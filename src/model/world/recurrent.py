import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from dataclasses import dataclass

from config.world.world_model import MegaTransformerRecurrentConfig
from model import recurrent_criteria
from model.transformer import MegaTransformerBlock
from model.world.kv_cache import RecurrentKVCache
from utils.megatransformer_utils import transformer_weight_init


class MegatransformerRecurrentBlock(nn.Module):
    def __init__(self, config: MegaTransformerRecurrentConfig):
        super().__init__()

        self.config = config
        self.mean_thinking_steps = config.mean_thinking_steps
        self.backprop_depth = config.backprop_depth
        self.thought_initialization_method = self.config.thought_initialization_method
        self.exit_criteria = self.config.exit_criteria
        self.exit_criteria_threshold = self.config.exit_criteria_threshold

        self.lockstep_n = self.config.lockstep_n
        self.lockstep_k = self.config.lockstep_k
        
        self.recurrent_block = MegaTransformerBlock(config.block_config)

        # block_config.d_model is 2*base_d_model (for concatenated [x_0, thought_state])
        # Project back to base_d_model
        self.projection = nn.Linear(config.block_config.d_model, config.block_config.d_model // 2)
        
        if self.exit_criteria == 'kl_divergence':
            self.exit_criteria = recurrent_criteria.KLDivergenceCriteria(self.exit_criteria_threshold)
        else:
            # todo: implement other exit criteria
            raise ValueError(f"Invalid exit criteria: {self.exit_criteria}")
        
        self.step = 0

        self._init_weights()

    def _init_weights(self):
        self.apply(transformer_weight_init())

    def initialize_thinking_state(self, input_embeds):
        """
        Taken directly from the original Huginn implementation: https://github.com/seal-rg/recurrent-pretraining/blob/main/recpre/model_dynamic.py
        """
        if self.thought_initialization_method == "none":
            return input_embeds
        if self.thought_initialization_method == "normal":
            x = torch.randn_like(input_embeds)
        elif self.thought_initialization_method == "embed":
            x = torch.randn_like(input_embeds).mul(1 / math.sqrt(input_embeds.shape[-1]))
        elif self.thought_initialization_method == "like-init":
            # initializes a thought state with the same shape as the input embeddings
            # eg if a batch input is of the shape (N, T, D), then the thought state is of the shape (N, T, D)
            # for example, (16, 1024, 512) is sampled from torch.randn_like
            # and then the thought state is truncated to a normal distribution of the same shape as the embeddings
            # where values outside of -3*0.02 and 3*0.02 are redrawn
            x = torch.randn_like(input_embeds)
            std = 0.02
            torch.nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3 * std, b=3 * std)
        elif self.thought_initialization_method == "zero":
            x = torch.zeros_like(input_embeds)
        elif self.thought_initialization_method == "unit":
            x = torch.randn_like(input_embeds)
            std, mean = torch.std_mean(x, dim=-1, keepdim=True)
            x = (x - mean) / std
        return x
    
    def n_k_steps(self, mean_steps, backprop_depth):
        """
        This randomly samples the number of total steps that the model will think for (to get a diverse range of thinking steps during training)
        and the number of backpropagation steps is always at most backprop_depth.
        1. During training, we sample from a Poisson log-normal distribution to get the total number of thinking steps.
        2. During evaluation, the exit criteria is used to determine the number of thinking steps.
        3. The number of backpropagation steps is always at most backprop_depth.
        4. The random sampling is seeded with the current step number to ensure reproducibility during checkpointing and across devices.
        """
        seed_n = 514229 + self.step  # easiest way to make the sampler re-runnable in checkpointing
        seed_k = 317811 + self.step
        if not self.lockstep_n and torch.distributed.is_initialized():
            seed_n = seed_n * (torch.distributed.get_rank() + 1)
        if not self.lockstep_k and torch.distributed.is_initialized():
            seed_k = seed_k * (torch.distributed.get_rank() + 1)

        # todo: get seeding working here
        n_generator = torch.Generator(device="cpu")
        n_generator.manual_seed(seed_n % (2**31 - 1))
        k_generator = torch.Generator(device="cpu")
        k_generator.manual_seed(seed_k % (2**31 - 1))

        t = max(mean_steps - backprop_depth, 0)
        s = backprop_depth

        if self.training:
            # poisson log normal filling
            sigma = 0.5
            mu = math.log(t + s) - (sigma**2 / 2)
            rate = torch.zeros((1,)).log_normal_(mean=mu, std=sigma, generator=n_generator)
            p = torch.poisson(torch.tensor([rate], dtype=torch.float), generator=n_generator) + 1
            n = torch.clamp(p - s, min=0)
            k = torch.as_tensor(torch.minimum(torch.as_tensor(s), p))
            self.step += 1
        else:
            # during evaluation, we use the mean number of steps as a maximum and let exit criteria determine when to stop
            # if it can do so before then. no backprop necessary
            n, k = torch.tensor(mean_steps), torch.tensor(0)

        return n.to(torch.long), k.to(torch.long)

    def forward(
        self,
        x_0: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[RecurrentKVCache] = None,
        position_offset: int = 0,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[RecurrentKVCache]]:
        """
        Forward pass through recurrent block with optional KV caching.

        For training: standard forward without caching (use_cache=False).
        For generation: uses Huginn-style shared KV cache (use_cache=True).

        The Huginn approach (Section 6.2):
        - Uses a circular buffer with cache_budget slots
        - At iteration i, uses slot (i % cache_budget)
        - Previous tokens' KVs may be from different iteration depths
        - Model zero-shot adapts to this mixed-depth attention

        Args:
            x_0: Input embeddings, shape (batch, seq_len, d_model)
            attention_mask: Optional attention mask
            kv_cache: Optional RecurrentKVCache for efficient generation
            position_offset: Position offset for RoPE (for cached generation)
            use_cache: Whether to use and update KV cache

        Returns:
            thought_states: Output thought states, shape (batch, seq_len, d_model)
            new_kv_cache: Updated RecurrentKVCache if use_cache=True, else None
        """
        n_steps_no_grad, k_steps_grad = self.n_k_steps(self.mean_thinking_steps, self.backprop_depth)

        last_thought_state = self.initialize_thinking_state(x_0)  # (batch_size, seq_len, d_model)

        # Initialize or use existing cache
        new_kv_cache = kv_cache if use_cache and kv_cache is not None else None
        if use_cache and new_kv_cache is None:
            new_kv_cache = RecurrentKVCache(strategy="huginn", cache_budget=16)

        iteration = 0

        if n_steps_no_grad > 0:
            with torch.no_grad():
                for _ in range(n_steps_no_grad):
                    # Get cache for this iteration (Huginn circular buffer)
                    iter_cache = None
                    if use_cache and new_kv_cache is not None:
                        iter_cache = new_kv_cache.get_layer_at_iteration(iteration, layer_idx=0)

                    block_output, updated_cache = self.recurrent_block(
                        torch.cat([x_0, last_thought_state], dim=-1),
                        attention_mask=attention_mask,
                        kv_cache=iter_cache,
                        position_offset=position_offset,
                        use_cache=use_cache,
                    )

                    # Update cache for this iteration slot
                    if use_cache and updated_cache is not None and new_kv_cache is not None:
                        slot = iteration % new_kv_cache.cache_budget
                        layer_cache = new_kv_cache.get_layer_at_iteration(iteration, layer_idx=0)
                        layer_cache.key_cache = updated_cache.key_cache
                        layer_cache.value_cache = updated_cache.value_cache

                    # Project back down to d_model
                    thought_states = self.projection(block_output)

                    if not self.training and self.exit_criteria is not None and self.exit_criteria.should_exit(last_thought_state, thought_states):
                        return thought_states, new_kv_cache

                    last_thought_state = thought_states
                    iteration += 1

        if k_steps_grad > 0:
            for _ in range(k_steps_grad):
                # Get cache for this iteration
                iter_cache = None
                if use_cache and new_kv_cache is not None:
                    iter_cache = new_kv_cache.get_layer_at_iteration(iteration, layer_idx=0)

                block_output, updated_cache = self.recurrent_block(
                    torch.cat([x_0, last_thought_state], dim=-1),
                    attention_mask=attention_mask,
                    kv_cache=iter_cache,
                    position_offset=position_offset,
                    use_cache=use_cache,
                )

                # Update cache for this iteration slot
                if use_cache and updated_cache is not None and new_kv_cache is not None:
                    layer_cache = new_kv_cache.get_layer_at_iteration(iteration, layer_idx=0)
                    layer_cache.key_cache = updated_cache.key_cache
                    layer_cache.value_cache = updated_cache.value_cache

                thought_states = self.projection(block_output)
                last_thought_state = thought_states
                iteration += 1

        return last_thought_state, new_kv_cache

    def generate_step(
        self,
        x_0: torch.Tensor,
        kv_cache: RecurrentKVCache,
        position_offset: int,
        attention_mask: Optional[torch.Tensor] = None,
        max_iterations: Optional[int] = None,
    ) -> Tuple[torch.Tensor, RecurrentKVCache, int]:
        """
        Single generation step with KV caching.

        Runs recurrent iterations until convergence or max_iterations.
        Uses Huginn-style shared KV cache.

        Args:
            x_0: Input embedding for new token(s), shape (batch, new_seq_len, d_model)
            kv_cache: RecurrentKVCache with cached KVs from previous tokens
            position_offset: Position of new token(s) in sequence
            attention_mask: Optional attention mask for full sequence
            max_iterations: Maximum iterations (defaults to mean_thinking_steps)

        Returns:
            output: Output thought state for new token(s)
            updated_cache: Updated KVCache
            num_iterations: Number of iterations performed
        """
        if max_iterations is None:
            max_iterations = self.mean_thinking_steps

        last_thought_state = self.initialize_thinking_state(x_0)

        for iteration in range(max_iterations):
            # Get cache for this iteration slot
            iter_cache = kv_cache.get_layer_at_iteration(iteration, layer_idx=0)

            block_output, updated_cache = self.recurrent_block(
                torch.cat([x_0, last_thought_state], dim=-1),
                attention_mask=attention_mask,
                kv_cache=iter_cache,
                position_offset=position_offset,
                use_cache=True,
            )

            # Update cache
            if updated_cache is not None:
                layer_cache = kv_cache.get_layer_at_iteration(iteration, layer_idx=0)
                layer_cache.key_cache = updated_cache.key_cache
                layer_cache.value_cache = updated_cache.value_cache

            thought_states = self.projection(block_output)

            # Check exit criteria
            if self.exit_criteria is not None and self.exit_criteria.should_exit(last_thought_state, thought_states):
                return thought_states, kv_cache, iteration + 1

            last_thought_state = thought_states

        return last_thought_state, kv_cache, max_iterations
