import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

from config.world.world_model import MegaTransformerRecurrentConfig
from model import recurrent_criteria
from model.transformer import MegaTransformerEncoderBlock
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
        self.thought_init_std = self.config.thought_init_std

        self.lockstep_n = self.config.lockstep_n
        self.lockstep_k = self.config.lockstep_k

        self.n_recurrent_blocks = config.n_recurrent_blocks
        self.injection_type = config.injection_type

        # Bank of recurrent blocks. If share_block_weights, all entries reference
        # the same module (deeper application of 1 block per iteration).
        if config.share_block_weights:
            shared_block = MegaTransformerEncoderBlock(config.block_config)
            self.recurrent_blocks = nn.ModuleList([shared_block] * self.n_recurrent_blocks)
        else:
            self.recurrent_blocks = nn.ModuleList([
                MegaTransformerEncoderBlock(config.block_config)
                for _ in range(self.n_recurrent_blocks)
            ])

        if self.injection_type == "concat":
            # block_config.d_model is 2*base_d_model (for concatenated [x_0, thought_state])
            # Project back to base_d_model
            self.projection = nn.Linear(config.block_config.d_model, config.block_config.d_model // 2)
        else:
            # Additive injection: blocks operate at base d_model, no projection needed
            self.projection = None

        # Per-iteration normalization: prevents activation growth across iterations
        # when using multiple blocks per iteration. Without this, 4 residual blocks
        # compound the signal magnitude, causing unbounded growth over 32 iterations.
        self.iteration_norm = config.iteration_norm
        self.pre_projection_norm = None
        self.post_projection_norm = None
        if self.iteration_norm == "pre_projection":
            from model.norms import RMSNorm
            self.pre_projection_norm = RMSNorm(config.block_config.d_model)
        elif self.iteration_norm == "post_projection":
            from model.norms import RMSNorm
            out_dim = config.block_config.d_model // 2 if self.injection_type == "concat" else config.block_config.d_model
            self.post_projection_norm = RMSNorm(out_dim)
        
        if self.exit_criteria == 'kl_divergence':
            self.exit_criteria = recurrent_criteria.KLDivergenceCriteria(self.exit_criteria_threshold)
        else:
            # todo: implement other exit criteria
            raise ValueError(f"Invalid exit criteria: {self.exit_criteria}")
        
        self.step = 0
        self.track_iteration_stats = False

        self._init_weights()

    def _combine(self, x_0: torch.Tensor, thought: torch.Tensor) -> torch.Tensor:
        """Combine input embedding with thought state for block input."""
        if self.injection_type == "concat":
            return torch.cat([x_0, thought], dim=-1)
        else:
            return x_0 + thought

    def _extract_thought(self, block_output: torch.Tensor) -> torch.Tensor:
        """Extract thought state from block output."""
        if self.pre_projection_norm is not None:
            block_output = self.pre_projection_norm(block_output)
        if self.projection is not None:
            block_output = self.projection(block_output)
        if self.post_projection_norm is not None:
            block_output = self.post_projection_norm(block_output)
        return block_output

    def _run_iteration(
        self,
        x_0: torch.Tensor,
        thought: torch.Tensor,
        iteration: int,
        attention_mask: Optional[torch.Tensor],
        kv_cache: Optional[RecurrentKVCache],
        position_offset: int,
        use_cache: bool,
        share_kv_cache: bool,
        _extend_mask_fn=None,
    ) -> torch.Tensor:
        """Run one full recurrent iteration through all blocks sequentially.

        Each iteration: combine(x_0, thought) → block_0 → block_1 → ... → block_N → extract → new_thought
        """
        h = self._combine(x_0, thought)

        for block_idx, block in enumerate(self.recurrent_blocks):
            iter_cache = None
            if use_cache and kv_cache is not None:
                # Each block gets its own layer_idx in the cache (unless sharing)
                layer_idx = 0 if share_kv_cache else block_idx
                iter_cache = kv_cache.get_layer_at_iteration(iteration, layer_idx=layer_idx)

            mask = _extend_mask_fn(attention_mask, iter_cache) if _extend_mask_fn else attention_mask

            h, updated_cache = block(
                h,
                attention_mask=mask,
                kv_cache=iter_cache,
                position_offset=position_offset,
                use_cache=use_cache,
            )

            if use_cache and updated_cache is not None and kv_cache is not None:
                layer_idx = 0 if share_kv_cache else block_idx
                layer_cache = kv_cache.get_layer_at_iteration(iteration, layer_idx=layer_idx)
                layer_cache.key_cache = updated_cache.key_cache
                layer_cache.value_cache = updated_cache.value_cache

        return self._extract_thought(h)

    def _init_weights(self):
        self.apply(transformer_weight_init())
        # Re-init recurrent blocks with configurable gain (default 0.02 = same as transformer_weight_init)
        if self.config.block_init_gain != 0.02:
            gain = self.config.block_init_gain
            for block in self.recurrent_blocks:
                for module in block.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_normal_(module.weight, gain=gain)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
        # Optionally re-init projection with explicit gain (default 1.0 = no override,
        # keeps the gain=0.02 from transformer_weight_init)
        if self.projection is not None and self.config.projection_init_gain != 1.0:
            nn.init.xavier_uniform_(self.projection.weight, gain=self.config.projection_init_gain)
            if self.projection.bias is not None:
                nn.init.zeros_(self.projection.bias)
        if not self.config.depth_scaled_init:
            return
        # Depth-scaled init for output projections (Huginn-inspired).
        # Each block appears mean_thinking_steps times (not n_blocks * mean_steps),
        # so per-block effective depth = mean_thinking_steps.
        # Use base d_model (half of concat d_model) to avoid over-shrinking.
        d_model = self.config.block_config.d_model
        if self.injection_type == "concat":
            d_model = d_model // 2  # use base width, not concat width
        l_eff = self.mean_thinking_steps
        out_std = (1.0 / (5 * d_model * l_eff)) ** 0.5
        for block in self.recurrent_blocks:
            if hasattr(block, 'self_attn') and hasattr(block.self_attn, 'o_proj'):
                nn.init.normal_(block.self_attn.o_proj.weight, std=out_std)
            if hasattr(block, 'ffn') and hasattr(block.ffn, 'condense'):
                nn.init.normal_(block.ffn.condense.weight, std=out_std)

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
            x = torch.randn_like(input_embeds)
            std = self.thought_init_std
            torch.nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3 * std, b=3 * std)
        elif self.thought_initialization_method == "zero":
            x = torch.zeros_like(input_embeds)
        elif self.thought_initialization_method == "unit":
            x = torch.randn_like(input_embeds)
            std, mean = torch.std_mean(x, dim=-1, keepdim=True)
            x = (x - mean) / std
        return x
    
    @torch.no_grad()
    def _compute_thought_stats(self, thought: torch.Tensor) -> dict:
        """Compute per-token activation stats for a thought state tensor.

        Args:
            thought: (batch, seq_len, d_model)

        Returns:
            Dict with per-token stats, each shaped (batch, seq_len).
        """
        t = thought.float()
        return {
            "std": t.std(dim=-1).cpu(),       # (batch, seq_len)
            "mean": t.mean(dim=-1).cpu(),
            "max": t.max(dim=-1).values.cpu(),
            "min": t.min(dim=-1).values.cpu(),
            "norm": t.norm(dim=-1).cpu(),
        }

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
        share_kv_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[RecurrentKVCache], int, List[float]]:
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
            num_iterations: Total number of recurrent iterations performed
            kl_per_iteration: Per-iteration KL divergence between consecutive
                thought states (mean over batch and sequence). Empty during training.
        """
        n_steps_no_grad, k_steps_grad = self.n_k_steps(self.mean_thinking_steps, self.backprop_depth)

        last_thought_state = self.initialize_thinking_state(x_0)  # (batch_size, seq_len, d_model)
        track_kl = not self.training
        kl_per_iteration: List[float] = []

        # Per-iteration activation stats (controlled by flag, works in both train and eval)
        iteration_stats: Optional[List[dict]] = None
        if getattr(self, 'track_iteration_stats', False):
            iteration_stats = []

        # Per-token convergence tracking (eval only)
        converged: Optional[torch.Tensor] = None
        if not self.training and self.exit_criteria is not None:
            converged = torch.zeros(
                x_0.shape[0], x_0.shape[1],
                dtype=torch.bool, device=x_0.device,
            )

        # Initialize or use existing cache
        new_kv_cache = kv_cache if use_cache and kv_cache is not None else None
        if use_cache and new_kv_cache is None:
            new_kv_cache = RecurrentKVCache(strategy="huginn", cache_budget=16)

        iteration = 0

        def _extend_mask_for_cache(mask, iter_cache):
            """Extend attention mask to cover cached KV positions.

            When the Huginn circular buffer reuses a slot, the cached keys
            from a previous iteration are concatenated with new keys, making
            the key dimension larger than the query dimension.  The attention
            mask must cover [cached_positions | new_positions]; cached
            positions are always attendable (1).
            """
            if mask is None or iter_cache is None or iter_cache.key_cache is None:
                return mask
            cached_len = iter_cache.seq_len
            if cached_len == 0:
                return mask
            ones = torch.ones(
                mask.shape[0], cached_len,
                device=mask.device, dtype=mask.dtype,
            )
            return torch.cat([ones, mask], dim=-1)

        if n_steps_no_grad > 0:
            with torch.no_grad():
                for _ in range(n_steps_no_grad):
                    new_thought = self._run_iteration(
                        x_0, last_thought_state, iteration,
                        attention_mask, new_kv_cache, position_offset,
                        use_cache, share_kv_cache, _extend_mask_for_cache,
                    )

                    kl_per_token = None
                    if track_kl or iteration_stats is not None:
                        kl_per_token = F.kl_div(
                            last_thought_state, new_thought,
                            reduction="none", log_target=True,
                        ).sum(dim=-1)  # (batch, seq_len)
                    if track_kl and kl_per_token is not None:
                        kl_per_iteration.append(kl_per_token.mean().item())

                    # Per-token freeze: only update non-converged tokens
                    if converged is not None and hasattr(self.exit_criteria, 'converged_mask'):
                        newly_converged = self.exit_criteria.converged_mask(last_thought_state, new_thought)
                        converged = converged | newly_converged
                        thought_states = torch.where(converged.unsqueeze(-1), last_thought_state, new_thought)
                        if converged.all():
                            if iteration_stats is not None:
                                stats = self._compute_thought_stats(thought_states)
                                if kl_per_token is not None:
                                    stats["kl"] = kl_per_token.cpu()
                                iteration_stats.append(stats)
                            return thought_states, new_kv_cache, iteration + 1, kl_per_iteration, iteration_stats
                    else:
                        thought_states = new_thought

                    if iteration_stats is not None:
                        stats = self._compute_thought_stats(thought_states)
                        if kl_per_token is not None:
                            stats["kl"] = kl_per_token.cpu()
                        iteration_stats.append(stats)

                    last_thought_state = thought_states
                    iteration += 1

        if k_steps_grad > 0:
            for _ in range(k_steps_grad):
                thought_states = self._run_iteration(
                    x_0, last_thought_state, iteration,
                    attention_mask, new_kv_cache, position_offset,
                    use_cache, share_kv_cache, _extend_mask_for_cache,
                )

                kl_per_token = None
                if track_kl or iteration_stats is not None:
                    kl_per_token = F.kl_div(
                        last_thought_state.detach(), thought_states.detach(),
                        reduction="none", log_target=True,
                    ).sum(dim=-1)  # (batch, seq_len)
                if track_kl and kl_per_token is not None:
                    kl_per_iteration.append(kl_per_token.mean().item())

                if iteration_stats is not None:
                    stats = self._compute_thought_stats(thought_states)
                    if kl_per_token is not None:
                        stats["kl"] = kl_per_token.cpu()
                    iteration_stats.append(stats)

                last_thought_state = thought_states
                iteration += 1

        return last_thought_state, new_kv_cache, iteration, kl_per_iteration, iteration_stats

    def generate_step(
        self,
        x_0: torch.Tensor,
        kv_cache: RecurrentKVCache,
        position_offset: int,
        attention_mask: Optional[torch.Tensor] = None,
        max_iterations: Optional[int] = None,
        share_kv_cache: bool = False,
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

        def _extend_mask_for_cache(mask, cache):
            if mask is None or cache is None or cache.key_cache is None:
                return mask
            cached_len = cache.seq_len
            if cached_len == 0:
                return mask
            ones = torch.ones(mask.shape[0], cached_len, device=mask.device, dtype=mask.dtype)
            return torch.cat([ones, mask], dim=-1)

        for iteration in range(max_iterations):
            thought_states = self._run_iteration(
                x_0, last_thought_state, iteration,
                attention_mask, kv_cache, position_offset,
                use_cache=True, share_kv_cache=share_kv_cache,
                _extend_mask_fn=_extend_mask_for_cache,
            )

            # Check exit criteria
            if self.exit_criteria is not None and self.exit_criteria.should_exit(last_thought_state, thought_states):
                return thought_states, kv_cache, iteration + 1

            last_thought_state = thought_states

        return last_thought_state, kv_cache, max_iterations
