import math
from typing import Optional

import torch
import torch.nn as nn

from dataclasses import dataclass

from model import Mult, Sum, recurrent_criteria
from model.world.transformer import MegaTransformerBlock
from utils.configuration import TransformerBlockConfig
from utils.megatransformer_utils import transformer_weight_init


@dataclass
class MegatransformerRecurrentConfig:
    """Configuration for the recurrent (thought vector) block.

    The recurrent block operates at 2*d_model internally to concatenate
    input embeddings with thought state, then projects back to d_model.
    """
    block_config: TransformerBlockConfig
    mean_thinking_steps: int = 32
    backprop_depth: int = 8
    thought_initialization_method: str = "like-init"
    exit_criteria: str = "kl_divergence"
    exit_criteria_threshold: float = 1e-4
    lockstep_n: bool = False
    lockstep_k: bool = False

class MegatransformerRecurrentBlock(nn.Module):
    def __init__(self, config: MegatransformerRecurrentConfig):
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

    def forward(self, x_0: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        n_steps_no_grad, k_steps_grad = self.n_k_steps(self.mean_thinking_steps, self.backprop_depth)

        x = x_0  # (batch_size, seq_len, d_model)

        last_thought_state = self.initialize_thinking_state(x)  # (batch_size, seq_len, d_model)

        if n_steps_no_grad > 0:
            with torch.no_grad():
                for _ in range(n_steps_no_grad):
                    thought_states = self.recurrent_block(
                        torch.cat([x_0, last_thought_state], dim=-1),
                        attention_mask=attention_mask
                    )

                    # back down to d_model
                    thought_states = self.projection(thought_states)  # (batch_size, seq_len, d_model)

                    if not self.training and self.exit_criteria is not None and self.exit_criteria.should_exit(last_thought_state, thought_states):
                        # get out early if exit criteria is met
                        return thought_states

                    last_thought_state = thought_states

        if k_steps_grad > 0:
            for _ in range(k_steps_grad):
                thought_states = self.recurrent_block(
                    torch.cat([x_0, last_thought_state], dim=-1),
                    attention_mask=attention_mask
                )

                # back down to d_model
                thought_states = self.projection(thought_states)

                # exit criteria doesn't need to run in the loop with grads

                last_thought_state = thought_states

        return last_thought_state
