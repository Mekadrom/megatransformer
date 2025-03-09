from torch import nn
from typing import Optional, Union

from model import huginn_criteria, megatransformer_attn, megatransformer_ffn, mult, sum

import math
import megatransformer_utils
import torch


class BlockOutput:
    def __init__(self, hidden_states, key_value=None, attention_probs=None):
        self.hidden_states = hidden_states
        self.key_value = key_value
        self.attention_probs = attention_probs

class MegaTransformerBlock(nn.Module):
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__()
        self.attention = megatransformer_attn.MegaTransformerSelfAttention(config)
        
        self.ffn: nn.Module
        if config.ffn_type == "mlp":
            self.ffn = megatransformer_ffn.SimpleFFN(config)
        elif config.ffn_type == "gated":
            self.ffn = megatransformer_ffn.GateFFN(config)
        else:
            raise ValueError(f"Unknown ffn_type: {config.ffn_type}")

        if config.pre_attn_norm:
            self.pre_attn_norm = megatransformer_utils.create_norm(config)
        else:
            self.pre_attn_norm = None
        if config.post_attn_norm:
            self.post_attn_norm = megatransformer_utils.create_norm(config)
        else:
            self.post_attn_norm = None
        if config.pre_ffn_norm:
            self.pre_ffn_norm = megatransformer_utils.create_norm(config)
        else:
            self.pre_ffn_norm = None
        if config.post_ffn_norm:
            self.post_ffn_norm = megatransformer_utils.create_norm(config)
        else:
            self.post_ffn_norm = None
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False
    ):
        if self.pre_attn_norm is not None:
            pre_attn_input = self.pre_attn_norm(hidden_states)
        else:
            pre_attn_input = hidden_states

        attn_outputs = self.attention(
            pre_attn_input,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )

        attn_output = attn_outputs.hidden_states
        attention_probs = attn_outputs.attention_probs

        hidden_states = hidden_states + attn_output

        if self.post_attn_norm is not None:
            attn_outputs = self.post_attn_norm(attn_outputs)

        pre_ffn_input = self.pre_ffn_norm(hidden_states)
        ffn_output = self.ffn(pre_ffn_input)
        
        hidden_states = hidden_states + ffn_output

        if self.post_ffn_norm is not None:
            hidden_states = self.post_ffn_norm(hidden_states)

        return BlockOutput(
            hidden_states=hidden_states,
            attention_probs=attention_probs,
        )


class RecurrentBlockOutput:
    def __init__(self, hidden_states, last_thought_state, next_cache=None, all_hidden_states=None, attention_probs=None):
        self.hidden_states = hidden_states
        self.last_thought_state = last_thought_state
        self.next_cache = next_cache
        self.all_hidden_states = all_hidden_states
        self.attention_probs = attention_probs


class MegaTransformerRecurrentBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mean_thinking_steps = config.huginn_mean_thinking_steps
        self.backprop_depth = config.huginn_backprop_depth
        self.thought_initialization_method = self.config.huginn_thought_initialization_method
        self.adapter_method = self.config.huginn_adapter_method
        self.exit_criteria = self.config.huginn_exit_criteria
        self.exit_criteria_threshold = self.config.huginn_exit_criteria_threshold

        self.blocks = nn.ModuleList([MegaTransformerBlock(config) for _ in range(config.num_recurrent_layers)])

        self.adapter: nn.Module
        if self.adapter_method == 'sum':
            self.adapter = sum.Sum() # todo: implement other adapter methods
        elif self.adapter_method == "gate":
            self.adapter = mult.Mult()
        elif self.adapter_method == "linear":
            self.adapter = nn.Linear(config.hidden_size * 2, config.hidden_size)
        else:
            raise ValueError(f"Invalid adapter method: {self.adapter_method}")
        
        if self.exit_criteria == 'kl_divergence':
            self.exit_criteria = huginn_criteria.KLDivergenceCriteria(self.exit_criteria_threshold) # todo: implement other exit criteria
        else:
            raise ValueError(f"Invalid exit criteria: {self.exit_criteria}")

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
        # todo: get seeding working here
        n_generator = torch.Generator(device="cpu")
        n_generator.manual_seed(42 % (2**31 - 1))
        k_generator = torch.Generator(device="cpu")
        k_generator.manual_seed(42 % (2**31 - 1))

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
        else:
            n, k = torch.tensor(mean_steps), torch.tensor(0)
        return n.to(torch.long), k.to(torch.long)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        past_key_value=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False
    ):
        if self.training:
            n_steps_no_grad, k_steps_grad = self.n_k_steps(self.mean_thinking_steps, self.backprop_depth)
        else:
            n_steps_no_grad, k_steps_grad = self.mean_thinking_steps, 0

        x = self.initialize_thinking_state(hidden_states)
        last_thought_state = x
        if n_steps_no_grad > 0:
            with torch.no_grad():
                outputs = self.apply_thinking_layers(x, last_thought_state, n_steps_no_grad, hidden_states, attention_mask, head_mask, past_key_value, use_cache, output_attentions, output_hidden_states)
                x = outputs.hidden_states
                last_thought_state = outputs.last_thought_state

        if k_steps_grad > 0:
            outputs = self.apply_thinking_layers(x, last_thought_state, k_steps_grad, hidden_states, attention_mask, head_mask, past_key_value, use_cache, output_attentions, output_hidden_states)

        return RecurrentBlockOutput(
            hidden_states=outputs.hidden_states,
            last_thought_state=outputs.last_thought_state,
            next_cache=outputs.next_cache,
            all_hidden_states=outputs.all_hidden_states,
            attention_probs=outputs.attention_probs,
        )

    def apply_thinking_layers(self,
                              x: torch.Tensor,
                              last_thought_state: Optional[torch.Tensor],
                              n_steps: Union[int, torch.Tensor],
                              hidden_states: torch.Tensor,
                              attention_mask: Optional[torch.Tensor],
                              head_mask,
                              past_key_values=None,
                              use_cache=False,
                              output_attentions: bool=False,
                              output_hidden_states: bool=False):
        all_hidden_states: Optional[list] = [] if output_hidden_states else None
        all_attentions: Optional[list] = [] if output_attentions else None

        for _ in range(n_steps):
            if self.adapter_method == "linear":
                x = self.adapter(torch.cat([x, hidden_states], dim=-1))
            else:
                x = self.adapter(x, hidden_states)

            for i, (thinking_layer, past_key_value) in enumerate(zip(self.blocks, past_key_values)):
                if all_hidden_states is not None:
                    all_hidden_states.append(hidden_states)

                outputs = thinking_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                )

                hidden_states = outputs.hidden_states
                attention_probs = outputs.attention_probs

                if all_attentions is not None:
                    all_attentions.append(attention_probs)

                last_thought_state = hidden_states

            if not self.training and self.exit_criteria.should_exit(last_thought_state, x):
                break

        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        return RecurrentBlockOutput(
            hidden_states=hidden_states,
            last_thought_state=last_thought_state,
            next_cache=None,
            all_hidden_states=all_hidden_states,
            attention_probs=all_attentions,
        )
