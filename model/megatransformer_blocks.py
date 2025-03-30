from torch import nn
from typing import Optional, Union

import torch.utils.tensorboard

from model import megatransformer_attn, megatransformer_ffn, megatransformer_modules, recurrent_criteria

import math
import megatransformer_utils
import torch


class MegaTransformerBlock(nn.Module):
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__()
        self.attention = megatransformer_attn.MegaTransformerSelfAttention(config)
        
        self.ffn: nn.Module
        if config.ffn_type == "mlp":
            self.ffn = megatransformer_ffn.SimpleFFN(config)
        else:
            raise ValueError(f"Unknown ffn_type: {config.ffn_type}")

        if config.pre_attn_norm:
            self.pre_attn_norm = megatransformer_utils.create_norm(config.hidden_size, config.norm_type, config.norm_eps)
        else:
            self.pre_attn_norm = None
        if config.post_attn_norm:
            self.post_attn_norm = megatransformer_utils.create_norm(config.hidden_size, config.norm_type, config.norm_eps)
        else:
            self.post_attn_norm = None
        if config.pre_ffn_norm:
            self.pre_ffn_norm = megatransformer_utils.create_norm(config.hidden_size, config.norm_type, config.norm_eps)
        else:
            self.pre_ffn_norm = None
        if config.post_ffn_norm:
            self.post_ffn_norm = megatransformer_utils.create_norm(config.hidden_size, config.norm_type, config.norm_eps)
        else:
            self.post_ffn_norm = None

        self._init_weights()

    def _init_weights(self):
        self.apply(megatransformer_utils.transformer_weight_init)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        past_key_values: megatransformer_utils.KVCache=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ) -> Union[tuple[torch.Tensor, megatransformer_utils.KVCache, torch.Tensor], megatransformer_utils.BlockOutput]:
        if self.pre_attn_norm is not None:
            pre_attn_input = self.pre_attn_norm(hidden_states)
        else:
            pre_attn_input = hidden_states

        if use_cache:
            if past_key_values is None:
                past_key_values = megatransformer_utils.KVCache()

        attn_outputs = self.attention(
            pre_attn_input,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        if not return_dict:
            attn_output = attn_outputs[0]
            attention_probs = attn_outputs[2]
        else:
            attn_output = attn_outputs.hidden_states
            attention_probs = attn_outputs.attention_probs

        hidden_states = hidden_states + attn_output

        if self.post_attn_norm is not None:
            attn_outputs = self.post_attn_norm(attn_output)

        pre_ffn_input = self.pre_ffn_norm(hidden_states)
        ffn_output = self.ffn(pre_ffn_input)

        hidden_states = hidden_states + ffn_output

        if self.post_ffn_norm is not None:
            hidden_states = self.post_ffn_norm(hidden_states)

        if not return_dict:
            return (
                hidden_states,
                past_key_values,
                attention_probs,
            )

        return megatransformer_utils.BlockOutput(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attention_probs=attention_probs,
        )

class RecurrentBlockOutput:
    def __init__(self, hidden_states, last_thought_state, past_key_values=None, all_hidden_states=None, attention_probs=None, n_steps_no_grad=None, k_steps_grad=None):
        self.hidden_states = hidden_states
        self.last_thought_state = last_thought_state
        self.past_key_values = past_key_values
        self.all_hidden_states = all_hidden_states
        self.attention_probs = attention_probs
        self.n_steps_no_grad = n_steps_no_grad
        self.k_steps_grad = k_steps_grad


# todo: implement kv caching
class MegaTransformerRecurrentBlock(nn.Module):
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__()
        self.config = config
        self.mean_thinking_steps = config.recurrent_mean_thinking_steps
        self.backprop_depth = config.recurrent_backprop_depth
        self.thought_initialization_method = self.config.recurrent_thought_initialization_method
        self.adapter_method = self.config.recurrent_adapter_method
        self.exit_criteria = self.config.recurrent_exit_criteria
        self.exit_criteria_threshold = self.config.recurrent_exit_criteria_threshold

        self.lockstep_n = self.config.recurrent_lockstep_n
        self.lockstep_k = self.config.recurrent_lockstep_k

        self.blocks = nn.ModuleList([MegaTransformerBlock(config) for _ in range(config.n_recurrent_layers)])

        self.adapter: nn.Module
        if self.adapter_method == 'sum':
            self.adapter = megatransformer_modules.Sum() # todo: implement other adapter methods
        elif self.adapter_method == "gate":
            self.adapter = megatransformer_modules.Mult()
        elif self.adapter_method == "linear":
            self.adapter = nn.Linear(config.hidden_size * 2, config.hidden_size)
        else:
            raise ValueError(f"Invalid adapter method: {self.adapter_method}")
        
        if self.exit_criteria == 'kl_divergence':
            self.exit_criteria = recurrent_criteria.KLDivergenceCriteria(self.exit_criteria_threshold) # todo: implement other exit criteria
        else:
            raise ValueError(f"Invalid exit criteria: {self.exit_criteria}")
        
        self.step = 0

        self._init_weights()

    def _init_weights(self):
        self.apply(megatransformer_utils.transformer_weight_init)
        
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
            n, k = torch.tensor(mean_steps), torch.tensor(0)

        return n.to(torch.long), k.to(torch.long)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        past_key_values: Optional[list[list[megatransformer_utils.KVCache]]]=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        n_steps_no_grad, k_steps_grad = self.n_k_steps(self.mean_thinking_steps, self.backprop_depth)

        all_hidden_states = [] if output_hidden_states else None

        x = self.initialize_thinking_state(hidden_states)
        last_thought_state = x
        actual_n_steps = n_steps_no_grad

        if n_steps_no_grad > 0:
            with torch.no_grad():
                outputs, actual_n_steps = self.apply_thinking_layers(
                    x,
                    last_thought_state,
                    n_steps_no_grad,
                    hidden_states,
                    attention_mask,
                    head_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    output_hidden_states,
                    return_dict=return_dict,
                )
                if not return_dict:
                    x = outputs[0]
                    last_thought_state = outputs[1]
                    if output_hidden_states:
                        all_hidden_states.extend(outputs[2])
                else:
                    x = outputs.hidden_states
                    last_thought_state = outputs.last_thought_state
                    if output_hidden_states:
                        all_hidden_states.extend(outputs.all_hidden_states)

        if k_steps_grad > 0:
            outputs, _ = self.apply_thinking_layers(
                x,
                last_thought_state,
                k_steps_grad,
                hidden_states,
                attention_mask,
                head_mask,
                past_key_values,
                use_cache,
                output_attentions,
                output_hidden_states,
                start_step_idx=n_steps_no_grad,
                return_dict=return_dict,
            )
            if not return_dict:
                x = outputs[0]
                last_thought_state = outputs[1]
                if output_hidden_states:
                    all_hidden_states.extend(outputs[2])
            else:
                x = outputs.hidden_states
                last_thought_state = outputs.last_thought_state
                if output_hidden_states:
                    all_hidden_states.extend(outputs.all_hidden_states)

        if not return_dict:
            return (
                x,
                last_thought_state,
                past_key_values,
                all_hidden_states,
                outputs.attention_probs,
                n_steps_no_grad,
                k_steps_grad,
            )

        return RecurrentBlockOutput(
            hidden_states=x,
            last_thought_state=outputs.last_thought_state,
            past_key_values=outputs.past_key_values,
            all_hidden_states=all_hidden_states,
            attention_probs=outputs.attention_probs,
            n_steps_no_grad=min(n_steps_no_grad, actual_n_steps),
            k_steps_grad=k_steps_grad,
        )

    def apply_thinking_layers(self,
                              x: torch.Tensor,
                              last_thought_state: Optional[torch.Tensor],
                              n_steps: Union[int, torch.Tensor],
                              hidden_states: torch.Tensor,
                              attention_mask: Optional[torch.Tensor],
                              head_mask,
                              past_key_values: Optional[list[list[megatransformer_utils.KVCache]]]=None,
                              use_cache: bool=False,
                              output_attentions: bool=False,
                              output_hidden_states: bool=False,
                              start_step_idx=0,
                              return_dict: bool=False):
        all_hidden_states: Optional[list] = [] if output_hidden_states else None
        all_attentions: Optional[list] = [] if output_attentions else None

        actual_n_steps = 0
        for _ in range(n_steps.item()):
            if self.adapter_method == "linear":
                x = self.adapter(torch.cat([x, hidden_states], dim=-1))
            else:
                x = self.adapter(x, hidden_states)

            for i, thinking_layer in enumerate(self.blocks):
                outputs = thinking_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i] if head_mask is not None else None,
                    # past_key_values=past_key_values[self.config.n_prelude_layers + i][start_step_idx+step] if past_key_values is not None else None,
                    # use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

                if not return_dict:
                    hidden_states = outputs[0]
                    attention_probs = outputs[2]
                else:
                    hidden_states = outputs.hidden_states
                    attention_probs = outputs.attention_probs

                if output_attentions:
                    all_attentions.append(attention_probs)

                if output_hidden_states:
                    all_hidden_states.append(hidden_states)

                last_thought_state = hidden_states

            actual_n_steps += 1
            if not self.training and self.exit_criteria.should_exit(last_thought_state, x):
                break

        if not return_dict:
            return (
                hidden_states,
                past_key_values,
                all_hidden_states,
                all_attentions,
            ), actual_n_steps

        return RecurrentBlockOutput(
            hidden_states=hidden_states,
            last_thought_state=last_thought_state,
            past_key_values=past_key_values,
            all_hidden_states=all_hidden_states,
            attention_probs=all_attentions,
        ), actual_n_steps
