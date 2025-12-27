import math

import torch

from typing import Optional, Union

from torch import nn
from transformers import PreTrainedModel, GenerationMixin

from model import causal, kv_cache, Mult, recurrent_criteria, Sum
from utils import configuration
from utils.megatransformer_utils import transformer_weight_init
from utils.model_utils import create_norm, create_sinusoidal_1d_pos_encoding


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
    def __init__(self, config: configuration.MegaTransformerConfig):
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

        self.blocks = nn.ModuleList([causal.MegaTransformerBlock(config) for _ in range(config.n_recurrent_layers)])

        self.adapter: nn.Module
        if self.adapter_method == 'sum':
            self.adapter = Sum() # todo: implement other adapter methods
        elif self.adapter_method == "gate":
            self.adapter = Mult()
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
        past_key_values: Optional[list[list[kv_cache.KVCache]]]=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        n_steps_no_grad, k_steps_grad = self.n_k_steps(self.mean_thinking_steps, self.backprop_depth)

        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None

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
                    if not self.training:
                        if output_hidden_states:
                            all_hidden_states.extend(outputs[2])
                        if output_attentions:
                            all_attentions.extend(outputs[3])
                else:
                    x = outputs.hidden_states
                    last_thought_state = outputs.last_thought_state
                    if not self.training:
                        if output_hidden_states:
                            all_hidden_states.extend(outputs.all_hidden_states)
                        if output_attentions:
                            all_attentions.extend(outputs.attention_probs)

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
                if not self.training:
                    if output_hidden_states:
                        all_hidden_states.extend(outputs[2])
                    if output_attentions:
                        all_attentions.extend(outputs[3])
            else:
                x = outputs.hidden_states
                last_thought_state = outputs.last_thought_state
                if not self.training:
                    if output_hidden_states:
                        all_hidden_states.extend(outputs.all_hidden_states)
                    if output_attentions:
                        all_attentions.extend(outputs.attention_probs)

        if not return_dict:
            return (
                x,
                last_thought_state,
                past_key_values,
                all_hidden_states,
                all_attentions,
                n_steps_no_grad,
                k_steps_grad,
            )

        return RecurrentBlockOutput(
            hidden_states=x,
            last_thought_state=outputs.last_thought_state,
            past_key_values=past_key_values,
            all_hidden_states=all_hidden_states,
            attention_probs=all_attentions,
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
                              past_key_values: Optional[list[list[kv_cache.KVCache]]]=None,
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

                if not self.training:
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


class MegaTransformerRawEmbedsRecurrentCausalModel(PreTrainedModel, GenerationMixin):
    config_class = configuration.MegaTransformerConfig
    
    def __init__(self, config: configuration.MegaTransformerConfig):
        super().__init__(config)
        self.config = config
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        prelude = nn.ModuleList([causal.MegaTransformerBlock(config) for _ in range(config.n_prelude_layers)])
        recurrent = causal.MegaTransformerRecurrentBlock(config)
        coda = nn.ModuleList([causal.MegaTransformerBlock(config) for _ in range(config.n_coda_layers)])

        self.transformer = nn.ModuleList([*prelude, recurrent, *coda])
        
        if config.use_final_norm:
            self.norm_final = create_norm(config.hidden_size, config.norm_type, config.norm_eps)
        else:
            self.norm_final = None

    def gradient_checkpointing_enable(self, **kwargs):
        print(f"Enabling gradient checkpointing with kwargs: {kwargs}")
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self, **kwargs):
        print(f"Disabling gradient checkpointing with kwargs: {kwargs}")
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask=None,
        head_mask=None,
        past_key_values: list[kv_cache.KVCache]=None,
        use_cache=False,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        batch_size, seq_length = inputs_embeds.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=inputs_embeds.device)
        
        hidden_states = self.dropout(inputs_embeds)
        
        # Initialize lists to store outputs for each layer
        all_hidden_states: Optional[list] = [] if output_hidden_states else None
        all_attentions: Optional[list] = [] if output_attentions else None

        for i, block in enumerate(self.transformer):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if not self.training and all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            
            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    head_mask,
                    past_key_value,
                    use_cache,
                    output_attentions,
                    output_hidden_states,
                    return_dict,
                )
            else:
                outputs = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    past_key_values=past_key_value,
                    use_cache=use_cache,
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

            # [3] is past_key_value

            if isinstance(block, causal.MegaTransformerRecurrentBlock):
                if not return_dict:
                    n_steps_no_grad = outputs[5]
                    k_steps_grad = outputs[6]
                else:
                    n_steps_no_grad = outputs.n_steps_no_grad
                    k_steps_grad = outputs.k_steps_grad

            if not self.training:
                if hasattr(outputs, "all_hidden_states") and all_hidden_states is not None:
                    all_hidden_states.extend(outputs.all_hidden_states)
                elif hasattr(outputs, "hidden_states") and all_hidden_states is not None:
                    all_hidden_states.append(outputs.hidden_states)
            
                if all_attentions:
                    all_attentions.append(attention_probs)
        
        if self.norm_final is not None:
            hidden_states = self.norm_final(hidden_states)
        
        if not self.training and all_hidden_states is not None:
            all_hidden_states.append(hidden_states)
        
        if not return_dict:
            return (
                hidden_states,
                past_key_values,
                all_hidden_states,
                all_attentions,
                n_steps_no_grad,
                k_steps_grad,
            )
        
        return causal.MegaTransformerCausalOutput(
            logits=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            n_steps_no_grad=n_steps_no_grad,
            k_steps_grad=k_steps_grad,
        )
    
    def _gradient_checkpointing_func(self, module, *args, **kwargs):
        """Wrapper for gradient checkpointing."""
        def custom_forward(*inputs):
            return module(*inputs)
        
        return torch.utils.checkpoint.checkpoint(
            custom_forward,
            *args,
            use_reentrant=True,
        )

    def prepare_inputs_for_generation(self, inputs_embeds, past_key_values=None, attention_mask: torch.Tensor=None, **kwargs):
        if past_key_values is not None and past_key_values[0] is not None:
            inputs_embeds = inputs_embeds[:, -1:]

        use_cache = kwargs.get("use_cache", True)
        position_ids = kwargs.get("position_ids", None)
        
        if position_ids is not None:
            position_ids = position_ids[:, -1:] if position_ids is not None else None

        return {
            "inputs_embeds": inputs_embeds,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }


class MegaTransformerRecurrentCausalModel(PreTrainedModel, GenerationMixin):
    config_class = configuration.MegaTransformerConfig
    
    def __init__(self, config: configuration.MegaTransformerConfig):
        super().__init__(config)
        self.config = config
        
        # embedding and prelude
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)

        self.wpe: Optional[Union[nn.Embedding, nn.Parameter]] = None
        if config.use_positional_embedding:
            self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        elif config.use_sinusoidal_embedding:
            self.wpe = nn.Parameter(create_sinusoidal_1d_pos_encoding(config.max_position_embeddings, config.hidden_size))
            self.wpe.requires_grad = config.sinusoidal_embedding_learnable
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.recurrent = MegaTransformerRawEmbedsRecurrentCausalModel(config)
        
        self.apply(self._init_weights)

    def gradient_checkpointing_enable(self, **kwargs):
        print(f"Enabling gradient checkpointing with kwargs: {kwargs}")
        self.recurrent.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self, **kwargs):
        print(f"Disabling gradient checkpointing with kwargs: {kwargs}")
        self.recurrent.gradient_checkpointing_disable(**kwargs)

    def _init_weights(self, module):
        """Initialize weights as in the original implementation"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_input_embeddings(self):
        return self.wte
    
    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values: list[kv_cache.KVCache]=None,
        use_cache=False,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        else:
            batch_size, seq_length = inputs_embeds.shape[:2]
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)
        
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        
        if self.wpe is not None:
            if isinstance(self.wpe, nn.Parameter):
                position_embeds = self.wpe
                position_embeds = position_embeds[:, :seq_length, :].expand(batch_size, -1, -1)
            else:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
        else:
            # positional embedding likely applied in self attention block
            hidden_states = inputs_embeds
        
        hidden_states = self.dropout(hidden_states)

        # Initialize lists to store outputs for each layer
        all_hidden_states: Optional[list] = [] if output_hidden_states else None
        all_attentions: Optional[list] = [] if output_attentions else None
        
        recurrent_outputs = self.recurrent(
            hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if not self.training:
            if all_hidden_states is not None:
                all_hidden_states.extend(recurrent_outputs.all_hidden_states)
                all_hidden_states.append(recurrent_outputs.hidden_states)
            if all_attentions is not None:
                all_attentions.extend(recurrent_outputs.all_attentions)

        if not return_dict:
            return (
                recurrent_outputs.logits,
                past_key_values,
                all_hidden_states,
                all_attentions,
                recurrent_outputs.n_steps_no_grad,
                recurrent_outputs.k_steps_grad,
            )
        
        return causal.MegaTransformerCausalOutput(
            logits=recurrent_outputs.logits,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            n_steps_no_grad=recurrent_outputs.n_steps_no_grad,
            k_steps_grad=recurrent_outputs.k_steps_grad,
        )
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask: torch.Tensor=None, **kwargs):
        if past_key_values is not None and past_key_values[0] is not None:
            input_ids = input_ids[:, -1:]

        use_cache = kwargs.get("use_cache", True)
        position_ids = kwargs.get("position_ids", None)
        
        if position_ids is not None:
            position_ids = position_ids[:, -1:] if position_ids is not None else None

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }


class MegaTransformerRecurrentCausalLMHead(PreTrainedModel, GenerationMixin):
    config_class = configuration.MegaTransformerConfig
    
    def __init__(self, config: configuration.MegaTransformerConfig):
        super().__init__(config)
        self.transformer = MegaTransformerRecurrentCausalModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.apply(self._init_weights)
        
        if config.tie_word_embeddings:
            self.tie_weights()
    
    def tie_weights(self):
        self._tie_or_clone_weights(self.lm_head, self.transformer.get_input_embeddings())
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_input_embeddings(self):
        return self.transformer.wte
    
    def set_input_embeddings(self, new_embeddings):
        self.transformer.wte = new_embeddings
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        past_key_values: list[kv_cache.KVCache]=None,
        use_cache=False,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        past_key_values = past_key_values if past_key_values is not None else [None] * (self.config.n_prelude_layers + 1 + self.config.n_coda_layers)
        # if use_cache:
        #     # initialize kv caches for prelude layers as past_key_values[0:n_prelude_layers] = KVCache()
        #     for i in range(len(self.config.n_prelude_layers)):
        #         if past_key_values[i] is None:
        #             past_key_values[i] = kv_cache.KVCache()

        #     # initialize kv caches for recurrent layers as past_key_values[n_prelude_layers] = list[KVCache()]
        #     if past_key_values[self.config.n_prelude_layers] is None:
        #         # use a list of KVCache() for the recurrent layer
        #         recurrent_kv_cache = [None] * (self.config.recurrent_mean_thinking_steps * 2)
        #         for j in range(len(recurrent_kv_cache)):
        #             recurrent_kv_cache[j] = kv_cache.KVCache()
        #         past_key_values[self.config.n_prelude_layers] = recurrent_kv_cache

        #     # initialize kv caches for coda layers as past_key_values[n_prelude_layers+1:] = KVCache()
        #     for i in range(self.config.n_prelude_layers + 1, len(past_key_values)):
        #         if past_key_values[i] is None:
        #             past_key_values[i] = kv_cache.KVCache()
        # print(f"initialized past_key_values: {past_key_values}")
        
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if return_dict:
            # dict
            hidden_states = transformer_outputs.logits
        else:
            # tuple
            hidden_states = transformer_outputs[0]
        
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        if not return_dict:
            output = (
                lm_logits,
                *transformer_outputs[1:],
            )
            return ((loss,) + output) if loss is not None else output
        
        return causal.MegaTransformerCausalOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            n_steps_no_grad=transformer_outputs.n_steps_no_grad,
            k_steps_grad=transformer_outputs.k_steps_grad,
        )
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return self.transformer.prepare_inputs_for_generation(input_ids, **kwargs)


tiny_config = lambda tokenizer, max_position_embeddings: configuration.MegaTransformerConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=max_position_embeddings,
    n_layers=None,
    hidden_size=64,
    d_queries=16,
    d_values=16,
    n_query_groups=4,
    n_heads=4,
    intermediate_size=256,
    n_prelude_layers=1,
    n_recurrent_layers=1,
    n_coda_layers=1,
    intermediate_activation="relu",
    norm_type="layernorm",
    ffn_type="mlp",
    use_positional_embedding=False,
    use_sinusoidal_embedding=False,
    use_rotary_embedding=True,
    use_alibi_bias=False,
    use_qkv_bias=False,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

def create_model(config: configuration.MegaTransformerConfig):
    return MegaTransformerRecurrentCausalLMHead(config)


model_config_lookup = { 
    "tiny": lambda tokenizer, max_position_embeddings, **kwargs: create_model(
        config=tiny_config(tokenizer, max_position_embeddings),
        **kwargs
    ),
}
