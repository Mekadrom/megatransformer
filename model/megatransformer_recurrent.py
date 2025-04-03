from torch import nn
from transformers import PreTrainedModel, GenerationMixin
from typing import Optional, Union

from model import megatransformer_blocks

import megatransformer_utils
import torch


class MegaTransformerRawEmbedsRecurrentCausalModel(PreTrainedModel, GenerationMixin):
    config_class = megatransformer_utils.MegaTransformerConfig
    
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__(config)
        self.config = config
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        prelude = nn.ModuleList([megatransformer_blocks.MegaTransformerBlock(config) for _ in range(config.n_prelude_layers)])
        recurrent = megatransformer_blocks.MegaTransformerRecurrentBlock(config)
        coda = nn.ModuleList([megatransformer_blocks.MegaTransformerBlock(config) for _ in range(config.n_coda_layers)])

        self.transformer = nn.ModuleList([*prelude, recurrent, *coda])
        
        if config.use_final_norm:
            self.norm_final = megatransformer_utils.create_norm(config.hidden_size, config.norm_type, config.norm_eps)
        else:
            self.norm_final = None

    def forward(
        self,
        inputs_embeds,
        attention_mask=None,
        past_key_values: list[megatransformer_utils.KVCache]=None,
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
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
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

            if isinstance(block, megatransformer_blocks.MegaTransformerRecurrentBlock):
                if not return_dict:
                    n_steps_no_grad = outputs[4]
                    k_steps_grad = outputs[5]
                else:
                    n_steps_no_grad = outputs.n_steps_no_grad
                    k_steps_grad = outputs.k_steps_grad
            else:
                n_steps_no_grad = None
                k_steps_grad = None

            if hasattr(outputs, "all_hidden_states") and all_hidden_states is not None:
                all_hidden_states.extend(outputs.all_hidden_states)
            elif hasattr(outputs, "hidden_states") and all_hidden_states is not None:
                all_hidden_states.append(outputs.hidden_states)
            
            if all_attentions:
                all_attentions.append(attention_probs)
        
        if self.norm_final is not None:
            hidden_states = self.norm_final(hidden_states)
        
        if all_hidden_states is not None:
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
        
        return megatransformer_utils.MegaTransformerCausalOutput(
            logits=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            n_steps_no_grad=n_steps_no_grad,
            k_steps_grad=k_steps_grad,
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

class MegaTransformerRecurrentCausalLMHead(PreTrainedModel, GenerationMixin):
    config_class = megatransformer_utils.MegaTransformerConfig
    
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
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
        past_key_values: list[megatransformer_utils.KVCache]=None,
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
        #             past_key_values[i] = megatransformer_utils.KVCache()

        #     # initialize kv caches for recurrent layers as past_key_values[n_prelude_layers] = list[KVCache()]
        #     if past_key_values[self.config.n_prelude_layers] is None:
        #         # use a list of KVCache() for the recurrent layer
        #         recurrent_kv_cache = [None] * (self.config.recurrent_mean_thinking_steps * 2)
        #         for j in range(len(recurrent_kv_cache)):
        #             recurrent_kv_cache[j] = megatransformer_utils.KVCache()
        #         past_key_values[self.config.n_prelude_layers] = recurrent_kv_cache

        #     # initialize kv caches for coda layers as past_key_values[n_prelude_layers+1:] = KVCache()
        #     for i in range(self.config.n_prelude_layers + 1, len(past_key_values)):
        #         if past_key_values[i] is None:
        #             past_key_values[i] = megatransformer_utils.KVCache()
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
        
        return megatransformer_utils.MegaTransformerCausalOutput(
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

def create_recurrent_small_model(tokenizer, max_position_embeddings):
    # uses a recurrent approach to emulate a deeper model (~120M params)
    return MegaTransformerRecurrentCausalLMHead(megatransformer_utils.MegaTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=max_position_embeddings,
        n_layers=None,
        n_prelude_layers=2,
        n_recurrent_layers=4,
        n_coda_layers=2,
        intermediate_activation="swiglu",
        norm_type="rmsnorm",
        ffn_type="mlp",
        use_positional_embedding=False,
        use_sinusoidal_embedding=False,
        use_rotary_embedding=False,
        use_alibi_bias=True,
        use_qkv_bias=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    ))

def create_recurrent_small_model_with_sandwich_norm(tokenizer, max_position_embeddings):
    # uses a recurrent approach to emulate a deeper model
    return MegaTransformerRecurrentCausalLMHead(megatransformer_utils.MegaTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=max_position_embeddings,
        n_layers=None,
        n_prelude_layers=2,
        n_recurrent_layers=4,
        n_coda_layers=2,
        intermediate_activation="swiglu",
        norm_type="rmsnorm",
        ffn_type="mlp",
        # all norms enabled for sandwich norm
        post_attn_norm=True,
        post_ffn_norm=True,
        use_positional_embedding=False,
        use_sinusoidal_embedding=False,
        use_rotary_embedding=False,
        use_alibi_bias=True,
        use_qkv_bias=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    ))

def create_recurrent_medium_model(tokenizer, max_position_embeddings):
    # uses a recurrent approach to emulate a deeper model (~200M params)
    return MegaTransformerRecurrentCausalLMHead(megatransformer_utils.MegaTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=max_position_embeddings,
        hidden_size=1024,
        n_layers=None,
        n_prelude_layers=2,
        n_recurrent_layers=4,
        n_coda_layers=2,
        d_queries=64,
        d_values=64,
        n_query_groups=16,
        n_heads=16,
        intermediate_size=4096,
        intermediate_activation="swiglu",
        norm_type="rmsnorm",
        ffn_type="mlp",
        use_positional_embedding=False,
        use_sinusoidal_embedding=False,
        use_rotary_embedding=False,
        use_alibi_bias=True,
        use_qkv_bias=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    ))


def create_test_tiny_recurrent_model(tokenizer, max_position_embeddings):
    # uses a recurrent approach to emulate a deeper model (~2M params)
    return MegaTransformerRecurrentCausalLMHead(megatransformer_utils.MegaTransformerConfig(
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
    ))

lookup = { 
    "recurrent_small": create_recurrent_small_model,
    "recurrent_small_sandwich": create_recurrent_small_model_with_sandwich_norm,
    "recurrent_medium": create_recurrent_medium_model,
    "test_tiny_recurrent": create_test_tiny_recurrent_model,
}

def model_config_lookup(config):
    if config not in lookup:
        raise ValueError(f"Unknown model configuration: {config}")
    return lookup[config]
