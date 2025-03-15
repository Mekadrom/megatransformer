from torch import nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from typing import Optional, Union

from model import megatransformer

import megatransformer_utils
import torch

class MegaTransformerCausalOutput(dict):
    def __init__(self,
        loss: Optional[torch.Tensor]=None,
        logits: Optional[torch.Tensor]=None,
        past_key_values: Optional[list[megatransformer_utils.KVCache]]=None,
        hidden_states: Optional[list]=None,
        attentions: Optional[list]=None,
        n_steps_no_grad: Optional[int]=None,
        k_steps_grad: Optional[int]=None,
    ):
        self.loss = loss
        self.logits = logits
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.n_steps_no_grad = n_steps_no_grad
        self.k_steps_grad = k_steps_grad

        super().__init__(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
            n_steps_no_grad=n_steps_no_grad,
            k_steps_grad=k_steps_grad,
        )


class MegaTransformerSimpleCausalModel(PreTrainedModel, GenerationMixin):
    config_class = megatransformer_utils.MegaTransformerConfig
    
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__(config)
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.wpe: Optional[Union[nn.Embedding, nn.Parameter]] = None
        if config.use_positional_embedding:
            self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        elif config.use_sinusoidal_embedding:
            self.wpe = nn.Parameter(megatransformer_utils.create_sinusoidal_embedding(config.max_position_embeddings, config.hidden_size))
            self.wpe.requires_grad = config.sinusoidal_embedding_learnable

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        if config.n_layers is not None:
            self.transformer = nn.ModuleList([megatransformer.MegaTransformerBlock(config) for _ in range(config.n_layers)])
        else:
            prelude = nn.ModuleList([megatransformer.MegaTransformerBlock(config) for _ in range(config.n_prelude_layers)])
            recurrent = megatransformer.MegaTransformerRecurrentBlock(config)
            coda = nn.ModuleList([megatransformer.MegaTransformerBlock(config) for _ in range(config.n_coda_layers)])

            self.transformer = nn.ModuleList([*prelude, recurrent, *coda])
        
        self.norm_final = megatransformer_utils.create_norm(config)
        
        self.apply(self._init_weights)

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
        past_key_values: list[megatransformer_utils.KVCache]=None,
        use_cache=False,
        token_type_ids=None,
        position_ids=None,
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
        
        # todo: find out what this does and if it is needed for huginn
        if self.config.n_layers is not None:
            head_mask = self.get_head_mask(head_mask, self.config.n_layers)
        
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        
        if self.wpe is not None:
            if isinstance(self.wpe, nn.Parameter):
                position_embeds = self.wpe
                position_embeds = position_embeds[:, :seq_length, :].expand(batch_size, -1, -1)
            else:
                if position_ids is None:
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
        
        for i, (block, past_key_value) in enumerate(zip(self.transformer, past_key_values)):
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i] if head_mask is not None else None,
                past_key_values=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states
            )

            hidden_states = outputs.hidden_states
            attention_probs = outputs.attention_probs

            if hasattr(outputs, "all_hidden_states") and all_hidden_states is not None:
                all_hidden_states.extend(outputs.all_hidden_states)
            
            if all_attentions:
                all_attentions.append(attention_probs)
        
        hidden_states = self.norm_final(hidden_states)
        
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)
        
        if not return_dict:
            return (
                hidden_states,
                past_key_values,
                all_hidden_states,
                all_attentions,
                outputs.n_steps_no_grad if hasattr(outputs, "n_steps_no_grad") else None,
                outputs.k_steps_grad if hasattr(outputs, "k_steps_grad") else None,
            )
        
        return MegaTransformerCausalOutput(
            logits=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            n_steps_no_grad=outputs.n_steps_no_grad if hasattr(outputs, "n_steps_no_grad") else None,
            k_steps_grad=outputs.k_steps_grad if hasattr(outputs, "k_steps_grad") else None,
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
    

class MegaTransformerCausalLMHead(PreTrainedModel, GenerationMixin):
    config_class = megatransformer_utils.MegaTransformerConfig
    
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__(config)
        self.transformer = MegaTransformerSimpleCausalModel(config)
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

        if self.config.n_layers is not None:
            past_key_values = past_key_values if past_key_values is not None else [None] * self.config.n_layers
            if use_cache:
                for i in range(len(past_key_values)):
                    if past_key_values[i] is None:
                        past_key_values[i] = megatransformer_utils.KVCache()
        else:
            past_key_values = past_key_values if past_key_values is not None else [None] * (self.config.n_prelude_layers + 1 + self.config.n_coda_layers)
            # if use_cache:
            #     # initialize kv caches for prelude layers as past_key_values[0:n_prelude_layers] = KVCache()
            #     for i in range(len(self.config.n_prelude_layers)):
            #         if past_key_values[i] is None:
            #             past_key_values[i] = megatransformer_utils.KVCache()

            #     # initialize kv caches for recurrent layers as past_key_values[n_prelude_layers] = list[KVCache()]
            #     if past_key_values[self.config.n_prelude_layers] is None:
            #         # use a list of KVCache() for the recurrent layer
            #         huginn_kv_cache = [None] * (self.config.huginn_mean_thinking_steps * 2)
            #         for j in range(len(huginn_kv_cache)):
            #             huginn_kv_cache[j] = megatransformer_utils.KVCache()
            #         past_key_values[self.config.n_prelude_layers] = huginn_kv_cache

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
        
        return MegaTransformerCausalOutput(
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


def create_gpt2_model(tokenizer, max_position_embeddings):
    # gpt2-small closest equivalent (~124M params)
    return MegaTransformerCausalLMHead(megatransformer_utils.MegaTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=max_position_embeddings,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    ))

def create_modern_model(tokenizer, max_position_embeddings):
    # uses more modern approaches to causal language modeling (~148M params)
    return MegaTransformerCausalLMHead(megatransformer_utils.MegaTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=max_position_embeddings,
        intermediate_activation="swiglu",
        norm_type="rmsnorm",
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

def create_modern_medium_model(tokenizer, max_position_embeddings):
    # uses more modern approaches to causal language modeling (~536M params)
    return MegaTransformerCausalLMHead(megatransformer_utils.MegaTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=max_position_embeddings,
        hidden_size=1024,
        n_layers=24,
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
        use_rotary_embedding=True,
        use_alibi_bias=False,
        use_qkv_bias=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    ))

def create_huginn_model(tokenizer, max_position_embeddings):
    # uses a recurrent approach to emulate a deeper model
    return MegaTransformerCausalLMHead(megatransformer_utils.MegaTransformerConfig(
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

def create_huginn_sandwich_model(tokenizer, max_position_embeddings):
    # uses a recurrent approach to emulate a deeper model
    return MegaTransformerCausalLMHead(megatransformer_utils.MegaTransformerConfig(
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

lookup = { 
    "gpt2": create_gpt2_model,
    "modern": create_modern_model,
    "modern_medium": create_modern_medium_model,
    "huginn": create_huginn_model,
    "huginn_sandwich": create_huginn_sandwich_model,
}

def model_config_lookup(config):
    if config not in lookup:
        raise ValueError(f"Unknown model configuration: {config}")
    return lookup[config]
