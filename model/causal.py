import torch

from typing import Optional, Union

from torch import nn
from transformers import PreTrainedModel, GenerationMixin

from . import activations, attention, causal, kv_cache
from utils import configuration
from utils.megatransformer_utils import transformer_weight_init
from utils.model_utils import create_norm, create_sinusoidal_1d_pos_encoding, get_activation_type


class BlockOutput:
    def __init__(self, hidden_states, past_key_values: kv_cache.KVCache=None, attention_probs=None):
        self.hidden_states = hidden_states
        self.past_key_values = past_key_values
        self.attention_probs = attention_probs


class MegaTransformerCausalOutput(dict):
    def __init__(self,
        loss: Optional[torch.Tensor]=None,
        logits: Optional[torch.Tensor]=None,
        past_key_values: Optional[list[kv_cache.KVCache]]=None,
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


class SimpleFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.expand = nn.Linear(config.hidden_size, config.intermediate_size)
        self.condense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        activation_type = get_activation_type(config.intermediate_activation)
        if activation_type == activations.SwiGLU:
            self.activation = activations.SwiGLU(config.intermediate_size)
        else:
            self.activation = activation_type()
    
    def forward(self, hidden_states):
        hidden_states = self.expand(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.condense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class MegaTransformerBlock(nn.Module):
    """Self Attention followed by FFN with optional pre/post layer norms."""
    def __init__(self, config: configuration.MegaTransformerConfig):
        super().__init__()
        self.self_attn = attention.MegaTransformerSelfAttention(config)
        
        self.ffn: nn.Module
        if config.ffn_type == "mlp":
            self.ffn = causal.SimpleFFN(config)
        else:
            raise ValueError(f"Unknown ffn_type: {config.ffn_type}")

        if config.pre_attn_norm:
            self.pre_attn_norm = create_norm(config.hidden_size, config.norm_type, config.norm_eps)
        else:
            self.pre_attn_norm = None
        if config.post_attn_norm:
            self.post_attn_norm = create_norm(config.hidden_size, config.norm_type, config.norm_eps)
        else:
            self.post_attn_norm = None
        if config.pre_ffn_norm:
            self.pre_ffn_norm = create_norm(config.hidden_size, config.norm_type, config.norm_eps)
        else:
            self.pre_ffn_norm = None
        if config.post_ffn_norm:
            self.post_ffn_norm = create_norm(config.hidden_size, config.norm_type, config.norm_eps)
        else:
            self.post_ffn_norm = None

        self._init_weights()

    def _init_weights(self):
        self.apply(transformer_weight_init())
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        past_key_values: kv_cache.KVCache=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ) -> Union[tuple[torch.Tensor, kv_cache.KVCache, torch.Tensor], BlockOutput]:
        if self.pre_attn_norm is not None:
            pre_attn_input = self.pre_attn_norm(hidden_states)
        else:
            pre_attn_input = hidden_states

        if use_cache:
            if past_key_values is None:
                past_key_values = kv_cache.KVCache()

        attn_outputs = self.self_attn(
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

        return BlockOutput(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attention_probs=attention_probs,
        )

class MegaTransformerSimpleCausalModel(PreTrainedModel, GenerationMixin):
    config_class = configuration.MegaTransformerConfig
    
    def __init__(self, config: configuration.MegaTransformerConfig):
        super().__init__(config)
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.wpe: Optional[Union[nn.Embedding, nn.Parameter]] = None
        if config.use_positional_embedding:
            self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        elif config.use_sinusoidal_embedding:
            self.wpe = nn.Parameter(create_sinusoidal_1d_pos_encoding(config.max_position_embeddings, config.hidden_size))
            self.wpe.requires_grad = config.sinusoidal_embedding_learnable

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.transformer = nn.ModuleList([MegaTransformerBlock(config) for _ in range(config.n_layers)])

        if config.use_final_norm:
            self.norm_final = create_norm(config.hidden_size, config.norm_type, config.norm_eps)
        else:
            self.norm_final = None
        
        self.gradient_checkpointing = False

        self.apply(self._init_weights)

    def gradient_checkpointing_enable(self, **kwargs):
        print(f"Unexpected kwargs in gradient_checkpointing_enable: {kwargs}")
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self, **kwargs):
        print(f"Unexpected kwargs in gradient_checkpointing_disable: {kwargs}")
        self.gradient_checkpointing = False

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
            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i] if head_mask is not None else None,
                    past_key_values=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            else:
                if not self.training and all_hidden_states is not None:
                    all_hidden_states.append(hidden_states)
                
                outputs = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i] if head_mask is not None else None,
                    past_key_values=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

            hidden_states = outputs.hidden_states
            attention_probs = outputs.attention_probs

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
            )
        
        return kv_cache.MegaTransformerCausalOutput(
            logits=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
    
    def _gradient_checkpointing_func(self, module_call, *args, **kwargs):
        """Wrapper for gradient checkpointing."""
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward
        
        return torch.utils.checkpoint.checkpoint(
            create_custom_forward(module_call),
            *args,
            **kwargs,
            use_reentrant=True
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
    config_class = configuration.MegaTransformerConfig
    
    def __init__(self, config: configuration.MegaTransformerConfig):
        super().__init__(config)
        self.transformer = MegaTransformerSimpleCausalModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.gradient_checkpointing = False
        
        self.apply(self._init_weights)
        
        if config.tie_word_embeddings:
            self.tie_weights()
    
    def gradient_checkpointing_enable(self, **kwargs):
        print(f"Unexpected kwargs in gradient_checkpointing_enable: {kwargs}")
        self.transformer.gradient_checkpointing_enable()
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self, **kwargs):
        print(f"Unexpected kwargs in gradient_checkpointing_disable: {kwargs}")
        self.transformer.gradient_checkpointing_disable()
        self.gradient_checkpointing = False

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

        past_key_values = past_key_values if past_key_values is not None else [None] * self.config.n_layers
        if use_cache:
            for i in range(len(past_key_values)):
                if past_key_values[i] is None:
                    past_key_values[i] = kv_cache.KVCache()
        
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
        )
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return self.transformer.prepare_inputs_for_generation(input_ids, **kwargs)


def create_gpt2_tiny_model(tokenizer, max_position_embeddings):
    # debug model size, ~9.5M params
    return MegaTransformerCausalLMHead(configuration.MegaTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=max_position_embeddings,
        hidden_size=64,
        n_layers=12,
        d_queries=64,
        d_values=64,
        n_query_groups=12,
        n_heads=12,
        intermediate_size=3072,
        intermediate_activation="relu",
        use_qkv_bias=False,
        use_cache=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    ))

def create_gpt2_small_model(tokenizer, max_position_embeddings):
    # gpt2-small closest equivalent (~124M params)
    return MegaTransformerCausalLMHead(configuration.MegaTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=max_position_embeddings,
        use_cache=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    ))

def create_modern_small_model(tokenizer, max_position_embeddings):
    # uses more modern approaches to causal language modeling (~148M params)
    return MegaTransformerCausalLMHead(configuration.MegaTransformerConfig(
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
    return MegaTransformerCausalLMHead(configuration.MegaTransformerConfig(
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

def create_test_tiny_model(tokenizer, max_position_embeddings):
    # uses a recurrent approach to emulate a deeper model (~2M params)
    return MegaTransformerCausalLMHead(configuration.MegaTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=max_position_embeddings,
        n_layers=3,
        hidden_size=64,
        d_queries=16,
        d_values=16,
        n_query_groups=4,
        n_heads=4,
        intermediate_size=256,
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
    "gpt2_tiny": create_gpt2_tiny_model,
    "gpt2_small": create_gpt2_small_model,
    "modern_small": create_modern_small_model,
    "modern_medium": create_modern_medium_model,
    "test_tiny": create_test_tiny_model,
}

def model_config_lookup(config):
    if config not in lookup:
        raise ValueError(f"Unknown model configuration: {config}")
    return lookup[config]
