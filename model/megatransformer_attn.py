from rotary_embedding_torch import RotaryEmbedding
from torch import nn

from model import swiglu

import math
import megatransformer_utils
import torch
import torch.nn.functional as F


class SelfAttentionOutput:
    def __init__(self, hidden_states: torch.Tensor, past_key_values, attention_probs: torch.Tensor=None):
        self.hidden_states: torch.Tensor = hidden_states
        self.past_key_values = past_key_values
        self.attention_probs: torch.Tensor = attention_probs

    def __deepspeed_tensor_attributes__(self):
        return ['hidden_states', 'attention_probs']

class MegaTransformerSelfAttention(nn.Module):
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.hidden_size = config.hidden_size
        self.use_grok_scaled_attn = config.use_grok_scaled_attn

        self.d_queries = config.d_queries
        self.d_values = config.d_values
        self.n_query_groups = config.n_query_groups
        self.n_heads = config.n_heads

        self.q_proj = nn.Linear(config.hidden_size, self.n_query_groups * self.d_queries, bias=config.use_qkv_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.n_heads * self.d_queries, bias=config.use_qkv_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.n_heads * self.d_values, bias=config.use_qkv_bias)

        self.heads_activation = None
        if config.heads_activation is not None:
            activation_type = megatransformer_utils.get_activation_function(config.heads_activation)
            if activation_type == swiglu.SwiGLU:
                self.heads_activation = swiglu.SwiGLU(config.d_values)
            else:
                self.heads_activation = activation_type()

        self.rotary_embedding = None
        if bool(config.use_rotary_embedding):
            self.rotary_embedding = RotaryEmbedding(dim=config.rotary_embedding_dim, learned_freq=config.rotary_embedding_learnable)

        if bool(config.use_alibi_bias):
            self.register_buffer('alibi_bias', megatransformer_utils.create_alibi_bias(n_heads=config.n_heads, maxlen=config.max_position_embeddings))
        else:
            self.register_buffer('alibi_bias', None)
        
        self.o_proj = nn.Linear(self.n_heads * self.d_values, config.hidden_size)
        
        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.proj_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_position_embeddings, config.max_position_embeddings)).view(1, 1, config.max_position_embeddings, config.max_position_embeddings),
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor=None,
        head_mask: torch.Tensor=None,
        past_key_values: megatransformer_utils.KVCache=None,
        use_cache=False,
        output_attentions: bool=False,
    ) -> SelfAttentionOutput:
        N, _ = hidden_states.shape[:2]

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        if self.heads_activation is not None:
            queries = self.heads_activation(queries)
            keys = self.heads_activation(keys)
            values = self.heads_activation(values)

        if self.rotary_embedding is not None:
            queries = self.rotary_embedding.rotate_queries_or_keys(queries)
            keys = self.rotary_embedding.rotate_queries_or_keys(keys)
        
        queries = queries.view(N, -1, self.n_query_groups, self.d_queries).permute(0, 2, 1, 3).contiguous()
        keys_to_cache = keys.view(N, -1, self.n_heads, self.d_queries).permute(0, 2, 1, 3).contiguous()
        values_to_cache = values.view(N, -1, self.n_heads, self.d_values).permute(0, 2, 1, 3).contiguous()

        # cache keys and values in the shape (B, N, S, H/N) where N is the number of heads
        if past_key_values is not None:
            past_key, past_value = past_key_values
            keys = torch.cat([past_key, keys_to_cache], dim=-2)
            values = torch.cat([past_value, values_to_cache], dim=-2)
        else:
            keys = keys_to_cache
            values = values_to_cache

        if use_cache:
            if past_key_values is None:
                past_key_values = megatransformer_utils.KVCache()
            past_key_values.update(key=keys_to_cache, value=values_to_cache)

        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))

        _, _, t = queries.shape[:3]
        _, _, T = keys.shape[:3]

        if self.alibi_bias is not None:
            attention_scores = attention_scores + self.alibi_bias[:, :t, :t].unsqueeze(0).repeat(N, 1, 1, 1)

        attention_scores = attention_scores / math.sqrt(self.d_queries)
        if bool(self.use_grok_scaled_attn):
            attention_scores = 30.0 * torch.tanh(attention_scores / 30.0)
        
        causal_mask_slice = self.causal_mask[:, :, :t, :T]
        attention_scores = attention_scores.masked_fill(causal_mask_slice == 0, float("-inf"))
        
        if attention_mask is not None:
            # HuggingFace uses 1 for tokens to attend to and 0 for masked tokens
            # which is the opposite of the original transformer implementation
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float("-inf"))
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        context_layer = torch.matmul(attention_probs, values)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Apply output projection
        output = self.o_proj(context_layer)
        output = self.proj_dropout(output)
        
        return SelfAttentionOutput(
            hidden_states=output,
            past_key_values=past_key_values,
            attention_probs=attention_probs if output_attentions else None,
        )
