from rotary_embedding_torch import RotaryEmbedding
from torch import nn

import math
import megatransformer_utils
import torch
import torch.nn.functional as F


class SelfAttentionOutput:
    def __init__(self, hidden_states: torch.Tensor, past_key_values: megatransformer_utils.KVCache, attention_probs: torch.Tensor=None):
        self.hidden_states: torch.Tensor = hidden_states
        self.past_key_values: megatransformer_utils.KVCache = past_key_values
        self.attention_probs: torch.Tensor = attention_probs

class MegaTransformerSelfAttention(nn.Module):
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.use_grok_scaled_attn = config.use_grok_scaled_attn

        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_qkv_bias)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_qkv_bias)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_qkv_bias)

        self.rotary_embedding = None
        if bool(config.use_rotary_embedding):
            self.rotary_embedding = RotaryEmbedding(dim=config.rotary_embedding_dim, learned_freq=config.rotary_embedding_learnable)

        if bool(config.use_alibi_bias):
            self.register_buffer('alibi_bias', megatransformer_utils.create_alibi_bias(n_heads=config.num_attention_heads, maxlen=config.max_position_embeddings))
        else:
            self.register_buffer('alibi_bias', None)
        
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.proj_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_position_embeddings, config.max_position_embeddings)).view(1, 1, config.max_position_embeddings, config.max_position_embeddings),
        )
    
    def transpose_for_scores(self, x):
        """Reshape from [B, S, H] to [B, N, S, H/N] where N is the number of heads"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
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

        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        if self.rotary_embedding is not None:
            query_layer = self.rotary_embedding.rotate_queries_or_keys(query_layer)
            key_layer = self.rotary_embedding.rotate_queries_or_keys(key_layer)
        
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        # cache keys and values in the shape (B, N, S, H/N) where N is the number of heads
        if past_key_values is not None:
            past_key, past_value = past_key_values.key, past_key_values.value
            if past_key is not None and past_value is not None:
                k = torch.cat([past_key, key_layer], dim=-2)
                v = torch.cat([past_value, value_layer], dim=-2)
            else:
                k = key_layer
                v = value_layer
        else:
            k = key_layer
            v = value_layer

        if use_cache:
            if past_key_values is None:
                past_key_values = megatransformer_utils.KVCache()
            past_key_values.update(key=key_layer, value=value_layer)

        attention_scores = torch.matmul(query_layer, k.transpose(-1, -2))

        if self.alibi_bias is not None:
            attention_scores = attention_scores + self.alibi_bias[:, :t, :t].unsqueeze(0).repeat(N, 1, 1, 1)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if bool(self.use_grok_scaled_attn):
            attention_scores = 30.0 * torch.tanh(attention_scores / 30.0)
        
        _, _, t = query_layer.shape[:3]
        _, _, T = key_layer.shape[:3]

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
        
        context_layer = torch.matmul(attention_probs, v)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Apply output projection
        output = self.output(context_layer)
        output = self.proj_dropout(output)
        
        return SelfAttentionOutput(
            hidden_states=output,
            past_key_values=past_key_values,
            attention_probs=attention_probs if output_attentions else None,
        )
