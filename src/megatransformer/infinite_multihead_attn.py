from rotary_embedding_torch import RotaryEmbedding
from torch import nn
from typing import Optional

from . import transformer_utils

import torch
import torch.nn.functional as F

class InfiniteMultiHeadAttention(nn.Module):
    def __init__(self, device, model_config, attn_config, self_attn, in_decoder=False):
        super(InfiniteMultiHeadAttention, self).__init__()

        self.device = device
        self.attn_config = attn_config
        self.self_attn = self_attn
        self.in_decoder = in_decoder  

        self.maxlen = model_config.maxlen
        self.d_model = model_config.d_model
        self.dropout = model_config.dropout
        self.positional_encoding_type = model_config.positional_encoding_type
        self.positional_encoding_dim = model_config.positional_encoding_dim
        self.learnable_positional_encoding = model_config.learnable_positional_encoding
        self.norm = model_config.norm
        self.norm_eps = model_config.norm_eps

        self.n_gqa_groups = attn_config.n_gqa_groups
        self.n_heads = attn_config.n_heads
        self.d_queries = attn_config.d_queries
        self.d_values = attn_config.d_values
        self.q_bias = attn_config.q_bias
        self.k_bias = attn_config.k_bias
        self.v_bias = attn_config.v_bias
        self.o_bias = attn_config.o_bias
        self.heads_activation_function = attn_config.heads_activation_function
        self.infinite_attention_n_segments = attn_config.infinite_attention_n_segments
        self.infinite_attention_update = attn_config.infinite_attention_update
        self.use_grok_scaled_attn = attn_config.use_grok_scaled_attn

        if self.positional_encoding_type == 'rotary':
            self.rotary_embedding = RotaryEmbedding(dim=self.positional_encoding_dim, learned_freq=self.learnable_positional_encoding).to(device)
            self.alibi_bias = None
        elif self.positional_encoding_type == 'alibi':
            self.alibi_bias = nn.Parameter(transformer_utils.create_alibi_bias(n_heads=self.n_heads, maxlen=self.maxlen), requires_grad=False)
            self.rotary_embedding = None
        else:
            self.rotary_embedding = None
            self.alibi_bias = None

        self.n_q_heads = self.n_gqa_groups * self.n_heads

        # A linear projection to cast (n_kv_heads sets of) queries from the input query sequences
        self.q_proj = nn.Linear(self.d_model, self.n_q_heads * self.d_queries, bias=self.q_bias) # (N, query_sequence_pad_length, n_kv_heads * d_queries)
        # A linear projection to cast (n_kv_heads sets of) keys and values from the input reference sequences
        self.k_proj = nn.Linear(self.d_model, self.n_heads * self.d_queries, bias=self.k_bias) # (N, key_value_sequence_pad_length, n_kv_heads * d_queries)
        self.v_proj = nn.Linear(self.d_model, self.n_heads * self.d_values, bias=self.v_bias) # (N, key_value_sequence_pad_length, n_kv_heads * d_values)

        # a linear projection to cast (n_q_heads sets of) computed attention-weighted vectors to output vectors
        self.o_proj = nn.Linear(self.n_heads * self.d_values, self.d_model, bias=self.o_bias)

        self.qkv_norm = self.norm(self.d_model, self.norm_eps)

        self.attn_softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.output_dropout = nn.Dropout(self.dropout)

        self.heads_activation = None
        if self.heads_activation_function is not None:
            self.heads_activation = transformer_utils.create_activation_function(self.d_model, self.heads_activation_function)

        assert self.maxlen % self.infinite_attention_n_segments == 0, "maxlen must be divisible by infinite_attention_n_segments"

        self.infinite_attn_segment_size = (self.maxlen // self.infinite_attention_n_segments)

        # eg: 6 * 8 * 64 -> 48 * 64 = 2048 -> 64
        self.k_memory_compression = nn.Sequential(
            nn.Linear(self.infinite_attn_segment_size * self.d_queries, self.d_queries),
            nn.ELU(),
        )
        self.v_memory_compression = nn.Sequential(
            nn.Linear(self.infinite_attn_segment_size * self.d_values, self.d_values),
            nn.ELU(),
        )

        self.register_buffer('causal_mask', torch.tril(torch.ones(self.maxlen + 1, self.maxlen + 1).to(self.device)).to(torch.bool))

    def mask_attention(self, attention_weights: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # mask away tokens by setting such weights to a large negative number, so that they evaluate to 0 under the softmax
        # if attention_mask is not None:
        #     assert attention_mask.shape[0] == attention_weights.shape[0], f"batch dimension for padding is wrong: {attention_mask.shape[0]} != {attention_weights.shape[0]}. overall shape: {attention_mask.shape} != {attention_weights.shape}"
        #     assert attention_mask.shape[1] == attention_weights.shape[3], f"padding mask length is wrong: {attention_mask.shape[1]} != {attention_weights.shape[3]}. overall shape: {attention_mask.shape} != {attention_weights.shape}"

        #     attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).to(torch.bool)

        #     attention_weights = attention_weights.masked_fill_(attention_mask, -float('inf'))
        if self.self_attn:
            attention_weights = attention_weights.masked_fill_(~self.causal_mask[:attention_weights.shape[-2], :attention_weights.shape[-1]], -float('inf'))

        return attention_weights

    def forward(self,
                query_sequences: torch.Tensor,
                key_sequences: torch.Tensor,
                value_sequences: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_attn_values: Optional[bool] = False) -> tuple[torch.Tensor, list[torch.Tensor]]:
        query_sequences = self.qkv_norm(query_sequences)

        # if this isn't self-attention, they will already have been normed in the last layer of the Encoder (from whence they came)
        if self.self_attn:
            key_sequences = self.qkv_norm(key_sequences)
            value_sequences = self.qkv_norm(value_sequences)

        N, t, _ = query_sequences.shape
        N, T, _ = key_sequences.shape

        q_heads: torch.Tensor = self.q_proj(query_sequences)
        k_heads: torch.Tensor = self.k_proj(key_sequences)
        v_heads: torch.Tensor = self.v_proj(value_sequences)

        if self.heads_activation is not None:
            q_heads = self.heads_activation(q_heads)
            k_heads = self.heads_activation(k_heads)
            v_heads = self.heads_activation(v_heads)

        if t < self.maxlen:
            # pad to maxlen with zeros prepended along the sequence length
            q_heads = torch.cat([torch.zeros(N, self.maxlen - t, self.n_heads * self.n_gqa_groups * self.d_queries).to(self.device), q_heads], dim=1)
            t = self.maxlen
        if T < self.maxlen:
            # pad to maxlen with zeros prepended along the sequence length
            k_heads = torch.cat([torch.zeros(N, self.maxlen - T, self.n_heads * self.d_values).to(self.device), k_heads], dim=1)
            v_heads = torch.cat([torch.zeros(N, self.maxlen - T, self.n_heads * self.d_values).to(self.device), v_heads], dim=1)
            T = self.maxlen

        # Split the last dimension by the n_kv_heads subspaces
        q_heads = q_heads.contiguous().view(N, t, self.n_gqa_groups, self.n_heads, self.d_queries) # (N, query_sequence_pad_length, n_gqa_groups, n_heads, d_queries)
        k_heads = k_heads.contiguous().view(N, T, self.n_heads, self.d_queries) # (N, key_value_sequence_pad_length, n_heads, d_queries)
        v_heads = v_heads.contiguous().view(N, T, self.n_heads, self.d_values) # (N, key_value_sequence_pad_length, n_heads, d_values)

        q_heads = q_heads.permute(0, 2, 3, 1, 4) # Nghtq
        k_heads = k_heads.permute(0, 2, 1, 3) # NhTq
        v_heads = v_heads.permute(0, 2, 1, 3) # NhTv

        if self.rotary_embedding is not None:
            q_heads = self.rotary_embedding.rotate_queries_or_keys(q_heads, seq_dim=-2)
            k_heads = self.rotary_embedding.rotate_queries_or_keys(k_heads.unsqueeze(0), seq_dim=-2).squeeze(0) # adds a singleton dimension for the rotation operation and then removes it for the torch compiler

        # produces a list of tensors of shape Nhsq or Nhsv where s = maxlen // infinite_attention_n_segments
        k_segments = k_heads.split(self.infinite_attn_segment_size, dim=-2)
        v_segments = v_heads.split(self.infinite_attn_segment_size, dim=-2)

        # grabs last two infinite attention chunks, produces tensor of shape Nhrq or Nhrv where r = 2 * (maxlen // infinite_attention_n_segments) eg r = 2 * (256 // 8) = 64, so tensor would be of shape Nh(64)q or Nh(64)v
        recent_k = torch.cat(k_segments[-2:], dim=-2)
        recent_v = torch.cat(v_segments[-2:], dim=-2)

        # Nhiq or Nhiv where i = (maxlen // infinite_attention_n_segments) - 2 = 30, so tensor would be of shape Nh(30)q or Nh(30)v
        k_segments = torch.stack(k_segments[:-2], dim=2).to(k_heads.device).to(k_heads.dtype)
        v_segments = torch.stack(v_segments[:-2], dim=2).to(v_heads.device).to(v_heads.dtype)

        k_summaries = self.k_memory_compression(k_segments.view(N, self.n_heads, self.infinite_attention_n_segments - 2, -1)) # Nh(i*q)
        v_summaries = self.v_memory_compression(v_segments.view(N, self.n_heads, self.infinite_attention_n_segments - 2, -1)) # Nh(i*v)

        k_heads = torch.cat([k_summaries, recent_k], dim=-2) # Nhrq
        v_heads = torch.cat([v_summaries, recent_v], dim=-2) # Nhrv

        # generate attention weights by taking the dot product of queries and keys
        attention_weights = torch.einsum('...ghtq,...hTq->...htT', q_heads, k_heads)

        if self.alibi_bias is not None:
            attention_weights = attention_weights + self.alibi_bias

        attention_weights = (1.0 / (self.d_queries ** 0.5)) * attention_weights
        if bool(self.use_grok_scaled_attn):
            attention_weights = 30.0 * torch.tanh(attention_weights / 30.0) # grok version of scaled attention

        attention_weights = self.mask_attention(attention_weights, attention_mask)
        attention_weights = self.attn_softmax(attention_weights)
        attention_weights_for_visualization = [attention_weights.clone().detach()]

        attention_weights = self.attn_dropout(attention_weights)

        # Calculate sequences as the weighted sums of values based on these softmax weights
        sequences = torch.einsum('...htT,...hTv->...htv', attention_weights, v_heads)

        sequences = sequences.permute(0, 2, 1, 3)

        # Concatenate the n_heads subspaces (each with an output of size d_values)
        sequences = sequences.contiguous().view(N, t, -1)

        sequences = self.o_proj(sequences)
        sequences = self.output_dropout(sequences)

        return sequences, attention_weights_for_visualization
