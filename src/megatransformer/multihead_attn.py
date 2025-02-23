from rotary_embedding_torch import RotaryEmbedding
from torch import nn
from typing import Optional

from . import transformer_utils

import torch
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, device, model_config, attn_config, self_attn, in_decoder=False):
        super(MultiHeadAttention, self).__init__()

        self.debug = False

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

        self.n_heads = attn_config.n_heads
        self.d_queries = attn_config.d_queries
        self.d_values = attn_config.d_values
        self.q_bias = attn_config.q_bias
        self.k_bias = attn_config.k_bias
        self.v_bias = attn_config.v_bias
        self.o_bias = attn_config.o_bias
        self.heads_activation_function = attn_config.heads_activation_function
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

        # A linear projection to cast (n_kv_heads sets of) queries from the input query sequences
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.d_queries, bias=self.q_bias) # (N, query_sequence_pad_length, n_kv_heads * d_queries)
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

        self.register_buffer('causal_mask', ~torch.tril(torch.ones(self.maxlen + 1, self.maxlen + 1).to(self.device)).to(torch.bool))

    def mask_attention(self, attention_weights: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # mask away tokens by setting such weights to a large negative number, so that they evaluate to 0 under the softmax
        # masks should be either bool tensors with True values where values should be masked, or float/integer tensors with 1 values where values should be masked
        if attention_mask is not None:
            assert attention_mask.shape[0] == attention_weights.shape[0], f"batch dimension for padding is wrong: {attention_mask.shape[0]} != {attention_weights.shape[0]}. overall shape: {attention_mask.shape} != {attention_weights.shape}"
            assert attention_mask.shape[1] == attention_weights.shape[3], f"padding mask length is wrong: {attention_mask.shape[1]} != {attention_weights.shape[3]}. overall shape: {attention_mask.shape} != {attention_weights.shape}"

            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).to(torch.bool)

            attention_weights = attention_weights.masked_fill_(attention_mask, -float('inf'))
        if self.self_attn:
            attention_weights = attention_weights.masked_fill_(self.causal_mask[:attention_weights.shape[-2], :attention_weights.shape[-1]], -float('inf'))

        return attention_weights

    def forward(self,
                query_sequences: torch.Tensor,
                key_sequences: torch.Tensor,
                value_sequences: torch.Tensor,
                attention_mask: Optional[torch.Tensor]=None,
                k_cache: Optional[list[torch.Tensor]]=None,
                v_cache: Optional[list[torch.Tensor]]=None,
                return_attn_values: Optional[bool]=False) -> tuple[torch.Tensor, list[torch.Tensor]]:
        query_sequences = self.qkv_norm(query_sequences)

        # if this isn't self-attention, they will already have been normed in the last layer of the Encoder (from whence they came)
        if self.self_attn:
            key_sequences = self.qkv_norm(key_sequences)
            value_sequences = self.qkv_norm(value_sequences)

        if k_cache is not None and v_cache is not None:
            k_new = self.k_proj(key_sequences[:, -1:])
            v_new = self.v_proj(value_sequences[:, -1:])

            k_cache.append(k_new)
            v_cache.append(v_new)

            k_heads = torch.cat(k_cache, dim=1)
            v_heads = torch.cat(v_cache, dim=1)

            q_heads = self.q_proj(query_sequences[:, -1:])
        else:
            q_heads: torch.Tensor = self.q_proj(query_sequences)
            k_heads: torch.Tensor = self.k_proj(key_sequences)
            v_heads: torch.Tensor = self.v_proj(value_sequences)

        if self.heads_activation is not None:
            q_heads = self.heads_activation(q_heads)
            k_heads = self.heads_activation(k_heads)
            v_heads = self.heads_activation(v_heads)

        N, t, _ = q_heads.shape
        N, T, _ = k_heads.shape

        # Split the last dimension by the n_kv_heads subspaces
        q_heads = q_heads.contiguous().view(N, t, self.n_heads, self.d_queries) # (N, query_sequence_pad_length, n_heads, d_queries)
        k_heads = k_heads.contiguous().view(N, T, self.n_heads, self.d_queries) # (N, key_value_sequence_pad_length, n_heads, d_queries)
        v_heads = v_heads.contiguous().view(N, T, self.n_heads, self.d_values) # (N, key_value_sequence_pad_length, n_heads, d_values)

        q_heads = q_heads.permute(0, 2, 1, 3) # Nhtd
        k_heads = k_heads.permute(0, 2, 1, 3) # NhTd
        v_heads = v_heads.permute(0, 2, 1, 3) # NhTv

        if self.rotary_embedding is not None:
            q_heads = self.rotary_embedding.rotate_queries_or_keys(q_heads, seq_dim=-2)
            k_heads = self.rotary_embedding.rotate_queries_or_keys(k_heads.unsqueeze(0), seq_dim=-2).squeeze(0) # adds a singleton dimension for the rotation operation and then removes it for the torch compiler

        # if return_attn_values:
        attention_weights = torch.einsum('...htq,...hTq->...htT', q_heads, k_heads)

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
        # else:
        #     attention_weights_for_visualization = []
        #     sequences = F.scaled_dot_product_attention(
        #         q_heads,
        #         k_heads,
        #         v_heads,
        #         attn_mask=attention_mask,
        #         dropout_p=self.dropout,
        #         is_causal=self.self_attn
        #     )

        sequences = sequences.permute(0, 2, 1, 3)

        # Concatenate the n_heads subspaces (each with an output of size d_values)
        sequences = sequences.contiguous().view(N, t, -1)

        sequences = self.o_proj(sequences)
        sequences = self.output_dropout(sequences)

        return sequences, attention_weights_for_visualization
