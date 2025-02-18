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

        self.n_q_heads = self.n_gqa_groups * self.n_heads

        # A linear projection to cast (n_kv_heads sets of) queries from the input query sequences
        self.q_proj = nn.Linear(self.d_model, self.n_q_heads * self.d_queries, bias=self.q_bias) # (N, query_sequence_pad_length, n_kv_heads * d_queries)
        # A linear projection to cast (n_kv_heads sets of) keys and values from the input reference sequences
        self.k_proj = nn.Linear(self.d_model, self.n_heads * self.d_queries, bias=self.k_bias) # (N, key_value_sequence_pad_length, n_kv_heads * d_queries)
        self.v_proj = nn.Linear(self.d_model, self.n_heads * self.d_values, bias=self.v_bias) # (N, key_value_sequence_pad_length, n_kv_heads * d_values)

        # a linear projection to cast (n_q_heads sets of) computed attention-weighted vectors to output vectors
        self.o_proj = nn.Linear(self.n_heads * self.d_values, self.d_model, bias=self.o_bias)

        self.qkv_norm = self.norm(self.d_model, self.norm_eps)

        self.attn_dropout = nn.Dropout(self.dropout)

        self.heads_activation = None
        if self.heads_activation_function is not None:
            self.heads_activation = transformer_utils.create_activation_function(self.d_model, self.heads_activation_function)

        self.infinite_attn_elu: Optional[nn.ELU] = None
        self.infinite_attn_beta: Optional[nn.Parameter] = None

        assert self.maxlen % self.infinite_attention_n_segments == 0, "maxlen must be divisible by infinite_attention_n_segments"

        self.infinite_attn_beta = nn.Parameter(torch.ones((1,)))
        self.infinite_attn_elu = nn.ELU()
        self.register_buffer('causal_mask', torch.tril(torch.ones((self.maxlen // self.infinite_attention_n_segments) + 1, (self.maxlen // self.infinite_attention_n_segments) + 1).to(device)))

    def mask_attention(self, attention_weights: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # mask away tokens by setting such weights to a large negative number, so that they evaluate to 0 under the softmax
        if attention_mask is not None:
            assert attention_mask.shape[0] == attention_weights.shape[0], f"batch dimension for padding is wrong: {attention_mask.shape[0]} != {attention_weights.shape[0]}. overall shape: {attention_mask.shape} != {attention_weights.shape}"
            assert attention_mask.shape[1] == attention_weights.shape[3], f"padding mask length is wrong: {attention_mask.shape[1]} != {attention_weights.shape[3]}. overall shape: {attention_mask.shape} != {attention_weights.shape}"

            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).to(torch.bool)

            attention_weights = attention_weights.masked_fill_(~attention_mask, -float('inf'))

        if self.self_attn:
            attention_weights = attention_weights.masked_fill_(self.causal_mask[:attention_weights.shape[-2], :attention_weights.shape[-1]] == 0, -float('inf'))

        return attention_weights

    def forward(self, query_sequences: torch.Tensor, key_sequences: torch.Tensor, value_sequences: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        query_sequences = self.qkv_norm(query_sequences)

        # if this isn't self-attention, they will already have been normed in the last layer of the Encoder (from whence they came)
        if self.self_attn:
            key_sequences = self.qkv_norm(key_sequences)
            value_sequences = self.qkv_norm(value_sequences)

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
        q_heads = q_heads.contiguous().view(N, t, self.n_gqa_groups, self.n_heads, self.d_queries) # (N, query_sequence_pad_length, n_gqa_groups, n_heads, d_queries)
        k_heads = k_heads.contiguous().view(N, T, self.n_heads, self.d_queries) # (N, key_value_sequence_pad_length, n_heads, d_queries)
        v_heads = v_heads.contiguous().view(N, T, self.n_heads, self.d_values) # (N, key_value_sequence_pad_length, n_heads, d_values)

        q_heads = q_heads.permute(0, 2, 3, 1, 4) # Nghtd
        k_heads = k_heads.permute(0, 2, 1, 3) # NhTd
        v_heads = v_heads.permute(0, 2, 1, 3) # NhTv

        if hasattr(self, 'rotary_embedding') and self.rotary_embedding is not None:
            q_heads = self.rotary_embedding.rotate_queries_or_keys(q_heads, seq_dim=-2)
            k_heads = self.rotary_embedding.rotate_queries_or_keys(k_heads.unsqueeze(0), seq_dim=-2).squeeze(0) # adds a singleton dimension for the rotation operation and then removes it for the torch compiler

        memory = torch.zeros((self.n_heads, self.d_queries, self.d_queries)).to(query_sequences.device)
        z = torch.zeros((self.n_heads, self.d_queries, 1)).to(query_sequences.device)

        q_heads = q_heads.view(N, self.n_gqa_groups, self.n_heads, self.infinite_attention_n_segments, t // self.infinite_attention_n_segments, self.d_queries) # Nghitq
        k_heads = k_heads.view(N, self.n_heads, self.infinite_attention_n_segments, T // self.infinite_attention_n_segments, self.d_queries) # NhiTq
        v_heads = v_heads.view(N, self.n_heads, self.infinite_attention_n_segments, T // self.infinite_attention_n_segments, self.d_values) # NhiTv

        output = []
        for idx in range(self.infinite_attention_n_segments):
            sigma_q: torch.Tensor = self.infinite_attn_elu(q_heads[:, :, :, idx, :, :]) + 1.0
            sigma_k: torch.Tensor = self.infinite_attn_elu(k_heads[:, :, idx, :, :]) + 1.0

            A_mem = (sigma_q @ memory) / ((sigma_q @ z) + (1e-6))

            attention_weights: torch.Tensor = torch.einsum('...ghtq,...hTq->...htT', q_heads[:, :, :, idx, :, :], k_heads[:, :, idx, :, :])

            # scaled attention
            attention_weights = (1.0 / (self.d_queries ** 0.5)) * attention_weights
            if bool(self.use_grok_scaled_attn):
                attention_weights = 30.0 * torch.tanh(attention_weights / 30.0)

            attention_weights = self.mask_attention(attention_weights, None)
            attention_weights = F.softmax(attention_weights, dim=-1)

            attention_weights_for_visualization = [attention_weights.clone().detach().contiguous().view(N, self.n_gqa_groups, self.n_heads, t // self.infinite_attention_n_segments, T // self.infinite_attention_n_segments)]

            # not included in paper for some reason? experiment
            # attention_weights = self.dropout(attention_weights)
            attention_weights = attention_weights @ v_heads[:, :, idx, :, :]

            attention_weights = (F.sigmoid(self.infinite_attn_beta) * A_mem) + ((1 - F.sigmoid(self.infinite_attn_beta)) * attention_weights)

            if self.infinite_attention_update == 'linear':
                memory = memory + (sigma_k.transpose(-2, -1) @ v_heads[:, :, idx, :, :])
            else:
                delta = (sigma_k @ memory) / ((sigma_k @ z) + 1e-6)
                memory = memory + (sigma_k.transpose(-2, -1) @ (v_heads[:, :, idx, :, :] - delta))

            z = z + sigma_k.sum(dim=-2, keepdim=True)

            output.append(attention_weights)

        sequences = torch.concat(output, dim = 2) # NhiTv

        # Concatenate the n_heads subspaces (each with an output of size d_values)
        sequences = sequences.contiguous().view(N, t, -1)

        sequences = self.attn_dropout(sequences)

        sequences = self.o_proj(sequences)

        return sequences, attention_weights_for_visualization
