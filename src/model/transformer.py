import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rotary_embedding_torch import RotaryEmbedding

from config.common import MegaTransformerBlockConfig
from model import activations
from model.activations import get_activation_type
from model.norms import create_norm
from model.world.kv_cache import KVCache
from utils.megatransformer_utils import transformer_weight_init


def create_alibi_bias(n_heads, maxlen):
    slopes = torch.pow(2, -torch.arange(1, n_heads + 1) * 8 / n_heads)
    # Create position differences matrix
    pos = torch.arange(maxlen)
    diff = pos.unsqueeze(-1) - pos.unsqueeze(-2)  # [seq_len, seq_len]
    # Calculate bias for each head
    bias = -torch.abs(diff).unsqueeze(0) * slopes.unsqueeze(-1).unsqueeze(-1)
    return bias  # [n_heads, seq_len, seq_len]


class MegaTransformerCausalSelfAttention(nn.Module):
    def __init__(self, config: MegaTransformerBlockConfig):
        super().__init__()

        self.config = config

        self.n_heads = config.n_heads
        self.hidden_size = config.d_model
        self.use_grok_scaled_attn = config.use_grok_scaled_attn

        self.d_queries = config.d_queries
        self.d_values = config.d_values
        self.n_query_groups = config.n_query_groups
        self.n_heads = config.n_heads

        # GQA: queries have full n_heads, keys/values have fewer n_query_groups (shared across query heads)
        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.d_queries, bias=config.use_qkv_bias)
        self.k_proj = nn.Linear(config.d_model, self.n_query_groups * self.d_queries, bias=config.use_qkv_bias)
        self.v_proj = nn.Linear(config.d_model, self.n_query_groups * self.d_values, bias=config.use_qkv_bias)

        # Number of times to repeat K/V heads to match Q heads
        assert self.n_heads % self.n_query_groups == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_query_groups ({self.n_query_groups})"
        self.n_rep = self.n_heads // self.n_query_groups

        self.heads_activation = None
        if config.heads_activation is not None:
            activation_type = get_activation_type(config.heads_activation)
            if activation_type == activations.SwiGLU:
                self.heads_activation = activations.SwiGLU(config.d_values)
            else:
                self.heads_activation = activation_type()

        self.rotary_embedding = None
        if bool(config.use_rotary_embedding):
            self.rotary_embedding = RotaryEmbedding(dim=config.rotary_embedding_dim, learned_freq=config.rotary_embedding_learnable)

        if bool(config.use_alibi_bias):
            self.register_buffer('alibi_bias', create_alibi_bias(n_heads=config.n_heads, maxlen=config.max_position_embeddings))
        else:
            self.register_buffer('alibi_bias', None)
        
        self.o_proj = nn.Linear(self.n_heads * self.d_values, config.d_model)
        
        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.proj_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        max_positions = config.max_position_embeddings

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_positions, max_positions)).view(1, 1, max_positions, max_positions),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        kv_cache: Optional[KVCache] = None,
        position_offset: int = 0,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Forward pass with optional KV caching for efficient generation.

        Args:
            hidden_states: Input tensor, shape (batch, seq_len, d_model)
            attention_mask: Optional attention mask
            head_mask: Optional head mask
            kv_cache: Optional KVCache with cached keys/values from previous positions
            position_offset: Position offset for RoPE (number of cached tokens)
            use_cache: Whether to return updated KV cache

        Returns:
            output: Attention output, shape (batch, seq_len, d_model)
            new_kv_cache: Updated KVCache if use_cache=True, else None
        """
        N, seq_len = hidden_states.shape[:2]

        queries: torch.Tensor = self.q_proj(hidden_states)
        keys: torch.Tensor = self.k_proj(hidden_states)
        values: torch.Tensor = self.v_proj(hidden_states)

        if self.heads_activation is not None:
            queries = self.heads_activation(queries)
            keys = self.heads_activation(keys)
            values = self.heads_activation(values)

        # Reshape to head format before applying RoPE
        # Q: (N, seq, n_heads * d_queries) -> (N, n_heads, seq, d_queries)
        # K: (N, seq, n_query_groups * d_queries) -> (N, n_query_groups, seq, d_queries)
        # V: (N, seq, n_query_groups * d_values) -> (N, n_query_groups, seq, d_values)
        queries = queries.view(N, -1, self.n_heads, self.d_queries).permute(0, 2, 1, 3).contiguous()
        keys = keys.view(N, -1, self.n_query_groups, self.d_queries).permute(0, 2, 1, 3).contiguous()
        values = values.view(N, -1, self.n_query_groups, self.d_values).permute(0, 2, 1, 3).contiguous()

        # Apply RoPE with position offset for correct positions during generation
        if self.rotary_embedding is not None:
            # rotate_queries_or_keys supports offset parameter for cached generation
            queries = self.rotary_embedding.rotate_queries_or_keys(queries, offset=position_offset)
            keys = self.rotary_embedding.rotate_queries_or_keys(keys, offset=position_offset)

        # Handle KV cache
        new_kv_cache = None
        if kv_cache is not None and kv_cache.key_cache is not None:
            # Concatenate cached keys/values with new ones
            # Cached K/V are already expanded to n_heads if GQA was used
            keys_for_cache = keys  # Before expansion
            values_for_cache = values

            # Expand new K/V heads to match Q heads for GQA before concatenation
            if self.n_rep > 1:
                keys = keys.repeat_interleave(self.n_rep, dim=1)
                values = values.repeat_interleave(self.n_rep, dim=1)

            # Concatenate with cache (cache stores expanded K/V)
            keys = torch.cat([kv_cache.key_cache, keys], dim=2)
            values = torch.cat([kv_cache.value_cache, values], dim=2)

            if use_cache:
                # Store expanded K/V in cache for next step
                new_kv_cache = KVCache()
                new_kv_cache.key_cache = keys.clone()
                new_kv_cache.value_cache = values.clone()
        else:
            # No cache - expand K/V for GQA
            if self.n_rep > 1:
                keys = keys.repeat_interleave(self.n_rep, dim=1)
                values = values.repeat_interleave(self.n_rep, dim=1)

            if use_cache:
                # Initialize cache with current K/V (already expanded)
                new_kv_cache = KVCache()
                new_kv_cache.key_cache = keys.clone()
                new_kv_cache.value_cache = values.clone()

        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))

        _, _, t = queries.shape[:3]  # Query sequence length (new tokens)
        _, _, T = keys.shape[:3]  # Total key sequence length (cached + new)

        if self.alibi_bias is not None:
            # For ALiBi, we need the full position range
            # Query positions: [position_offset, position_offset + t)
            # Key positions: [0, T)
            alibi_slice = self.alibi_bias[:, position_offset:position_offset + t, :T]
            attention_scores = attention_scores + alibi_slice.unsqueeze(0).repeat(N, 1, 1, 1)

        attention_scores = attention_scores / math.sqrt(self.d_queries)
        if bool(self.use_grok_scaled_attn):
            attention_scores = 30.0 * torch.tanh(attention_scores / 30.0)

        # Causal masking: queries at position i can attend to keys at positions <= i
        # For cached generation: query position = position_offset + local_pos
        # Key position = 0 to T-1
        # Query at pos p can attend to keys at pos <= p
        max_seq_len = position_offset + t
        if max_seq_len > min(self.causal_mask.shape[-1], self.causal_mask.shape[-2]):
            # Recalculate causal mask with new longest length
            new_size = max(max_seq_len, T)
            self.causal_mask = torch.tril(
                torch.ones(new_size, new_size, device=hidden_states.device)
            ).view(1, 1, new_size, new_size)

        # Slice causal mask for current query/key positions
        causal_mask_slice = self.causal_mask[:, :, position_offset:position_offset + t, :T]
        causal_mask_slice = causal_mask_slice.to(attention_scores.device)
        attention_scores = attention_scores.masked_fill(causal_mask_slice == 0, float("-inf"))

        if attention_mask is not None:
            # HuggingFace uses 1 for tokens to attend to and 0 for masked tokens
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float("-inf"))

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, values)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.d_values * self.n_heads,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Apply output projection
        output = self.o_proj(context_layer)
        output = self.proj_dropout(output)

        return output, new_kv_cache


class SimpleFFN(nn.Module):
    def __init__(self, config: MegaTransformerBlockConfig):
        super().__init__()
        self.expand = nn.Linear(config.d_model, config.d_inner)
        self.condense = nn.Linear(config.d_inner, config.d_model)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        activation_type = get_activation_type(config.activation_function)
        if activation_type == activations.SwiGLU:
            self.activation = activations.SwiGLU(config.d_inner)
        else:
            self.activation = activation_type()
    
    def forward(self, hidden_states):
        hidden_states = self.expand(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.condense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class MegaTransformerBlock(nn.Module):
    """Causal Self Attention followed by FFN with optional pre/post layer norms."""
    def __init__(self, config: MegaTransformerBlockConfig):
        super().__init__()
        self.self_attn = MegaTransformerCausalSelfAttention(config)

        self.ffn: nn.Module
        if config.ffn_type == "mlp":
            self.ffn = SimpleFFN(config)
        else:
            raise ValueError(f"Unknown ffn_type: {config.ffn_type}")

        if config.pre_attn_norm:
            self.pre_attn_norm = create_norm(config.d_model, config.norm_type, config.norm_eps)
        else:
            self.pre_attn_norm = None
        if config.post_attn_norm:
            self.post_attn_norm = create_norm(config.d_model, config.norm_type, config.norm_eps)
        else:
            self.post_attn_norm = None
        if config.pre_ffn_norm:
            self.pre_ffn_norm = create_norm(config.d_model, config.norm_type, config.norm_eps)
        else:
            self.pre_ffn_norm = None
        if config.post_ffn_norm:
            self.post_ffn_norm = create_norm(config.d_model, config.norm_type, config.norm_eps)
        else:
            self.post_ffn_norm = None

        self._init_weights()

    def _init_weights(self):
        self.apply(transformer_weight_init())

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        position_offset: int = 0,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Forward pass with optional KV caching.

        Args:
            hidden_states: Input tensor, shape (batch, seq_len, d_model)
            attention_mask: Optional attention mask
            head_mask: Optional head mask
            kv_cache: Optional KVCache for efficient generation
            position_offset: Position offset for RoPE
            use_cache: Whether to return updated KV cache

        Returns:
            hidden_states: Output tensor, shape (batch, seq_len, d_model)
            new_kv_cache: Updated KVCache if use_cache=True, else None
        """
        if self.pre_attn_norm is not None:
            attn_input = self.pre_attn_norm(hidden_states)
        else:
            attn_input = hidden_states

        attn_output, new_kv_cache = self.self_attn(
            attn_input,
            attention_mask=attention_mask,
            head_mask=head_mask,
            kv_cache=kv_cache,
            position_offset=position_offset,
            use_cache=use_cache,
        )

        hidden_states = hidden_states + attn_output

        if self.post_attn_norm is not None:
            hidden_states = self.post_attn_norm(hidden_states)

        if self.pre_ffn_norm is not None:
            ffn_input = self.pre_ffn_norm(hidden_states)
        else:
            ffn_input = hidden_states

        ffn_output = self.ffn(ffn_input)

        hidden_states = hidden_states + ffn_output

        if self.post_ffn_norm is not None:
            hidden_states = self.post_ffn_norm(hidden_states)

        return hidden_states, new_kv_cache
