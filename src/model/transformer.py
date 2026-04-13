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
from utils.megatransformer_utils import linear_weight_init


def create_alibi_bias(n_heads, maxlen):
    slopes = torch.pow(2, -torch.arange(1, n_heads + 1) * 8 / n_heads)
    # Create position differences matrix
    pos = torch.arange(maxlen)
    diff = pos.unsqueeze(-1) - pos.unsqueeze(-2)  # [seq_len, seq_len]
    # Calculate bias for each head
    bias = -torch.abs(diff).unsqueeze(0) * slopes.unsqueeze(-1).unsqueeze(-1)
    return bias  # [n_heads, seq_len, seq_len]


class MegaTransformerAttention(nn.Module):
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
        
        self.is_causal = config.causal
        max_positions = config.max_position_embeddings

        if self.is_causal:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(max_positions, max_positions)).view(1, 1, max_positions, max_positions),
                persistent=False,
            )
        else:
            self.register_buffer("causal_mask", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        kv_cache: Optional[KVCache] = None,
        position_offset: int = 0,
        use_cache: bool = False,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Forward pass with optional KV caching for efficient generation.

        Args:
            hidden_states: Input tensor, shape (batch, seq_len, d_model).
                Queries always come from here.
            attention_mask: Optional attention mask for self-attention keys.
            head_mask: Optional head mask
            kv_cache: Optional KVCache with cached keys/values from previous positions
            position_offset: Position offset for RoPE (number of cached tokens)
            use_cache: Whether to return updated KV cache
            encoder_hidden_states: If provided, keys and values come from this tensor
                instead of hidden_states (cross-attention mode).
            encoder_attention_mask: Attention mask for encoder states in cross-attention.

        Returns:
            output: Attention output, shape (batch, seq_len, d_model)
            new_kv_cache: Updated KVCache if use_cache=True, else None
        """
        is_cross_attention = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross_attention else hidden_states

        N, seq_len = hidden_states.shape[:2]

        queries: torch.Tensor = self.q_proj(hidden_states)
        keys: torch.Tensor = self.k_proj(kv_input)
        values: torch.Tensor = self.v_proj(kv_input)

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

        # Causal masking (skip for bidirectional attention and cross-attention)
        if self.is_causal and not is_cross_attention:
            # For cached generation with Huginn-style cache, position_offset tracks the
            # global position while T (key length) may differ across cache slots.
            # Use T-based offset so the mask always matches the actual key dimension.
            eff_offset = max(T - t, 0)  # query positions relative to key positions
            required_size = max(eff_offset + t, T)
            if required_size > min(self.causal_mask.shape[-1], self.causal_mask.shape[-2]):
                self.causal_mask = torch.tril(
                    torch.ones(required_size, required_size, device=hidden_states.device)
                ).view(1, 1, required_size, required_size)

            # Slice causal mask for current query/key positions
            causal_mask_slice = self.causal_mask[:, :, eff_offset:eff_offset + t, :T]
            causal_mask_slice = causal_mask_slice.to(attention_scores.device)
            attention_scores = attention_scores.masked_fill(causal_mask_slice == 0, float("-inf"))

        # Apply attention mask (use encoder_attention_mask for cross-attention)
        active_mask = encoder_attention_mask if is_cross_attention else attention_mask
        if active_mask is not None:
            active_mask = active_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(active_mask == 0, float("-inf"))

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


class MegaTransformerAxial2DAttention(nn.Module):
    """Self-attention with 2D axial rotary positional encoding for image patches.

    Uses separate RoPE frequencies for row and column positions, giving the
    attention pattern awareness of 2D spatial relationships. Non-causal by
    default (image patches attend to all other patches).

    Args:
        config: MegaTransformerBlockConfig (uses d_model, n_heads, d_queries, d_values,
                n_query_groups, rotary_embedding_dim).
        grid_size: (H, W) tuple — number of patches per axis.
        causal: If False, no causal mask is applied (default for images).
    """

    def __init__(self, config: MegaTransformerBlockConfig, grid_size: Tuple[int, int]):
        super().__init__()

        self.config = config
        self.n_heads = config.n_heads
        self.d_queries = config.d_queries
        self.d_values = config.d_values
        self.n_query_groups = config.n_query_groups
        self.causal = config.causal

        assert self.n_heads % self.n_query_groups == 0
        self.n_rep = self.n_heads // self.n_query_groups

        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.d_queries, bias=config.use_qkv_bias)
        self.k_proj = nn.Linear(config.d_model, self.n_query_groups * self.d_queries, bias=config.use_qkv_bias)
        self.v_proj = nn.Linear(config.d_model, self.n_query_groups * self.d_values, bias=config.use_qkv_bias)
        self.o_proj = nn.Linear(self.n_heads * self.d_values, config.d_model)

        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.proj_dropout = nn.Dropout(config.hidden_dropout_prob)

        # 2D axial RoPE: half the dims for row, half for column
        rope_dim = config.rotary_embedding_dim
        self.rotary_embedding = RotaryEmbedding(dim=rope_dim // 2, freqs_for='pixel')

        # Precompute and register axial frequencies
        H, W = grid_size
        axial_freqs = self.rotary_embedding.get_axial_freqs(H, W)  # (H, W, rope_dim)
        self.register_buffer('axial_freqs', axial_freqs.reshape(H * W, rope_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        kv_cache: Optional[KVCache] = None,
        position_offset: int = 0,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        from rotary_embedding_torch import apply_rotary_emb

        N, seq_len = hidden_states.shape[:2]

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(N, -1, self.n_heads, self.d_queries).permute(0, 2, 1, 3).contiguous()
        keys = keys.view(N, -1, self.n_query_groups, self.d_queries).permute(0, 2, 1, 3).contiguous()
        values = values.view(N, -1, self.n_query_groups, self.d_values).permute(0, 2, 1, 3).contiguous()

        # Apply 2D axial RoPE using precomputed frequencies
        freqs = self.axial_freqs[:seq_len]  # (seq_len, rope_dim)
        queries = apply_rotary_emb(freqs, queries)
        keys = apply_rotary_emb(freqs, keys)

        # GQA expansion
        if self.n_rep > 1:
            keys = keys.repeat_interleave(self.n_rep, dim=1)
            values = values.repeat_interleave(self.n_rep, dim=1)

        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.d_queries)

        # Optional causal masking (off by default for images)
        if self.causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device))
            attention_scores = attention_scores.masked_fill(causal_mask.view(1, 1, seq_len, seq_len) == 0, float("-inf"))

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float("-inf"))

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, values)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(N, seq_len, self.d_values * self.n_heads)

        output = self.o_proj(context_layer)
        output = self.proj_dropout(output)

        return output, None


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


class MegaTransformerEncoderBlock(nn.Module):
    """Causal Self Attention followed by FFN with optional pre/post layer norms."""
    def __init__(self, config: MegaTransformerBlockConfig):
        super().__init__()
        self.self_attn = MegaTransformerAttention(config)

        self.ffn: nn.Module
        if config.ffn_type == "mlp":
            self.ffn = SimpleFFN(config)
        else:
            raise ValueError(f"Unknown ffn_type: {config.ffn_type}")

        self.pre_attn_norm = create_norm(config.d_model, config.norm_type, config.norm_eps) if config.pre_attn_norm else None
        self.post_attn_norm = create_norm(config.d_model, config.norm_type, config.norm_eps) if config.post_attn_norm else None
        self.pre_ffn_norm = create_norm(config.d_model, config.norm_type, config.norm_eps) if config.pre_ffn_norm else None
        self.post_ffn_norm = create_norm(config.d_model, config.norm_type, config.norm_eps) if config.post_ffn_norm else None

        self._init_weights()

    def _init_weights(self):
        # Per-block default: standard Xavier (gain=1.0). Parent stacks should
        # call `apply_depth_scaled_residual_init` after constructing all blocks
        # to depth-scale the residual-output layers (o_proj, condense).
        self.apply(linear_weight_init(gain=1.0))

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


class MegaTransformerAxial2DEncoderBlock(nn.Module):
    """Transformer block with 2D axial RoPE for image patch sequences.

    Drop-in replacement for MegaTransformerBlock in image preludes/codas.
    Uses non-causal bidirectional attention with 2D positional encoding
    so patches attend based on spatial proximity, not sequence order.

    Args:
        config: MegaTransformerBlockConfig
        grid_size: (H, W) patch grid dimensions
        causal: Whether to apply causal mask (default False for images)
    """

    def __init__(self, config: MegaTransformerBlockConfig, grid_size: Tuple[int, int]):
        super().__init__()
        self.self_attn = MegaTransformerAxial2DAttention(config, grid_size=grid_size)

        if config.ffn_type == "mlp":
            self.ffn = SimpleFFN(config)
        else:
            raise ValueError(f"Unknown ffn_type: {config.ffn_type}")

        self.pre_attn_norm = create_norm(config.d_model, config.norm_type, config.norm_eps) if config.pre_attn_norm else None
        self.post_attn_norm = create_norm(config.d_model, config.norm_type, config.norm_eps) if config.post_attn_norm else None
        self.pre_ffn_norm = create_norm(config.d_model, config.norm_type, config.norm_eps) if config.pre_ffn_norm else None
        self.post_ffn_norm = create_norm(config.d_model, config.norm_type, config.norm_eps) if config.post_ffn_norm else None

        self._init_weights()

    def _init_weights(self):
        # Per-block default: standard Xavier (gain=1.0). Parent stacks should
        # call `apply_depth_scaled_residual_init` after constructing all blocks
        # to depth-scale the residual-output layers (o_proj, condense).
        self.apply(linear_weight_init(gain=1.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        position_offset: int = 0,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        if self.pre_attn_norm is not None:
            attn_input = self.pre_attn_norm(hidden_states)
        else:
            attn_input = hidden_states

        attn_output, _ = self.self_attn(attn_input, attention_mask=attention_mask, head_mask=head_mask)
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

        return hidden_states, None


class MegaTransformerDecoderBlock(nn.Module):
    """Self-attention + cross-attention + FFN decoder block.

    When encoder_hidden_states is provided, cross-attention attends to it.
    When not provided, cross-attention falls back to self-attention on hidden_states
    (making this block behave like an encoder block with an extra attention layer).
    """
    def __init__(self, config: MegaTransformerBlockConfig):
        super().__init__()
        self.self_attn = MegaTransformerAttention(config)

        # Cross-attention uses the same config but is always non-causal
        import copy
        cross_config = copy.deepcopy(config)
        cross_config.causal = False
        self.cross_attn = MegaTransformerAttention(cross_config)

        self.ffn: nn.Module
        if config.ffn_type == "mlp":
            self.ffn = SimpleFFN(config)
        else:
            raise ValueError(f"Unknown ffn_type: {config.ffn_type}")

        self.pre_attn_norm = create_norm(config.d_model, config.norm_type, config.norm_eps) if config.pre_attn_norm else None
        self.inter_attn_norm = create_norm(config.d_model, config.norm_type, config.norm_eps) if config.inter_attn_norm else None
        self.post_attn_norm = create_norm(config.d_model, config.norm_type, config.norm_eps) if config.post_attn_norm else None
        self.pre_ffn_norm = create_norm(config.d_model, config.norm_type, config.norm_eps) if config.pre_ffn_norm else None
        self.post_ffn_norm = create_norm(config.d_model, config.norm_type, config.norm_eps) if config.post_ffn_norm else None

        self._init_weights()

    def _init_weights(self):
        # Per-block default: standard Xavier (gain=1.0). Parent stacks should
        # call `apply_depth_scaled_residual_init` after constructing all blocks
        # to depth-scale the residual-output layers (o_proj, condense).
        self.apply(linear_weight_init(gain=1.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        position_offset: int = 0,
        use_cache: bool = False,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Forward pass with self-attention, cross-attention, and FFN.

        Args:
            hidden_states: Decoder input, shape (batch, seq_len, d_model)
            attention_mask: Mask for self-attention keys
            head_mask: Optional head mask
            kv_cache: Optional KVCache for self-attention
            position_offset: Position offset for RoPE
            use_cache: Whether to return updated KV cache
            encoder_hidden_states: Encoder output for cross-attention.
                If None, cross-attention uses hidden_states (self-attention fallback).
            encoder_attention_mask: Mask for encoder states in cross-attention.

        Returns:
            hidden_states: Output tensor, shape (batch, seq_len, d_model)
            new_kv_cache: Updated KVCache if use_cache=True, else None
        """
        # Self-attention
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

        # Cross-attention
        if self.inter_attn_norm is not None:
            cross_input = self.inter_attn_norm(hidden_states)
        else:
            cross_input = hidden_states

        cross_output, _ = self.cross_attn(
            cross_input,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        hidden_states = hidden_states + cross_output

        if self.post_attn_norm is not None:
            hidden_states = self.post_attn_norm(hidden_states)

        # FFN
        if self.pre_ffn_norm is not None:
            ffn_input = self.pre_ffn_norm(hidden_states)
        else:
            ffn_input = hidden_states

        ffn_output = self.ffn(ffn_input)

        hidden_states = hidden_states + ffn_output

        if self.post_ffn_norm is not None:
            hidden_states = self.post_ffn_norm(hidden_states)

        return hidden_states, new_kv_cache