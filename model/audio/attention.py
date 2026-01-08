import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import megatransformer_utils


# =============================================================================
# 2D Rotary Position Embedding (RoPE) for frequency-time attention
# =============================================================================

def _compute_2d_rope_freqs(
    dim: int,
    max_freq: int,
    max_time: int,
    base: float = 10000.0,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute rotary embedding frequencies for 2D positions (frequency, time).

    Splits the embedding dimension in half:
    - First half encodes frequency position (mel bin)
    - Second half encodes time position

    Args:
        dim: Head dimension (must be divisible by 4 for 2D split)
        max_freq: Maximum frequency positions (M, mel bins)
        max_time: Maximum time positions (T, timesteps)
        base: Base for frequency computation
        device: Device for tensors

    Returns:
        freq_freqs: [max_freq, dim//2] rotation frequencies for freq dimension
        time_freqs: [max_time, dim//2] rotation frequencies for time dimension
    """
    half_dim = dim // 2
    quarter_dim = dim // 4

    # Compute inverse frequencies for each dimension
    inv_freq = 1.0 / (base ** (torch.arange(0, quarter_dim, 2, device=device).float() / quarter_dim))

    # Position indices
    freq_pos = torch.arange(max_freq, device=device).float()
    time_pos = torch.arange(max_time, device=device).float()

    # Outer product: [positions, freqs]
    freq_freqs = torch.einsum('p,f->pf', freq_pos, inv_freq)  # [max_freq, quarter_dim//2]
    time_freqs = torch.einsum('p,f->pf', time_pos, inv_freq)  # [max_time, quarter_dim//2]

    # Duplicate for sin/cos pairs: [pos, quarter_dim//2] -> [pos, quarter_dim]
    freq_freqs = torch.cat([freq_freqs, freq_freqs], dim=-1)  # [max_freq, quarter_dim]
    time_freqs = torch.cat([time_freqs, time_freqs], dim=-1)  # [max_time, quarter_dim]

    return freq_freqs, time_freqs


def _apply_2d_rope(
    x: torch.Tensor,
    freq_freqs: torch.Tensor,
    time_freqs: torch.Tensor,
    freq_positions: torch.Tensor,
    time_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Apply 2D rotary position embeddings.

    Args:
        x: [B, n_heads, seq_len, head_dim] where seq_len = M * T
        freq_freqs: [max_freq, head_dim//2] precomputed frequency embeddings
        time_freqs: [max_time, head_dim//2] precomputed time embeddings
        freq_positions: [seq_len] frequency position for each token (0 to M-1)
        time_positions: [seq_len] time position for each token (0 to T-1)

    Returns:
        x_rotated: [B, n_heads, seq_len, head_dim] with 2D positional info
    """
    head_dim = x.shape[-1]
    quarter_dim = head_dim // 4

    # Get the rotation angles for each position in the sequence
    # freq_positions: [seq_len] indices into freq_freqs
    # time_positions: [seq_len] indices into time_freqs
    freq_angles = freq_freqs[freq_positions]  # [seq_len, quarter_dim]
    time_angles = time_freqs[time_positions]  # [seq_len, quarter_dim]

    # Combine into full rotation: first half for freq, second half for time
    # This gives each position unique 2D encoding
    angles = torch.cat([freq_angles, time_angles], dim=-1)  # [seq_len, head_dim//2]

    # Compute sin/cos
    cos = torch.cos(angles)  # [seq_len, head_dim//2]
    sin = torch.sin(angles)  # [seq_len, head_dim//2]

    # Expand for batch and heads: [1, 1, seq_len, head_dim//2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Split x into two halves for rotation
    x1, x2 = x[..., :head_dim // 2], x[..., head_dim // 2:]

    # Apply rotation: complex multiplication in real form
    # (x1 + i*x2) * (cos + i*sin) = (x1*cos - x2*sin) + i*(x1*sin + x2*cos)
    x_rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)

    return x_rotated


def _prepare_2d_attention_mask(
    key_padding_mask: torch.Tensor | None,
    B: int,
    M: int,
    T: int,
) -> torch.Tensor | None:
    """
    Prepare attention mask for 2D (M*T) sequence attention.

    Args:
        key_padding_mask: [B, T] where True indicates positions to mask.
        B: Batch size
        M: Frequency bins (mel bins)
        T: Time steps

    Returns:
        attn_mask: [B, 1, M*T, M*T] float mask with -inf for masked positions
    """
    if key_padding_mask is None:
        return None

    seq_len = M * T

    # key_padding_mask is [B, T] - need to expand to [B, M*T]
    # Each frequency bin at time t should be masked if t is masked
    # Reshape: [B, T] -> [B, 1, T] -> [B, M, T] -> [B, M*T]
    mask_2d = key_padding_mask.unsqueeze(1).expand(-1, M, -1)  # [B, M, T]
    mask_2d = mask_2d.reshape(B, seq_len)  # [B, M*T]

    # Expand for attention: [B, M*T] -> [B, 1, 1, M*T] for key masking
    attn_mask = mask_2d.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, M*T]
    attn_mask = attn_mask.expand(-1, -1, seq_len, -1)  # [B, 1, M*T, M*T]

    # Convert to float mask
    attn_mask = attn_mask.to(torch.float32)
    attn_mask = attn_mask.masked_fill(attn_mask == 1, float('-inf'))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


def _prepare_attention_mask(key_padding_mask: torch.Tensor | None, B: int, C: int, M: int, T: int) -> torch.Tensor | None:
    """
    Prepare attention mask for batched multi-head attention.

    Args:
        key_padding_mask: [B, T] where True indicates positions to mask (ignore).
        B: Batch size
        C: Channels / hidden dim (unused, kept for API consistency)
        M: Mel bins / width dim
        T: Time dim (sequence length)
    Returns:
        attn_mask: [B*M, T, T] float mask with -inf for masked positions, or None
    """
    if key_padding_mask is None:
        return None

    # Convert to float mask for attention
    attn_mask = key_padding_mask.unsqueeze(1).expand(-1, T, -1)  # [B*M, T, T]
    attn_mask = attn_mask.to(torch.float32)
    attn_mask = attn_mask.masked_fill(attn_mask == 1, float('-inf'))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
    attn_mask = attn_mask.unsqueeze(1)  # [B*M, 1, T, T] for head broadcasting

    return attn_mask

class AudioLinearSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, d_queries, d_values, use_flash_attention=True, dropout_p=0.1):
        super().__init__()
        self.hidden_dim = hidden_size
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.use_flash_attention = use_flash_attention
        
        self.q_proj = nn.Linear(hidden_size, d_queries * n_heads)
        self.k_proj = nn.Linear(hidden_size, d_queries * n_heads)
        self.v_proj = nn.Linear(hidden_size, d_values * n_heads)
        
        self.out_proj = nn.Linear(self.d_values * n_heads, hidden_size)
        
        self.dropout = nn.Dropout(dropout_p)

        self._init_weights()

    def _init_weights(self):
        self.apply(megatransformer_utils.transformer_weight_init())
        
    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Normal multi-head self attention, but it expects 4D input where it will batch by the first and third dimensions,
        and outputs the same shape.
        Args:
            x: [B, C, M, T] where B is batch size, C is channels, M is mel bins and T is time.
        Returns:
            output: [B, C, M, T] where B is batch size, C is channels, M is mel bins and T is time. Attention is applied
            along the T dimension, between the M dimension values, batched along B*M.
        """
        B, C, M, T = x.shape
        
        attn_mask = _prepare_attention_mask(key_padding_mask, B, C, M, T)

        x = x.permute(0, 2, 1, 3)  # [B, M, C, T]

        x = x.contiguous().view(-1, C, T)  # [B*M, C, T]
        x = x.permute(0, 2, 1)  # [B*M, T, C]
        
        q: torch.Tensor = self.q_proj(x)  # [B*M, T, n_heads*d_queries]
        k: torch.Tensor = self.k_proj(x)
        v: torch.Tensor = self.v_proj(x)
        
        q = q.view(-1, T, self.n_heads, self.d_queries)  # [B*M, T, n_heads, d_queries]
        k = k.view(-1, T, self.n_heads, self.d_queries)
        v = v.view(-1, T, self.n_heads, self.d_values)
        
        q = q.transpose(1, 2)  # [B*M, n_heads, T, d_queries]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        output: torch.Tensor
        if self.use_flash_attention:
            output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )  # [B*M, n_heads, T, d_values]
        else:
            scale = 1.0 / math.sqrt(self.d_queries)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B*M, n_heads, T, T]
            
            if attn_mask is not None:
                attn_scores = attn_scores + attn_mask

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            output = torch.matmul(attn_weights, v)  # [B*M, n_heads, T, d_queries]
        
        output = output.transpose(1, 2).contiguous()  # [B*M, T, n_heads, d_queries]

        output = output.view(-1, T, self.n_heads*self.d_values)  # [B*M, T, H]
        
        output = self.out_proj(output)  # [B*M, T, H]

        output = output.permute(0, 2, 1)  # [B*M, H, T]

        # restore input shape by splitting the hidden dim into width and height
        output = output.view(B, M, C, T)

        output = output.permute(0, 2, 1, 3)  # [B, C, M, T]
        
        return output


class AudioConvSelfAttentionBlock(nn.Module):
    """
    Multi-head self attention with convolutional Q/K/V projections.

    Uses 1D convolutions along the time dimension to capture local context
    before computing attention. This can help attention be more "locally aware"
    of spectral patterns like harmonics and formant transitions.
    """
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        d_queries: int,
        d_values: int,
        kernel_size: int = 3,
        use_depthwise: bool = True,
        use_flash_attention: bool = True,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.kernel_size = kernel_size
        self.use_depthwise = use_depthwise
        self.use_flash_attention = use_flash_attention

        # Convolutional Q/K/V projections
        # These operate along the time dimension to capture local context
        if use_depthwise:
            # Depthwise separable: depthwise conv -> pointwise conv
            # More parameter efficient while still capturing local patterns
            self.q_conv = nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2, groups=hidden_size),
                nn.Conv1d(hidden_size, d_queries * n_heads, 1),
            )
            self.k_conv = nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2, groups=hidden_size),
                nn.Conv1d(hidden_size, d_queries * n_heads, 1),
            )
            self.v_conv = nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2, groups=hidden_size),
                nn.Conv1d(hidden_size, d_values * n_heads, 1),
            )
        else:
            # Standard convolutions
            self.q_conv = nn.Conv1d(hidden_size, d_queries * n_heads, kernel_size, padding=kernel_size // 2)
            self.k_conv = nn.Conv1d(hidden_size, d_queries * n_heads, kernel_size, padding=kernel_size // 2)
            self.v_conv = nn.Conv1d(hidden_size, d_values * n_heads, kernel_size, padding=kernel_size // 2)

        self.out_proj = nn.Linear(d_values * n_heads, hidden_size)

        self.dropout = nn.Dropout(dropout_p)

        self._init_weights()

    def _init_weights(self):
        # Initialize conv layers
        for module in [self.q_conv, self.k_conv, self.v_conv]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Conv1d):
                        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='linear')
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='linear')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize output projection
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head self attention with convolutional Q/K/V.

        Args:
            x: [B, C, M, T] where B is batch size, C is channels,
               M is mel bins, T is time.
            key_padding_mask: [B, T] where True indicates positions to mask (ignore).
                              These are typically padded positions in variable-length sequences.
            return_attention_weights: If True, also return attention weights [B, M, n_heads, T, T].
                                      Note: This disables flash attention for this forward pass.
        Returns:
            output: [B, C, M, T] - same shape as input. Attention is applied
            along the T dimension with local context from convolutions.
            attn_weights (optional): [B, M, n_heads, T, T] if return_attention_weights=True
        """
        B, C, M, T = x.shape

        attn_mask = _prepare_attention_mask(key_padding_mask, B, C, M, T)

        # Reshape for processing: [B, C, M, T] -> [B*M, C, T]
        x = x.permute(0, 2, 1, 3)  # [B, M, C, T]
        x = x.contiguous().view(-1, C, T)  # [B*M, C, T]

        # Apply convolutional projections (conv1d expects [N, C, L])
        q: torch.Tensor = self.q_conv(x)  # [B*M, n_heads*d_queries, T]
        k: torch.Tensor = self.k_conv(x)  # [B*M, n_heads*d_queries, T]
        v: torch.Tensor = self.v_conv(x)  # [B*M, n_heads*d_values, T]

        # Reshape for attention: [B*M, n_heads*d, T] -> [B*M, n_heads, T, d]
        q = q.view(-1, self.n_heads, self.d_queries, T).permute(0, 1, 3, 2)  # [B*M, n_heads, T, d_queries]
        k = k.view(-1, self.n_heads, self.d_queries, T).permute(0, 1, 3, 2)  # [B*M, n_heads, T, d_queries]
        v = v.view(-1, self.n_heads, self.d_values, T).permute(0, 1, 3, 2)   # [B*M, n_heads, T, d_values]

        # Compute attention
        output: torch.Tensor
        attn_weights: torch.Tensor | None = None

        # Use manual attention if we need to return weights (flash attention doesn't return them)
        if self.use_flash_attention and not return_attention_weights:
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )  # [B*M, n_heads, T, d_values]
        else:
            scale = 1.0 / math.sqrt(self.d_queries)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B*M, n_heads, T, T]

            # Apply attention mask if provided
            if attn_mask is not None:
                attn_scores = attn_scores + attn_mask

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights_dropped = self.dropout(attn_weights) if self.training else attn_weights

            output = torch.matmul(attn_weights_dropped, v)  # [B*M, n_heads, T, d_values]

        # Reshape back: [B*M, n_heads, T, d_values] -> [B*M, T, n_heads*d_values]
        output = output.transpose(1, 2).contiguous()  # [B*M, T, n_heads, d_values]
        output = output.view(-1, T, self.n_heads * self.d_values)  # [B*M, T, n_heads*d_values]

        # Output projection
        output = self.out_proj(output)  # [B*M, T, C]

        # Reshape to original format
        output = output.permute(0, 2, 1)  # [B*M, C, T]
        output = output.view(B, M, C, T)  # [B, M, C, T]
        output = output.permute(0, 2, 1, 3)  # [B, C, M, T]

        if return_attention_weights and attn_weights is not None:
            # Reshape attention weights: [B*M, n_heads, T, T] -> [B, M, n_heads, T, T]
            attn_weights = attn_weights.view(B, M, self.n_heads, T, T)
            return output, attn_weights

        return output


class AudioConvSelfAttentionBlockWithGating(nn.Module):
    """
    Convolutional self attention with GLU-style gating on Q/K/V.

    Combines local convolutions with gating mechanism for more expressive
    feature extraction before attention.
    """
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        d_queries: int,
        d_values: int,
        kernel_size: int = 3,
        use_flash_attention: bool = True,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.kernel_size = kernel_size
        self.use_flash_attention = use_flash_attention

        # Gated convolutional projections
        # Output 2x channels for GLU gating
        self.q_conv = nn.Conv1d(hidden_size, 2 * d_queries * n_heads, kernel_size, padding=kernel_size // 2)
        self.k_conv = nn.Conv1d(hidden_size, 2 * d_queries * n_heads, kernel_size, padding=kernel_size // 2)
        self.v_conv = nn.Conv1d(hidden_size, 2 * d_values * n_heads, kernel_size, padding=kernel_size // 2)

        self.out_proj = nn.Linear(d_values * n_heads, hidden_size)

        self.dropout = nn.Dropout(dropout_p)

        self._init_weights()

    def _init_weights(self):
        for conv in [self.q_conv, self.k_conv, self.v_conv]:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='linear')
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _glu(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GLU activation: split in half, sigmoid gate one half."""
        x, gate = x.chunk(2, dim=1)
        return x * torch.sigmoid(gate)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, M, T]
            key_padding_mask: [B, T] where True indicates positions to mask (ignore).
                              These are typically padded positions in variable-length sequences.
            return_attention_weights: If True, also return attention weights [B, M, n_heads, T, T].
                                      Note: This disables flash attention for this forward pass.
        Returns:
            output: [B, C, M, T]
            attn_weights (optional): [B, M, n_heads, T, T] if return_attention_weights=True
        """
        B, C, M, T = x.shape

        attn_mask = _prepare_attention_mask(key_padding_mask, B, C, M, T)

        x = x.permute(0, 2, 1, 3)  # [B, M, C, T]
        x = x.contiguous().view(-1, C, T)  # [B*M, C, T]

        # Apply gated convolutional projections
        q = self._glu(self.q_conv(x))  # [B*M, n_heads*d_queries, T]
        k = self._glu(self.k_conv(x))  # [B*M, n_heads*d_queries, T]
        v = self._glu(self.v_conv(x))  # [B*M, n_heads*d_values, T]

        # Reshape for attention: [B*M, n_heads*d, T] -> [B*M, n_heads, T, d]
        q = q.view(-1, self.n_heads, self.d_queries, T).permute(0, 1, 3, 2)  # [B*M, n_heads, T, d_queries]
        k = k.view(-1, self.n_heads, self.d_queries, T).permute(0, 1, 3, 2)  # [B*M, n_heads, T, d_queries]
        v = v.view(-1, self.n_heads, self.d_values, T).permute(0, 1, 3, 2)   # [B*M, n_heads, T, d_values]

        # Compute attention
        output: torch.Tensor
        attn_weights: torch.Tensor | None = None

        # Use manual attention if we need to return weights (flash attention doesn't return them)
        if self.use_flash_attention and not return_attention_weights:
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )  # [B*M, n_heads, T, d_values]
        else:
            scale = 1.0 / math.sqrt(self.d_queries)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B*M, n_heads, T, T]

            # Apply attention mask if provided
            if attn_mask is not None:
                attn_scores = attn_scores + attn_mask

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights_dropped = self.dropout(attn_weights) if self.training else attn_weights

            output = torch.matmul(attn_weights_dropped, v)  # [B*M, n_heads, T, d_values]

        # Reshape back: [B*M, n_heads, T, d_values] -> [B*M, T, n_heads*d_values]
        output = output.transpose(1, 2).contiguous()  # [B*M, T, n_heads, d_values]
        output = output.view(-1, T, self.n_heads * self.d_values)  # [B*M, T, n_heads*d_values]

        # Output projection
        output = self.out_proj(output)  # [B*M, T, C]

        # Reshape to original format
        output = output.permute(0, 2, 1)  # [B*M, C, T]
        output = output.view(B, M, C, T)  # [B, M, C, T]
        output = output.permute(0, 2, 1, 3)  # [B, C, M, T]

        if return_attention_weights and attn_weights is not None:
            # Reshape attention weights: [B*M, n_heads, T, T] -> [B, M, n_heads, T, T]
            attn_weights = attn_weights.view(B, M, self.n_heads, T, T)
            return output, attn_weights

        return output


class AudioDiffusionCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, d_queries, d_values, context_dim=None, use_flash_attention=True, dropout_p=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.context_dim = context_dim or hidden_size  # If None, use hidden_dim
        self.use_flash_attention = use_flash_attention
        
        self.q_proj = nn.Linear(hidden_size, n_heads*d_queries)
        self.k_proj = nn.Linear(self.context_dim, n_heads*d_queries)
        self.v_proj = nn.Linear(self.context_dim, n_heads*d_values)
        
        self.out_proj = nn.Linear(n_heads*d_values, hidden_size)

        self.dropout = nn.Dropout(dropout_p)

        self._init_weights()
        self._init_context_projections()

    def _init_weights(self):
        self.apply(megatransformer_utils.transformer_weight_init())

    def _init_context_projections(self):
        """
        Properly initialize K/V projections for text conditioning.

        Text embeddings (e.g., from T5) often have std ~0.2-0.3, not 1.0.
        Standard xavier init results in very weak conditioning signal.

        This ensures K/V outputs have std ≈ 1.0 for proper attention scaling.
        """
        assumed_input_std = 0.2  # Common for T5, CLIP embeddings

        fan_in_k = self.context_dim
        target_k_std = 1.0 / (assumed_input_std * math.sqrt(fan_in_k))
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=target_k_std)

        fan_in_v = self.context_dim
        target_v_std = 1.0 / (assumed_input_std * math.sqrt(fan_in_v))
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=target_v_std)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, H, W, T = x.size()
        BC, N, CH = context.size()

        assert B == BC, f"Batch size mismatch: {B} vs {BC}. Shapes: {x.shape}, {context.shape}"

        x = x.permute(0, 2, 1, 3)  # [B, W, H, T]
        x = x.contiguous().view(B*W, H, T)    # [B*W, H, T]
        x = x.permute(0, 2, 1)  # [B*W, T, H]

        # context is 3D batched linear feature tokens, broadcast along the width dimension for attention
        context = context.unsqueeze(2).expand(-1, -1, W, -1)  # [B, N, W, CH]
        context = context.permute(0, 2, 3, 1)  # [B, W, CH, N]
        context = context.contiguous().view(B*W, CH, N)   # [B*W, CH, N]
        context = context.permute(0, 2, 1)  # [B*W, N, CH]

        q: torch.Tensor = self.q_proj(x)        # [B*W, T, n_heads*d_queries]
        k: torch.Tensor = self.k_proj(context)  # [B*W, N, n_heads*d_queries]
        v: torch.Tensor = self.v_proj(context)  # [B*W, N, n_heads*d_values]

        q = q.view(-1, T, self.n_heads, self.d_queries).transpose(1, 2)  # [B*W, n_heads, T, d_queries]
        k = k.view(-1, N, self.n_heads, self.d_queries).transpose(1, 2)  # [B*W, n_heads, N, d_queries]
        v = v.view(-1, N, self.n_heads, self.d_values).transpose(1, 2)  # [B*W, n_heads, N, d_values]

        output: torch.Tensor
        if self.use_flash_attention:
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )  # [B*W, n_heads, T, d_values]
        else:
            scale = 1.0 / math.sqrt(self.d_queries)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B*W, n_heads, T, N]
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            output = torch.matmul(attn_weights, v)  # [B*W, n_heads, T, d_values]
        
        output = output.transpose(1, 2).contiguous()  # [B*W, T, n_heads, head_dim]
        output = output.view(-1, T, self.n_heads*self.d_values)  # [B*W, T, n_heads*d_values]
        
        output = self.out_proj(output)  # [B*W, T, H]

        output = output.permute(0, 2, 1)  # [B*W, H, T]

        # restore input shape by splitting the hidden dim into width and height
        output = output.view(B, W, H, T)

        output = output.permute(0, 2, 1, 3)  # [B, H, W, T]

        return output


class AudioConv2DSelfAttentionBlock(nn.Module):
    """
    Full 2D self-attention over frequency (M) and time (T) dimensions with 2D RoPE.

    Unlike AudioConvSelfAttentionBlock which treats each mel bin independently,
    this module flattens M×T into a single sequence and applies attention across
    all positions. This enables:
    - Cross-frequency attention (harmonics, formant relationships)
    - Cross-time attention (temporal dependencies)
    - 2D positional encoding via RoPE

    For a bottleneck with M=10, T=75, this creates a 750-token sequence.
    Attention complexity is O((M*T)²) = O(562,500) vs O(M * T²) = O(56,250) for 1D.

    Uses 2D RoPE where:
    - First half of head dimension encodes frequency position
    - Second half encodes time position
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        d_queries: int,
        d_values: int,
        max_freq_positions: int = 128,  # Max M (mel bins after downsampling)
        max_time_positions: int = 512,  # Max T (timesteps after downsampling)
        kernel_size: int = 3,
        use_depthwise: bool = True,
        use_flash_attention: bool = True,
        dropout_p: float = 0.1,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.kernel_size = kernel_size
        self.use_depthwise = use_depthwise
        self.use_flash_attention = use_flash_attention
        self.max_freq_positions = max_freq_positions
        self.max_time_positions = max_time_positions
        self.rope_base = rope_base

        # Ensure head dimension is divisible by 4 for 2D RoPE
        assert d_queries % 4 == 0, f"d_queries must be divisible by 4 for 2D RoPE, got {d_queries}"

        # 2D convolutional Q/K/V projections
        # Use 2D conv to capture local 2D context before attention
        if use_depthwise:
            self.q_conv = nn.Sequential(
                nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2, groups=hidden_size),
                nn.Conv2d(hidden_size, d_queries * n_heads, 1),
            )
            self.k_conv = nn.Sequential(
                nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2, groups=hidden_size),
                nn.Conv2d(hidden_size, d_queries * n_heads, 1),
            )
            self.v_conv = nn.Sequential(
                nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2, groups=hidden_size),
                nn.Conv2d(hidden_size, d_values * n_heads, 1),
            )
        else:
            self.q_conv = nn.Conv2d(hidden_size, d_queries * n_heads, kernel_size, padding=kernel_size // 2)
            self.k_conv = nn.Conv2d(hidden_size, d_queries * n_heads, kernel_size, padding=kernel_size // 2)
            self.v_conv = nn.Conv2d(hidden_size, d_values * n_heads, kernel_size, padding=kernel_size // 2)

        self.out_proj = nn.Linear(d_values * n_heads, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

        # Precompute RoPE frequencies (will be moved to device on first forward)
        self._rope_freqs_computed = False
        self.register_buffer('freq_freqs', torch.zeros(max_freq_positions, d_queries // 2), persistent=False)
        self.register_buffer('time_freqs', torch.zeros(max_time_positions, d_queries // 2), persistent=False)

        self._init_weights()

    def _init_weights(self):
        for module in [self.q_conv, self.k_conv, self.v_conv]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, (nn.Conv1d, nn.Conv2d)):
                        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='linear')
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
            elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='linear')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _ensure_rope_freqs(self, device: torch.device):
        """Lazily compute RoPE frequencies on the correct device."""
        if not self._rope_freqs_computed or self.freq_freqs.device != device:
            freq_freqs, time_freqs = _compute_2d_rope_freqs(
                dim=self.d_queries,
                max_freq=self.max_freq_positions,
                max_time=self.max_time_positions,
                base=self.rope_base,
                device=device,
            )
            self.freq_freqs = freq_freqs
            self.time_freqs = time_freqs
            self._rope_freqs_computed = True

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Full 2D self-attention with 2D RoPE positional encoding.

        Args:
            x: [B, C, M, T] where B is batch size, C is channels,
               M is frequency bins (mel bins), T is time.
            key_padding_mask: [B, T] where True indicates time positions to mask.
                              Mask is expanded to cover all frequency bins at masked times.
            return_attention_weights: If True, return attention weights [B, n_heads, M*T, M*T].

        Returns:
            output: [B, C, M, T] - same shape as input
            attn_weights (optional): [B, n_heads, M*T, M*T] if return_attention_weights=True
        """
        B, C, M, T = x.shape
        seq_len = M * T

        # Ensure RoPE frequencies are computed
        self._ensure_rope_freqs(x.device)

        # Prepare 2D attention mask
        attn_mask = _prepare_2d_attention_mask(key_padding_mask, B, M, T)

        # Apply 2D convolutional projections: [B, C, M, T] -> [B, n_heads*d, M, T]
        q: torch.Tensor = self.q_conv(x)  # [B, n_heads*d_queries, M, T]
        k: torch.Tensor = self.k_conv(x)  # [B, n_heads*d_queries, M, T]
        v: torch.Tensor = self.v_conv(x)  # [B, n_heads*d_values, M, T]

        # Flatten spatial dimensions: [B, n_heads*d, M, T] -> [B, n_heads*d, M*T]
        q = q.view(B, self.n_heads * self.d_queries, seq_len)
        k = k.view(B, self.n_heads * self.d_queries, seq_len)
        v = v.view(B, self.n_heads * self.d_values, seq_len)

        # Reshape for attention: [B, n_heads*d, seq] -> [B, n_heads, seq, d]
        q = q.view(B, self.n_heads, self.d_queries, seq_len).permute(0, 1, 3, 2)  # [B, n_heads, M*T, d_queries]
        k = k.view(B, self.n_heads, self.d_queries, seq_len).permute(0, 1, 3, 2)  # [B, n_heads, M*T, d_queries]
        v = v.view(B, self.n_heads, self.d_values, seq_len).permute(0, 1, 3, 2)   # [B, n_heads, M*T, d_values]

        # Create 2D position indices for RoPE
        # Positions are laid out as: (0,0), (0,1), ..., (0,T-1), (1,0), ..., (M-1,T-1)
        freq_positions = torch.arange(M, device=x.device).unsqueeze(1).expand(M, T).reshape(-1)  # [M*T]
        time_positions = torch.arange(T, device=x.device).unsqueeze(0).expand(M, T).reshape(-1)  # [M*T]

        # Apply 2D RoPE to queries and keys
        q = _apply_2d_rope(q, self.freq_freqs, self.time_freqs, freq_positions, time_positions)
        k = _apply_2d_rope(k, self.freq_freqs, self.time_freqs, freq_positions, time_positions)

        # Compute attention
        output: torch.Tensor
        attn_weights: torch.Tensor | None = None

        if self.use_flash_attention and not return_attention_weights:
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )  # [B, n_heads, M*T, d_values]
        else:
            scale = 1.0 / math.sqrt(self.d_queries)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, n_heads, M*T, M*T]

            if attn_mask is not None:
                attn_scores = attn_scores + attn_mask

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights_dropped = self.dropout(attn_weights) if self.training else attn_weights

            output = torch.matmul(attn_weights_dropped, v)  # [B, n_heads, M*T, d_values]

        # Reshape: [B, n_heads, M*T, d_values] -> [B, M*T, n_heads*d_values]
        output = output.transpose(1, 2).contiguous()  # [B, M*T, n_heads, d_values]
        output = output.view(B, seq_len, self.n_heads * self.d_values)  # [B, M*T, n_heads*d_values]

        # Output projection
        output = self.out_proj(output)  # [B, M*T, C]

        # Reshape back to 2D: [B, M*T, C] -> [B, C, M, T]
        output = output.permute(0, 2, 1)  # [B, C, M*T]
        output = output.view(B, C, M, T)  # [B, C, M, T]

        if return_attention_weights and attn_weights is not None:
            # attn_weights shape: [B, n_heads, M*T, M*T]
            return output, attn_weights

        return output
