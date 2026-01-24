"""
GuBERT: Speaker-Invariant Speech Encoder

A compact speech encoder trained with ASR (CTC) + GRL speaker disentanglement.
Produces content-focused features that explicitly exclude speaker characteristics.

The key insight: ASR loss wants content, GRL removes speaker - these objectives
are aligned rather than conflicting (unlike reconstruction + GRL in VAEs).

Usage:
    model = GuBERTEncoder.from_config("small", num_speakers=992)
    features, asr_logits, speaker_logits = model(mel_spec, grl_alpha=1.0)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

from model import activations
from utils.model_utils import get_activation_type


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial training.
    Forward: identity
    Backward: negates gradients scaled by alpha
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.alpha * grad_output, None


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            [B, T, D] with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ConvSubsampling(nn.Module):
    """
    Convolutional subsampling frontend (similar to Conformer/wav2vec2).
    Reduces temporal resolution while extracting local features.

    Uses Dropout1d which drops entire channels rather than individual elements,
    providing more structured regularization for convolutional features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list = [5, 3, 3],
        strides: list = [2, 2, 1],
        dropout: float = 0.05,  # Dropout1d drops channels - use lower values (0.02-0.1)
    ):
        super().__init__()

        layers = []
        channels = [in_channels] + [out_channels] * len(kernel_sizes)

        for i, (k, s) in enumerate(zip(kernel_sizes, strides)):
            layers.extend([
                nn.Conv1d(channels[i], channels[i + 1], kernel_size=k, stride=s, padding=k // 2),
                nn.GroupNorm(min(32, channels[i + 1]), channels[i + 1]),
                nn.GELU(),
                nn.Dropout1d(dropout),  # Channel dropout - more principled for conv
            ])

        self.conv = nn.Sequential(*layers)

        # Compute total stride for length calculation
        self.total_stride = 1
        for s in strides:
            self.total_stride *= s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, T]
        Returns:
            [B, out_channels, T // total_stride]
        """
        return self.conv(x)

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input length."""
        return (input_length + self.total_stride - 1) // self.total_stride


class HeadDropout(nn.Module):
    """
    DropHead: Randomly drops entire attention heads during training.

    This regularization technique forces the model to not over-rely on any
    single attention head, improving generalization. Similar to dropout but
    operates on entire head channels rather than individual elements.

    Reference: "Scheduled DropHead" (Zhou et al., 2020)

    Args:
        num_heads: Number of attention heads
        head_dim: Dimension per head (d_model // num_heads)
        drop_prob: Probability of dropping each head (0.0 = no dropout)
    """

    def __init__(self, num_heads: int, head_dim: int, drop_prob: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply head dropout to attention output.

        Args:
            x: [B, T, D] attention output where D = num_heads * head_dim

        Returns:
            [B, T, D] with randomly dropped heads (scaled appropriately)
        """
        if not self.training or self.drop_prob == 0.0:
            return x

        B, T, D = x.shape

        # Reshape to expose heads: [B, T, num_heads, head_dim]
        x = x.view(B, T, self.num_heads, self.head_dim)

        # Generate head mask: [B, 1, num_heads, 1] - same mask for all timesteps
        # Each head is independently dropped with probability drop_prob
        head_mask = torch.bernoulli(
            torch.full((B, 1, self.num_heads, 1), 1.0 - self.drop_prob, device=x.device)
        )

        # Ensure at least one head is kept (avoid all-zeros)
        # If all heads would be dropped, keep a random one
        all_dropped = (head_mask.sum(dim=2, keepdim=True) == 0)
        if all_dropped.any():
            # Pick a random head to keep for each sample where all were dropped
            random_head = torch.randint(0, self.num_heads, (B, 1, 1, 1), device=x.device)
            keep_mask = (torch.arange(self.num_heads, device=x.device).view(1, 1, -1, 1) == random_head)
            head_mask = torch.where(all_dropped, keep_mask.float(), head_mask)

        # Apply mask and scale (like standard dropout)
        # Scale by 1/(1-p) to maintain expected value, accounting for kept heads
        keep_prob = head_mask.mean(dim=2, keepdim=True).clamp(min=1e-6)
        x = x * head_mask / keep_prob

        # Reshape back: [B, T, D]
        return x.view(B, T, D)

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, head_dim={self.head_dim}, drop_prob={self.drop_prob}"


class ConformerConvModule(nn.Module):
    """
    Conformer-style convolution module for speech processing.

    Architecture: LayerNorm -> Pointwise Conv -> GLU -> Depthwise Conv -> BatchNorm -> Swish -> Pointwise Conv -> Dropout

    The depthwise separable convolution captures local acoustic patterns that
    self-attention alone may miss, making it particularly effective for speech.

    Reference: "Conformer: Convolution-augmented Transformer for Speech Recognition"
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        inner_dim = d_model * expansion_factor

        self.norm = nn.LayerNorm(d_model)

        # Pointwise expansion with GLU (doubles channels, GLU halves them)
        self.pointwise_conv1 = nn.Conv1d(d_model, inner_dim * 2, kernel_size=1)

        # Depthwise conv (processes each channel independently)
        self.depthwise_conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=inner_dim,  # Depthwise: each channel has its own filter
        )

        self.batch_norm = nn.BatchNorm1d(inner_dim)
        self.activation = nn.SiLU()  # Swish

        # Pointwise projection back to d_model
        self.pointwise_conv2 = nn.Conv1d(inner_dim, d_model, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            [B, T, D]
        """
        # Pre-norm
        x = self.norm(x)

        # [B, T, D] -> [B, D, T] for conv
        x = x.transpose(1, 2)

        # Pointwise expansion + GLU
        x = self.pointwise_conv1(x)  # [B, inner_dim*2, T]
        x = F.glu(x, dim=1)  # [B, inner_dim, T]

        # Depthwise conv
        x = self.depthwise_conv(x)  # [B, inner_dim, T]
        x = self.batch_norm(x)
        x = self.activation(x)

        # Pointwise projection
        x = self.pointwise_conv2(x)  # [B, D, T]
        x = self.dropout(x)

        # [B, D, T] -> [B, T, D]
        return x.transpose(1, 2)


class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block with pre-norm, optional RoPE, Conformer conv, and configurable activation.

    Supports:
    - Pre-norm architecture (more stable training)
    - Rotary Position Embeddings (RoPE) for better position modeling
    - Conformer-style convolution module for local acoustic patterns
    - Macaron-style FFN (half-step FFN before and after attention)
    - SwiGLU or GELU activation in FFN
    - DropHead regularization
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        head_drop_prob: float = 0.0,
        # Architectural options
        use_rotary_embedding: bool = False,
        use_conformer_conv: bool = False,
        conformer_kernel_size: int = 31,
        use_macaron: bool = False,  # Macaron-style: ½FFN → Attn → Conv → ½FFN
        activation: str = "gelu",  # "gelu" or "swiglu"
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.use_rotary_embedding = use_rotary_embedding
        self.use_macaron = use_macaron

        # Rotary position embeddings
        if use_rotary_embedding:
            self.rotary_embedding = RotaryEmbedding(dim=self.head_dim)
            # Custom attention with RoPE (can't use nn.MultiheadAttention)
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.o_proj = nn.Linear(d_model, d_model)
            self.attn_dropout = nn.Dropout(dropout)
        else:
            self.rotary_embedding = None
            self.self_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )

        # Head dropout: drops entire attention heads for regularization
        self.head_dropout = HeadDropout(
            num_heads=n_heads,
            head_dim=self.head_dim,
            drop_prob=head_drop_prob,
        ) if head_drop_prob > 0 else nn.Identity()

        # Conformer conv module (between attention and FFN)
        self.conv_module = ConformerConvModule(
            d_model=d_model,
            kernel_size=conformer_kernel_size,
            dropout=dropout,
        ) if use_conformer_conv else None

        # Helper to build FFN
        def build_ffn():
            activation_type = get_activation_type(activation)
            if activation_type == activations.SwiGLU:
                return nn.Sequential(
                    nn.Linear(d_model, d_ff * 2),
                    activations.SwiGLUSimple(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout),
                )
            else:
                return nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    activation_type(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout),
                )

        # Macaron-style: two half-step FFNs (each scaled by 0.5)
        # Standard: single FFN
        if use_macaron:
            self.ff1 = build_ffn()  # First half-step FFN
            self.ff2 = build_ffn()  # Second half-step FFN
            self.norm_ff1 = nn.LayerNorm(d_model)
            self.norm_ff2 = nn.LayerNorm(d_model)
            self.ff = None  # Not used in Macaron mode
            self.norm2 = None  # Not used in Macaron mode
        else:
            self.ff = build_ffn()
            self.ff1 = None
            self.ff2 = None
            self.norm_ff1 = None
            self.norm_ff2 = None
            self.norm2 = nn.LayerNorm(d_model)

        self.norm1 = nn.LayerNorm(d_model)  # For attention
        if use_conformer_conv:
            self.norm_conv = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _rope_attention(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Self-attention with rotary position embeddings."""
        B, T, D = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D_h]
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        q = self.rotary_embedding.rotate_queries_or_keys(q)
        k = self.rotary_embedding.rotate_queries_or_keys(k)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, T, T]

        # Apply key padding mask
        if key_padding_mask is not None:
            # key_padding_mask: [B, T] True for padded positions
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # [B, 1, 1, T]
                float('-inf')
            )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [B, H, T, D_h]
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]

        return self.o_proj(out)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            attn_mask: [T, T] attention mask (not used with RoPE)
            key_padding_mask: [B, T] True for padded positions
        Returns:
            [B, T, D]

        Macaron-style (if enabled): ½FFN → Attn → Conv → ½FFN
        Standard: Attn → Conv → FFN
        """
        if self.use_macaron:
            # Macaron: First half-step FFN (scaled by 0.5)
            residual = x
            x = self.norm_ff1(x)
            x = 0.5 * self.ff1(x)
            x = residual + x

        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        if self.use_rotary_embedding:
            x = self._rope_attention(x, key_padding_mask=key_padding_mask)
        else:
            x, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.head_dropout(x)  # DropHead: randomly drop entire attention heads
        x = self.dropout(x)
        x = residual + x

        # Conformer conv module (if enabled)
        if self.conv_module is not None:
            residual = x
            x = self.conv_module(x)
            x = residual + x

        # Feedforward
        if self.use_macaron:
            # Macaron: Second half-step FFN (scaled by 0.5)
            residual = x
            x = self.norm_ff2(x)
            x = 0.5 * self.ff2(x)
            x = residual + x
        else:
            # Standard: Full FFN
            residual = x
            x = self.norm2(x)
            x = self.ff(x)
            x = residual + x

        return x


class SpeakerClassifier(nn.Module):
    """
    Speaker classifier head with modern pooling strategies.
    Used with gradient reversal to push speaker info out of features.

    Pooling options:
    - "mean": Simple mean pooling
    - "statistics": Mean + std concatenation (2x input dim)
    - "attention": Learnable attention weights
    - "attentive_statistics": ASP from ECAPA-TDNN (attention-weighted mean + std)
    - "multi_head_attention": Multi-head self-attention pooling
    """

    def __init__(
        self,
        d_model: int,
        num_speakers: int,
        hidden_dim: Optional[int] = None,
        pooling: str = "attentive_statistics",
        dropout: float = 0.1,
        num_attention_heads: int = 4,
    ):
        super().__init__()

        hidden_dim = hidden_dim or d_model * 2
        self.pooling = pooling
        self.d_model = d_model

        # Determine pooled dimension based on pooling type
        if pooling in ("statistics", "attentive_statistics"):
            pooled_dim = d_model * 2  # mean + std
        else:
            pooled_dim = d_model

        # Pooling-specific layers
        if pooling == "attention":
            self.attn_pool = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Tanh(),
                nn.Linear(d_model, 1),
            )

        elif pooling == "attentive_statistics":
            # Attentive Statistics Pooling (ASP) from ECAPA-TDNN
            # Uses attention to compute weighted mean and std
            self.asp_linear = nn.Linear(d_model, d_model)
            self.asp_attention = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm1d(d_model),
                nn.Conv1d(d_model, d_model, kernel_size=1),
                nn.Softmax(dim=2),
            )

        elif pooling == "multi_head_attention":
            # Learnable query for pooling
            self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
            self.mha_pool = nn.MultiheadAttention(
                d_model, num_attention_heads, dropout=dropout, batch_first=True
            )
            self.mha_norm = nn.LayerNorm(d_model)

        # Classifier MLP
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_speakers),
        )

    def _statistics_pooling(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute mean and std pooling."""
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()  # [B, T, 1]
            lengths = mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]

            # Masked mean
            mean = (x * mask_expanded).sum(dim=1) / lengths

            # Masked std
            diff_sq = ((x - mean.unsqueeze(1)) ** 2) * mask_expanded
            var = diff_sq.sum(dim=1) / lengths.clamp(min=1)
            std = (var + 1e-6).sqrt()
        else:
            mean = x.mean(dim=1)
            std = x.std(dim=1) + 1e-6

        return torch.cat([mean, std], dim=-1)  # [B, 2*D]

    def _attentive_statistics_pooling(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Attentive Statistics Pooling (ASP) from ECAPA-TDNN.
        Computes attention-weighted mean and std.
        """
        # x: [B, T, D]
        h = self.asp_linear(x)  # [B, T, D]

        # Compute attention weights using Conv1d (expects [B, D, T])
        h_transposed = h.transpose(1, 2)  # [B, D, T]
        attn_weights = self.asp_attention(h_transposed)  # [B, D, T] with softmax over T
        attn_weights = attn_weights.transpose(1, 2)  # [B, T, D]

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()  # [B, T, 1]
            attn_weights = attn_weights * mask_expanded
            # Re-normalize after masking
            attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

        # Weighted mean
        weighted_mean = (x * attn_weights).sum(dim=1)  # [B, D]

        # Weighted std
        diff_sq = (x - weighted_mean.unsqueeze(1)) ** 2
        weighted_var = (diff_sq * attn_weights).sum(dim=1)
        weighted_std = (weighted_var + 1e-6).sqrt()  # [B, D]

        return torch.cat([weighted_mean, weighted_std], dim=-1)  # [B, 2*D]

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] features
            mask: [B, T] True for valid positions
        Returns:
            [B, num_speakers] speaker logits
        """
        if self.pooling == "mean":
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled = x.mean(dim=1)

        elif self.pooling == "max":
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            pooled = x.max(dim=1)[0]

        elif self.pooling == "statistics":
            pooled = self._statistics_pooling(x, mask)

        elif self.pooling == "attentive_statistics":
            pooled = self._attentive_statistics_pooling(x, mask)

        elif self.pooling == "attention":
            attn_weights = self.attn_pool(x).squeeze(-1)  # [B, T]
            if mask is not None:
                attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1)  # [B, T]
            pooled = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]

        elif self.pooling == "multi_head_attention":
            B = x.size(0)
            query = self.pool_query.expand(B, -1, -1)  # [B, 1, D]

            # Create key_padding_mask (True = ignore)
            key_padding_mask = ~mask if mask is not None else None

            pooled, _ = self.mha_pool(query, x, x, key_padding_mask=key_padding_mask)
            pooled = self.mha_norm(pooled.squeeze(1))  # [B, D]

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")

        return self.classifier(pooled)


@dataclass
class GuBERTConfig:
    """Configuration for GuBERT model."""
    n_mels: int = 80
    encoder_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ff_dim: int = 1024
    conv_kernel_sizes: list = None
    conv_strides: list = None
    vocab_size: int = 32  # Characters + blank for CTC
    num_speakers: int = 992
    dropout: float = 0.1
    max_seq_len: int = 5000
    speaker_pooling: str = "mean"

    # Dropout regularization (helps prevent memorization)
    conv_dropout: float = 0.0       # Dropout1d in conv frontend (0 = use standard dropout)
    feature_dropout: float = 0.0    # Dropout on features before heads
    head_dropout: float = 0.0       # Dropout in ASR head
    attention_head_drop: float = 0.0  # DropHead on attention

    # Architectural options
    use_rotary_embedding: bool = False  # Use RoPE instead of sinusoidal positional encoding
    use_conformer_conv: bool = False    # Add Conformer-style conv module to each block
    conformer_kernel_size: int = 31     # Kernel size for Conformer conv (standard from paper)
    use_macaron: bool = False           # Macaron-style FFN: ½FFN → Attn → Conv → ½FFN
    activation: str = "gelu"            # FFN activation: "gelu" or "swiglu"

    # Speaker normalization (strips speaker-specific statistics from features)
    # Instance norm normalizes each sample across time, removing per-utterance mean/variance
    # which often encodes speaker identity. More direct than adversarial GRL.
    use_instance_norm: bool = False     # Add InstanceNorm after each transformer block
    instance_norm_affine: bool = False  # Learn scale/shift (False = pure normalization)

    # Variance regularization (for VAE-friendly features)
    use_variance_reg: bool = False
    temporal_var_weight: float = 0.01
    temporal_var_min: float = 0.1
    dim_var_weight: float = 0.01
    dim_var_min: float = 0.1
    temporal_smoothness_weight: float = 0.1
    temporal_smoothness_max: float = 0.95

    # CTC upsampling (relaxes CTC length constraint without increasing transformer cost)
    # Upsamples features before CTC head using linear interpolation
    # factor=1: no upsampling (default), factor=2: 2x more CTC frames, etc.
    ctc_upsample_factor: int = 1

    speaker_classifier_hidden_dim: Optional[int] = None

    def __post_init__(self):
        if self.conv_kernel_sizes is None:
            self.conv_kernel_sizes = [7, 3, 3]  # Larger first kernel for more acoustic context
        if self.conv_strides is None:
            self.conv_strides = [2, 2, 1]  # 4x downsampling

        if self.speaker_classifier_hidden_dim is None:
            self.speaker_classifier_hidden_dim = self.encoder_dim * 2

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        import dataclasses
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        import json
        return json.dumps(self.to_dict(), indent=2)


# Predefined configurations
GUBERT_CONFIGS = {
    # ~1.3M params - very fast, good for experimentation
    "tiny": GuBERTConfig(
        encoder_dim=128,
        num_layers=3,
        num_heads=4,
        ff_dim=512,
        dropout=0.1,
    ),
    # ~7M params w/ macaron and swiglu, ~2.5M w/o
    "tiny_deep": GuBERTConfig(
        encoder_dim=128,
        num_layers=12,
        num_heads=8,
        ff_dim=512,
        dropout=0.1,
    ),
    # ~9.6M params w/ macaron and swiglu, ~3.7M w/o
    "small": GuBERTConfig(
        encoder_dim=256,
        num_layers=4,
        num_heads=4,
        ff_dim=1024,
        dropout=0.1,
    ),
    # ~30M params w/ macaron and swiglu, ~11M w/o
    "small_deep": GuBERTConfig(
        encoder_dim=256,
        num_layers=12,
        num_heads=8,
        ff_dim=1152,
        dropout=0.1,
    ),
    # ~32M params w/ macaron and swiglu, ~11M w/o
    "medium": GuBERTConfig(
        encoder_dim=384,
        num_layers=6,
        num_heads=6,
        ff_dim=1536,
        dropout=0.1,
    ),
    # ~61M params w/ macaron and swiglu, ~23M w/o
    "medium_deep": GuBERTConfig(
        encoder_dim=384,
        num_layers=12,
        num_heads=6,
        ff_dim=1536,
        dropout=0.1,
    ),
    # ~74M params w/ macaron and swiglu, ~27M w/o
    "large": GuBERTConfig(
        encoder_dim=512,
        num_layers=8,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1,
    ),
    # 110M params w/ macaron and swiglu, ~40M w/o
    "large_deep": GuBERTConfig(
        encoder_dim=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1,
    ),
}


class GuBERTEncoder(nn.Module):
    """
    GuBERT: Speaker-Invariant Speech Encoder

    Trained with dual objectives:
    1. ASR (CTC loss) - forces encoding of linguistic content
    2. GRL speaker classification - removes speaker information

    The resulting features are content-focused and speaker-agnostic,
    suitable as targets for a content VAE or direct use in TTS.
    """

    def __init__(self, config: GuBERTConfig):
        super().__init__()
        self.config = config

        # Convolutional subsampling frontend with optional Dropout1d
        self.conv_subsample = ConvSubsampling(
            in_channels=config.n_mels,
            out_channels=config.encoder_dim,
            kernel_sizes=config.conv_kernel_sizes,
            strides=config.conv_strides,
            dropout=config.conv_dropout if config.conv_dropout > 0 else config.dropout,
        )

        # Positional encoding (only used when not using RoPE)
        if not config.use_rotary_embedding:
            self.pos_enc = SinusoidalPositionalEncoding(
                d_model=config.encoder_dim,
                max_len=config.max_seq_len,
                dropout=config.dropout,
            )
        else:
            self.pos_enc = None  # RoPE handles position in each attention block

        # Transformer encoder blocks with architectural options
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=config.encoder_dim,
                n_heads=config.num_heads,
                d_ff=config.ff_dim,
                dropout=config.dropout,
                head_drop_prob=config.attention_head_drop,
                use_rotary_embedding=config.use_rotary_embedding,
                use_conformer_conv=config.use_conformer_conv,
                conformer_kernel_size=config.conformer_kernel_size,
                use_macaron=config.use_macaron,
                activation=config.activation,
            )
            for _ in range(config.num_layers)
        ])

        # Instance normalization for speaker removal (optional)
        # Normalizes across time dimension, stripping per-utterance mean/variance
        # which typically encodes speaker identity
        if config.use_instance_norm:
            self.instance_norms = nn.ModuleList([
                nn.InstanceNorm1d(config.encoder_dim, affine=config.instance_norm_affine)
                for _ in range(config.num_layers)
            ])
        else:
            self.instance_norms = None

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.encoder_dim)

        # Feature dropout (applied before heads)
        self.feature_dropout = nn.Dropout(config.feature_dropout) if config.feature_dropout > 0 else nn.Identity()

        # ASR head (CTC) with optional dropout
        if config.head_dropout > 0:
            self.asr_head = nn.Sequential(
                nn.Linear(config.encoder_dim, config.encoder_dim),
                nn.GELU(),
                nn.Dropout(config.head_dropout),
                nn.Linear(config.encoder_dim, config.vocab_size),
            )
        else:
            self.asr_head = nn.Linear(config.encoder_dim, config.vocab_size)

        # Speaker classifier (for GRL)
        self.speaker_classifier = SpeakerClassifier(
            d_model=config.encoder_dim,
            num_speakers=config.num_speakers,
            hidden_dim=config.speaker_classifier_hidden_dim,
            pooling=config.speaker_pooling,
            dropout=config.dropout,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @classmethod
    def from_config(cls, config_name: str, **overrides) -> "GuBERTEncoder":
        """
        Create model from predefined config with optional overrides.

        Args:
            config_name: One of "tiny", "small", "medium", "large"
            **overrides: Override any config parameter

        Example:
            model = GuBERTEncoder.from_config("small", num_speakers=500)
        """
        if config_name not in GUBERT_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(GUBERT_CONFIGS.keys())}")

        config = GUBERT_CONFIGS[config_name]
        # Apply overrides
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = GuBERTConfig(**config_dict)

        return cls(config)

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input mel spectrogram length."""
        return self.conv_subsample.get_output_length(input_length)

    def forward(
        self,
        mel_spec: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        grl_alpha: float = 1.0,
        return_all_hiddens: bool = False,
    ) -> dict:
        """
        Forward pass through GuBERT encoder.

        Args:
            mel_spec: [B, n_mels, T] mel spectrogram
            lengths: [B] original lengths before padding (optional)
            grl_alpha: Gradient reversal strength (0=no reversal, 1=full reversal)
            return_all_hiddens: If True, return hidden states from all layers

        Returns:
            dict with keys:
                - features: [B, T', encoder_dim] normalized content features
                - features_unnorm: [B, T', encoder_dim] pre-LayerNorm features
                - asr_logits: [B, T', vocab_size] for CTC loss
                - speaker_logits: [B, num_speakers] for GRL loss
                - feature_lengths: [B] output sequence lengths
                - variance_loss: scalar loss for variance regularization (if enabled)
                - temporal_smoothness: scalar metric (if variance_reg enabled and training)
                - all_hiddens: list of [B, T', D] if return_all_hiddens=True
        """
        # Convolutional frontend
        x = self.conv_subsample(mel_spec)  # [B, encoder_dim, T']
        x = x.permute(0, 2, 1)  # [B, T', encoder_dim]

        # Calculate output lengths
        feature_lengths = None
        padding_mask = None
        if lengths is not None:
            feature_lengths = torch.tensor(
                [self.get_output_length(l.item()) for l in lengths],
                device=mel_spec.device,
                dtype=torch.long,
            )
            # Create padding mask [B, T'] - True for padded positions
            max_len = x.size(1)
            padding_mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= feature_lengths.unsqueeze(1)

        # Positional encoding (skip if using RoPE - handled in attention)
        if self.pos_enc is not None:
            x = self.pos_enc(x)

        # Transformer encoder
        all_hiddens = [x] if return_all_hiddens else None
        for i, block in enumerate(self.encoder_blocks):
            x = block(x, key_padding_mask=padding_mask)

            # Apply instance normalization to strip speaker statistics (if enabled)
            # InstanceNorm1d expects [B, C, T], we have [B, T, C]
            if self.instance_norms is not None:
                x = x.transpose(1, 2)  # [B, C, T]
                x = self.instance_norms[i](x)
                x = x.transpose(1, 2)  # [B, T, C]

            if return_all_hiddens:
                all_hiddens.append(x)

        # Store pre-LayerNorm features
        features_unnorm = x

        # Final normalization
        features = self.final_norm(x)

        # Compute variance regularization loss (for VAE-friendly features)
        variance_loss = torch.tensor(0.0, device=mel_spec.device)
        temporal_smoothness = None
        if self.config.use_variance_reg:
            # Use normalized features for variance computation
            feat_for_var = features

            # Temporal variance: want features to change over time (not constant)
            if feat_for_var.size(1) > 1:
                temporal_diff = feat_for_var[:, 1:, :] - feat_for_var[:, :-1, :]
                temporal_var = temporal_diff.var(dim=1).mean()
                temporal_loss = F.relu(self.config.temporal_var_min - temporal_var)

                # Temporal smoothness (for logging) - cosine similarity between adjacent frames
                with torch.no_grad():
                    feat_norm = F.normalize(feat_for_var, dim=-1)
                    cos_sim = (feat_norm[:, 1:, :] * feat_norm[:, :-1, :]).sum(dim=-1)
                    temporal_smoothness = cos_sim.mean()
            else:
                temporal_loss = torch.tensor(0.0, device=mel_spec.device)

            # Per-dimension variance: want each dimension to be used (not dead)
            dim_var = feat_for_var.var(dim=(0, 1))
            dim_loss = F.relu(self.config.dim_var_min - dim_var).mean()

            # Temporal smoothness penalty: penalize if features are too smooth
            smoothness_loss = torch.tensor(0.0, device=mel_spec.device)
            if temporal_smoothness is not None:
                smoothness_loss = F.relu(temporal_smoothness - self.config.temporal_smoothness_max)

            variance_loss = (
                self.config.temporal_var_weight * temporal_loss +
                self.config.dim_var_weight * dim_loss +
                self.config.temporal_smoothness_weight * smoothness_loss
            )

        # Apply feature dropout before heads
        features_for_heads = self.feature_dropout(features)

        # CTC upsampling: upsample features before ASR head to relax CTC length constraint
        # This keeps transformer efficient (operates on T/4) while giving CTC more frames
        ctc_lengths = feature_lengths
        if self.config.ctc_upsample_factor > 1:
            # Linear interpolation upsampling: [B, T, D] -> [B, T*factor, D]
            features_for_asr = features_for_heads.transpose(1, 2)  # [B, D, T]
            features_for_asr = F.interpolate(
                features_for_asr,
                scale_factor=float(self.config.ctc_upsample_factor),
                mode='linear',
                align_corners=False,
            )
            features_for_asr = features_for_asr.transpose(1, 2)  # [B, T*factor, D]

            # Update CTC lengths to reflect upsampling
            if feature_lengths is not None:
                ctc_lengths = feature_lengths * self.config.ctc_upsample_factor
        else:
            features_for_asr = features_for_heads

        # ASR head (operates on potentially upsampled features)
        asr_logits = self.asr_head(features_for_asr)

        # Speaker classifier with gradient reversal (operates on original resolution)
        # Use ~padding_mask to get valid positions mask
        valid_mask = ~padding_mask if padding_mask is not None else None
        reversed_features = GradientReversalFunction.apply(features_for_heads, grl_alpha)
        speaker_logits = self.speaker_classifier(reversed_features, mask=valid_mask)

        result = {
            "features": features,  # Normalized features (preferred for VAE)
            "features_unnorm": features_unnorm,  # Pre-LayerNorm (for comparison)
            "asr_logits": asr_logits,
            "speaker_logits": speaker_logits,
            "feature_lengths": feature_lengths,  # Original feature lengths (for feature extraction)
            "ctc_lengths": ctc_lengths,  # CTC lengths (potentially upsampled, for CTC loss)
            "variance_loss": variance_loss,
            "temporal_smoothness": temporal_smoothness if self.config.use_variance_reg and self.training else None,
        }

        if return_all_hiddens:
            result["all_hiddens"] = all_hiddens

        return result

    def extract_features(
        self,
        mel_spec: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        layer: int = -1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract features without computing ASR/speaker heads.
        Useful for inference after training.

        Args:
            mel_spec: [B, n_mels, T]
            lengths: [B] optional lengths
            layer: Which layer to extract from (-1 = final)

        Returns:
            features: [B, T', encoder_dim]
            feature_lengths: [B] or None
        """
        with torch.no_grad():
            result = self.forward(
                mel_spec,
                lengths=lengths,
                grl_alpha=0.0,  # No gradient reversal needed for inference
                return_all_hiddens=(layer != -1),
            )

        if layer == -1:
            features = result["features"]
        else:
            features = result["all_hiddens"][layer]

        return features, result["feature_lengths"]

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# CTC vocabulary utilities
class CTCVocab:
    """Character-level vocabulary for CTC ASR."""

    # Standard character set for English ASR
    CHARS = " 'abcdefghijklmnopqrstuvwxyz"
    BLANK = "<blank>"
    UNK = "<unk>"

    def __init__(self, chars: str = None):
        chars = chars or self.CHARS
        self.blank_idx = 0
        self.unk_idx = 1
        self.chars = chars

        # Build vocab: [blank, unk, ...chars...]
        self.idx_to_char = [self.BLANK, self.UNK] + list(chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.idx_to_char)}
        self.vocab_size = len(self.idx_to_char)

    def encode(self, text: str) -> list:
        """Convert text to token indices."""
        text = text.lower()
        return [self.char_to_idx.get(c, self.unk_idx) for c in text]

    def decode(self, indices: list, remove_blanks: bool = True, collapse_repeats: bool = True) -> str:
        """
        Convert token indices back to text.

        Args:
            indices: List of token indices
            remove_blanks: Remove blank tokens
            collapse_repeats: Collapse repeated characters (CTC decoding)
        """
        if collapse_repeats:
            # CTC collapse: remove consecutive duplicates
            collapsed = []
            prev = None
            for idx in indices:
                if idx != prev:
                    collapsed.append(idx)
                    prev = idx
            indices = collapsed

        chars = []
        for idx in indices:
            if remove_blanks and idx == self.blank_idx:
                continue
            if idx == self.unk_idx:
                chars.append('?')
            elif idx < len(self.idx_to_char):
                char = self.idx_to_char[idx]
                if char not in [self.BLANK, self.UNK]:
                    chars.append(char)

        return ''.join(chars)

    def ctc_decode_greedy(self, logits: torch.Tensor) -> list:
        """
        Greedy CTC decoding.

        Args:
            logits: [T, vocab_size] or [B, T, vocab_size]

        Returns:
            List of decoded strings
        """
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)

        # Greedy: take argmax at each timestep
        predictions = logits.argmax(dim=-1)  # [B, T]

        decoded = []
        for pred in predictions:
            text = self.decode(pred.tolist())
            decoded.append(text)

        return decoded

    def build_ctc_decoder(self, kenlm_model_path: Optional[str] = None, alpha: float = 0.5, beta: float = 1.0):
        """
        Build a pyctcdecode decoder with optional LM support.

        Args:
            kenlm_model_path: Path to KenLM .arpa or .bin file (optional)
            alpha: LM weight (higher = more LM influence)
            beta: Word insertion bonus (higher = longer words)

        Returns:
            pyctcdecode decoder or None if pyctcdecode not available
        """
        import os

        try:
            from pyctcdecode import build_ctcdecoder
        except ImportError:
            print("pyctcdecode not installed. Install with: pip install pyctcdecode")
            return None

        # Check if LM file exists
        if kenlm_model_path and not os.path.exists(kenlm_model_path):
            print(f"WARNING: KenLM model not found at {kenlm_model_path}")
            print("  CTC decoder will be built WITHOUT language model")
            print("  Download from: https://www.openslr.org/11/ (e.g., 4-gram.arpa.gz)")
            kenlm_model_path = None

        # Build labels list in vocab order (pyctcdecode expects this)
        labels = self.idx_to_char.copy()

        # pyctcdecode expects "" for blank token
        labels[self.blank_idx] = ""

        # Extract unigrams from ARPA file for word-level LM
        unigrams = None
        if kenlm_model_path and kenlm_model_path.endswith('.arpa'):
            print(f"Loading LM from {kenlm_model_path}...")
            unigrams = self._extract_unigrams_from_arpa(kenlm_model_path)

        decoder = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=kenlm_model_path,
            unigrams=unigrams,
            alpha=alpha,
            beta=beta,
        )

        return decoder

    def _extract_unigrams_from_arpa(self, arpa_path: str) -> list:
        """Extract unigrams from ARPA file for pyctcdecode.

        Note: We extract ALL words, not just those matching our char vocab.
        pyctcdecode handles the character-to-word mapping internally.
        """
        unigrams = []
        in_unigrams = False

        try:
            with open(arpa_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()

                    if line == '\\1-grams:':
                        in_unigrams = True
                        continue
                    elif line.startswith('\\') and in_unigrams:
                        break  # Done with unigrams

                    if in_unigrams and line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            word = parts[1]
                            # Skip special tokens
                            if word and word not in ('<s>', '</s>', '<unk>', '<UNK>'):
                                unigrams.append(word.lower())

            print(f"  Extracted {len(unigrams):,} unigrams from ARPA")
        except Exception as e:
            print(f"Warning: Failed to extract unigrams from ARPA: {e}")
            return None

        return unigrams if unigrams else None

    def ctc_decode_beam(
        self,
        logits: torch.Tensor,
        decoder=None,
        beam_width: int = 100,
    ) -> list:
        """
        Beam search CTC decoding with optional LM.

        Args:
            logits: [T, vocab_size] or [B, T, vocab_size]
            decoder: pyctcdecode decoder (from build_ctc_decoder)
            beam_width: Beam width for search

        Returns:
            List of decoded strings
        """
        if decoder is None:
            # Fall back to greedy if no decoder
            return self.ctc_decode_greedy(logits)

        import numpy as np

        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()

        if logits.ndim == 2:
            logits = logits[np.newaxis, ...]

        # Convert to log probabilities (pyctcdecode expects log probs)
        # Softmax then log
        logits_max = logits.max(axis=-1, keepdims=True)
        logits_stable = logits - logits_max
        probs = np.exp(logits_stable) / np.exp(logits_stable).sum(axis=-1, keepdims=True)
        log_probs = np.log(probs + 1e-10)

        decoded = []
        for log_prob in log_probs:
            text = decoder.decode(log_prob, beam_width=beam_width)
            decoded.append(text)

        return decoded


def create_gubert(
    config: str = "small",
    num_speakers: int = 992,
    vocab_size: int = 30,  # 26 letters + space + ' + blank + unk
    **kwargs,
) -> GuBERTEncoder:
    """
    Convenience function to create a GuBERT model.

    Args:
        config: Model size ("tiny", "small", "medium", "large")
        num_speakers: Number of speakers for GRL classifier
        vocab_size: CTC vocabulary size
        **kwargs: Additional config overrides

    Returns:
        GuBERTEncoder model
    """
    return GuBERTEncoder.from_config(
        config,
        num_speakers=num_speakers,
        vocab_size=vocab_size,
        **kwargs,
    )
