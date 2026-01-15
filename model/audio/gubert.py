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


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block with pre-norm and optional head dropout."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        head_drop_prob: float = 0.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

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

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            attn_mask: [T, T] attention mask
            key_padding_mask: [B, T] True for padded positions
        Returns:
            [B, T, D]
        """
        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.head_dropout(x)  # DropHead: randomly drop entire attention heads
        x = self.dropout(x)
        x = residual + x

        # Pre-norm feedforward
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x

        return x


class SpeakerClassifier(nn.Module):
    """
    Speaker classifier head with pooling strategies.
    Used with gradient reversal to push speaker info out of features.
    """

    def __init__(
        self,
        d_model: int,
        num_speakers: int,
        hidden_dim: Optional[int] = None,
        pooling: str = "mean",  # "mean", "attention", "max"
        dropout: float = 0.1,
    ):
        super().__init__()

        hidden_dim = hidden_dim or d_model
        self.pooling = pooling

        if pooling == "attention":
            self.attn_pool = nn.Sequential(
                nn.Linear(d_model, 1),
            )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_speakers),
        )

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
                # Masked mean pooling
                mask_expanded = mask.unsqueeze(-1).float()  # [B, T, 1]
                pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled = x.mean(dim=1)

        elif self.pooling == "max":
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            pooled = x.max(dim=1)[0]

        elif self.pooling == "attention":
            attn_weights = self.attn_pool(x).squeeze(-1)  # [B, T]
            if mask is not None:
                attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1)  # [B, T]
            pooled = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]

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

    def __post_init__(self):
        if self.conv_kernel_sizes is None:
            self.conv_kernel_sizes = [5, 3, 3]
        if self.conv_strides is None:
            self.conv_strides = [2, 2, 1]  # 4x downsampling

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
    # ~2M params - very fast, good for experimentation
    "tiny": GuBERTConfig(
        encoder_dim=128,
        num_layers=3,
        num_heads=4,
        ff_dim=512,
        dropout=0.1,
    ),
    # ~5M params - balanced speed/quality
    "small": GuBERTConfig(
        encoder_dim=256,
        num_layers=4,
        num_heads=4,
        ff_dim=1024,
        dropout=0.1,
    ),
    # ~15M params - good quality
    "medium": GuBERTConfig(
        encoder_dim=384,
        num_layers=6,
        num_heads=6,
        ff_dim=1536,
        dropout=0.1,
    ),
    # ~30M params - high quality
    "large": GuBERTConfig(
        encoder_dim=512,
        num_layers=8,
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

        # Convolutional subsampling frontend
        self.conv_subsample = ConvSubsampling(
            in_channels=config.n_mels,
            out_channels=config.encoder_dim,
            kernel_sizes=config.conv_kernel_sizes,
            strides=config.conv_strides,
            dropout=config.dropout,
        )

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model=config.encoder_dim,
            max_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=config.encoder_dim,
                n_heads=config.num_heads,
                d_ff=config.ff_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.encoder_dim)

        # ASR head (CTC)
        self.asr_head = nn.Linear(config.encoder_dim, config.vocab_size)

        # Speaker classifier (for GRL)
        self.speaker_classifier = SpeakerClassifier(
            d_model=config.encoder_dim,
            num_speakers=config.num_speakers,
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
                - features: [B, T', encoder_dim] content features
                - asr_logits: [B, T', vocab_size] for CTC loss
                - speaker_logits: [B, num_speakers] for GRL loss
                - feature_lengths: [B] output sequence lengths
                - all_hiddens: list of [B, T', D] if return_all_hiddens=True
        """
        batch_size = mel_spec.size(0)

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

        # Positional encoding
        x = self.pos_enc(x)

        # Transformer encoder
        all_hiddens = [x] if return_all_hiddens else None
        for block in self.encoder_blocks:
            x = block(x, key_padding_mask=padding_mask)
            if return_all_hiddens:
                all_hiddens.append(x)

        # Final normalization
        features = self.final_norm(x)

        # ASR head
        asr_logits = self.asr_head(features)

        # Speaker classifier with gradient reversal
        # Use ~padding_mask to get valid positions mask
        valid_mask = ~padding_mask if padding_mask is not None else None
        reversed_features = GradientReversalFunction.apply(features, grl_alpha)
        speaker_logits = self.speaker_classifier(reversed_features, mask=valid_mask)

        result = {
            "features": features,
            "asr_logits": asr_logits,
            "speaker_logits": speaker_logits,
            "feature_lengths": feature_lengths,
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


# =============================================================================
# Masked Prediction (HuBERT-style) + GRL
# =============================================================================

@dataclass
class MaskedGuBERTConfig:
    """Configuration for MaskedGuBERT model."""
    n_mels: int = 80
    encoder_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ff_dim: int = 1024
    conv_kernel_sizes: list = None
    conv_strides: list = None
    num_speakers: int = 992
    dropout: float = 0.1
    conv_dropout: float = 0.05     # Dropout1d in conv frontend (drops entire channels)
    feature_dropout: float = 0.0   # Dropout on features before prediction head
    head_dropout: float = 0.0      # Dropout inside prediction head
    attention_head_drop: float = 0.0  # DropHead: probability of dropping entire attention heads
    max_seq_len: int = 5000
    speaker_pooling: str = "mean"

    # Masked prediction mode
    use_vq: bool = False            # If False, use regression; if True, use VQ targets

    # Masking parameters
    mask_prob: float = 0.08         # Probability of starting a mask span
    mask_length: int = 10           # Length of each mask span
    min_masks: int = 2              # Minimum number of mask spans per sequence

    # Regression-specific (used when use_vq=False)
    regression_loss_type: str = "smooth_l1"  # "smooth_l1", "mse", or "huber"
    target_layer_norm: bool = True  # Normalize targets (data2vec style)

    # VQ-specific (used when use_vq=True)
    num_codebooks: int = 2          # Number of parallel codebooks (product quantization)
    codebook_size: int = 320        # Entries per codebook
    codebook_dim: int = None        # Dimension per codebook (encoder_dim // num_codebooks if None)
    codebook_temp: float = 1.0      # Temperature for codebook softmax (Gumbel-softmax)
    commitment_weight: float = 0.25 # VQ commitment loss weight

    # Feature variance regularization (for VAE-friendly features)
    use_variance_reg: bool = False      # Enable variance regularization
    temporal_var_weight: float = 0.01   # Weight for temporal variance loss
    temporal_var_min: float = 0.1       # Minimum temporal variance threshold
    dim_var_weight: float = 0.01        # Weight for dimension variance loss
    dim_var_min: float = 0.1            # Minimum dimension variance threshold
    # Temporal smoothness penalty (penalize when adjacent frames too similar)
    temporal_smoothness_weight: float = 0.1   # Weight for smoothness penalty
    temporal_smoothness_max: float = 0.95     # Max allowed smoothness (penalize above this)

    def __post_init__(self):
        if self.conv_kernel_sizes is None:
            self.conv_kernel_sizes = [5, 3, 3]
        if self.conv_strides is None:
            self.conv_strides = [2, 2, 1]  # 4x downsampling
        if self.codebook_dim is None:
            self.codebook_dim = self.encoder_dim // self.num_codebooks

    def to_dict(self) -> dict:
        """Convert config to dictionary (for HuggingFace compatibility)."""
        import dataclasses
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Convert config to JSON string (for HuggingFace compatibility)."""
        import json
        return json.dumps(self.to_dict(), indent=2)


# Predefined configurations for MaskedGuBERT
# By default, use regression (use_vq=False) for continuous representations
MASKED_GUBERT_CONFIGS = {
    "tiny": MaskedGuBERTConfig(
        encoder_dim=128,
        num_layers=3,
        num_heads=4,
        ff_dim=512,
        use_vq=False,  # Regression by default
        dropout=0.1,
    ),
    # guber is ~4.8M params, GRL is ~432k params
    "small": MaskedGuBERTConfig(
        encoder_dim=288,
        num_layers=4,
        num_heads=4,
        ff_dim=1152,
        use_vq=False,  # Regression by default
        dropout=0.1,
    ),
    # guber is ~6.8M params, GRL is ~432k params
    "small_deep": MaskedGuBERTConfig(
        encoder_dim=288,
        num_layers=6,
        num_heads=6,
        ff_dim=1152,
        use_vq=False,  # Regression by default
        dropout=0.1,
    ),
    # gubert is ~12M params, GRL is ~650k params
    "medium": MaskedGuBERTConfig(
        encoder_dim=384,
        num_layers=6,
        num_heads=6,
        ff_dim=1536,
        use_vq=False,  # Regression by default
        dropout=0.1,
    ),
    "large": MaskedGuBERTConfig(
        encoder_dim=512,
        num_layers=8,
        num_heads=8,
        ff_dim=2048,
        use_vq=False,  # Regression by default
        dropout=0.1,
    ),
}


class VectorQuantizer(nn.Module):
    """
    Online Vector Quantizer with learnable codebook.

    Uses straight-through estimator for gradients (forward uses hard assignment,
    backward uses soft gradients through the codebook).

    Supports product quantization with multiple codebooks for efficiency.
    """

    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        codebook_dim: int,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
        use_ema: bool = False,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_weight = commitment_weight
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # Learnable codebook embeddings: [num_codebooks, codebook_size, codebook_dim]
        self.codebook = nn.Parameter(
            torch.randn(num_codebooks, codebook_size, codebook_dim) * 0.02
        )

        if use_ema:
            # EMA cluster sizes and embeddings for codebook update
            self.register_buffer('ema_cluster_size', torch.zeros(num_codebooks, codebook_size))
            self.register_buffer('ema_embed_sum', torch.zeros(num_codebooks, codebook_size, codebook_dim))

    def forward(
        self,
        x: torch.Tensor,
        return_indices: bool = True,
    ) -> dict:
        """
        Quantize input features.

        Args:
            x: [B, T, D] where D = num_codebooks * codebook_dim
            return_indices: Whether to return codebook indices

        Returns:
            dict with:
                - quantized: [B, T, D] quantized features
                - indices: [B, T, num_codebooks] codebook indices
                - commitment_loss: scalar commitment loss
                - codebook_loss: scalar codebook loss (if not using EMA)
        """
        B, T, D = x.shape

        # Split into codebook groups: [B, T, num_codebooks, codebook_dim]
        x_split = x.view(B, T, self.num_codebooks, self.codebook_dim)

        # Compute distances to codebook entries
        # x_split: [B, T, G, d], codebook: [G, K, d]
        # distances: [B, T, G, K]
        x_flat = x_split.reshape(-1, self.num_codebooks, self.codebook_dim)  # [B*T, G, d]

        distances = torch.zeros(B * T, self.num_codebooks, self.codebook_size, device=x.device)
        for g in range(self.num_codebooks):
            # [B*T, d] vs [K, d] -> [B*T, K]
            distances[:, g, :] = torch.cdist(
                x_flat[:, g, :].unsqueeze(1),  # [B*T, 1, d]
                self.codebook[g].unsqueeze(0),  # [1, K, d]
            ).squeeze(1)

        distances = distances.view(B, T, self.num_codebooks, self.codebook_size)

        # Hard assignment (argmin)
        indices = distances.argmin(dim=-1)  # [B, T, G]

        # Gather quantized vectors
        # indices: [B, T, G] -> one-hot: [B, T, G, K]
        one_hot = F.one_hot(indices, self.codebook_size).float()  # [B, T, G, K]

        # quantized: [B, T, G, d]
        quantized_split = torch.einsum('btgk,gkd->btgd', one_hot, self.codebook)

        # Straight-through estimator: use quantized in forward, but gradient flows through x
        quantized_split = x_split + (quantized_split - x_split).detach()

        # Reshape back: [B, T, D]
        quantized = quantized_split.view(B, T, D)

        # Commitment loss: encourage encoder output to stay close to codebook
        commitment_loss = F.mse_loss(x_split, quantized_split.detach())

        # Codebook loss: encourage codebook to stay close to encoder output
        codebook_loss = F.mse_loss(quantized_split, x_split.detach())

        result = {
            "quantized": quantized,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
        }

        if return_indices:
            result["indices"] = indices

        return result

    def get_codebook_usage(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Compute codebook usage statistics.

        Args:
            indices: [B, T, num_codebooks]

        Returns:
            usage: [num_codebooks, codebook_size] usage counts
        """
        usage = torch.zeros(self.num_codebooks, self.codebook_size, device=indices.device)
        for g in range(self.num_codebooks):
            usage[g] = torch.bincount(
                indices[:, :, g].reshape(-1),
                minlength=self.codebook_size
            ).float()
        return usage


def generate_mask_spans(
    seq_len: int,
    batch_size: int,
    mask_prob: float = 0.08,
    mask_length: int = 10,
    min_masks: int = 2,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Generate span masks for masked audio prediction (HuBERT/wav2vec2 style).

    Args:
        seq_len: Sequence length
        batch_size: Batch size
        mask_prob: Probability of starting a mask span at each position
        mask_length: Length of each mask span
        min_masks: Minimum number of mask spans per sequence
        device: Device for output tensor

    Returns:
        mask: [B, T] boolean tensor, True = masked
    """
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    for b in range(batch_size):
        # Calculate number of mask spans
        num_masks = max(min_masks, int(seq_len * mask_prob))

        # Ensure we don't mask more than 80% of sequence
        max_masked = int(seq_len * 0.8)
        num_masks = min(num_masks, max_masked // mask_length)

        if num_masks == 0:
            continue

        # Randomly select span start positions (avoid overlapping with end)
        valid_starts = seq_len - mask_length
        if valid_starts <= 0:
            # Sequence too short, mask everything
            mask[b, :] = True
            continue

        # Sample start positions
        starts = torch.randint(0, valid_starts, (num_masks,), device=device)

        # Create spans
        for start in starts:
            end = min(start + mask_length, seq_len)
            mask[b, start:end] = True

    return mask


class MaskedGuBERTEncoder(nn.Module):
    """
    MaskedGuBERT: Speaker-Invariant Speech Encoder with Masked Prediction

    Trained with dual objectives:
    1. Masked prediction (HuBERT-style) - forces encoding of acoustic structure
    2. GRL speaker classification - removes speaker information

    Unlike CTC-based GuBERT which requires transcriptions, this version is
    self-supervised and can be trained on any audio data.

    The resulting features capture phonetic content + prosody while being
    speaker-agnostic - ideal for TTS and voice conversion.
    """

    def __init__(self, config: MaskedGuBERTConfig):
        super().__init__()
        self.config = config

        # Convolutional subsampling frontend with channel dropout (Dropout1d)
        self.conv_subsample = ConvSubsampling(
            in_channels=config.n_mels,
            out_channels=config.encoder_dim,
            kernel_sizes=config.conv_kernel_sizes,
            strides=config.conv_strides,
            dropout=config.conv_dropout,  # Uses Dropout1d internally
        )

        # Learned mask embedding (replaces masked positions)
        self.mask_embedding = nn.Parameter(torch.randn(config.encoder_dim) * 0.02)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model=config.encoder_dim,
            max_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # Transformer encoder blocks with optional DropHead
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=config.encoder_dim,
                n_heads=config.num_heads,
                d_ff=config.ff_dim,
                dropout=config.dropout,
                head_drop_prob=config.attention_head_drop,
            )
            for _ in range(config.num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.encoder_dim)

        # Feature dropout: applied to features before prediction head
        # Helps prevent memorization by forcing prediction head to be robust
        self.feature_dropout = nn.Dropout(config.feature_dropout) if config.feature_dropout > 0 else nn.Identity()

        # Mode-specific components
        if config.use_vq:
            # VQ mode: discrete codebook targets
            self.quantizer = VectorQuantizer(
                num_codebooks=config.num_codebooks,
                codebook_size=config.codebook_size,
                codebook_dim=config.codebook_dim,
                commitment_weight=config.commitment_weight,
            )

            # Projection to quantizer space (if dimensions don't match)
            if config.encoder_dim != config.num_codebooks * config.codebook_dim:
                self.pre_quantize_proj = nn.Linear(
                    config.encoder_dim,
                    config.num_codebooks * config.codebook_dim,
                )
            else:
                self.pre_quantize_proj = nn.Identity()

            # Prediction head: predicts codebook indices for masked positions
            # head_dropout applied after GELU activation
            head_dropout_layer = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()
            self.prediction_head = nn.Sequential(
                nn.Linear(config.encoder_dim, config.encoder_dim),
                nn.GELU(),
                head_dropout_layer,
                nn.LayerNorm(config.encoder_dim),
                nn.Linear(config.encoder_dim, config.num_codebooks * config.codebook_size),
            )
        else:
            # Regression mode: continuous normalized targets (data2vec style)
            self.quantizer = None
            self.pre_quantize_proj = None

            # Prediction head: predicts continuous features for masked positions
            # head_dropout applied after GELU activation
            head_dropout_layer = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()
            self.prediction_head = nn.Sequential(
                nn.Linear(config.encoder_dim, config.encoder_dim),
                nn.GELU(),
                head_dropout_layer,
                nn.LayerNorm(config.encoder_dim),
                nn.Linear(config.encoder_dim, config.encoder_dim),
            )

        # Speaker classifier (for GRL)
        self.speaker_classifier = SpeakerClassifier(
            d_model=config.encoder_dim,
            num_speakers=config.num_speakers,
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
    def from_config(cls, config_name: str, **overrides) -> "MaskedGuBERTEncoder":
        """
        Create model from predefined config with optional overrides.

        Args:
            config_name: One of "tiny", "small", "medium", "large"
            **overrides: Override any config parameter

        Example:
            model = MaskedGuBERTEncoder.from_config("small", num_speakers=500)
        """
        if config_name not in MASKED_GUBERT_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(MASKED_GUBERT_CONFIGS.keys())}")

        config = MASKED_GUBERT_CONFIGS[config_name]
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = MaskedGuBERTConfig(**config_dict)

        return cls(config)

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input mel spectrogram length."""
        return self.conv_subsample.get_output_length(input_length)

    def _generate_targets_vq(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate VQ targets from unmasked (teacher) features.

        Uses the features BEFORE masking to create targets, ensuring the model
        must predict the original content from context.

        Args:
            features: [B, T, D] features before masking

        Returns:
            target_indices: [B, T, num_codebooks] quantization targets
            quantizer_output: dict with quantizer losses
        """
        # Project to quantizer space
        features_proj = self.pre_quantize_proj(features)

        # Quantize to get targets
        quantizer_output = self.quantizer(features_proj, return_indices=True)

        return quantizer_output["indices"], quantizer_output

    def _generate_targets_regression(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate regression targets from unmasked (teacher) features.

        Uses layer normalization on detached features (data2vec style) to create
        stable targets that don't collapse during training.

        Args:
            features: [B, T, D] features before masking

        Returns:
            targets: [B, T, D] normalized continuous targets
        """
        targets = features.detach()

        # Apply layer normalization for stable targets (data2vec insight)
        if self.config.target_layer_norm:
            targets = F.layer_norm(targets, [targets.shape[-1]])

        return targets

    def forward(
        self,
        mel_spec: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        grl_alpha: float = 1.0,
        return_all_hiddens: bool = False,
    ) -> dict:
        """
        Forward pass through MaskedGuBERT encoder.

        Args:
            mel_spec: [B, n_mels, T] mel spectrogram
            lengths: [B] original lengths before padding (optional)
            mask: [B, T'] mask for masked prediction (True = masked)
                  If None, generates mask automatically during training
            grl_alpha: Gradient reversal strength (0=no reversal, 1=full reversal)
            return_all_hiddens: If True, return hidden states from all layers

        Returns:
            dict with keys (common):
                - features: [B, T', encoder_dim] content features
                - speaker_logits: [B, num_speakers] for GRL loss
                - feature_lengths: [B] output sequence lengths
                - mask: [B, T'] the mask used
                - all_hiddens: list of [B, T', D] if return_all_hiddens=True

            VQ mode (use_vq=True):
                - prediction_logits: [B, T', num_codebooks, codebook_size] for masked prediction
                - target_indices: [B, T', num_codebooks] quantization targets
                - commitment_loss: scalar VQ commitment loss
                - codebook_loss: scalar VQ codebook loss

            Regression mode (use_vq=False):
                - predictions: [B, T', encoder_dim] predicted features
                - targets: [B, T', encoder_dim] normalized target features
        """
        batch_size = mel_spec.size(0)

        # Convolutional frontend
        x = self.conv_subsample(mel_spec)  # [B, encoder_dim, T']
        x = x.permute(0, 2, 1)  # [B, T', encoder_dim]

        seq_len = x.size(1)

        # Calculate output lengths and padding mask
        feature_lengths = None
        padding_mask = None
        if lengths is not None:
            feature_lengths = torch.tensor(
                [self.get_output_length(l.item()) for l in lengths],
                device=mel_spec.device,
                dtype=torch.long,
            )
            max_len = x.size(1)
            padding_mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= feature_lengths.unsqueeze(1)

        # Generate targets BEFORE masking (teacher features)
        if self.config.use_vq:
            target_indices, quantizer_output = self._generate_targets_vq(x.detach())
        else:
            targets = self._generate_targets_regression(x)

        # Generate mask if not provided
        if mask is None and self.training:
            mask = generate_mask_spans(
                seq_len=seq_len,
                batch_size=batch_size,
                mask_prob=self.config.mask_prob,
                mask_length=self.config.mask_length,
                min_masks=self.config.min_masks,
                device=x.device,
            )
            # Don't mask padding
            if padding_mask is not None:
                mask = mask & ~padding_mask

        # Apply masking: replace masked positions with learned embedding
        if mask is not None and mask.any():
            x = x.masked_fill(mask.unsqueeze(-1), 0) + \
                self.mask_embedding.unsqueeze(0).unsqueeze(0) * mask.unsqueeze(-1).float()

        # Positional encoding
        x = self.pos_enc(x)

        # Transformer encoder
        all_hiddens = [x] if return_all_hiddens else None
        for block in self.encoder_blocks:
            x = block(x, key_padding_mask=padding_mask)
            if return_all_hiddens:
                all_hiddens.append(x)

        # Store pre-norm features for VAE-friendly extraction
        features_unnorm = x

        # Final normalization
        features = self.final_norm(x)

        # Speaker classifier with gradient reversal
        valid_mask = ~padding_mask if padding_mask is not None else None
        reversed_features = GradientReversalFunction.apply(features, grl_alpha)
        speaker_logits = self.speaker_classifier(reversed_features, mask=valid_mask)

        # Compute feature variance regularization (for VAE-friendly features)
        variance_loss = torch.tensor(0.0, device=features.device)
        if self.config.use_variance_reg and self.training:
            # Use unnormalized features for variance computation (LayerNorm constrains variance)
            feat_for_var = features_unnorm

            # Mask out padding for variance computation
            if valid_mask is not None:
                # valid_mask: [B, T] where True = valid position
                feat_masked = feat_for_var * valid_mask.unsqueeze(-1).float()
                # Count valid positions per sample
                valid_counts = valid_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
            else:
                feat_masked = feat_for_var
                valid_counts = torch.tensor(feat_for_var.shape[1], device=feat_for_var.device, dtype=torch.float32)

            # Temporal variance: variance across time dimension (should be high)
            # Compute per-sample temporal mean then variance
            if valid_mask is not None:
                temporal_mean = feat_masked.sum(dim=1) / valid_counts  # [B, D]
                temporal_var = ((feat_masked - temporal_mean.unsqueeze(1)) ** 2 * valid_mask.unsqueeze(-1).float()).sum(dim=1) / valid_counts
                temporal_var = temporal_var.mean()  # Average across batch and dimensions
            else:
                temporal_var = feat_for_var.var(dim=1).mean()

            # Penalize if temporal variance is below threshold
            temporal_loss = F.relu(self.config.temporal_var_min - temporal_var)

            # Dimension variance: variance across dimensions (should utilize all dimensions)
            # Flatten batch and time, then compute variance per dimension
            if valid_mask is not None:
                # Gather only valid positions
                valid_features = feat_for_var[valid_mask]  # [N_valid, D]
                dim_var = valid_features.var(dim=0).mean() if valid_features.shape[0] > 1 else torch.tensor(0.0, device=features.device)
            else:
                dim_var = feat_for_var.reshape(-1, feat_for_var.shape[-1]).var(dim=0).mean()

            # Penalize if dimension variance is below threshold
            dim_loss = F.relu(self.config.dim_var_min - dim_var)

            # Temporal smoothness penalty: compute WITHOUT dropout to get true smoothness
            # This prevents the model from hiding behind dropout noise during training
            # We re-forward through transformer with model in eval mode (disables all dropout)
            was_training = self.training
            self.eval()  # Disable all dropout, batchnorm stats, etc.

            # Re-forward through transformer to get clean features (with gradients)
            # Note: even in eval mode, gradients still flow (no torch.no_grad)
            x_clean = self.conv_subsample(mel_spec)  # [B, encoder_dim, T']
            x_clean = x_clean.permute(0, 2, 1)  # [B, T', encoder_dim]
            x_clean = self.pos_enc(x_clean)
            for block in self.encoder_blocks:
                x_clean = block(x_clean, key_padding_mask=padding_mask)

            # Restore training mode
            if was_training:
                self.train()

            # Compute temporal smoothness on clean (no-dropout) features
            feat_for_smooth = x_clean  # Pre-LayerNorm, no dropout
            feat_t = feat_for_smooth[:, 1:, :]  # [B, T-1, D]
            feat_t_minus_1 = feat_for_smooth[:, :-1, :]  # [B, T-1, D]
            temporal_smoothness = F.cosine_similarity(feat_t, feat_t_minus_1, dim=-1)  # [B, T-1]

            # Mask out padding positions for smoothness calculation
            if valid_mask is not None:
                # valid_mask[:, 1:] AND valid_mask[:, :-1] = both positions valid
                smooth_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
                temporal_smoothness = (temporal_smoothness * smooth_mask.float()).sum() / smooth_mask.sum().clamp(min=1)
            else:
                temporal_smoothness = temporal_smoothness.mean()

            # Penalize if smoothness exceeds max threshold
            smoothness_loss = F.relu(temporal_smoothness - self.config.temporal_smoothness_max)

            variance_loss = (
                self.config.temporal_var_weight * temporal_loss +
                self.config.dim_var_weight * dim_loss +
                self.config.temporal_smoothness_weight * smoothness_loss
            )

        # Build result dictionary
        result = {
            "features": features,
            "features_unnorm": features_unnorm,  # Pre-LayerNorm features for VAE extraction
            "speaker_logits": speaker_logits,
            "feature_lengths": feature_lengths,
            "mask": mask,
            "variance_loss": variance_loss,
            # Metrics for logging (only present when variance_reg is enabled)
            "temporal_smoothness": temporal_smoothness if self.config.use_variance_reg and self.training else None,
        }

        # Apply feature dropout before prediction head (helps prevent memorization)
        features_for_pred = self.feature_dropout(features)

        if self.config.use_vq:
            # VQ mode: return logits and indices for cross-entropy
            prediction_logits = self.prediction_head(features_for_pred)  # [B, T', num_codebooks * codebook_size]
            prediction_logits = prediction_logits.view(
                batch_size, seq_len, self.config.num_codebooks, self.config.codebook_size
            )
            result["prediction_logits"] = prediction_logits
            result["target_indices"] = target_indices
            result["commitment_loss"] = quantizer_output["commitment_loss"]
            result["codebook_loss"] = quantizer_output["codebook_loss"]
        else:
            # Regression mode: return continuous predictions and targets
            predictions = self.prediction_head(features_for_pred)  # [B, T', encoder_dim]
            result["predictions"] = predictions
            result["targets"] = targets

        if return_all_hiddens:
            result["all_hiddens"] = all_hiddens

        return result

    def extract_features(
        self,
        mel_spec: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        layer: int = -1,
        normalize: bool = True,
        layers: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract features without masking or computing prediction heads.
        Useful for inference after training.

        Args:
            mel_spec: [B, n_mels, T]
            lengths: [B] optional lengths
            layer: Which layer to extract from (-1 = final). Ignored if `layers` is provided.
            normalize: If True, apply final LayerNorm. If False, return raw pre-norm features
                       (more dynamic range, better for VAE training).
            layers: Optional list of layer indices for multi-scale features.
                    If provided, concatenates features from specified layers.
                    Example: [-1, -3, -5] extracts from last, third-to-last, fifth-to-last layers.
                    Each intermediate layer is NOT normalized; only final layer uses normalize flag.

        Returns:
            features: [B, T', encoder_dim] or [B, T', encoder_dim * len(layers)] if multi-scale
            feature_lengths: [B] or None
        """
        was_training = self.training
        self.eval()  # Temporarily set to eval mode to skip mask generation

        # Determine if we need all hidden states
        need_all_hiddens = (layer != -1) or (layers is not None)

        with torch.no_grad():
            # Forward without mask (mask=None + eval mode = no masking)
            result = self.forward(
                mel_spec,
                lengths=lengths,
                mask=None,  # No mask for feature extraction
                grl_alpha=0.0,  # No gradient reversal for inference
                return_all_hiddens=need_all_hiddens,
            )

        if was_training:
            self.train()  # Restore training mode

        # Multi-scale feature extraction
        if layers is not None:
            all_hiddens = result["all_hiddens"]  # List of [B, T', D] from each layer
            num_layers = len(all_hiddens)

            multi_scale_features = []
            for l in layers:
                # Convert negative indices to positive
                layer_idx = l if l >= 0 else num_layers + l
                layer_idx = max(0, min(layer_idx, num_layers - 1))

                layer_features = all_hiddens[layer_idx]

                # Only apply normalization to the final layer if requested
                if l == -1 and normalize:
                    layer_features = self.final_norm(layer_features)

                multi_scale_features.append(layer_features)

            # Concatenate along feature dimension: [B, T', D * num_layers]
            features = torch.cat(multi_scale_features, dim=-1)

        elif layer == -1:
            # Use normalized or unnormalized features based on flag
            if normalize:
                features = result["features"]  # Already normalized in forward()
            else:
                features = result["features_unnorm"]  # Pre-LayerNorm features
        else:
            features = result["all_hiddens"][layer]

        return features, result["feature_lengths"]

    def compute_masked_prediction_loss(
        self,
        forward_output: dict,
    ) -> torch.Tensor:
        """
        Compute the masked prediction loss from forward output.

        For VQ mode: cross-entropy over codebook indices
        For regression mode: smooth L1 / MSE / Huber over continuous features

        Args:
            forward_output: Output dict from forward() call

        Returns:
            loss: Scalar loss tensor
        """
        mask = forward_output["mask"]

        if mask is None or not mask.any():
            # No masked positions - return zero loss
            device = forward_output["features"].device
            return torch.tensor(0.0, device=device, requires_grad=True)

        if self.config.use_vq:
            # VQ mode: cross-entropy loss
            prediction_logits = forward_output["prediction_logits"]  # [B, T, G, K]
            target_indices = forward_output["target_indices"]  # [B, T, G]

            # Flatten for cross-entropy: [B*T*G, K] vs [B*T*G]
            B, T, G, K = prediction_logits.shape
            logits_flat = prediction_logits.view(-1, K)  # [B*T*G, K]
            targets_flat = target_indices.view(-1)  # [B*T*G]

            # Expand mask to match codebook dimension: [B, T] -> [B*T*G]
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, G).reshape(-1)

            # Only compute loss on masked positions
            loss = F.cross_entropy(
                logits_flat[mask_expanded],
                targets_flat[mask_expanded],
                reduction="mean",
            )

            # Add VQ losses
            loss = loss + self.config.commitment_weight * forward_output["commitment_loss"]
            loss = loss + forward_output["codebook_loss"]

        else:
            # Regression mode: continuous loss
            predictions = forward_output["predictions"]  # [B, T, D]
            targets = forward_output["targets"]  # [B, T, D]

            # Get masked predictions and targets
            masked_preds = predictions[mask]  # [N_masked, D]
            masked_targets = targets[mask]  # [N_masked, D]

            # Compute loss based on configured type
            if self.config.regression_loss_type == "smooth_l1":
                loss = F.smooth_l1_loss(masked_preds, masked_targets, reduction="mean")
            elif self.config.regression_loss_type == "mse":
                loss = F.mse_loss(masked_preds, masked_targets, reduction="mean")
            elif self.config.regression_loss_type == "huber":
                loss = F.huber_loss(masked_preds, masked_targets, reduction="mean", delta=1.0)
            else:
                raise ValueError(f"Unknown regression loss type: {self.config.regression_loss_type}")

        return loss

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_masked_gubert(
    config: str = "small",
    num_speakers: int = 992,
    **kwargs,
) -> MaskedGuBERTEncoder:
    """
    Convenience function to create a MaskedGuBERT model.

    Args:
        config: Model size ("tiny", "small", "medium", "large")
        num_speakers: Number of speakers for GRL classifier
        **kwargs: Additional config overrides

    Returns:
        MaskedGuBERTEncoder model
    """
    return MaskedGuBERTEncoder.from_config(
        config,
        num_speakers=num_speakers,
        **kwargs,
    )
