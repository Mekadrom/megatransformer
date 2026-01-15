"""
Speech Reconstruction Model

Architecture for reconstructing mel spectrograms from GuBERT features + speaker embeddings.

Components:
1. SpeakerEncoder: mel spec -> speaker embedding (attention-pooled, no structural info)
2. MelReconstructor: GuBERT features + speaker embedding -> mel spec

The attention pooling in SpeakerEncoder enforces that speaker embeddings contain
no temporal/structural information - only speaker identity characteristics.
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SpeakerEncoderConfig:
    """Configuration for SpeakerEncoder."""
    n_mels: int = 80
    encoder_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1
    speaker_dim: int = 256  # Output speaker embedding dimension

    # Conv subsampling
    conv_channels: List[int] = None
    conv_kernel_sizes: List[int] = None
    conv_strides: List[int] = None

    def __post_init__(self):
        if self.conv_channels is None:
            self.conv_channels = [self.encoder_dim // 2, self.encoder_dim]
        if self.conv_kernel_sizes is None:
            self.conv_kernel_sizes = [5, 3]
        if self.conv_strides is None:
            self.conv_strides = [2, 2]  # 4x total downsampling


@dataclass
class MelReconstructorConfig:
    """Configuration for MelReconstructor."""
    gubert_dim: int = 256  # Input GuBERT feature dimension
    speaker_dim: int = 256  # Speaker embedding dimension
    n_mels: int = 80

    # Architecture
    hidden_dim: int = 512
    num_layers: int = 4
    kernel_size: int = 5
    dropout: float = 0.1

    # Upsampling to match mel resolution
    upsample_factors: List[int] = None  # Should match GuBERT's downsampling

    # FiLM conditioning
    film_hidden_dim: int = 256

    def __post_init__(self):
        if self.upsample_factors is None:
            self.upsample_factors = [2, 2]  # 4x upsampling (matches GuBERT's 4x downsampling)


@dataclass
class SpeechReconstructionConfig:
    """Configuration for the complete speech reconstruction model."""
    speaker_encoder: SpeakerEncoderConfig = None
    mel_reconstructor: MelReconstructorConfig = None

    # ArcFace settings (optional, for speaker classification)
    use_arcface: bool = False
    arcface_scale: float = 30.0
    arcface_margin: float = 0.5
    num_speakers: int = 0  # Required if use_arcface=True

    def __post_init__(self):
        if self.speaker_encoder is None:
            self.speaker_encoder = SpeakerEncoderConfig()
        if self.mel_reconstructor is None:
            self.mel_reconstructor = MelReconstructorConfig()


# =============================================================================
# Speaker Encoder Components
# =============================================================================

class ConvSubsampling2D(nn.Module):
    """
    2D convolutional subsampling for mel spectrograms.

    Input: [B, 1, n_mels, T] or [B, n_mels, T]
    Output: [B, T', encoder_dim]
    """
    def __init__(
        self,
        n_mels: int,
        out_channels: int,
        channels: List[int],
        kernel_sizes: List[int],
        strides: List[int],
    ):
        super().__init__()
        self.n_mels = n_mels

        # Build conv layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_channels = 1
        current_mels = n_mels

        for i, (ch, k, s) in enumerate(zip(channels, kernel_sizes, strides)):
            # Use stride only in time dimension, keep mel dimension
            self.convs.append(nn.Conv2d(
                in_channels, ch,
                kernel_size=(3, k),
                stride=(1, s),
                padding=(1, k // 2),
            ))
            self.norms.append(nn.GroupNorm(max(1, ch // 4), ch))
            in_channels = ch

        # Project to encoder dim
        # After conv, shape is [B, channels[-1], n_mels, T']
        # Flatten mel dimension into channels, then project
        self.out_proj = nn.Linear(channels[-1] * n_mels, out_channels)

        # Store strides for length calculation
        self.strides = strides
        self.total_stride = math.prod(strides)

    def get_output_length(self, input_length: int) -> int:
        """Calculate output length given input length."""
        length = input_length
        for stride in self.strides:
            length = (length + stride - 1) // stride
        return length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, n_mels, T] or [B, 1, n_mels, T]
        Returns:
            [B, T', encoder_dim]
        """
        # Ensure 4D input
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, 1, n_mels, T]

        # Apply conv layers
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            x = norm(x)
            x = F.silu(x)

        # x: [B, C, n_mels, T']
        B, C, M, T = x.shape

        # Flatten mel and channel dims, then project
        x = x.permute(0, 3, 1, 2)  # [B, T', C, n_mels]
        x = x.reshape(B, T, C * M)  # [B, T', C * n_mels]
        x = self.out_proj(x)  # [B, T', encoder_dim]

        return x


class TransformerEncoderLayer(nn.Module):
    """Standard transformer encoder layer with pre-norm."""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
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
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.dropout(x)
        x = residual + x

        # Pre-norm FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class AttentionPooling(nn.Module):
    """
    Attention-based pooling to aggregate sequence into single vector.

    This enforces that the output contains no positional/structural information -
    the same set of frames in any order produces the same output.

    Uses a learnable query that attends to all positions.
    """
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Learnable query
        self.query = nn.Parameter(torch.randn(1, 1, d_model))

        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] sequence
            key_padding_mask: [B, T] True where padded
        Returns:
            [B, D] pooled representation
        """
        B = x.shape[0]

        # Expand query for batch
        query = self.query.expand(B, -1, -1)  # [B, 1, D]

        # Attend to sequence
        pooled, _ = self.attn(
            query, x, x,
            key_padding_mask=key_padding_mask,
        )  # [B, 1, D]

        pooled = self.norm(pooled.squeeze(1))  # [B, D]

        return pooled


class SpeakerEncoder(nn.Module):
    """
    Speaker encoder that extracts a single speaker embedding from mel spectrograms.

    Architecture:
    1. Conv subsampling (reduces temporal resolution)
    2. Transformer encoder layers
    3. Attention pooling to single vector (removes structural info)
    4. Output projection + L2 normalization

    The attention pooling enforces that the output is order-invariant,
    meaning it captures speaker characteristics without temporal structure.
    """
    def __init__(self, config: SpeakerEncoderConfig):
        super().__init__()
        self.config = config

        # Conv subsampling
        self.conv_subsample = ConvSubsampling2D(
            n_mels=config.n_mels,
            out_channels=config.encoder_dim,
            channels=config.conv_channels,
            kernel_sizes=config.conv_kernel_sizes,
            strides=config.conv_strides,
        )

        # Positional encoding (sinusoidal)
        self.pos_encoding = self._create_pos_encoding(
            max_len=2000,
            d_model=config.encoder_dim,
        )

        # Transformer layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=config.encoder_dim,
                n_heads=config.num_heads,
                d_ff=config.ff_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        # Final norm before pooling
        self.pre_pool_norm = nn.LayerNorm(config.encoder_dim)

        # Attention pooling
        self.attention_pool = AttentionPooling(
            d_model=config.encoder_dim,
            n_heads=config.num_heads,
        )

        # Output projection to speaker_dim
        self.out_proj = nn.Sequential(
            nn.Linear(config.encoder_dim, config.speaker_dim),
            nn.LayerNorm(config.speaker_dim),
        )

        self._init_weights()

    def _create_pos_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_output_length(self, input_length: int) -> int:
        """Get output sequence length before pooling."""
        return self.conv_subsample.get_output_length(input_length)

    def forward(
        self,
        mel_spec: torch.Tensor,
        lengths: torch.Tensor = None,
        return_sequence: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract speaker embedding from mel spectrogram.

        Args:
            mel_spec: [B, n_mels, T] or [B, 1, n_mels, T] mel spectrogram
            lengths: [B] optional valid lengths
            return_sequence: If True, also return pre-pooling sequence

        Returns:
            dict with:
                - speaker_embedding: [B, speaker_dim] L2-normalized speaker embedding
                - sequence (optional): [B, T', encoder_dim] if return_sequence=True
        """
        # Handle input shape
        if mel_spec.dim() == 4:
            mel_spec = mel_spec.squeeze(1)  # [B, n_mels, T]

        B, _, T = mel_spec.shape

        # Conv subsampling
        x = self.conv_subsample(mel_spec)  # [B, T', encoder_dim]

        # Add positional encoding
        seq_len = x.shape[1]
        x = x + self.pos_encoding[:, :seq_len, :]

        # Compute padding mask for attention
        key_padding_mask = None
        if lengths is not None:
            # Compute downsampled lengths
            ds_lengths = lengths.clone()
            for stride in self.conv_subsample.strides:
                ds_lengths = (ds_lengths + stride - 1) // stride

            # Create mask: True where padded
            key_padding_mask = torch.arange(seq_len, device=x.device).unsqueeze(0) >= ds_lengths.unsqueeze(1)

        # Transformer layers
        for layer in self.encoder_layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        # Pre-pooling norm
        x = self.pre_pool_norm(x)

        # Store sequence if needed
        sequence = x if return_sequence else None

        # Attention pooling
        pooled = self.attention_pool(x, key_padding_mask=key_padding_mask)  # [B, encoder_dim]

        # Output projection
        speaker_emb = self.out_proj(pooled)  # [B, speaker_dim]

        # L2 normalize
        speaker_emb = F.normalize(speaker_emb, p=2, dim=-1)

        result = {"speaker_embedding": speaker_emb}
        if return_sequence:
            result["sequence"] = sequence

        return result

    def get_num_params(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Mel Reconstructor Components
# =============================================================================

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.

    Applies affine transformation conditioned on speaker embedding:
    output = scale * input + shift

    where scale and shift are predicted from the speaker embedding.
    """
    def __init__(
        self,
        feature_dim: int,
        speaker_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(speaker_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim * 2),
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize to near-identity transform
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Initialize final layer to output zeros (identity FiLM)
        final_linear = self.projection[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def forward(
        self,
        x: torch.Tensor,
        speaker_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] or [B, D, T] features
            speaker_emb: [B, speaker_dim]
        Returns:
            Modulated features, same shape as x
        """
        # Predict scale and shift
        film_params = self.projection(speaker_emb)  # [B, D*2]
        scale, shift = film_params.chunk(2, dim=-1)  # [B, D] each

        # Apply bounded transformation
        scale = torch.tanh(scale) * 0.5  # [-0.5, 0.5]
        shift = torch.tanh(shift) * 0.5  # [-0.5, 0.5]

        # Reshape for broadcasting
        if x.dim() == 3:
            if x.shape[1] != scale.shape[1]:
                # x is [B, T, D], need scale as [B, 1, D]
                scale = scale.unsqueeze(1)
                shift = shift.unsqueeze(1)
            else:
                # x is [B, D, T], need scale as [B, D, 1]
                scale = scale.unsqueeze(-1)
                shift = shift.unsqueeze(-1)

        return x * (1 + scale) + shift


class ResidualConvBlock(nn.Module):
    """Residual 1D conv block with FiLM conditioning."""
    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        speaker_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(max(1, channels // 4), channels)
        self.norm2 = nn.GroupNorm(max(1, channels // 4), channels)
        self.dropout = nn.Dropout(dropout)

        # FiLM layer
        self.film = FiLMLayer(channels, speaker_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv2.bias)
        self.conv2.weight.data *= 0.1  # Small init for residual

    def forward(
        self,
        x: torch.Tensor,
        speaker_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
            speaker_emb: [B, speaker_dim]
        """
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x)
        x = self.dropout(x)

        # Apply FiLM conditioning
        x = self.film(x, speaker_emb)

        x = self.conv2(x)
        x = self.norm2(x)

        x = residual + x
        x = F.silu(x)

        return x


class MelReconstructor(nn.Module):
    """
    Reconstructs mel spectrograms from GuBERT features + speaker embedding.

    Architecture:
    1. Initial projection from GuBERT dim to hidden dim
    2. Temporal upsampling to match mel resolution
    3. Multiple residual conv blocks with FiLM conditioning
    4. Output projection to mel channels
    """
    def __init__(self, config: MelReconstructorConfig):
        super().__init__()
        self.config = config

        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.gubert_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )

        # Upsampling layers
        self.upsample_layers = nn.ModuleList()
        for factor in config.upsample_factors:
            self.upsample_layers.append(nn.Sequential(
                nn.Upsample(scale_factor=factor, mode='nearest'),
                nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(max(1, config.hidden_dim // 4), config.hidden_dim),
                nn.SiLU(),
            ))

        # FiLM layers for upsampling stages
        self.upsample_films = nn.ModuleList([
            FiLMLayer(config.hidden_dim, config.speaker_dim, config.film_hidden_dim)
            for _ in config.upsample_factors
        ])

        # Residual conv blocks
        self.conv_blocks = nn.ModuleList([
            ResidualConvBlock(
                channels=config.hidden_dim,
                kernel_size=config.kernel_size,
                speaker_dim=config.speaker_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(config.hidden_dim, config.hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(max(1, config.hidden_dim // 8), config.hidden_dim // 2),
            nn.SiLU(),
            nn.Conv1d(config.hidden_dim // 2, config.n_mels, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_output_length(self, input_length: int) -> int:
        """Calculate output length given input (GuBERT feature) length."""
        length = input_length
        for factor in self.config.upsample_factors:
            length = length * factor
        return length

    def forward(
        self,
        gubert_features: torch.Tensor,
        speaker_embedding: torch.Tensor,
        target_length: int = None,
    ) -> torch.Tensor:
        """
        Reconstruct mel spectrogram from GuBERT features and speaker embedding.

        Args:
            gubert_features: [B, T, D] GuBERT features
            speaker_embedding: [B, speaker_dim] speaker embedding
            target_length: Optional target output length for trimming

        Returns:
            mel_recon: [B, n_mels, T'] reconstructed mel spectrogram
        """
        # Initial projection: [B, T, D] -> [B, T, hidden_dim]
        x = self.input_proj(gubert_features)

        # Permute for conv: [B, T, C] -> [B, C, T]
        x = x.permute(0, 2, 1)

        # Upsample with FiLM conditioning
        for upsample, film in zip(self.upsample_layers, self.upsample_films):
            x = upsample(x)
            x = film(x, speaker_embedding)

        # Residual conv blocks with FiLM
        for block in self.conv_blocks:
            x = block(x, speaker_embedding)

        # Output projection: [B, hidden_dim, T] -> [B, n_mels, T]
        mel_recon = self.output_proj(x)

        # Trim to target length if specified
        if target_length is not None and mel_recon.shape[-1] > target_length:
            mel_recon = mel_recon[..., :target_length]

        return mel_recon

    def get_num_params(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# ArcFace Loss
# =============================================================================

class ArcFaceHead(nn.Module):
    """
    ArcFace head for speaker classification with angular margin.

    Adds angular margin to encourage better speaker separation in embedding space.
    This creates an embedding space where:
    - Same-speaker embeddings cluster tightly together
    - Different speakers are separated by at least the angular margin
    - Interpolation between nearby points produces valid speaker characteristics

    Reference: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    """
    def __init__(
        self,
        speaker_dim: int,
        num_speakers: int,
        scale: float = 30.0,
        margin: float = 0.5,
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_speakers = num_speakers

        # Weight matrix (class centers on hypersphere)
        self.weight = nn.Parameter(torch.FloatTensor(num_speakers, speaker_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin constants
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        # Easy margin threshold: when theta > pi - margin, use linear penalty instead
        # This prevents cos(theta + m) from becoming too negative
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute ArcFace logits.

        Args:
            embeddings: [B, speaker_dim] L2-normalized speaker embeddings
            labels: [B] speaker labels (required during training for margin)

        Returns:
            logits: [B, num_speakers] classification logits
        """
        # Normalize weights (class centers on unit hypersphere)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity = cos(theta) since both are normalized
        cosine = F.linear(embeddings, weight_norm)  # [B, num_speakers]

        if labels is not None and self.training:
            # Add angular margin to target class
            # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
            sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)

            # Easy margin: when theta > pi - m, cos(theta + m) becomes problematic
            # Use linear penalty instead: cos(theta) - m*sin(m)
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

            # One-hot for target class
            one_hot = F.one_hot(labels, num_classes=self.num_speakers).float()

            # Apply margin only to target class
            logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        else:
            logits = cosine

        # Scale (temperature) - higher scale = sharper softmax
        logits = logits * self.scale

        return logits

    def get_speaker_centers(self) -> torch.Tensor:
        """Get L2-normalized speaker class centers for inference/interpolation."""
        return F.normalize(self.weight, p=2, dim=1)


# =============================================================================
# GE2E Loss (Generalized End-to-End)
# =============================================================================

class GE2ELoss(nn.Module):
    """
    Generalized End-to-End (GE2E) loss for speaker verification.

    This loss enforces content-invariance in speaker embeddings by:
    1. Computing per-speaker centroids within each batch
    2. Pushing each utterance toward its own speaker's centroid
    3. Pushing it away from other speakers' centroids

    Requires structured batches: [N speakers × M utterances] = N*M total samples.
    The batch should be organized so that samples 0..M-1 are speaker 0,
    samples M..2M-1 are speaker 1, etc.

    Reference: "Generalized End-to-End Loss for Speaker Verification" (Google, 2018)

    Args:
        init_w: Initial value for learnable scale parameter
        init_b: Initial value for learnable bias parameter
    """

    def __init__(self, init_w: float = 10.0, init_b: float = -5.0):
        super().__init__()
        # Learnable scale and bias for similarity scores
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(
        self,
        embeddings: torch.Tensor,
        n_speakers: int,
        n_utterances: int,
    ) -> torch.Tensor:
        """
        Compute GE2E loss.

        Args:
            embeddings: [N*M, D] L2-normalized speaker embeddings
                Organized as [spk0_utt0, spk0_utt1, ..., spk0_uttM-1,
                             spk1_utt0, spk1_utt1, ..., spk1_uttM-1, ...]
            n_speakers: Number of speakers (N)
            n_utterances: Number of utterances per speaker (M)

        Returns:
            GE2E loss (scalar)
        """
        # Reshape to [N, M, D]
        embeddings = embeddings.view(n_speakers, n_utterances, -1)

        # Compute centroids for each speaker: [N, D]
        # Use all M utterances for centroid
        centroids = embeddings.mean(dim=1)  # [N, D]

        # For each utterance, compute centroid EXCLUDING that utterance
        # This is the "leave-one-out" centroid for more stable training
        # centroid_exclusive[j, i] = mean of speaker j's embeddings excluding utterance i
        centroids_exclusive = []
        for i in range(n_utterances):
            # Mask out utterance i
            mask = torch.ones(n_utterances, dtype=torch.bool, device=embeddings.device)
            mask[i] = False
            # Compute centroid from remaining utterances: [N, D]
            centroid_excl = embeddings[:, mask, :].mean(dim=1)
            centroids_exclusive.append(centroid_excl)
        # Stack: [M, N, D] -> we'll index appropriately below

        # Compute similarity matrix
        # For each utterance (j, i), compute similarity to:
        #   - Its own speaker's exclusive centroid (positive)
        #   - All other speakers' centroids (negatives)

        # Similarity to all centroids: [N, M, N]
        # sim[j, i, k] = cosine_sim(embedding[j, i], centroid[k])
        sim_matrix = torch.zeros(n_speakers, n_utterances, n_speakers, device=embeddings.device)

        for j in range(n_speakers):
            for i in range(n_utterances):
                emb = embeddings[j, i]  # [D]

                for k in range(n_speakers):
                    if k == j:
                        # Use exclusive centroid for own speaker
                        centroid = centroids_exclusive[i][j]  # [D]
                    else:
                        # Use full centroid for other speakers
                        centroid = centroids[k]  # [D]

                    # Cosine similarity (embeddings should already be L2-normalized)
                    sim = F.cosine_similarity(emb.unsqueeze(0), centroid.unsqueeze(0))
                    sim_matrix[j, i, k] = sim

        # Apply learnable scale and bias
        sim_matrix = self.w * sim_matrix + self.b

        # Softmax loss: for each utterance, the target is its own speaker
        # Reshape for cross-entropy: [N*M, N]
        sim_flat = sim_matrix.view(n_speakers * n_utterances, n_speakers)
        targets = torch.arange(n_speakers, device=embeddings.device).repeat_interleave(n_utterances)

        loss = F.cross_entropy(sim_flat, targets)

        return loss

    def compute_eer(
        self,
        embeddings: torch.Tensor,
        n_speakers: int,
        n_utterances: int,
    ) -> float:
        """
        Compute Equal Error Rate (EER) for evaluation.

        Returns approximate EER based on similarity scores.
        """
        with torch.no_grad():
            embeddings = embeddings.view(n_speakers, n_utterances, -1)
            centroids = embeddings.mean(dim=1)  # [N, D]

            # Compute all pairwise similarities
            # [N, M, D] @ [N, D].T -> [N, M, N]
            sim_matrix = torch.einsum('nmd,kd->nmk', embeddings, centroids)

            # Positive pairs: diagonal (same speaker)
            # Negative pairs: off-diagonal (different speakers)
            positive_sims = []
            negative_sims = []

            for j in range(n_speakers):
                for i in range(n_utterances):
                    for k in range(n_speakers):
                        sim = sim_matrix[j, i, k].item()
                        if k == j:
                            positive_sims.append(sim)
                        else:
                            negative_sims.append(sim)

            # Simple EER approximation: find threshold where FAR = FRR
            positive_sims = sorted(positive_sims)
            negative_sims = sorted(negative_sims, reverse=True)

            # Binary search for EER threshold
            n_pos = len(positive_sims)
            n_neg = len(negative_sims)

            eer = 0.5  # Default if can't compute
            for threshold_idx in range(min(n_pos, n_neg)):
                frr = threshold_idx / n_pos  # False rejection rate
                far = threshold_idx / n_neg  # False acceptance rate
                if frr >= far:
                    eer = (frr + far) / 2
                    break

            return eer


# =============================================================================
# Combined Model
# =============================================================================

class SpeechReconstructionModel(nn.Module):
    """
    Combined model for speech reconstruction.

    Components:
    1. SpeakerEncoder: mel_spec -> speaker_embedding
    2. MelReconstructor: gubert_features + speaker_embedding -> mel_spec_recon
    3. (Optional) ArcFaceHead: speaker_embedding -> speaker_logits

    Training:
    - Input: mel_spec, gubert_features, (optional) speaker_ids
    - Output: reconstructed mel_spec, speaker_embedding, (optional) speaker_logits
    """
    def __init__(self, config: SpeechReconstructionConfig):
        super().__init__()
        self.config = config

        # Speaker encoder
        self.speaker_encoder = SpeakerEncoder(config.speaker_encoder)

        # Mel reconstructor
        self.mel_reconstructor = MelReconstructor(config.mel_reconstructor)

        # Optional ArcFace head
        self.arcface_head = None
        if config.use_arcface and config.num_speakers > 0:
            self.arcface_head = ArcFaceHead(
                speaker_dim=config.speaker_encoder.speaker_dim,
                num_speakers=config.num_speakers,
                scale=config.arcface_scale,
                margin=config.arcface_margin,
            )

    def forward(
        self,
        mel_spec: torch.Tensor,
        gubert_features: torch.Tensor,
        mel_lengths: torch.Tensor = None,
        gubert_lengths: torch.Tensor = None,
        speaker_ids: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            mel_spec: [B, n_mels, T] input mel spectrogram
            gubert_features: [B, T', D] GuBERT features
            mel_lengths: [B] valid mel lengths
            gubert_lengths: [B] valid GuBERT feature lengths
            speaker_ids: [B] speaker IDs for ArcFace (optional)

        Returns:
            dict with:
                - mel_recon: [B, n_mels, T] reconstructed mel spectrogram
                - speaker_embedding: [B, speaker_dim] speaker embedding
                - speaker_logits: [B, num_speakers] (if use_arcface)
        """
        # Extract speaker embedding from mel spec
        speaker_result = self.speaker_encoder(
            mel_spec,
            lengths=mel_lengths,
        )
        speaker_embedding = speaker_result["speaker_embedding"]

        # Reconstruct mel from GuBERT features + speaker embedding
        target_length = mel_spec.shape[-1]
        mel_recon = self.mel_reconstructor(
            gubert_features,
            speaker_embedding,
            target_length=target_length,
        )

        result = {
            "mel_recon": mel_recon,
            "speaker_embedding": speaker_embedding,
        }

        # ArcFace logits
        if self.arcface_head is not None:
            speaker_logits = self.arcface_head(speaker_embedding, speaker_ids)
            result["speaker_logits"] = speaker_logits

        return result

    def encode_speaker(
        self,
        mel_spec: torch.Tensor,
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """Extract speaker embedding only."""
        result = self.speaker_encoder(mel_spec, lengths=lengths)
        return result["speaker_embedding"]

    def reconstruct_mel(
        self,
        gubert_features: torch.Tensor,
        speaker_embedding: torch.Tensor,
        target_length: int = None,
    ) -> torch.Tensor:
        """Reconstruct mel from features + embedding."""
        return self.mel_reconstructor(
            gubert_features,
            speaker_embedding,
            target_length=target_length,
        )

    def get_num_params(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Predefined Configs
# =============================================================================

SPEAKER_ENCODER_CONFIGS = {
    "tiny": SpeakerEncoderConfig(
        encoder_dim=128,
        num_layers=2,
        num_heads=4,
        ff_dim=512,
        speaker_dim=128,
        conv_channels=[64, 128],
        conv_kernel_sizes=[5, 3],
        conv_strides=[2, 2],
    ),
    "small": SpeakerEncoderConfig(
        encoder_dim=256,
        num_layers=4,
        num_heads=4,
        ff_dim=1024,
        speaker_dim=256,
        conv_channels=[128, 256],
        conv_kernel_sizes=[5, 3],
        conv_strides=[2, 2],
    ),
    "medium": SpeakerEncoderConfig(
        encoder_dim=384,
        num_layers=6,
        num_heads=6,
        ff_dim=1536,
        speaker_dim=256,
        conv_channels=[192, 384],
        conv_kernel_sizes=[5, 3],
        conv_strides=[2, 2],
    ),
    "large": SpeakerEncoderConfig(
        encoder_dim=512,
        num_layers=8,
        num_heads=8,
        ff_dim=2048,
        speaker_dim=512,
        conv_channels=[256, 512],
        conv_kernel_sizes=[5, 3],
        conv_strides=[2, 2],
    ),
}

MEL_RECONSTRUCTOR_CONFIGS = {
    "tiny": MelReconstructorConfig(
        hidden_dim=256,
        num_layers=2,
        kernel_size=5,
        film_hidden_dim=128,
    ),
    "small": MelReconstructorConfig(
        hidden_dim=384,
        num_layers=4,
        kernel_size=5,
        film_hidden_dim=192,
    ),
    "medium": MelReconstructorConfig(
        hidden_dim=512,
        num_layers=6,
        kernel_size=5,
        film_hidden_dim=256,
    ),
    "large": MelReconstructorConfig(
        hidden_dim=768,
        num_layers=8,
        kernel_size=7,
        film_hidden_dim=384,
    ),
}


def create_speech_reconstruction_model(
    config: str = "small",
    gubert_dim: int = 256,
    n_mels: int = 80,
    use_arcface: bool = False,
    num_speakers: int = 0,
    **kwargs,
) -> SpeechReconstructionModel:
    """
    Create a SpeechReconstructionModel from predefined config.

    Args:
        config: Config size (tiny, small, medium, large)
        gubert_dim: GuBERT feature dimension
        n_mels: Number of mel channels
        use_arcface: Whether to use ArcFace loss
        num_speakers: Number of speakers (required if use_arcface)
        **kwargs: Additional config overrides

    Returns:
        SpeechReconstructionModel
    """
    if config not in SPEAKER_ENCODER_CONFIGS:
        raise ValueError(f"Unknown config: {config}. Available: {list(SPEAKER_ENCODER_CONFIGS.keys())}")

    speaker_config = SPEAKER_ENCODER_CONFIGS[config]
    speaker_config.n_mels = n_mels

    mel_config = MEL_RECONSTRUCTOR_CONFIGS[config]
    mel_config.gubert_dim = gubert_dim
    mel_config.speaker_dim = speaker_config.speaker_dim
    mel_config.n_mels = n_mels

    full_config = SpeechReconstructionConfig(
        speaker_encoder=speaker_config,
        mel_reconstructor=mel_config,
        use_arcface=use_arcface,
        num_speakers=num_speakers,
        **kwargs,
    )

    return SpeechReconstructionModel(full_config)
