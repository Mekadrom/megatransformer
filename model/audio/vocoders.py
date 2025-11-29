import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from typing import Optional, Literal

from model.megatransformer_modules import Snake

class VocoderResidualBlock(nn.Module):
    """Residual block with dilated convolutions for the vocoder."""
    def __init__(
        self,
        channels: int,
        dilation: int = 1,
        kernel_size: int = 3,
        activation_fn: str = 'leaky_relu',
        snake_alpha_init: float = 1.0,
        leaky_relu_slope: float = 0.1
    ):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope
        self.convs = nn.ModuleList([
            nn.Conv1d(
                channels, 
                channels, 
                kernel_size=kernel_size, 
                dilation=dilation,
                padding=dilation * (kernel_size - 1) // 2
            ),
            nn.Conv1d(
                channels, 
                channels, 
                kernel_size=kernel_size, 
                dilation=dilation,
                padding=dilation * (kernel_size - 1) // 2
            )
        ])
        self.gate_conv = nn.Conv1d(channels, channels, kernel_size=1)

        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)

        match activation_fn:
            case 'leaky_relu':
                self.act1 = nn.LeakyReLU(negative_slope=leaky_relu_slope)
                self.act2 = nn.LeakyReLU(negative_slope=leaky_relu_slope)
            case 'snake':
                self.act1 = Snake(channels, snake_alpha_init)
                self.act2 = Snake(channels, snake_alpha_init)
            case _:
                raise ValueError(f"Unsupported activation function: {activation_fn}")

        self._init_weights()

    def _init_weights(self):
        # Initialize weights for the Conv1d layers
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, nonlinearity='leaky_relu')
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        nn.init.kaiming_normal_(self.gate_conv.weight, nonlinearity='leaky_relu')
        if self.gate_conv.bias is not None:
            nn.init.zeros_(self.gate_conv.bias)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        
        features = self.norm1(features)
        features = self.act1(features)
        features = self.convs[0](features)
        
        features = self.norm2(features)
        features = self.act2(features)
        features = self.convs[1](features)
        
        # Gate mechanism for controlled information flow
        gate = torch.sigmoid(self.gate_conv(residual))
        return residual + gate * features

class VocoderUpsampleBlock(nn.Module):
    """Upsampling block for the vocoder."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_factor: int,
        kernel_size: int = 8,
        leaky_relu_slope: float = 0.1,
        norm_type: Literal['batch', 'weight', 'group'] = 'batch'
    ):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope
        
        # Transposed convolution for upsampling
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=upsample_factor,
            padding=(kernel_size - upsample_factor) // 2
        )

        if norm_type == 'batch':
            self.norm = nn.BatchNorm1d(out_channels)
            self._init_weights()
        elif norm_type == 'weight':
            self.conv = weight_norm(self.conv)
        elif norm_type == 'group':
            # num_groups=32 is common, but adjust if out_channels isn't divisible
            num_groups = min(32, out_channels)
            self.norm = nn.GroupNorm(num_groups, out_channels)
            self._init_weights()
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

    def _init_weights(self):
        # Initialize weights for the ConvTranspose1d layer
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='leaky_relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.conv(features)
        if hasattr(self, "norm"):
            features = self.norm(features)
        features = F.leaky_relu(features, negative_slope=self.leaky_relu_slope)
        return features


class VocoderCrossAttention(nn.Module):
    """
    Cross-attention module for conditioning the vocoder on text embeddings.
    Applied at mel-spectrogram resolution for efficiency.
    Uses F.scaled_dot_product_attention for Flash Attention support.
    """
    def __init__(
        self,
        audio_channels: int,
        conditioning_channels: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = audio_channels // n_heads
        self.dropout_p = dropout

        assert audio_channels % n_heads == 0, "audio_channels must be divisible by n_heads"

        # Project audio features to queries
        self.to_q = nn.Conv1d(audio_channels, audio_channels, 1)
        # Project conditioning to keys and values
        self.to_k = nn.Linear(conditioning_channels, audio_channels)
        self.to_v = nn.Linear(conditioning_channels, audio_channels)
        # Output projection
        self.to_out = nn.Conv1d(audio_channels, audio_channels, 1)

        # Layer norm for stability
        self.norm_audio = nn.GroupNorm(1, audio_channels)  # equivalent to LayerNorm for conv
        self.norm_cond = nn.LayerNorm(conditioning_channels)

    def forward(
        self,
        audio_features: torch.Tensor,   # [B, C, T_audio]
        condition: torch.Tensor,         # [B, T_text, C_cond]
        condition_mask: Optional[torch.Tensor] = None  # [B, T_text], True for valid tokens
    ) -> torch.Tensor:
        B, C, T_audio = audio_features.shape
        T_text = condition.shape[1]

        # Normalize inputs
        audio_norm = self.norm_audio(audio_features)
        cond_norm = self.norm_cond(condition)

        # Queries from audio: [B, C, T_audio] -> [B, heads, T_audio, head_dim]
        q = self.to_q(audio_norm)
        q = q.view(B, self.n_heads, self.head_dim, T_audio).transpose(2, 3)

        # Keys and values from text: [B, T_text, C] -> [B, heads, T_text, head_dim]
        k = self.to_k(cond_norm).view(B, T_text, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(cond_norm).view(B, T_text, self.n_heads, self.head_dim).transpose(1, 2)

        # Create attention mask for scaled_dot_product_attention
        # SDPA expects: True = masked (ignore), so we invert our mask
        attn_mask = None
        if condition_mask is not None:
            # [B, T_text] -> [B, 1, 1, T_text] broadcast to [B, heads, T_audio, T_text]
            attn_mask = ~condition_mask[:, None, None, :]

        # Flash attention / memory efficient attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )  # [B, heads, T_audio, head_dim]

        # Reshape back: [B, heads, T_audio, head_dim] -> [B, C, T_audio]
        out = out.transpose(2, 3).reshape(B, C, T_audio)

        # Residual connection
        return audio_features + self.to_out(out)


class AudioVocoder(nn.Module):
    """
    Neural vocoder that converts mel spectrograms to audio waveforms.

    This vocoder can be trained alongside the diffusion model and is designed
    to efficiently convert mel spectrograms to high-quality audio waveforms.

    Supports two conditioning modes:
    - 'interpolate': Linear interpolation of conditioning to waveform resolution (default, legacy)
    - 'attention': Cross-attention at mel-spectrogram resolution (more flexible, efficient)
    """
    def __init__(
        self,
        hidden_channels,
        in_channels: int,  # Number of mel bands
        conditioning_channels: int,
        upsample_factors: list[int] = [8, 8, 8],  # Total upsampling of 512 (matching hop_length)
        n_residual_layers: int = 4,
        dilation_cycle: int = 4,
        kernel_size: int = 3,
        leaky_relu_slope: float = 0.1,
        conditioning_enabled: bool = True,
        conditioning_mode: Literal['interpolate', 'attention'] = 'interpolate',  # 'interpolate' or 'attention'
        attention_n_heads: int = 4,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.has_condition = conditioning_enabled
        self.conditioning_mode = conditioning_mode

        assert conditioning_mode in ('interpolate', 'attention'), \
            f"conditioning_mode must be 'interpolate' or 'attention', got {conditioning_mode}"

        # Initial convolution
        self.initial_conv = nn.Conv1d(
            in_channels,
            hidden_channels,
            kernel_size=7,
            padding=3
        )

        # Conditioning at mel resolution (before upsampling) for attention mode
        if conditioning_enabled and conditioning_mode == 'attention':
            self.conditioning_attention = VocoderCrossAttention(
                audio_channels=hidden_channels,
                conditioning_channels=conditioning_channels,
                n_heads=attention_n_heads,
                dropout=attention_dropout,
            )

        # Upsampling blocks
        self.upsample_blocks = nn.ModuleList()
        current_channels = hidden_channels
        for factor in upsample_factors:
            self.upsample_blocks.append(
                VocoderUpsampleBlock(
                    current_channels,
                    current_channels // 2,
                    upsample_factor=factor,
                    leaky_relu_slope=leaky_relu_slope
                )
            )
            current_channels //= 2

        # Residual blocks with increasing dilation
        self.residual_blocks = nn.ModuleList()
        for i in range(n_residual_layers):
            dilation = 2 ** (i % dilation_cycle)
            self.residual_blocks.append(
                VocoderResidualBlock(
                    channels=current_channels,
                    dilation=dilation,
                    kernel_size=kernel_size,
                    leaky_relu_slope=leaky_relu_slope
                )
            )

        # Conditioning layer for interpolate mode (applied after upsampling)
        if conditioning_enabled and conditioning_mode == 'interpolate':
            self.conditioning_layer = nn.Conv1d(
                conditioning_channels,
                current_channels,
                kernel_size=1
            )

        # Final output layers
        self.final_layers = nn.Sequential(
            nn.LeakyReLU(leaky_relu_slope),
            nn.Conv1d(
                current_channels,
                current_channels,
                kernel_size=3,
                padding=1
            ),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Conv1d(
                current_channels,
                1,  # Single channel output for mono waveform
                kernel_size=3,
                padding=1
            ),
            nn.Tanh()  # Output in range [-1, 1]
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize weights for the Conv1d layers
        nn.init.kaiming_normal_(self.initial_conv.weight, nonlinearity='leaky_relu')
        if self.initial_conv.bias is not None:
            nn.init.zeros_(self.initial_conv.bias)

        for layer in self.final_layers:
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(
        self,
        mel_spec: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the vocoder.

        Args:
            mel_spec: [B, n_mels, T_mel] Mel spectrogram
            condition: Optional [B, T_cond, C] text/conditioning embeddings
            condition_mask: Optional [B, T_cond] boolean mask where True = valid token
                           (only used for attention mode)

        Returns:
            [B, T_audio] Audio waveform
        """
        if torch.isnan(mel_spec).any():
            print("NaN detected in mel_spec input to vocoder")
        if torch.isinf(mel_spec).any():
            print("Inf detected in mel_spec input to vocoder")

        # remove channel dimension
        if mel_spec.dim() == 4:
            mel_spec = mel_spec.squeeze(1)  # [B, n_mels, T_mel]

        # Initial processing
        x = self.initial_conv(mel_spec)  # [B, hidden_channels, T_mel]

        # Apply attention conditioning at mel resolution (before upsampling)
        if self.has_condition and self.conditioning_mode == 'attention' and condition is not None:
            x = self.conditioning_attention(x, condition, condition_mask)

        # Upsampling
        for i, upsample in enumerate(self.upsample_blocks):
            x = upsample(x)

        # Apply interpolation conditioning at waveform resolution (after upsampling)
        if self.has_condition and self.conditioning_mode == 'interpolate' and condition is not None:
            # Reshape conditioning to match the waveform sequence length
            condition_upsample = F.interpolate(
                condition.permute(0, 2, 1),  # [B, T_cond, C] -> [B, C, T_cond]
                size=x.size(2),
                mode='linear',
                align_corners=False
            )

            cond_proj = self.conditioning_layer(condition_upsample)

            x = x + cond_proj

        # Apply residual blocks
        for i, res_block in enumerate(self.residual_blocks):
            x = res_block(x)

        # Final layers
        waveform = self.final_layers(x)
        waveform = waveform.squeeze(1)  # Remove channel dimension

        return waveform

class AudioVocoderInterleaveUpsampleResidual(nn.Module):
    def __init__(
        self,
        hidden_channels,
        in_channels: int,  # Number of mel bands
        conditioning_channels: int,
        upsample_factors: list[int] = [8, 8, 8],  # Total upsampling of 512 (matching hop_length)
        n_residual_layers_per_scale: int = 3,
        dilation_cycle: int = 4,
        kernel_size: int = 3,
        leaky_relu_slope: float = 0.1,
        conditioning_enabled: bool = True,
        attention_n_heads: int = 4,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.has_condition = conditioning_enabled

        # Initial convolution
        self.initial_conv = nn.Conv1d(
            in_channels,
            hidden_channels,
            kernel_size=7,
            padding=3
        )

        # Conditioning at mel resolution (before upsampling) for attention mode
        self.conditioning_attention = VocoderCrossAttention(
            audio_channels=hidden_channels,
            conditioning_channels=conditioning_channels,
            n_heads=attention_n_heads,
            dropout=attention_dropout,
        )

        current_channels = hidden_channels

        # Upsampling blocks
        self.upsample_blocks = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()

        for scale_idx, factor in enumerate(upsample_factors):
            self.upsample_blocks.append(
                VocoderUpsampleBlock(
                    current_channels,
                    current_channels // 2,
                    upsample_factor=factor,
                    leaky_relu_slope=leaky_relu_slope
                )
            )
            current_channels //= 2
            
            # Residual blocks at this scale - dilations cycle within each scale
            scale_residual_blocks = nn.ModuleList()
            for i in range(n_residual_layers_per_scale):
                dilation = 2 ** (i % dilation_cycle)
                scale_residual_blocks.append(
                    VocoderResidualBlock(
                        channels=current_channels,
                        dilation=dilation,
                        kernel_size=kernel_size,
                        leaky_relu_slope=leaky_relu_slope
                    )
                )
            self.residual_blocks.append(scale_residual_blocks)

        # Final output layers
        self.final_layers = nn.Sequential(
            nn.LeakyReLU(leaky_relu_slope),
            nn.Conv1d(
                current_channels,
                current_channels,
                kernel_size=3,
                padding=1
            ),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Conv1d(
                current_channels,
                1,  # Single channel output for mono waveform
                kernel_size=3,
                padding=1
            ),
            nn.Tanh()  # Output in range [-1, 1]
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize weights for the Conv1d layers
        nn.init.kaiming_normal_(self.initial_conv.weight, nonlinearity='leaky_relu')
        if self.initial_conv.bias is not None:
            nn.init.zeros_(self.initial_conv.bias)

        for layer in self.final_layers:
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(
        self,
        mel_spec: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the vocoder.

        Args:
            mel_spec: [B, n_mels, T_mel] Mel spectrogram
            condition: Optional [B, T_cond, C] text/conditioning embeddings
            condition_mask: Optional [B, T_cond] boolean mask where True = valid token
                           (only used for attention mode)

        Returns:
            [B, T_audio] Audio waveform
        """
        if torch.isnan(mel_spec).any():
            print("NaN detected in mel_spec input to vocoder")
        if torch.isinf(mel_spec).any():
            print("Inf detected in mel_spec input to vocoder")

        # remove channel dimension
        if mel_spec.dim() == 4:
            mel_spec = mel_spec.squeeze(1)  # [B, n_mels, T_mel]

        # Initial processing
        x = self.initial_conv(mel_spec)  # [B, hidden_channels, T_mel]

        # Apply attention conditioning at mel resolution (before upsampling)
        if self.has_condition and condition is not None:
            x = self.conditioning_attention(x, condition, condition_mask)

        for upsample, res_blocks in zip(self.upsample_blocks, self.residual_blocks):
            x = upsample(x)
            for res_block in res_blocks:
                x = res_block(x)

        # Final layers
        waveform = self.final_layers(x)
        waveform = waveform.squeeze(1)  # Remove channel dimension

        return waveform
