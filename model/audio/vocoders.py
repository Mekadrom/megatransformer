import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from typing import Dict, Optional, Literal

from transformers import T5EncoderModel, T5Tokenizer

import megatransformer_utils
from model.audio.criteria import MultiResolutionSTFTLoss, StableMelSpectrogramLoss
from model.activations import Snake

class VocoderResidualBlock(nn.Module):
    """Residual block with dilated convolutions for the vocoder."""
    def __init__(
        self,
        channels: int,
        dilation: int = 1,
        kernel_size: int = 3,
        activation_fn: Literal["leaky_relu", "snake"] = 'leaky_relu',
        snake_alpha_init: float = 1.0,
        leaky_relu_slope: float = 0.1
    ):
        super().__init__()
        self.activation_fn = activation_fn
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
        for conv in self.convs:
            if self.activation_fn == 'snake':
                nn.init.xavier_uniform_(conv.weight, gain=1.0)
            else:
                nn.init.kaiming_normal_(conv.weight, a=self.leaky_relu_slope, mode='fan_in', nonlinearity='leaky_relu')
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        
        nn.init.xavier_uniform_(self.gate_conv.weight, gain=1.0)
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
        activation_fn: Literal["leaky_relu", "snake"] = 'leaky_relu',
        snake_alpha_init: float = 1.0,
        leaky_relu_slope: float = 0.1,
        norm_type: Literal['batch', 'weight', 'group'] = 'weight'
    ):
        super().__init__()
        self.activation_fn = activation_fn
        self.leaky_relu_slope = leaky_relu_slope
        
        # Transposed convolution for upsampling
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=upsample_factor,
            padding=(kernel_size - upsample_factor) // 2
        )

        match activation_fn:
            case 'leaky_relu':
                self.act1 = nn.LeakyReLU(negative_slope=leaky_relu_slope)
            case 'snake':
                self.act1 = Snake(out_channels, snake_alpha_init)
            case _:
                raise ValueError(f"Unsupported activation function: {activation_fn}")

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
        if self.activation_fn == 'snake':
            nn.init.xavier_uniform_(self.conv.weight, gain=1.0)
        else:
            nn.init.kaiming_normal_(self.conv.weight, a=self.leaky_relu_slope, mode='fan_out', nonlinearity='leaky_relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.conv(features)
        if hasattr(self, "norm"):
            features = self.norm(features)
        features = self.act1(features)
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

        self._init_weights()

    def _init_weights(self):
        # Attention projections - use xavier with small gain
        nn.init.xavier_uniform_(self.to_q.weight, gain=0.02)
        nn.init.xavier_uniform_(self.to_k.weight, gain=0.02)
        nn.init.xavier_uniform_(self.to_v.weight, gain=0.02)
        nn.init.xavier_uniform_(self.to_out.weight, gain=0.02)
        
        if self.to_q.bias is not None:
            nn.init.zeros_(self.to_q.bias)
        if self.to_k.bias is not None:
            nn.init.zeros_(self.to_k.bias)
        if self.to_v.bias is not None:
            nn.init.zeros_(self.to_v.bias)
        if self.to_out.bias is not None:
            nn.init.zeros_(self.to_out.bias)

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
    def __init__(
        self,
        hidden_channels,
        in_channels: int,  # Number of mel bands
        conditioning_channels: int,
        upsample_factors: list[int] = [8, 8, 8],  # Total upsampling of 512 (matching hop_length)
        n_residual_layers: int = 3,
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
        self.leaky_relu_slope = leaky_relu_slope
        self.conditioning_enabled = conditioning_enabled

        # Initial convolution
        self.initial_conv = nn.Conv1d(
            in_channels,
            hidden_channels,
            kernel_size=7,
            padding=3
        )

        # Conditioning at mel resolution (before upsampling) for attention mode
        if conditioning_enabled:
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
            
            # Residual blocks at this scale - dilations cycle within each scale
            scale_residual_blocks = nn.ModuleList()
            for i in range(n_residual_layers):
                dilation = 2 ** (i % dilation_cycle)
                scale_residual_blocks.append(
                    VocoderResidualBlock(
                        channels=current_channels,
                        dilation=dilation,
                        kernel_size=kernel_size,
                        activation_fn='snake',
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
                padding=1,
                bias=False
            ),
            nn.Tanh()  # Output in range [-1, 1]
        )

        self._init_weights()

    def _init_weights(self):
        # Initial conv - standard kaiming with correct a parameter
        nn.init.kaiming_normal_(self.initial_conv.weight, a=self.leaky_relu_slope, mode='fan_in', nonlinearity='leaky_relu')
        if self.initial_conv.bias is not None:
            nn.init.zeros_(self.initial_conv.bias)
        
        # Final layers - need special handling
        for i, layer in enumerate(self.final_layers):
            if isinstance(layer, nn.Conv1d):
                if i == len(self.final_layers) - 2:  # Last conv before Tanh
                    # Small init for layer feeding into Tanh to avoid saturation
                    nn.init.normal_(layer.weight, mean=0.0, std=0.001)
                else:
                    nn.init.kaiming_normal_(layer.weight, a=self.leaky_relu_slope, mode='fan_in', nonlinearity='leaky_relu')
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
        if self.conditioning_enabled and condition is not None:
            x = self.conditioning_attention(x, condition, condition_mask)

        for upsample, res_blocks in zip(self.upsample_blocks, self.residual_blocks):
            x = upsample(x)
            for res_block in res_blocks:
                x = res_block(x)

        # Final layers
        waveform = self.final_layers(x)
        waveform = waveform.squeeze(1)  # Remove channel dimension

        return waveform


class VocoderWithT5Conditioning(nn.Module):
    """
    Wrapper model that combines AudioVocoder with frozen T5 text embeddings for conditioning.
    """
    def __init__(self,
                 config: megatransformer_utils.MegaTransformerConfig,
                 sc_loss_weight: float = 1.0,
                 mag_loss_weight: float = 3.0,
                 waveform_l1_loss_weight: float = 0.1,
                 mel_recon_loss_weight: float = 1.0,
                 complex_stft_loss_weight: float = 2.0,
                 t5_model_name: str = "google/t5-v1_1-base",
                 conditioning_enabled: bool = True):
        super().__init__()
        self.config = config
        self.sc_loss_weight = sc_loss_weight
        self.mag_loss_weight = mag_loss_weight
        self.waveform_l1_loss_weight = waveform_l1_loss_weight
        self.mel_recon_loss_weight = mel_recon_loss_weight
        self.complex_stft_loss_weight = complex_stft_loss_weight
        self.conditioning_enabled = conditioning_enabled

        # self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db=80)

        if conditioning_enabled:
            # Frozen T5 encoder for text conditioning
            self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
            self.t5_encoder = T5EncoderModel.from_pretrained(t5_model_name)

            # Freeze T5 parameters
            for param in self.t5_encoder.parameters():
                param.requires_grad = False

            # Get T5 hidden size
            self.t5_hidden_size = self.t5_encoder.config.d_model

            # Project T5 embeddings to vocoder conditioning size
            self.condition_proj = nn.Linear(self.t5_hidden_size, config.hidden_size)

        # AudioVocoder from megatransformer_audio_decoder
        self.vocoder = AudioVocoder(
            hidden_channels=config.audio_vocoder_hidden_channels,
            in_channels=config.audio_n_mels,
            conditioning_channels=config.hidden_size,
            upsample_factors=config.audio_vocoder_upsample_factors,
            n_residual_layers=config.audio_vocoder_n_residual_layers,
            conditioning_enabled=conditioning_enabled,
            attention_n_heads=4,
        )

        # Loss functions
        self.stft_loss = MultiResolutionSTFTLoss()
        self.mel_recon_loss = StableMelSpectrogramLoss(
            sample_rate=config.audio_sample_rate,
            n_fft=config.audio_n_fft,
            hop_length=config.audio_hop_length,
            n_mels=config.audio_n_mels,
        )

    def get_text_embeddings(self, text_input_ids: torch.Tensor, text_attention_mask: torch.Tensor) -> torch.Tensor:
        """Get frozen T5 embeddings for text conditioning."""
        with torch.no_grad():
            t5_outputs = self.t5_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
            )
            text_embeddings = t5_outputs.last_hidden_state

        # Project to vocoder conditioning dimension
        text_embeddings = self.condition_proj(text_embeddings)
        return text_embeddings

    def forward(
        self,
        mel_spec: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        waveform_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the vocoder with T5 text conditioning.

        Args:
            mel_spec: [B, n_mels, T_mel] Mel spectrogram input
            text_input_ids: [B, seq_len] T5 tokenized text
            text_attention_mask: [B, seq_len] Attention mask for T5
            waveform_labels: [B, T_audio] Target waveform for training

        Returns:
            Dictionary with loss and outputs
        """
        # Get text conditioning from frozen T5
        if self.conditioning_enabled:
            text_condition = self.get_text_embeddings(text_input_ids, text_attention_mask)
        else:
            text_condition = None

        # Generate waveform through vocoder
        pred_waveform: torch.Tensor = self.vocoder(mel_spec, condition=text_condition)

        outputs = {"pred_waveform": pred_waveform}

        if waveform_labels is not None:
            # Ensure waveform_labels has batch dimension to match pred_waveform
            if waveform_labels.dim() == 1:
                waveform_labels = waveform_labels.unsqueeze(0)

            # Align waveform lengths
            min_len = min(pred_waveform.shape[-1], waveform_labels.shape[-1])
            pred_waveform_aligned = pred_waveform[..., :min_len]
            waveform_labels_aligned = waveform_labels[..., :min_len]

            # Compute losses
            waveform_l1 = F.l1_loss(pred_waveform_aligned, waveform_labels_aligned)
            # STFT loss expects [B, 1, T] shape
            sc_loss, mag_loss, complex_stft_loss = self.stft_loss(
                pred_waveform_aligned.unsqueeze(1) if pred_waveform_aligned.dim() == 2 else pred_waveform_aligned,
                waveform_labels_aligned.unsqueeze(1) if waveform_labels_aligned.dim() == 2 else waveform_labels_aligned
            )

            mel_recon_loss = self.mel_recon_loss(pred_waveform_aligned, mel_spec)

            total_loss = (self.sc_loss_weight * sc_loss) + (self.mag_loss_weight * mag_loss) + (self.complex_stft_loss_weight * complex_stft_loss) + (self.waveform_l1_loss_weight * waveform_l1) + (self.mel_recon_loss_weight * mel_recon_loss)

            outputs.update({
                "loss": total_loss,
                "waveform_l1": waveform_l1,
                "sc_loss": sc_loss,
                "mag_loss": mag_loss,
                "mel_recon_loss": mel_recon_loss,
                "complex_stft_loss": complex_stft_loss,
            })

        return outputs


def create_small_vocoder_model(sc_loss_weight, mag_loss_weight, waveform_l1_loss_weight, mel_recon_loss_weight, complex_stft_loss_weight, conditioning_enabled: bool = True) -> VocoderWithT5Conditioning:
    """Create a small vocoder model for testing."""
    config = megatransformer_utils.MegaTransformerConfig(
        hidden_size=256,
        audio_n_mels=128,
        audio_n_fft=1024,
        audio_hop_length=512,
        audio_max_duration=10.0,
        audio_sample_rate=16000,
        audio_vocoder_hidden_channels=512,
        audio_vocoder_upsample_factors=[8, 8, 8],
        audio_vocoder_n_residual_layers=3,
    )
    return VocoderWithT5Conditioning(
        config,
        sc_loss_weight=sc_loss_weight,
        mag_loss_weight=mag_loss_weight,
        waveform_l1_loss_weight=waveform_l1_loss_weight,
        mel_recon_loss_weight=mel_recon_loss_weight,
        complex_stft_loss_weight=complex_stft_loss_weight,
        t5_model_name="google/t5-v1_1-small",
        conditioning_enabled=conditioning_enabled
    )


model_config_lookup = {
    "small_vocoder": create_small_vocoder_model,
}
