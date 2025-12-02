import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from typing import Dict, Optional, Literal

import megatransformer_utils
from model.audio.criteria import HighFreqSTFTLoss, MultiResolutionSTFTLoss, PhaseLoss, StableMelSpectrogramLoss
from model.activations import Snake
from model.audio.shared_window_buffer import SharedWindowBuffer


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


class AudioVocoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        in_channels: int,  # Number of mel bands
        # Total upsampling of 256 (matching hop_length)
        upsample_factors: list[int] = [8, 8, 4],
        n_residual_layers: int = 3,
        dilation_cycle: int = 4,
        kernel_size: int = 3,
        leaky_relu_slope: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.leaky_relu_slope = leaky_relu_slope

        # Initial convolution
        self.initial_conv = nn.Conv1d(
            in_channels,
            hidden_channels,
            kernel_size=7,
            padding=3
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
    ) -> torch.Tensor:
        """
        Forward pass of the vocoder.

        Args:
            mel_spec: [B, n_mels, T_mel] Mel spectrogram

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

        for upsample, res_blocks in zip(self.upsample_blocks, self.residual_blocks):
            x = upsample(x)
            for res_block in res_blocks:
                x = res_block(x)

        # Final layers
        waveform = self.final_layers(x)
        waveform = waveform.squeeze(1)  # Remove channel dimension

        return waveform


class VocoderWithLoss(nn.Module):
    def __init__(self,
                 shared_window_buffer: SharedWindowBuffer,
                 config: megatransformer_utils.MegaTransformerConfig,
                 sc_loss_weight: float = 1.0,
                 mag_loss_weight: float = 3.0,
                 waveform_l1_loss_weight: float = 0.1,
                 mel_recon_loss_weight: float = 1.0,
                 mel_recon_loss_weight_linspace_max: float = 1.0,
                 complex_stft_loss_weight: float = 2.0,
                 phase_loss_weight: float = 1.0,
                 phase_ip_loss_weight: float = 1.0,
                 phase_iaf_loss_weight: float = 1.0,
                 phase_gd_loss_weight: float = 1.0,
                 high_freq_stft_loss_weight: float = 0.0,
                 high_freq_stft_cutoff_bin: int = 256,
    ):
        super().__init__()
        self.config = config
        self.sc_loss_weight = sc_loss_weight
        self.mag_loss_weight = mag_loss_weight
        self.waveform_l1_loss_weight = waveform_l1_loss_weight
        self.mel_recon_loss_weight = mel_recon_loss_weight
        self.mel_recon_loss_weight_linspace_max = mel_recon_loss_weight_linspace_max
        self.complex_stft_loss_weight = complex_stft_loss_weight
        self.phase_loss_weight = phase_loss_weight
        self.phase_ip_loss_weight = phase_ip_loss_weight
        self.phase_iaf_loss_weight = phase_iaf_loss_weight
        self.phase_gd_loss_weight = phase_gd_loss_weight
        self.high_freq_stft_loss_weight = high_freq_stft_loss_weight

        self.n_fft = config.audio_n_fft
        self.hop_length = config.audio_hop_length

        self.shared_window_buffer = shared_window_buffer

        # AudioVocoder from megatransformer_audio_decoder
        self.vocoder = AudioVocoder(
            hidden_channels=config.audio_vocoder_hidden_channels,
            in_channels=config.audio_n_mels,
            upsample_factors=config.audio_vocoder_upsample_factors,
            n_residual_layers=config.audio_vocoder_n_residual_layers,
        )

        # Loss functions
        self.stft_loss = MultiResolutionSTFTLoss(shared_window_buffer=shared_window_buffer)
        self.mel_recon_loss = StableMelSpectrogramLoss(
            shared_window_buffer=shared_window_buffer,
            sample_rate=config.audio_sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=config.audio_n_mels,
            mel_recon_loss_weight_linspace_max=self.mel_recon_loss_weight_linspace_max
        )
        if phase_loss_weight > 0.0:
            self.phase_loss = PhaseLoss(shared_window_buffer=shared_window_buffer, config=config)
        else:
            self.phase_loss = None
        if high_freq_stft_loss_weight > 0.0:
            self.high_freq_stft_loss = HighFreqSTFTLoss(shared_window_buffer=shared_window_buffer, n_fft=config.audio_n_fft, cutoff_bin=high_freq_stft_cutoff_bin)
        else:
            self.high_freq_stft_loss = None

    def forward(
        self,
        mel_spec: torch.Tensor,
        waveform_labels: Optional[torch.Tensor] = None,
        target_complex_stfts: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Generate waveform through vocoder
        pred_waveform: torch.Tensor = self.vocoder(mel_spec)

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

            mel_recon_loss_value = self.mel_recon_loss(pred_waveform_aligned, mel_spec)

            if target_complex_stfts is None:
                target_complex_stfts = torch.stft(
                    waveform_labels_aligned.to(torch.float32), self.n_fft, self.hop_length,
                    window=self.shared_window_buffer.get_window(self.n_fft, waveform_labels_aligned.device), return_complex=True
                )

            if self.phase_loss is not None:
                ip_loss, iaf_loss, gd_loss = self.phase_loss(
                    pred_waveform_aligned,
                    waveform_labels_aligned,
                    target_complex_stfts=target_complex_stfts
                )
                phase_loss_value = (self.phase_ip_loss_weight * ip_loss +
                                    self.phase_iaf_loss_weight * iaf_loss +
                                    self.phase_gd_loss_weight * gd_loss)
            else:
                ip_loss = iaf_loss = gd_loss = phase_loss_value = 0.0

            if self.high_freq_stft_loss is not None:
                high_freq_stft_loss_value = self.high_freq_stft_loss(
                    pred_waveform_aligned,
                    waveform_labels_aligned,
                    target_complex_stfts=target_complex_stfts
                )
            else:
                high_freq_stft_loss_value = 0.0

            total_loss = (self.sc_loss_weight * sc_loss +
                          self.mag_loss_weight * mag_loss +
                          self.complex_stft_loss_weight * complex_stft_loss +
                          self.waveform_l1_loss_weight * waveform_l1 + 
                          self.mel_recon_loss_weight * mel_recon_loss_value +
                          self.phase_loss_weight * phase_loss_value +
                          self.high_freq_stft_loss_weight * high_freq_stft_loss_value)

            outputs.update({
                "loss": total_loss,
                "waveform_l1": waveform_l1,
                "sc_loss": sc_loss,
                "mag_loss": mag_loss,
                "mel_recon_loss": mel_recon_loss_value,
                "complex_stft_loss": complex_stft_loss,
                "phase_loss": phase_loss_value,
                "phase_ip_loss": ip_loss,
                "phase_iaf_loss": iaf_loss,
                "phase_gd_loss": gd_loss,
                "high_freq_stft_loss": high_freq_stft_loss_value,
            })

        return outputs


def create_small_vocoder_model(
        shared_window_buffer: SharedWindowBuffer,
        sc_loss_weight,
        mag_loss_weight,
        waveform_l1_loss_weight,
        mel_recon_loss_weight,
        mel_recon_loss_weight_linspace_max,
        complex_stft_loss_weight,
        phase_loss_weight,
        phase_ip_loss_weight,
        phase_iaf_loss_weight,
        phase_gd_loss_weight,
        high_freq_stft_loss_weight,
        high_freq_stft_cutoff_bin,
    ) -> VocoderWithLoss:
    """Create a small vocoder model for testing."""
    config = megatransformer_utils.MegaTransformerConfig(
        hidden_size=512,
        audio_n_mels=80,
        audio_n_fft=1024,
        audio_hop_length=256,
        audio_max_duration=10.0,
        audio_sample_rate=16000,
        audio_vocoder_hidden_channels=256,
        audio_vocoder_upsample_factors=[8, 8, 4],
        audio_vocoder_n_residual_layers=4,
    )
    return VocoderWithLoss(
        shared_window_buffer=shared_window_buffer,
        config=config,
        sc_loss_weight=sc_loss_weight,
        mag_loss_weight=mag_loss_weight,
        waveform_l1_loss_weight=waveform_l1_loss_weight,
        mel_recon_loss_weight=mel_recon_loss_weight,
        mel_recon_loss_weight_linspace_max=mel_recon_loss_weight_linspace_max,
        complex_stft_loss_weight=complex_stft_loss_weight,
        phase_loss_weight=phase_loss_weight,
        phase_ip_loss_weight=phase_ip_loss_weight,
        phase_iaf_loss_weight=phase_iaf_loss_weight,
        phase_gd_loss_weight=phase_gd_loss_weight,
        high_freq_stft_loss_weight=high_freq_stft_loss_weight,
        high_freq_stft_cutoff_bin=high_freq_stft_cutoff_bin,
    )


model_config_lookup = {
    "small_vocoder": create_small_vocoder_model,
}
