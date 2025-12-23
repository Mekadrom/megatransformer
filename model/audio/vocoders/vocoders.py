import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Optional, Literal

from model.activations import Snake
from model.audio.criteria import HighFreqSTFTLoss, MultiResolutionSTFTLoss, PhaseLoss, StableMelSpectrogramLoss
from model.audio.shared_window_buffer import SharedWindowBuffer
from model.audio.vocoders.convtranspose1d_vocoder import ConvTranspose1DVocoderUpsampleBlock
from model.audio.vocoders.freq_domain_vocoder import SplitBandLowFreqMeanFreqDomainVocoder, HeavyHeadedFrequencyDomainVocoder, LightHeadedFrequencyDomainVocoder, SplitBandFrequencyDomainVocoder
from model.audio.vocoders.upsample_vocoder import AntiAliasedUpsampleVocoderUpsampleBlock
from utils import configuration


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


class Vocoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        in_channels: int,  # Number of mel bands
        upsample_block_class,
        residual_block_class = VocoderResidualBlock,
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
                upsample_block_class(
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
                    residual_block_class(
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
                 vocoder: nn.Module,
                 shared_window_buffer: SharedWindowBuffer,
                 config: configuration.MegaTransformerConfig,
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
                 direct_mag_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.vocoder = vocoder
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
        self.direct_mag_loss_weight = direct_mag_loss_weight

        self.n_fft = config.audio_n_fft
        self.hop_length = config.audio_hop_length

        self.shared_window_buffer = shared_window_buffer


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
            self.high_freq_stft_loss = HighFreqSTFTLoss(
                shared_window_buffer=shared_window_buffer,
                n_fft=config.audio_n_fft,
                hop_length=config.audio_hop_length,
                cutoff_bin=high_freq_stft_cutoff_bin
            )
        else:
            self.high_freq_stft_loss = None

    def forward(
        self,
        mel_spec: torch.Tensor,
        waveform_labels: Optional[torch.Tensor] = None,
        target_complex_stfts: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Generate waveform through vocoder
        o = self.vocoder(mel_spec)

        if isinstance(o, tuple):
            pred_waveform = o[0]
            pred_stft = o[1]
        else:
            pred_waveform = o
            pred_stft = None

        outputs = {"pred_waveform": pred_waveform}
        if pred_stft is not None:
            outputs["pred_stft"] = pred_stft

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
                waveform_labels_aligned.unsqueeze(1) if waveform_labels_aligned.dim() == 2 else waveform_labels_aligned,
            )

            if pred_stft is not None and target_complex_stfts is not None:
                pred_mag = pred_stft.abs()
                target_mag = target_complex_stfts.abs()
                # Use 1e-5 minimum for bf16 numerical stability
                direct_mag_loss = F.l1_loss(
                    torch.log(pred_mag.clamp(min=1e-5)),
                    torch.log(target_mag.clamp(min=1e-5))
                )
            else:
                direct_mag_loss = 0.0

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
                    target_complex_stfts=target_complex_stfts,
                    precomputed_stft=pred_stft,
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
                    target_complex_stfts=target_complex_stfts,
                    precomputed_stft=pred_stft
                )
            else:
                high_freq_stft_loss_value = 0.0

            total_loss = (self.sc_loss_weight * sc_loss +
                          self.mag_loss_weight * mag_loss +
                          self.complex_stft_loss_weight * complex_stft_loss +
                          self.waveform_l1_loss_weight * waveform_l1 +
                          self.mel_recon_loss_weight * mel_recon_loss_value +
                          self.phase_loss_weight * phase_loss_value +
                          self.high_freq_stft_loss_weight * high_freq_stft_loss_value +
                          self.direct_mag_loss_weight * direct_mag_loss)

            # Debug: log all losses to catch spikes
            # megatransformer_utils.print_debug_tensor('loss_waveform_l1', waveform_l1)
            # megatransformer_utils.print_debug_tensor('loss_sc', sc_loss)
            # megatransformer_utils.print_debug_tensor('loss_mag', mag_loss)
            # megatransformer_utils.print_debug_tensor('loss_complex_stft', complex_stft_loss)
            # megatransformer_utils.print_debug_tensor('loss_mel_recon', mel_recon_loss_value)
            # megatransformer_utils.print_debug_tensor('loss_phase', phase_loss_value)
            # megatransformer_utils.print_debug_tensor('loss_high_freq_stft', high_freq_stft_loss_value)
            # megatransformer_utils.print_debug_tensor('loss_direct_mag', direct_mag_loss)
            # megatransformer_utils.print_debug_tensor('loss_total', total_loss)

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
                "direct_mag_loss": direct_mag_loss,
            })

        return outputs


really_tiny_freq_domain_config = configuration.MegaTransformerConfig(
    hidden_size=256,
    audio_n_mels=80,
    audio_n_fft=1024,
    audio_hop_length=256,
    audio_max_duration=10.0,
    audio_sample_rate=16000,
    audio_vocoder_hidden_channels=128,
    audio_vocoder_upsample_factors=[8],  # this is the convnext mult instead
    audio_vocoder_n_residual_layers=3,
)

tiny_freq_domain_config = configuration.MegaTransformerConfig(
    hidden_size=256,
    audio_n_mels=80,
    audio_n_fft=1024,
    audio_hop_length=256,
    audio_max_duration=10.0,
    audio_sample_rate=16000,
    audio_vocoder_hidden_channels=128,
    audio_vocoder_upsample_factors=[8],  # this is the convnext mult instead
    audio_vocoder_n_residual_layers=3,
)

tiny_config = configuration.MegaTransformerConfig(
    hidden_size=256,
    audio_n_mels=80,
    audio_n_fft=1024,
    audio_hop_length=256,
    audio_max_duration=10.0,
    audio_sample_rate=16000,
    audio_vocoder_hidden_channels=128,
    audio_vocoder_upsample_factors=[8, 8, 4],
    audio_vocoder_n_residual_layers=3,
)

small_config = configuration.MegaTransformerConfig(
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

def create_vocoder(
        vocoder: nn.Module,
        shared_window_buffer: SharedWindowBuffer,
        config: configuration.MegaTransformerConfig,
        **kwargs,
) -> VocoderWithLoss:
    return VocoderWithLoss(
        vocoder=vocoder,
        shared_window_buffer=shared_window_buffer,
        config=config,
        **kwargs,
    )

def create_convtranspose1d_vocoder(
        shared_window_buffer: SharedWindowBuffer,
        config: configuration.MegaTransformerConfig,
        **kwargs,
) -> VocoderWithLoss:
    return create_vocoder(
        vocoder=Vocoder(
            upsample_block_class=ConvTranspose1DVocoderUpsampleBlock,
            hidden_channels=config.audio_vocoder_hidden_channels,
            in_channels=config.audio_n_mels,
            upsample_factors=config.audio_vocoder_upsample_factors,
            n_residual_layers=config.audio_vocoder_n_residual_layers,
        ),
        shared_window_buffer=shared_window_buffer,
        config=config,
        **kwargs,
    )

def create_upsample_vocoder(
        shared_window_buffer: SharedWindowBuffer,
        config: configuration.MegaTransformerConfig,
        **kwargs,
) -> VocoderWithLoss:
    return create_vocoder(
        vocoder=Vocoder(
            upsample_block_class=AntiAliasedUpsampleVocoderUpsampleBlock,
            hidden_channels=config.audio_vocoder_hidden_channels,
            in_channels=config.audio_n_mels,
            upsample_factors=config.audio_vocoder_upsample_factors,
            n_residual_layers=config.audio_vocoder_n_residual_layers,
        ),
        shared_window_buffer=shared_window_buffer,
        config=config,
        **kwargs,
    )

def create_heavy_headed_freq_domain_vocoder(
        shared_window_buffer: SharedWindowBuffer,
        config: configuration.MegaTransformerConfig,
        **kwargs,
) -> VocoderWithLoss:
    return create_vocoder(
        vocoder=HeavyHeadedFrequencyDomainVocoder(
            shared_window_buffer,
            n_mels=config.audio_n_mels,
            n_fft=config.audio_n_fft,
            hop_length=config.audio_hop_length,
            hidden_dim=config.audio_vocoder_hidden_channels,
            num_layers=config.audio_vocoder_n_residual_layers,
            convnext_mult=config.audio_vocoder_upsample_factors[-1],  # 4 most of the time
        ),
        shared_window_buffer=shared_window_buffer,
        config=config,
        **kwargs,
    )

def create_light_headed_freq_domain_vocoder(
        shared_window_buffer: SharedWindowBuffer,
        config: configuration.MegaTransformerConfig,
        **kwargs,
) -> VocoderWithLoss:
    return create_vocoder(
        vocoder=LightHeadedFrequencyDomainVocoder(
            shared_window_buffer,
            n_mels=config.audio_n_mels,
            n_fft=config.audio_n_fft,
            hop_length=config.audio_hop_length,
            hidden_dim=config.audio_vocoder_hidden_channels,
            num_layers=config.audio_vocoder_n_residual_layers,
            convnext_mult=config.audio_vocoder_upsample_factors[0],  # 8 most of the time
        ),
        shared_window_buffer=shared_window_buffer,
        config=config,
        **kwargs,
    )


def create_split_band_freq_domain_vocoder(
        shared_window_buffer: SharedWindowBuffer,
        config: configuration.MegaTransformerConfig,
        cutoff_bin: int = 128,
        low_freq_kernel: int = 7,
        high_freq_kernel: int = 3,
        **kwargs,
) -> VocoderWithLoss:
    return create_vocoder(
        vocoder=SplitBandFrequencyDomainVocoder(
            shared_window_buffer,
            n_mels=config.audio_n_mels,
            n_fft=config.audio_n_fft,
            hop_length=config.audio_hop_length,
            hidden_dim=config.audio_vocoder_hidden_channels,
            num_layers=config.audio_vocoder_n_residual_layers,
            convnext_mult=config.audio_vocoder_upsample_factors[0],  # 8 most of the time
            cutoff_bin=cutoff_bin,
            low_freq_kernel=low_freq_kernel,
            high_freq_kernel=high_freq_kernel,
        ),
        shared_window_buffer=shared_window_buffer,
        config=config,
        **kwargs,
    )

def create_experimental_vocoder(
        shared_window_buffer: SharedWindowBuffer,
        config: configuration.MegaTransformerConfig,
        cutoff_bin: int = 128,
        low_freq_kernel: int = 7,
        high_freq_kernel: int = 3,
        **kwargs,
) -> VocoderWithLoss:
    return create_vocoder(
        vocoder=SplitBandLowFreqMeanFreqDomainVocoder(
            shared_window_buffer,
            n_mels=config.audio_n_mels,
            n_fft=config.audio_n_fft,
            hop_length=config.audio_hop_length,
            hidden_dim=config.audio_vocoder_hidden_channels,
            num_layers=config.audio_vocoder_n_residual_layers,
            convnext_mult=config.audio_vocoder_upsample_factors[0],  # 8 most of the time
            cutoff_bin=cutoff_bin,
            low_freq_kernel=low_freq_kernel,
            high_freq_kernel=high_freq_kernel,
        ),
        shared_window_buffer=shared_window_buffer,
        config=config,
        **kwargs,
    )


model_config_lookup = {
    "small_vocoder_convtranspose1d": lambda shared_window_buffer, **kwargs: create_convtranspose1d_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=small_config,
        **kwargs
    ),
    "small_vocoder_upsample": lambda shared_window_buffer, **kwargs: create_upsample_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=small_config,
        **kwargs
    ),
    "small_freq_domain_vocoder": lambda shared_window_buffer, **kwargs: create_heavy_headed_freq_domain_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=small_config,
        **kwargs
    ),
    "tiny_freq_domain_vocoder": lambda shared_window_buffer, **kwargs: create_heavy_headed_freq_domain_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=tiny_config,
        **kwargs
    ),
    "really_tiny_freq_domain_vocoder": lambda shared_window_buffer, **kwargs: create_heavy_headed_freq_domain_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=really_tiny_freq_domain_config,
        **kwargs
    ),
    "tiny_lightheaded_freq_domain_vocoder": lambda shared_window_buffer, **kwargs: create_light_headed_freq_domain_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=tiny_config,
        **kwargs
    ),
    "tiny_splitband_freq_domain_vocoder": lambda shared_window_buffer, **kwargs: create_split_band_freq_domain_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=tiny_config,
        **kwargs
    ),
    "experimental": lambda shared_window_buffer, **kwargs: create_experimental_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=tiny_config,
        **kwargs,
    )
}
