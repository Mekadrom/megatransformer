import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Optional, Literal

from model.activations import Snake
from model.audio.criteria import HighFreqSTFTLoss, MultiResolutionSTFTLoss, PhaseLoss, StableMelSpectrogramLoss
from model.audio.vocoders.convtranspose1d_vocoder import ConvTranspose1DVocoderUpsampleBlock
from model.audio.vocoders.freq_domain_vocoder import SplitBandLowFreqMeanFreqDomainVocoder, HeavyHeadedFrequencyDomainVocoder, LightHeadedFrequencyDomainVocoder, SplitBandFrequencyDomainVocoder, FrequencyDomainVocoderWithAttention
from model.audio.vocoders.upsample_vocoder import AntiAliasedUpsampleVocoderUpsampleBlock
from utils import configuration
from utils.audio_utils import SharedWindowBuffer


class HiFiGANResBlock(nn.Module):
    """
    HiFi-GAN ResBlock1 implementation.

    Each block iterates through dilations, applying:
    - LeakyReLU/Snake activation
    - Dilated convolution
    - LeakyReLU/Snake activation
    - Non-dilated convolution (dilation=1)
    - Residual connection

    Uses weight normalization as in the original HiFi-GAN paper.
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 3, 5),
        activation_fn: Literal["leaky_relu", "snake"] = 'leaky_relu',
        snake_alpha_init: float = 1.0,
        leaky_relu_slope: float = 0.1
    ):
        super().__init__()
        self.activation_fn = activation_fn
        self.leaky_relu_slope = leaky_relu_slope

        # Dilated convolutions (one per dilation value)
        self.convs1 = nn.ModuleList([
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(
                    channels, channels, kernel_size,
                    dilation=d,
                    padding=d * (kernel_size - 1) // 2
                )
            )
            for d in dilations
        ])

        # Non-dilated convolutions (dilation=1 always, one per dilation value)
        self.convs2 = nn.ModuleList([
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(
                    channels, channels, kernel_size,
                    dilation=1,
                    padding=(kernel_size - 1) // 2
                )
            )
            for _ in dilations
        ])

        # Create activation functions
        if activation_fn == 'snake':
            # For Snake, we need separate instances for each position
            self.acts1 = nn.ModuleList([Snake(channels, snake_alpha_init) for _ in dilations])
            self.acts2 = nn.ModuleList([Snake(channels, snake_alpha_init) for _ in dilations])
        else:
            # LeakyReLU can be shared (stateless)
            self.acts1 = nn.ModuleList([nn.LeakyReLU(leaky_relu_slope) for _ in dilations])
            self.acts2 = nn.ModuleList([nn.LeakyReLU(leaky_relu_slope) for _ in dilations])

        self._init_weights()

    def _init_weights(self):
        # With weight_norm parametrization, we must init the underlying weight tensor
        # which is stored in conv.parametrizations.weight.original0
        for conv in self.convs1:
            weight = conv.parametrizations.weight.original0 if hasattr(conv, 'parametrizations') else conv.weight
            if self.activation_fn == 'snake':
                nn.init.xavier_uniform_(weight, gain=1.0)
            else:
                nn.init.kaiming_normal_(weight, a=self.leaky_relu_slope, mode='fan_in', nonlinearity='leaky_relu')
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

        for conv in self.convs2:
            weight = conv.parametrizations.weight.original0 if hasattr(conv, 'parametrizations') else conv.weight
            if self.activation_fn == 'snake':
                nn.init.xavier_uniform_(weight, gain=1.0)
            else:
                nn.init.kaiming_normal_(weight, a=self.leaky_relu_slope, mode='fan_in', nonlinearity='leaky_relu')
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResBlock.

        For each dilation:
            xt = act(x)
            xt = dilated_conv(xt)
            xt = act(xt)
            xt = non_dilated_conv(xt)
            x = x + xt
        """
        for c1, c2, act1, act2 in zip(self.convs1, self.convs2, self.acts1, self.acts2):
            xt = act1(x)
            xt = c1(xt)
            xt = act2(xt)
            xt = c2(xt)
            x = x + xt
        return x

    def remove_weight_norm(self):
        """Remove weight normalization for inference optimization."""
        for conv in self.convs1:
            nn.utils.parametrize.remove_parametrizations(conv, 'weight')
        for conv in self.convs2:
            nn.utils.parametrize.remove_parametrizations(conv, 'weight')


# Keep old class name as alias for backwards compatibility
VocoderResidualBlock = HiFiGANResBlock


class MRFResidualStack(nn.Module):
    """
    Single residual stack for one kernel size in MRF block.

    This is now just a thin wrapper around HiFiGANResBlock,
    since HiFiGANResBlock handles all dilations internally.
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: list[int] = [1, 3, 5],
        activation_fn: Literal["leaky_relu", "snake"] = 'snake',
        snake_alpha_init: float = 1.0,
        leaky_relu_slope: float = 0.1,
    ):
        super().__init__()
        self.block = HiFiGANResBlock(
            channels=channels,
            kernel_size=kernel_size,
            dilations=tuple(dilations),
            activation_fn=activation_fn,
            snake_alpha_init=snake_alpha_init,
            leaky_relu_slope=leaky_relu_slope,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

    def remove_weight_norm(self):
        """Remove weight normalization for inference optimization."""
        self.block.remove_weight_norm()


class MultiReceptiveFieldBlock(nn.Module):
    """
    Multi-Receptive Field (MRF) block from HiFi-GAN.

    Runs parallel residual stacks with different kernel sizes to capture
    patterns at multiple scales, then averages the outputs.

    This significantly improves audio quality by allowing the model to
    learn both fine-grained and coarse patterns simultaneously.
    """
    def __init__(
        self,
        channels: int,
        kernel_sizes: list[int] = [3, 7, 11],
        dilations: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        activation_fn: Literal["leaky_relu", "snake"] = 'snake',
        snake_alpha_init: float = 1.0,
        leaky_relu_slope: float = 0.1,
    ):
        super().__init__()

        assert len(kernel_sizes) == len(dilations), \
            f"kernel_sizes ({len(kernel_sizes)}) must match dilations ({len(dilations)})"

        self.n_stacks = len(kernel_sizes)

        self.stacks = nn.ModuleList([
            MRFResidualStack(
                channels=channels,
                kernel_size=k,
                dilations=d,
                activation_fn=activation_fn,
                snake_alpha_init=snake_alpha_init,
                leaky_relu_slope=leaky_relu_slope,
            )
            for k, d in zip(kernel_sizes, dilations)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass averaging outputs from all kernel size stacks.

        Args:
            x: [B, C, T] input tensor

        Returns:
            [B, C, T] averaged output from all stacks
        """
        # Run all stacks in parallel and average
        outputs = [stack(x) for stack in self.stacks]
        return sum(outputs) / self.n_stacks

    def remove_weight_norm(self):
        """Remove weight normalization for inference optimization."""
        for stack in self.stacks:
            stack.remove_weight_norm()


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


class MRFVocoder(nn.Module):
    """
    HiFi-GAN style vocoder with Multi-Receptive Field (MRF) blocks.

    Uses parallel residual stacks with different kernel sizes at each scale
    to capture patterns at multiple receptive field sizes simultaneously.
    This significantly improves audio quality compared to single-kernel vocoders.
    """
    def __init__(
        self,
        hidden_channels: int,
        in_channels: int,  # Number of mel bands
        upsample_block_class,
        upsample_factors: list[int] = [8, 8, 4],
        upsample_kernel_sizes: list[int] = [16, 16, 8],
        mrf_kernel_sizes: list[int] = [3, 7, 11],
        mrf_dilations: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        activation_fn: Literal["leaky_relu", "snake"] = 'snake',
        leaky_relu_slope: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.leaky_relu_slope = leaky_relu_slope

        # Initial convolution with larger kernel for better initial receptive field
        self.initial_conv = nn.Conv1d(
            in_channels,
            hidden_channels,
            kernel_size=7,
            padding=3
        )

        current_channels = hidden_channels

        # Upsampling blocks with MRF
        self.upsample_blocks = nn.ModuleList()
        self.mrf_blocks = nn.ModuleList()

        for factor, kernel_size in zip(upsample_factors, upsample_kernel_sizes):
            upsample_block = None
            if upsample_block_class == AntiAliasedUpsampleVocoderUpsampleBlock:
                upsample_block = upsample_block_class(
                    current_channels,
                    current_channels // 2,
                    upsample_factor=factor,
                    kernel_size=kernel_size,
                )
            else:
                upsample_block = upsample_block_class(
                    current_channels,
                    current_channels // 2,
                    upsample_factor=factor,
                    kernel_size=kernel_size,
                    leaky_relu_slope=leaky_relu_slope
                )
            self.upsample_blocks.append(upsample_block)
            current_channels //= 2

            # MRF block at this scale
            self.mrf_blocks.append(
                MultiReceptiveFieldBlock(
                    channels=current_channels,
                    kernel_sizes=mrf_kernel_sizes,
                    dilations=mrf_dilations,
                    activation_fn=activation_fn,
                    leaky_relu_slope=leaky_relu_slope,
                )
            )

        # Final output layers
        self.final_layers = nn.Sequential(
            nn.LeakyReLU(leaky_relu_slope),
            nn.Conv1d(
                current_channels,
                current_channels,
                kernel_size=7,  # Larger kernel for final smoothing
                padding=3
            ),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Conv1d(
                current_channels,
                1,  # Single channel output for mono waveform
                kernel_size=7,
                padding=3,
                bias=False
            ),
            nn.Tanh()  # Output in range [-1, 1]
        )

        self._init_weights()

    def _init_weights(self):
        # Initial conv
        nn.init.kaiming_normal_(self.initial_conv.weight, a=self.leaky_relu_slope, mode='fan_in', nonlinearity='leaky_relu')
        if self.initial_conv.bias is not None:
            nn.init.zeros_(self.initial_conv.bias)

        # Final layers
        for i, layer in enumerate(self.final_layers):
            if isinstance(layer, nn.Conv1d):
                if i == len(self.final_layers) - 2:  # Last conv before Tanh
                    nn.init.normal_(layer.weight, mean=0.0, std=0.001)
                else:
                    nn.init.kaiming_normal_(layer.weight, a=self.leaky_relu_slope, mode='fan_in', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MRF vocoder.

        Args:
            mel_spec: [B, n_mels, T_mel] or [B, 1, n_mels, T_mel] Mel spectrogram

        Returns:
            [B, T_audio] Audio waveform
        """
        # Remove channel dimension if present
        if mel_spec.dim() == 4:
            mel_spec = mel_spec.squeeze(1)

        # Initial processing
        x = self.initial_conv(mel_spec)

        # Upsample + MRF at each scale
        for upsample, mrf in zip(self.upsample_blocks, self.mrf_blocks):
            x = upsample(x)
            x = mrf(x)

        # Final layers
        waveform = self.final_layers(x)
        waveform = waveform.squeeze(1)

        return waveform

    def remove_weight_norm(self):
        """Remove weight normalization for inference optimization."""
        for mrf in self.mrf_blocks:
            mrf.remove_weight_norm()


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


# =============================================================================
# Config factory functions - create configs with customizable audio parameters
# =============================================================================

def _create_config(
    base_config: dict,
    audio_n_fft: int = 1024,
    audio_hop_length: int = 256,
    audio_sample_rate: int = 16000,
    audio_n_mels: int = 80,
) -> configuration.MegaTransformerConfig:
    """Create a config with custom audio parameters."""
    config_dict = base_config.copy()
    config_dict["audio_n_fft"] = audio_n_fft
    config_dict["audio_hop_length"] = audio_hop_length
    config_dict["audio_sample_rate"] = audio_sample_rate
    config_dict["audio_n_mels"] = audio_n_mels
    return configuration.MegaTransformerConfig(**config_dict)


# Base config templates (without audio params that will be overridden)
_really_tiny_freq_domain_base = {
    "hidden_size": 256,
    "audio_n_mels": 80,
    "audio_n_fft": 1024,
    "audio_hop_length": 256,
    "audio_max_duration": 10.0,
    "audio_sample_rate": 16000,
    "audio_vocoder_hidden_channels": 128,
    "audio_vocoder_upsample_factors": [8],  # this is the convnext mult instead
    "audio_vocoder_n_residual_layers": 3,
}

_tiny_freq_domain_base = {
    "hidden_size": 256,
    "audio_n_mels": 80,
    "audio_n_fft": 1024,
    "audio_hop_length": 256,
    "audio_max_duration": 10.0,
    "audio_sample_rate": 16000,
    "audio_vocoder_hidden_channels": 128,
    "audio_vocoder_upsample_factors": [8],  # this is the convnext mult instead
    "audio_vocoder_n_residual_layers": 3,
}

_debug_base = {
    "hidden_size": 4,
    "audio_n_mels": 80,
    "audio_n_fft": 1024,
    "audio_hop_length": 256,
    "audio_max_duration": 10.0,
    "audio_sample_rate": 16000,
    "audio_vocoder_hidden_channels": 4,
    "audio_vocoder_upsample_factors": [8, 8, 4],
    "audio_vocoder_n_residual_layers": 1,
}

_micro_base = {
    "hidden_size": 256,
    "audio_n_mels": 80,
    "audio_n_fft": 1024,
    "audio_hop_length": 256,
    "audio_max_duration": 10.0,
    "audio_sample_rate": 16000,
    "audio_vocoder_hidden_channels": 64,
    "audio_vocoder_upsample_factors": [8, 8, 4],
    "audio_vocoder_n_residual_layers": 2,
}

_tiny_base = {
    "audio_n_mels": 80,
    "audio_n_fft": 1024,
    "audio_hop_length": 256,
    "audio_max_duration": 10.0,
    "audio_sample_rate": 16000,
    "audio_vocoder_hidden_channels": 128,
    "audio_vocoder_upsample_factors": [8, 8, 4],
    "audio_vocoder_n_residual_layers": 3,
}

_small_base = {
    "audio_n_mels": 80,
    "audio_n_fft": 1024,
    "audio_hop_length": 256,
    "audio_max_duration": 10.0,
    "audio_sample_rate": 16000,
    "audio_vocoder_hidden_channels": 128,
    "audio_vocoder_upsample_factors": [8, 8, 4],
    "audio_vocoder_n_residual_layers": 3,
}

_medium_base = {
    "audio_n_mels": 80,
    "audio_n_fft": 1024,
    "audio_hop_length": 256,
    "audio_max_duration": 10.0,
    "audio_sample_rate": 16000,
    "audio_vocoder_hidden_channels": 192,
    "audio_vocoder_upsample_factors": [8, 8, 4],
    "audio_vocoder_n_residual_layers": 4,
}

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


def create_attention_freq_domain_vocoder(
        shared_window_buffer: SharedWindowBuffer,
        config: configuration.MegaTransformerConfig,
        num_conv_layers: int = 6,
        num_attn_layers: int = 2,
        attn_heads: int = 4,
        cutoff_bin: int = 128,
        use_gradient_checkpointing: bool = False,
        **kwargs,
) -> VocoderWithLoss:
    """Create frequency-domain vocoder with attention for phase coherence."""
    return create_vocoder(
        vocoder=FrequencyDomainVocoderWithAttention(
            shared_window_buffer,
            n_mels=config.audio_n_mels,
            n_fft=config.audio_n_fft,
            hop_length=config.audio_hop_length,
            hidden_dim=config.audio_vocoder_hidden_channels,
            num_conv_layers=num_conv_layers,
            num_attn_layers=num_attn_layers,
            attn_heads=attn_heads,
            convnext_mult=config.audio_vocoder_upsample_factors[0],
            cutoff_bin=cutoff_bin,
            use_gradient_checkpointing=use_gradient_checkpointing,
        ),
        shared_window_buffer=shared_window_buffer,
        config=config,
        **kwargs,
    )


def create_mrf_vocoder(
        shared_window_buffer: SharedWindowBuffer,
        config: configuration.MegaTransformerConfig,
        mrf_kernel_sizes: list[int] = [3, 7, 11],
        mrf_dilations: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        use_gradient_checkpointing: bool = False,
        **kwargs,
) -> VocoderWithLoss:
    """Create MRF vocoder with ConvTranspose1D upsampling."""
    return create_vocoder(
        vocoder=MRFVocoder(
            hidden_channels=config.audio_vocoder_hidden_channels,
            in_channels=config.audio_n_mels,
            upsample_block_class=ConvTranspose1DVocoderUpsampleBlock,
            upsample_factors=config.audio_vocoder_upsample_factors,
            mrf_kernel_sizes=mrf_kernel_sizes,
            mrf_dilations=mrf_dilations,
        ),
        shared_window_buffer=shared_window_buffer,
        config=config,
        **kwargs,
    )


def create_mrf_upsample_vocoder(
        shared_window_buffer: SharedWindowBuffer,
        config: configuration.MegaTransformerConfig,
        upsample_kernel_sizes: list[int] = [16, 16, 8],
        mrf_kernel_sizes: list[int] = [3, 7, 11],
        mrf_dilations: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        use_gradient_checkpointing: bool = False,
        **kwargs,
) -> VocoderWithLoss:
    """Create MRF vocoder with anti-aliased upsampling."""
    return create_vocoder(
        vocoder=MRFVocoder(
            hidden_channels=config.audio_vocoder_hidden_channels,
            in_channels=config.audio_n_mels,
            upsample_block_class=AntiAliasedUpsampleVocoderUpsampleBlock,
            upsample_factors=config.audio_vocoder_upsample_factors,
            upsample_kernel_sizes=upsample_kernel_sizes,
            mrf_kernel_sizes=mrf_kernel_sizes,
            mrf_dilations=mrf_dilations,
        ),
        shared_window_buffer=shared_window_buffer,
        config=config,
        **kwargs,
    )


# MRF-specific config bases
_mrf_small_base = {
    "audio_n_mels": 80,
    "audio_n_fft": 1024,
    "audio_hop_length": 256,
    "audio_max_duration": 10.0,
    "audio_sample_rate": 16000,
    "audio_vocoder_hidden_channels": 512,  # HiFi-GAN V1 uses 512
    "audio_vocoder_upsample_factors": [8, 8, 4],
    "audio_vocoder_n_residual_layers": 3,  # 3 dilations per MRF stack
}

_mrf_medium_base = {
    "audio_n_mels": 80,
    "audio_n_fft": 1024,
    "audio_hop_length": 256,
    "audio_max_duration": 10.0,
    "audio_sample_rate": 16000,
    "audio_vocoder_hidden_channels": 288,  # Smaller for faster training
    "audio_vocoder_upsample_factors": [8, 8, 2, 2],
    "audio_vocoder_upsample_kernel_sizes": [16, 16, 4, 4],
}

def _get_config(base_config: dict, audio_n_fft: int = None, audio_hop_length: int = None,
                audio_sample_rate: int = None, audio_n_mels: int = None) -> configuration.MegaTransformerConfig:
    """Get config with optional audio parameter overrides."""
    if audio_n_fft is None and audio_hop_length is None and audio_sample_rate is None and audio_n_mels is None:
        # Use defaults from base config
        return _create_config(base_config)
    return _create_config(
        base_config,
        audio_n_fft=audio_n_fft if audio_n_fft is not None else base_config.get("audio_n_fft", 1024),
        audio_hop_length=audio_hop_length if audio_hop_length is not None else base_config.get("audio_hop_length", 256),
        audio_sample_rate=audio_sample_rate if audio_sample_rate is not None else base_config.get("audio_sample_rate", 16000),
        audio_n_mels=audio_n_mels if audio_n_mels is not None else base_config.get("audio_n_mels", 80),
    )


model_config_lookup = {
    # Legacy single-kernel vocoders
    "small_vocoder_convtranspose1d": lambda shared_window_buffer, audio_n_fft=None, audio_hop_length=None, audio_sample_rate=None, audio_n_mels=None, **kwargs: create_convtranspose1d_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=_get_config(_small_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels),
        **kwargs
    ),
    "small_vocoder_upsample": lambda shared_window_buffer, audio_n_fft=None, audio_hop_length=None, audio_sample_rate=None, audio_n_mels=None, **kwargs: create_upsample_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=_get_config(_small_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels),
        **kwargs
    ),

    # MRF (Multi-Receptive Field) vocoders - HiFi-GAN style
    "mrf_small": lambda shared_window_buffer, audio_n_fft=None, audio_hop_length=None, audio_sample_rate=None, audio_n_mels=None, **kwargs: create_mrf_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=_get_config(_mrf_small_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels),
        **kwargs
    ),
    "mrf_medium": lambda shared_window_buffer, audio_n_fft=None, audio_hop_length=None, audio_sample_rate=None, audio_n_mels=None, **kwargs: create_mrf_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=_get_config(_mrf_medium_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels),
        **kwargs
    ),
    "mrf_small_upsample": lambda shared_window_buffer, audio_n_fft=None, audio_hop_length=None, audio_sample_rate=None, audio_n_mels=None, **kwargs: create_mrf_upsample_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=_get_config(_mrf_small_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels),
        **kwargs
    ),
    "mrf_medium_upsample": lambda shared_window_buffer, audio_n_fft=None, audio_hop_length=None, audio_sample_rate=None, audio_n_mels=None, **kwargs: create_mrf_upsample_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=_get_config(_mrf_medium_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels),
        **kwargs
    ),

    # Frequency domain vocoders
    "small_freq_domain_vocoder": lambda shared_window_buffer, audio_n_fft=None, audio_hop_length=None, audio_sample_rate=None, audio_n_mels=None, **kwargs: create_heavy_headed_freq_domain_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=_get_config(_small_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels),
        **kwargs
    ),
    "tiny_freq_domain_vocoder": lambda shared_window_buffer, audio_n_fft=None, audio_hop_length=None, audio_sample_rate=None, audio_n_mels=None, **kwargs: create_heavy_headed_freq_domain_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=_get_config(_tiny_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels),
        **kwargs
    ),
    "really_tiny_freq_domain_vocoder": lambda shared_window_buffer, audio_n_fft=None, audio_hop_length=None, audio_sample_rate=None, audio_n_mels=None, **kwargs: create_heavy_headed_freq_domain_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=_get_config(_really_tiny_freq_domain_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels),
        **kwargs
    ),
    "tiny_lightheaded_freq_domain_vocoder": lambda shared_window_buffer, audio_n_fft=None, audio_hop_length=None, audio_sample_rate=None, audio_n_mels=None, **kwargs: create_light_headed_freq_domain_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=_get_config(_tiny_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels),
        **kwargs
    ),
    "tiny_splitband_freq_domain_vocoder": lambda shared_window_buffer, audio_n_fft=None, audio_hop_length=None, audio_sample_rate=None, audio_n_mels=None, **kwargs: create_split_band_freq_domain_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=_get_config(_tiny_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels),
        **kwargs
    ),
    "experimental": lambda shared_window_buffer, audio_n_fft=None, audio_hop_length=None, audio_sample_rate=None, audio_n_mels=None, **kwargs: create_experimental_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=_get_config(_tiny_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels),
        **kwargs,
    ),

    # Attention-based frequency domain vocoders
    "debug_attention_config": lambda shared_window_buffer, audio_n_fft=None, audio_hop_length=None, audio_sample_rate=None, audio_n_mels=None, use_gradient_checkpointing=False, **kwargs: create_attention_freq_domain_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=_get_config(_debug_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels),
        use_gradient_checkpointing=use_gradient_checkpointing,
        **kwargs,
    ),
    "micro_attention_freq_domain_vocoder": lambda shared_window_buffer, audio_n_fft=None, audio_hop_length=None, audio_sample_rate=None, audio_n_mels=None, use_gradient_checkpointing=False, **kwargs: create_attention_freq_domain_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=_get_config(_micro_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels),
        use_gradient_checkpointing=use_gradient_checkpointing,
        **kwargs,
    ),
    "tiny_attention_freq_domain_vocoder": lambda shared_window_buffer, audio_n_fft=None, audio_hop_length=None, audio_sample_rate=None, audio_n_mels=None, use_gradient_checkpointing=False, **kwargs: create_attention_freq_domain_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=_get_config(_tiny_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels),
        use_gradient_checkpointing=use_gradient_checkpointing,
        **kwargs,
    ),
    "small_attention_freq_domain_vocoder": lambda shared_window_buffer, audio_n_fft=None, audio_hop_length=None, audio_sample_rate=None, audio_n_mels=None, use_gradient_checkpointing=False, **kwargs: create_attention_freq_domain_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=_get_config(_small_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels),
        use_gradient_checkpointing=use_gradient_checkpointing,
        **kwargs,
    ),
    "medium_attention_freq_domain_vocoder": lambda shared_window_buffer, audio_n_fft=None, audio_hop_length=None, audio_sample_rate=None, audio_n_mels=None, use_gradient_checkpointing=False, **kwargs: create_attention_freq_domain_vocoder(
        shared_window_buffer=shared_window_buffer,
        config=_get_config(_medium_base, audio_n_fft, audio_hop_length, audio_sample_rate, audio_n_mels),
        use_gradient_checkpointing=use_gradient_checkpointing,
        num_conv_layers=6,
        num_attn_layers=2,
        attn_heads=4,
        **kwargs,
    )
}
