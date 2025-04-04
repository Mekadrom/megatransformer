from model import megatransformer_diffusion
from typing import Optional

import megatransformer_utils
import torch
import torch.nn as nn
import torch.nn.functional as F


class VocoderResidualBlock(nn.Module):
    """Residual block with dilated convolutions for the vocoder."""
    def __init__(
        self, 
        channels: int, 
        dilation: int = 1, 
        kernel_size: int = 3, 
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
        
        features = F.leaky_relu(features, negative_slope=self.leaky_relu_slope)
        features = self.convs[0](features)
        
        features = F.leaky_relu(features, negative_slope=self.leaky_relu_slope)
        features = self.convs[1](features)
        
        # Gate mechanism for controlled information flow
        gate = torch.sigmoid(self.gate_conv(residual))

        gated = residual + gate * features

        return gated

class VocoderUpsampleBlock(nn.Module):
    """Upsampling block for the vocoder."""
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        upsample_factor: int,
        kernel_size: int = 8, 
        leaky_relu_slope: float = 0.1
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
        self.norm = nn.BatchNorm1d(out_channels)

        self._init_weights()

    def _init_weights(self):
        # Initialize weights for the ConvTranspose1d layer
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='leaky_relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.conv(features)
        features = self.norm(features)
        features = F.leaky_relu(features, negative_slope=self.leaky_relu_slope)
        return features


class AudioVocoder(nn.Module):
    """
    Neural vocoder that converts mel spectrograms to audio waveforms.
    
    This vocoder can be trained alongside the diffusion model and is designed
    to efficiently convert mel spectrograms to high-quality audio waveforms.
    """
    def __init__(
        self,
        hidden_channels,
        in_channels: int,  # Number of mel bands
        conditioning_channels: int,
        upsample_factors: list[int] = [8, 8, 8],  # Total upsampling of 256 (matching hop_length)
        n_residual_layers: int = 4, 
        dilation_cycle: int = 4,
        kernel_size: int = 3,
        leaky_relu_slope: float = 0.1,
        conditioning_enabled: bool = True
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
        
        # Optional conditioning layer
        if conditioning_enabled:
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
                1,  # Single channel output for waveform
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
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the vocoder.
        
        Args:
            mel_spec: [B, n_mels, T_mel] Mel spectrogram
            conditioning: Optional [B, C, T_cond] additional conditioning
                          from the language model or diffusion model
            
        Returns:
            [B, 1, T_audio] Audio waveform
        """
        if torch.isnan(mel_spec).any():
            print("NaN detected in mel_spec input to vocoder")
        if torch.isinf(mel_spec).any():
            print("Inf detected in mel_spec input to vocoder")

        # remove channel dimension
        if mel_spec.dim() == 4:
            mel_spec = mel_spec.squeeze(1)  # [B, n_mels, T_mel]

        # Initial processing
        x = self.initial_conv(mel_spec)
        
        # Upsampling
        for i, upsample in enumerate(self.upsample_blocks):
            x = upsample(x)
        
        # Apply conditioning if provided
        if self.has_condition and condition is not None:
            # Reshape conditioning to match the mel sequence length
            if condition.size(2) != x.size(2):
                condition = F.interpolate(
                    condition,
                    size=x.size(2),
                    mode='nearest'
                )
            
            cond = self.conditioning_layer(condition)
            x = x + cond
        
        # Apply residual blocks
        for i, res_block in enumerate(self.residual_blocks):
            x = res_block(x)
        
        # Final layers
        waveform = self.final_layers(x)
        waveform = waveform.squeeze(1)  # Remove channel dimension

        return waveform

# Multi-Resolution STFT Loss for better vocoder training
class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss for better audio quality.
    
    This loss computes the L1 loss between the STFT of the predicted and
    ground truth waveforms at multiple resolutions, which helps capture
    both fine and coarse time-frequency structures.
    """
    def __init__(
        self,
        fft_sizes: list[int] = [512, 1024, 2048],
        hop_sizes: list[int] = [128, 256, 512],
        win_lengths: list[int] = [512, 1024, 2048],
        window: str = "hann_window"
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        
        # Create window buffers
        self.register_buffer(
            "window",
            torch.hann_window(max(win_lengths)) if window == "hann_window" else
            torch.hamming_window(max(win_lengths))
        )
    
    def stft_magnitude(
        self, 
        x: torch.Tensor, 
        fft_size: int, 
        hop_size: int, 
        win_length: int
    ) -> torch.Tensor:
        """Calculate STFT magnitude."""

        # stft requires float32 input
        x_float32 = x.float()
        x_stft = torch.stft(
            x_float32.squeeze(1),
            fft_size,
            hop_size,
            win_length,
            self.window[:win_length].to(x_float32.dtype),
            return_complex=True
        )
        return torch.abs(x_stft).to(x.dtype)
    
    def forward(
        self, 
        pred_waveform: torch.Tensor, 
        target_waveform: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate multi-resolution STFT loss.
        
        Args:
            pred_waveform: [B, 1, T] Predicted waveform
            target_waveform: [B, 1, T] Target waveform
            
        Returns:
            Tuple of (sc_loss, mag_loss) - spectral convergence and magnitude losses
        """
        sc_loss = 0.0
        mag_loss = 0.0
        
        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            pred_mag = self.stft_magnitude(
                pred_waveform, fft_size, hop_size, win_length
            )
            target_mag = self.stft_magnitude(
                target_waveform, fft_size, hop_size, win_length
            )
            
            # Spectral convergence loss
            sc_loss += torch.norm(target_mag - pred_mag, p="fro") / (torch.norm(target_mag, p="fro") + 1e-7)
            
            # Log magnitude loss
            log_pred_mag = torch.log(pred_mag.clamp(min=1e-7))
            log_target_mag = torch.log(target_mag.clamp(min=1e-7))
            mag_loss += F.l1_loss(log_pred_mag, log_target_mag)
        
        # Normalize by number of STFT resolutions
        sc_loss = sc_loss / len(self.fft_sizes)
        mag_loss = mag_loss / len(self.fft_sizes)
        
        return sc_loss, mag_loss


class AudioGenerationLoss(nn.Module):
    """
    Combined loss function for training both diffusion model and vocoder.
    """
    def __init__(
        self,
        diffusion_loss_weight: float = 1.0,
        mel_loss_weight: float = 10.0,
        waveform_loss_weight: float = 1.0,
        stft_loss_weight: float = 2.0
    ):
        super().__init__()
        self.diffusion_loss_weight = diffusion_loss_weight
        self.mel_loss_weight = mel_loss_weight
        self.waveform_loss_weight = waveform_loss_weight
        self.stft_loss_weight = stft_loss_weight
        
        self.stft_loss = MultiResolutionSTFTLoss()
    
    def forward(
        self,
        pred_noise: torch.Tensor,
        noise: torch.Tensor,
        pred_mel: Optional[torch.Tensor] = None,
        target_mel: Optional[torch.Tensor] = None,
        pred_waveform: Optional[torch.Tensor] = None,
        target_waveform: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Calculate the combined loss.
        
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        losses = {}
        
        # Diffusion loss (noise prediction)
        diffusion_loss = F.mse_loss(pred_noise, noise)
        losses["diffusion"] = diffusion_loss
        
        total_loss = self.diffusion_loss_weight * diffusion_loss
        
        # Optional mel loss
        if pred_mel is not None and target_mel is not None:
            mel_loss = F.l1_loss(pred_mel, target_mel)
            losses["mel"] = mel_loss
            total_loss = total_loss + self.mel_loss_weight * mel_loss
        
        # Optional waveform and STFT losses
        if pred_waveform is not None and target_waveform is not None:
            # Direct waveform loss
            waveform_loss = F.l1_loss(pred_waveform, target_waveform)
            losses["waveform"] = waveform_loss
            total_loss = total_loss + self.waveform_loss_weight * waveform_loss
            
            # Multi-resolution STFT loss
            sc_loss, mag_loss = self.stft_loss(pred_waveform, target_waveform)
            losses["sc"] = sc_loss
            losses["mag"] = mag_loss
            total_loss = total_loss + self.stft_loss_weight * (sc_loss + mag_loss)
        
        return total_loss, losses

class AudioConditionalGaussianDiffusion(megatransformer_diffusion.GaussianDiffusion):
    def __init__(self, *args, **kwargs):
        self.config: megatransformer_utils.MegaTransformerConfig = kwargs.pop("config")

        super().__init__(*args, **kwargs)

        self.vocoder = AudioVocoder(
            in_channels=self.config.audio_n_mels,
            hidden_channels=self.config.audio_vocoder_hidden_channels,
            conditioning_channels=self.config.hidden_size,
            upsample_factors=self.config.audio_vocoder_upsample_factors,
            n_residual_layers=self.config.audio_vocoder_n_residual_layers,
            conditioning_enabled=True
        )

        self.stft_loss = MultiResolutionSTFTLoss()
    
    def q_sample(self, x_start, t, noise=None, condition=None):
        # Same as before, condition not needed for forward process
        return super().q_sample(x_start, t, noise)
    
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, t_index, condition=None):
        """Single step of the reverse diffusion process with conditioning"""
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x.shape)
        
        # Model forward pass with condition
        model_output = self.unet(x, t.to(x.dtype), condition=condition)
        
        if self.predict_epsilon:
            # Model predicts noise ε
            pred_epsilon = model_output
            pred_x0 = sqrt_recip_alphas_t * x - sqrt_recip_alphas_t * sqrt_one_minus_alphas_cumprod_t * pred_epsilon
            pred_x0 = torch.clamp(pred_x0, -1., 1.)
            
            # Calculate posterior mean using betas_t
            denominator = torch.sqrt(1 - self._extract(self.alphas_cumprod, t, x.shape) + 1e-8)
            posterior_mean = (
                x * (1 - betas_t) / denominator +
                pred_x0 * betas_t / denominator
            )
        else:
            # Model directly predicts x_0
            pred_x0 = model_output
            pred_x0 = torch.clamp(pred_x0, -1., 1.)
            
            # Calculate posterior mean using betas_t
            denominator = torch.sqrt(1 - self._extract(self.alphas_cumprod, t, x.shape) + 1e-8)
            posterior_mean = (
                x * (1 - betas_t) / denominator +
                pred_x0 * betas_t / denominator
            )
        
        # Calculate posterior variance using betas_t
        posterior_variance = betas_t * (1 - self._extract(self.alphas_cumprod, t-1, x.shape)) / (1 - self._extract(self.alphas_cumprod, t, x.shape))
        
        if t_index == 0:
            # No noise at the last step (t=0)
            return posterior_mean
        else:
            noise = torch.randn_like(x)
            return posterior_mean + torch.sqrt(posterior_variance) * noise
    
    @torch.no_grad()
    def sample(self, device, condition, batch_size=1, n_mels=128):
        """Sample with conditioning"""
        # Start from pure noise
        x = torch.randn(batch_size, 1, n_mels, condition.shape[-1], dtype=condition.dtype, device=device)

        # Iteratively denoise with conditioning
        for time_step in reversed(range(1, self.num_timesteps)):
            t = torch.full((batch_size,), time_step, device=device, dtype=torch.long)
            x = self.p_sample(x, t, time_step, condition=condition)
            
        # Scale to [0, 1] range
        x = (x + 1) / 2
        x = torch.clamp(x, 0.0, 1.0)
        
        return x
    
    def forward(self, x_0: torch.Tensor, condition=None, audio_waveform_labels=None):
        """Training forward pass with conditioning"""
        batch_size = x_0.shape[0]

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        
        # Model forward pass with condition
        model_output = self.unet(x_t, t.to(x_0.dtype), condition=condition)

        loss_dict = {}
        if self.predict_epsilon:
            # Loss to the added noise
            loss_dict["noise_mse"] = F.mse_loss(model_output, noise)

            pred_x0 = self.predict_start_from_noise(x_t, t, model_output)
            # megatransformer_utils.print_debug_tensor('initial pred_x0', pred_x0)
            pred_x0 = torch.clamp(pred_x0, -1., 1.)
            model_output = pred_x0

            loss_dict["mel_l1"] = F.l1_loss(pred_x0, x_0)

            B, E, M, T = pred_x0.shape

            # combine batch and element dimensions again
            pred_x0 = pred_x0.view(B*E, M, T)
            condition = condition.unsqueeze(1).expand(-1, E, -1, -1)  # [B, E, T, M]
            condition = condition.reshape(B*E, condition.shape[2], condition.shape[3])
            condition = condition.permute(0, 2, 1)  # [B*E, M, T]

            # megatransformer_utils.print_debug_tensor('condition', condition)
            # megatransformer_utils.print_debug_tensor('final pred_x0', pred_x0)

            # use mel spec label as input to vocoder instead of potentially unstable output from diffusion model
            pred_waveform = self.vocoder(x_0, condition=condition)

            pred_waveform = F.pad(pred_waveform, (0, audio_waveform_labels.shape[-1] - pred_waveform.shape[-1]), value=0)

            loss_dict["waveform_l1"] = F.l1_loss(pred_waveform, audio_waveform_labels)

            # extract example dimension
            pred_waveform = pred_waveform.view(B, E, pred_waveform.shape[-1])

            # megatransformer_utils.print_debug_tensor('pred_waveform', pred_waveform)

            # Multi-resolution STFT loss
            sc_loss, mag_loss = self.stft_loss(pred_waveform, audio_waveform_labels)
            loss_dict["sc_loss"] = sc_loss
            loss_dict["mag_loss"] = mag_loss
        else:
            pred_waveform = None
            # Loss to the original audio
            loss_dict["direct"] = F.mse_loss(model_output, x_0)

        total_loss = (
            1.0 * loss_dict.get("noise_mse", 0.0) + 
            1.0 * loss_dict.get("direct", 0.0) +
            3.0 * loss_dict.get("mel_l1", 0.0) +
            1.0 * loss_dict.get("waveform_l1", 0.0) +
            1.5 * (loss_dict.get("sc_loss", 0.0) + loss_dict.get("mag_loss", 0.0))
        )

        return model_output, total_loss, pred_waveform
