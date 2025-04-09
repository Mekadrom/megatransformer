from einops import reduce
from model import megatransformer_diffusion
from typing import Optional

import megatransformer_utils

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioDiffusionSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, d_queries, d_values, use_flash_attention=True, dropout=0.1):
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
        
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        self.apply(megatransformer_utils.transformer_weight_init)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normal multi-head self attention, but it expects 4D input where it will batch by the first and third dimensions,
        and outputs the same shape.
        Args:
            x: [B, H, W, T] where B is batch size, H is height, W is width and T is time.
        Returns:
            output: [B, H, W, T] where B is batch size, H is height, W is width and T is time. Attention is applied
            along the T dimension, between the W dimension values, batched along B*W.
        """
        B, H, W, T = x.shape

        x = x.permute(0, 2, 1, 3)  # [B, W, H, T]

        x = x.contiguous().view(-1, H, T)  # [B*W, H, T]

        x = x.permute(0, 2, 1)  # [B*W, T, H]
        
        q: torch.Tensor = self.q_proj(x)  # [B*W, T, n_heads*d_queries]
        k: torch.Tensor = self.k_proj(x)
        v: torch.Tensor = self.v_proj(x)
        
        q = q.view(-1, T, self.n_heads, self.d_queries)  # [B*W, T, n_heads, d_queries]
        k = k.view(-1, T, self.n_heads, self.d_queries)
        v = v.view(-1, T, self.n_heads, self.d_values)
        
        q = q.transpose(1, 2)  # [B*W, n_heads, T, d_queries]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
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
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B*W, n_heads, T, T]
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            output = torch.matmul(attn_weights, v)  # [B*W, n_heads, T, d_queries]
        
        output = output.transpose(1, 2).contiguous()  # [B*W, T, n_heads, d_queries]

        output = output.view(-1, T, self.n_heads*self.d_values)  # [B*W, T, H]
        
        output = self.out_proj(output)  # [B*W, T, H]

        output = output.permute(0, 2, 1)  # [B*W, H, T]

        # restore input shape by splitting the hidden dim into width and height
        output = output.view(B, W, H, T)

        output = output.permute(0, 2, 1, 3)  # [B, H, W, T]
        
        return output

class AudioDiffusionCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, d_queries, d_values, context_dim=None, use_flash_attention=True, dropout=0.1):
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

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        self.apply(megatransformer_utils.transformer_weight_init)
    
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
    
    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, noise=None, condition=None):
        b, c, h, w = x_start.shape
        if noise is None:
            noise = torch.randn_like(x_start)

        # noise sample

        x_start = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_out = self.unet(x_start, t, condition)

        target = noise

        loss = F.l1_loss(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * self._extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, x_0, condition=None, audio_waveform_labels=None):
        if len(x_0.shape) == 5:
            # squish batch and example dimensions if necessary
            b, e, c, h, w = x_0.shape
            x_0 = x_0.view(-1, c, h, w)
        else:
            b, c, h, w = x_0.shape
            e = None

        if condition is not None:
            if len(condition.shape) == 5:
                # squish batch and example dimensions if necessary
                *_, c, h, w = condition.shape
                condition = condition.view(-1, c, h, w)

        t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device= x_0.device).long()

        if self.normalize:
            x_0 = self.normalize_to_neg_one_to_one(x_0)

        mel_l1_outputs, mel_l1_loss = self.p_losses(x_0, t, condition=condition)

        if e is not None:
            # restore batch and example dimensions
            mel_l1_outputs = mel_l1_outputs.view(b, e, c, h, w)
            # leave loss alone, already means across combined batch and example dimension

        total_loss = mel_l1_loss

        if audio_waveform_labels is not None:
            # do vocoder forward and loss using waveform labels and clean mel specs (x_0)
            pred_waveform = self.vocoder(x_0, condition=condition)

            pred_waveform = F.pad(pred_waveform, (0, audio_waveform_labels.shape[-1] - pred_waveform.shape[-1]), value=0)

            waveform_l1 = F.l1_loss(pred_waveform, audio_waveform_labels)
            sc_loss, mag_loss = self.stft_loss(pred_waveform, audio_waveform_labels)

            total_loss = (
                total_loss +
                1 * waveform_l1 +
                1.5 * (sc_loss + mag_loss)
            )

        return mel_l1_outputs, total_loss, pred_waveform
