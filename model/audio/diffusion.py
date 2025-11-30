from einops import reduce
from model import megatransformer_diffusion
from typing import Optional

import megatransformer_utils

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.audio.criteria import MultiResolutionSTFTLoss
from model.audio.vocoders import AudioVocoder


class AudioDiffusionSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, d_queries, d_values, use_flash_attention=True, dropout_p=0.1, is_linear_attention=False):
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
        
        self.dropout = nn.Dropout(dropout_p)

        self._init_weights()

    def _init_weights(self):
        self.apply(megatransformer_utils.transformer_weight_init())
        
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
    def __init__(self, hidden_size, n_heads, d_queries, d_values, context_dim=None, use_flash_attention=True, dropout_p=0.1):
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

        self.dropout = nn.Dropout(dropout_p)

        self._init_weights()

    def _init_weights(self):
        self.apply(megatransformer_utils.transformer_weight_init())
    
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

class AudioConditionalGaussianDiffusion(megatransformer_diffusion.GaussianDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vocoder = AudioVocoder(
            hidden_channels=self.config.audio_vocoder_hidden_channels,
            in_channels=self.config.audio_n_mels,
            upsample_factors=self.config.audio_vocoder_upsample_factors,
            n_residual_layers=self.config.audio_vocoder_n_residual_layers,
        )

        self.mel_min = -11.5  # log(1e-5)
        self.mel_max = 5.0    # typical max for speech

        self.stft_loss = MultiResolutionSTFTLoss()
    
    def normalize(self, x_0):
        return 2 * (x_0 - self.mel_min) / (self.mel_max - self.mel_min) - 1
    
    def unnormalize(self, x):
        return (x + 1) / 2 * (self.mel_max - self.mel_min) + self.mel_min

    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, noise=None, condition=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        # noise sample

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_out = self.unet(x_noisy, t, condition)

        target = noise

        loss = F.mse_loss(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * self._extract(self.loss_weight, t, loss.shape)
        return model_out, loss.mean()

    def forward(self, x_0, condition=None, waveform_labels=None):
        if x_0.dim() == 3:
            x_0 = x_0.unsqueeze(1)

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

        t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device = x_0.device).long()

        if self.normalize:
            x_0 = self.normalize(x_0)

        mel_l1_outputs, mel_l1_loss = self.p_losses(x_0, t, condition=condition)

        if e is not None:
            # restore batch and example dimensions
            mel_l1_outputs = mel_l1_outputs.view(b, e, c, h, w)
            # leave loss alone, already means across combined batch and example dimension

        total_loss = mel_l1_loss
        waveform_l1 = None
        sc_loss = None
        mag_loss = None

        if waveform_labels is not None:
            # do vocoder forward and loss using waveform labels and clean mel specs (x_0)
            pred_waveform = self.vocoder(mel_l1_outputs, condition=condition)

            pred_waveform = F.pad(pred_waveform, (0, waveform_labels.shape[-1] - pred_waveform.shape[-1]), value=0)

            waveform_l1 = F.l1_loss(pred_waveform, waveform_labels)
            sc_loss, mag_loss = self.stft_loss(pred_waveform, waveform_labels)

            total_loss = (
                total_loss +
                1 * waveform_l1 +
                1.5 * (sc_loss + mag_loss)
            )

        return mel_l1_outputs, [total_loss, mel_l1_loss, waveform_l1, sc_loss, mag_loss], pred_waveform

    @torch.no_grad()
    def sample(self, device, batch_size: int, condition: Optional[torch.Tensor]=None, return_intermediate: bool=False, override_ddim_sampling_steps: Optional[int]=None, generator=None, **kwargs) -> torch.Tensor:
        x = torch.randn(batch_size, 1, self.config.audio_n_mels, self.config.audio_max_frames, device=device, generator=generator)

        if self.is_ddim_sampling or override_ddim_sampling_steps is not None:
            x = self.ddim_sample_loop(x, condition=condition, return_intermediate=return_intermediate, override_sampling_steps=override_ddim_sampling_steps)
        else:
            x = self.p_sample_loop(x, condition=condition, return_intermediate=return_intermediate)

        if isinstance(x, tuple):
            audio = x[0]
        else:
            audio = x

        if self.normalize:
            audio = self.unnormalize(audio)

        return (audio, *x[1:]) if return_intermediate else audio
