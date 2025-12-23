import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from einops import reduce

from model import diffusion
from utils import configuration, megatransformer_utils


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
        self._init_context_projections()

    def _init_weights(self):
        self.apply(megatransformer_utils.transformer_weight_init())

    def _init_context_projections(self):
        """
        Properly initialize K/V projections for text conditioning.

        Text embeddings (e.g., from T5) often have std ~0.2-0.3, not 1.0.
        Standard xavier init results in very weak conditioning signal.

        This ensures K/V outputs have std ≈ 1.0 for proper attention scaling.
        """
        assumed_input_std = 0.2  # Common for T5, CLIP embeddings

        fan_in_k = self.context_dim
        target_k_std = 1.0 / (assumed_input_std * math.sqrt(fan_in_k))
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=target_k_std)

        fan_in_v = self.context_dim
        target_v_std = 1.0 / (assumed_input_std * math.sqrt(fan_in_v))
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=target_v_std)

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


class AudioConditionalGaussianDiffusion(diffusion.GaussianDiffusion):
    def __init__(self, *args, **kwargs):
        # Remove shared_window_buffer from kwargs before passing to parent
        kwargs.pop('shared_window_buffer', None)
        super().__init__(*args, **kwargs)

        self.mel_min = -11.5  # log(1e-5)
        self.mel_max = 2.0    # typical max for speech (empirically ~1.6, add small margin)
    
    def normalize(self, x_0):
        return 2 * (x_0 - self.mel_min) / (self.mel_max - self.mel_min) - 1
    
    def unnormalize(self, x):
        return (x + 1) / 2 * (self.mel_max - self.mel_min) + self.mel_min

    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, noise=None, condition=None, return_diagnostics=False):

        if noise is None:
            noise = torch.randn_like(x_start)

        # Offset noise: add constant noise across spatial dimensions
        if self.offset_noise_strength > 0 and self.training:
            offset_noise = torch.randn(x_start.shape[0], x_start.shape[1], 1, 1, device=x_start.device)
            noise = noise + self.offset_noise_strength * offset_noise

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # CFG: randomly drop conditioning during training
        if condition is not None and self.cfg_dropout_prob > 0 and self.training:
            batch_size = x_start.shape[0]
            dropout_mask = torch.rand(batch_size, device=x_start.device) < self.cfg_dropout_prob
            condition = condition.clone()
            condition[dropout_mask] = 0.0

        model_out = self.unet(x_noisy, t, condition)

        if self.prediction_type == "v":
            # Model predicts v, target is v
            target = self.get_v_target(x_start, noise, t)
        else:
            # Model predicts noise (epsilon)
            target = noise

        loss = F.mse_loss(model_out, target, reduction='none')
        loss_per_sample = reduce(loss, 'b ... -> b', 'mean')

        loss_weighted = loss_per_sample * self._extract(self.loss_weight, t, loss_per_sample.shape)

        if return_diagnostics:
            diagnostics = {
                "timesteps": t,
                "loss_per_sample": loss_per_sample.detach(),
                "loss_weighted_per_sample": loss_weighted.detach(),
            }
            return model_out, loss_weighted.mean(), diagnostics

        return model_out, loss_weighted.mean()

    def forward(self, x_0: torch.Tensor, condition: Optional[torch.Tensor]=None, return_diagnostics: bool=False):
        """
        Forward pass for audio diffusion training.

        Args:
            x_0: [B, n_mels, T] or [B, 1, n_mels, T] mel spectrogram (log scale)
            condition: Optional [B, N, C] conditioning embeddings (e.g., T5 embeddings)
            return_diagnostics: If True, return additional diagnostic info for debugging

        Returns:
            predicted_noise: The noise predicted by the model
            loss: MSE loss between predicted and actual noise
            diagnostics (optional): Dict with per-timestep loss info and mel statistics
        """
        if x_0.dim() == 3:
            x_0 = x_0.unsqueeze(1)

        if len(x_0.shape) == 5:
            # squish batch and example dimensions if necessary
            b, e, c, h, w = x_0.shape
            x_0 = x_0.view(-1, c, h, w)
        else:
            b, c, h, w = x_0.shape
            e = None

        # Capture mel statistics before normalization for diagnostics
        mel_stats = None
        if return_diagnostics:
            mel_stats = {
                "mel_min": x_0.min().item(),
                "mel_max": x_0.max().item(),
                "mel_mean": x_0.mean().item(),
                "mel_std": x_0.std().item(),
            }

        if condition is not None:
            if len(condition.shape) == 5:
                # squish batch and example dimensions if necessary
                *_, c, h, w = condition.shape
                condition = condition.view(-1, c, h, w)

        t = self._sample_timesteps(x_0.shape[0], x_0.device)

        if self.is_normalize:
            x_0 = self.normalize(x_0)
            if return_diagnostics:
                mel_stats["mel_normalized_min"] = x_0.min().item()
                mel_stats["mel_normalized_max"] = x_0.max().item()
                mel_stats["mel_normalized_mean"] = x_0.mean().item()
                mel_stats["mel_normalized_std"] = x_0.std().item()

        if return_diagnostics:
            predicted_noise, loss, diagnostics = self.p_losses(x_0, t, condition=condition, return_diagnostics=True)
            diagnostics["mel_stats"] = mel_stats
        else:
            predicted_noise, loss = self.p_losses(x_0, t, condition=condition)

        if e is not None:
            # restore batch and example dimensions
            predicted_noise = predicted_noise.view(b, e, c, h, w)

        if return_diagnostics:
            return predicted_noise, loss, diagnostics
        return predicted_noise, loss

    @torch.no_grad()
    def sample(
        self,
        device,
        batch_size: int,
        condition: Optional[torch.Tensor]=None,
        return_intermediate: bool=False,
        override_sampling_steps: Optional[int]=None,
        generator=None,
        guidance_scale: float=1.0,
        sampler: str="dpm_solver_pp",
        dpm_solver_order: int=2,
        **_
    ) -> torch.Tensor:
        x = torch.randn(batch_size, 1, self.config.audio_n_mels, self.config.audio_max_frames, device=device, generator=generator)

        sampling_steps = override_sampling_steps if override_sampling_steps is not None else self.sampling_timesteps

        if sampler == "dpm_solver_pp":
            x = self.dpm_solver_pp_sample_loop(
                x, condition=condition, return_intermediate=return_intermediate,
                override_sampling_steps=sampling_steps, guidance_scale=guidance_scale,
                order=dpm_solver_order
            )
        elif sampler == "ddim" or (self.is_ddim_sampling and sampler != "ddpm"):
            x = self.ddim_sample_loop(x, condition=condition, return_intermediate=return_intermediate, override_sampling_steps=sampling_steps, guidance_scale=guidance_scale)
        else:
            x = self.p_sample_loop(x, condition=condition, return_intermediate=return_intermediate, guidance_scale=guidance_scale)

        if isinstance(x, tuple):
            audio = x[0]
            noise_preds = x[1]
            x_start_preds = x[2]
        else:
            audio = x
            noise_preds = None
            x_start_preds = None

        if self.is_normalize:
            audio = self.unnormalize(audio)
            # Also unnormalize intermediate x_start predictions
            if x_start_preds is not None:
                x_start_preds = [self.unnormalize(x_start) for x_start in x_start_preds]

        if return_intermediate:
            return audio, noise_preds, x_start_preds
        return audio


def create_unet(config: configuration.MegaTransformerConfig, context_dim: Optional[int] = None):
    return diffusion.ConvDenoisingUNet(
        activation=config.audio_decoder_activation if hasattr(config, 'audio_decoder_activation') else "silu",
        stride=(2, 2),
        self_attn_class=AudioDiffusionSelfAttentionBlock,
        cross_attn_class=AudioDiffusionCrossAttentionBlock,
        norm_class=nn.LayerNorm,
        in_channels=1,
        model_channels=config.audio_decoder_model_channels,
        out_channels=1,
        channel_multipliers=config.audio_decoder_model_channels,
        time_embedding_dim=config.audio_decoder_time_embedding_dim,
        attention_levels=config.audio_decoder_attention_levels,
        num_res_blocks=config.audio_decoder_num_res_blocks,
        has_condition=True,
        context_dim=context_dim,
        dropout_p=config.audio_decoder_unet_dropout_p,
        down_block_self_attn_n_heads=config.audio_decoder_down_block_self_attn_n_heads,
        down_block_self_attn_d_queries=config.audio_decoder_down_block_self_attn_d_queries,
        down_block_self_attn_d_values=config.audio_decoder_down_block_self_attn_d_values,
        down_block_self_attn_use_flash_attention=config.audio_decoder_down_block_self_attn_use_flash_attention,
        up_block_self_attn_n_heads=config.audio_decoder_up_block_self_attn_n_heads,
        up_block_self_attn_d_queries=config.audio_decoder_up_block_self_attn_d_queries,
        up_block_self_attn_d_values=config.audio_decoder_up_block_self_attn_d_values,
        up_block_self_attn_use_flash_attention=config.audio_decoder_up_block_self_attn_use_flash_attention,
        cross_attn_n_heads=config.audio_decoder_cross_attn_n_heads,
        cross_attn_d_queries=config.audio_decoder_cross_attn_d_queries,
        cross_attn_d_values=config.audio_decoder_cross_attn_d_values,
        cross_attn_use_flash_attention=config.audio_decoder_cross_attn_use_flash_attention,
    )


small_config = configuration.MegaTransformerConfig(
    hidden_size=192,
    audio_decoder_model_channels=80,
    audio_decoder_time_embedding_dim=192,
    audio_decoder_attention_levels=[False, True, True],
    audio_decoder_num_res_blocks=2,
    audio_decoder_down_block_self_attn_n_heads=4,
    audio_decoder_down_block_self_attn_d_queries=48,
    audio_decoder_down_block_self_attn_d_values=48,
    audio_decoder_down_block_self_attn_use_flash_attention=True,
    audio_decoder_up_block_self_attn_n_heads=4,
    audio_decoder_up_block_self_attn_d_queries=48,
    audio_decoder_up_block_self_attn_d_values=48,
    audio_decoder_up_block_self_attn_use_flash_attention=True,
    audio_decoder_cross_attn_n_heads=4,
    audio_decoder_cross_attn_d_queries=48,
    audio_decoder_cross_attn_d_values=48,
    audio_decoder_cross_attn_use_flash_attention=True,
    audio_decoder_channel_multipliers=[1, 2, 3],  # Gradual growth
)


def create_diffusion_model(
    config: configuration.MegaTransformerConfig,
    context_dim: int = 512,
    num_timesteps: int = 1000,
    sampling_timesteps: int = 50,
    betas_schedule: str = "cosine",
    normalize: bool = True,
    min_snr_loss_weight: bool = True,
    min_snr_gamma: float = 5.0,
    prediction_type: str = "epsilon",  # "epsilon" or "v"
    cfg_dropout_prob: float = 0.1,  # Default 10% dropout for CFG training
    zero_terminal_snr: bool = True,  # Recommended for audio generation
    offset_noise_strength: float = 0.0,  # Less common for audio, default off
    timestep_sampling: str = "logit_normal",  # "uniform" or "logit_normal"
    logit_normal_mean: float = 0.0,  # Mean for logit-normal (0 = centered on middle timesteps)
    logit_normal_std: float = 1.0,  # Std for logit-normal (lower = more peaked)
) -> AudioConditionalGaussianDiffusion:
    return AudioConditionalGaussianDiffusion(
        config,
        create_unet(config, context_dim),
        1,  # Single channel mel spectrogram
        num_timesteps=num_timesteps,
        betas_schedule=betas_schedule,
        min_snr_loss_weight=min_snr_loss_weight,
        min_snr_gamma=min_snr_gamma,
        normalize=normalize,
        sampling_timesteps=sampling_timesteps,
        prediction_type=prediction_type,
        cfg_dropout_prob=cfg_dropout_prob,
        zero_terminal_snr=zero_terminal_snr,
        offset_noise_strength=offset_noise_strength,
        timestep_sampling=timestep_sampling,
        logit_normal_mean=logit_normal_mean,
        logit_normal_std=logit_normal_std,
    )


model_config_lookup = {
    "small": lambda **kwargs: create_diffusion_model(
        config=small_config,
        **kwargs
    )
}
