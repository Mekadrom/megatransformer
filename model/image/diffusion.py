import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from model import diffusion
from model.diffusion import GaussianDiffusion
from utils import configuration
from utils.megatransformer_utils import transformer_weight_init


class ImageRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C]
        x = x.permute(0, 3, 1, 2).contiguous()  # to [B, C, H, W]
        x_norm = F.normalize(x, dim=1)
        shift: torch.Tensor = x.shape[-1] ** 0.5
        norm = x_norm * self.g * shift
        norm = norm.permute(0, 2, 3, 1).contiguous()  # back to [B, H, W, C]
        return norm


class ImageDiffusionSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, d_queries, d_values, use_flash_attention=True, dropout_p=0.1, is_linear_attention=False):
        super().__init__()
        self.hidden_dim = hidden_size
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.use_flash_attention = use_flash_attention
        self.dropout_p = dropout_p
        self.is_linear_attention = is_linear_attention

        self.q_proj = nn.Conv2d(hidden_size, d_queries * n_heads, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(hidden_size, d_queries * n_heads, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(hidden_size, d_values * n_heads, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(self.d_values * n_heads, hidden_size, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout_p)

        self._init_weights()

    def _init_weights(self):
        self.apply(transformer_weight_init())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-head self attention for image data, treating spatial dimensions (H,W) as the sequence length.
        Args:
            x: [B, C, H, W] where B is batch size, C is channels, H is height, W is width.
        Returns:
            output: [B, C, H, W] where attention is applied along the spatial dimensions.
        """
        B, C, H, W = x.shape

        q: torch.Tensor = self.q_proj(x)  # [B, n_heads*d_queries, H, W]
        k: torch.Tensor = self.k_proj(x)  # [B, n_heads*d_queries, H, W]
        v: torch.Tensor = self.v_proj(x)  # [B, n_heads*d_values, H, W]
        
        q = q.view(B, self.n_heads, self.d_queries, -1).transpose(-2, -1)  # [B, n_heads, H*W, d_queries]
        k = k.view(B, self.n_heads, self.d_queries, -1).transpose(-2, -1)  # [B, n_heads, H*W, d_queries]
        v = v.view(B, self.n_heads, self.d_values, -1).transpose(-2, -1)  # [B, n_heads, H*W, d_values]

        if self.is_linear_attention:
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-2)
        
        output: torch.Tensor
        if self.use_flash_attention:
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False
            )  # [B, n_heads, H*W, d_values]
        else:
            if self.is_linear_attention:
                # Linear attention: (Q^T)·(K·V) - more efficient for long sequences
                kv = torch.matmul(k.transpose(-2, -1), v)  # [B, n_heads, d_queries, d_values]
                output = torch.matmul(q, kv)  # [B, n_heads, seq_len, d_values]
            else:
                # Standard dot-product attention
                scale = 1.0 / math.sqrt(self.d_queries)
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, n_heads, seq_len, seq_len]
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                output = torch.matmul(attn_weights, v)  # [B, n_heads, seq_len, d_values]
        
        # Reshape back to image format
        output = output.transpose(2, 3).contiguous()  # [B, n_heads, d_values, seq_len]
        output = output.view(B, self.n_heads * self.d_values, H, W)  # [B, n_heads*d_values, H, W]
        output = self.out_proj(output)  # [B, C, H, W]

        return output


class ImageDiffusionCrossAttentionBlock(nn.Module):
    """
    Simpler cross-attention block for image data that works with any spatial size.
    Uses 1x1 convolutions for query projection (instead of large patch embeddings).
    This is the standard approach used in UNet-based diffusion models like Stable Diffusion.
    """
    def __init__(self, hidden_size, n_heads=8, d_queries=64, d_values=64, context_dim=None, use_flash_attention=True, dropout_p=0.1, is_linear_attention=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.context_dim = context_dim or hidden_size
        self.use_flash_attention = use_flash_attention
        self.dropout_p = dropout_p
        self.is_linear_attention = is_linear_attention

        # pointwise conv for query (works with any spatial size)
        self.q_proj = nn.Conv2d(hidden_size, n_heads * d_queries, kernel_size=1, bias=False)
        # linear projections for context (key/value)
        self.k_proj = nn.Linear(self.context_dim, n_heads * d_queries, bias=False)
        self.v_proj = nn.Linear(self.context_dim, n_heads * d_values, bias=False)
        # another pointwise conv for output projection
        self.out_proj = nn.Conv2d(n_heads * d_values, hidden_size, kernel_size=1)

        self.dropout = nn.Dropout(dropout_p)

        self._init_weights()

    def _init_weights(self):
        self.apply(transformer_weight_init())
        self._init_context_projections()

    def _init_context_projections(self):
        """
        Properly initialize K/V projections for text conditioning.

        Text embeddings (e.g., from T5) often have std ~0.2-0.3, not 1.0.
        Standard xavier init with gain=0.02 results in output_std ≈ 0.005,
        which makes the conditioning signal negligible.

        This initialization ensures the K/V projections produce outputs
        with std ≈ 1.0 regardless of input embedding statistics.
        Target: output_std ≈ 1.0 / sqrt(d_queries) for proper attention scaling.
        """
        # For scaled dot-product attention, we want Q·K^T / sqrt(d_k) to have
        # reasonable variance. With input_std ≈ 0.2 and fan_in = context_dim:
        # output_std = input_std * weight_std * sqrt(fan_in)
        #
        # To get output_std ≈ 1.0:
        # weight_std = 1.0 / (input_std * sqrt(fan_in))
        #
        # Assuming text embeddings have std ≈ 0.2 (common for T5, CLIP, etc.):
        assumed_input_std = 0.2

        # For k_proj: want output_std ≈ 1.0
        fan_in_k = self.context_dim
        target_k_std = 1.0 / (assumed_input_std * math.sqrt(fan_in_k))
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=target_k_std)

        # For v_proj: same reasoning
        fan_in_v = self.context_dim
        target_v_std = 1.0 / (assumed_input_std * math.sqrt(fan_in_v))
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=target_v_std)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Cross-attention between image features and context (e.g., text embeddings).

        Args:
            x: Image features [B, C, H, W]
            context: Context embeddings [B, T, D] (e.g., text embeddings)

        Returns:
            output: [B, C, H, W] with cross-attended features
        """
        B, C, H, W = x.size()
        BC, T, CC = context.size()

        assert B == BC, f"Batch size mismatch: {B} vs {BC}"

        # Query from image features
        q: torch.Tensor = self.q_proj(x)  # [B, n_heads*d_queries, H, W]
        q = q.view(B, self.n_heads, self.d_queries, -1).transpose(-2, -1)  # [B, n_heads, H*W, d_queries]

        # Key and value from context
        k: torch.Tensor = self.k_proj(context)  # [B, T, n_heads*d_queries]
        v: torch.Tensor = self.v_proj(context)  # [B, T, n_heads*d_values]

        k = k.view(B, T, self.n_heads, self.d_queries).transpose(1, 2)  # [B, n_heads, T, d_queries]
        v = v.view(B, T, self.n_heads, self.d_values).transpose(1, 2)   # [B, n_heads, T, d_values]

        output: torch.Tensor
        if self.use_flash_attention:
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False
            )  # [B, n_heads, H*W, d_values]
        else:
            if self.is_linear_attention:
                # Linear attention
                kv = torch.matmul(k.transpose(-2, -1), v)  # [B, n_heads, d_queries, d_values]
                output = torch.matmul(q, kv)  # [B, n_heads, H*W, d_values]
            else:
                # Standard attention
                scale = 1.0 / math.sqrt(self.d_queries)
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, n_heads, H*W, T]
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                output = torch.matmul(attn_weights, v)  # [B, n_heads, H*W, d_values]

        # Reshape back to image format
        output = output.transpose(2, 3).contiguous()  # [B, n_heads, d_values, H*W]
        output = output.view(B, self.n_heads * self.d_values, H, W)  # [B, n_heads*d_values, H, W]
        output = self.out_proj(output)  # [B, C, H, W]

        return output


class ImageConditionalGaussianDiffusion(GaussianDiffusion):
    """
    Image diffusion model for text-to-image generation.
    Operates on VAE latent space with text conditioning via cross-attention.
    """

    def __init__(
        self,
        config: configuration.MegaTransformerConfig,
        *args,
        **kwargs
    ):
        super().__init__(config, *args, **kwargs)

    def forward(self, x_0, condition=None, return_diagnostics=False):
        """
        Forward pass for image diffusion training.

        Args:
            x_0: Input latent tensor [B, C, H, W]
            condition: Text embeddings [B, T, D] for cross-attention
            return_diagnostics: If True, return additional diagnostics dict

        Returns:
            predicted_noise: Model's noise prediction
            loss: Diffusion loss
            diagnostics: (optional) Dict with training diagnostics
        """
        model_output, losses = super().forward(x_0, condition=condition)

        if return_diagnostics:
            diagnostics = {
                "latent_stats": {
                    "latent_min": x_0.min().item(),
                    "latent_max": x_0.max().item(),
                    "latent_mean": x_0.mean().item(),
                    "latent_std": x_0.std().item(),
                }
            }
            return model_output, losses[0], diagnostics

        return model_output, losses[0]

    def sample(self, device, batch_size, condition=None, return_intermediate=False, override_sampling_steps=None, guidance_scale=1.0, sampler="dpm_solver_pp", dpm_solver_order=2, **kwargs):
        """
        Sample from the diffusion model.

        Args:
            device: Device to run on
            batch_size: Number of samples to generate
            condition: Text embeddings for conditioning
            return_intermediate: If True, return intermediate denoising steps
            override_sampling_steps: Override sampling steps
            guidance_scale: Classifier-free guidance scale. 1.0 = no guidance, >1.0 = stronger conditioning.
            sampler: Sampling algorithm ("dpm_solver_pp", "ddim", or "ddpm")
            dpm_solver_order: Order for DPM-Solver++ (1, 2, or 3)

        Returns:
            samples: Generated latent samples [B, C, H, W]
            noise_preds: (optional) Intermediate noise predictions
            x_start_preds: (optional) Intermediate x_start predictions
        """
        result = super().sample(
            device=device,
            batch_size=batch_size,
            condition=condition,
            return_intermediate=return_intermediate,
            override_sampling_steps=override_sampling_steps,
            guidance_scale=guidance_scale,
            sampler=sampler,
            dpm_solver_order=dpm_solver_order,
            **kwargs
        )

        if return_intermediate:
            samples, noise_preds, x_start_preds = result
            # Return samples as list for consistency with other diffusion implementations
            return [samples], noise_preds, x_start_preds

        return [result]


def create_unet(config: configuration.MegaTransformerConfig, latent_channels, context_dim: Optional[int] = None):
    return diffusion.ConvDenoisingUNet(
        activation=config.image_decoder_activation if hasattr(config, 'image_decoder_activation') else "silu",
        stride=(2, 2),
        self_attn_class=ImageDiffusionSelfAttentionBlock,
        cross_attn_class=ImageDiffusionCrossAttentionBlock,
        norm_class=ImageRMSNorm,
        in_channels=latent_channels,
        model_channels=config.image_decoder_model_channels,
        out_channels=latent_channels,
        channel_multipliers=config.image_decoder_channel_multipliers,
        time_embedding_dim=config.image_decoder_time_embedding_dim,
        attention_levels=config.image_decoder_attention_levels,
        num_res_blocks=config.image_decoder_num_res_blocks,
        has_condition=True,
        context_dim=context_dim,
        dropout_p=config.image_decoder_unet_dropout_p,
        down_block_self_attn_n_heads=config.image_decoder_down_block_self_attn_n_heads,
        down_block_self_attn_d_queries=config.image_decoder_down_block_self_attn_d_queries,
        down_block_self_attn_d_values=config.image_decoder_down_block_self_attn_d_values,
        down_block_self_attn_use_flash_attention=config.image_decoder_down_block_self_attn_use_flash_attention,
        up_block_self_attn_n_heads=config.image_decoder_up_block_self_attn_n_heads,
        up_block_self_attn_d_queries=config.image_decoder_up_block_self_attn_d_queries,
        up_block_self_attn_d_values=config.image_decoder_up_block_self_attn_d_values,
        up_block_self_attn_use_flash_attention=config.image_decoder_up_block_self_attn_use_flash_attention,
        cross_attn_n_heads=config.image_decoder_cross_attn_n_heads,
        cross_attn_d_queries=config.image_decoder_cross_attn_d_queries,
        cross_attn_d_values=config.image_decoder_cross_attn_d_values,
        cross_attn_use_flash_attention=config.image_decoder_cross_attn_use_flash_attention,
    )

def create_dit_unet(config: configuration.MegaTransformerConfig, latent_channels, context_dim: int):
    return diffusion.DiTBackbone(
        config.hidden_size,
        config.image_decoder_num_res_blocks,
        config.image_decoder_down_block_self_attn_n_heads,
        config.image_decoder_cross_attn_n_heads,
        config.image_decoder_unet_dropout_p,
        config.image_decoder_channel_multipliers[0],
        2,
        latent_channels,
        context_dim,
        config.image_size,
    )


# ~80K params - tiny model for quick tests/proving architecture out
tiny_dit_config = configuration.MegaTransformerConfig(
    hidden_size=32,
    image_size=32,
    image_decoder_model_channels=16,
    image_decoder_time_embedding_dim=32,
    image_decoder_num_res_blocks=2,
    image_decoder_down_block_self_attn_n_heads=2,
    image_decoder_down_block_self_attn_d_queries=16,
    image_decoder_down_block_self_attn_d_values=16,
    image_decoder_down_block_self_attn_use_flash_attention=True,
    image_decoder_cross_attn_n_heads=2,
    image_decoder_cross_attn_d_queries=16,
    image_decoder_cross_attn_d_values=16,
    image_decoder_cross_attn_use_flash_attention=True,
    image_decoder_channel_multipliers=[4],  # MLP ratio for dit
)

# ~14M params - small DiT model for memorization tests
small_dit_config = configuration.MegaTransformerConfig(
    hidden_size=248,
    image_size=32,
    image_decoder_model_channels=80,
    image_decoder_time_embedding_dim=192,
    image_decoder_attention_levels=[False, True, True],
    image_decoder_num_res_blocks=8,
    image_decoder_down_block_self_attn_n_heads=4,
    image_decoder_down_block_self_attn_d_queries=48,
    image_decoder_down_block_self_attn_d_values=48,
    image_decoder_down_block_self_attn_use_flash_attention=True,
    image_decoder_cross_attn_n_heads=4,
    image_decoder_cross_attn_d_queries=48,
    image_decoder_cross_attn_d_values=48,
    image_decoder_cross_attn_use_flash_attention=True,
    image_decoder_channel_multipliers=[4],
)

# ~46M params, medium DiT model for higher-capacity testing and generalization probing
medium_dit_config = configuration.MegaTransformerConfig(
    hidden_size=384,
    image_size=32,
    image_decoder_model_channels=128,
    image_decoder_time_embedding_dim=256,
    image_decoder_attention_levels=[False, True, True],
    image_decoder_num_res_blocks=12,
    image_decoder_down_block_self_attn_n_heads=6,
    image_decoder_down_block_self_attn_d_queries=64,
    image_decoder_down_block_self_attn_d_values=64,
    image_decoder_down_block_self_attn_use_flash_attention=True,
    image_decoder_cross_attn_n_heads=6,
    image_decoder_cross_attn_d_queries=64,
    image_decoder_cross_attn_d_values=64,
    image_decoder_cross_attn_use_flash_attention=True,
    image_decoder_channel_multipliers=[4],
)


# ~13M params - Good for experimentation and overfitting tests. Forces cross attention at all but the largest levels.
small_config = configuration.MegaTransformerConfig(
    hidden_size=192,
    image_size=32,
    image_decoder_model_channels=80,
    image_decoder_time_embedding_dim=192,
    image_decoder_attention_levels=[False, True, True],
    image_decoder_num_res_blocks=2,
    image_decoder_down_block_self_attn_n_heads=4,
    image_decoder_down_block_self_attn_d_queries=48,
    image_decoder_down_block_self_attn_d_values=48,
    image_decoder_down_block_self_attn_use_flash_attention=True,
    image_decoder_up_block_self_attn_n_heads=4,
    image_decoder_up_block_self_attn_d_queries=48,
    image_decoder_up_block_self_attn_d_values=48,
    image_decoder_up_block_self_attn_use_flash_attention=True,
    image_decoder_cross_attn_n_heads=4,
    image_decoder_cross_attn_d_queries=48,
    image_decoder_cross_attn_d_values=48,
    image_decoder_cross_attn_use_flash_attention=True,
    image_decoder_channel_multipliers=[1, 2, 3],  # Gradual growth
)


# ~20M params - Optimized for conditioning-based memorization with larger attention capacity
# Prioritizes cross-attention dim (320) to preserve more conditioning info (512→320 = 62.5%)
# Larger bottleneck (4x) for more capacity at 8x8 where semantic decisions happen
medium_config = configuration.MegaTransformerConfig(
    hidden_size=320,
    image_size=32,
    image_decoder_model_channels=80,
    image_decoder_time_embedding_dim=192,
    image_decoder_attention_levels=[False, True, True],
    image_decoder_num_res_blocks=2,
    image_decoder_channel_multipliers=[1, 2, 4],  # 4x at bottleneck for more capacity
    # 320-dim attention (4 heads × 80) - 67% larger than small config
    image_decoder_down_block_self_attn_n_heads=4,
    image_decoder_down_block_self_attn_d_queries=80,
    image_decoder_down_block_self_attn_d_values=80,
    image_decoder_down_block_self_attn_use_flash_attention=True,
    image_decoder_up_block_self_attn_n_heads=4,
    image_decoder_up_block_self_attn_d_queries=80,
    image_decoder_up_block_self_attn_d_values=80,
    image_decoder_up_block_self_attn_use_flash_attention=True,
    image_decoder_cross_attn_n_heads=4,
    image_decoder_cross_attn_d_queries=80,
    image_decoder_cross_attn_d_values=80,
    image_decoder_cross_attn_use_flash_attention=True,
)


def create_diffusion_model(
    config: configuration.MegaTransformerConfig,
    unet: nn.Module,
    latent_channels: int = 4,
    num_timesteps: int = 1000,
    sampling_timesteps: int = 50,
    betas_schedule: str = "cosine",
    normalize: bool = True,
    min_snr_loss_weight: bool = True,
    min_snr_gamma: float = 5.0,
    prediction_type: str = "epsilon",
    cfg_dropout_prob: float = 0.1,  # Default 10% dropout for CFG training
    zero_terminal_snr: bool = True,  # Recommended for image generation
    offset_noise_strength: float = 0.1,  # Recommended for image generation
    timestep_sampling: str = "logit_normal",  # "uniform" or "logit_normal"
    logit_normal_mean: float = 0.0,  # Mean for logit-normal (0 = centered on middle timesteps)
    logit_normal_std: float = 1.0,  # Std for logit-normal (lower = more peaked)
) -> ImageConditionalGaussianDiffusion:
    return ImageConditionalGaussianDiffusion(
        config=config,
        unet=unet,
        in_channels=latent_channels,
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
    # ~13M params - good for experimentation and overfitting tests
    "small": lambda context_dim, **kwargs: create_diffusion_model(
        config=small_config,
        unet=create_unet(config=small_config, latent_channels=4, context_dim=context_dim),
        **kwargs
    ),
    # ~20M params - optimized for conditioning-based memorization
    "medium": lambda context_dim, **kwargs: create_diffusion_model(
        config=medium_config,
        unet=create_unet(config=medium_config, latent_channels=4, context_dim=context_dim),
        **kwargs
    ),
    "tiny_dit": lambda context_dim, **kwargs: create_diffusion_model(
        config=tiny_dit_config,
        unet=create_dit_unet(config=tiny_dit_config, latent_channels=4, context_dim=context_dim),
        **kwargs
    ),
    "small_dit": lambda context_dim, **kwargs: create_diffusion_model(
        config=small_dit_config,
        unet=create_dit_unet(config=small_dit_config, latent_channels=4, context_dim=context_dim),
        **kwargs
    ),
    "tiny_dit_flow": lambda context_dim, cfg_dropout_prob, timestep_sampling, **_: diffusion.create_flow_matching_model(
        config=tiny_dit_config,
        unet=create_dit_unet(config=tiny_dit_config, latent_channels=4, context_dim=context_dim),
        latent_channels=4,
        cfg_dropout_prob=cfg_dropout_prob,
        timestep_sampling=timestep_sampling,
    ),
    "small_dit_flow": lambda context_dim, cfg_dropout_prob, timestep_sampling, **_: diffusion.create_flow_matching_model(
        config=small_dit_config,
        unet=create_dit_unet(config=small_dit_config, latent_channels=4, context_dim=context_dim),
        latent_channels=4,
        cfg_dropout_prob=cfg_dropout_prob,
        timestep_sampling=timestep_sampling,
    ),
    "medium_dit_flow": lambda context_dim, cfg_dropout_prob, timestep_sampling, **_: diffusion.create_flow_matching_model(
        config=medium_dit_config,
        unet=create_dit_unet(config=medium_dit_config, latent_channels=4, context_dim=context_dim),
        latent_channels=4,
        cfg_dropout_prob=cfg_dropout_prob,
        timestep_sampling=timestep_sampling,
    ),
}
