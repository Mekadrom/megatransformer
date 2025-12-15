import torch
import torch.nn as nn

import megatransformer_utils
from model.image.vae import VAE
from model.megatransformer_diffusion import GaussianDiffusion
from model.megatransformer_image_decoder import ImageSelfAttentionBlock, ImageCrossAttentionBlockSimple
from model import norms


class ImageConditionalGaussianDiffusion(GaussianDiffusion):
    """
    Image diffusion model for text-to-image generation.
    Operates on VAE latent space with text conditioning via cross-attention.
    """

    def __init__(
        self,
        config: megatransformer_utils.MegaTransformerConfig,
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

    def sample(self, device, batch_size, condition=None, return_intermediate=False, override_ddim_sampling_steps=None, **kwargs):
        """
        Sample from the diffusion model.

        Args:
            device: Device to run on
            batch_size: Number of samples to generate
            condition: Text embeddings for conditioning
            return_intermediate: If True, return intermediate denoising steps
            override_ddim_sampling_steps: Override DDIM sampling steps

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
            override_ddim_sampling_steps=override_ddim_sampling_steps,
            **kwargs
        )

        if return_intermediate:
            samples, noise_preds, x_start_preds = result
            # Return samples as list for consistency with audio diffusion
            return [samples], noise_preds, x_start_preds

        return [result]


# Model configurations for image diffusion
tiny_image_diffusion_config = megatransformer_utils.MegaTransformerConfig(
    hidden_size=64,
    image_size=32,  # Latent size (e.g., 256/8=32 with 8x compression VAE)
    image_decoder_model_channels=64,
    image_decoder_time_embedding_dim=64,
    image_decoder_num_res_blocks=2,
    image_decoder_down_block_self_attn_n_heads=2,
    image_decoder_down_block_self_attn_d_queries=16,
    image_decoder_down_block_self_attn_d_values=16,
    image_decoder_down_block_self_attn_use_flash_attention=True,
    image_decoder_up_block_self_attn_n_heads=2,
    image_decoder_up_block_self_attn_d_queries=16,
    image_decoder_up_block_self_attn_d_values=16,
    image_decoder_up_block_self_attn_use_flash_attention=True,
    image_decoder_cross_attn_n_heads=2,
    image_decoder_cross_attn_d_queries=16,
    image_decoder_cross_attn_d_values=16,
    image_decoder_cross_attn_use_flash_attention=True,
    image_decoder_channel_multipliers=[2, 4],
)

small_image_diffusion_config = megatransformer_utils.MegaTransformerConfig(
    hidden_size=384,
    image_size=32,
    image_decoder_model_channels=128,
    image_decoder_time_embedding_dim=256,
    image_decoder_num_res_blocks=2,
    image_decoder_down_block_self_attn_n_heads=6,
    image_decoder_down_block_self_attn_d_queries=64,
    image_decoder_down_block_self_attn_d_values=64,
    image_decoder_down_block_self_attn_use_flash_attention=True,
    image_decoder_up_block_self_attn_n_heads=6,
    image_decoder_up_block_self_attn_d_queries=64,
    image_decoder_up_block_self_attn_d_values=64,
    image_decoder_up_block_self_attn_use_flash_attention=True,
    image_decoder_cross_attn_n_heads=6,
    image_decoder_cross_attn_d_queries=64,
    image_decoder_cross_attn_d_values=64,
    image_decoder_cross_attn_use_flash_attention=True,
)

medium_image_diffusion_config = megatransformer_utils.MegaTransformerConfig(
    hidden_size=512,
    image_size=32,
    image_decoder_model_channels=256,
    image_decoder_time_embedding_dim=512,
    image_decoder_num_res_blocks=3,
    image_decoder_down_block_self_attn_n_heads=8,
    image_decoder_down_block_self_attn_d_queries=64,
    image_decoder_down_block_self_attn_d_values=64,
    image_decoder_down_block_self_attn_use_flash_attention=True,
    image_decoder_up_block_self_attn_n_heads=8,
    image_decoder_up_block_self_attn_d_queries=64,
    image_decoder_up_block_self_attn_d_values=64,
    image_decoder_up_block_self_attn_use_flash_attention=True,
    image_decoder_cross_attn_n_heads=8,
    image_decoder_cross_attn_d_queries=64,
    image_decoder_cross_attn_d_values=64,
    image_decoder_cross_attn_use_flash_attention=True,
)


def create_image_diffusion_model(
    config: megatransformer_utils.MegaTransformerConfig,
    latent_channels: int = 4,
    num_timesteps: int = 1000,
    sampling_timesteps: int = 50,
    betas_schedule: str = "cosine",
    context_dim: int = 512,
    normalize: bool = True,
    min_snr_loss_weight: bool = True,
    min_snr_gamma: float = 5.0,
    prediction_type: str = "epsilon",
) -> ImageConditionalGaussianDiffusion:
    """Create an image diffusion model from config."""
    model = ImageConditionalGaussianDiffusion(
        config=config,
        activation=config.image_decoder_activation if hasattr(config, 'image_decoder_activation') else "silu",
        scale_factor=2,
        stride=2,
        self_attn_class=ImageSelfAttentionBlock,
        cross_attn_class=ImageCrossAttentionBlockSimple,
        norm_class=norms.RMSNorm,
        in_channels=latent_channels,
        model_channels=config.image_decoder_model_channels,
        out_channels=latent_channels,
        time_embedding_dim=config.image_decoder_time_embedding_dim,
        num_res_blocks=config.image_decoder_num_res_blocks,
        unet_dropout_p=config.image_decoder_unet_dropout_p if hasattr(config, 'image_decoder_unet_dropout_p') else 0.1,
        num_timesteps=num_timesteps,
        betas_schedule=betas_schedule,
        has_condition=True,
        context_dim=context_dim,
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
        min_snr_loss_weight=min_snr_loss_weight,
        min_snr_gamma=min_snr_gamma,
        normalize=normalize,
        sampling_timesteps=sampling_timesteps,
        prediction_type=prediction_type,
    )

    return model


model_config_lookup = {
    "tiny_image_diffusion": lambda **kwargs: create_image_diffusion_model(
        config=tiny_image_diffusion_config,
        **kwargs
    ),
    "small_image_diffusion": lambda **kwargs: create_image_diffusion_model(
        config=small_image_diffusion_config,
        **kwargs
    ),
    "medium_image_diffusion": lambda **kwargs: create_image_diffusion_model(
        config=medium_image_diffusion_config,
        **kwargs
    ),
    "tiny_v_image_diffusion": lambda **kwargs: create_image_diffusion_model(
        config=tiny_image_diffusion_config,
        prediction_type="v",
        **kwargs
    ),
}


class VAEDiffusionWrapper(nn.Module):
    """Wrapper for VAE to be used as a diffuser model."""
    def __init__(self, latent_scale_factor: float, vae: VAE, diffusion: GaussianDiffusion):
        super().__init__()
        
        self.latent_scale_factor = latent_scale_factor
        self.vae = vae
        self.diffusion = diffusion

        # freeze VAE
        self.vae.eval()
        self.vae.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        self.vae.requires_grad_(False)
        return self

    def forward(self, x: torch.Tensor, **kwargs):
        # for training, encode input to latent space
        mu, _ = self.vae.encoder(x)
        mu = mu * self.latent_scale_factor

        # take only mean for diffusion
        latent_diffusion_output, losses = self.diffusion.forward(mu, **kwargs)
        mse_loss = losses[0]

        return latent_diffusion_output, mse_loss

    def sample(self, device, batch_size: int, **kwargs) -> torch.Tensor:
        # sample in latent space (pass latent image dimensions as "image_size" in kwargs)
        latent_samples = self.diffusion.sample(device, batch_size, **kwargs)
        latent_samples = latent_samples / self.latent_scale_factor        

        # decode to image space
        recon_images = self.vae.decoder(latent_samples)

        return recon_images
