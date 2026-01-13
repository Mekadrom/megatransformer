import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Optional

from model.image.vae import ImageVAEDecoder
from model.world.transformer import MegaTransformerBlock
from utils import configuration
from utils.megatransformer_utils import conv2d_weight_init, transformer_weight_init


@dataclass
class ImageCodaAndVAEConfig:
    coda_config: configuration.TransformerBlockConfig
    image_config: configuration.ImageConfig


class Unpatchify(nn.Module):
    """
    Reconstructs spatial latent from patch token sequence.

    Uses upsample + conv instead of transposed convolution to avoid
    checkerboard artifacts.
    """

    def __init__(self, d_model: int, latent_channels: int, patch_size: int):
        super(Unpatchify, self).__init__()

        self.patch_size = patch_size

        # Upsample spatially, then conv to reduce channels
        self.upsample = nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(d_model, latent_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unpatchify the input tensor from patches back to image latent format.

        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
                where seq_length = (latent_h / patch_size) ^ 2

        Returns:
            Tensor of shape (batch_size, latent_channels, latent_h, latent_w)
        """
        x = x.permute(0, 2, 1)  # (batch, d_model, seq_length)
        N, D, T = x.shape
        spatial_size = int(T ** 0.5)
        x = x.view(N, D, spatial_size, spatial_size)  # (batch, d_model, H/patch, W/patch)
        x = self.upsample(x)  # (batch, d_model, H, W)
        return self.conv(x)   # (batch, latent_channels, H, W)


class ImageCodaAndVAEWithLoss(nn.Module):
    """
    Image output head for the multimodal world model.

    Supports two modes:
    1. Latent-only (training): Output latent predictions, compute loss in latent space
    2. Full decode (inference): Decode latents to RGB images via VAE decoder

    The pipeline is:
    1. Coda transformer with residual connection
    2. Linear projection to (latent_spatial_size^2 * latent_channels)
    3. Reshape to VAE latent format (batch, channels, height, width)
    4. (Optional) VAE decoder produces RGB images
    """

    def __init__(self, config: ImageCodaAndVAEConfig, vae_decoder: Optional[ImageVAEDecoder] = None):
        super(ImageCodaAndVAEWithLoss, self).__init__()

        self.config = config
        self.latent_channels = config.image_config.latent_channels
        self.latent_spatial_size = config.image_config.image_size // config.image_config.latent_compression_factor

        coda_config = config.coda_config
        self.coda = MegaTransformerBlock(coda_config)

        self.unpatchify = Unpatchify(
            coda_config.d_model,
            config.image_config.latent_channels,
            patch_size=config.image_config.latent_patch_size
        )

        self.vae_decoder = vae_decoder

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = nn.MSELoss()  # TODO: placeholder for actual perceptual loss

        self._init_weights()

    def _init_weights(self):
        # Only init coda and projection; VAE decoder has its own init
        self.coda.apply(transformer_weight_init())
        self.unpatchify.apply(conv2d_weight_init())

    def forward(
        self,
        x: torch.Tensor,
        latent_labels: Optional[torch.Tensor] = None,
        image_labels: Optional[torch.Tensor] = None,
        decode_to_image: bool = False,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Process hidden states through coda and optionally decode to images.

        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model).
            latent_labels: Target latents for loss computation during training.
                Shape: (batch_size, latent_channels, latent_h, latent_w)
            image_labels: Target RGB images for loss computation (only used if decode_to_image=True).
            decode_to_image: If True, decode latents to RGB images via VAE decoder.
                If False, only output latent predictions (more efficient for training).

        Returns:
            Dictionary containing:
            - "latent_preds": Predicted latents (batch, channels, h, w)
            - "reconstructed_images": Decoded images (only if decode_to_image=True)
            - "latent_l1_loss", "latent_mse_loss": Latent space losses (if latent_labels provided)
            - "image_l1_loss", "image_mse_loss", "image_perceptual_loss": Image losses (if decode_to_image and image_labels provided)
        """
        coda_output: torch.Tensor = x + self.coda(x)  # (batch, seq_length, d_model)

        # seq_length should be 256 for a 256x256 image. this is because the vae latent space is 32x32 and we patchify into 2x2 patches

        latent_preds = self.unpatchify(coda_output)  # (batch, latent_channels, latent_h, latent_w)

        outputs = {"image_latent_preds": latent_preds}

        # Latent space loss (for training with precomputed latents)
        if latent_labels is not None:
            outputs["image_latent_l1_loss"] = self.l1_loss(latent_preds, latent_labels)
            outputs["image_latent_mse_loss"] = self.mse_loss(latent_preds, latent_labels)

        # Optionally decode to images
        if decode_to_image:
            if self.vae_decoder is None:
                raise ValueError("VAE decoder required for image decoding. "
                                 "Either provide vae_decoder at init or use decode_to_image=False.")
            reconstructed_images = self.vae_decoder(
                latent_preds,
                return_attention_weights=kwargs.pop("return_attention_weights", False),
            )
            outputs["image_reconstructed_images"] = reconstructed_images

            if image_labels is not None:
                outputs["image_l1_loss"] = self.l1_loss(reconstructed_images, image_labels)
                outputs["image_mse_loss"] = self.mse_loss(reconstructed_images, image_labels)
                outputs["image_perceptual_loss"] = self.perceptual_loss(reconstructed_images, image_labels)

        return outputs
