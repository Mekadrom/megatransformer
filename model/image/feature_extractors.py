import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Optional

from model.image.vae import ImageVAEEncoder
from model.world.transformer import MegaTransformerBlock
from utils import configuration
from utils.megatransformer_utils import conv2d_weight_init, transformer_weight_init


@dataclass
class ImageVAEPreludeFeatureExtractorConfig:
    prelude_config: configuration.TransformerBlockConfig
    image_config: configuration.ImageConfig


class PatchEmbedding(nn.Module):
    """
    Converts a 2D feature map into a sequence of patch embeddings.

    Uses a strided convolution to extract non-overlapping patches from the input
    and project each patch to d_model dimensions in a single operation.
    """

    def __init__(self, in_channels: int, patch_size: int, d_model: int):
        super().__init__()

        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model).
        """
        x = self.projection(x)  # Shape: (batch_size, d_model, H/patch_size, W/patch_size)
        N, C, H, W = x.size()  # C == d_model
        x = x.flatten(2).transpose(1, 2)  # Shape: (batch_size, seq_length, d_model)
        return x


class ImageVAEPreludeFeatureExtractor(nn.Module):
    """
    Encodes images into token embeddings for the multimodal world model.

    Supports two modes:
    1. Live encoding: Pass raw RGB images, VAE encoder compresses to latents
    2. Precomputed latents: Pass precomputed VAE latents directly (for efficient training)

    The pipeline after obtaining latents:
    1. Patch embedding extracts and projects non-overlapping patches to d_model
    2. 2D positional encoding is added
    3. Image-specific prelude transformer processes the sequence

    The prelude allows image-specific processing before interleaving with other modalities.
    Uses a residual connection around the prelude transformer.
    """

    def __init__(self, config: ImageVAEPreludeFeatureExtractorConfig, vae_encoder: Optional[ImageVAEEncoder] = None):
        super().__init__()

        self.config = config

        prelude_config = config.prelude_config
        self.prelude = MegaTransformerBlock(prelude_config)

        self.patch_embedding = PatchEmbedding(
            in_channels=config.image_config.latent_channels,
            patch_size=config.image_config.latent_patch_size,
            d_model=prelude_config.d_model
        )

        self.vae_encoder = vae_encoder

        # 2D positional encoding for patches
        # Compute number of patches: (latent_size / patch_size)^2
        latent_size = config.image_config.image_size // config.image_config.latent_compression_factor
        num_patches_per_side = latent_size // config.image_config.latent_patch_size
        num_patches = num_patches_per_side ** 2
        self.num_patches_per_side = num_patches_per_side

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_patches, prelude_config.d_model)
        )

        self._init_weights()

    def _init_weights(self):
        # Patch embedding uses conv2d
        self.patch_embedding.apply(conv2d_weight_init())
        # Prelude transformer
        self.prelude.apply(transformer_weight_init())
        # Initialize positional embedding with small values
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor, precomputed_latents: bool = False) -> torch.Tensor:
        """
        Process image input into token embeddings.

        Args:
            x: Either raw RGB images or precomputed VAE latents.
                - Raw images: (batch_size, 3, image_size, image_size)
                - Precomputed latents: (batch_size, latent_channels, latent_h, latent_w)
            precomputed_latents: If True, x is treated as precomputed VAE latents.
                If False, x is passed through the VAE encoder first.

        Returns:
            Token embeddings of shape (batch_size, num_patches, d_model).
        """
        if precomputed_latents:
            latent_images = x
        else:
            if self.vae_encoder is None:
                raise ValueError("VAE encoder required for raw image input. "
                                 "Either provide vae_encoder at init or use precomputed_latents=True.")
            latent_images = self.vae_encoder(x)

        # Shape: (batch_size, latent_channels, latent_spatial_size, latent_spatial_size)
        patch_embeddings = self.patch_embedding(latent_images)  # (batch_size, num_patches, d_model)

        # Add 2D positional encoding
        patch_embeddings = patch_embeddings + self.pos_embedding

        prelude_hidden, _ = self.prelude(patch_embeddings)  # (batch_size, num_patches, d_model)
        prelude_output = patch_embeddings + prelude_hidden

        return prelude_output