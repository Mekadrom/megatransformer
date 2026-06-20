"""
Image decoder for the world model.

Takes content tokens from the recurrent block and produces a 2D image latent.
Two architectural modes are supported via `ImageDecoderConfig.mode`:

1. "direct" (default, recommended):
       content tokens (B, n_patches, d_model)
         → encoder layers (self-attention)
         → encoder_output_norm
         → unpatchify → (B, latent_channels, H, W)

   Sample-specific content flows directly into the unpatchify, mirroring the
   voice coda's architecture. Easy to bootstrap from random init because every
   layer's gradient signal is grounded in the input.

2. "cross_attention" (DETR-style):
       encoder layers refine content tokens into encoded keys/values
       learned spatial_queries cross-attend to encoded content
       decoder layers refine queries
       unpatchify → (B, latent_channels, H, W)

   More flexible (queries and content can have different sequence lengths) but
   harder to train from scratch — the decoder must learn to use the cross-
   attention to pull content into queries that start sample-independent. Known
   bootstrap deadlock if cross-attention `o_proj` collapses to ~0.

The default mode is "direct" because it has been empirically observed to escape
the predict-the-mean attractor while "cross_attention" gets stuck in it.
"""

import copy

import torch
import torch.nn as nn

from typing import Optional

from megatransformer.config.image.decoder import ImageDecoderConfig
from megatransformer.model.sinusoidal_positional_encoding import Sinusoidal2DPositionalEmbedding
from megatransformer.model.transformer import MegaTransformerDecoderBlock, MegaTransformerEncoderBlock
from megatransformer.model.image.generator import PixelShuffleUnpatchify, Unpatchify
from megatransformer.utils.megatransformer_utils import (
    apply_depth_scaled_residual_init,
    linear_weight_init,
)


class ImageDecoder(nn.Module):
    """Image decoder for the world model.

    Always has:
        - encoder_layers (self-attention over content tokens)
        - encoder_output_norm
        - unpatchify (sequence → 2D spatial latent)
        - optional output denormalization (learnable per-channel scale+bias)

    In cross_attention mode, additionally has:
        - spatial_queries (learned per-position embeddings)
        - pos_embedding (2D sinusoidal positional encoding for queries)
        - decoder_layers (self-attn + cross-attn + FFN)
    """

    def __init__(self, config: ImageDecoderConfig):
        super().__init__()
        self.config = config
        self.mode = config.mode
        bc = config.block_config

        # Encoder: self-attention over content tokens (bidirectional).
        # Used in both modes — refines the recurrent block's content tokens.
        encoder_block_config = copy.deepcopy(bc)
        encoder_block_config.causal = False
        self.encoder_layers = nn.ModuleList([
            MegaTransformerEncoderBlock(encoder_block_config)
            for _ in range(config.n_encoder_layers)
        ])
        # Final encoder norm: prevents activation growth across layers.
        self.encoder_output_norm = nn.LayerNorm(bc.d_model)

        num_patches_per_side = config.latent_spatial_size // config.patch_size
        self.num_patches_per_side = num_patches_per_side

        # Cross-attention specific components.
        if self.mode == "cross_attention":
            n_patches = config.n_patches
            # Learned spatial queries — each query represents a patch position.
            # Init at std=1/sqrt(d_model) so queries have unit-ish magnitude
            # comparable to (LayerNorm'd) keys, ensuring cross-attention scores
            # are well-distributed at init rather than near-zero.
            self.spatial_queries = nn.Parameter(
                torch.randn(1, n_patches, bc.d_model) * (bc.d_model ** -0.5)
            )
            self.pos_embedding = Sinusoidal2DPositionalEmbedding(
                num_patches_per_side, bc.d_model,
            )
            # Decoder layers: self-attn(queries) + cross-attn(queries→encoded) + FFN
            self.decoder_layers = nn.ModuleList([
                MegaTransformerDecoderBlock(bc)
                for _ in range(config.n_decoder_layers)
            ])

        # Unpatchify: sequence of patch tokens → spatial latent (B, C, H, W)
        if config.unpatchify_mode == "pixel_shuffle":
            self.unpatchify = PixelShuffleUnpatchify(
                bc.d_model, config.latent_channels, config.patch_size,
            )
        else:
            self.unpatchify = Unpatchify(
                bc.d_model, config.latent_channels, config.patch_size,
            )

        # Optional learnable per-channel output denormalization.
        # Init scale=2.5 to compensate for unpatchify attenuation; bias=0.
        self.use_output_denorm = config.use_output_denorm
        if self.use_output_denorm:
            self.output_scale = nn.Parameter(torch.ones(config.latent_channels) * 2.5)
            self.output_bias = nn.Parameter(torch.zeros(config.latent_channels))

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        self._init_weights()

    def _init_weights(self):
        # 1. Standard Xavier on every Linear in encoder + decoder layers.
        self.apply(linear_weight_init(gain=1.0))

        # 2. Depth-scaled init for residual outputs in each stack independently.
        apply_depth_scaled_residual_init(self.encoder_layers)
        if self.mode == "cross_attention":
            apply_depth_scaled_residual_init(self.decoder_layers)

        # 3. Unpatchify convolutions: Xavier with gain=0.5 so the encoder/decoder's
        #    unit-ish output produces latent preds with reasonable magnitude.
        for module in self.unpatchify.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        latent_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Generate image latents from content tokens.

        Args:
            encoder_hidden_states: Content tokens from the recurrent block,
                shape (B, n_content, d_model). For direct mode, n_content
                must equal n_patches.
            encoder_attention_mask: Optional mask for content tokens.
            latent_labels: Optional target image latents for loss computation.

        Returns:
            Dict with:
                - "image_latent_preds": (B, latent_channels, H, W)
                - "image_latent_l1_loss" / "image_latent_mse_loss": if labels given
        """
        # Encoder: self-attention refinement of content tokens. Used in both modes.
        encoded = encoder_hidden_states
        for layer in self.encoder_layers:
            encoded, _ = layer(encoded, attention_mask=encoder_attention_mask)
        encoded = self.encoder_output_norm(encoded)

        if self.mode == "direct":
            # Direct mode: encoded content tokens IS the patch sequence.
            # Just unpatchify them. Sample-specific content flows straight
            # through to the output.
            patches = encoded
        else:
            # Cross-attention mode: learned spatial queries pull from encoded.
            B = encoded.shape[0]
            queries = self.pos_embedding(self.spatial_queries.expand(B, -1, -1))
            for layer in self.decoder_layers:
                queries, _ = layer(
                    queries,
                    encoder_hidden_states=encoded,
                    encoder_attention_mask=encoder_attention_mask,
                )
            patches = queries

        # Unpatchify: (B, n_patches, d_model) → (B, C, H, W)
        latent_preds = self.unpatchify(patches)

        # Optional output denormalization
        if self.use_output_denorm:
            latent_preds = (
                latent_preds * self.output_scale[None, :, None, None]
                + self.output_bias[None, :, None, None]
            )

        outputs = {"image_latent_preds": latent_preds}

        if latent_labels is not None:
            outputs["image_latent_l1_loss"] = self.l1_loss(latent_preds, latent_labels)
            outputs["image_latent_mse_loss"] = self.mse_loss(latent_preds, latent_labels)

        return outputs
