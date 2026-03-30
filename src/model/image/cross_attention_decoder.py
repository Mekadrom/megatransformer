"""
Encoder-decoder image generator for the world model.

Takes variable-length content tokens from the recurrent block, refines them
through self-attention encoder layers, then generates fixed-size image latents
via cross-attention decoder layers.

Architecture:
    content tokens (B, N_content, d_model) — from recurrent block

    Encoder (self-attention over content tokens):
        N_enc layers of: self_attn(content) → ffn
        Builds globally-aware semantic representations.

    Decoder (cross-attention from spatial queries to encoded content):
        N_dec layers of: self_attn(patches) → cross_attn(patches, encoded_content) → ffn
        Spatial queries progressively pull content into a 2D layout.

    unpatchify → (B, latent_channels, H, W)
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple
from dataclasses import dataclass

from config.common import MegaTransformerBlockConfig
from model.transformer import MegaTransformerDecoderBlock, MegaTransformerEncoderBlock
from model.image.generator import PixelShuffleUnpatchify, Unpatchify
from utils.megatransformer_utils import transformer_weight_init


@dataclass
class CrossAttentionImageDecoderConfig:
    """Config for the encoder-decoder image generator."""
    # Shared transformer config for both encoder and decoder blocks
    decoder_config: MegaTransformerBlockConfig = None

    # Layer counts
    n_encoder_layers: int = 4  # self-attention over content tokens
    n_layers: int = 6  # cross-attention decoder layers (kept as n_layers for backward compat)

    # Image spatial dimensions
    latent_channels: int = 12
    latent_spatial_size: int = 32
    patch_size: int = 2

    # Unpatchify mode
    unpatchify_mode: str = "pixel_shuffle"

    # Whether to use learnable output denormalization
    use_output_denorm: bool = True

    def __post_init__(self):
        if self.decoder_config is None:
            self.decoder_config = MegaTransformerBlockConfig(
                d_model=512, n_heads=8, d_queries=64, d_values=64,
                n_query_groups=8, d_inner=2048,
                causal=False,
                pre_attn_norm=True, inter_attn_norm=True, pre_ffn_norm=True,
            )
        elif isinstance(self.decoder_config, dict):
            self.decoder_config = MegaTransformerBlockConfig(**self.decoder_config)

    @property
    def n_patches(self) -> int:
        patches_per_side = self.latent_spatial_size // self.patch_size
        return patches_per_side * patches_per_side

    @property
    def grid_size(self) -> Tuple[int, int]:
        s = self.latent_spatial_size // self.patch_size
        return (s, s)


class CrossAttentionImageDecoder(nn.Module):
    """Encoder-decoder image generator.

    Encoder: self-attention layers refine the content tokens from the recurrent
    block into globally-aware semantic representations. All content positions
    can attend to each other (bidirectional), enriching each token with context
    from the full sequence.

    Decoder: cross-attention layers where learned spatial queries attend to the
    encoded content tokens, progressively building a spatial representation
    that is unpatchified into image latents.
    """

    def __init__(self, config: CrossAttentionImageDecoderConfig):
        super().__init__()
        self.config = config
        dc = config.decoder_config

        n_patches = config.n_patches

        # Encoder: self-attention over content tokens (bidirectional)
        encoder_config = copy.deepcopy(dc)
        encoder_config.causal = False
        self.encoder_layers = nn.ModuleList([
            MegaTransformerEncoderBlock(encoder_config)
            for _ in range(config.n_encoder_layers)
        ])
        # Final encoder norm: prevents activation growth from residual
        # compounding across encoder layers (std doubles per layer without this)
        self.encoder_output_norm = nn.LayerNorm(dc.d_model)

        # Learned spatial queries — each query represents a patch position
        self.spatial_queries = nn.Parameter(
            torch.randn(1, n_patches, dc.d_model) * 0.02
        )

        # 2D positional encoding for spatial queries
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, n_patches, dc.d_model)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Decoder layers: self-attn (patches) + cross-attn (patches→content) + FFN
        self.layers = nn.ModuleList([
            MegaTransformerDecoderBlock(dc)
            for _ in range(config.n_layers)
        ])

        # Unpatchify: patch tokens → spatial latent
        if config.unpatchify_mode == "pixel_shuffle":
            self.unpatchify = PixelShuffleUnpatchify(
                dc.d_model, config.latent_channels, config.patch_size,
            )
        else:
            self.unpatchify = Unpatchify(
                dc.d_model, config.latent_channels, config.patch_size,
            )

        # Learnable denormalization. Init scale to ~2.5 to compensate for
        # the unpatchify attenuation (decoder std~2 → unpatchify std~0.4).
        # Target latent std is ~1.15, so scale ≈ 1.15/0.4 ≈ 2.5.
        self.use_output_denorm = config.use_output_denorm
        if self.use_output_denorm:
            self.output_scale = nn.Parameter(torch.ones(config.latent_channels) * 2.5)
            self.output_bias = nn.Parameter(torch.zeros(config.latent_channels))

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        self._init_weights()

    def _init_weights(self):
        self.apply(transformer_weight_init())
        # Re-init unpatchify convolutions with Xavier (gain=0.5) so that
        # decoder output at std~2.0 produces latent preds at std~1.0.
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
        Generate image latents from content tokens via encoder-decoder.

        Args:
            encoder_hidden_states: Content tokens from recurrent block,
                shape (batch, N_content, d_model).
            encoder_attention_mask: Mask for content tokens,
                shape (batch, N_content). True=attend, False=ignore.
            latent_labels: Target image latents for loss computation,
                shape (batch, latent_channels, H, W).

        Returns:
            Dict with:
                - "image_latent_preds": Predicted latents (batch, C, H, W)
                - "image_latent_l1_loss": L1 loss (if latent_labels provided)
                - "image_latent_mse_loss": MSE loss (if latent_labels provided)
        """
        B = encoder_hidden_states.shape[0]

        # Encoder: refine content tokens with bidirectional self-attention
        encoded = encoder_hidden_states
        for layer in self.encoder_layers:
            hidden, _ = layer(encoded, attention_mask=encoder_attention_mask)
            encoded = encoded + hidden
        encoded = self.encoder_output_norm(encoded)

        # Expand spatial queries for batch
        queries = self.spatial_queries.expand(B, -1, -1) + self.pos_embedding

        # Decoder: spatial queries attend to encoded content
        for layer in self.layers:
            queries, _ = layer(
                queries,
                encoder_hidden_states=encoded,
                encoder_attention_mask=encoder_attention_mask,
            )

        # Unpatchify to spatial latent
        latent_preds = self.unpatchify(queries)

        # Denormalize
        if self.use_output_denorm:
            latent_preds = (
                latent_preds * self.output_scale[None, :, None, None]
                + self.output_bias[None, :, None, None]
            )

        outputs = {"image_latent_preds": latent_preds}

        # Compute losses if labels provided
        if latent_labels is not None:
            outputs["image_latent_l1_loss"] = self.l1_loss(latent_preds, latent_labels)
            outputs["image_latent_mse_loss"] = self.mse_loss(latent_preds, latent_labels)

        return outputs
