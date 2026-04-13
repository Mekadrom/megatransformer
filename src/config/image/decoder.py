"""Configuration for the world model's image decoder(s).

This module exposes two decoder configs that the world model can dispatch on:

1. `ImageDecoderConfig` (`src/model/image/decoder.py`):
     Direct latent prediction. Two architecture modes selected by `mode`:
     - "direct" (default, recommended): encoder layers refine the recurrent
       block's content tokens, then unpatchify directly.
     - "cross_attention" (DETR-style): learned spatial queries cross-attend
       to encoded content. More flexible but harder to train from scratch.

2. `DiffusionBridgeImageDecoderConfig` (`src/model/image/diffusion_decoder.py`):
     Flow-matching DiT with a Q-Former-style bridge from the recurrent block
     output. Drop-in replacement for `ImageDecoderConfig`. Trains on the same
     latent labels but uses a flow-matching loss that doesn't have the
     predict-the-mean attractor of L1+MSE.
"""

import dataclasses
import json

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from config.common import MegaTransformerBlockConfig


@dataclass
class ImageDecoderConfig:
    """Configuration for the image decoder.

    `mode` selects between the direct and cross-attention architectures.
    Other fields apply to either, with `n_decoder_layers` only used in
    cross-attention mode.
    """

    # Architecture mode: "direct" or "cross_attention".
    mode: str = "direct"

    # Shared transformer block config used by both encoder and decoder layers
    # (in cross-attention mode) or just by the encoder layers (in direct mode).
    block_config: Optional[MegaTransformerBlockConfig] = None

    # Number of self-attention layers over content tokens. Used in both modes.
    n_encoder_layers: int = 4

    # Number of cross-attention decoder layers. Only used in cross_attention
    # mode; ignored in direct mode.
    n_decoder_layers: int = 6

    # Image latent spatial dimensions
    latent_channels: int = 12
    latent_spatial_size: int = 32
    patch_size: int = 2

    # Unpatchify implementation
    unpatchify_mode: str = "pixel_shuffle"

    # Whether to use a learnable per-channel scale+bias on the output
    use_output_denorm: bool = True

    def __post_init__(self):
        if self.block_config is None:
            self.block_config = MegaTransformerBlockConfig(
                d_model=512, n_heads=8, d_queries=64, d_values=64,
                n_query_groups=8, d_inner=2048,
                causal=False,
                pre_attn_norm=True, inter_attn_norm=True, pre_ffn_norm=True,
            )
        elif isinstance(self.block_config, dict):
            # Defensive: allow dict for hand-written configs / from-disk loading.
            self.block_config = MegaTransformerBlockConfig(**self.block_config)
        if self.mode not in ("direct", "cross_attention"):
            raise ValueError(
                f"Invalid ImageDecoderConfig.mode={self.mode!r}; "
                f"must be 'direct' or 'cross_attention'."
            )

    @property
    def n_patches(self) -> int:
        patches_per_side = self.latent_spatial_size // self.patch_size
        return patches_per_side * patches_per_side

    @property
    def grid_size(self) -> Tuple[int, int]:
        s = self.latent_spatial_size // self.patch_size
        return (s, s)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class DiffusionBridgeImageDecoderConfig:
    """Config for the flow-matching diffusion bridge image decoder.

    Architecture (Flux-style, but sized to match the world model and trained
    from scratch on whatever latents the dataset provides):

        recurrent block image positions  (B, n_recurrent, src_d_model)
          → bridge (cross-attention from learnable queries)
          → conditioning tokens                (B, n_bridge_queries, d_model)

        noisy latent + timestep + conditioning
          → DiT (self-attention + cross-attention to bridge tokens
                 + AdaLN-Zero modulation from timestep)
          → predicted velocity                 (B, latent_channels, H, W)

        loss = MSE(v_pred, v_target)   where v_target = noise - x_0

    `src_d_model` is the recurrent block's d_model and is supplied at module
    construction time (it's normalized via the world model's image_coda_input_norm
    before reaching this decoder, so values are well-scaled).

    Defaults to a small DiT (12 layers, d_model=768) that matches the user's
    `huginn_small_*` recurrent block. Override fields for larger backbones.

    Note on Flux pretrained weights:
        Loading actual Flux.1-schnell weights requires the dataset to be
        re-encoded with Flux's VAE — its DiT was trained against a specific
        latent space. This config does NOT load Flux weights; it only borrows
        the general architecture. Set `pretrained_path=None` (default).
    """

    # ── Bridge module ─────────────────────────────────────────────────────
    # Number of learnable query tokens the bridge produces. These are what
    # the DiT cross-attends to as conditioning. ~32–128 is typical.
    n_bridge_queries: int = 64
    # Bridge transformer depth.
    n_bridge_layers: int = 4
    # Block config for the bridge layers (self-attn + cross-attn + FFN).
    bridge_block_config: Optional[MegaTransformerBlockConfig] = None

    # ── DiT backbone ─────────────────────────────────────────────────────
    # Hidden size of the DiT. Should match (or be reachable from) the recurrent
    # block's d_model so the bridge doesn't have to span a huge gap.
    d_model: int = 768
    # Number of DiT blocks.
    n_dit_layers: int = 12
    # Block config for the DiT layers (self-attn + cross-attn + FFN, with
    # AdaLN-Zero modulation injected by the DiT block class itself).
    dit_block_config: Optional[MegaTransformerBlockConfig] = None

    # ── Image latent dimensions (must match dataset) ─────────────────────
    latent_channels: int = 12
    latent_spatial_size: int = 32
    patch_size: int = 2

    # ── Flow matching ────────────────────────────────────────────────────
    # Sampling distribution for the timestep during training.
    # "uniform" → t ~ U(0, 1).
    # "logit_normal" → t = sigmoid(N(0, 1)), biases mid-range timesteps,
    #                  used by SD3/Flux as a quality improvement.
    timestep_sampling: str = "logit_normal"
    # Number of Euler integration steps for inference sampling.
    num_inference_steps: int = 16

    # ── Latent scaling (SD-style) ────────────────────────────────────────
    # Multiplicative scale applied to clean latents (`x_0`) at the diffusion
    # input, and divided back at sampling output. Brings VAE latents into
    # roughly unit variance, which the flow-matching formulation assumes.
    #
    # Three formats accepted:
    #   - None (default): no scaling, equivalent to all-ones.
    #   - float: global scalar applied uniformly to every channel
    #     (the SD1.x / SDXL approach; e.g. SD1.x uses 0.18215).
    #   - List[float] of length `latent_channels`: per-channel scaling
    #     (the SD3 / Flux approach). Each channel `c` is multiplied by
    #     `latent_scale[c]` independently.
    #
    # For LiteVAE: per-channel statistics show std varies from 0.79 to 1.92
    # across channels (2.4× ratio). Per-channel scaling is recommended for
    # this dataset; global scaling is barely needed since the global std is
    # already ≈1.12.
    latent_scale: Optional[Union[float, List[float]]] = None

    # ── Pretrained loading (optional, not used in option 2) ──────────────
    # Path to a Flux.1-schnell checkpoint. If set, the DiT loads pretrained
    # weights — but only works correctly if the dataset uses Flux's VAE latent
    # space. Default None = train from scratch on whatever latents the dataset
    # provides.
    pretrained_path: Optional[str] = None

    def __post_init__(self):
        if self.bridge_block_config is None:
            self.bridge_block_config = MegaTransformerBlockConfig(
                d_model=self.d_model,
                n_heads=max(self.d_model // 64, 1),
                d_queries=64,
                d_values=64,
                n_query_groups=max(self.d_model // 64, 1),
                d_inner=self.d_model * 4,
                causal=False,
                pre_attn_norm=True,
                inter_attn_norm=True,
                pre_ffn_norm=True,
                use_rotary_embedding=False,
            )
        elif isinstance(self.bridge_block_config, dict):
            self.bridge_block_config = MegaTransformerBlockConfig(**self.bridge_block_config)

        if self.dit_block_config is None:
            self.dit_block_config = MegaTransformerBlockConfig(
                d_model=self.d_model,
                n_heads=max(self.d_model // 64, 1),
                d_queries=64,
                d_values=64,
                n_query_groups=max(self.d_model // 64, 1),
                d_inner=self.d_model * 4,
                causal=False,
                use_rotary_embedding=False,
            )
        elif isinstance(self.dit_block_config, dict):
            self.dit_block_config = MegaTransformerBlockConfig(**self.dit_block_config)

        if self.timestep_sampling not in ("uniform", "logit_normal"):
            raise ValueError(
                f"Invalid timestep_sampling={self.timestep_sampling!r}; "
                f"must be 'uniform' or 'logit_normal'."
            )

        # Validate latent_scale shape if it's a list/tuple.
        if isinstance(self.latent_scale, (list, tuple)):
            if len(self.latent_scale) != self.latent_channels:
                raise ValueError(
                    f"latent_scale list length {len(self.latent_scale)} does not "
                    f"match latent_channels={self.latent_channels}."
                )

    @property
    def n_patches(self) -> int:
        return (self.latent_spatial_size // self.patch_size) ** 2

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
