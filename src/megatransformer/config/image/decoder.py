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

from megatransformer.config.common import MegaTransformerBlockConfig


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
    # If False, the Q-Former bridge is removed entirely and the DiT cross-attends
    # DIRECTLY to the recurrent block's image-position outputs (the gen-query
    # hiddens). The bridge's original job — resampling a variable-length input to
    # a fixed token grid — is moot here: the trunk output is already fixed-length
    # and the DiT accepts arbitrary conditioning lengths. Removing it gives a
    # shorter, wider conditioning path (candidate lever for weak text conditioning;
    # see the DiT decomposition discussion). REQUIRES encoder d_model == DiT
    # d_model (small_sum matches by design, see the d_model=768 comment below).
    use_bridge: bool = True
    # Number of learnable query tokens the bridge produces. These are what
    # the DiT cross-attends to as conditioning. ~32–128 is typical.
    # Ignored when use_bridge=False (DiT cross-attends the trunk outputs directly).
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
    # Min-SNR loss weighting (Hang et al., 2023). Downweights easy low-noise
    # timesteps where the model can predict the target almost perfectly,
    # preventing them from dominating gradients. For flow matching with linear
    # interpolation x_t = (1-t)*x_0 + t*noise, SNR(t) = (1-t)^2 / t^2.
    # The per-sample weight is min(SNR(t), gamma) / SNR(t).
    #   gamma=5.0: recommended default from the paper
    #   gamma=None: disabled (uniform weighting)
    min_snr_gamma: Optional[float] = 5.0
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


@dataclass
class SDXLAdapterConfig:
    """Configuration for the SDXL conditioning adapter (frozen-SDXL image path).

    Selected by the world model when `image_coda_config` is this type (existing
    ImageDecoderConfig / DiffusionBridgeImageDecoderConfig configs are untouched,
    so this is fully backwards compatible). Instead of predicting an image latent,
    the adapter maps the trunk's K image gen-query outputs to the conditioning a
    frozen SDXL UNet consumes: a 77x2048 sequence (CLIP ViT-L 768 + OpenCLIP bigG
    1280, concatenated) plus a 1280-dim bigG pooled embedding. K is decoupled from
    the fixed 77 by cross-attention. See model/image/sdxl_adapter.py.
    """

    # Trunk hidden size (input gen-query dim). Must match the recurrent d_model.
    d_model: int = 768
    # Adapter internal width.
    adapter_dim: int = 768
    n_heads: int = 12
    n_layers: int = 2          # self-attn layers over the K gen queries
    n_cross_layers: int = 2    # cross-attn layers (77 slots attend the K queries)
    dropout: float = 0.0

    # SDXL conditioning target shapes (fixed by SDXL; don't change).
    seq_len: int = 77
    seq_dim: int = 2048        # CLIP-L(768) + bigG(1280)
    pool_dim: int = 1280       # bigG pooled

    # Loss: MSE regression + InfoNCE discriminability. The tolerance probe showed
    # SDXL tolerates random error but not semantic bias, so the contrastive term
    # keeps predictions in the right basin. Set contrastive_weight=0 for pure MSE.
    contrastive_weight: float = 1.0
    contrastive_temp: float = 0.07

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class ZImageAdapterConfig:
    """Configuration for the Z-Image conditioning adapter (frozen Z-Image-Turbo path).

    Selected by the world model when `image_coda_config` is this type (existing
    ImageDecoderConfig / DiffusionBridgeImageDecoderConfig / SDXLAdapterConfig configs
    are untouched, so this is fully backwards compatible). Instead of predicting an
    image latent, the adapter maps the trunk's K image gen-query outputs to the Qwen3-4B
    text conditioning Z-Image's S3-DiT cross-attends over: a fixed seq_len x 2560
    sequence (Qwen3-4B penultimate hidden states, resampled to seq_len). No pooled
    vector (Z-Image has none). See model/image/zimage_adapter.py and
    utils/zimage_text_encoder.py.
    """

    # Trunk hidden size (input gen-query dim). Must match the recurrent d_model.
    d_model: int = 768
    # Adapter internal width.
    adapter_dim: int = 768
    n_heads: int = 12
    n_layers: int = 2          # self-attn layers over the K gen queries
    n_cross_layers: int = 2    # cross-attn layers (seq_len slots attend the K queries)
    dropout: float = 0.0

    # Z-Image conditioning target shapes.
    seq_len: int = 64          # K conditioning tokens the adapter emits (Z-Image cross-
                               # attends over any count, so K is free; the trainer
                               # resamples the variable-length Qwen3 target to this K)
    seq_dim: int = 2560        # Qwen3-4B hidden size

    # BASELINE = pure MSE (contrastive_weight=0). Tier-1 discriminability = InfoNCE (via a
    # learned projection head — raw cosine on LM features is weak) between predicted and
    # target conditioning, pushing different captions to distinct outputs (anti-interference).
    # The trainer RAMPS this weight from 0 over --image_contrastive_ramp_steps.
    contrastive_weight: float = 0.0        # max InfoNCE weight (0 = off)
    contrastive_temp: float = 0.07
    contrastive_proj_dim: int = 256        # InfoNCE projection-head width
    # MoCo-style memory queue of past targets (0 = in-batch negatives only). In-batch
    # InfoNCE at batch_size 8 is an 8-way task that saturates near 0 loss within a few
    # hundred steps and stops producing gradient; the queue makes it (B+queue_size)-way.
    # No momentum encoder is needed here because the targets come from a FROZEN encoder.
    # Non-persistent buffer (not checkpointed); refills in queue_size/batch_size steps.
    contrastive_queue_size: int = 0

    # Tier-0 whitened MSE: regress in per-dim z-scored Qwen3 space (removes massive
    # constant dims + equalizes per-dim loss -> attacks MSE-mean collapse). Stats are
    # injected by the trainer (--image_whiten_stats_path) into adapter buffers that
    # persist in the checkpoint; eval/chat must construct with whiten_target=True to
    # de-whiten. Off = naive baseline.
    whiten_target: bool = False

    # INFERENCE-ONLY dispersion correction. An MSE-optimal point estimate is necessarily
    # shrunk toward the target mean by the fraction of variance it cannot explain (measured:
    # alpha 0.745 vs R^2 0.738 -- the conditional-mean signature), and the frozen DiT reads
    # that under-dispersion as a washed-out, generic scene. Scaling the WHITENED prediction
    # by ~1/alpha before de-whitening makes the whitened MSE strictly WORSE and the render
    # measurably BETTER: on the diverse probe, gain 1.34 took CLIPScore 0.2873 -> 0.3025
    # against a 0.3448 GT ceiling (~26% of the remaining gap, no retraining). Broad optimum
    # 1.15-1.55, collapses by 2.1. Estimate it as 1/alpha on held-out captions with
    # scripts_local/zimage_shrinkage_probe.py -- use the VAL alpha, not an OOD probe's.
    # Applies ONLY to the surfaced prediction, never to the training loss (which is computed
    # in whitened space before this). Requires whiten_target. 1.0 = off (default, so recorded
    # eval trends stay comparable).
    output_gain: float = 1.0

    # ── TIER-3: flow-matching sampler instead of a point estimate ──────────────────────
    # An MSE point head is structurally under-dispersed (shrinks toward the conditional mean
    # by 1-R^2) and the frozen DiT renders that hedge as bland/generic; output_gain patches
    # it with a global scalar, but the OPTIMAL gain varies per caption (sparse scenes ~1.2,
    # dense ~1.8). This replaces the estimate with a rectified-flow sampler over the whitened
    # conditioning, conditioned on the Q-Former context: samples land on-manifold at full
    # dispersion per-caption, no gain to tune. See model/image/cond_flow_head.py.
    # Warm-start from a whitened regression checkpoint: the Q-Former loads, the head is fresh.
    flow_head: bool = False
    flow_dim: int = 512                    # head width (~29M params at these defaults)
    flow_heads: int = 8
    flow_layers: int = 4
    flow_steps: int = 8                    # Euler steps at inference (compute <-> fidelity)
    flow_time_sampling: str = "logit_normal"   # "logit_normal" (SD3) or "uniform"
    # Small auxiliary weight on the point head so seq_pred stays a comparable diagnostic
    # (alpha / R^2 / retrieval) against the T0/T1 regression runs. 0 = pure sampler.
    flow_aux_mse_weight: float = 0.1
    # Classifier-free guidance. Training drops the context to a learned null with this
    # probability so the head also learns the UNCONDITIONAL field; sampling then extrapolates
    # v = v_uncond + w*(v_cond - v_uncond). This is the principled form of output_gain: it
    # pushes away from the model's own unconditional prediction per-sample and per-timestep
    # instead of scaling deviation-from-the-mean by one global constant. Needed because a
    # sampler's MEAN draw loses to a conditional-mean point estimate on CLIPScore (measured
    # 0.251 vs 0.287) even while best-of-6 wins (0.319) -- guidance is the knob on that axis.
    # NOTE: a checkpoint trained with cfg_dropout=0 has an untrained null, so guidance only
    # works after training WITH dropout.
    flow_cfg_dropout: float = 0.1
    flow_guidance: float = 1.0             # inference w; 1.0 = off (single forward per step)

    # Target encoder (Z-Image's frozen Qwen3-4B), used by the world trainer to build
    # regression targets. 4-bit keeps it ~2.5GB so it fits alongside training.
    target_model: str = "Tongyi-MAI/Z-Image-Turbo"
    target_max_length: int = 512
    target_load_in_4bit: bool = True

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
