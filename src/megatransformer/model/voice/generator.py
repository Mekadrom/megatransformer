import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from typing import List, Optional

from megatransformer.config.voice.generator import (
    VOICE_CODA_CONFIGS,
    VoiceCodaAndSMGConfig,
)
from megatransformer.model.norms import create_norm
from megatransformer.model.transformer import MegaTransformerEncoderBlock
from megatransformer.utils.megatransformer_utils import (
    apply_depth_scaled_residual_init,
    conv2d_weight_init,
    linear_weight_init,
)


class TemporalRefine(nn.Module):
    """Temporal refinement via Conv1d for SIVE feature predictions.

    Two Conv1d(kernel_size=3) layers with GELU activation give a 5-frame
    receptive field (~240ms at 16kHz / hop=256 / SIVE 3x downsample),
    roughly one phoneme — enough for local coherence without smearing
    across phoneme boundaries.

    WARNING: This module has a train/inference mismatch for autoregressive
    generation. During training the Conv1d sees 3-frame context, but during
    autoregressive inference (seq_len=1) it sees only zero-padded single
    frames. Use ``FramewiseRefine`` for autoregressive codas.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, timesteps)"""
        return x + self.conv2(self.act(self.conv1(x)))


class FramewiseRefine(nn.Module):
    """Per-frame nonlinear refinement for SIVE feature predictions.

    A small residual MLP operating independently on each frame in the
    feature channel space (post-projection, typically 128-dim). Unlike
    ``TemporalRefine`` (Conv1d-based), this has no temporal dependency —
    no train/inference mismatch for autoregressive generation.

    The coda's causal transformer already provides sequential context in
    d_model space via KV caching. This adds nonlinear capacity in the
    lower-dimensional feature space where the transformer cannot operate.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(channels * 2, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, timesteps)"""
        # Linear layers need (batch, timesteps, channels)
        h = x.permute(0, 2, 1)
        h = self.fc2(self.act(self.fc1(h)))
        return x + h.permute(0, 2, 1)


class VoiceCodaAndSMGWithLoss(nn.Module):
    """
    Voice output head for the multimodal world model.

    Predicts SIVE features from transformer hidden states.
    Output shape: (batch, feature_channels, timesteps).

    The pipeline:
    1. Coda transformer with residual connection
    2. Linear projection from d_model to feature_channels
    3. Transpose to (batch, feature_channels, timesteps)
    4. (Optional) Conv1d temporal refinement

    Decoding SIVE features to mel/waveform is handled externally (SMG + vocoder).
    """

    def __init__(self, prefix: str, config: VoiceCodaAndSMGConfig):
        super(VoiceCodaAndSMGWithLoss, self).__init__()

        self.prefix = prefix
        self.config = config
        self.feature_channels = config.feature_channels

        coda_config = config.coda_config

        if config.use_input_norm:
            self.input_norm = create_norm(coda_config.d_model, config.input_norm_type, config.norm_epsilon)

        self.coda = nn.ModuleList([
            MegaTransformerEncoderBlock(coda_config)
            for _ in range(config.n_layers)
        ])

        self.feature_projection = nn.Linear(coda_config.d_model, config.feature_channels)

        # Optional refinement after linear projection
        output_mode = getattr(config, 'output_mode', 'linear')
        if output_mode == "conv_refine":
            self.temporal_refine = TemporalRefine(config.feature_channels)
        elif output_mode == "framewise_refine":
            self.temporal_refine = FramewiseRefine(config.feature_channels)
        else:
            self.temporal_refine = None

        # Learnable denormalization: maps from normalized space back to original
        # latent distribution. Initialized to identity (scale=1, bias=0).
        if config.use_output_norm:
            if config.output_norm_type == "scale_shift":
                self.output_scale = nn.Parameter(torch.ones(config.feature_channels))
                self.output_bias = nn.Parameter(torch.zeros(config.feature_channels))
            else:
                self.output_norm = create_norm(coda_config.d_model, config.output_norm_type, config.norm_epsilon)

        # Stop prediction head: single linear → scalar logit per frame.
        # Predicts whether the current frame is the last real frame (or past it).
        self.stop_head = nn.Linear(coda_config.d_model, 1)

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        self.gradient_checkpointing = False
        self._init_weights()

    def _init_weights(self):
        init_linear = linear_weight_init(gain=1.0)
        for block in self.coda:
            block.apply(init_linear)
        apply_depth_scaled_residual_init(self.coda)
        self.feature_projection.apply(init_linear)
        # Bias the stop head toward "continue" at init so an untrained model
        # generates full sequences instead of stopping immediately.
        nn.init.zeros_(self.stop_head.weight)
        nn.init.constant_(self.stop_head.bias, -5.0)  # sigmoid(-5) ≈ 0.007
        if self.temporal_refine is not None:
            if isinstance(self.temporal_refine, TemporalRefine):
                # TemporalRefine uses Conv1d — Kaiming init for ReLU/GELU.
                self.temporal_refine.apply(conv2d_weight_init())
            elif isinstance(self.temporal_refine, FramewiseRefine):
                self.temporal_refine.apply(init_linear)

    @classmethod
    def from_config(cls, prefix: str, config_name: str, **overrides) -> "VoiceCodaAndSMGWithLoss":
        if config_name not in VOICE_CODA_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(VOICE_CODA_CONFIGS.keys())}")

        config = VOICE_CODA_CONFIGS[config_name]
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = VoiceCodaAndSMGConfig(**config_dict)

        return cls(prefix, config)

    def forward(
        self,
        x: torch.Tensor,
        latent_labels: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        decode_to_mel: bool = False,
        kv_caches: Optional[List] = None,
        position_offset: int = 0,
        use_cache: bool = False,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Process hidden states through coda to predict SIVE features.

        Args:
            x: Input tensor of shape (batch, seq_length, d_model).
            latent_labels: Target SIVE features for loss computation.
                Shape: (batch, feature_channels, timesteps)
            lengths: Unused (kept for interface compatibility).
            decode_to_mel: Unused (kept for interface compatibility).
            kv_caches: Optional list of KVCache objects, one per coda layer.
                When provided, the coda's self-attention attends to cached
                representations from previous positions (inference only).
            position_offset: RoPE position offset for cached generation.
            use_cache: If True, return updated KV caches in the output dict.

        Returns:
            Dictionary containing:
            - "{prefix}_latent_preds": Predicted SIVE features (batch, feature_channels, timesteps)
            - "{prefix}_latent_l1_loss", "{prefix}_latent_mse_loss": Losses (if latent_labels provided)
            - "kv_caches": Updated KV caches (if use_cache=True)
        """
        if hasattr(self, 'input_norm'):
            x = self.input_norm(x)

        # MegaTransformerEncoderBlock.forward already adds residuals internally,
        # so the loop just chains layers without re-adding the input.
        h = x
        new_kv_caches = []
        for i, block in enumerate(self.coda):
            block_cache = kv_caches[i] if kv_caches is not None else None
            if self.gradient_checkpointing and self.training and not use_cache:
                h, new_cache = torch_checkpoint(
                    block, h, None, None, block_cache, position_offset, use_cache,
                    use_reentrant=False,
                )
            else:
                h, new_cache = block(
                    h,
                    kv_cache=block_cache,
                    position_offset=position_offset,
                    use_cache=use_cache,
                )
            if use_cache:
                new_kv_caches.append(new_cache)

        feature_preds = self.feature_projection(h)  # (batch, seq_length, feature_channels)
        # Denormalize: learnable scale and bias map back to original latent range
        if hasattr(self, 'output_scale') and hasattr(self, 'output_bias'):
            feature_preds = feature_preds * self.output_scale + self.output_bias
        elif hasattr(self, 'output_norm'):
            feature_preds = self.output_norm(feature_preds)

        feature_preds = feature_preds.permute(0, 2, 1)  # (batch, feature_channels, timesteps)

        if self.temporal_refine is not None:
            feature_preds = self.temporal_refine(feature_preds)

        # Stop logits: (batch, timesteps) — probability that this frame is at/past the end
        stop_logits = self.stop_head(h).squeeze(-1)  # (batch, seq_length)

        outputs = {
            f"{self.prefix}_latent_preds": feature_preds,
            f"{self.prefix}_stop_logits": stop_logits,
        }
        if use_cache:
            outputs["kv_caches"] = new_kv_caches

        if latent_labels is not None:
            outputs[f"{self.prefix}_latent_l1_loss"] = self.l1_loss(feature_preds, latent_labels)
            outputs[f"{self.prefix}_latent_mse_loss"] = self.mse_loss(feature_preds, latent_labels)

        return outputs
