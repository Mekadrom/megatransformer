"""Flow-matching DiT image decoder for the world model.

Drop-in alternative to `model.image.decoder.ImageDecoder`. Trains a flow-
matching DiT (Flux-style architecture) on the same image latent labels as the
direct decoder, with a Q-Former-style bridge from the recurrent block's image-
position outputs into the DiT's cross-attention conditioning.

Why this exists:
    Direct latent prediction with L1+MSE has a deep local minimum at "predict
    the mean" because predicting zero gives MSE = var(labels), which the
    optimizer locks onto when content matching is hard. Flow matching does NOT
    have this attractor — at every timestep `t`, the trivial baseline is "copy
    the input minus a tiny bit," not "predict zero." This makes the loss
    landscape much friendlier for hard generation tasks.

Architecture:
    Bridge: a small transformer with `n_bridge_queries` learnable queries that
    cross-attend to the recurrent block's image-position outputs and produce
    a fixed-size sequence of conditioning tokens. This translates from the
    world model's embedding space to the DiT's expected conditioning space.

    DiT backbone: standard DiT with AdaLN-Zero modulation from the timestep
    embedding, plus per-block cross-attention to the bridge tokens. AdaLN-Zero
    means the modulation is initialized to zero so each block starts as the
    identity function — same trick as Flux, SD3, and the original DiT paper.

    Loss: linear flow matching. Sample t∈[0,1], form `x_t = (1-t)·x_0 + t·noise`,
    target velocity `v_target = noise - x_0`, MSE between predicted and target
    velocity.

Note on Flux pretrained weights:
    This module does NOT load Flux.1-schnell's pretrained weights, only borrows
    its general architecture. Loading the actual checkpoint would require the
    dataset to be re-encoded with Flux's VAE, since the diffusion priors are
    baked into a specific latent space. See the config docstring.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

from config.image.decoder import DiffusionBridgeImageDecoderConfig
from config.common import MegaTransformerBlockConfig
from model.transformer import (
    MegaTransformerAttention,
    MegaTransformerDecoderBlock,
    SimpleFFN,
)
from utils.megatransformer_utils import (
    apply_depth_scaled_residual_init,
    linear_weight_init,
)


# ─── Helpers ──────────────────────────────────────────────────────────────


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN modulation: x * (1 + scale) + shift.

    `scale` and `shift` have shape (B, d_model); broadcast over the sequence dim.
    The (1 + scale) form means scale=0 is identity, which combined with the
    AdaLN-Zero init makes blocks start as identity transforms.
    """
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP, à la DDPM/DiT."""

    def __init__(self, d_model: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    @staticmethod
    def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[..., :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0, 1]
        # The sinusoidal embedding is computed in float32 for numerical
        # accuracy, then cast to the MLP's working dtype (which under
        # mixed-precision training will be bf16/fp16) so the linear layers
        # don't see a dtype mismatch.
        emb = self._sinusoidal_embedding(t, self.freq_dim)
        emb = emb.to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(emb)


# ─── DiT block ────────────────────────────────────────────────────────────


class DiTBlock(nn.Module):
    """DiT block with AdaLN-Zero modulation and cross-attention to conditioning.

    Structure (with AdaLN modulation from a global conditioning vector `c`):

        x = x + gate_sa  · self_attn(  modulate(norm(x), shift_sa,  scale_sa)  )
        x = x + gate_xa  · cross_attn( modulate(norm(x), shift_xa,  scale_xa), kv )
        x = x + gate_mlp · ffn(        modulate(norm(x), shift_mlp, scale_mlp) )

    The 9 modulation params come from a small SiLU+Linear MLP fed by `c`.
    The MLP weights are initialized to zero so all gates start at 0 → each
    block is the identity at init. This is the AdaLN-Zero trick from DiT.
    """

    def __init__(self, block_config: MegaTransformerBlockConfig):
        super().__init__()
        d_model = block_config.d_model
        self.d_model = d_model

        # Standard self-attention; force non-causal since image latent patches
        # have no inherent order.
        sa_config = MegaTransformerBlockConfig(**{
            **{k: v for k, v in block_config.__dict__.items()},
        })
        sa_config.causal = False
        self.self_attn = MegaTransformerAttention(sa_config)

        # Cross-attention to bridge tokens; also non-causal.
        xa_config = MegaTransformerBlockConfig(**{
            **{k: v for k, v in block_config.__dict__.items()},
        })
        xa_config.causal = False
        self.cross_attn = MegaTransformerAttention(xa_config)

        self.ffn = SimpleFFN(block_config)

        # AdaLN: 9 modulation params per block (3 stages × {shift, scale, gate}).
        # No learned affine in the layer norms — AdaLN replaces it.
        self.norm_sa = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm_xa = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm_ffn = nn.LayerNorm(d_model, elementwise_affine=False)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 9 * d_model),
        )

    def _compute_modulation(self, c: torch.Tensor) -> torch.Tensor:
        """Compute the AdaLN modulation tensor in float32 for stability.

        DiT, SD3 and Flux all keep the modulation MLP and the subsequent
        `(1 + scale)·x + shift` operation in float32 even under mixed
        precision training, because:
          1. SiLU has small derivatives near zero where modulation params
             often live, and bf16's ~3-digit precision can throw away the
             gradient signal across many training steps.
          2. `(1 + scale)·x` is a multiply-add chain whose accumulated
             rounding error can drift significantly across 12 blocks, and
             the resulting conditioning misalignment can cause loss spikes.

        We cast the input `c` and the modulation linear's weights/bias to
        float32 just for this op (the MLP is tiny — one Linear, negligible
        cost), then cast the modulation outputs back to the block's working
        dtype so the rest of the block stays in bf16 consistently.
        """
        target_dtype = c.dtype
        c_f32 = c.float()
        silu_c = F.silu(c_f32)
        lin = self.adaLN_modulation[1]
        w = lin.weight.float()
        b = lin.bias.float() if lin.bias is not None else None
        modulation = F.linear(silu_c, w, b)
        return modulation.to(dtype=target_dtype)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        kv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, n_patches, d_model) — current patch token state
            c: (B, d_model) — global modulation signal (timestep + bridge pool)
            kv: (B, n_cond, d_model) — bridge tokens (cross-attention K/V)
        """
        modulation = self._compute_modulation(c)
        (
            shift_sa, scale_sa, gate_sa,
            shift_xa, scale_xa, gate_xa,
            shift_mlp, scale_mlp, gate_mlp,
        ) = modulation.chunk(9, dim=-1)

        # Self-attention stage
        x_norm = _modulate(self.norm_sa(x), shift_sa, scale_sa)
        sa_out, _ = self.self_attn(x_norm)
        x = x + gate_sa.unsqueeze(1) * sa_out

        # Cross-attention to conditioning stage
        x_norm = _modulate(self.norm_xa(x), shift_xa, scale_xa)
        xa_out, _ = self.cross_attn(x_norm, encoder_hidden_states=kv)
        x = x + gate_xa.unsqueeze(1) * xa_out

        # FFN stage
        x_norm = _modulate(self.norm_ffn(x), shift_mlp, scale_mlp)
        ffn_out = self.ffn(x_norm)
        x = x + gate_mlp.unsqueeze(1) * ffn_out

        return x


# ─── Bridge module ────────────────────────────────────────────────────────


class _Bridge(nn.Module):
    """Q-Former-style bridge from recurrent block image positions to DiT cond.

    `n_bridge_queries` learnable queries cross-attend to the recurrent block's
    image-position outputs and produce a fixed-size sequence of conditioning
    tokens. The DiT then cross-attends to these tokens.

    The queries don't see the recurrent block's text positions directly — only
    the image positions, which already absorbed cross-modal context inside the
    recurrent block via self-attention.
    """

    def __init__(self, config: DiffusionBridgeImageDecoderConfig):
        super().__init__()
        d_model = config.d_model
        self.queries = nn.Parameter(
            torch.randn(1, config.n_bridge_queries, d_model) * (d_model ** -0.5)
        )
        self.layers = nn.ModuleList([
            MegaTransformerDecoderBlock(config.bridge_block_config)
            for _ in range(config.n_bridge_layers)
        ])

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # encoder_hidden_states: (B, n_image_positions, d_model)
        B = encoder_hidden_states.shape[0]
        q = self.queries.expand(B, -1, -1)
        for layer in self.layers:
            q, _ = layer(q, encoder_hidden_states=encoder_hidden_states)
        return q  # (B, n_bridge_queries, d_model)


# ─── DiT backbone ─────────────────────────────────────────────────────────


class _DiTBackbone(nn.Module):
    """Standard DiT backbone with patch embedding, AdaLN-Zero blocks, and unpatchify."""

    def __init__(self, config: DiffusionBridgeImageDecoderConfig):
        super().__init__()
        d_model = config.d_model
        c = config.latent_channels
        p = config.patch_size
        spatial = config.latent_spatial_size

        self.patch_size = p
        self.latent_channels = c
        self.spatial_size = spatial
        self.n_patches_per_side = spatial // p
        self.n_patches = self.n_patches_per_side ** 2

        # Patchify: Conv2d with stride=patch_size projects (B, C, H, W)
        # into (B, d_model, H/p, W/p), which then flattens to a token sequence.
        self.patch_embed = nn.Conv2d(c, d_model, kernel_size=p, stride=p)

        # Learned 2D positional embedding for the patch tokens.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))

        # Timestep embedder produces a per-sample modulation signal.
        self.t_embedder = TimestepEmbedder(d_model)

        # Stack of DiT blocks.
        self.blocks = nn.ModuleList([
            DiTBlock(config.dit_block_config) for _ in range(config.n_dit_layers)
        ])

        # Final layer: norm → AdaLN modulation → linear projection to patch dim.
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )
        self.final_linear = nn.Linear(d_model, p * p * c)

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        # (B, n_patches, p²·C) → (B, C, H, W)
        B, _, _ = x.shape
        nh = nw = self.n_patches_per_side
        p = self.patch_size
        c = self.latent_channels
        x = x.reshape(B, nh, nw, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # (B, c, nh, p, nw, p)
        return x.reshape(B, c, nh * p, nw * p)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_global: torch.Tensor,
        cond_kv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_t: (B, C, H, W) — noisy latent
            t: (B,) — timestep in [0, 1]
            cond_global: (B, d_model) — pooled bridge output for AdaLN
            cond_kv: (B, n_cond, d_model) — full bridge output for cross-attention
        """
        # Patchify
        x = self.patch_embed(x_t)             # (B, d_model, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)      # (B, n_patches, d_model)
        x = x + self.pos_embed

        # Per-sample modulation signal: timestep + global bridge pooled vector.
        t_emb = self.t_embedder(t)
        c = t_emb + cond_global

        for block in self.blocks:
            x = block(x, c, cond_kv)

        # Final modulation → linear → unpatchify.
        # The final modulation is computed in float32 for the same reason
        # as the per-block modulation (see DiTBlock._compute_modulation).
        target_dtype = c.dtype
        c_f32 = c.float()
        silu_c = F.silu(c_f32)
        fm_lin = self.final_modulation[1]
        fm_w = fm_lin.weight.float()
        fm_b = fm_lin.bias.float() if fm_lin.bias is not None else None
        final_mod = F.linear(silu_c, fm_w, fm_b).to(dtype=target_dtype)
        shift, scale = final_mod.chunk(2, dim=-1)
        x = _modulate(self.final_norm(x), shift, scale)
        x = self.final_linear(x)              # (B, n_patches, p²·C)
        return self._unpatchify(x)


# ─── Decoder module ───────────────────────────────────────────────────────


class DiffusionBridgeImageDecoder(nn.Module):
    """Drop-in image decoder using a flow-matching DiT with a bridge from the
    recurrent block.

    Forward signature matches `ImageDecoder` so the world model can dispatch
    on the config type without changing call sites:

        forward(encoder_hidden_states, encoder_attention_mask=None,
                latent_labels=None, **kwargs)
        → dict with "image_diffusion_loss" (training) and "image_latent_preds"

    During training (when `latent_labels` is provided): runs one flow-matching
    step at a randomly sampled timestep and returns the velocity-prediction MSE
    as `image_diffusion_loss`. Also returns a rough clean-latent estimate as
    `image_latent_preds` for monitoring; this is NOT a high-quality sample.

    During inference (no labels): runs `num_inference_steps` Euler steps from
    pure noise to a clean latent and returns it as `image_latent_preds`.
    """

    def __init__(self, config: DiffusionBridgeImageDecoderConfig):
        super().__init__()
        self.config = config

        self.bridge = _Bridge(config)
        self.dit = _DiTBackbone(config)

        # Latent scaling buffer (SD-style). Stored as a (1, C, 1, 1) tensor
        # so it broadcasts cleanly over (B, C, H, W) latents. Three input
        # formats accepted (see DiffusionBridgeImageDecoderConfig.latent_scale):
        #   None        → all-ones (no-op)
        #   float       → broadcast scalar to all channels
        #   list[float] → per-channel scaling
        c = config.latent_channels
        if config.latent_scale is None:
            scale_tensor = torch.ones(c)
        elif isinstance(config.latent_scale, (int, float)):
            scale_tensor = torch.full((c,), float(config.latent_scale))
        else:
            scale_tensor = torch.tensor(list(config.latent_scale), dtype=torch.float)
        # Shape (1, C, 1, 1) for broadcasting over (B, C, H, W).
        self.register_buffer("latent_scale", scale_tensor.view(1, c, 1, 1))

        self._init_weights()

    def _init_weights(self):
        # Standard Xavier on every Linear in the bridge and DiT...
        self.apply(linear_weight_init(gain=1.0))

        # ...then depth-scale the residual outputs of the bridge stack.
        apply_depth_scaled_residual_init(self.bridge.layers)

        # Re-init the DiT block residual outputs with depth scaling. Each block
        # has self_attn.o_proj, cross_attn.o_proj, and ffn.condense feeding the
        # residual stream; depth-scale all three.
        apply_depth_scaled_residual_init(self.dit.blocks)

        # Patch positional embedding: small init.
        nn.init.trunc_normal_(self.dit.pos_embed, std=0.02)

        # Patch embedding Conv2d: standard Xavier (it's a "linear" projection).
        nn.init.xavier_normal_(self.dit.patch_embed.weight, gain=1.0)
        if self.dit.patch_embed.bias is not None:
            nn.init.zeros_(self.dit.patch_embed.bias)

        # ── AdaLN-Zero (modified) ──
        # Zero out every per-block adaLN_modulation Linear so each DiT block
        # starts as the identity transform. This is the critical "Zero" part
        # of AdaLN-Zero — it gives the network depth-independent stability at
        # init.
        for block in self.dit.blocks:
            nn.init.zeros_(block.adaLN_modulation[1].weight)
            nn.init.zeros_(block.adaLN_modulation[1].bias)

        # Final modulation MLP also zeroed → final layer's shift/scale start at 0
        # (the identity modulation), but the final_linear KEEPS its standard
        # Xavier init below.
        nn.init.zeros_(self.dit.final_modulation[1].weight)
        nn.init.zeros_(self.dit.final_modulation[1].bias)

        # NOTE: We deliberately do NOT zero the final_linear. The original DiT
        # paper does, and it produces a model that outputs exactly v=0 at init
        # → bootstraps from a single layer over the first few training steps
        # before any other gradient flows. For our use case we specifically
        # want gradient to reach the bridge and from there back to the recurrent
        # block on step 1, so we keep final_linear at standard Xavier init. The
        # initial v_pred is a "random projection" of the patches — not zero,
        # but not unstable either (the AdaLN-Zero blocks above mean the actual
        # block contributions are zero, so the model is just patch_embed →
        # final_norm → final_linear at init, a single non-deep linear chain).

    # ── Conditioning ──────────────────────────────────────────────────────

    def _compute_conditioning(
        self, encoder_hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cond_kv = self.bridge(encoder_hidden_states)
        cond_global = cond_kv.mean(dim=1)  # mean-pool for global modulation
        return cond_kv, cond_global

    # ── Timestep sampling ─────────────────────────────────────────────────

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.config.timestep_sampling == "uniform":
            return torch.rand(batch_size, device=device)
        elif self.config.timestep_sampling == "logit_normal":
            # SD3/Flux-style: sample from a logit-normal distribution which
            # biases the timestep distribution toward the middle of [0, 1].
            # Mid-range timesteps benefit most from training because both the
            # high-noise and low-noise ends are easier learning signals.
            u = torch.randn(batch_size, device=device)
            return torch.sigmoid(u)
        else:
            raise ValueError(f"unknown timestep_sampling: {self.config.timestep_sampling}")

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        latent_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """Dispatch on whether labels are provided.

        - With labels: cheap flow-matching training step (one DiT forward at
          a random timestep). Used by the trainer's `compute_loss` for BOTH
          training and validation loss computation. Returns the velocity-
          prediction loss and a rough clean-latent estimate.

        - Without labels: full Euler sampling (slow, `num_inference_steps`
          DiT forwards). Used by the visualization callback's reconstruction
          path and by autoregressive generation. Returns clean sampled latents
          in the original VAE latent space.

        Note: this dispatches on `latent_labels`, NOT on `self.training`. We
        want `model.eval()` + labels (the standard validation-loss path) to
        still use the cheap training forward — only the visualization callback
        invokes the model without labels and pays the sampling cost.
        """
        if latent_labels is not None:
            return self._training_forward(encoder_hidden_states, latent_labels)
        return self._sampling_forward(encoder_hidden_states)

    def _training_forward(
        self, encoder_hidden_states: torch.Tensor, x_0: torch.Tensor
    ) -> dict:
        """Single flow-matching training step.

        - Scales `x_0` into the diffusion model's working space (~unit variance).
        - Samples a timestep `t` per batch element.
        - Forms `x_t = (1 - t) · x_0_scaled + t · noise`.
        - Predicts velocity and computes MSE against `v_target = noise - x_0_scaled`.

        The returned `image_latent_preds` is unscaled back to the original
        VAE latent space so the trainer's monitoring metrics are comparable
        to the labels.
        """
        # The dataset feeds latents as float32 even when DeepSpeed has cast the
        # model to bf16/fp16. Match the model's working dtype (taken from the
        # already-processed encoder hidden states) so the Conv2d patch_embed
        # doesn't get a dtype-mismatched input. Note we have to cast `t` too —
        # `torch.rand` returns float32 by default, and the `(1 - t) * x_0`
        # interpolation would otherwise promote x_0 back to float32 even
        # though it's already in bf16.
        dtype = encoder_hidden_states.dtype
        x_0 = x_0.to(dtype=dtype)

        B = x_0.shape[0]
        device = x_0.device

        # Scale x_0 into diffusion working space (where target std ≈ 1).
        x_0_scaled = x_0 * self.latent_scale

        t = self._sample_timesteps(B, device).to(dtype=dtype)
        t_view = t.view(-1, 1, 1, 1)

        noise = torch.randn_like(x_0_scaled)
        x_t = (1.0 - t_view) * x_0_scaled + t_view * noise
        v_target = noise - x_0_scaled

        cond_kv, cond_global = self._compute_conditioning(encoder_hidden_states)
        v_pred = self.dit(x_t, t, cond_global, cond_kv)

        loss_raw = F.mse_loss(v_pred, v_target)

        # Whiten the loss so the trivial baseline (predict zero velocity) gives
        # ~1.0, matching the whitened text and voice loss baselines. The target
        # velocity v_target = noise - x_0_scaled has var ≈ var(noise) + var(x_0_scaled)
        # ≈ 1 + 1 = 2 (since both are ~unit variance after latent scaling). Dividing
        # by this puts the diffusion loss on the same scale as the other modalities,
        # so `image_latent_loss_weight` becomes a true emphasis multiplier.
        loss = loss_raw / 2.0

        # Rough clean-latent estimate for monitoring (NOT a high quality sample).
        # In scaled space: x_0_scaled ≈ x_t - t · v_pred. Unscale to original
        # latent space so the trainer's whitened-L1 metric is meaningful.
        with torch.no_grad():
            x_0_est_scaled = x_t - t_view * v_pred
            x_0_est = x_0_est_scaled / self.latent_scale

        return {
            "image_diffusion_loss": loss,
            "image_diffusion_loss_raw": loss_raw.detach(),
            "image_latent_preds": x_0_est,
        }

    @torch.no_grad()
    def _sampling_forward(self, encoder_hidden_states: torch.Tensor) -> dict:
        """Sample a clean latent by Euler-integrating from noise.

        Sampling happens in the scaled (~unit variance) diffusion working
        space, then the final result is divided back by `latent_scale` to
        return latents in the original VAE space the dataset uses.
        """
        B = encoder_hidden_states.shape[0]
        device = encoder_hidden_states.device
        # Match the model's working dtype (bf16/fp16 under DeepSpeed) so the
        # noise tensor is compatible with the patch_embed Conv2d weights.
        dtype = encoder_hidden_states.dtype
        cfg = self.config

        cond_kv, cond_global = self._compute_conditioning(encoder_hidden_states)

        x = torch.randn(
            B, cfg.latent_channels, cfg.latent_spatial_size, cfg.latent_spatial_size,
            device=device, dtype=dtype,
        )

        n_steps = cfg.num_inference_steps
        dt = 1.0 / n_steps
        for i in range(n_steps):
            # Walk t from 1.0 down to 0 in n_steps. At t=1 the input is pure
            # noise; at t=0 it should be a clean latent (in scaled space).
            # `t` must be in the model's working dtype to avoid promoting
            # `x` back to float32 inside the DiT block computations.
            t = torch.full((B,), 1.0 - i * dt, device=device, dtype=dtype)
            v = self.dit(x, t, cond_global, cond_kv)
            x = x - dt * v

        # Unscale back to the original VAE latent space.
        x = x / self.latent_scale
        return {"image_latent_preds": x}
