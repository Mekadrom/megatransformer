"""
Recurrent VAE with adaptive exit criteria.

Follows the same recurrence patterns as MegaTransformerRecurrentBlock:
- Random iteration count sampled via Poisson log-normal each batch
- Deterministic seeding based on step count for reproducibility
- GPU synchronization via lockstep options
- Truncated backprop (n steps no grad, k steps with grad)
- Gradient injection from initial conv output
- KL-based exit criteria (using recurrent_criteria.py)
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal, Optional, Tuple, Dict, Any, List

from model import recurrent_criteria
from model.vae import VGGPerceptualLoss, LPIPSLoss
from utils.model_utils import get_activation_type


# Debug logger for numerical stability diagnostics
_logger = logging.getLogger(__name__)


def _tensor_stats(t: torch.Tensor, name: str) -> Dict[str, float]:
    """Compute statistics for a tensor."""
    with torch.no_grad():
        return {
            f"{name}_mean": t.mean().item(),
            f"{name}_std": t.std().item(),
            f"{name}_min": t.min().item(),
            f"{name}_max": t.max().item(),
            f"{name}_abs_max": t.abs().max().item(),
            f"{name}_has_nan": torch.isnan(t).any().item(),
            f"{name}_has_inf": torch.isinf(t).any().item(),
        }


def _log_tensor_stats(t: torch.Tensor, name: str, level: int = logging.DEBUG) -> None:
    """Log tensor statistics."""
    stats = _tensor_stats(t, name)
    _logger.log(
        level,
        f"{name}: mean={stats[f'{name}_mean']:.4f}, std={stats[f'{name}_std']:.4f}, "
        f"min={stats[f'{name}_min']:.4f}, max={stats[f'{name}_max']:.4f}, "
        f"abs_max={stats[f'{name}_abs_max']:.4f}, "
        f"nan={stats[f'{name}_has_nan']}, inf={stats[f'{name}_has_inf']}"
    )


# =============================================================================
# Alternative Exit Criteria (kept for easy experimentation)
# =============================================================================
# To use these instead of the recurrent_criteria.KLDivergenceCriteria:
# 1. In RecurrentEncoder.__init__, replace:
#      self.exit_criteria = recurrent_criteria.KLDivergenceCriteria(exit_threshold)
#    with:
#      self.exit_criteria = LatentKLDeltaCriteria(exit_threshold)
# 2. In RecurrentEncoder.forward, update the should_exit call as needed
# =============================================================================

class LatentKLDeltaCriteria:
    """
    Alternative encoder exit criteria: exits when the KL divergence to standard
    normal stops changing significantly between iterations.

    This measures: |KL(q(z|x) || N(0,1))_t - KL(q(z|x) || N(0,1))_{t-1}| < threshold

    Simpler than comparing distributions between iterations - just checks if the
    encoder's "confidence" (measured by KL to prior) has stabilized.
    """
    def __init__(self, threshold: float = 0.001):
        self.threshold = threshold
        self.last_kl: Optional[float] = None

    def reset(self):
        self.last_kl = None

    def compute_kl_to_prior(self, mu: torch.Tensor, logvar: torch.Tensor) -> float:
        """KL divergence from q(z|x) = N(mu, sigma^2) to p(z) = N(0, 1)."""
        kl = 0.5 * torch.mean(mu.pow(2) + logvar.exp() - 1 - logvar)
        return kl.item()

    def should_exit(self, mu: torch.Tensor, logvar: torch.Tensor) -> bool:
        """Check if KL to prior has stabilized."""
        current_kl = self.compute_kl_to_prior(mu, logvar)

        if self.last_kl is not None:
            delta = abs(current_kl - self.last_kl)
            if delta < self.threshold:
                return True

        self.last_kl = current_kl
        return False


class PreviewDeltaCriteria:
    """
    Alternative decoder exit criteria: exits when the preview reconstruction
    stops changing significantly between iterations.

    This measures: MSE(preview_t, preview_{t-1}) < threshold

    The "preview" is a lightweight 1x1 conv that estimates the final output
    at the current latent resolution, used purely for exit decision.
    """
    def __init__(self, threshold: float = 0.001):
        self.threshold = threshold
        self.last_preview: Optional[torch.Tensor] = None

    def reset(self):
        self.last_preview = None

    def should_exit(self, preview: torch.Tensor) -> Tuple[bool, float]:
        """
        Check if preview has stabilized.
        Returns (should_exit, delta) where delta is the MSE change.
        """
        if self.last_preview is not None:
            delta = F.mse_loss(preview, self.last_preview).item()
            self.last_preview = preview.detach()
            if delta < self.threshold:
                return True, delta
            return False, delta

        self.last_preview = preview.detach()
        return False, float('inf')


# =============================================================================
# Main Implementation
# =============================================================================

def _make_activation(activation_fn: str, channels: int) -> nn.Module:
    """
    Create an activation module using megatransformer_utils.get_activation_type.
    Handles parameterized activations (SwiGLU, Snake) that require channel count.
    """
    from model import activations
    activation_type = get_activation_type(activation_fn)

    # Check if it's a parameterized activation that needs channel count
    if activation_type in [activations.SwiGLU, activations.Snake]:
        return activation_type(channels)
    else:
        return activation_type()


class RecurrentConvBlock(nn.Module):
    """
    A single convolutional block that will be applied repeatedly.
    Maintains spatial dimensions.

    Numerical stability features:
    - Post-residual GroupNorm to keep activations centered over many iterations
    - Learnable residual gate (sigmoid-based) for smooth residual contribution
    - Activation clamping to prevent runaway values
    - Zero-initialized output conv for identity-like initialization
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        expansion: int = 2,
        dropout: float = 0.0,
        residual_scale: float = 1.0,  # Scale residual (1.0 = standard, <1.0 for stability)
        post_residual_norm: bool = True,  # GroupNorm after residual add for stability
        use_residual_gate: bool = True,  # Learnable gate for residual contribution
        activation_clamp: float = 50.0,  # Clamp activations to prevent overflow (0 = disabled)
        h_injection_type: Literal['additive', 'concat'] = 'additive',
        activation_fn: str = 'gelu',  # Activation function (gelu recommended for recurrent stability)
        use_final_spectral_norm: bool = False,  # Whether to apply spectral norm to final conv
    ):
        super().__init__()

        hidden = channels * expansion
        padding = kernel_size // 2
        self.residual_scale = residual_scale
        self.activation_clamp = activation_clamp
        self.h_injection_type = h_injection_type

        # Create activation modules
        input_channels = channels * 2 if h_injection_type == 'concat' else channels
        self.activation1 = _make_activation(activation_fn, input_channels)
        self.activation2 = _make_activation(activation_fn, hidden)

        if h_injection_type == 'concat':
            self.norm1 = nn.GroupNorm(min(32, channels*2), channels*2)
            self.conv1 = nn.Conv2d(channels*2, hidden, kernel_size, padding=padding)
        else:
            self.norm1 = nn.GroupNorm(min(32, channels), channels)
            self.conv1 = nn.Conv2d(channels, hidden, kernel_size, padding=padding)

        self.norm2 = nn.GroupNorm(min(32, hidden), hidden)
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size, padding=padding)

        if use_final_spectral_norm:
            # Initialize to small values before spectral norm for stable residual
            # Spectral norm will normalize spectral norm to 1, but small init
            # ensures the output starts small (identity-like behavior)
            nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.conv2.bias)
            self.conv2 = nn.utils.spectral_norm(self.conv2)
        else:
            # Initialize final conv to near-zero for stable residual
            nn.init.zeros_(self.conv2.weight)
            nn.init.zeros_(self.conv2.bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Post-residual normalization to keep values centered across iterations
        self.post_residual_norm = None
        if post_residual_norm:
            self.post_residual_norm = nn.GroupNorm(min(32, channels), channels)

        # Learnable residual gate: starts at ~0.5 (sigmoid(0) = 0.5)
        # This allows the network to learn how much of the residual update to use
        self.residual_gate = None
        if use_residual_gate:
            # Initialize to small negative value so sigmoid gives ~0.1-0.3 initially
            # This makes early iterations more stable (small updates)
            self.residual_gate = nn.Parameter(torch.full((1, channels, 1, 1), -1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x[..., :x.shape[1] // 2, :, :] if self.h_injection_type == 'concat' else x

        x = self.norm1(x)
        x = self.activation1(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.activation2(x)
        x = self.dropout(x)
        x = self.conv2(x)

        # Apply residual scaling
        if self.residual_scale != 1.0:
            x = self.residual_scale * x

        # Apply learnable residual gate (element-wise per channel)
        if self.residual_gate is not None:
            gate = torch.sigmoid(self.residual_gate)
            x = gate * x

        # Residual connection
        x = residual + x

        # Post-residual normalization for stability across many iterations
        if self.post_residual_norm is not None:
            x = self.post_residual_norm(x)

        # Activation clamping to prevent numerical instability
        if self.activation_clamp > 0:
            x = torch.clamp(x, -self.activation_clamp, self.activation_clamp)

        return x


class RecurrentEncoder(nn.Module):
    """
    Recurrent VAE encoder with KL-based exit criteria.

    Architecture:
        input → init_conv (downsample) → [recurrent_block × N] → to_latent → μ, logσ²

    The recurrent_block is applied repeatedly with:
    - Same parameters each iteration
    - Truncated backprop (n steps no grad, k steps with grad)
    - Gradient injection from init_conv output
    - Exit when KL divergence between consecutive latent estimates is below threshold
    """
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        latent_channels: int = 4,
        downsample_factor: int = 8,  # 256 → 32
        recurrent_expansion: int = 2,
        dropout: float = 0.0,
        mean_iterations: int = 16,
        backprop_depth: int = 8,
        exit_threshold: float = 0.01,
        gradient_injection_scale: float = 0.1,
        lockstep_n: bool = True,
        lockstep_k: bool = True,
        h_injection_type: Literal['additive', 'concat'] = 'additive',
        use_injection_scale: bool = True,  # Whether to scale injection over iterations
        debug: bool = False,  # Enable debug logging for numerical stability diagnostics
        # Numerical stability settings
        post_residual_norm: bool = True,  # GroupNorm after residual in recurrent block
        use_residual_gate: bool = True,  # Learnable gate for residual contribution
        activation_clamp: float = 50.0,  # Clamp activations (0 = disabled)
        activation_fn: str = 'gelu',  # Activation function (gelu recommended for recurrent stability)
    ):
        super().__init__()

        self.mean_iterations = mean_iterations
        self.backprop_depth = backprop_depth
        self.exit_threshold = exit_threshold
        self.gradient_injection_scale = gradient_injection_scale
        self.lockstep_n = lockstep_n
        self.lockstep_k = lockstep_k
        self.h_injection_type = h_injection_type
        self.use_injection_scale = use_injection_scale
        self.debug = debug

        # Step counter for deterministic seeding
        self.step = 0

        # Exit criteria
        self.exit_criteria = recurrent_criteria.KLDivergenceCriteria(exit_threshold)

        # Initial conv: downsample and project to hidden channels
        num_downsamples = int(math.log2(downsample_factor))

        init_layers = []
        ch_in = in_channels
        ch_out = hidden_channels // (2 ** (num_downsamples - 1))

        for i in range(num_downsamples):
            if i == num_downsamples - 1:
                ch_out = hidden_channels
            init_layers.extend([
                nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(min(32, ch_out), ch_out),
                _make_activation(activation_fn, ch_out),
            ])
            ch_in = ch_out
            if i < num_downsamples - 2:
                ch_out = ch_out * 2

        # Add final centering normalization after init_conv for zero-centered output
        # This ensures h_init is well-conditioned before recurrent iterations
        init_layers.append(nn.GroupNorm(min(32, hidden_channels), hidden_channels))

        self.init_conv = nn.Sequential(*init_layers)

        # Recurrent block (same params used repeatedly)
        self.recurrent_block = RecurrentConvBlock(
            hidden_channels,
            expansion=recurrent_expansion,
            dropout=dropout,
            post_residual_norm=post_residual_norm,
            use_residual_gate=use_residual_gate,
            activation_clamp=activation_clamp,
            h_injection_type=h_injection_type,
            activation_fn=activation_fn,
        )

        # To latent: μ and logσ² (these are "preview" heads for exit criteria)
        self.to_mu = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1)

        # Initialize to_logvar to produce small variance initially
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.constant_(self.to_logvar.bias, -2.0)  # exp(-2) ≈ 0.14 std

    def n_k_steps(self):
        """
        Sample (n, k) steps using Poisson log-normal distribution.
        Matches MegaTransformerRecurrentBlock.n_k_steps for consistency.

        n = number of steps without gradient (detached)
        k = number of steps with gradient (backprop enabled)
        """
        seed_n = 514229 + self.step
        seed_k = 317811 + self.step

        if not self.lockstep_n and torch.distributed.is_initialized():
            seed_n = seed_n * (torch.distributed.get_rank() + 1)
        if not self.lockstep_k and torch.distributed.is_initialized():
            seed_k = seed_k * (torch.distributed.get_rank() + 1)

        n_generator = torch.Generator(device="cpu")
        n_generator.manual_seed(seed_n % (2**31 - 1))
        k_generator = torch.Generator(device="cpu")
        k_generator.manual_seed(seed_k % (2**31 - 1))

        t = max(self.mean_iterations - self.backprop_depth, 0)
        s = self.backprop_depth

        if self.training:
            # Poisson log-normal sampling
            sigma = 0.5
            mu = math.log(t + s) - (sigma**2 / 2)
            rate = torch.zeros((1,)).log_normal_(mean=mu, std=sigma, generator=n_generator)
            p = torch.poisson(torch.tensor([rate], dtype=torch.float), generator=n_generator) + 1
            n = torch.clamp(p - s, min=0)
            k = torch.as_tensor(torch.minimum(torch.as_tensor(s), p))
            self.step += 1
        else:
            # At inference, use fixed mean_iterations with no gradient
            n, k = torch.tensor(self.mean_iterations), torch.tensor(0)

        return n.to(torch.long).item(), k.to(torch.long).item()

    def compute_kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence from standard normal.

        Uses sum over spatial dims + channels, mean over batch (matches regular VAE).
        KL(N(mu, sigma^2) || N(0, 1)) = 0.5 * (mu^2 + sigma^2 - 1 - log(sigma^2))
        """
        # Sum over channels and spatial dimensions [1, 2, 3], mean over batch [0]
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])
        return torch.mean(kl)

    def forward(
        self,
        x: torch.Tensor,
        force_n: Optional[int] = None,
        force_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            x: Input image [B, C, H, W]
            force_n: If set, use this many no-grad iterations
            force_k: If set, use this many with-grad iterations

        Returns:
            mu: Latent mean [B, latent_channels, H', W']
            logvar: Latent log-variance [B, latent_channels, H', W']
            info: Dict with iteration counts, KL history, etc.
        """
        # Sample or use forced iteration counts
        if force_n is not None and force_k is not None:
            n_steps, k_steps = force_n, force_k
        else:
            n_steps, k_steps = self.n_k_steps()

        # Initial projection + downsample
        h = self.init_conv(x)
        h_init = h  # Save for gradient injection

        # Debug: collect per-iteration stats
        debug_stats: List[Dict[str, Any]] = [] if self.debug else None

        if self.debug:
            _logger.debug(f"[Encoder] Starting forward: n_steps={n_steps}, k_steps={k_steps}")
            _log_tensor_stats(x, "enc_input")
            _log_tensor_stats(h, "enc_h_init")

        kl_history = []
        last_logvar = None
        actual_n_steps = n_steps

        # Track total iteration for proper injection scaling
        total_iter = 0

        # Phase 1: n steps without gradient
        if n_steps > 0:
            with torch.no_grad():
                for i in range(n_steps):
                    if self.h_injection_type == "additive":
                        # Gradient injection with decay: scale by 1/(iter+1) to prevent accumulation
                        # First iter gets full scale, subsequent iters get diminishing contribution
                        if self.use_injection_scale:
                            injection_scale = self.gradient_injection_scale / (total_iter + 1)
                        else:
                            injection_scale = 1.0
                        h = h + injection_scale * h_init
                    elif self.h_injection_type == "concat":
                        h = torch.cat([h, h_init], dim=1)
                    total_iter += 1
                    

                    # Apply recurrent block
                    h = self.recurrent_block(h)

                    # Debug logging
                    if self.debug:
                        iter_stats = {
                            "phase": "n",
                            "iter": total_iter,
                            "injection_scale": injection_scale,
                            **_tensor_stats(h, "h"),
                        }
                        debug_stats.append(iter_stats)
                        _logger.debug(
                            f"[Encoder] n-phase iter {total_iter}: "
                            f"injection={injection_scale:.4f}, "
                            f"h_abs_max={iter_stats['h_abs_max']:.4f}, "
                            f"h_std={iter_stats['h_std']:.4f}"
                        )
                        # Warn on potential instability
                        if iter_stats["h_has_nan"] or iter_stats["h_has_inf"]:
                            _logger.warning(f"[Encoder] NaN/Inf detected at n-phase iter {total_iter}!")
                        if iter_stats["h_abs_max"] > 100:
                            _logger.warning(f"[Encoder] Large activation at n-phase iter {total_iter}: abs_max={iter_stats['h_abs_max']:.2f}")

                    # Compute current latent estimate for exit criteria
                    logvar = self.to_logvar(h)

                    # Check exit criteria
                    if last_logvar is not None:
                        # Use log_softmax for KL divergence input
                        curr_log = F.log_softmax(logvar.flatten(1), dim=-1)
                        last_log = F.log_softmax(last_logvar.flatten(1), dim=-1)
                        if self.exit_criteria.should_exit(last_log, curr_log):
                            actual_n_steps = i + 1
                            if self.debug:
                                _logger.debug(f"[Encoder] Early exit at n-phase iter {i+1}")
                            break

                    last_logvar = logvar
                    kl_history.append(self.compute_kl(self.to_mu(h), logvar).item())

        # Phase 2: k steps with gradient
        actual_k_steps = k_steps
        for i in range(k_steps):
            if self.h_injection_type == "additive":
                # Gradient injection with decay (continues from phase 1's count)
                injection_scale = self.gradient_injection_scale / (total_iter + 1)
                h = h + injection_scale * h_init
            elif self.h_injection_type == "concat":
                h = torch.cat([h, h_init], dim=1)
            total_iter += 1

            # Apply recurrent block
            h = self.recurrent_block(h)

            # Debug logging
            if self.debug:
                iter_stats = {
                    "phase": "k",
                    "iter": total_iter,
                    "injection_scale": injection_scale,
                    **_tensor_stats(h, "h"),
                }
                debug_stats.append(iter_stats)
                _logger.debug(
                    f"[Encoder] k-phase iter {total_iter}: "
                    f"injection={injection_scale:.4f}, "
                    f"h_abs_max={iter_stats['h_abs_max']:.4f}, "
                    f"h_std={iter_stats['h_std']:.4f}"
                )
                if iter_stats["h_has_nan"] or iter_stats["h_has_inf"]:
                    _logger.warning(f"[Encoder] NaN/Inf detected at k-phase iter {total_iter}!")
                if iter_stats["h_abs_max"] > 100:
                    _logger.warning(f"[Encoder] Large activation at k-phase iter {total_iter}: abs_max={iter_stats['h_abs_max']:.2f}")

            # Compute current latent estimate
            logvar = self.to_logvar(h)

            # Check exit criteria (but don't break during grad phase for stable training)
            if last_logvar is not None:
                curr_log = F.log_softmax(logvar.flatten(1), dim=-1)
                last_log = F.log_softmax(last_logvar.flatten(1), dim=-1)
                # We track but don't exit during grad phase
                _ = self.exit_criteria.should_exit(last_log, curr_log)

            last_logvar = logvar
            kl_history.append(self.compute_kl(self.to_mu(h), logvar).item())

        # Final latent
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)

        if self.debug:
            _log_tensor_stats(mu, "enc_mu")
            _log_tensor_stats(logvar, "enc_logvar")
            _logger.debug(f"[Encoder] Completed: total_iters={total_iter}, kl_final={kl_history[-1] if kl_history else 0:.4f}")

        info = {
            "n_steps": actual_n_steps,
            "k_steps": actual_k_steps,
            "total_iterations": actual_n_steps + actual_k_steps,
            "kl_history": kl_history,
            "kl_final": kl_history[-1] if kl_history else 0.0,
        }
        if self.debug:
            info["debug_stats"] = debug_stats

        return mu, logvar, info


class RecurrentDecoder(nn.Module):
    """
    Recurrent VAE decoder with output-delta exit criteria.

    Architecture:
        z → init_conv → [recurrent_block × N] → final_conv (upsample) → output

    Exit when output stops changing significantly between iterations.
    """
    def __init__(
        self,
        out_channels: int = 3,
        hidden_channels: int = 128,
        latent_channels: int = 4,
        upsample_factor: int = 8,  # 32 → 256
        recurrent_expansion: int = 2,
        dropout: float = 0.0,
        mean_iterations: int = 16,
        backprop_depth: int = 8,
        exit_threshold: float = 0.001,
        gradient_injection_scale: float = 0.1,
        lockstep_n: bool = True,
        lockstep_k: bool = True,
        h_injection_type: Literal['additive', 'concat'] = 'additive',
        debug: bool = False,  # Enable debug logging for numerical stability diagnostics
        # Numerical stability settings
        post_residual_norm: bool = True,  # GroupNorm after residual in recurrent block
        use_residual_gate: bool = True,  # Learnable gate for residual contribution
        activation_clamp: float = 50.0,  # Clamp activations (0 = disabled)
        activation_fn: str = 'gelu',  # Activation function (gelu recommended for recurrent stability)
    ):
        super().__init__()

        self.mean_iterations = mean_iterations
        self.backprop_depth = backprop_depth
        self.exit_threshold = exit_threshold
        self.gradient_injection_scale = gradient_injection_scale
        self.lockstep_n = lockstep_n
        self.lockstep_k = lockstep_k
        self.h_injection_type = h_injection_type
        self.debug = debug

        # Step counter for deterministic seeding
        self.step = 0

        # Initial conv: project latent to hidden channels
        # Ends with GroupNorm for zero-centered output (stable for recurrent iterations)
        self.init_conv = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, hidden_channels), hidden_channels),
            _make_activation(activation_fn, hidden_channels),
            # Final centering normalization for zero-centered h_init
            nn.GroupNorm(min(32, hidden_channels), hidden_channels),
        )

        # Recurrent block (same params used repeatedly)
        self.recurrent_block = RecurrentConvBlock(
            hidden_channels,
            expansion=recurrent_expansion,
            dropout=dropout,
            post_residual_norm=post_residual_norm,
            use_residual_gate=use_residual_gate,
            activation_clamp=activation_clamp,
            h_injection_type=h_injection_type,
            activation_fn=activation_fn,
        )

        # Preview conv: get current reconstruction estimate (for exit criteria)
        self.preview_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

        # Final conv: upsample to output resolution
        num_upsamples = int(math.log2(upsample_factor))

        final_layers = []
        ch_in = hidden_channels

        for i in range(num_upsamples):
            ch_out = ch_in // 2 if i < num_upsamples - 1 else ch_in
            final_layers.extend([
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
                nn.GroupNorm(min(32, ch_out), ch_out),
                _make_activation(activation_fn, ch_out) if i < num_upsamples - 1 else nn.Identity(),
            ])
            ch_in = ch_out

        final_layers.append(nn.Conv2d(ch_in, out_channels, kernel_size=3, padding=1))
        self.final_conv = nn.Sequential(*final_layers)

    def n_k_steps(self):
        """
        Sample (n, k) steps using Poisson log-normal distribution.
        Uses different seed offset than encoder to allow independent sampling.
        """
        # Different base seeds than encoder
        seed_n = 832040 + self.step
        seed_k = 1346269 + self.step

        if not self.lockstep_n and torch.distributed.is_initialized():
            seed_n = seed_n * (torch.distributed.get_rank() + 1)
        if not self.lockstep_k and torch.distributed.is_initialized():
            seed_k = seed_k * (torch.distributed.get_rank() + 1)

        n_generator = torch.Generator(device="cpu")
        n_generator.manual_seed(seed_n % (2**31 - 1))
        k_generator = torch.Generator(device="cpu")
        k_generator.manual_seed(seed_k % (2**31 - 1))

        t = max(self.mean_iterations - self.backprop_depth, 0)
        s = self.backprop_depth

        if self.training:
            sigma = 0.5
            mu = math.log(t + s) - (sigma**2 / 2)
            rate = torch.zeros((1,)).log_normal_(mean=mu, std=sigma, generator=n_generator)
            p = torch.poisson(torch.tensor([rate], dtype=torch.float), generator=n_generator) + 1
            n = torch.clamp(p - s, min=0)
            k = torch.as_tensor(torch.minimum(torch.as_tensor(s), p))
            self.step += 1
        else:
            n, k = torch.tensor(self.mean_iterations), torch.tensor(0)

        return n.to(torch.long).item(), k.to(torch.long).item()

    def forward(
        self,
        z: torch.Tensor,
        force_n: Optional[int] = None,
        force_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            z: Latent tensor [B, latent_channels, H', W']
            force_n: If set, use this many no-grad iterations
            force_k: If set, use this many with-grad iterations

        Returns:
            output: Reconstructed image [B, out_channels, H, W]
            info: Dict with iteration counts, delta history, etc.
        """
        if force_n is not None and force_k is not None:
            n_steps, k_steps = force_n, force_k
        else:
            n_steps, k_steps = self.n_k_steps()

        # Initial projection
        h = self.init_conv(z)
        h_init = h

        # Debug: collect per-iteration stats
        debug_stats: List[Dict[str, Any]] = [] if self.debug else None

        if self.debug:
            _logger.debug(f"[Decoder] Starting forward: n_steps={n_steps}, k_steps={k_steps}")
            _log_tensor_stats(z, "dec_input_z")
            _log_tensor_stats(h, "dec_h_init")

        delta_history = []
        preview_prev = None
        actual_n_steps = n_steps

        # Track total iteration for proper injection scaling
        total_iter = 0

        # Phase 1: n steps without gradient
        if n_steps > 0:
            with torch.no_grad():
                for i in range(n_steps):
                    if self.h_injection_type == "additive":
                        # Gradient injection with decay: scale by 1/(iter+1) to prevent accumulation
                        injection_scale = self.gradient_injection_scale / (total_iter + 1)
                        h = h + injection_scale * h_init
                    elif self.h_injection_type == "concat":
                        h = torch.cat([h, h_init], dim=1)
                    total_iter += 1

                    h = self.recurrent_block(h)

                    # Debug logging
                    if self.debug:
                        iter_stats = {
                            "phase": "n",
                            "iter": total_iter,
                            "injection_scale": injection_scale,
                            **_tensor_stats(h, "h"),
                        }
                        debug_stats.append(iter_stats)
                        _logger.debug(
                            f"[Decoder] n-phase iter {total_iter}: "
                            f"injection={injection_scale:.4f}, "
                            f"h_abs_max={iter_stats['h_abs_max']:.4f}, "
                            f"h_std={iter_stats['h_std']:.4f}"
                        )
                        if iter_stats["h_has_nan"] or iter_stats["h_has_inf"]:
                            _logger.warning(f"[Decoder] NaN/Inf detected at n-phase iter {total_iter}!")
                        if iter_stats["h_abs_max"] > 100:
                            _logger.warning(f"[Decoder] Large activation at n-phase iter {total_iter}: abs_max={iter_stats['h_abs_max']:.2f}")

                    # Get preview for exit criteria
                    preview = self.preview_conv(h)

                    if preview_prev is not None:
                        delta = F.mse_loss(preview, preview_prev).item()
                        delta_history.append(delta)
                        if delta < self.exit_threshold:
                            actual_n_steps = i + 1
                            if self.debug:
                                _logger.debug(f"[Decoder] Early exit at n-phase iter {i+1}, delta={delta:.6f}")
                            break

                    preview_prev = preview

        # Phase 2: k steps with gradient
        actual_k_steps = k_steps
        for i in range(k_steps):
            if self.h_injection_type == "additive":
                # Gradient injection with decay (continues from phase 1's count)
                injection_scale = self.gradient_injection_scale / (total_iter + 1)
                h = h + injection_scale * h_init
            elif self.h_injection_type == "concat":
                h = torch.cat([h, h_init], dim=1)
            total_iter += 1

            h = self.recurrent_block(h)

            # Debug logging
            if self.debug:
                iter_stats = {
                    "phase": "k",
                    "iter": total_iter,
                    "injection_scale": injection_scale,
                    **_tensor_stats(h, "h"),
                }
                debug_stats.append(iter_stats)
                _logger.debug(
                    f"[Decoder] k-phase iter {total_iter}: "
                    f"injection={injection_scale:.4f}, "
                    f"h_abs_max={iter_stats['h_abs_max']:.4f}, "
                    f"h_std={iter_stats['h_std']:.4f}"
                )
                if iter_stats["h_has_nan"] or iter_stats["h_has_inf"]:
                    _logger.warning(f"[Decoder] NaN/Inf detected at k-phase iter {total_iter}!")
                if iter_stats["h_abs_max"] > 100:
                    _logger.warning(f"[Decoder] Large activation at k-phase iter {total_iter}: abs_max={iter_stats['h_abs_max']:.2f}")

            preview = self.preview_conv(h)
            if preview_prev is not None:
                delta = F.mse_loss(preview, preview_prev.detach()).item()
                delta_history.append(delta)

            preview_prev = preview

        # Final upsampling to full resolution
        output = self.final_conv(h)

        if self.debug:
            _log_tensor_stats(output, "dec_output")
            _logger.debug(f"[Decoder] Completed: total_iters={total_iter}")

        info = {
            "n_steps": actual_n_steps,
            "k_steps": actual_k_steps,
            "total_iterations": actual_n_steps + actual_k_steps,
            "delta_history": delta_history,
        }
        if self.debug:
            info["debug_stats"] = debug_stats

        return output, info


class RecurrentVAE(nn.Module):
    """
    Full Recurrent VAE with adaptive computation.

    Follows MegaTransformerRecurrentBlock patterns:
    - Random iteration count via Poisson log-normal each batch
    - Deterministic seeding based on step count for reproducibility
    - GPU sync via lockstep options
    - Truncated backprop (n steps no grad, k steps with grad)
    - Gradient injection to preserve flow to early layers
    - KL-based exit criteria for encoder
    - Output-delta exit criteria for decoder
    """
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        latent_channels: int = 4,
        scale_factor: int = 8,
        recurrent_expansion: int = 2,
        dropout: float = 0.0,
        # Recurrence settings
        mean_iterations: int = 16,
        backprop_depth: int = 8,
        encoder_exit_threshold: float = 0.01,
        decoder_exit_threshold: float = 0.001,
        gradient_injection_scale: float = 0.1,
        # Lockstep settings (sync iterations across GPUs)
        lockstep_n: bool = True,
        lockstep_k: bool = True,
        h_injection_type: Literal['additive', 'concat'] = 'additive',
        use_injection_scale: bool = True,  # Whether to scale injection over iterations
        # Numerical stability settings (recurrent block)
        post_residual_norm: bool = True,  # GroupNorm after residual in recurrent block
        use_residual_gate: bool = True,  # Learnable gate for residual contribution
        activation_clamp: float = 50.0,  # Clamp activations (0 = disabled)
        # Numerical stability settings (latent space)
        use_latent_norm: bool = True,  # Normalize latent z before decoder
        logvar_clamp_max: float = 4.0,  # Max logvar (std=exp(2)≈7.4), prevents channel blowup
        # Loss weights
        kl_weight: float = 1e-6,
        perceptual_loss_weight: float = 0.1,
        iteration_cost_weight: float = 0.0,  # Penalize using more iterations (ACT-style)
        # Perceptual loss settings
        perceptual_loss_type: str = "vgg",  # "vgg", "lpips", or "none"
        lpips_net: str = "alex",  # "alex", "vgg", or "squeeze" (only used if perceptual_loss_type="lpips")
        # Debug mode
        debug: bool = False,  # Enable debug logging for numerical stability diagnostics
        # Activation function
        activation_fn: str = 'gelu',  # Activation function (gelu recommended for recurrent stability)
    ):
        super().__init__()

        self.logvar_clamp_max = logvar_clamp_max

        self.kl_weight = kl_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.iteration_cost_weight = iteration_cost_weight
        self.mean_iterations = mean_iterations
        self._debug = debug

        # Perceptual loss
        self.perceptual_loss_type = perceptual_loss_type
        if perceptual_loss_type == "vgg":
            self.perceptual_loss = VGGPerceptualLoss()
        elif perceptual_loss_type == "lpips":
            self.perceptual_loss = LPIPSLoss(net=lpips_net)
        else:
            self.perceptual_loss = None

        # Latent space normalization to prevent channel-specific blowup
        # GroupNorm on z ensures no single channel dominates before decoder
        if use_latent_norm:
            self.latent_norm = nn.GroupNorm(
                num_groups=min(4, latent_channels),  # Few groups for small channel count
                num_channels=latent_channels,
                affine=True,  # Learnable scale/shift
            )
        else:
            self.latent_norm = None

        self.encoder = RecurrentEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            downsample_factor=scale_factor,
            recurrent_expansion=recurrent_expansion,
            dropout=dropout,
            mean_iterations=mean_iterations,
            backprop_depth=backprop_depth,
            exit_threshold=encoder_exit_threshold,
            gradient_injection_scale=gradient_injection_scale,
            lockstep_n=lockstep_n,
            lockstep_k=lockstep_k,
            h_injection_type=h_injection_type,
            use_injection_scale=use_injection_scale,
            debug=debug,
            post_residual_norm=post_residual_norm,
            use_residual_gate=use_residual_gate,
            activation_clamp=activation_clamp,
            activation_fn=activation_fn,
        )

        self.decoder = RecurrentDecoder(
            out_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            upsample_factor=scale_factor,
            recurrent_expansion=recurrent_expansion,
            dropout=dropout,
            mean_iterations=mean_iterations,
            backprop_depth=backprop_depth,
            exit_threshold=decoder_exit_threshold,
            gradient_injection_scale=gradient_injection_scale,
            lockstep_n=lockstep_n,
            lockstep_k=lockstep_k,
            h_injection_type=h_injection_type,
            use_injection_scale=use_injection_scale,
            debug=debug,
            post_residual_norm=post_residual_norm,
            use_residual_gate=use_residual_gate,
            activation_clamp=activation_clamp,
            activation_fn=activation_fn,
        )

    @property
    def debug(self) -> bool:
        """Get debug mode status."""
        return self._debug

    @debug.setter
    def debug(self, value: bool) -> None:
        """Set debug mode for encoder and decoder."""
        self._debug = value
        self.encoder.debug = value
        self.decoder.debug = value

    def set_debug(self, enabled: bool = True) -> "RecurrentVAE":
        """
        Enable or disable debug logging for numerical stability diagnostics.

        When enabled, logs per-iteration statistics including:
        - Hidden state mean, std, min, max, abs_max
        - NaN/Inf detection with warnings
        - Large activation warnings (abs_max > 100)
        - Injection scale values
        - Early exit events

        To see debug output, set the logger level:
            import logging
            logging.getLogger('model.image.recurrent_vae').setLevel(logging.DEBUG)

        Args:
            enabled: Whether to enable debug logging

        Returns:
            self (for chaining)
        """
        self.debug = enabled
        return self

    def sync_steps(self):
        """Sync encoder and decoder step counters (call after loading checkpoint)."""
        max_step = max(self.encoder.step, self.decoder.step)
        self.encoder.step = max_step
        self.decoder.step = max_step

    def encode(
        self,
        x: torch.Tensor,
        force_n: Optional[int] = None,
        force_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Encode input to latent distribution."""
        return self.encoder(x, force_n=force_n, force_k=force_k)

    def decode(
        self,
        z: torch.Tensor,
        force_n: Optional[int] = None,
        force_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Decode latent to output."""
        return self.decoder(z, force_n=force_n, force_k=force_k)

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reparameterization trick with logvar clamping and latent normalization.

        Args:
            mu: Mean of latent distribution [B, C, H, W]
            logvar: Log-variance of latent distribution [B, C, H, W]

        Returns:
            Sampled latent z (normalized if latent_norm is enabled)
        """
        # Clamp logvar to prevent numerical instability
        # logvar_clamp_max=4.0 gives max std of exp(2) ≈ 7.4
        # This prevents specific channels from having huge variance
        logvar = torch.clamp(logvar, min=-10.0, max=self.logvar_clamp_max)

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu  # Use mean at inference

        # Debug: log per-channel stats to identify problematic channels
        if self._debug:
            with torch.no_grad():
                for c in range(z.shape[1]):
                    ch_data = z[:, c]
                    ch_max = ch_data.abs().max().item()
                    if ch_max > 10.0:
                        _logger.warning(
                            f"[Latent] Channel {c} has large values: "
                            f"abs_max={ch_max:.2f}, mean={ch_data.mean():.4f}, std={ch_data.std():.4f}"
                        )

        # Normalize latent to prevent channel-specific blowup
        # This keeps z well-conditioned before entering decoder
        if self.latent_norm is not None:
            z = self.latent_norm(z)

        return z

    def forward(
        self,
        x: torch.Tensor,
        force_encoder_n: Optional[int] = None,
        force_encoder_k: Optional[int] = None,
        force_decoder_n: Optional[int] = None,
        force_decoder_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Full forward pass.

        Returns:
            recon: Reconstructed image
            info: Dict with losses, iteration counts, etc.
        """
        # Encode
        mu, logvar, enc_info = self.encode(
            x, force_n=force_encoder_n, force_k=force_encoder_k
        )

        # Sample latent
        z = self.reparameterize(mu, logvar)

        # Decode
        recon, dec_info = self.decode(
            z, force_n=force_decoder_n, force_k=force_decoder_k
        )

        # Compute losses
        recon_loss = F.mse_loss(recon, x)
        kl_loss = self.encoder.compute_kl(mu, logvar)

        # Perceptual loss
        perceptual_loss = torch.tensor(0.0, device=x.device)
        if self.perceptual_loss is not None:
            perceptual_loss = self.perceptual_loss(recon, x)

        # Optional: penalize using more iterations (encourages early exit)
        if self.iteration_cost_weight > 0:
            enc_iter_cost = enc_info["total_iterations"] / (2 * self.mean_iterations)
            dec_iter_cost = dec_info["total_iterations"] / (2 * self.mean_iterations)
            iteration_cost = (enc_iter_cost + dec_iter_cost) / 2
        else:
            iteration_cost = 0.0

        total_loss = (
            recon_loss +
            self.kl_weight * kl_loss +
            self.perceptual_loss_weight * perceptual_loss +
            self.iteration_cost_weight * iteration_cost
        )

        info = {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "perceptual_loss": perceptual_loss,
            "encoder_n_steps": enc_info["n_steps"],
            "encoder_k_steps": enc_info["k_steps"],
            "encoder_iterations": enc_info["total_iterations"],
            "decoder_n_steps": dec_info["n_steps"],
            "decoder_k_steps": dec_info["k_steps"],
            "decoder_iterations": dec_info["total_iterations"],
            "kl_history": enc_info["kl_history"],
            "delta_history": dec_info["delta_history"],
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

        return recon, info


# Factory functions for different sizes
def recurrent_vae_tiny(**kwargs) -> RecurrentVAE:
    """~1M params - for quick experiments."""
    defaults = dict(
        hidden_channels=64,
        latent_channels=4,
        recurrent_expansion=2,
        mean_iterations=8,
        backprop_depth=4,
    )
    defaults.update(kwargs)
    return RecurrentVAE(**defaults)


def recurrent_vae_small(**kwargs) -> RecurrentVAE:
    """~4M params - reasonable quality."""
    defaults = dict(
        hidden_channels=128,
        latent_channels=4,
        recurrent_expansion=2,
        mean_iterations=12,
        backprop_depth=6,
    )
    defaults.update(kwargs)
    return RecurrentVAE(**defaults)


def recurrent_vae_base(**kwargs) -> RecurrentVAE:
    """~10M params - good quality."""
    defaults = dict(
        hidden_channels=192,
        latent_channels=8,
        recurrent_expansion=2,
        mean_iterations=16,
        backprop_depth=8,
    )
    defaults.update(kwargs)
    return RecurrentVAE(**defaults)


# Model config lookup (matches your existing pattern)
model_config_lookup = {
    "recurrent_vae_tiny": recurrent_vae_tiny,
    "recurrent_vae_small": recurrent_vae_small,
    "recurrent_vae_base": recurrent_vae_base,
}


if __name__ == "__main__":
    # Quick test
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Testing RecurrentVAE...")

    model = recurrent_vae_small().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass (training mode - random iterations)
    model.train()
    x = torch.randn(2, 3, 256, 256, device=device)

    recon, info = model(x)

    print(f"\nTraining mode (random iterations):")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {recon.shape}")
    print(f"Encoder: n={info['encoder_n_steps']}, k={info['encoder_k_steps']}, total={info['encoder_iterations']}")
    print(f"Decoder: n={info['decoder_n_steps']}, k={info['decoder_k_steps']}, total={info['decoder_iterations']}")
    print(f"Recon loss: {info['recon_loss']:.4f}")
    print(f"KL loss: {info['kl_loss']:.4f}")

    # Test backward pass
    print("\nTesting backward pass...")
    info["loss"].backward()
    print("Backward pass successful!")

    # Check gradient flow
    enc_init_grad = model.encoder.init_conv[0].weight.grad
    dec_init_grad = model.decoder.init_conv[0].weight.grad
    print(f"Encoder init_conv grad norm: {enc_init_grad.norm():.6f}")
    print(f"Decoder init_conv grad norm: {dec_init_grad.norm():.6f}")

    # Test inference mode (fixed iterations)
    model.eval()
    with torch.no_grad():
        recon, info = model(x)

    print(f"\nInference mode (fixed mean_iterations):")
    print(f"Encoder: n={info['encoder_n_steps']}, k={info['encoder_k_steps']}, total={info['encoder_iterations']}")
    print(f"Decoder: n={info['decoder_n_steps']}, k={info['decoder_k_steps']}, total={info['decoder_iterations']}")

    # Test with forced iterations
    print("\nWith forced iterations (n=2, k=3 each):")
    model.train()
    with torch.no_grad():
        recon, info = model(x, force_encoder_n=2, force_encoder_k=3, force_decoder_n=2, force_decoder_k=3)
    print(f"Encoder: n={info['encoder_n_steps']}, k={info['encoder_k_steps']}")
    print(f"Decoder: n={info['decoder_n_steps']}, k={info['decoder_k_steps']}")

    # Test iteration sampling consistency
    print("\nTesting iteration sampling (5 batches):")
    model.train()
    for i in range(5):
        with torch.no_grad():
            _, info = model(x)
        print(f"  Batch {i}: enc={info['encoder_iterations']}, dec={info['decoder_iterations']}")
