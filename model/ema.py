"""
Exponential Moving Average (EMA) for model weights.

EMA maintains a shadow copy of model weights that is updated as:
    ema_weights = decay * ema_weights + (1 - decay) * model_weights

This provides smoother, more stable weights for inference, which is
particularly important for diffusion models.
"""

import copy
from typing import Optional

import torch
import torch.nn as nn


class EMAModel:
    """
    Exponential Moving Average of model weights.

    Usage:
        model = MyModel()
        ema = EMAModel(model, decay=0.9999)

        # During training
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            ema.update()  # Update EMA weights

        # For inference/sampling
        with ema.apply_ema():
            samples = model.sample(...)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_after_step: int = 0,
        update_every: int = 1,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: The model to track
            decay: EMA decay rate (0.9999 is typical for diffusion)
            update_after_step: Don't update EMA until after this many steps
            update_every: Update EMA every N steps (1 = every step)
            device: Device to store EMA weights (None = same as model)
        """
        self.model = model
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.device = device

        self.step = 0

        # Create shadow weights
        self.shadow_params = {}
        self._init_shadow_params()

        # For context manager
        self._backup_params = None

    def _init_shadow_params(self):
        """Initialize shadow parameters as a copy of model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.device is not None:
                    self.shadow_params[name] = param.data.clone().to(self.device)
                else:
                    self.shadow_params[name] = param.data.clone()

    def get_decay(self, step: int) -> float:
        """
        Get decay rate, optionally with warmup.

        Uses inverse decay warmup: decay increases from 0 to target over early steps.
        """
        if step < self.update_after_step:
            return 0.0

        # Optional: warmup decay (can help stability in early training)
        # Ramp from 0.9 to target decay over first 1000 updates
        warmup_steps = 1000
        if step < self.update_after_step + warmup_steps:
            progress = (step - self.update_after_step) / warmup_steps
            return self.decay * progress + 0.9 * (1 - progress)

        return self.decay

    @torch.no_grad()
    def update(self):
        """Update EMA weights. Call after each optimizer step."""
        self.step += 1

        if self.step % self.update_every != 0:
            return

        decay = self.get_decay(self.step)

        if decay == 0.0:
            # Just copy weights directly
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    self.shadow_params[name].copy_(param.data)
            return

        # EMA update: shadow = decay * shadow + (1 - decay) * param
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.shadow_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    def apply_shadow(self):
        """Apply EMA weights to model (for inference)."""
        self._backup_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self._backup_params[name] = param.data.clone()
                param.data.copy_(self.shadow_params[name])

    def restore(self):
        """Restore original weights (after inference)."""
        if self._backup_params is None:
            return

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self._backup_params:
                param.data.copy_(self._backup_params[name])

        self._backup_params = None

    def apply_ema(self):
        """Context manager for using EMA weights."""
        return EMAContext(self)

    def state_dict(self) -> dict:
        """Get state dict for saving."""
        return {
            'shadow_params': {k: v.cpu() for k, v in self.shadow_params.items()},
            'step': self.step,
            'decay': self.decay,
        }

    def load_state_dict(self, state_dict: dict):
        """Load state dict."""
        self.step = state_dict.get('step', 0)
        self.decay = state_dict.get('decay', self.decay)

        shadow_params = state_dict.get('shadow_params', {})
        for name, param in shadow_params.items():
            if name in self.shadow_params:
                device = self.shadow_params[name].device
                self.shadow_params[name].copy_(param.to(device))

    def to(self, device: torch.device):
        """Move EMA weights to device."""
        self.device = device
        for name in self.shadow_params:
            self.shadow_params[name] = self.shadow_params[name].to(device)
        return self


class EMAContext:
    """Context manager for temporarily applying EMA weights."""

    def __init__(self, ema: EMAModel):
        self.ema = ema

    def __enter__(self):
        self.ema.apply_shadow()
        return self.ema.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ema.restore()
        return False