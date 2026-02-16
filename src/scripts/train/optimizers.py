"""
Combined Muon + AdamW optimizer for training neural networks.

Muon (MomentUm Orthogonalized by Newton-schulz) is designed for 2D+ parameters
(linear weights, conv filters) while AdamW handles 1D parameters (biases, norms).

Based on Keller Jordan's Muon implementation: https://github.com/KellerJordan/Muon

Usage:
    optimizer = MuonAdamW(
        model.named_parameters(),
        lr_muon=0.02,
        lr_adamw=1e-4,
        # Optional: specify boundary layers to keep on AdamW
        first_layer_names=["conv_subsample.conv.0"],
        last_layer_names=["asr_head", "speaker_classifier"],
    )
"""

from typing import Optional, Iterable, Tuple, List, Set, Callable
import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the orthogonalization of G.

    Uses a quintic polynomial iteration with coefficients that maximize
    convergence rate at the fixed point. Produces approximately U @ S' @ V.T
    where S' is diagonal with values roughly between 0.5 and 1.5.

    Args:
        G: Input tensor of shape [..., M, N] where M, N >= 1
        steps: Number of Newton-Schulz iterations (default 5)

    Returns:
        Orthogonalized tensor of same shape as G
    """
    assert G.ndim >= 2, f"Expected 2D+ tensor, got {G.ndim}D"

    # Quintic polynomial coefficients (optimized for convergence)
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Work in bfloat16 for efficiency
    X = G.bfloat16()

    # Transpose if M > N for numerical stability
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT

    # Normalize spectral norm to at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.mT

    return X


def muon_update(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> torch.Tensor:
    """
    Compute Muon update: momentum + Newton-Schulz orthogonalization.

    Args:
        grad: Parameter gradient
        momentum_buffer: Momentum buffer (modified in-place)
        beta: Momentum coefficient
        ns_steps: Newton-Schulz iteration steps
        nesterov: Whether to use Nesterov momentum

    Returns:
        Orthogonalized update tensor
    """
    # Update momentum buffer: buf = beta * buf + (1 - beta) * grad
    momentum_buffer.lerp_(grad, 1 - beta)

    # Nesterov: use grad + beta * momentum, else just momentum
    if nesterov:
        update = grad.lerp_(momentum_buffer, beta)
    else:
        update = momentum_buffer.clone()

    # Flatten conv filters (4D) to 2D for orthogonalization
    original_shape = update.shape
    if update.ndim == 4:
        update = update.view(update.size(0), -1)
    elif update.ndim == 3:
        # Conv1d: [out_ch, in_ch, kernel] -> [out_ch, in_ch * kernel]
        update = update.view(update.size(0), -1)

    # Orthogonalize via Newton-Schulz
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)

    # Scale by sqrt(M/N) to match RMS of original gradient
    update = update * max(1, update.size(-2) / update.size(-1)) ** 0.5

    # Reshape back to original shape
    update = update.view(original_shape)

    return update.to(grad.dtype)


class Muon(Optimizer):
    """
    Muon optimizer for 2D+ parameters (matrices, conv filters).

    Applies momentum with Newton-Schulz orthogonalization, which prevents
    gradient collapse into low-rank subspaces.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.02)
        momentum: Momentum coefficient (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        ns_steps: Newton-Schulz iterations (default: 5)
        weight_decay: Weight decay coefficient (default: 0.0)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Initialize state
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                # Apply weight decay (decoupled, like AdamW)
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)

                # Compute orthogonalized update
                update = muon_update(
                    grad,
                    state["momentum_buffer"],
                    beta=momentum,
                    ns_steps=ns_steps,
                    nesterov=nesterov,
                )

                # Apply update
                p.add_(update, alpha=-lr)

        return loss


class MuonAdamW(Optimizer):
    """
    Combined Muon + AdamW optimizer.

    Automatically routes parameters to the appropriate optimizer:
    - Muon: 2D+ parameters (Linear weights, Conv1d/Conv2d filters)
    - AdamW: 1D parameters (biases, LayerNorm, GroupNorm, BatchNorm, embeddings)

    Boundary layers (first/last) can be kept on AdamW for stability.

    Args:
        named_params: Iterator of (name, param) tuples from model.named_parameters()
        lr_muon: Learning rate for Muon parameters (default: 0.02)
        lr_adamw: Learning rate for AdamW parameters (default: 1e-4)
        momentum_muon: Momentum for Muon (default: 0.95)
        betas_adamw: Betas for AdamW (default: (0.9, 0.999))
        weight_decay_muon: Weight decay for Muon params (default: 0.0)
        weight_decay_adamw: Weight decay for AdamW params (default: 0.01)
        ns_steps: Newton-Schulz iterations for Muon (default: 5)
        nesterov: Use Nesterov momentum in Muon (default: True)
        first_layer_names: List of substrings to match first layer param names (kept on AdamW)
        last_layer_names: List of substrings to match last layer param names (kept on AdamW)
        adamw_override_names: Additional param name substrings to force onto AdamW
        verbose: Print parameter routing information (default: False)

    Example:
        optimizer = MuonAdamW(
            model.named_parameters(),
            lr_muon=0.02,
            lr_adamw=1e-4,
            first_layer_names=["conv_subsample.conv.0"],
            last_layer_names=["asr_head", "classifier"],
        )
    """

    def __init__(
        self,
        named_params: Iterable[Tuple[str, torch.nn.Parameter]],
        lr_muon: float = 0.02,
        lr_adamw: float = 1e-4,
        momentum_muon: float = 0.95,
        betas_adamw: Tuple[float, float] = (0.9, 0.999),
        eps_adamw: float = 1e-8,
        weight_decay_muon: float = 0.0,
        weight_decay_adamw: float = 0.01,
        ns_steps: int = 5,
        nesterov: bool = True,
        first_layer_names: Optional[List[str]] = None,
        last_layer_names: Optional[List[str]] = None,
        adamw_override_names: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        self.lr_muon = lr_muon
        self.lr_adamw = lr_adamw
        self.momentum_muon = momentum_muon
        self.betas_adamw = betas_adamw
        self.eps_adamw = eps_adamw
        self.weight_decay_muon = weight_decay_muon
        self.weight_decay_adamw = weight_decay_adamw
        self.ns_steps = ns_steps
        self.nesterov = nesterov
        self.verbose = verbose

        # Boundary layer patterns
        self.first_layer_names = set(first_layer_names or [])
        self.last_layer_names = set(last_layer_names or [])
        self.adamw_override_names = set(adamw_override_names or [])

        # Patterns that always go to AdamW (norms, biases, embeddings)
        self.adamw_patterns = {
            "bias", "LayerNorm", "layernorm", "layer_norm",
            "GroupNorm", "groupnorm", "group_norm",
            "BatchNorm", "batchnorm", "batch_norm",
            "Embedding", "embedding", "embed",
            ".norm.", "_norm.", "final_norm",
        }

        # Collect and route parameters
        muon_params = []
        adamw_params = []

        named_params_list = list(named_params)

        for name, param in named_params_list:
            if not param.requires_grad:
                continue

            route = self._route_parameter(name, param)

            if route == "muon":
                muon_params.append(param)
                if verbose:
                    print(f"[Muon]  {name}: {tuple(param.shape)}")
            else:
                adamw_params.append(param)
                if verbose:
                    print(f"[AdamW] {name}: {tuple(param.shape)}")

        if verbose:
            muon_count = sum(p.numel() for p in muon_params)
            adamw_count = sum(p.numel() for p in adamw_params)
            print(f"\nMuon params: {len(muon_params)} tensors, {muon_count:,} parameters")
            print(f"AdamW params: {len(adamw_params)} tensors, {adamw_count:,} parameters")

        # Store param lists for the optimizers
        self.muon_params = muon_params
        self.adamw_params = adamw_params

        # Create sub-optimizers
        self.muon_optimizer = None
        if muon_params:
            self.muon_optimizer = Muon(
                muon_params,
                lr=lr_muon,
                momentum=momentum_muon,
                nesterov=nesterov,
                ns_steps=ns_steps,
                weight_decay=weight_decay_muon,
            )

        self.adamw_optimizer = None
        if adamw_params:
            self.adamw_optimizer = AdamW(
                adamw_params,
                lr=lr_adamw,
                betas=betas_adamw,
                eps=eps_adamw,
                weight_decay=weight_decay_adamw,
            )

        # Initialize parent with all params (for compatibility)
        all_params = muon_params + adamw_params
        defaults = dict(lr=lr_muon)  # Nominal, actual LRs are in sub-optimizers
        super().__init__(all_params, defaults)

    def _route_parameter(self, name: str, param: torch.nn.Parameter) -> str:
        """
        Determine whether a parameter should use Muon or AdamW.

        Returns:
            "muon" or "adamw"
        """
        # 1D params always go to AdamW
        if param.ndim < 2:
            return "adamw"

        # Check explicit AdamW overrides
        for pattern in self.adamw_override_names:
            if pattern in name:
                return "adamw"

        # Check first/last layer patterns
        for pattern in self.first_layer_names:
            if pattern in name:
                return "adamw"

        for pattern in self.last_layer_names:
            if pattern in name:
                return "adamw"

        # Check norm/bias/embedding patterns
        for pattern in self.adamw_patterns:
            if pattern in name:
                return "adamw"

        # 2D+ params go to Muon
        return "muon"

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step with both optimizers."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.muon_optimizer is not None:
            self.muon_optimizer.step()

        if self.adamw_optimizer is not None:
            self.adamw_optimizer.step()

        return loss

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients for all parameters."""
        if self.muon_optimizer is not None:
            self.muon_optimizer.zero_grad(set_to_none=set_to_none)
        if self.adamw_optimizer is not None:
            self.adamw_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """Return combined state dict."""
        return {
            "muon": self.muon_optimizer.state_dict() if self.muon_optimizer else None,
            "adamw": self.adamw_optimizer.state_dict() if self.adamw_optimizer else None,
        }

    def load_state_dict(self, state_dict):
        """Load combined state dict."""
        if self.muon_optimizer is not None and state_dict.get("muon") is not None:
            self.muon_optimizer.load_state_dict(state_dict["muon"])
        if self.adamw_optimizer is not None and state_dict.get("adamw") is not None:
            self.adamw_optimizer.load_state_dict(state_dict["adamw"])

    @property
    def param_groups(self):
        """Return combined param groups for compatibility."""
        groups = []
        if self.muon_optimizer is not None:
            for g in self.muon_optimizer.param_groups:
                groups.append({**g, "optimizer": "muon"})
        if self.adamw_optimizer is not None:
            for g in self.adamw_optimizer.param_groups:
                groups.append({**g, "optimizer": "adamw"})
        return groups

    def get_lr(self) -> dict:
        """Get current learning rates."""
        return {
            "muon": self.lr_muon,
            "adamw": self.lr_adamw,
        }

    def set_lr(self, lr_muon: Optional[float] = None, lr_adamw: Optional[float] = None):
        """Set learning rates."""
        if lr_muon is not None and self.muon_optimizer is not None:
            self.lr_muon = lr_muon
            for g in self.muon_optimizer.param_groups:
                g["lr"] = lr_muon

        if lr_adamw is not None and self.adamw_optimizer is not None:
            self.lr_adamw = lr_adamw
            for g in self.adamw_optimizer.param_groups:
                g["lr"] = lr_adamw


def create_muon_adamw_optimizer(
    model: nn.Module,
    lr_muon: float = 0.02,
    lr_adamw: float = 1e-4,
    weight_decay_muon: float = 0.0,
    weight_decay_adamw: float = 0.01,
    momentum_muon: float = 0.95,
    ns_steps: int = 5,
    first_layer_names: Optional[List[str]] = None,
    last_layer_names: Optional[List[str]] = None,
    verbose: bool = False,
) -> MuonAdamW:
    """
    Convenience function to create a MuonAdamW optimizer from a model.

    Args:
        model: PyTorch model
        lr_muon: Muon learning rate
        lr_adamw: AdamW learning rate
        weight_decay_muon: Muon weight decay
        weight_decay_adamw: AdamW weight decay
        momentum_muon: Muon momentum
        ns_steps: Newton-Schulz iterations
        first_layer_names: Patterns for first layer params (kept on AdamW)
        last_layer_names: Patterns for last layer params (kept on AdamW)
        verbose: Print routing info

    Returns:
        Configured MuonAdamW optimizer
    """
    return MuonAdamW(
        model.named_parameters(),
        lr_muon=lr_muon,
        lr_adamw=lr_adamw,
        weight_decay_muon=weight_decay_muon,
        weight_decay_adamw=weight_decay_adamw,
        momentum_muon=momentum_muon,
        ns_steps=ns_steps,
        first_layer_names=first_layer_names,
        last_layer_names=last_layer_names,
        verbose=verbose,
    )