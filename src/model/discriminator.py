import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_adaptive_weight(
    nll_loss: torch.Tensor,
    g_loss: torch.Tensor,
    last_layer: nn.Parameter,
    discriminator_weight: float = 1.0,
) -> torch.Tensor:
    """
    Compute adaptive discriminator weight (VQGAN-style).

    This balances the GAN loss contribution with the reconstruction loss by
    computing the ratio of their gradient norms with respect to the last
    decoder layer. This prevents the discriminator from dominating training.

    Reference: Esser et al., "Taming Transformers for High-Resolution Image Synthesis"
    https://arxiv.org/abs/2012.09841

    Args:
        nll_loss: Reconstruction loss (MSE, L1, perceptual, etc.) - must have grad enabled
        g_loss: Generator's GAN loss - must have grad enabled
        last_layer: The last layer's weight parameter (e.g., decoder.final_conv.weight)
        discriminator_weight: Base discriminator weight to scale by

    Returns:
        Adaptive weight to multiply with the GAN loss
    """
    # Compute gradients of reconstruction loss w.r.t. last decoder layer
    nll_grads = torch.autograd.grad(
        nll_loss, last_layer, retain_graph=True, allow_unused=True
    )[0]

    # Compute gradients of GAN loss w.r.t. last decoder layer
    g_grads = torch.autograd.grad(
        g_loss, last_layer, retain_graph=True, allow_unused=True
    )[0]

    # Handle case where gradients are None (shouldn't happen in normal training)
    if nll_grads is None or g_grads is None:
        return torch.tensor(discriminator_weight, device=g_loss.device)

    # Compute adaptive weight as ratio of gradient norms
    # This ensures GAN gradients are scaled to match reconstruction gradient magnitude
    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)

    # Clamp to prevent extreme values
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()

    return discriminator_weight * d_weight


def discriminator_loss(
    disc_real_outputs: list[torch.Tensor],
    disc_fake_outputs: list[torch.Tensor],
) -> torch.Tensor:
    """
    Discriminator loss: real samples should be classified as 1, fake as 0.
    Uses least-squares GAN loss.
    """
    loss = 0.0
    for dr, df in zip(disc_real_outputs, disc_fake_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        f_loss = torch.mean(df ** 2)
        loss += r_loss + f_loss
    return loss / (2 * len(disc_real_outputs))


def discriminator_hinge_loss(
    disc_real_outputs: list[torch.Tensor],
    disc_fake_outputs: list[torch.Tensor],
) -> torch.Tensor:
    """
    Discriminator hinge loss.
    Real samples should produce positive values, fake should produce negative.
    """
    loss = 0.0
    for real, fake in zip(disc_real_outputs, disc_fake_outputs):
        loss += torch.mean(F.relu(1 - real))
        loss += torch.mean(F.relu(1 + fake))
    return loss / (2 * len(disc_real_outputs))


def compute_adaptive_weight(
    nll_loss: torch.Tensor,
    g_loss: torch.Tensor,
    last_parameters: nn.Parameter,
    discriminator_weight: float = 1.0,
) -> torch.Tensor:
    """
    Compute adaptive discriminator weight (VQGAN-style).

    This balances the GAN loss contribution with the reconstruction loss by
    computing the ratio of their gradient norms with respect to the last
    decoder layer. This prevents the discriminator from dominating training.

    Reference: Esser et al., "Taming Transformers for High-Resolution Image Synthesis"
    https://arxiv.org/abs/2012.09841

    Args:
        nll_loss: Reconstruction loss (MSE, L1, perceptual, etc.) - must have grad enabled
        g_loss: Generator's GAN loss - must have grad enabled
        last_layer: The last layer's weight parameter (e.g., decoder.final_conv.weight)
        discriminator_weight: Base discriminator weight to scale by

    Returns:
        Adaptive weight to multiply with the GAN loss
    """
    # Compute gradients of reconstruction loss w.r.t. last decoder layer
    nll_grads = torch.autograd.grad(
        nll_loss, last_parameters, retain_graph=True, allow_unused=True
    )[0]

    # Compute gradients of GAN loss w.r.t. last decoder layer
    g_grads = torch.autograd.grad(
        g_loss, last_parameters, retain_graph=True, allow_unused=True
    )[0]

    # Handle case where gradients are None (shouldn't happen in normal training)
    if nll_grads is None or g_grads is None:
        return torch.tensor(discriminator_weight, device=g_loss.device)

    # Compute adaptive weight as ratio of gradient norms
    # This ensures GAN gradients are scaled to match reconstruction gradient magnitude
    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)

    # Clamp to prevent extreme values
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()

    return discriminator_weight * d_weight
