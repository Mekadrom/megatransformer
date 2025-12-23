#!/usr/bin/env python3
"""
Targeted overfitting test for image diffusion on actual VAE latents.
Tests whether the model can memorize small subsets of real data.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.image.diffusion import model_config_lookup


def load_real_data(cache_dir: str, num_samples: int = 100):
    """Load actual VAE latents and text embeddings from cache."""
    pt_files = sorted(glob(os.path.join(cache_dir, "*.pt")))[:num_samples]

    latents = []
    conditions = []

    for f in pt_files:
        data = torch.load(f, weights_only=True)
        latents.append(data["latent_mu"])
        conditions.append(data["text_embeddings"])

    latents = torch.stack(latents)
    conditions = torch.stack(conditions)

    print(f"Loaded {len(latents)} samples")
    print(f"Latent shape: {latents.shape}, mean={latents.mean():.4f}, std={latents.std():.4f}")
    print(f"Condition shape: {conditions.shape}, mean={conditions.mean():.4f}, std={conditions.std():.4f}")

    return latents, conditions


def test_overfitting(
    model_name: str,
    cache_dir: str,
    num_samples: int = 10,
    num_steps: int = 1000,
    lr: float = 1e-3,
    batch_size: int = None,
    device: str = "cuda"
):
    """Test if model can overfit to a small number of real samples."""
    print(f"\n{'='*70}")
    print(f"OVERFITTING TEST: {model_name}")
    print(f"Samples: {num_samples}, Steps: {num_steps}, LR: {lr}, Batch: {batch_size or num_samples}")
    print(f"{'='*70}")

    # Load data
    latents, conditions = load_real_data(cache_dir, num_samples)
    latents = latents.to(device)
    conditions = conditions.to(device)

    # Use full batch by default (true overfitting)
    if batch_size is None:
        batch_size = num_samples

    # Create model
    context_dim = conditions.shape[-1]
    model = model_config_lookup[model_name](
        latent_channels=4,
        num_timesteps=1000,
        sampling_timesteps=20,
        betas_schedule="cosine",
        context_dim=context_dim,
        normalize=False,
        min_snr_loss_weight=False,
        cfg_dropout_prob=0.0,  # Disable for overfitting test
        zero_terminal_snr=False,
        offset_noise_strength=0.0,
        timestep_sampling="uniform",
    )
    model.train()
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer with no weight decay for overfitting
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)

    losses = []
    best_loss = float('inf')

    pbar = tqdm(range(num_steps), desc="Training")
    for step in pbar:
        # Sample batch (with replacement if batch_size > num_samples)
        if batch_size >= num_samples:
            batch_idx = torch.arange(num_samples)
        else:
            batch_idx = torch.randint(0, num_samples, (batch_size,))

        x_0 = latents[batch_idx]
        cond = conditions[batch_idx]

        optimizer.zero_grad()
        output, loss = model(x_0, condition=cond)
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        best_loss = min(best_loss, loss_val)

        if step % 100 == 0:
            pbar.set_postfix({
                'loss': f'{loss_val:.4f}',
                'best': f'{best_loss:.4f}',
                'out_std': f'{output.std().item():.3f}',
                'grad': f'{grad_norm:.3f}'
            })

    # Results
    print(f"\nResults:")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Best loss: {best_loss:.4f} (at step {losses.index(best_loss)})")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Improvement: {(losses[0] - best_loss) / losses[0] * 100:.1f}%")

    if best_loss < 0.05:
        print(f"  ✓ EXCELLENT: Model can fully memorize {num_samples} samples")
    elif best_loss < 0.1:
        print(f"  ✓ GOOD: Model nearly memorized {num_samples} samples")
    elif best_loss < 0.2:
        print(f"  ~ PARTIAL: Model is learning but not fully memorizing")
    else:
        print(f"  ✗ POOR: Model struggles to memorize {num_samples} samples")

    return losses, best_loss


def test_conditioning_ablation(
    model_name: str,
    cache_dir: str,
    num_samples: int = 10,
    num_steps: int = 500,
    lr: float = 1e-3,
    device: str = "cuda"
):
    """Test if conditioning helps or hurts memorization."""
    print(f"\n{'='*70}")
    print(f"CONDITIONING ABLATION: {model_name}")
    print(f"{'='*70}")

    # Load data
    latents, conditions = load_real_data(cache_dir, num_samples)
    latents = latents.to(device)
    conditions = conditions.to(device)

    context_dim = conditions.shape[-1]

    results = {}

    for cond_mode in ["with_real_cond", "with_zero_cond", "with_random_cond", "unconditional"]:
        print(f"\n--- Testing: {cond_mode} ---")

        model = model_config_lookup[model_name](
            latent_channels=4,
            num_timesteps=1000,
            sampling_timesteps=20,
            betas_schedule="cosine",
            context_dim=context_dim,
            normalize=False,
            min_snr_loss_weight=False,
            cfg_dropout_prob=0.0,
            zero_terminal_snr=False,
            offset_noise_strength=0.0,
            timestep_sampling="uniform",
        )
        model.train()
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0)

        losses = []
        for step in tqdm(range(num_steps), desc=cond_mode):
            x_0 = latents

            if cond_mode == "with_real_cond":
                cond = conditions
            elif cond_mode == "with_zero_cond":
                cond = torch.zeros_like(conditions)
            elif cond_mode == "with_random_cond":
                cond = torch.randn_like(conditions) * 0.2  # Match real std
            else:  # unconditional
                cond = None

            optimizer.zero_grad()
            output, loss = model(x_0, condition=cond)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        best_loss = min(losses)
        results[cond_mode] = best_loss
        print(f"  Best loss: {best_loss:.4f}")

    print(f"\n{'='*70}")
    print("ABLATION SUMMARY:")
    print(f"{'='*70}")
    for mode, loss in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {mode}: {loss:.4f}")

    return results


def test_lr_sweep(
    model_name: str,
    cache_dir: str,
    num_samples: int = 10,
    num_steps: int = 300,
    device: str = "cuda"
):
    """Find optimal learning rate for this model/data combo."""
    print(f"\n{'='*70}")
    print(f"LEARNING RATE SWEEP: {model_name}")
    print(f"{'='*70}")

    learning_rates = [5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5]

    results = {}

    for lr in learning_rates:
        _, best_loss = test_overfitting(
            model_name=model_name,
            cache_dir=cache_dir,
            num_samples=num_samples,
            num_steps=num_steps,
            lr=lr,
            device=device
        )
        results[lr] = best_loss

    print(f"\n{'='*70}")
    print("LR SWEEP SUMMARY:")
    print(f"{'='*70}")
    for lr, loss in sorted(results.items(), key=lambda x: x[1]):
        print(f"  LR={lr:.0e}: best_loss={loss:.4f}")

    best_lr = min(results.keys(), key=lambda x: results[x])
    print(f"\nBest LR: {best_lr:.0e} (loss={results[best_lr]:.4f})")

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Find cache directory
    cache_dirs = glob("cached_datasets/*diffusion*latent*")
    if not cache_dirs:
        print("No diffusion latent cache found!")
        return

    cache_dir = cache_dirs[0]
    print(f"Using cache: {cache_dir}")

    model_name = "balanced_small_image_diffusion"

    # Test 1: Can the model overfit to 10 samples with high LR?
    test_overfitting(
        model_name=model_name,
        cache_dir=cache_dir,
        num_samples=10,
        num_steps=1000,
        lr=1e-3,  # Higher LR
        device=device
    )

    # Test 2: LR sweep to find optimal
    test_lr_sweep(
        model_name=model_name,
        cache_dir=cache_dir,
        num_samples=10,
        num_steps=300,
        device=device
    )

    # Test 3: Does conditioning help or hurt?
    test_conditioning_ablation(
        model_name=model_name,
        cache_dir=cache_dir,
        num_samples=10,
        num_steps=500,
        lr=1e-3,
        device=device
    )


if __name__ == "__main__":
    main()
