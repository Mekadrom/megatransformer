#!/usr/bin/env python3
"""
Compute latent statistics (mean, std) from a trained VAE for diffusion training.

This script loads a trained VAE checkpoint, encodes a dataset, and computes
the mean and standard deviation of the latent representations. These values
can be passed to diffusion training scripts via --latent_mean and --latent_std
to normalize latents to zero-mean unit-variance.

Usage (Image VAE):
    python scripts/compute_latent_stats.py \
        --modality image \
        --vae_checkpoint runs/image_vae/my_run/checkpoint-10000 \
        --vae_config small \
        --cache_dir cached_datasets/my_dataset_vae_cached \
        --num_samples 10000 \
        --batch_size 32

Usage (Audio VAE):
    python scripts/compute_latent_stats.py \
        --modality audio \
        --vae_checkpoint runs/audio_vae/my_run/checkpoint-10000 \
        --vae_config medium \
        --cache_dir cached_datasets/my_audio_dataset_vae_cached \
        --num_samples 10000 \
        --batch_size 16

Output:
    Prints latent statistics and the command line args to use for diffusion training.
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_image_vae(checkpoint_path: str, config: str, latent_channels: int = 4) -> nn.Module:
    """Load image VAE from checkpoint."""
    from model.image.vae import model_config_lookup
    from utils.model_loading_utils import load_model

    if config not in model_config_lookup:
        raise ValueError(f"Unknown image VAE config: {config}. Available: {list(model_config_lookup.keys())}")

    model = model_config_lookup[config](latent_channels=latent_channels)
    model, _ = load_model(False, model, checkpoint_path)
    return model


def load_audio_vae(checkpoint_path: str, config: str, latent_channels: int = 32) -> nn.Module:
    """Load audio VAE from checkpoint."""
    from model.audio.vae import model_config_lookup
    from utils.model_loading_utils import load_model

    if config not in model_config_lookup:
        raise ValueError(f"Unknown audio VAE config: {config}. Available: {list(model_config_lookup.keys())}")

    model = model_config_lookup[config](latent_channels=latent_channels)
    model, _ = load_model(False, model, checkpoint_path)
    return model


def load_recurrent_vae(checkpoint_path: str, config: str, latent_channels: int = 4) -> nn.Module:
    """Load recurrent VAE from checkpoint."""
    from model.image.recurrent_vae import model_config_lookup
    from utils.model_loading_utils import load_model

    if config not in model_config_lookup:
        raise ValueError(f"Unknown recurrent VAE config: {config}. Available: {list(model_config_lookup.keys())}")

    model = model_config_lookup[config](latent_channels=latent_channels)
    model, _ = load_model(False, model, checkpoint_path)
    return model


def compute_latent_stats(
    model: nn.Module,
    dataloader: DataLoader,
    modality: str,
    num_samples: int,
    device: torch.device,
    use_fp16: bool = False,
) -> Tuple[float, float, dict]:
    """
    Compute mean and std of latent representations.

    Returns:
        mean: Global mean of latents
        std: Global std of latents
        stats: Dict with per-channel and other detailed statistics
    """
    model.eval()
    model.to(device)

    all_means = []
    all_vars = []
    all_mins = []
    all_maxs = []
    channel_means = None
    channel_vars = None
    total_samples = 0

    dtype = torch.float16 if use_fp16 else torch.float32

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing latent statistics"):
            if total_samples >= num_samples:
                break

            # Get input based on modality
            if modality == "image":
                x = batch["image"].to(device, dtype=dtype)
            elif modality == "audio":
                x = batch["spectrogram"].to(device, dtype=dtype)
            else:
                raise ValueError(f"Unknown modality: {modality}")

            # Encode to get latent mean (mu)
            if hasattr(model, 'encode'):
                # Standard VAE or RecurrentVAE
                result = model.encode(x)
                if isinstance(result, tuple):
                    if len(result) == 2:
                        mu, logvar = result
                    else:
                        mu, logvar, _ = result  # RecurrentVAE returns info dict
                else:
                    mu = result
            elif hasattr(model, 'encoder'):
                # Direct encoder access
                result = model.encoder(x)
                if isinstance(result, tuple):
                    mu, logvar = result[:2]
                else:
                    mu = result
            else:
                raise ValueError("Model doesn't have encode() or encoder attribute")

            # Convert to float32 for statistics
            mu = mu.float()

            # Compute batch statistics
            batch_mean = mu.mean().item()
            batch_var = mu.var().item()
            batch_min = mu.min().item()
            batch_max = mu.max().item()

            all_means.append(batch_mean)
            all_vars.append(batch_var)
            all_mins.append(batch_min)
            all_maxs.append(batch_max)

            # Per-channel statistics (running mean)
            # Shape: [B, C, ...] -> compute mean over batch and spatial dims
            if mu.dim() >= 2:
                num_channels = mu.shape[1]
                # Flatten all dims except channel: [B, C, *] -> [B, C, -1] -> mean over [0, 2]
                mu_flat = mu.flatten(2) if mu.dim() > 2 else mu.unsqueeze(-1)
                batch_channel_means = mu_flat.mean(dim=(0, 2))  # [C]
                batch_channel_vars = mu_flat.var(dim=(0, 2))  # [C]

                if channel_means is None:
                    channel_means = batch_channel_means.cpu()
                    channel_vars = batch_channel_vars.cpu()
                else:
                    # Welford's online algorithm for running mean/var
                    n = total_samples
                    m = x.shape[0]
                    delta = batch_channel_means.cpu() - channel_means
                    channel_means = channel_means + delta * m / (n + m)
                    # Simplified variance update (approximate)
                    channel_vars = (channel_vars * n + batch_channel_vars.cpu() * m) / (n + m)

            total_samples += x.shape[0]

    # Compute global statistics
    global_mean = sum(all_means) / len(all_means)
    global_std = (sum(all_vars) / len(all_vars)) ** 0.5
    global_min = min(all_mins)
    global_max = max(all_maxs)

    stats = {
        "global_mean": global_mean,
        "global_std": global_std,
        "global_min": global_min,
        "global_max": global_max,
        "num_samples": total_samples,
        "channel_means": channel_means.tolist() if channel_means is not None else None,
        "channel_stds": (channel_vars ** 0.5).tolist() if channel_vars is not None else None,
    }

    return global_mean, global_std, stats


def main():
    parser = argparse.ArgumentParser(description="Compute latent statistics from trained VAE")
    parser.add_argument("--modality", type=str, required=True, choices=["image", "audio"],
                        help="Modality type (image or audio)")
    parser.add_argument("--vae_checkpoint", type=str, required=True,
                        help="Path to VAE checkpoint directory")
    parser.add_argument("--vae_config", type=str, required=True,
                        help="VAE config name (e.g., 'small', 'medium')")
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Path to cached dataset directory")
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Number of samples to use for statistics (default: 10000)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for encoding (default: 32)")
    parser.add_argument("--latent_channels", type=int, default=None,
                        help="Number of latent channels (default: 4 for image, 32 for audio)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (default: cuda if available)")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 for encoding")
    parser.add_argument("--recurrent", action="store_true",
                        help="Use recurrent VAE instead of standard VAE (image only)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional: save statistics to JSON file")

    args = parser.parse_args()

    device = torch.device(args.device)

    # Set default latent channels based on modality
    if args.latent_channels is None:
        args.latent_channels = 4 if args.modality == "image" else 32

    print(f"Computing latent statistics for {args.modality} VAE")
    print(f"  Checkpoint: {args.vae_checkpoint}")
    print(f"  Config: {args.vae_config}")
    print(f"  Dataset: {args.cache_dir}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Latent channels: {args.latent_channels}")
    print(f"  Device: {device}")
    print()

    # Load model
    print("Loading VAE model...")
    if args.modality == "image":
        if args.recurrent:
            model = load_recurrent_vae(args.vae_checkpoint, args.vae_config, args.latent_channels)
        else:
            model = load_image_vae(args.vae_checkpoint, args.vae_config, args.latent_channels)
    else:
        model = load_audio_vae(args.vae_checkpoint, args.vae_config, args.latent_channels)

    # Load dataset
    print("Loading dataset...")
    if args.modality == "image":
        from dataset_loading.image_vae_dataset import CachedImageVAEDataset, ImageVAEDataCollator
        dataset = CachedImageVAEDataset(cache_dir=args.cache_dir)
        collator = ImageVAEDataCollator()
    else:
        from dataset_loading.audio_vae_dataset import CachedAudioVAEDataset, AudioVAEDataCollator
        dataset = CachedAudioVAEDataset(cache_dir=args.cache_dir)
        collator = AudioVAEDataCollator()

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,  # Shuffle to get representative samples
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )

    # Compute statistics
    print(f"Encoding {min(args.num_samples, len(dataset))} samples...")
    mean, std, stats = compute_latent_stats(
        model=model,
        dataloader=dataloader,
        modality=args.modality,
        num_samples=args.num_samples,
        device=device,
        use_fp16=args.fp16,
    )

    # Print results
    print()
    print("=" * 60)
    print("LATENT STATISTICS")
    print("=" * 60)
    print(f"  Global Mean: {mean:.6f}")
    print(f"  Global Std:  {std:.6f}")
    print(f"  Global Min:  {stats['global_min']:.6f}")
    print(f"  Global Max:  {stats['global_max']:.6f}")
    print(f"  Samples:     {stats['num_samples']}")
    print()

    if stats['channel_means'] is not None:
        print("Per-channel statistics:")
        for i, (ch_mean, ch_std) in enumerate(zip(stats['channel_means'], stats['channel_stds'])):
            print(f"  Channel {i}: mean={ch_mean:.6f}, std={ch_std:.6f}")
        print()

    print("=" * 60)
    print("DIFFUSION TRAINING ARGS")
    print("=" * 60)
    print(f"  --latent_mean {mean:.6f} --latent_std {std:.6f}")
    print()

    # Save to JSON if requested
    if args.output_json:
        import json
        with open(args.output_json, 'w') as f:
            json.dump({
                "mean": mean,
                "std": std,
                "stats": stats,
                "args": {
                    "modality": args.modality,
                    "vae_checkpoint": args.vae_checkpoint,
                    "vae_config": args.vae_config,
                    "cache_dir": args.cache_dir,
                    "num_samples": args.num_samples,
                    "latent_channels": args.latent_channels,
                }
            }, f, indent=2)
        print(f"Statistics saved to {args.output_json}")


if __name__ == "__main__":
    main()