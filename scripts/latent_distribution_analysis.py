#!/usr/bin/env python3
"""
Analyze VAE latent distribution statistics across different dataset qualities.

Compares latent statistics (mean, variance, etc.) between LibriSpeech clean
and other splits to verify the latent space is consistent across quality levels.

Usage:
    python scripts/latent_distribution_analysis.py \
        --vae_checkpoint runs/audio_vae/my_run/checkpoint-10000 \
        --vae_config mini \
        --latent_channels 16 \
        --max_samples 500
"""

import os
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset, Audio

from dataset_loading.audio_loading import extract_mels
from utils.audio_utils import SharedWindowBuffer


def load_audio_vae(checkpoint_path: str, vae_config: str, latent_channels: int, device: str = "cuda"):
    """Load an audio VAE from a checkpoint."""
    from model.audio.vae import model_config_lookup

    if vae_config not in model_config_lookup:
        raise ValueError(f"Unknown VAE config: {vae_config}. Available: {list(model_config_lookup.keys())}")

    model = model_config_lookup[vae_config](
        latent_channels=latent_channels,
        perceptual_loss_type="none",
    )

    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded VAE from {safetensors_path}")
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded VAE from {pytorch_path}")
    else:
        raise FileNotFoundError(f"No model checkpoint found at {checkpoint_path}")

    model = model.to(device)
    model.eval()
    return model


def compute_latent_stats(
    vae,
    dataset,
    shared_window_buffer: SharedWindowBuffer,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    max_samples: int,
    max_frames: int,
    device: str,
) -> Dict[str, np.ndarray]:
    """
    Compute latent statistics for a dataset.

    Returns dict with:
        - means: [N, C] per-sample channel means
        - stds: [N, C] per-sample channel stds
        - global_mean: [C] mean across all samples
        - global_std: [C] std across all samples
        - latent_norms: [N] L2 norm of each latent
        - kl_divs: [N] KL divergence from N(0,1) per sample
    """
    all_means = []
    all_stds = []
    all_mus = []
    all_logvars = []
    latent_norms = []

    num_samples = min(len(dataset), max_samples)

    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Encoding"):
            try:
                example = dataset[i]
                audio = example["audio"]
                waveform = torch.tensor(audio["array"], dtype=torch.float32)

                # Skip very short or silent audio
                if len(waveform) < n_fft or waveform.abs().max() < 0.01:
                    continue

                # Extract mel spectrogram
                mel = extract_mels(
                    shared_window_buffer,
                    waveform,
                    sr=sample_rate,
                    n_mels=n_mels,
                    n_fft=n_fft,
                    hop_length=hop_length,
                )

                # Pad/truncate to max_frames
                if mel.shape[-1] < max_frames:
                    mel = F.pad(mel, (0, max_frames - mel.shape[-1]), value=0)
                elif mel.shape[-1] > max_frames:
                    mel = mel[..., :max_frames]

                # Add batch and channel dims: [n_mels, T] -> [1, 1, n_mels, T]
                mel = mel.unsqueeze(0).unsqueeze(0).to(device)

                # Encode
                mu, logvar = vae.encoder(mel)

                # Store statistics
                mu_np = mu.cpu().numpy().squeeze()  # [C, H, W]
                logvar_np = logvar.cpu().numpy().squeeze()

                # Per-channel statistics (flatten spatial dims)
                channel_means = mu_np.mean(axis=(1, 2)) if mu_np.ndim == 3 else mu_np.mean(axis=-1)
                channel_stds = mu_np.std(axis=(1, 2)) if mu_np.ndim == 3 else mu_np.std(axis=-1)

                all_means.append(channel_means)
                all_stds.append(channel_stds)
                all_mus.append(mu_np.flatten())
                all_logvars.append(logvar_np.flatten())
                latent_norms.append(np.linalg.norm(mu_np))

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

    all_means = np.array(all_means)  # [N, C]
    all_stds = np.array(all_stds)    # [N, C]
    all_mus = np.array(all_mus)      # [N, D]
    all_logvars = np.array(all_logvars)  # [N, D]
    latent_norms = np.array(latent_norms)

    # Compute KL divergence from N(0,1) per sample
    # KL(N(mu, sigma^2) || N(0, 1)) = 0.5 * (mu^2 + sigma^2 - log(sigma^2) - 1)
    kl_divs = 0.5 * (all_mus**2 + np.exp(all_logvars) - all_logvars - 1).mean(axis=1)

    return {
        "means": all_means,
        "stds": all_stds,
        "global_mean": all_means.mean(axis=0),
        "global_std": all_means.std(axis=0),
        "latent_norms": latent_norms,
        "kl_divs": kl_divs,
        "all_mus_flat": all_mus,
        "all_logvars_flat": all_logvars,
    }


def plot_comparison(
    stats_clean: Dict[str, np.ndarray],
    stats_other: Dict[str, np.ndarray],
    output_path: str,
):
    """Generate comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    num_channels = stats_clean["global_mean"].shape[0]
    channels = np.arange(num_channels)

    # 1. Per-channel mean comparison
    ax = axes[0, 0]
    width = 0.35
    ax.bar(channels - width/2, stats_clean["global_mean"], width, label="Clean", alpha=0.8)
    ax.bar(channels + width/2, stats_other["global_mean"], width, label="Other", alpha=0.8)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Mean")
    ax.set_title("Per-Channel Latent Mean")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Per-channel std comparison
    ax = axes[0, 1]
    ax.bar(channels - width/2, stats_clean["global_std"], width, label="Clean", alpha=0.8)
    ax.bar(channels + width/2, stats_other["global_std"], width, label="Other", alpha=0.8)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Std")
    ax.set_title("Per-Channel Latent Std (across samples)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Latent norm distribution
    ax = axes[0, 2]
    ax.hist(stats_clean["latent_norms"], bins=30, alpha=0.6, label="Clean", density=True)
    ax.hist(stats_other["latent_norms"], bins=30, alpha=0.6, label="Other", density=True)
    ax.set_xlabel("L2 Norm")
    ax.set_ylabel("Density")
    ax.set_title("Latent L2 Norm Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. KL divergence distribution
    ax = axes[1, 0]
    ax.hist(stats_clean["kl_divs"], bins=30, alpha=0.6, label="Clean", density=True)
    ax.hist(stats_other["kl_divs"], bins=30, alpha=0.6, label="Other", density=True)
    ax.set_xlabel("KL Divergence from N(0,1)")
    ax.set_ylabel("Density")
    ax.set_title("Per-Sample KL Divergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Mean of means distribution (per sample)
    ax = axes[1, 1]
    clean_sample_means = stats_clean["means"].mean(axis=1)
    other_sample_means = stats_other["means"].mean(axis=1)
    ax.hist(clean_sample_means, bins=30, alpha=0.6, label="Clean", density=True)
    ax.hist(other_sample_means, bins=30, alpha=0.6, label="Other", density=True)
    ax.set_xlabel("Sample Mean (avg across channels)")
    ax.set_ylabel("Density")
    ax.set_title("Per-Sample Mean Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Summary statistics text
    ax = axes[1, 2]
    ax.axis("off")

    summary_text = (
        "Summary Statistics\n"
        "==================\n\n"
        f"Clean samples: {len(stats_clean['latent_norms'])}\n"
        f"Other samples: {len(stats_other['latent_norms'])}\n\n"
        "Latent Norm:\n"
        f"  Clean: {stats_clean['latent_norms'].mean():.2f} +/- {stats_clean['latent_norms'].std():.2f}\n"
        f"  Other: {stats_other['latent_norms'].mean():.2f} +/- {stats_other['latent_norms'].std():.2f}\n\n"
        "KL Divergence:\n"
        f"  Clean: {stats_clean['kl_divs'].mean():.2f} +/- {stats_clean['kl_divs'].std():.2f}\n"
        f"  Other: {stats_other['kl_divs'].mean():.2f} +/- {stats_other['kl_divs'].std():.2f}\n\n"
        "Global Mean (avg across channels):\n"
        f"  Clean: {stats_clean['global_mean'].mean():.4f}\n"
        f"  Other: {stats_other['global_mean'].mean():.4f}\n\n"
        "Channel Mean Correlation:\n"
        f"  {np.corrcoef(stats_clean['global_mean'], stats_other['global_mean'])[0,1]:.4f}\n\n"
        "Channel Std Correlation:\n"
        f"  {np.corrcoef(stats_clean['global_std'], stats_other['global_std'])[0,1]:.4f}"
    )
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze VAE latent distribution across dataset qualities")

    # VAE settings
    parser.add_argument("--vae_checkpoint", type=str, required=True,
                        help="Path to VAE checkpoint directory")
    parser.add_argument("--vae_config", type=str, default="mini",
                        help="VAE config name")
    parser.add_argument("--latent_channels", type=int, default=16,
                        help="Number of latent channels")

    # Audio settings
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--max_frames", type=int, default=625,
                        help="Max mel frames (~10sec at default settings)")

    # Dataset settings
    parser.add_argument("--dataset_name", type=str, default="openslr/librispeech_asr")
    parser.add_argument("--clean_split", type=str, default="validation.clean",
                        help="Clean validation split")
    parser.add_argument("--other_split", type=str, default="validation.other",
                        help="Other validation split")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Max samples per split")

    # Output
    parser.add_argument("--output_dir", type=str, default="./analysis",
                        help="Output directory for plots")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading VAE from {args.vae_checkpoint}...")
    vae = load_audio_vae(args.vae_checkpoint, args.vae_config, args.latent_channels, device)

    shared_window_buffer = SharedWindowBuffer()

    # Load clean dataset
    print(f"\nLoading {args.dataset_name} split {args.clean_split}...")
    dataset_clean = load_dataset(args.dataset_name, "clean", split=args.clean_split.split(".")[-1])
    dataset_clean = dataset_clean.cast_column("audio", Audio(sampling_rate=args.sample_rate))

    print(f"Computing latent statistics for clean split ({min(len(dataset_clean), args.max_samples)} samples)...")
    stats_clean = compute_latent_stats(
        vae, dataset_clean, shared_window_buffer,
        args.sample_rate, args.n_mels, args.n_fft, args.hop_length,
        args.max_samples, args.max_frames, device
    )

    # Load other dataset
    print(f"\nLoading {args.dataset_name} split {args.other_split}...")
    dataset_other = load_dataset(args.dataset_name, "other", split=args.other_split.split(".")[-1])
    dataset_other = dataset_other.cast_column("audio", Audio(sampling_rate=args.sample_rate))

    print(f"Computing latent statistics for other split ({min(len(dataset_other), args.max_samples)} samples)...")
    stats_other = compute_latent_stats(
        vae, dataset_other, shared_window_buffer,
        args.sample_rate, args.n_mels, args.n_fft, args.hop_length,
        args.max_samples, args.max_frames, device
    )

    # Generate plots
    output_path = os.path.join(args.output_dir, "latent_distribution_comparison.png")
    plot_comparison(stats_clean, stats_other, output_path)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nClean samples analyzed: {len(stats_clean['latent_norms'])}")
    print(f"Other samples analyzed: {len(stats_other['latent_norms'])}")

    print(f"\nLatent L2 Norm:")
    print(f"  Clean: {stats_clean['latent_norms'].mean():.2f} +/- {stats_clean['latent_norms'].std():.2f}")
    print(f"  Other: {stats_other['latent_norms'].mean():.2f} +/- {stats_other['latent_norms'].std():.2f}")

    print(f"\nKL Divergence from N(0,1):")
    print(f"  Clean: {stats_clean['kl_divs'].mean():.2f} +/- {stats_clean['kl_divs'].std():.2f}")
    print(f"  Other: {stats_other['kl_divs'].mean():.2f} +/- {stats_other['kl_divs'].std():.2f}")

    # Compute distribution shift metrics
    mean_shift = np.abs(stats_clean['global_mean'] - stats_other['global_mean']).mean()
    std_shift = np.abs(stats_clean['global_std'] - stats_other['global_std']).mean()
    mean_corr = np.corrcoef(stats_clean['global_mean'], stats_other['global_mean'])[0, 1]
    std_corr = np.corrcoef(stats_clean['global_std'], stats_other['global_std'])[0, 1]

    print(f"\nDistribution Shift Metrics:")
    print(f"  Mean absolute channel mean shift: {mean_shift:.4f}")
    print(f"  Mean absolute channel std shift: {std_shift:.4f}")
    print(f"  Channel mean correlation: {mean_corr:.4f}")
    print(f"  Channel std correlation: {std_corr:.4f}")

    if mean_corr > 0.95 and std_corr > 0.9:
        print("\n[OK] Latent distributions appear well-aligned between clean and other.")
    elif mean_corr > 0.8:
        print("\n[WARN] Some distribution shift detected. Diffusion may need exposure to both.")
    else:
        print("\n[ALERT] Significant distribution shift. Consider training diffusion on mixed data.")

    # Save raw stats
    stats_path = os.path.join(args.output_dir, "latent_stats.npz")
    np.savez(
        stats_path,
        clean_means=stats_clean["means"],
        clean_stds=stats_clean["stds"],
        clean_norms=stats_clean["latent_norms"],
        clean_kl=stats_clean["kl_divs"],
        other_means=stats_other["means"],
        other_stds=stats_other["stds"],
        other_norms=stats_other["latent_norms"],
        other_kl=stats_other["kl_divs"],
    )
    print(f"\nSaved raw statistics to {stats_path}")


if __name__ == "__main__":
    main()