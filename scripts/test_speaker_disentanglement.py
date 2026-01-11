#!/usr/bin/env python3
"""
Test speaker/content disentanglement in audio VAE.

This script runs several diagnostics to verify that:
1. The latent space captures content (phonetics, prosody) but NOT speaker identity
2. The speaker embedding controls speaker identity in reconstruction

Tests:
1. Voice conversion: Encode speaker A, decode with speaker B's embedding
2. Speaker classifier: Train classifier to predict speaker from latents (low accuracy = good)
3. Latent visualization: t-SNE colored by speaker (no clusters = good)
4. Speaker embedding ablation: Decode with zero/mean embedding
5. Speaker interpolation: Same latent, interpolate between speaker embeddings
6. Content similarity: Same content from different speakers should have similar latents

Additional Disentanglement Methods (beyond speaker embedding dropout):
- Adversarial speaker classifier with gradient reversal layer (GRL)
- Mutual information minimization between latents and speaker embeddings
- Contrastive losses (push same-content different-speaker pairs together)
- FactorVAE-style total correlation penalty
- Instance normalization in encoder (removes speaker-correlated statistics)
- Domain adversarial training (speaker as domain)

Usage:
    python scripts/test_speaker_disentanglement.py \
        --vae_checkpoint runs/audio_vae/my_run/checkpoint-10000 \
        --vae_config small \
        --cache_dir cached_datasets/my_audio_dataset_vae_cached \
        --output_dir disentanglement_results \
        --num_samples 500
"""

import argparse
import os
import sys
import json
import random
from collections import defaultdict

# Add repo root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_audio_vae(
    checkpoint_path: str,
    config: str,
    latent_channels: int = 32,
    speaker_embedding_dim: int = 192,
    normalize_speaker_embedding: bool = True,
    film_scale_bound: float = 0.5,
    film_shift_bound: float = 0.5,
    zero_init_film_bias: bool = False,
) -> nn.Module:
    """Load audio VAE from checkpoint."""
    from model.audio.vae import model_config_lookup
    from utils.model_loading_utils import load_model

    if config not in model_config_lookup:
        raise ValueError(f"Unknown audio VAE config: {config}. Available: {list(model_config_lookup.keys())}")

    model = model_config_lookup[config](
        latent_channels=latent_channels,
        speaker_embedding_dim=speaker_embedding_dim,
        normalize_speaker_embedding=normalize_speaker_embedding,
        film_scale_bound=film_scale_bound,
        film_shift_bound=film_shift_bound,
        zero_init_film_bias=zero_init_film_bias,
    )

    # Check checkpoint for FiLM weights before loading
    import glob
    checkpoint_files = glob.glob(os.path.join(checkpoint_path, "checkpoint-*", "pytorch_model.bin"))
    if checkpoint_files:
        sorted_checkpoints = sorted(checkpoint_files, key=lambda x: int(x.split("-")[-1].split(os.path.sep)[0]))
        latest_checkpoint = sorted_checkpoints[-1]
        checkpoint_state = torch.load(latest_checkpoint, map_location='cpu')

        # Check for FiLM projection keys
        film_keys = [k for k in checkpoint_state.keys() if 'speaker_projection' in k or 'early_film' in k]
        if film_keys:
            print(f"\n[FiLM Diagnostic] Found {len(film_keys)} FiLM projection keys in checkpoint:")
            for k in film_keys[:5]:  # Show first 5
                print(f"  - {k}")
            if len(film_keys) > 5:
                print(f"  ... and {len(film_keys) - 5} more")
        else:
            print("\n[FiLM Diagnostic] WARNING: No FiLM projection keys found in checkpoint!")
            print("  The checkpoint may have been trained with speaker_embedding_dim=0")
            print("  FiLM projections will be randomly initialized and won't respond to speaker embeddings!")

    model, _ = load_model(False, model, checkpoint_path)

    # Verify FiLM projections exist and test their behavior
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'speaker_projections'):
        print(f"\n[FiLM Diagnostic] Model has {len(model.decoder.speaker_projections)} speaker projections")
        if hasattr(model.decoder, 'early_film_projection'):
            print("[FiLM Diagnostic] Model has early_film_projection")

        # Test FiLM projection behavior with different embeddings
        with torch.no_grad():
            test_emb1 = torch.randn(1, speaker_embedding_dim)
            test_emb2 = torch.randn(1, speaker_embedding_dim)
            test_emb1 = torch.nn.functional.normalize(test_emb1, p=2, dim=-1)
            test_emb2 = torch.nn.functional.normalize(test_emb2, p=2, dim=-1)

            proj = model.decoder.speaker_projections[0]
            out1 = proj(test_emb1)
            out2 = proj(test_emb2)
            diff = (out1 - out2).abs().mean().item()

            print(f"[FiLM Diagnostic] FiLM projection test:")
            print(f"  - Output diff for 2 random embeddings: {diff:.6f}")
            if diff < 0.001:
                print("  - WARNING: FiLM projection outputs nearly identical for different inputs!")
                print("    This suggests FiLM is not responding to speaker embeddings.")
            else:
                print(f"  - FiLM projection IS responding to different speaker embeddings")

    return model


def load_vocoder(checkpoint_path: str, config: str, device: torch.device) -> nn.Module:
    """Load vocoder from checkpoint for audio generation."""
    from model.audio.vocoders.vocoders import model_config_lookup as vocoder_config_lookup
    from utils.model_loading_utils import load_model
    from utils.audio_utils import SharedWindowBuffer

    if config not in vocoder_config_lookup:
        raise ValueError(f"Unknown vocoder config: {config}. Available: {list(vocoder_config_lookup.keys())}")

    shared_window_buffer = SharedWindowBuffer()
    vocoder = vocoder_config_lookup[config](shared_window_buffer=shared_window_buffer)
    vocoder, _ = load_model(False, vocoder, checkpoint_path)

    # Remove weight normalization for inference optimization
    if hasattr(vocoder, 'vocoder') and hasattr(vocoder.vocoder, 'remove_weight_norm'):
        vocoder.vocoder.remove_weight_norm()

    vocoder = vocoder.to(device)
    vocoder.eval()
    return vocoder


def trim_to_length(mel_spec: torch.Tensor, length: int) -> torch.Tensor:
    """Trim mel spectrogram to actual length (remove padding)."""
    return mel_spec[..., :length]


def save_audio(
    vocoder: nn.Module,
    mel_spec: torch.Tensor,
    output_path: str,
    sample_rate: int = 16000,
    device: torch.device = None,
):
    """Generate audio from mel spectrogram using vocoder and save to file."""
    import soundfile as sf

    if device is None:
        device = next(vocoder.parameters()).device

    mel_spec = mel_spec.to(device)

    with torch.no_grad():
        # Vocoder expects [B, 1, n_mels, T] or [B, n_mels, T]
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(0)  # Add batch dim
        waveform = vocoder(mel_spec)["pred_waveform"]

    # Convert to numpy and save
    waveform = waveform.squeeze().cpu().numpy()
    sf.write(output_path, waveform, sample_rate)
    return output_path


def cluster_speaker_embeddings(embeddings: torch.Tensor, n_clusters: int = 20) -> list:
    """
    Cluster speaker embeddings to create pseudo speaker IDs.
    Useful when speaker_id is not available in dataset.
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("WARNING: sklearn not installed, using random pseudo-IDs")
        return [f"pseudo_{i % n_clusters}" for i in range(len(embeddings))]

    # Flatten if needed (embeddings might be [N, 1, D] or [N, D])
    if embeddings.dim() == 3:
        embeddings = embeddings.squeeze(1)

    print(f"Clustering {len(embeddings)} speaker embeddings into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings.numpy())

    return [f"cluster_{label}" for label in cluster_labels]


def select_diverse_speaker_pairs(
    samples_by_speaker: dict,
    num_pairs: int = 5,
    min_samples_per_speaker: int = 1,
) -> list:
    """
    Select speaker pairs that are maximally diverse based on embedding cosine distance.

    This helps avoid tests choosing similar speakers or low-quality samples that
    may not represent distinct voice characteristics.

    Args:
        samples_by_speaker: Dict mapping speaker_id -> list of samples with embeddings
        num_pairs: Number of diverse pairs to select
        min_samples_per_speaker: Minimum samples required for a speaker to be considered

    Returns:
        List of (speaker_a, speaker_b, distance) tuples sorted by distance (most diverse first)
    """
    # Filter speakers with enough samples
    valid_speakers = [
        sid for sid, samples in samples_by_speaker.items()
        if len(samples) >= min_samples_per_speaker
    ]

    if len(valid_speakers) < 2:
        print(f"WARNING: Only {len(valid_speakers)} valid speakers found")
        return list(zip(valid_speakers[:1] * 2, valid_speakers[:1] * 2, [0.0]))

    # Compute mean embedding for each speaker
    speaker_mean_embeddings = {}
    for sid in valid_speakers:
        embeddings = torch.stack([s["speaker_embedding"].squeeze() for s in samples_by_speaker[sid]])
        speaker_mean_embeddings[sid] = embeddings.mean(dim=0)

    # Compute pairwise cosine distances between all speaker pairs
    pairs_with_distances = []
    for i, sid_a in enumerate(valid_speakers):
        for sid_b in valid_speakers[i+1:]:
            emb_a = speaker_mean_embeddings[sid_a]
            emb_b = speaker_mean_embeddings[sid_b]
            # Cosine distance = 1 - cosine_similarity (higher = more different)
            cos_sim = F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
            cos_dist = 1.0 - cos_sim
            pairs_with_distances.append((sid_a, sid_b, cos_dist))

    # Sort by distance (descending) to get most diverse pairs first
    pairs_with_distances.sort(key=lambda x: x[2], reverse=True)

    # Return top N most diverse pairs
    selected = pairs_with_distances[:num_pairs]

    if selected:
        print(f"Selected {len(selected)} diverse speaker pairs:")
        for sid_a, sid_b, dist in selected:
            print(f"  {sid_a} <-> {sid_b}: cosine distance = {dist:.4f}")

    return selected


def extract_latents(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int,
    max_mel_frames: int = 1875,  # 30s of audio at default hop length
) -> dict:
    """Extract latents and metadata from dataset."""
    model.eval()

    latents = []
    speaker_embeddings = []
    speaker_ids = []
    sample_indices = []
    mel_lengths = []  # Track original lengths before padding
    has_speaker_ids = None  # Will be set on first batch

    total = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting latents")):
            if total >= num_samples:
                break

            # Collator returns "mel_spec", not "spectrogram"
            spec = batch["mel_spec"].to(device)

            # Get original lengths before padding
            if "mel_spec_lengths" in batch:
                batch_lengths = batch["mel_spec_lengths"].tolist()
            else:
                # Use actual spec length if no explicit lengths provided
                batch_lengths = [spec.shape[-1]] * spec.shape[0]
            mel_lengths.extend(batch_lengths)

            # Pad/truncate to fixed length to ensure consistent latent sizes across batches
            if spec.shape[-1] < max_mel_frames:
                spec = F.pad(spec, (0, max_mel_frames - spec.shape[-1]), value=0)
            elif spec.shape[-1] > max_mel_frames:
                spec = spec[..., :max_mel_frames]

            mu, logvar = model.encode(spec)

            # Store latents (use mu, not sampled z, for consistency)
            latents.append(mu.cpu())

            # Store speaker embeddings if available
            if "speaker_embedding" in batch and batch["speaker_embedding"] is not None:
                speaker_embeddings.append(batch["speaker_embedding"].cpu())

            # Check if we have speaker IDs on first batch
            if has_speaker_ids is None:
                has_speaker_ids = "speaker_id" in batch or "speaker_ids" in batch or "speaker" in batch

            # Store speaker IDs if available
            if "speaker_id" in batch:
                speaker_ids.extend(batch["speaker_id"])
            elif "speaker_ids" in batch:
                speaker_ids.extend(batch["speaker_ids"])
            elif "speaker" in batch:
                speaker_ids.extend(batch["speaker"])
            # Don't assign batch index yet - we'll cluster embeddings if no IDs

            sample_indices.extend(range(total, total + spec.shape[0]))
            total += spec.shape[0]

    all_latents = torch.cat(latents)[:num_samples]
    all_embeddings = torch.cat(speaker_embeddings)[:num_samples] if speaker_embeddings else None
    all_lengths = mel_lengths[:num_samples]

    # If no speaker IDs were in the dataset, cluster embeddings to create pseudo-IDs
    if not has_speaker_ids and all_embeddings is not None:
        print("\nNo speaker_id in dataset, clustering speaker embeddings for pseudo-IDs...")
        speaker_ids = cluster_speaker_embeddings(all_embeddings, n_clusters=min(20, num_samples // 10))
    elif not speaker_ids:
        # No speaker IDs and no embeddings - use sample index as pseudo-ID
        speaker_ids = [f"sample_{i}" for i in range(len(all_latents))]

    return {
        "latents": all_latents,
        "speaker_embeddings": all_embeddings,
        "speaker_ids": speaker_ids[:num_samples],
        "sample_indices": sample_indices[:num_samples],
        "mel_lengths": all_lengths,
    }


def test_speaker_classifier(
    latents: torch.Tensor,
    speaker_ids: list,
    device: torch.device,
    num_epochs: int = 50,
) -> dict:
    """
    Train a simple classifier to predict speaker from latents.
    Low accuracy = good disentanglement (speaker info not in latents).
    """
    print("\n" + "=" * 60)
    print("TEST: Speaker Classifier on Latents")
    print("=" * 60)

    # Create speaker ID to index mapping
    unique_speakers = sorted(set(speaker_ids))
    speaker_to_idx = {s: i for i, s in enumerate(unique_speakers)}
    num_speakers = len(unique_speakers)

    print(f"Number of unique speakers: {num_speakers}")

    if num_speakers < 2:
        print("WARNING: Need at least 2 speakers to test classifier")
        return {"accuracy": None, "chance_accuracy": None, "num_speakers": num_speakers}

    # Flatten latents: [N, C, H, W] -> [N, C*H*W] or global average pool
    if latents.dim() == 4:
        # Global average pool over spatial dims
        latents_flat = latents.mean(dim=[2, 3])  # [N, C]
    elif latents.dim() == 3:
        latents_flat = latents.mean(dim=2)  # [N, C]
    else:
        latents_flat = latents

    # Create labels
    labels = torch.tensor([speaker_to_idx[s] for s in speaker_ids])

    # Train/test split
    n_samples = len(latents_flat)
    indices = torch.randperm(n_samples)
    split = int(0.8 * n_samples)
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = latents_flat[train_idx].to(device), latents_flat[test_idx].to(device)
    y_train, y_test = labels[train_idx].to(device), labels[test_idx].to(device)

    # Simple linear classifier
    input_dim = latents_flat.shape[1]
    classifier = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_speakers),
    ).to(device)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train
    classifier.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logits = classifier(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate
    classifier.eval()
    with torch.no_grad():
        test_logits = classifier(X_test)
        predictions = test_logits.argmax(dim=1)
        accuracy = (predictions == y_test).float().mean().item()

    chance_accuracy = 1.0 / num_speakers

    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Chance accuracy: {chance_accuracy:.2%}")
    print(f"Ratio to chance: {accuracy / chance_accuracy:.2f}x")

    if accuracy < chance_accuracy * 1.5:
        print("RESULT: GOOD disentanglement - classifier near chance level")
    elif accuracy < chance_accuracy * 3:
        print("RESULT: MODERATE disentanglement - some speaker info leaking")
    else:
        print("RESULT: POOR disentanglement - significant speaker info in latents")

    return {
        "accuracy": accuracy,
        "chance_accuracy": chance_accuracy,
        "ratio_to_chance": accuracy / chance_accuracy,
        "num_speakers": num_speakers,
    }


def test_latent_visualization(
    latents: torch.Tensor,
    speaker_ids: list,
    output_dir: str,
    mel_lengths: list = None,
    max_samples: int = 1000,
) -> dict:
    """
    Visualize latents with t-SNE, colored by speaker and by duration.
    Good disentanglement = no clear speaker clusters.

    Args:
        latents: Latent representations [N, C, ...]
        speaker_ids: List of speaker IDs for each sample
        output_dir: Directory to save plots
        mel_lengths: List of mel spectrogram lengths (frames) for each sample
        max_samples: Maximum samples to visualize
    """
    print("\n" + "=" * 60)
    print("TEST: Latent Space Visualization (t-SNE)")
    print("=" * 60)

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("WARNING: sklearn not installed, skipping t-SNE visualization")
        return {"status": "skipped", "reason": "sklearn not installed"}

    # Flatten and subsample if needed
    if latents.dim() == 4:
        latents_flat = latents.mean(dim=[2, 3])
    elif latents.dim() == 3:
        latents_flat = latents.mean(dim=2)
    else:
        latents_flat = latents

    if len(latents_flat) > max_samples:
        indices = random.sample(range(len(latents_flat)), max_samples)
        latents_flat = latents_flat[indices]
        speaker_ids = [speaker_ids[i] for i in indices]
        if mel_lengths is not None:
            mel_lengths = [mel_lengths[i] for i in indices]

    print(f"Running t-SNE on {len(latents_flat)} samples...")

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents_flat) - 1))
    latents_2d = tsne.fit_transform(latents_flat.numpy())

    # Create color mapping for speakers
    unique_speakers = sorted(set(speaker_ids))
    speaker_to_color = {s: i for i, s in enumerate(unique_speakers)}
    speaker_colors = [speaker_to_color[s] for s in speaker_ids]

    os.makedirs(output_dir, exist_ok=True)
    plot_paths = []

    # Plot 1: Colored by speaker
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(latents_2d[:, 0], latents_2d[:, 1], c=speaker_colors, cmap='tab20', alpha=0.6, s=20)

    if len(unique_speakers) <= 20:
        legend = ax.legend(*scatter.legend_elements(), title="Speakers", loc="upper right")
        ax.add_artist(legend)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"Latent Space by Speaker (N={len(latents_flat)}, {len(unique_speakers)} speakers)\n"
                 f"Good disentanglement = mixed colors, no clusters")

    save_path_speaker = os.path.join(output_dir, "tsne_by_speaker.png")
    plt.savefig(save_path_speaker, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths.append(save_path_speaker)
    print(f"Saved t-SNE by speaker plot to {save_path_speaker}")

    # Plot 2: Colored by duration (mel_lengths)
    if mel_lengths is not None:
        fig, ax = plt.subplots(figsize=(12, 10))

        # Convert lengths to seconds for readability (assuming hop_length=256, sr=16000)
        # frames * hop_length / sample_rate = seconds
        hop_length = 256
        sample_rate = 16000
        durations_sec = [length * hop_length / sample_rate for length in mel_lengths]

        scatter = ax.scatter(
            latents_2d[:, 0], latents_2d[:, 1],
            c=durations_sec, cmap='viridis', alpha=0.6, s=20
        )
        cbar = plt.colorbar(scatter, ax=ax, label='Duration (seconds)')

        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(f"Latent Space by Duration (N={len(latents_flat)})\n"
                     f"If horseshoe correlates with duration, latent encodes audio length")

        # Add duration statistics
        min_dur = min(durations_sec)
        max_dur = max(durations_sec)
        mean_dur = np.mean(durations_sec)
        ax.text(0.02, 0.98, f"Duration: {min_dur:.1f}s - {max_dur:.1f}s (mean: {mean_dur:.1f}s)",
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        save_path_duration = os.path.join(output_dir, "tsne_by_duration.png")
        plt.savefig(save_path_duration, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths.append(save_path_duration)
        print(f"Saved t-SNE by duration plot to {save_path_duration}")

        # Compute correlation between t-SNE position and duration
        # Using distance from center or position along principal axis
        tsne_magnitude = np.sqrt(latents_2d[:, 0]**2 + latents_2d[:, 1]**2)
        corr_magnitude = np.corrcoef(tsne_magnitude, durations_sec)[0, 1]
        corr_tsne1 = np.corrcoef(latents_2d[:, 0], durations_sec)[0, 1]
        corr_tsne2 = np.corrcoef(latents_2d[:, 1], durations_sec)[0, 1]

        print(f"Duration correlations:")
        print(f"  t-SNE 1 vs duration: r={corr_tsne1:.3f}")
        print(f"  t-SNE 2 vs duration: r={corr_tsne2:.3f}")
        print(f"  t-SNE magnitude vs duration: r={corr_magnitude:.3f}")
    else:
        print("No mel_lengths provided, skipping duration plot")
        save_path_duration = None
        corr_tsne1 = corr_tsne2 = corr_magnitude = None

    return {
        "status": "completed",
        "plot_path_speaker": save_path_speaker,
        "plot_path_duration": save_path_duration,
        "plot_paths": plot_paths,
        "num_samples": len(latents_flat),
        "duration_correlations": {
            "tsne1_vs_duration": corr_tsne1,
            "tsne2_vs_duration": corr_tsne2,
            "magnitude_vs_duration": corr_magnitude,
        } if mel_lengths else None
    }


def test_voice_conversion(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: str,
    num_conversions: int = 5,
    vocoder: nn.Module = None,
    sample_rate: int = 16000,
    use_diverse_pairs: bool = True,
) -> dict:
    """
    Test voice conversion: encode speaker A, decode with speaker B's embedding.

    Args:
        use_diverse_pairs: If True, select maximally diverse speaker pairs based on
                          embedding cosine distance to ensure clear voice differences.
    """
    print("\n" + "=" * 60)
    print("TEST: Voice Conversion")
    print("=" * 60)

    model.eval()

    # Collect samples with different speakers - gather more to enable diverse selection
    samples_by_speaker = defaultdict(list)
    min_speakers_needed = max(10, num_conversions * 2) if use_diverse_pairs else 2
    max_samples_per_speaker = 5  # Cap samples per speaker to avoid memory issues

    print(f"Collecting samples from at least {min_speakers_needed} speakers...")

    with torch.no_grad():
        for batch in dataloader:
            if "speaker_embedding" not in batch or batch["speaker_embedding"] is None:
                print("WARNING: No speaker embeddings in dataset, skipping voice conversion test")
                return {"status": "skipped", "reason": "no speaker embeddings"}

            spec = batch["mel_spec"]
            speaker_emb = batch["speaker_embedding"]
            lengths = batch.get("mel_spec_lengths", None)

            # Get speaker IDs
            if "speaker_id" in batch:
                speaker_ids = batch["speaker_id"]
            elif "speaker_ids" in batch:
                speaker_ids = batch["speaker_ids"]
            elif "speaker" in batch:
                speaker_ids = batch["speaker"]
            else:
                speaker_ids = [f"unknown_{i}" for i in range(spec.shape[0])]

            for i in range(spec.shape[0]):
                sid = speaker_ids[i] if isinstance(speaker_ids, list) else speaker_ids[i].item()
                # Skip if we already have enough samples for this speaker
                if len(samples_by_speaker[sid]) >= max_samples_per_speaker:
                    continue
                length = lengths[i].item() if lengths is not None else spec.shape[-1]
                samples_by_speaker[sid].append({
                    "mel_spec": spec[i:i+1],
                    "speaker_embedding": speaker_emb[i:i+1],
                    "length": length,
                })

            # Continue until we have enough speakers
            if len(samples_by_speaker) >= min_speakers_needed:
                break

    if len(samples_by_speaker) < 2:
        print("WARNING: Need at least 2 different speakers for voice conversion test")
        return {"status": "skipped", "reason": "insufficient speakers"}

    print(f"Collected samples from {len(samples_by_speaker)} speakers")

    # Select speaker pairs - either diverse or sequential
    if use_diverse_pairs and len(samples_by_speaker) >= 4:
        diverse_pairs = select_diverse_speaker_pairs(
            samples_by_speaker,
            num_pairs=num_conversions,
            min_samples_per_speaker=1
        )
        # Use the most diverse pair for the main conversion
        speaker_a, speaker_b, _ = diverse_pairs[0]
    else:
        # Fall back to first two speakers
        speakers = list(samples_by_speaker.keys())[:2]
        speaker_a, speaker_b = speakers[0], speakers[1]
        diverse_pairs = [(speaker_a, speaker_b, None)]

    print(f"Converting between Speaker A ({speaker_a}) and Speaker B ({speaker_b})")

    os.makedirs(output_dir, exist_ok=True)
    audio_dir = os.path.join(output_dir, "audio")
    if vocoder is not None:
        os.makedirs(audio_dir, exist_ok=True)
        print(f"Vocoder loaded - will generate audio files in {audio_dir}/")

    results = []
    audio_files = []

    for i in range(min(num_conversions, len(samples_by_speaker[speaker_a]))):
        sample_a = samples_by_speaker[speaker_a][i]
        sample_b = samples_by_speaker[speaker_b][min(i, len(samples_by_speaker[speaker_b]) - 1)]

        spec_a = sample_a["mel_spec"].to(device)
        emb_a = sample_a["speaker_embedding"].to(device)
        emb_b = sample_b["speaker_embedding"].to(device)
        length_a = sample_a["length"]
        length_b = sample_b["length"]

        # Encode speaker A's audio
        mu, logvar = model.encode(spec_a)
        z = model.reparameterize(mu, logvar)

        # Decode with different embeddings
        recon_a = model.decode(z, speaker_embedding=emb_a)  # Original reconstruction
        recon_b = model.decode(z, speaker_embedding=emb_b)  # Voice converted

        # Trim to actual length for visualization
        spec_a_trimmed = trim_to_length(spec_a, length_a)
        recon_a_trimmed = trim_to_length(recon_a, length_a)
        recon_b_trimmed = trim_to_length(recon_b, length_a)
        spec_b_trimmed = trim_to_length(sample_b["mel_spec"], length_b)

        # Save spectrograms for visual comparison
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(spec_a_trimmed[0, 0].cpu().numpy(), aspect='auto', origin='lower')
        axes[0].set_title(f"Original (Speaker {speaker_a})")

        axes[1].imshow(recon_a_trimmed[0, 0].cpu().detach().numpy(), aspect='auto', origin='lower')
        axes[1].set_title(f"Recon w/ Emb A")

        axes[2].imshow(recon_b_trimmed[0, 0].cpu().detach().numpy(), aspect='auto', origin='lower')
        axes[2].set_title(f"Converted to Speaker {speaker_b}")

        axes[3].imshow(spec_b_trimmed[0, 0].numpy(), aspect='auto', origin='lower')
        axes[3].set_title(f"Reference (Speaker {speaker_b})")

        for ax in axes:
            ax.set_xlabel("Time")
            ax.set_ylabel("Mel bin")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"voice_conversion_{i}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()

        results.append(save_path)
        print(f"Saved voice conversion {i} to {save_path}")

        # Generate audio if vocoder available
        if vocoder is not None:
            conversion_audio = {}
            # Original
            orig_path = os.path.join(audio_dir, f"conversion_{i}_original.wav")
            save_audio(vocoder, spec_a_trimmed, orig_path, sample_rate, device)
            conversion_audio["original"] = orig_path

            # Reconstruction with original embedding
            recon_a_path = os.path.join(audio_dir, f"conversion_{i}_recon_emb_a.wav")
            save_audio(vocoder, recon_a_trimmed, recon_a_path, sample_rate, device)
            conversion_audio["recon_emb_a"] = recon_a_path

            # Voice converted
            recon_b_path = os.path.join(audio_dir, f"conversion_{i}_converted_to_{speaker_b}.wav")
            save_audio(vocoder, recon_b_trimmed, recon_b_path, sample_rate, device)
            conversion_audio["converted"] = recon_b_path

            # Reference speaker B
            ref_path = os.path.join(audio_dir, f"conversion_{i}_reference_{speaker_b}.wav")
            save_audio(vocoder, spec_b_trimmed, ref_path, sample_rate, device)
            conversion_audio["reference"] = ref_path

            audio_files.append(conversion_audio)
            print(f"  Generated audio: {orig_path}, {recon_b_path}")

    return {"status": "completed", "num_conversions": len(results), "plots": results, "audio_files": audio_files}


def test_speaker_embedding_ablation(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: str,
    vocoder: nn.Module = None,
    sample_rate: int = 16000,
) -> dict:
    """
    Test reconstruction with zero and mean speaker embeddings.
    Content should be preserved with neutral/average voice.
    """
    print("\n" + "=" * 60)
    print("TEST: Speaker Embedding Ablation")
    print("=" * 60)

    model.eval()

    # Get a sample and speaker embedding dimension
    batch = next(iter(dataloader))

    if "speaker_embedding" not in batch or batch["speaker_embedding"] is None:
        print("WARNING: No speaker embeddings in dataset, skipping ablation test")
        return {"status": "skipped", "reason": "no speaker embeddings"}

    spec = batch["mel_spec"][:1].to(device)
    orig_emb = batch["speaker_embedding"][:1].to(device)
    lengths = batch.get("mel_spec_lengths", None)
    length = lengths[0].item() if lengths is not None else spec.shape[-1]
    emb_dim = orig_emb.shape[-1]

    print(f"Speaker embedding dimension: {emb_dim}")
    print(f"Mel spec length: {length} frames")

    # Create ablated embeddings
    zero_emb = torch.zeros_like(orig_emb)
    rand_emb = torch.randn_like(orig_emb)
    # Extreme embedding to test if FiLM responds at all
    extreme_emb = torch.ones_like(orig_emb) * 100.0

    # Compute mean embedding from dataset
    all_embs = []
    for b in dataloader:
        if "speaker_embedding" in b and b["speaker_embedding"] is not None:
            all_embs.append(b["speaker_embedding"])
        if len(all_embs) >= 50:
            break
    mean_emb = torch.cat(all_embs).mean(dim=0, keepdim=True).to(device)

    with torch.no_grad():
        mu, logvar = model.encode(spec)
        z = model.reparameterize(mu, logvar)

        # Get reconstructions with FiLM stats to diagnose conditioning
        recon_orig, film_stats_orig = model.decode(z, speaker_embedding=orig_emb, return_film_stats=True)
        recon_zero, film_stats_zero = model.decode(z, speaker_embedding=zero_emb, return_film_stats=True)
        recon_mean, film_stats_mean = model.decode(z, speaker_embedding=mean_emb, return_film_stats=True)
        recon_rand, film_stats_rand = model.decode(z, speaker_embedding=rand_emb, return_film_stats=True)
        recon_extreme, film_stats_extreme = model.decode(z, speaker_embedding=extreme_emb, return_film_stats=True)

        # Check normalized embeddings
        print("\n[FiLM Ablation Diagnostic] Embeddings after L2 normalization:")
        orig_norm = torch.nn.functional.normalize(orig_emb, p=2, dim=-1)
        zero_norm = torch.nn.functional.normalize(zero_emb, p=2, dim=-1)
        extreme_norm = torch.nn.functional.normalize(extreme_emb, p=2, dim=-1)
        rand_norm = torch.nn.functional.normalize(rand_emb, p=2, dim=-1)

        print(f"  orig_emb norm: L2={orig_emb.norm().item():.4f} -> {orig_norm.norm().item():.4f}, sample values: {orig_norm[0,:5].tolist()}")
        print(f"  zero_emb norm: L2={zero_emb.norm().item():.4f} -> {zero_norm.norm().item():.4f}, sample values: {zero_norm[0,:5].tolist()}")
        print(f"  extreme_emb norm: L2={extreme_emb.norm().item():.4f} -> {extreme_norm.norm().item():.4f}, sample values: {extreme_norm[0,:5].tolist()}")
        print(f"  rand_emb norm: L2={rand_emb.norm().item():.4f} -> {rand_norm.norm().item():.4f}, sample values: {rand_norm[0,:5].tolist()}")

        # Compare FiLM outputs across embeddings
        print("\n[FiLM Ablation Diagnostic] Comparing FiLM outputs for different embeddings:")
        if film_stats_orig and film_stats_zero:
            # Print key stats for each stage
            for prefix in ['early', 'stage0', 'stage1', 'stage2']:
                scale_key = f"{prefix}_scale_raw_mean"
                shift_key = f"{prefix}_shift_raw_mean"
                if scale_key in film_stats_orig:
                    print(f"\n  {prefix}:")
                    print(f"    scale_mean: orig={film_stats_orig[scale_key]:.4f}, zero={film_stats_zero[scale_key]:.4f}, extreme={film_stats_extreme[scale_key]:.4f}")
                    print(f"    shift_mean: orig={film_stats_orig[shift_key]:.4f}, zero={film_stats_zero[shift_key]:.4f}, extreme={film_stats_extreme[shift_key]:.4f}")
                    # Check if FiLM outputs are identical (the bug we're looking for)
                    scale_diff = abs(film_stats_orig[scale_key] - film_stats_zero[scale_key])
                    if scale_diff < 0.0001:
                        print(f"    ⚠️  WARNING: scale outputs are IDENTICAL for orig vs zero!")
        else:
            print("  No FiLM stats returned - decoder may not support return_film_stats")

    # Trim to actual length for visualization
    spec_trimmed = trim_to_length(spec, length)
    recon_orig_trimmed = trim_to_length(recon_orig, length)
    recon_zero_trimmed = trim_to_length(recon_zero, length)
    recon_mean_trimmed = trim_to_length(recon_mean, length)
    recon_rand_trimmed = trim_to_length(recon_rand, length)
    recon_extreme_trimmed = trim_to_length(recon_extreme, length)

    # Compute pairwise differences to quantify FiLM effect
    def mel_diff(a, b):
        return (a - b).abs().mean().item()

    diff_orig_zero = mel_diff(recon_orig_trimmed, recon_zero_trimmed)
    diff_orig_rand = mel_diff(recon_orig_trimmed, recon_rand_trimmed)
    diff_orig_extreme = mel_diff(recon_orig_trimmed, recon_extreme_trimmed)
    diff_zero_extreme = mel_diff(recon_zero_trimmed, recon_extreme_trimmed)

    print(f"\nPairwise reconstruction differences (mean absolute error):")
    print(f"  Original vs Zero:    {diff_orig_zero:.6f}")
    print(f"  Original vs Random:  {diff_orig_rand:.6f}")
    print(f"  Original vs Extreme: {diff_orig_extreme:.6f}")
    print(f"  Zero vs Extreme:     {diff_zero_extreme:.6f}")

    if diff_zero_extreme < 0.001:
        print("\n  WARNING: Zero and Extreme embeddings produce nearly identical output!")
        print("           This suggests FiLM is not responding to speaker embeddings.")
    elif diff_zero_extreme < 0.01:
        print("\n  CAUTION: Very small difference between Zero and Extreme embeddings.")
        print("           FiLM may have minimal effect on output.")

    # Plot
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))

    axes[0].imshow(spec_trimmed[0, 0].cpu().numpy(), aspect='auto', origin='lower')
    axes[0].set_title("Original Input")

    axes[1].imshow(recon_orig_trimmed[0, 0].cpu().detach().numpy(), aspect='auto', origin='lower')
    axes[1].set_title("Recon (Original Emb)")

    axes[2].imshow(recon_zero_trimmed[0, 0].cpu().detach().numpy(), aspect='auto', origin='lower')
    axes[2].set_title("Recon (Zero Emb)")

    axes[3].imshow(recon_mean_trimmed[0, 0].cpu().detach().numpy(), aspect='auto', origin='lower')
    axes[3].set_title("Recon (Mean Emb)")

    axes[4].imshow(recon_rand_trimmed[0, 0].cpu().detach().numpy(), aspect='auto', origin='lower')
    axes[4].set_title("Recon (Random Emb)")

    axes[5].imshow(recon_extreme_trimmed[0, 0].cpu().detach().numpy(), aspect='auto', origin='lower')
    axes[5].set_title("Recon (Extreme=100)")

    for ax in axes:
        ax.set_xlabel("Time")
        ax.set_ylabel("Mel bin")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "embedding_ablation.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved ablation plot to {save_path}")
    print("Check: Content should be similar across all, only voice characteristics should change")

    # Generate audio if vocoder available
    audio_files = {}
    if vocoder is not None:
        audio_dir = os.path.join(output_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)

        audio_files["original"] = os.path.join(audio_dir, "ablation_original.wav")
        save_audio(vocoder, spec_trimmed, audio_files["original"], sample_rate, device)

        audio_files["recon_orig_emb"] = os.path.join(audio_dir, "ablation_recon_orig_emb.wav")
        save_audio(vocoder, recon_orig_trimmed, audio_files["recon_orig_emb"], sample_rate, device)

        audio_files["recon_zero_emb"] = os.path.join(audio_dir, "ablation_recon_zero_emb.wav")
        save_audio(vocoder, recon_zero_trimmed, audio_files["recon_zero_emb"], sample_rate, device)

        audio_files["recon_mean_emb"] = os.path.join(audio_dir, "ablation_recon_mean_emb.wav")
        save_audio(vocoder, recon_mean_trimmed, audio_files["recon_mean_emb"], sample_rate, device)

        audio_files["recon_rand_emb"] = os.path.join(audio_dir, "ablation_recon_rand_emb.wav")
        save_audio(vocoder, recon_rand_trimmed, audio_files["recon_rand_emb"], sample_rate, device)

        audio_files["recon_extreme_emb"] = os.path.join(audio_dir, "ablation_recon_extreme_emb.wav")
        save_audio(vocoder, recon_extreme_trimmed, audio_files["recon_extreme_emb"], sample_rate, device)

        print(f"  Generated ablation audio files in {audio_dir}/")

    return {
        "status": "completed",
        "plot_path": save_path,
        "audio_files": audio_files,
        "pairwise_differences": {
            "orig_vs_zero": diff_orig_zero,
            "orig_vs_rand": diff_orig_rand,
            "orig_vs_extreme": diff_orig_extreme,
            "zero_vs_extreme": diff_zero_extreme,
        }
    }


def test_speaker_interpolation(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: str,
    num_interpolations: int = 5,
    num_steps: int = 5,
    vocoder: nn.Module = None,
    sample_rate: int = 16000,
    use_diverse_pairs: bool = True,
) -> dict:
    """
    Interpolate between speaker embeddings with fixed latent.
    Should show smooth transition between voices.

    Args:
        num_interpolations: Number of different interpolation examples to generate
        num_steps: Number of interpolation steps between speakers
        use_diverse_pairs: If True, select maximally diverse speaker pairs
    """
    print("\n" + "=" * 60)
    print("TEST: Speaker Embedding Interpolation")
    print("=" * 60)

    model.eval()

    # Collect samples from multiple speakers
    samples_by_speaker = defaultdict(list)
    min_speakers_needed = max(10, num_interpolations * 2) if use_diverse_pairs else 2
    max_samples_per_speaker = 3

    print(f"Collecting samples from at least {min_speakers_needed} speakers...")

    for batch in dataloader:
        if "speaker_embedding" not in batch or batch["speaker_embedding"] is None:
            print("WARNING: No speaker embeddings in dataset, skipping interpolation test")
            return {"status": "skipped", "reason": "no speaker embeddings"}

        spec = batch["mel_spec"]
        speaker_emb = batch["speaker_embedding"]
        lengths = batch.get("mel_spec_lengths", None)

        if "speaker_id" in batch:
            speaker_ids = batch["speaker_id"]
        elif "speaker_ids" in batch:
            speaker_ids = batch["speaker_ids"]
        elif "speaker" in batch:
            speaker_ids = batch["speaker"]
        else:
            speaker_ids = list(range(spec.shape[0]))

        for i in range(spec.shape[0]):
            sid = speaker_ids[i] if isinstance(speaker_ids, list) else speaker_ids[i].item()
            if len(samples_by_speaker[sid]) >= max_samples_per_speaker:
                continue
            length = lengths[i].item() if lengths is not None else spec.shape[-1]
            samples_by_speaker[sid].append({
                "mel_spec": spec[i:i+1],
                "speaker_embedding": speaker_emb[i:i+1],
                "length": length,
            })

        if len(samples_by_speaker) >= min_speakers_needed:
            break

    if len(samples_by_speaker) < 2:
        print("WARNING: Need at least 2 speakers for interpolation")
        return {"status": "skipped", "reason": "insufficient speakers"}

    print(f"Collected samples from {len(samples_by_speaker)} speakers")

    # Select speaker pairs
    if use_diverse_pairs and len(samples_by_speaker) >= 4:
        diverse_pairs = select_diverse_speaker_pairs(
            samples_by_speaker,
            num_pairs=num_interpolations,
            min_samples_per_speaker=1
        )
        speakers = [diverse_pairs[0][0], diverse_pairs[0][1]]
    else:
        speakers = list(samples_by_speaker.keys())[:2]

    sample_a = samples_by_speaker[speakers[0]][0]
    sample_b = samples_by_speaker[speakers[1]][0]

    spec = sample_a["mel_spec"].to(device)
    emb_a = sample_a["speaker_embedding"].to(device)
    emb_b = sample_b["speaker_embedding"].to(device)
    length = sample_a["length"]

    with torch.no_grad():
        mu, logvar = model.encode(spec)
        z = model.reparameterize(mu, logvar)

        # Interpolate
        alphas = np.linspace(0, 1, num_steps)
        recons = []
        recons_tensor = []  # For audio generation

        for alpha in alphas:
            interp_emb = (1 - alpha) * emb_a + alpha * emb_b
            recon = model.decode(z, speaker_embedding=interp_emb)
            recon_trimmed = trim_to_length(recon, length)
            recons.append(recon_trimmed[0, 0].cpu().numpy())
            recons_tensor.append(recon_trimmed)

    # Trim original spec for visualization
    spec_trimmed = trim_to_length(spec, length)

    # Plot
    fig, axes = plt.subplots(1, num_steps + 1, figsize=(4 * (num_steps + 1), 4))

    axes[0].imshow(spec_trimmed[0, 0].cpu().numpy(), aspect='auto', origin='lower')
    axes[0].set_title("Original Input")

    for i, (alpha, recon) in enumerate(zip(alphas, recons)):
        axes[i + 1].imshow(recon, aspect='auto', origin='lower')
        axes[i + 1].set_title(f"alpha={alpha:.2f}")

    for ax in axes:
        ax.set_xlabel("Time")
        ax.set_ylabel("Mel bin")

    plt.suptitle(f"Speaker Interpolation: {speakers[0]} -> {speakers[1]}")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "speaker_interpolation.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved interpolation plot to {save_path}")

    # Generate audio if vocoder available
    audio_files = {}
    if vocoder is not None:
        audio_dir = os.path.join(output_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)

        audio_files["original"] = os.path.join(audio_dir, "interpolation_original.wav")
        save_audio(vocoder, spec_trimmed, audio_files["original"], sample_rate, device)

        for i, (alpha, recon_t) in enumerate(zip(alphas, recons_tensor)):
            audio_path = os.path.join(audio_dir, f"interpolation_alpha_{alpha:.2f}.wav")
            save_audio(vocoder, recon_t, audio_path, sample_rate, device)
            audio_files[f"alpha_{alpha:.2f}"] = audio_path

        print(f"  Generated interpolation audio files in {audio_dir}/")

    return {"status": "completed", "plot_path": save_path, "speakers": speakers, "audio_files": audio_files}


def test_content_similarity(
    latents: torch.Tensor,
    speaker_ids: list,
) -> dict:
    """
    Test if same-content samples from different speakers have similar latents.

    LIMITATIONS:
    - This test is a PROXY measure, not a direct content similarity test
    - Ideally requires parallel data (same utterance from different speakers) which
      most datasets don't have
    - Instead, we compare within-speaker vs across-speaker latent similarity
    - If ALL latents have high similarity (e.g., cosine sim > 0.9), the difference
      between within/across will be small regardless of disentanglement quality
    - High baseline similarity can mask speaker clustering
    - For better results, consider:
      1. Using datasets with parallel recordings (e.g., VCTK, CMU Arctic)
      2. Computing similarity after PCA to amplify differences
      3. Using the speaker classifier test as the primary disentanglement metric
    """
    print("\n" + "=" * 60)
    print("TEST: Content Similarity Analysis")
    print("=" * 60)
    print("NOTE: This test uses within vs across-speaker similarity as a proxy.")
    print("      If all latents are highly similar, differences will be small.")

    # Flatten latents
    if latents.dim() == 4:
        latents_flat = latents.mean(dim=[2, 3])
    elif latents.dim() == 3:
        latents_flat = latents.mean(dim=2)
    else:
        latents_flat = latents

    # Normalize for cosine similarity
    latents_norm = F.normalize(latents_flat, dim=1)

    # Group by speaker
    speaker_to_indices = defaultdict(list)
    for i, sid in enumerate(speaker_ids):
        speaker_to_indices[sid].append(i)

    unique_speakers = list(speaker_to_indices.keys())

    if len(unique_speakers) < 2:
        print("WARNING: Need at least 2 speakers")
        return {"status": "skipped", "reason": "insufficient speakers"}

    # Compute within-speaker similarity
    within_sims = []
    for sid, indices in speaker_to_indices.items():
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                sim = F.cosine_similarity(
                    latents_norm[indices[i]:indices[i]+1],
                    latents_norm[indices[j]:indices[j]+1]
                ).item()
                within_sims.append(sim)

    # Compute across-speaker similarity (sample pairs)
    across_sims = []
    num_pairs = min(1000, len(latents_norm) * (len(latents_norm) - 1) // 2)

    for _ in range(num_pairs):
        # Pick two different speakers
        s1, s2 = random.sample(unique_speakers, 2)
        i = random.choice(speaker_to_indices[s1])
        j = random.choice(speaker_to_indices[s2])
        sim = F.cosine_similarity(
            latents_norm[i:i+1],
            latents_norm[j:j+1]
        ).item()
        across_sims.append(sim)

    within_mean = np.mean(within_sims) if within_sims else 0
    within_std = np.std(within_sims) if within_sims else 0
    across_mean = np.mean(across_sims) if across_sims else 0
    across_std = np.std(across_sims) if across_sims else 0

    print(f"Within-speaker similarity:  {within_mean:.4f} +/- {within_std:.4f}")
    print(f"Across-speaker similarity:  {across_mean:.4f} +/- {across_std:.4f}")
    print(f"Difference: {within_mean - across_mean:.4f}")

    if within_mean - across_mean < 0.1:
        print("RESULT: GOOD disentanglement - within/across speaker similarity similar")
        print("        (latents don't cluster by speaker)")
    else:
        print("RESULT: Some speaker clustering in latent space")
        print("        (within-speaker samples more similar than across-speaker)")

    return {
        "within_speaker_similarity": {"mean": within_mean, "std": within_std},
        "across_speaker_similarity": {"mean": across_mean, "std": across_std},
        "difference": within_mean - across_mean,
    }


def main():
    parser = argparse.ArgumentParser(description="Test speaker/content disentanglement in audio VAE")
    parser.add_argument("--vae_checkpoint", type=str, required=True,
                        help="Path to VAE checkpoint directory")
    parser.add_argument("--vae_config", type=str, required=True,
                        help="VAE config name (e.g., 'small', 'medium')")
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Path to cached dataset directory")
    parser.add_argument("--output_dir", type=str, default="disentanglement_results",
                        help="Directory to save results")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of samples to analyze")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for processing")
    parser.add_argument("--latent_channels", type=int, default=32,
                        help="Number of latent channels")
    parser.add_argument("--speaker_embedding_dim", type=int, default=192,
                        help="Speaker embedding dimension (default: 192, depends on speaker encoder used during preprocessing)")
    parser.add_argument("--normalize_speaker_embedding", action="store_true",
                        help="L2-normalize speaker embeddings before conditioning")
    parser.add_argument("--film_scale_bound", type=float, default=0.5,
                        help="FiLM scale bound (default: 0.5)")
    parser.add_argument("--film_shift_bound", type=float, default=0.5,
                        help="FiLM shift bound (default: 0.5)")
    parser.add_argument("--zero_init_film_bias", action="store_true",
                        help="Zero initialize FiLM bias parameters")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--skip_classifier", action="store_true",
                        help="Skip speaker classifier test")
    parser.add_argument("--skip_tsne", action="store_true",
                        help="Skip t-SNE visualization")
    parser.add_argument("--skip_conversion", action="store_true",
                        help="Skip voice conversion test")
    parser.add_argument("--skip_ablation", action="store_true",
                        help="Skip embedding ablation test")
    parser.add_argument("--skip_interpolation", action="store_true",
                        help="Skip speaker interpolation test")
    parser.add_argument("--max_mel_frames", type=int, default=1875,
                        help="Max mel frames to pad/truncate to (default: 1875 = 30s at 16kHz)")
    parser.add_argument("--vocoder_checkpoint", type=str, default=None,
                        help="Path to vocoder checkpoint for audio generation (optional)")
    parser.add_argument("--vocoder_config", type=str, default="tiny_attention_freq_domain_vocoder",
                        help="Vocoder config name (default: tiny_attention_freq_domain_vocoder)")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Audio sample rate (default: 16000)")
    parser.add_argument("--num_conversions", type=int, default=5,
                        help="Number of voice conversion examples to generate (default: 5)")
    parser.add_argument("--num_interpolations", type=int, default=5,
                        help="Number of speaker interpolation examples to generate (default: 5)")
    parser.add_argument("--num_ablations", type=int, default=3,
                        help="Number of embedding ablation examples to generate (default: 3)")
    parser.add_argument("--use_diverse_pairs", action="store_true", default=True,
                        help="Select maximally diverse speaker pairs for tests (default: True)")
    parser.add_argument("--no_diverse_pairs", action="store_false", dest="use_diverse_pairs",
                        help="Disable diverse speaker pair selection")

    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=False)

    print("=" * 60)
    print("SPEAKER/CONTENT DISENTANGLEMENT TEST")
    print("=" * 60)
    print(f"VAE checkpoint: {args.vae_checkpoint}")
    print(f"VAE config: {args.vae_config}")
    print(f"Dataset: {args.cache_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {device}")
    print()

    # Load model
    print("Loading VAE model...")
    print(f"  Speaker embedding dim: {args.speaker_embedding_dim}")
    model = load_audio_vae(
        args.vae_checkpoint,
        args.vae_config,
        args.latent_channels,
        args.speaker_embedding_dim,
        args.normalize_speaker_embedding,
        args.film_scale_bound,
        args.film_shift_bound,
        args.zero_init_film_bias,
    )
    model = model.to(device)
    model.eval()

    # Load vocoder if specified
    vocoder = None
    if args.vocoder_checkpoint:
        print(f"Loading vocoder from {args.vocoder_checkpoint}...")
        vocoder = load_vocoder(args.vocoder_checkpoint, args.vocoder_config, device)
        print(f"Vocoder loaded: {args.vocoder_config}")
    else:
        print("No vocoder specified - skipping audio generation")

    # Load dataset
    print("Loading dataset...")
    from shard_utils import AudioVAEShardedDataset, AudioVAEDataCollator

    dataset = AudioVAEShardedDataset(shard_dir=args.cache_dir)
    collator = AudioVAEDataCollator(
        audio_max_frames=args.max_mel_frames,
        speaker_embedding_dim=args.speaker_embedding_dim,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )

    # Extract latents
    print(f"\nExtracting latents from {args.num_samples} samples...")
    print(f"Padding mel specs to {args.max_mel_frames} frames for consistent latent sizes")
    data = extract_latents(model, dataloader, device, args.num_samples, args.max_mel_frames)

    results = {}

    # Run tests
    if not args.skip_classifier:
        results["speaker_classifier"] = test_speaker_classifier(
            data["latents"], data["speaker_ids"], device
        )

    if not args.skip_tsne:
        results["tsne_visualization"] = test_latent_visualization(
            data["latents"], data["speaker_ids"], args.output_dir,
            mel_lengths=data["mel_lengths"]
        )

    if not args.skip_conversion:
        results["voice_conversion"] = test_voice_conversion(
            model, dataloader, device, args.output_dir,
            num_conversions=args.num_conversions,
            vocoder=vocoder, sample_rate=args.sample_rate,
            use_diverse_pairs=args.use_diverse_pairs,
        )

    if not args.skip_ablation:
        results["embedding_ablation"] = test_speaker_embedding_ablation(
            model, dataloader, device, args.output_dir,
            vocoder=vocoder, sample_rate=args.sample_rate
        )

    if not args.skip_interpolation:
        results["speaker_interpolation"] = test_speaker_interpolation(
            model, dataloader, device, args.output_dir,
            num_interpolations=args.num_interpolations,
            vocoder=vocoder, sample_rate=args.sample_rate,
            use_diverse_pairs=args.use_diverse_pairs,
        )

    results["content_similarity"] = test_content_similarity(
        data["latents"], data["speaker_ids"]
    )

    # Save results summary
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        # Convert non-serializable items
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(results, f, indent=2, default=convert)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Results saved to {args.output_dir}/")
    print(f"  - results.json: Numerical metrics")
    print(f"  - tsne_by_speaker.png: Latent space colored by speaker")
    print(f"  - tsne_by_duration.png: Latent space colored by audio duration")
    print(f"  - voice_conversion_*.png: Voice conversion examples")
    print(f"  - embedding_ablation.png: Ablation study")
    print(f"  - speaker_interpolation.png: Interpolation visualization")


if __name__ == "__main__":
    main()
