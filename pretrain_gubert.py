#!/usr/bin/env python3
"""
GuBERT Pretraining Script

Supports two training modes:
1. CTC mode (--mode ctc): ASR-based training with transcriptions
   - CTC loss forces encoding of linguistic content
   - GRL speaker classification removes speaker information

2. Masked mode (--mode masked): Self-supervised HuBERT-style training
   - Masked prediction loss learns acoustic structure
   - GRL speaker classification removes speaker information

Visualizations logged to TensorBoard:
- t-SNE of features colored by speaker (both modes)
- Feature heatmaps (both modes)
- Sample transcriptions with WER/CER (CTC mode only)
- Speaker classifier accuracy over time (both modes)
- GRL alpha schedule (both modes)
- Codebook usage (masked mode only)
"""

import os
import time
from tqdm import tqdm

# Force non-interactive backend before importing pyplot
import matplotlib
matplotlib.use('Agg')

os.environ["DEEPSPEED_UNIT_TEST"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_TIMEOUT"] = "1200000"

import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset, Sampler
from shard_utils import ShardAwareSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers.integrations import TensorBoardCallback

from model.audio.gubert import (
    GuBERTEncoder, GuBERTConfig, CTCVocab, GUBERT_CONFIGS,
    MaskedGuBERTEncoder, MaskedGuBERTConfig, MASKED_GUBERT_CONFIGS,
)
from utils import megatransformer_utils


# ============================================================================
# Data Augmentation
# ============================================================================

class MelSpecAugmentation:
    """
    Mel spectrogram augmentation for GuBERT training.

    Helps prevent memorization by making each pass through the data different.
    Applied on-the-fly during training.

    Note: Avoids time/frequency masking as they would conflict with masked regression.
    """

    def __init__(
        self,
        # Noise injection
        noise_prob: float = 0.5,          # Probability of adding noise
        noise_std_range: tuple = (0.01, 0.1),  # Range for noise std
        # Gain augmentation
        gain_prob: float = 0.5,           # Probability of gain change
        gain_range: tuple = (0.8, 1.2),   # Min/max gain multiplier
        # Frequency shifting (approximate pitch shift)
        freq_shift_prob: float = 0.3,     # Probability of freq shift
        freq_shift_max: int = 4,          # Max bins to shift
        # Time warping (resample to slightly different length)
        time_warp_prob: float = 0.3,      # Probability of time warp
        time_warp_range: tuple = (0.9, 1.1),  # Speed factor range
    ):
        self.noise_prob = noise_prob
        self.noise_std_range = noise_std_range

        self.gain_prob = gain_prob
        self.gain_range = gain_range

        self.freq_shift_prob = freq_shift_prob
        self.freq_shift_max = freq_shift_max

        self.time_warp_prob = time_warp_prob
        self.time_warp_range = time_warp_range

    def __call__(self, mel_spec: torch.Tensor, mel_length: int) -> tuple:
        """
        Apply augmentation to mel spectrogram.

        Args:
            mel_spec: [n_mels, T] mel spectrogram
            mel_length: actual length (before padding)

        Returns:
            Tuple of (augmented mel spectrogram [n_mels, T'], new_length)
        """
        mel_spec = mel_spec.clone()
        n_mels, T = mel_spec.shape
        valid_T = min(mel_length, T)

        # Extract valid portion for augmentation
        valid_mel = mel_spec[:, :valid_T]

        # Noise injection (varying intensity)
        if torch.rand(1).item() < self.noise_prob:
            noise_std = torch.empty(1).uniform_(*self.noise_std_range).item()
            noise = torch.randn_like(valid_mel) * noise_std
            valid_mel = valid_mel + noise

        # Gain augmentation
        if torch.rand(1).item() < self.gain_prob:
            gain = torch.empty(1).uniform_(*self.gain_range).item()
            valid_mel = valid_mel * gain

        # Frequency shifting (approximate pitch shift)
        if torch.rand(1).item() < self.freq_shift_prob:
            shift = torch.randint(-self.freq_shift_max, self.freq_shift_max + 1, (1,)).item()
            if shift != 0:
                valid_mel = torch.roll(valid_mel, shifts=shift, dims=0)
                # Zero out the wrapped-around bins
                if shift > 0:
                    valid_mel[:shift, :] = 0
                else:
                    valid_mel[shift:, :] = 0

        # Time warping (resample to different length)
        if torch.rand(1).item() < self.time_warp_prob:
            warp_factor = torch.empty(1).uniform_(*self.time_warp_range).item()
            new_valid_T = int(valid_T * warp_factor)
            new_valid_T = max(10, new_valid_T)  # Ensure minimum length

            # Resample using interpolation
            valid_mel = valid_mel.unsqueeze(0)  # [1, n_mels, T]
            valid_mel = F.interpolate(valid_mel, size=new_valid_T, mode='linear', align_corners=False)
            valid_mel = valid_mel.squeeze(0)  # [n_mels, new_T]
            valid_T = new_valid_T

        # Reconstruct full tensor (with padding if needed)
        if valid_T != T:
            new_mel = torch.zeros(n_mels, max(valid_T, T), dtype=mel_spec.dtype)
            new_mel[:, :valid_T] = valid_mel
            mel_spec = new_mel[:, :T] if valid_T > T else new_mel
            if valid_T > T:
                valid_T = T  # Truncate if warped longer than original
        else:
            mel_spec[:, :valid_T] = valid_mel

        return mel_spec, valid_T


# ============================================================================
# Dataset
# ============================================================================

class GuBERTShardedDataset(Dataset):
    """
    Efficient dataset for loading preprocessed GuBERT training data from shards.

    Uses binary search on shard offsets instead of building a full index array,
    consistent with AudioVAEShardedDataset for memory efficiency.
    """

    SHARD_INDEX_FILE = "shard_index.json"

    def __init__(
        self,
        shard_dir: str,
        max_samples: Optional[int] = None,
        mode: str = None,
        augmentation: Optional[MelSpecAugmentation] = None,
    ):
        self.shard_dir = shard_dir
        self.max_samples = max_samples
        self.augmentation = augmentation

        # Load config
        config_path = os.path.join(shard_dir, "config.json")
        with open(config_path) as f:
            self.config = json.load(f)

        # Determine mode from config if not specified
        self.mode = mode or self.config.get("mode", "ctc")

        # Try to load cached shard index first (fast path)
        index_path = os.path.join(shard_dir, self.SHARD_INDEX_FILE)
        if os.path.exists(index_path):
            self._load_cached_index(index_path)
        else:
            self._build_and_cache_index(index_path)

        # Apply max_samples limit
        if self.max_samples is not None and self.max_samples < self.total_samples:
            self.total_samples = self.max_samples

        # LRU cache for loaded shards
        self._cache = {}
        self._cache_order = []
        self._cache_size = 3

        # CTC vocab only needed in CTC mode
        self.vocab = CTCVocab() if self.mode == "ctc" else None
        self.num_speakers = self.config.get("num_speakers", 1)

    def _load_cached_index(self, index_path: str):
        """Load pre-computed shard index from JSON file."""
        with open(index_path, "r") as f:
            index_data = json.load(f)

        self.shard_files = index_data["shard_files"]
        self.shard_offsets = index_data["shard_offsets"]
        self.total_samples = index_data["total_samples"]

    def _build_and_cache_index(self, index_path: str):
        """Build shard index by scanning all shards, then cache to JSON."""
        from tqdm import tqdm

        # Find all shards
        self.shard_files = sorted([
            f for f in os.listdir(self.shard_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        ])

        if not self.shard_files:
            raise ValueError(f"No shard files found in {self.shard_dir}")

        # Build index by loading each shard to get sample count
        self.shard_offsets = []
        self.total_samples = 0

        print(f"Building shard index for {len(self.shard_files)} GuBERT shards (first time only)...")
        for shard_file in tqdm(self.shard_files):
            self.shard_offsets.append(self.total_samples)
            shard_path = os.path.join(self.shard_dir, shard_file)
            # Load only to get num_samples, then discard
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            self.total_samples += shard["num_samples"]
            del shard

        print(f"Total samples: {self.total_samples:,}")

        # Cache the index for next time
        index_data = {
            "shard_files": self.shard_files,
            "shard_offsets": self.shard_offsets,
            "total_samples": self.total_samples,
        }
        try:
            with open(index_path, "w") as f:
                json.dump(index_data, f, indent=2)
            print(f"Cached shard index to {index_path}")
        except Exception as e:
            print(f"Warning: could not cache shard index: {e}")

    def _find_shard(self, idx: int) -> tuple:
        """Find which shard contains the given index using binary search."""
        lo, hi = 0, len(self.shard_offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.shard_offsets[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        return lo, idx - self.shard_offsets[lo]

    def _load_shard(self, shard_idx: int) -> dict:
        """Load shard with LRU caching."""
        if shard_idx in self._cache:
            # Move to end of LRU order
            self._cache_order.remove(shard_idx)
            self._cache_order.append(shard_idx)
            return self._cache[shard_idx]

        shard_path = os.path.join(self.shard_dir, self.shard_files[shard_idx])
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)

        self._cache[shard_idx] = shard
        self._cache_order.append(shard_idx)

        # Evict oldest if over capacity
        while len(self._cache) > self._cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        return shard

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int) -> dict:
        shard_idx, sample_idx = self._find_shard(idx)
        shard = self._load_shard(shard_idx)

        mel_spec = shard["mel_specs"][sample_idx]  # [n_mels, T]
        mel_length = shard["mel_lengths"][sample_idx]

        # Apply augmentation (only during training, not eval)
        if self.augmentation is not None:
            mel_spec, new_length = self.augmentation(mel_spec, mel_length.item())
            mel_length = torch.tensor(new_length, dtype=mel_length.dtype)

        result = {
            "mel_spec": mel_spec,
            "mel_length": mel_length,
            "speaker_id": shard["speaker_ids"][sample_idx],
        }

        # Add text data only in CTC mode
        if self.mode == "ctc" and "text_tokens" in shard:
            result["text_tokens"] = shard["text_tokens"][sample_idx]  # [L]
            result["text_length"] = shard["text_lengths"][sample_idx]

        return result

    def get_sampler(self, shuffle: bool = True, seed: int = 42) -> ShardAwareSampler:
        """
        Get a shard-aware sampler for efficient training.

        This sampler groups indices by shard, ensuring each shard is loaded only
        once per epoch instead of thrashing between shards on every random access.
        """
        return ShardAwareSampler(
            shard_offsets=self.shard_offsets,
            total_samples=self.total_samples,
            shuffle=shuffle,
            seed=seed,
        )


class GuBERTDataCollator:
    """Collator for batching GuBERT samples with padding."""

    def __init__(self, n_mels: int = 80, max_mel_frames: int = 1875, mode: str = "ctc"):
        self.n_mels = n_mels
        self.max_mel_frames = max_mel_frames
        self.mode = mode

    def __call__(self, batch: List[dict]) -> dict:
        # Find max lengths in batch
        max_mel_len = max(b["mel_length"].item() for b in batch)
        max_mel_len = min(max_mel_len, self.max_mel_frames)

        batch_size = len(batch)

        # Initialize tensors
        mel_specs = torch.zeros(batch_size, self.n_mels, max_mel_len)
        mel_lengths = torch.zeros(batch_size, dtype=torch.long)
        speaker_ids = torch.zeros(batch_size, dtype=torch.long)

        for i, b in enumerate(batch):
            mel_len = min(b["mel_length"].item(), max_mel_len)
            mel_specs[i, :, :mel_len] = b["mel_spec"][:, :mel_len]
            mel_lengths[i] = mel_len
            speaker_ids[i] = b["speaker_id"]

        result = {
            "mel_specs": mel_specs,
            "mel_lengths": mel_lengths,
            "speaker_ids": speaker_ids,
        }

        # Add text data only in CTC mode
        if self.mode == "ctc" and "text_tokens" in batch[0]:
            max_text_len = max(b["text_length"].item() for b in batch)
            text_tokens = torch.zeros(batch_size, max_text_len, dtype=torch.long)
            text_lengths = torch.zeros(batch_size, dtype=torch.long)

            for i, b in enumerate(batch):
                text_len = b["text_length"].item()
                text_tokens[i, :text_len] = b["text_tokens"][:text_len]
                text_lengths[i] = text_len

            result["text_tokens"] = text_tokens
            result["text_lengths"] = text_lengths

        return result


# ============================================================================
# Training Utilities
# ============================================================================

def get_writer(trainer: Trainer) -> Optional[SummaryWriter]:
    """Get TensorBoard writer from trainer callbacks."""
    if hasattr(trainer, "callback_handler"):
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback):
                if callback.tb_writer is not None:
                    return callback.tb_writer
    return None


def compute_wer_cer(pred: str, target: str) -> tuple:
    """Compute Word Error Rate and Character Error Rate."""
    # CER
    cer = levenshtein_distance(pred, target) / max(len(target), 1)

    # WER
    pred_words = pred.split()
    target_words = target.split()
    wer = levenshtein_distance(pred_words, target_words) / max(len(target_words), 1)

    return wer, cer


def levenshtein_distance(s1, s2) -> int:
    """Compute Levenshtein distance between two sequences."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


# ============================================================================
# GRL Alpha Scheduler
# ============================================================================

class GRLAlphaScheduler:
    """
    Schedule GRL alpha from 0 to max_alpha over warmup steps.

    Follows the original DANN paper recommendation:
    alpha = 2 / (1 + exp(-gamma * p)) - 1
    where p progresses from 0 to 1.

    Args:
        warmup_steps: Number of steps to ramp alpha from 0 to max_alpha
        max_alpha: Maximum alpha value (gradient reversal strength)
        gamma: Steepness of the sigmoid ramp
    """

    def __init__(
        self,
        warmup_steps: int = 5000,
        max_alpha: float = 1.0,
        gamma: float = 10.0,
    ):
        self.warmup_steps = warmup_steps
        self.max_alpha = max_alpha
        self.gamma = gamma

    def get_alpha(self, step: int) -> float:
        """Get alpha for a given step. Expects step to already include any offset."""
        if self.warmup_steps == 0:
            return self.max_alpha

        p = min(step / self.warmup_steps, 1.0)
        alpha = 2.0 / (1.0 + np.exp(-self.gamma * p)) - 1.0
        return float(alpha * self.max_alpha)


# ============================================================================
# Visualization Callback
# ============================================================================

class GuBERTVisualizationCallback(TrainerCallback):
    """
    Callback for logging GuBERT visualizations to TensorBoard.

    Logs:
    - t-SNE of features colored by speaker
    - Feature heatmaps
    - Sample transcriptions
    - CER/WER metrics
    """

    def __init__(
        self,
        eval_dataset: GuBERTShardedDataset,
        vocab: CTCVocab,
        visualization_steps: int = 1000,
        num_tsne_samples: int = 150,
        num_transcription_samples: int = 8,
        max_speakers_for_tsne: int = 15,
    ):
        self.eval_dataset = eval_dataset
        self.vocab = vocab
        self.visualization_steps = visualization_steps
        self.num_tsne_samples = num_tsne_samples
        self.num_transcription_samples = num_transcription_samples
        self.max_speakers_for_tsne = max_speakers_for_tsne
        self.trainer: Optional[Trainer] = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer = kwargs.get("model")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.visualization_steps != 0:
            return

        if model is None:
            return

        writer = get_writer(kwargs.get("trainer", self.trainer))
        if writer is None:
            return

        model.eval()
        device = next(model.parameters()).device

        try:
            self._log_tsne(model, writer, state.global_step, device)
            self._log_transcriptions(model, writer, state.global_step, device)
            self._log_feature_heatmap(model, writer, state.global_step, device)
        except Exception as e:
            print(f"Visualization error at step {state.global_step}: {e}")

        model.train()

    @torch.no_grad()
    def _log_tsne(self, model, writer, step, device):
        """Log t-SNE visualization of features colored by speaker."""
        # Collect features
        features_list = []
        speakers_list = []
        speaker_counts = {}

        indices = list(range(len(self.eval_dataset)))
        np.random.shuffle(indices)

        for idx in indices:
            if len(features_list) >= self.num_tsne_samples:
                break

            sample = self.eval_dataset[idx]
            speaker_id = sample["speaker_id"].item()

            # Limit samples per speaker for balance
            if speaker_counts.get(speaker_id, 0) >= self.num_tsne_samples // self.max_speakers_for_tsne:
                continue

            # Skip if we have enough speakers
            if len(speaker_counts) >= self.max_speakers_for_tsne and speaker_id not in speaker_counts:
                continue

            mel_spec = sample["mel_spec"].unsqueeze(0).to(device)
            mel_length = sample["mel_length"].unsqueeze(0)

            result = model(mel_spec, lengths=mel_length, grl_alpha=0.0)
            feat = result["features"]  # [1, T, D]

            # Mean pool over time
            feat_pooled = feat.mean(dim=1).cpu().numpy()  # [1, D]

            features_list.append(feat_pooled[0])
            speakers_list.append(speaker_id)
            speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1

        if len(features_list) < 10:
            return

        features = np.array(features_list)
        speakers = np.array(speakers_list)

        # Run PCA (much faster than t-SNE)
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(features)

        # Create figure - single scatter call for speed
        fig, ax = plt.subplots(figsize=(8, 8))
        unique_speakers = np.unique(speakers)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_speakers)))
        speaker_colors = [colors[list(unique_speakers).index(s) % len(colors)] for s in speakers]
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=speaker_colors, alpha=0.7, s=30)

        ax.set_title(f"PCA of GuBERT Features (Step {step})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        plt.tight_layout()
        writer.add_figure("visualizations/pca_by_speaker", fig, step)
        plt.close(fig)

    @torch.no_grad()
    def _log_transcriptions(self, model, writer, step, device):
        """Log sample transcriptions with CER/WER."""
        transcriptions = []
        total_cer = 0
        total_wer = 0

        indices = np.random.choice(
            len(self.eval_dataset),
            min(self.num_transcription_samples, len(self.eval_dataset)),
            replace=False,
        )

        for idx in indices:
            sample = self.eval_dataset[idx]

            mel_spec = sample["mel_spec"].unsqueeze(0).to(device)
            mel_length = sample["mel_length"].unsqueeze(0)
            text_tokens = sample["text_tokens"]
            text_length = sample["text_length"].item()

            # Get model prediction
            result = model(mel_spec, lengths=mel_length, grl_alpha=0.0)
            asr_logits = result["asr_logits"]  # [1, T, vocab]

            # Greedy decode
            pred_text = self.vocab.ctc_decode_greedy(asr_logits)[0]

            # Get ground truth
            target_text = self.vocab.decode(
                text_tokens[:text_length].tolist(),
                remove_blanks=True,
                collapse_repeats=False,
            )

            # Compute metrics
            wer, cer = compute_wer_cer(pred_text, target_text)
            total_wer += wer
            total_cer += cer

            transcriptions.append({
                "target": target_text,
                "pred": pred_text,
                "cer": cer,
                "wer": wer,
            })

        # Log metrics
        avg_cer = total_cer / len(transcriptions)
        avg_wer = total_wer / len(transcriptions)
        writer.add_scalar("eval/cer", avg_cer, step)
        writer.add_scalar("eval/wer", avg_wer, step)

        # Log sample transcriptions as text
        text_summary = f"**Step {step} Sample Transcriptions**\n\n"
        text_summary += f"Average CER: {avg_cer:.3f} | Average WER: {avg_wer:.3f}\n\n"
        for i, t in enumerate(transcriptions[:4]):
            text_summary += f"**Sample {i + 1}** (CER: {t['cer']:.3f}, WER: {t['wer']:.3f})\n"
            text_summary += f"  Target: `{t['target'][:100]}`\n"
            text_summary += f"  Pred:   `{t['pred'][:100]}`\n\n"

        writer.add_text("transcriptions/samples", text_summary, step)

    @torch.no_grad()
    def _log_feature_heatmap(self, model, writer, step, device):
        """Log feature heatmap for a sample."""
        sample = self.eval_dataset[0]

        mel_spec = sample["mel_spec"].unsqueeze(0).to(device)
        mel_length = sample["mel_length"].unsqueeze(0)

        result = model(mel_spec, lengths=mel_length, grl_alpha=0.0)
        features = result["features"][0].cpu().numpy()  # [T, D]

        # Transpose for heatmap (D, T)
        features = features.T

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Input mel spectrogram
        mel_np = sample["mel_spec"].numpy()
        im0 = axes[0].imshow(mel_np, aspect="auto", origin="lower", cmap="viridis")
        axes[0].set_title("Input Mel Spectrogram")
        axes[0].set_ylabel("Mel Bin")
        plt.colorbar(im0, ax=axes[0])

        # Output features
        im1 = axes[1].imshow(features, aspect="auto", origin="lower", cmap="viridis")
        axes[1].set_title(f"GuBERT Features (dim={features.shape[0]})")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Feature Dim")
        plt.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        writer.add_figure("visualizations/feature_heatmap", fig, step)
        plt.close(fig)


class MaskedGuBERTVisualizationCallback(TrainerCallback):
    """
    Callback for logging MaskedGuBERT visualizations to TensorBoard.

    Logs:
    - t-SNE of features colored by speaker
    - Feature heatmaps with mask overlay
    - Mask prediction visualizations
    """

    def __init__(
        self,
        eval_dataset: GuBERTShardedDataset,
        visualization_steps: int = 1000,
        num_tsne_samples: int = 150,
        max_speakers_for_tsne: int = 15,
        sync_with_eval: bool = False,
    ):
        self.eval_dataset = eval_dataset
        self.visualization_steps = visualization_steps
        self.num_tsne_samples = num_tsne_samples
        self.max_speakers_for_tsne = max_speakers_for_tsne
        self.sync_with_eval = sync_with_eval
        self.trainer: Optional[Trainer] = None

    def _run_visualizations(self, model, state, kwargs):
        """Run all visualizations and log to TensorBoard."""
        if model is None:
            return

        writer = get_writer(kwargs.get("trainer", self.trainer))
        if writer is None:
            return

        model.eval()
        device = next(model.parameters()).device

        try:
            print(f"  Running visualizations at step {state.global_step}...")
            t0 = time.time()
            self._log_tsne(model, writer, state.global_step, device)
            print(f"    PCA: {time.time() - t0:.1f}s")

            t0 = time.time()
            self._log_feature_heatmap(model, writer, state.global_step, device)
            print(f"    Feature heatmap: {time.time() - t0:.1f}s")

            t0 = time.time()
            self._log_mask_prediction(model, writer, state.global_step, device)
            print(f"    Mask prediction: {time.time() - t0:.1f}s")

            t0 = time.time()
            self._log_feature_space_health(model, writer, state.global_step, device)
            print(f"    Feature health: {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"Visualization error at step {state.global_step}: {e}")
            import traceback
            traceback.print_exc()

        model.train()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Skip step-based visualization if syncing with eval
        if self.sync_with_eval:
            return

        if not ((state.global_step == 1) or (state.global_step % self.visualization_steps == 0)) or not state.is_world_process_zero:
            return

        self._run_visualizations(model, state, kwargs)

    @torch.no_grad()
    def _log_tsne(self, model, writer, step, device):
        """Log t-SNE visualization of features colored by speaker."""
        features_list = []
        speakers_list = []
        speaker_counts = {}

        # Use sequential indices from start to avoid random shard access
        # This is much faster than random shuffling across the dataset
        indices = list(range(min(len(self.eval_dataset), self.num_tsne_samples * 10)))

        t_load = time.time()
        for idx in tqdm(indices, desc="t-SNE samples", leave=False):
            if len(features_list) >= self.num_tsne_samples:
                break

            sample = self.eval_dataset[idx]
            speaker_id = sample["speaker_id"].item()

            if speaker_counts.get(speaker_id, 0) >= self.num_tsne_samples // self.max_speakers_for_tsne:
                continue

            if len(speaker_counts) >= self.max_speakers_for_tsne and speaker_id not in speaker_counts:
                continue

            mel_spec = sample["mel_spec"].unsqueeze(0).to(device)
            mel_length = sample["mel_length"].unsqueeze(0)

            # Run in inference mode (no masking)
            result = model(mel_spec, lengths=mel_length, grl_alpha=0.0, mask=None)
            feat = result["features"]  # [1, T, D]

            feat_pooled = feat.mean(dim=1).cpu().numpy()
            features_list.append(feat_pooled[0])
            speakers_list.append(speaker_id)
            speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
        print(f"      Sample loading: {time.time() - t_load:.1f}s")

        if len(features_list) < 10:
            return

        features = np.array(features_list)
        speakers = np.array(speakers_list)

        # Precompute color indices
        unique_speakers = np.unique(speakers)
        speaker_to_idx = {s: i for i, s in enumerate(unique_speakers)}
        color_indices = np.array([speaker_to_idx[s] for s in speakers])

        # PCA
        t_pca = time.time()
        pca = PCA(n_components=2, random_state=42)
        pca_2d = pca.fit_transform(features)
        print(f"      PCA compute: {time.time() - t_pca:.1f}s")

        # t-SNE
        t_tsne = time.time()
        tsne = TSNE(n_components=2, perplexity=min(30, len(features) - 1), random_state=42)
        tsne_2d = tsne.fit_transform(features)
        print(f"      t-SNE compute: {time.time() - t_tsne:.1f}s")

        # Create side-by-side figure
        t_plot = time.time()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(pca_2d[:, 0], pca_2d[:, 1], c=color_indices, cmap='tab20', alpha=0.7, s=20)
        axes[0].set_title(f"PCA (Step {step})")
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")

        axes[1].scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=color_indices, cmap='tab20', alpha=0.7, s=20)
        axes[1].set_title(f"t-SNE (Step {step})")
        axes[1].set_xlabel("t-SNE 1")
        axes[1].set_ylabel("t-SNE 2")

        plt.tight_layout()
        writer.add_figure("visualizations/pca_tsne_by_speaker", fig, step)
        plt.close(fig)
        print(f"      Plot: {time.time() - t_plot:.1f}s")

    @torch.no_grad()
    def _log_feature_heatmap(self, model, writer, step, device):
        """Log feature heatmap for a sample."""
        sample = self.eval_dataset[0]

        mel_spec = sample["mel_spec"].unsqueeze(0).to(device)
        mel_length = sample["mel_length"].unsqueeze(0)
        actual_mel_length = mel_length.item()

        result = model(mel_spec, lengths=mel_length, grl_alpha=0.0, mask=None)
        features = result["features"][0].cpu().numpy()  # [T, D]
        feature_length = result.get("feature_lengths", [features.shape[0]])[0]
        if hasattr(feature_length, 'item'):
            feature_length = feature_length.item()

        # Crop to actual content (non-padded region)
        mel_np = sample["mel_spec"].numpy()[:, :actual_mel_length]
        features_cropped = features[:feature_length, :].T  # [D, T_actual]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        im0 = axes[0].imshow(mel_np, aspect="auto", origin="lower", cmap="viridis")
        axes[0].set_title(f"Input Mel Spectrogram (T={actual_mel_length})")
        axes[0].set_ylabel("Mel Bin")
        plt.colorbar(im0, ax=axes[0])

        # Use per-row normalization to show temporal variation within each dimension
        feat_normalized = (features_cropped - features_cropped.mean(axis=1, keepdims=True)) / (features_cropped.std(axis=1, keepdims=True) + 1e-8)
        im1 = axes[1].imshow(feat_normalized, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-3, vmax=3)
        axes[1].set_title(f"MaskedGuBERT Features (dim={features_cropped.shape[0]}, T={feature_length}) - normalized per dim")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Feature Dim")
        plt.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        writer.add_figure("visualizations/feature_heatmap", fig, step)
        plt.close(fig)

    @torch.no_grad()
    def _log_mask_prediction(self, model, writer, step, device):
        """Log mask prediction visualization."""
        sample = self.eval_dataset[0]

        mel_spec = sample["mel_spec"].unsqueeze(0).to(device)
        mel_length = sample["mel_length"].unsqueeze(0)
        actual_mel_length = sample["mel_length"].item()

        # Run with masking enabled
        model.train()  # Enable masking
        result = model(mel_spec, lengths=mel_length, grl_alpha=0.0)
        model.eval()

        mask = result.get("mask")  # [1, T'] where T' is feature length (subsampled)
        if mask is None:
            return

        mask = mask[0].cpu().numpy()  # [T']
        feature_lengths = result.get("feature_lengths")
        actual_feature_length = feature_lengths[0].item() if feature_lengths is not None else len(mask)

        # Crop mask to actual feature length (remove padding)
        mask_cropped = mask[:actual_feature_length]

        # Get the subsampling stride to rescale mask to mel spectrogram dimensions
        total_stride = model.conv_subsample.total_stride

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Crop mel spectrogram to actual length
        mel_np = sample["mel_spec"].numpy()[:, :actual_mel_length]
        im0 = axes[0].imshow(mel_np, aspect="auto", origin="lower", cmap="viridis")
        axes[0].set_title(f"Input Mel Spectrogram (cropped to {actual_mel_length} frames)")
        axes[0].set_ylabel("Mel Bin")
        plt.colorbar(im0, ax=axes[0])

        # Mask overlay - rescale mask from feature space to mel space
        # Each feature position covers 'total_stride' mel frames
        mask_mel_scale = np.repeat(mask_cropped, total_stride)[:actual_mel_length]
        mask_expanded = np.tile(mask_mel_scale, (mel_np.shape[0], 1))

        axes[1].imshow(mel_np, aspect="auto", origin="lower", cmap="viridis", alpha=0.7)
        axes[1].imshow(mask_expanded, aspect="auto", origin="lower", cmap="Reds", alpha=0.5)
        axes[1].set_title(f"Mask Overlay (red = masked, {mask_cropped.sum()}/{len(mask_cropped)} feature frames)")
        axes[1].set_ylabel("Mel Bin")

        # Prediction similarity (for regression mode)
        if not model.config.use_vq and "predictions" in result and "targets" in result:
            predictions = result["predictions"][0].cpu()[:actual_feature_length]  # [T', D]
            targets = result["targets"][0].cpu()[:actual_feature_length]  # [T', D]

            # Compute cosine similarity at each position
            pred_norm = F.normalize(predictions, dim=-1)
            targ_norm = F.normalize(targets, dim=-1)
            similarity = (pred_norm * targ_norm).sum(dim=-1).numpy()  # [T']

            axes[2].plot(similarity, label="Cosine Similarity", alpha=0.7)
            axes[2].fill_between(range(len(mask_cropped)), 0, 1, where=mask_cropped.astype(bool),
                                alpha=0.3, color='red', label="Masked")
            axes[2].set_title("Prediction Quality (cosine sim) at Each Position")
            axes[2].set_xlabel("Feature Frame")
            axes[2].set_ylabel("Cosine Similarity")
            axes[2].set_ylim(0, 1)
            axes[2].legend()
        else:
            axes[2].text(0.5, 0.5, "VQ mode - see codebook accuracy in metrics",
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title("Prediction Visualization")

        plt.tight_layout()
        writer.add_figure("visualizations/mask_prediction", fig, step)
        plt.close(fig)

    @torch.no_grad()
    def _log_feature_space_health(self, model, writer, step, device, num_samples: int = 50):
        """
        Log feature space health metrics for VAE compatibility.

        Tracks:
        - Per-dimension statistics (mean, std)
        - Feature norms distribution
        - Dimension utilization (dead dimension detection)
        - Temporal smoothness (adjacent frame similarity)
        - Activation histogram
        - Dimension correlation
        """
        all_features = []
        temporal_sims = []

        # Use sequential indices for shard locality
        indices = list(range(min(len(self.eval_dataset), num_samples)))

        temporal_sims_unnorm = []
        temporal_sims_norm = []

        for idx in tqdm(indices, desc="Feature health", leave=False):
            sample = self.eval_dataset[idx]
            mel_spec = sample["mel_spec"].unsqueeze(0).to(device)
            mel_length = sample["mel_length"].unsqueeze(0)

            result = model(mel_spec, lengths=mel_length, grl_alpha=0.0, mask=None)
            # Use pre-LayerNorm features for health metrics since that's what the VAE uses
            # (with normalize=False in extract_features)
            features_unnorm = result["features_unnorm"][0].cpu()  # [T, D]
            features_norm = result["features"][0].cpu()  # [T, D] - post LayerNorm

            all_features.append(features_unnorm)

            # Temporal smoothness: cosine similarity between adjacent frames
            if features_unnorm.shape[0] > 1:
                # Pre-LayerNorm (what VAE will use)
                feat_n = F.normalize(features_unnorm, dim=-1)
                sim = (feat_n[:-1] * feat_n[1:]).sum(dim=-1)  # [T-1]
                temporal_sims_unnorm.append(sim.mean().item())

                # Post-LayerNorm (for comparison)
                feat_n = F.normalize(features_norm, dim=-1)
                sim = (feat_n[:-1] * feat_n[1:]).sum(dim=-1)  # [T-1]
                temporal_sims_norm.append(sim.mean().item())

        # Keep temporal_sims for backward compatibility (use unnorm)
        temporal_sims = temporal_sims_unnorm

        # Concatenate all features: [total_frames, D]
        all_features = torch.cat(all_features, dim=0).numpy()
        total_frames, feat_dim = all_features.shape

        # === Scalar metrics ===
        # Overall statistics
        global_mean = np.mean(all_features)
        global_std = np.std(all_features)
        writer.add_scalar("feature_health/global_mean", global_mean, step)
        writer.add_scalar("feature_health/global_std", global_std, step)

        # Feature norms
        feature_norms = np.linalg.norm(all_features, axis=-1)
        writer.add_scalar("feature_health/mean_norm", np.mean(feature_norms), step)
        writer.add_scalar("feature_health/std_norm", np.std(feature_norms), step)

        # Per-dimension statistics
        dim_means = np.mean(all_features, axis=0)  # [D]
        dim_stds = np.std(all_features, axis=0)    # [D]

        # Dead dimension detection (std < 0.01)
        dead_dims = np.sum(dim_stds < 0.01)
        writer.add_scalar("feature_health/dead_dimensions", dead_dims, step)
        writer.add_scalar("feature_health/dead_dim_ratio", dead_dims / feat_dim, step)

        # Effective dimensionality (via explained variance ratio)
        # Approximate using ratio of squared stds
        var_per_dim = dim_stds ** 2
        var_normalized = var_per_dim / (var_per_dim.sum() + 1e-8)
        entropy = -np.sum(var_normalized * np.log(var_normalized + 1e-8))
        effective_dim = np.exp(entropy)
        writer.add_scalar("feature_health/effective_dimensionality", effective_dim, step)
        writer.add_scalar("feature_health/dim_utilization_ratio", effective_dim / feat_dim, step)

        # Temporal smoothness (log both pre-norm and post-norm for comparison)
        if temporal_sims_unnorm:
            mean_smoothness_unnorm = np.mean(temporal_sims_unnorm)
            writer.add_scalar("feature_health/temporal_smoothness_unnorm", mean_smoothness_unnorm, step)
            # Keep old metric name for backward compatibility
            writer.add_scalar("feature_health/temporal_smoothness", mean_smoothness_unnorm, step)
        if temporal_sims_norm:
            mean_smoothness_norm = np.mean(temporal_sims_norm)
            writer.add_scalar("feature_health/temporal_smoothness_norm", mean_smoothness_norm, step)

        # Debug print to console
        if temporal_sims_unnorm and temporal_sims_norm:
            print(f"    [Smoothness] pre-norm={mean_smoothness_unnorm:.4f}, post-norm={mean_smoothness_norm:.4f}")

        # === Visualizations ===
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Per-dimension mean distribution
        axes[0, 0].hist(dim_means, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(0, color='red', linestyle='--', label='zero')
        axes[0, 0].set_title(f"Per-Dimension Means (global μ={global_mean:.3f})")
        axes[0, 0].set_xlabel("Mean")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].legend()

        # 2. Per-dimension std distribution (dimension utilization)
        axes[0, 1].hist(dim_stds, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(0.01, color='red', linestyle='--', label='dead threshold')
        axes[0, 1].set_title(f"Per-Dimension Stds ({dead_dims}/{feat_dim} dead)")
        axes[0, 1].set_xlabel("Std")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].legend()

        # 3. Feature norms distribution
        axes[0, 2].hist(feature_norms, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 2].set_title(f"Feature Norms (μ={np.mean(feature_norms):.2f}, σ={np.std(feature_norms):.2f})")
        axes[0, 2].set_xlabel("L2 Norm")
        axes[0, 2].set_ylabel("Count")

        # 4. Overall activation histogram
        flat_features = all_features.flatten()
        # Clip for visualization (avoid extreme outliers dominating)
        clip_val = np.percentile(np.abs(flat_features), 99)
        clipped = np.clip(flat_features, -clip_val, clip_val)
        axes[1, 0].hist(clipped, bins=100, edgecolor='black', alpha=0.7)
        axes[1, 0].set_title(f"Activation Distribution (clipped to 99th %ile)")
        axes[1, 0].set_xlabel("Activation Value")
        axes[1, 0].set_ylabel("Count")

        # 5. Dimension correlation matrix (subsample dimensions for visibility)
        max_dims_to_show = 64
        if feat_dim > max_dims_to_show:
            dim_indices = np.linspace(0, feat_dim - 1, max_dims_to_show, dtype=int)
            features_subset = all_features[:, dim_indices]
        else:
            features_subset = all_features

        corr_matrix = np.corrcoef(features_subset.T)
        im = axes[1, 1].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        axes[1, 1].set_title(f"Dimension Correlations ({min(feat_dim, max_dims_to_show)} dims)")
        axes[1, 1].set_xlabel("Dimension")
        axes[1, 1].set_ylabel("Dimension")
        plt.colorbar(im, ax=axes[1, 1])

        # 6. Temporal smoothness histogram (compare pre-norm vs post-norm)
        if temporal_sims_unnorm and temporal_sims_norm:
            axes[1, 2].hist(temporal_sims_unnorm, bins=30, edgecolor='black', alpha=0.5,
                           label=f'pre-norm (μ={np.mean(temporal_sims_unnorm):.3f})', color='blue')
            axes[1, 2].hist(temporal_sims_norm, bins=30, edgecolor='black', alpha=0.5,
                           label=f'post-norm (μ={np.mean(temporal_sims_norm):.3f})', color='orange')
            axes[1, 2].axvline(np.mean(temporal_sims_unnorm), color='blue', linestyle='--')
            axes[1, 2].axvline(np.mean(temporal_sims_norm), color='orange', linestyle='--')
            axes[1, 2].set_title("Temporal Smoothness (adj. frame cos sim)")
            axes[1, 2].set_xlabel("Cosine Similarity")
            axes[1, 2].set_ylabel("Count")
            axes[1, 2].legend()
        elif temporal_sims_unnorm:
            axes[1, 2].hist(temporal_sims_unnorm, bins=30, edgecolor='black', alpha=0.7)
            axes[1, 2].axvline(np.mean(temporal_sims_unnorm), color='red', linestyle='--',
                              label=f'mean={np.mean(temporal_sims_unnorm):.3f}')
            axes[1, 2].set_title("Temporal Smoothness (pre-norm)")
            axes[1, 2].set_xlabel("Cosine Similarity")
            axes[1, 2].set_ylabel("Count")
            axes[1, 2].legend()
        else:
            axes[1, 2].text(0.5, 0.5, "No temporal data", ha='center', va='center')

        plt.suptitle(f"Feature Space Health (Step {step})", fontsize=14)
        plt.tight_layout()
        writer.add_figure("visualizations/feature_space_health", fig, step)
        plt.close(fig)

        # Log summary to console
        print(f"  Feature space: dim={feat_dim}, effective_dim={effective_dim:.1f} ({100*effective_dim/feat_dim:.1f}%), "
              f"dead={dead_dims}, temporal_sim={np.mean(temporal_sims):.3f}" if temporal_sims else
              f"  Feature space: dim={feat_dim}, effective_dim={effective_dim:.1f}, dead={dead_dims}")

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Compute and log eval metrics after evaluation, and optionally run visualizations."""
        if not state.is_world_process_zero:
            return

        if model is None:
            return

        writer = get_writer(kwargs.get("trainer", self.trainer))
        if writer is None:
            return

        model.eval()
        device = next(model.parameters()).device

        try:
            self._log_eval_metrics(model, writer, state.global_step, device)
        except Exception as e:
            print(f"Eval metrics error at step {state.global_step}: {e}")
            import traceback
            traceback.print_exc()

        # Run visualizations in lockstep with eval (if enabled)
        if self.sync_with_eval:
            self._run_visualizations(model, state, kwargs)

    @torch.no_grad()
    def _log_eval_metrics(self, model, writer, step, device, num_samples: int = 100):
        """Compute mask accuracy and speaker accuracy over eval samples."""
        mask_accs = []
        speaker_accs = []
        mask_losses = []

        # Use sequential indices for shard locality
        indices = list(range(min(len(self.eval_dataset), num_samples)))

        for idx in tqdm(indices, desc="Eval metrics", leave=False):
            sample = self.eval_dataset[idx]

            mel_spec = sample["mel_spec"].unsqueeze(0).to(device)
            mel_length = sample["mel_length"].unsqueeze(0)
            speaker_id = sample["speaker_id"].to(device)

            # Run with masking (training mode temporarily)
            model.train()
            result = model(mel_spec, lengths=mel_length, grl_alpha=0.0)
            model.eval()

            mask = result.get("mask")
            if mask is not None and mask.any():
                # Compute mask accuracy
                if model.config.use_vq:
                    prediction_logits = result["prediction_logits"]
                    target_indices = result["target_indices"]
                    masked_logits = prediction_logits[mask]
                    masked_targets = target_indices[mask]
                    pred_indices = masked_logits.argmax(dim=-1)
                    mask_acc = (pred_indices == masked_targets).float().mean().item()
                else:
                    predictions = result["predictions"]
                    targets = result["targets"]
                    masked_preds = F.normalize(predictions[mask], dim=-1)
                    masked_targets = F.normalize(targets[mask], dim=-1)
                    mask_acc = (masked_preds * masked_targets).sum(dim=-1).mean().item()

                mask_accs.append(mask_acc)

                # Compute mask loss
                mask_loss = model.compute_masked_prediction_loss(result)
                mask_losses.append(mask_loss.item())

            # Compute speaker accuracy
            speaker_logits = result["speaker_logits"]
            speaker_pred = speaker_logits.argmax(dim=-1)
            speaker_acc = (speaker_pred == speaker_id).float().item()
            speaker_accs.append(speaker_acc)

        # Log aggregated metrics
        if mask_accs:
            mean_mask_acc = np.mean(mask_accs)
            writer.add_scalar("eval/mask_accuracy", mean_mask_acc, step)
            print(f"  Eval mask accuracy: {mean_mask_acc:.4f}")

        if mask_losses:
            mean_mask_loss = np.mean(mask_losses)
            writer.add_scalar("eval/mask_loss", mean_mask_loss, step)

        if speaker_accs:
            mean_speaker_acc = np.mean(speaker_accs)
            writer.add_scalar("eval/speaker_accuracy", mean_speaker_acc, step)
            print(f"  Eval speaker accuracy: {mean_speaker_acc:.4f} (lower is better for GRL)")


# ============================================================================
# Custom Trainer
# ============================================================================

class GuBERTTrainer(Trainer):
    """
    Custom trainer for GuBERT with CTC + GRL losses.
    """

    def __init__(
        self,
        *args,
        vocab: CTCVocab,
        grl_alpha_scheduler: GRLAlphaScheduler,
        ctc_weight: float = 1.0,
        grl_weight: float = 0.1,
        cmdline: str = "",
        git_commit_hash: str = "",
        step_offset: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vocab = vocab
        self.grl_alpha_scheduler = grl_alpha_scheduler
        self.ctc_weight = ctc_weight
        self.grl_weight = grl_weight
        self.cmdline = cmdline
        self.git_commit_hash = git_commit_hash
        self.step_offset = step_offset if step_offset is not None else 0
        self.has_logged_cli = False

        # CTC loss
        self.ctc_criterion = nn.CTCLoss(blank=vocab.blank_idx, reduction="mean", zero_infinity=True)
        self.speaker_criterion = nn.CrossEntropyLoss()

        # Metrics tracking
        self._step_metrics = {}

        # Set up shard-aware sampler if dataset supports it
        self._shard_sampler = None
        if hasattr(self.train_dataset, 'get_sampler'):
            self._shard_sampler = self.train_dataset.get_sampler(shuffle=True, seed=42)

        # TensorBoard writer (lazily initialized)
        self.writer = None

    def _ensure_tensorboard_writer(self):
        """Get TensorBoard writer from callback."""
        if self.writer is not None:
            return
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback):
                self.writer = callback.tb_writer
                return

    def _get_train_sampler(self) -> Optional[Sampler]:
        """Override to use shard-aware sampler for efficient shard loading."""
        if self._shard_sampler is not None:
            epoch = int(self.state.epoch) if self.state and self.state.epoch else 0
            self._shard_sampler.set_epoch(epoch)
            return self._shard_sampler
        return super()._get_train_sampler()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        global_step = self.state.global_step + self.step_offset
        self._ensure_tensorboard_writer()

        # Log CLI and git hash on first call (logs at resumed step if resuming)
        if not self.has_logged_cli and self.writer is not None:
            self.writer.add_text("training/command_line", self.cmdline, global_step)
            self.writer.add_text("training/git_commit_hash", self.git_commit_hash, global_step)
            self.has_logged_cli = True

        mel_specs = inputs["mel_specs"]
        mel_lengths = inputs["mel_lengths"]
        text_tokens = inputs["text_tokens"]
        text_lengths = inputs["text_lengths"]
        speaker_ids = inputs["speaker_ids"]

        # Get GRL alpha for current step
        grl_alpha = self.grl_alpha_scheduler.get_alpha(global_step)

        # Forward pass
        result = model(mel_specs, lengths=mel_lengths, grl_alpha=grl_alpha)

        asr_logits = result["asr_logits"]  # [B, T, vocab]
        speaker_logits = result["speaker_logits"]  # [B, num_speakers]
        feature_lengths = result["feature_lengths"]  # [B]

        # CTC loss
        # CTC expects [T, B, vocab] and log probabilities
        log_probs = F.log_softmax(asr_logits, dim=-1).permute(1, 0, 2)  # [T, B, vocab]
        ctc_loss = self.ctc_criterion(log_probs, text_tokens, feature_lengths, text_lengths)

        # GRL speaker classification loss
        # We want the classifier to FAIL (be at chance level)
        # But we train it normally - GRL reverses gradients to encoder
        speaker_loss = self.speaker_criterion(speaker_logits, speaker_ids)

        # Speaker accuracy (for logging)
        with torch.no_grad():
            speaker_preds = speaker_logits.argmax(dim=-1)
            speaker_acc = (speaker_preds == speaker_ids).float().mean().item()

        # Combined loss
        total_loss = self.ctc_weight * ctc_loss + self.grl_weight * speaker_loss

        # Log to TensorBoard
        if self.writer is not None and global_step % self.args.logging_steps == 0:
            self.writer.add_scalar("train/ctc_loss", ctc_loss.item(), global_step)
            self.writer.add_scalar("train/speaker_loss", speaker_loss.item(), global_step)
            self.writer.add_scalar("train/speaker_accuracy", speaker_acc, global_step)
            self.writer.add_scalar("train/grl_alpha", grl_alpha, global_step)
            self.writer.add_scalar("train/total_loss", total_loss.item(), global_step)

        if return_outputs:
            return total_loss, result
        return total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to handle GuBERT inputs correctly during evaluation."""
        model.eval()

        with torch.no_grad():
            mel_specs = inputs["mel_specs"]
            mel_lengths = inputs["mel_lengths"]
            text_tokens = inputs["text_tokens"]
            text_lengths = inputs["text_lengths"]
            speaker_ids = inputs["speaker_ids"]

            # Forward pass (no GRL during eval)
            result = model(mel_specs, lengths=mel_lengths, grl_alpha=0.0)

            asr_logits = result["asr_logits"]
            speaker_logits = result["speaker_logits"]
            feature_lengths = result["feature_lengths"]

            # CTC loss
            log_probs = F.log_softmax(asr_logits, dim=-1).permute(1, 0, 2)
            ctc_loss = self.ctc_criterion(log_probs, text_tokens, feature_lengths, text_lengths)

            # Speaker loss
            speaker_loss = self.speaker_criterion(speaker_logits, speaker_ids)

            # Combined loss
            total_loss = self.ctc_weight * ctc_loss + self.grl_weight * speaker_loss

        return (total_loss, None, None)


class MaskedGuBERTTrainer(Trainer):
    """
    Custom trainer for MaskedGuBERT with masked prediction + GRL losses.
    """

    def __init__(
        self,
        *args,
        grl_alpha_scheduler: GRLAlphaScheduler,
        masked_weight: float = 1.0,
        grl_weight: float = 0.1,
        commitment_weight: float = 0.25,
        codebook_weight: float = 1.0,
        variance_weight: float = 1.0,
        cmdline: str = "",
        git_commit_hash: str = "",
        step_offset: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.grl_alpha_scheduler = grl_alpha_scheduler
        self.masked_weight = masked_weight
        self.grl_weight = grl_weight
        self.commitment_weight = commitment_weight
        self.codebook_weight = codebook_weight
        self.variance_weight = variance_weight
        self.cmdline = cmdline
        self.git_commit_hash = git_commit_hash
        self.step_offset = step_offset if step_offset is not None else 0
        self.has_logged_cli = False

        self.speaker_criterion = nn.CrossEntropyLoss()

        # Metrics tracking
        self._step_metrics = {}

        # Set up shard-aware sampler if dataset supports it
        self._shard_sampler = None
        if hasattr(self.train_dataset, 'get_sampler'):
            self._shard_sampler = self.train_dataset.get_sampler(shuffle=True, seed=42)

        # TensorBoard writer (lazily initialized)
        self.writer = None

    def _ensure_tensorboard_writer(self):
        """Get TensorBoard writer from callback."""
        if self.writer is not None:
            return
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback):
                self.writer = callback.tb_writer
                return

    def _get_train_sampler(self) -> Optional[Sampler]:
        """Override to use shard-aware sampler for efficient shard loading."""
        if self._shard_sampler is not None:
            epoch = int(self.state.epoch) if self.state and self.state.epoch else 0
            self._shard_sampler.set_epoch(epoch)
            return self._shard_sampler
        return super()._get_train_sampler()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        global_step = self.state.global_step + self.step_offset
        self._ensure_tensorboard_writer()

        # Log CLI and git hash on first call (logs at resumed step if resuming)
        if not self.has_logged_cli and self.writer is not None:
            self.writer.add_text("training/command_line", self.cmdline, global_step)
            self.writer.add_text("training/git_commit_hash", self.git_commit_hash, global_step)
            self.has_logged_cli = True

        mel_specs = inputs["mel_specs"]
        mel_lengths = inputs["mel_lengths"]
        speaker_ids = inputs["speaker_ids"]

        # Get GRL alpha for current step (use global_step with offset for proper resume)
        grl_alpha = self.grl_alpha_scheduler.get_alpha(global_step)

        # Forward pass (mask is generated automatically in training mode)
        result = model(mel_specs, lengths=mel_lengths, grl_alpha=grl_alpha)

        speaker_logits = result["speaker_logits"]  # [B, num_speakers]
        mask = result["mask"]  # [B, T]

        # Use the model's helper to compute masked prediction loss
        # This handles both VQ and regression modes automatically
        masked_loss = model.compute_masked_prediction_loss(result)

        # Compute prediction accuracy at masked positions (mode-dependent)
        mask_acc = 0.0
        if mask is not None and mask.any():
            with torch.no_grad():
                if model.config.use_vq:
                    # VQ mode: accuracy based on codebook index prediction
                    prediction_logits = result["prediction_logits"]
                    target_indices = result["target_indices"]
                    masked_logits = prediction_logits[mask]
                    masked_targets = target_indices[mask]
                    pred_indices = masked_logits.argmax(dim=-1)
                    mask_acc = (pred_indices == masked_targets).float().mean().item()
                else:
                    # Regression mode: use cosine similarity as accuracy proxy
                    predictions = result["predictions"]
                    targets = result["targets"]
                    masked_preds = F.normalize(predictions[mask], dim=-1)
                    masked_targets = F.normalize(targets[mask], dim=-1)
                    mask_acc = (masked_preds * masked_targets).sum(dim=-1).mean().item()

        # GRL speaker classification loss
        speaker_loss = self.speaker_criterion(speaker_logits, speaker_ids)

        # Speaker accuracy (for logging)
        with torch.no_grad():
            speaker_preds = speaker_logits.argmax(dim=-1)
            speaker_acc = (speaker_preds == speaker_ids).float().mean().item()

        # Combined loss (VQ losses only apply in VQ mode)
        total_loss = self.masked_weight * masked_loss + self.grl_weight * speaker_loss

        # VQ-specific losses (only in VQ mode)
        commitment_loss_val = 0.0
        codebook_loss_val = 0.0
        if model.config.use_vq:
            commitment_loss = result["commitment_loss"]
            codebook_loss = result["codebook_loss"]
            total_loss = total_loss + self.commitment_weight * commitment_loss + self.codebook_weight * codebook_loss
            commitment_loss_val = commitment_loss.item()
            codebook_loss_val = codebook_loss.item()

        # Variance regularization loss (for VAE-friendly features)
        variance_loss_val = 0.0
        temporal_smoothness_val = 0.0
        if model.config.use_variance_reg:
            variance_loss = result["variance_loss"]
            total_loss = total_loss + self.variance_weight * variance_loss
            variance_loss_val = variance_loss.item()
            # Extract temporal smoothness for logging
            if result.get("temporal_smoothness") is not None:
                temporal_smoothness_val = result["temporal_smoothness"].item()

        # Log to TensorBoard
        masked_loss_val = masked_loss.item() if torch.is_tensor(masked_loss) else masked_loss
        if self.writer is not None and self.state.global_step % self.args.logging_steps == 0:
            self.writer.add_scalar("train/masked_loss", masked_loss_val, global_step)
            self.writer.add_scalar("train/speaker_loss", speaker_loss.item(), global_step)
            self.writer.add_scalar("train/speaker_accuracy", speaker_acc, global_step)
            self.writer.add_scalar("train/mask_accuracy", mask_acc, global_step)
            self.writer.add_scalar("train/grl_alpha", grl_alpha, global_step)
            self.writer.add_scalar("train/total_loss", total_loss.item(), global_step)
            if model.config.use_vq:
                self.writer.add_scalar("train/commitment_loss", commitment_loss_val, global_step)
                self.writer.add_scalar("train/codebook_loss", codebook_loss_val, global_step)
            if model.config.use_variance_reg:
                self.writer.add_scalar("train/variance_loss", variance_loss_val, global_step)
                self.writer.add_scalar("train/temporal_smoothness", temporal_smoothness_val, global_step)

        if return_outputs:
            return total_loss, result
        return total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to handle MaskedGuBERT inputs correctly during evaluation."""
        with torch.no_grad():
            mel_specs = inputs["mel_specs"]
            mel_lengths = inputs["mel_lengths"]
            speaker_ids = inputs["speaker_ids"]

            # Generate mask for eval (need to temporarily set training mode for mask generation)
            model.train()  # Enable mask generation
            result = model(mel_specs, lengths=mel_lengths, grl_alpha=0.0)
            model.eval()  # Restore eval mode

            # Compute masked prediction loss (now has valid mask)
            masked_loss = model.compute_masked_prediction_loss(result)
            mask = result["mask"]

            # Compute mask prediction accuracy
            mask_acc = 0.0
            if mask is not None and mask.any():
                if model.config.use_vq:
                    prediction_logits = result["prediction_logits"]
                    target_indices = result["target_indices"]
                    masked_logits = prediction_logits[mask]
                    masked_targets = target_indices[mask]
                    pred_indices = masked_logits.argmax(dim=-1)
                    mask_acc = (pred_indices == masked_targets).float().mean().item()
                else:
                    predictions = result["predictions"]
                    targets = result["targets"]
                    masked_preds = F.normalize(predictions[mask], dim=-1)
                    masked_targets = F.normalize(targets[mask], dim=-1)
                    mask_acc = (masked_preds * masked_targets).sum(dim=-1).mean().item()

            # Speaker loss
            speaker_logits = result["speaker_logits"]
            speaker_loss = self.speaker_criterion(speaker_logits, speaker_ids)

            # Speaker accuracy
            speaker_preds = speaker_logits.argmax(dim=-1)
            speaker_acc = (speaker_preds == speaker_ids).float().mean().item()

            # Combined loss
            total_loss = self.masked_weight * masked_loss + self.grl_weight * speaker_loss

            # Add VQ losses if applicable
            commitment_loss_val = 0.0
            codebook_loss_val = 0.0
            if model.config.use_vq:
                total_loss = total_loss + self.commitment_weight * result["commitment_loss"]
                total_loss = total_loss + self.codebook_weight * result["codebook_loss"]
                commitment_loss_val = result["commitment_loss"].item()
                codebook_loss_val = result["codebook_loss"].item()

            # Add variance loss if applicable
            variance_loss_val = 0.0
            if model.config.use_variance_reg:
                total_loss = total_loss + self.variance_weight * result["variance_loss"]
                variance_loss_val = result["variance_loss"].item()

            # Accumulate metrics for logging at end of eval
            masked_loss_val = masked_loss.item() if torch.is_tensor(masked_loss) else masked_loss
            self._accumulate_eval_metrics({
                "masked_loss": masked_loss_val,
                "speaker_loss": speaker_loss.item(),
                "speaker_accuracy": speaker_acc,
                "mask_accuracy": mask_acc,
                "commitment_loss": commitment_loss_val,
                "codebook_loss": codebook_loss_val,
                "variance_loss": variance_loss_val,
            })

        return (total_loss, None, None)

    def _accumulate_eval_metrics(self, metrics: dict):
        """Accumulate metrics during evaluation for averaging."""
        if not hasattr(self, "_eval_metrics_accum"):
            self._eval_metrics_accum = {}
            self._eval_metrics_count = 0

        for key, value in metrics.items():
            if key not in self._eval_metrics_accum:
                self._eval_metrics_accum[key] = 0.0
            self._eval_metrics_accum[key] += value
        self._eval_metrics_count += 1

    def evaluate(self, *args, **kwargs):
        """Override to log accumulated eval metrics."""
        # Reset accumulators
        self._eval_metrics_accum = {}
        self._eval_metrics_count = 0

        # Run parent evaluate
        output = super().evaluate(*args, **kwargs)

        # Log averaged metrics to TensorBoard
        self._ensure_tensorboard_writer()
        if self.writer is not None and self._eval_metrics_count > 0:
            global_step = self.state.global_step + self.step_offset
            for key, total in self._eval_metrics_accum.items():
                avg_value = total / self._eval_metrics_count
                self.writer.add_scalar(f"eval/{key}", avg_value, global_step)

        return output


# ============================================================================
# Model Configuration
# ============================================================================

def create_model(
    config_name: str,
    num_speakers: int,
    mode: str = "ctc",
    vocab_size: int = 30,
    **overrides,
):
    """Create GuBERT or MaskedGuBERT model from config name with overrides."""
    if mode == "ctc":
        configs = GUBERT_CONFIGS
        if config_name not in configs:
            config_name = "small"
        return GuBERTEncoder.from_config(
            config_name,
            num_speakers=num_speakers,
            vocab_size=vocab_size,
            **overrides,
        )
    else:  # masked mode
        configs = MASKED_GUBERT_CONFIGS
        if config_name not in configs:
            config_name = "small"
        return MaskedGuBERTEncoder.from_config(
            config_name,
            num_speakers=num_speakers,
            **overrides,
        )


# ============================================================================
# Main
# ============================================================================

def main():
    args, unk = megatransformer_utils.parse_args()
    run_dir = os.path.join(args.logging_base_dir, args.run_name)

    # Parse extra arguments
    unk_dict = {}
    for i in range(0, len(unk), 2):
        unk_dict[unk[i].lstrip("-")] = unk[i + 1]

    # Training mode
    mode = unk_dict.get("mode", "ctc")
    if mode not in ["ctc", "masked"]:
        raise ValueError(f"Unknown mode: {mode}. Must be 'ctc' or 'masked'.")

    # Dataset settings
    train_cache_dir = unk_dict.get("train_cache_dir", "./cached_datasets/gubert_train")
    val_cache_dir = unk_dict.get("val_cache_dir", "./cached_datasets/gubert_val")

    # Model settings
    configs = GUBERT_CONFIGS if mode == "ctc" else MASKED_GUBERT_CONFIGS
    config_name = args.config if args.config in configs else "small"

    # Audio settings
    n_mels = int(unk_dict.get("n_mels", 80))
    max_mel_frames = int(unk_dict.get("max_mel_frames", 1875))

    # GRL settings
    grl_warmup_steps = int(unk_dict.get("grl_warmup_steps", 5000))
    grl_max_alpha = float(unk_dict.get("grl_max_alpha", 1.0))
    grl_weight = float(unk_dict.get("grl_weight", 0.1))

    # CTC-specific settings
    ctc_weight = float(unk_dict.get("ctc_weight", 1.0))

    # Masked-specific settings
    masked_weight = float(unk_dict.get("masked_weight", 1.0))
    use_vq = unk_dict.get("use_vq", "false").lower() in ("true", "1", "yes")
    commitment_weight = float(unk_dict.get("commitment_weight", 0.25))
    codebook_weight = float(unk_dict.get("codebook_weight", 1.0))
    num_codebooks = int(unk_dict.get("num_codebooks", 2))
    codebook_size = int(unk_dict.get("codebook_size", 320))

    # Feature variance regularization (for VAE-friendly features)
    use_variance_reg = unk_dict.get("use_variance_reg", "false").lower() in ("true", "1", "yes")
    variance_weight = float(unk_dict.get("variance_weight", 1.0))
    temporal_var_weight = float(unk_dict.get("temporal_var_weight", 0.01))
    temporal_var_min = float(unk_dict.get("temporal_var_min", 0.1))
    dim_var_weight = float(unk_dict.get("dim_var_weight", 0.01))
    dim_var_min = float(unk_dict.get("dim_var_min", 0.1))
    temporal_smoothness_weight = float(unk_dict.get("temporal_smoothness_weight", 0.1))
    temporal_smoothness_max = float(unk_dict.get("temporal_smoothness_max", 0.95))

    # Dropout settings for regularization (helps prevent memorization)
    conv_dropout = float(unk_dict.get("conv_dropout", 0.05))  # Dropout1d in conv frontend
    feature_dropout = float(unk_dict.get("feature_dropout", 0.0))
    head_dropout = float(unk_dict.get("head_dropout", 0.0))
    attention_head_drop = float(unk_dict.get("attention_head_drop", 0.0))  # DropHead on attention

    # Data augmentation settings (prevents memorization)
    use_augmentation = unk_dict.get("use_augmentation", "false").lower() in ("true", "1", "yes")
    aug_noise_prob = float(unk_dict.get("aug_noise_prob", 0.5))
    aug_noise_std_min = float(unk_dict.get("aug_noise_std_min", 0.01))
    aug_noise_std_max = float(unk_dict.get("aug_noise_std_max", 0.1))
    aug_gain_prob = float(unk_dict.get("aug_gain_prob", 0.5))
    aug_gain_min = float(unk_dict.get("aug_gain_min", 0.8))
    aug_gain_max = float(unk_dict.get("aug_gain_max", 1.2))
    aug_freq_shift_prob = float(unk_dict.get("aug_freq_shift_prob", 0.3))
    aug_freq_shift_max = int(unk_dict.get("aug_freq_shift_max", 4))
    aug_time_warp_prob = float(unk_dict.get("aug_time_warp_prob", 0.3))
    aug_time_warp_min = float(unk_dict.get("aug_time_warp_min", 0.9))
    aug_time_warp_max = float(unk_dict.get("aug_time_warp_max", 1.1))

    # Visualization settings (use standard generation_steps arg)
    visualization_steps = args.generation_steps

    # Eval strategy: "steps", "epoch", or "no"
    eval_strategy = unk_dict.get("eval_strategy", "steps" if args.eval_steps > 0 else "no")

    # Sync visualization with eval (run visualizations alongside eval instead of at fixed steps)
    sync_viz_with_eval = unk_dict.get("sync_viz_with_eval", "false").lower() in ("true", "1", "yes")

    # Max samples (for testing)
    max_train_samples = int(unk_dict.get("max_train_samples", 0)) or None
    max_val_samples = int(unk_dict.get("max_val_samples", 0)) or None

    print(f"GuBERT Pretraining")
    print(f"==================")
    print(f"Mode: {mode.upper()}")
    print(f"Config: {config_name}")
    print(f"Run dir: {run_dir}")
    print(f"Train cache: {train_cache_dir}")
    print(f"Val cache: {val_cache_dir}")
    if use_variance_reg:
        print(f"Variance regularization: ENABLED")
        print(f"  variance_weight: {variance_weight}")
        print(f"  temporal_var_weight: {temporal_var_weight}, min: {temporal_var_min}")
        print(f"  dim_var_weight: {dim_var_weight}, min: {dim_var_min}")
        print(f"  temporal_smoothness_weight: {temporal_smoothness_weight}, max: {temporal_smoothness_max}")
    if conv_dropout > 0 or feature_dropout > 0 or head_dropout > 0 or attention_head_drop > 0:
        print(f"Dropout regularization: ENABLED")
        print(f"  conv_dropout: {conv_dropout} (Dropout1d in conv frontend)")
        print(f"  feature_dropout: {feature_dropout}")
        print(f"  head_dropout: {head_dropout} (prediction head)")
        print(f"  attention_head_drop: {attention_head_drop} (DropHead on attention)")
    if use_vq:
        print(f"VQ mode: ENABLED (discrete codebook targets)")
        print(f"  num_codebooks: {num_codebooks}")
        print(f"  codebook_size: {codebook_size}")
        print(f"  commitment_weight: {commitment_weight}")
        print(f"  codebook_weight: {codebook_weight}")
    else:
        print(f"Regression mode: ENABLED (continuous targets)")
    if use_augmentation:
        print(f"Data augmentation: ENABLED")
        print(f"  noise: prob={aug_noise_prob}, std=[{aug_noise_std_min}, {aug_noise_std_max}]")
        print(f"  gain: prob={aug_gain_prob}, range=[{aug_gain_min}, {aug_gain_max}]")
        print(f"  freq_shift: prob={aug_freq_shift_prob}, max_bins={aug_freq_shift_max}")
        print(f"  time_warp: prob={aug_time_warp_prob}, range=[{aug_time_warp_min}, {aug_time_warp_max}]")
    print(f"")

    # Create augmentation for training (if enabled)
    augmentation = None
    if use_augmentation:
        augmentation = MelSpecAugmentation(
            noise_prob=aug_noise_prob,
            noise_std_range=(aug_noise_std_min, aug_noise_std_max),
            gain_prob=aug_gain_prob,
            gain_range=(aug_gain_min, aug_gain_max),
            freq_shift_prob=aug_freq_shift_prob,
            freq_shift_max=aug_freq_shift_max,
            time_warp_prob=aug_time_warp_prob,
            time_warp_range=(aug_time_warp_min, aug_time_warp_max),
        )

    # Load datasets
    print("Loading datasets...")
    # Train dataset gets augmentation; val dataset does not
    train_dataset = GuBERTShardedDataset(train_cache_dir, max_samples=max_train_samples, mode=mode, augmentation=augmentation)
    val_dataset = GuBERTShardedDataset(val_cache_dir, max_samples=max_val_samples, mode=mode)

    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples: {len(val_dataset):,}")
    print(f"  Num speakers: {train_dataset.num_speakers}")
    if mode == "ctc":
        print(f"  Vocab size: {train_dataset.vocab.vocab_size}")
    print(f"")

    # Create model
    print(f"Creating model ({config_name})...")
    if mode == "ctc":
        model = create_model(
            config_name=config_name,
            num_speakers=train_dataset.num_speakers,
            mode=mode,
            vocab_size=train_dataset.vocab.vocab_size,
            n_mels=n_mels,
        )
    else:
        model = create_model(
            config_name=config_name,
            num_speakers=train_dataset.num_speakers,
            mode=mode,
            n_mels=n_mels,
            # VQ mode settings
            use_vq=use_vq,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            # Dropout regularization (helps prevent memorization)
            conv_dropout=conv_dropout,  # Dropout1d in conv frontend
            feature_dropout=feature_dropout,
            head_dropout=head_dropout,
            attention_head_drop=attention_head_drop,  # DropHead on attention
            # Variance regularization for VAE-friendly features
            use_variance_reg=use_variance_reg,
            temporal_var_weight=temporal_var_weight,
            temporal_var_min=temporal_var_min,
            dim_var_weight=dim_var_weight,
            dim_var_min=dim_var_min,
            temporal_smoothness_weight=temporal_smoothness_weight,
            temporal_smoothness_max=temporal_smoothness_max,
        )

    num_params = model.get_num_params()
    print(f"Model: {model}")
    print(f"Total Parameters: {num_params:,}")

    # Debug: show actual config values to verify CLI overrides are applied
    if mode == "masked":
        cfg = model.config
        print(f"\nModel config (verify CLI overrides):")
        print(f"  use_vq: {cfg.use_vq}")
        print(f"  num_codebooks: {cfg.num_codebooks}")
        print(f"  codebook_size: {cfg.codebook_size}")
        print(f"  codebook_dim: {cfg.codebook_dim}")
        print(f"  commitment_weight: {cfg.commitment_weight}")
        print(f"  conv_dropout: {cfg.conv_dropout}")
        print(f"  feature_dropout: {cfg.feature_dropout}")
        print(f"  head_dropout: {cfg.head_dropout}")
        print(f"  attention_head_drop: {cfg.attention_head_drop}")
        print(f"  encoder_dim: {cfg.encoder_dim}")
        print(f"  Note: pre_quantize_proj is Identity when encoder_dim == num_codebooks * codebook_dim")
        print(f"        ({cfg.encoder_dim} == {cfg.num_codebooks} * {cfg.codebook_dim} = {cfg.num_codebooks * cfg.codebook_dim})")

    conv_upsample_params = sum(p.numel() for p in model.conv_subsample.parameters())
    encoder_blocks_params = sum(p.numel() for p in model.encoder_blocks.parameters())
    final_norm_params = sum(p.numel() for p in model.final_norm.parameters())
    prediction_head_params = sum(p.numel() for p in model.prediction_head.parameters())
    print(f"GuBERT Parameters: {conv_upsample_params + encoder_blocks_params + final_norm_params + prediction_head_params:,}")
    print(f"GRL Parameters: {sum(p.numel() for p in model.speaker_classifier.parameters()):,}")

    # Create data collator
    collator = GuBERTDataCollator(n_mels=n_mels, max_mel_frames=max_mel_frames, mode=mode)

    # Create GRL scheduler
    grl_scheduler = GRLAlphaScheduler(
        warmup_steps=grl_warmup_steps,
        max_alpha=grl_max_alpha,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=run_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs if args.num_train_epochs > 0 else 1,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_dir=run_dir,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps if eval_strategy == "steps" else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_safetensors=False,
        load_best_model_at_end=False,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=["tensorboard"],
        max_grad_norm=args.max_grad_norm,
        deepspeed=args.deepspeed_config if args.use_deepspeed else None,
        gradient_checkpointing=args.use_gradient_checkpointing,
        remove_unused_columns=False,  # Keep all columns for custom collator
        local_rank=args.local_rank,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        log_level=args.log_level,
    )

    # Create callbacks
    callbacks = []

    # Visualization callbacks
    if mode == "ctc":
        viz_callback = GuBERTVisualizationCallback(
            eval_dataset=val_dataset,
            vocab=train_dataset.vocab,
            visualization_steps=visualization_steps,
        )
        callbacks.append(viz_callback)
    else:  # masked mode
        viz_callback = MaskedGuBERTVisualizationCallback(
            eval_dataset=val_dataset,
            visualization_steps=visualization_steps,
            sync_with_eval=sync_viz_with_eval,
        )
        callbacks.append(viz_callback)

    # Build command line string for logging
    import sys
    cmdline = " ".join(sys.argv)

    # Create trainer based on mode
    step_offset = args.start_step or 0
    if mode == "ctc":
        trainer = GuBERTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            vocab=train_dataset.vocab,
            grl_alpha_scheduler=grl_scheduler,
            ctc_weight=ctc_weight,
            grl_weight=grl_weight,
            callbacks=callbacks,
            cmdline=cmdline,
            git_commit_hash=args.commit_hash or "",
            step_offset=step_offset,
        )
    else:  # masked mode
        trainer = MaskedGuBERTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            grl_alpha_scheduler=grl_scheduler,
            masked_weight=masked_weight,
            grl_weight=grl_weight,
            commitment_weight=commitment_weight,
            codebook_weight=codebook_weight,
            variance_weight=variance_weight,
            callbacks=callbacks,
            cmdline=cmdline,
            git_commit_hash=args.commit_hash or "",
            step_offset=step_offset,
        )

    # Set trainer reference for visualization callback
    viz_callback.trainer = trainer

    # Log configuration
    print("Training configuration:")
    print(f"  Mode: {mode}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    if mode == "ctc":
        print(f"  CTC weight: {ctc_weight}")
    else:
        print(f"  Masked weight: {masked_weight}")
        print(f"  Commitment weight: {commitment_weight}")
        print(f"  Codebook weight: {codebook_weight}")
    print(f"  GRL weight: {grl_weight}")
    print(f"  GRL warmup steps: {grl_warmup_steps}")
    print(f"  GRL max alpha: {grl_max_alpha}")
    print(f"")

    # Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    final_path = os.path.join(run_dir, "final")
    trainer.save_model(final_path)

    # Save config
    config_out = {
        "mode": mode,
        "config_name": config_name,
        "model_config": model.config.__dict__,
        "num_speakers": train_dataset.num_speakers,
        "grl_weight": grl_weight,
        "grl_warmup_steps": grl_warmup_steps,
        "grl_max_alpha": grl_max_alpha,
    }

    if mode == "ctc":
        config_out["vocab_size"] = train_dataset.vocab.vocab_size
        config_out["ctc_weight"] = ctc_weight
    else:
        config_out["masked_weight"] = masked_weight
        config_out["commitment_weight"] = commitment_weight
        config_out["codebook_weight"] = codebook_weight

    with open(os.path.join(run_dir, "training_config.json"), "w") as f:
        json.dump(config_out, f, indent=2, default=str)

    print(f"\nTraining complete!")
    print(f"Mode: {mode.upper()}")
    print(f"Model saved to: {final_path}")


if __name__ == "__main__":
    main()
