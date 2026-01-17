#!/usr/bin/env python3
"""
GuBERT Pretraining Script

ASR-based training with CTC loss and GRL speaker disentanglement.
- CTC loss forces encoding of linguistic content
- GRL speaker classification removes speaker information

Visualizations logged to TensorBoard:
- t-SNE of features colored by speaker
- Feature heatmaps
- Sample transcriptions with WER/CER
- Speaker classifier accuracy over time
- GRL alpha schedule
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
)
from model.audio.vocoders.vocoders import model_config_lookup as vocoder_config_lookup
from utils import megatransformer_utils
from utils.audio_utils import SharedWindowBuffer
from utils.model_loading_utils import load_model


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
    if not hasattr(trainer, "callback_handler"):
        return None

    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, TensorBoardCallback):
            # If writer exists, return it
            if callback.tb_writer is not None:
                return callback.tb_writer
            # Try to initialize the writer if it doesn't exist yet
            # This can happen if eval runs before any training logs
            if hasattr(trainer, "args") and trainer.args.logging_dir:
                try:
                    callback.tb_writer = SummaryWriter(log_dir=trainer.args.logging_dir)
                    return callback.tb_writer
                except Exception:
                    pass
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
    Callback for logging CTC GuBERT visualizations to TensorBoard.

    Logs:
    - PCA of features colored by speaker
    - Feature heatmaps with CTC alignment
    - Sample transcriptions with CER/WER
    - Feature health metrics (global_mean, std, temporal_smoothness)
    - Audio samples via vocoder
    """

    def __init__(
        self,
        eval_dataset: GuBERTShardedDataset,
        vocab: CTCVocab,
        visualization_steps: int = 1000,
        num_tsne_samples: int = 150,
        num_transcription_samples: int = 8,
        max_speakers_for_tsne: int = 15,
        sync_with_eval: bool = False,
        # Audio settings
        audio_sample_rate: int = 16000,
        audio_n_fft: int = 1024,
        audio_hop_length: int = 256,
        # Vocoder settings
        vocoder_checkpoint_path: Optional[str] = None,
        vocoder_config: str = "tiny_attention_freq_domain_vocoder",
        num_audio_samples: int = 4,
        # LM decoder settings (beam search with optional language model)
        kenlm_model_path: Optional[str] = None,
        lm_alpha: float = 0.5,
        lm_beta: float = 1.0,
        beam_width: int = 100,
    ):
        self.eval_dataset = eval_dataset
        self.vocab = vocab
        self.visualization_steps = visualization_steps
        self.num_tsne_samples = num_tsne_samples
        self.num_transcription_samples = num_transcription_samples
        self.max_speakers_for_tsne = max_speakers_for_tsne
        self.sync_with_eval = sync_with_eval
        self.trainer: Optional[Trainer] = None

        # Audio settings
        self.audio_sample_rate = audio_sample_rate
        self.audio_n_fft = audio_n_fft
        self.audio_hop_length = audio_hop_length
        self.num_audio_samples = num_audio_samples

        # Vocoder settings
        self.vocoder_checkpoint_path = vocoder_checkpoint_path
        self.vocoder_config = vocoder_config
        self.vocoder = None
        self._vocoder_load_attempted = False

        # LM decoder settings
        self.kenlm_model_path = kenlm_model_path
        self.lm_alpha = lm_alpha
        self.lm_beta = lm_beta
        self.beam_width = beam_width
        self.ctc_decoder = None
        self._decoder_build_attempted = False

        # Shared window buffer for vocoder
        self.shared_window_buffer = SharedWindowBuffer()

    def _load_vocoder(self):
        """Lazily load vocoder on first use."""
        if self._vocoder_load_attempted:
            return
        self._vocoder_load_attempted = True

        if self.vocoder_checkpoint_path is None:
            return

        if not os.path.exists(self.vocoder_checkpoint_path):
            print(f"Vocoder checkpoint not found at {self.vocoder_checkpoint_path}")
            return

        try:
            vocoder = vocoder_config_lookup[self.vocoder_config](
                shared_window_buffer=self.shared_window_buffer,
            )

            load_model(False, vocoder, self.vocoder_checkpoint_path)

            if hasattr(vocoder.vocoder, 'remove_weight_norm'):
                vocoder.vocoder.remove_weight_norm()

            vocoder.eval()
            self.vocoder = vocoder
            print(f"Loaded vocoder from {self.vocoder_checkpoint_path}")
        except Exception as e:
            print(f"Failed to load vocoder: {e}")
            self.vocoder = None

    def _build_ctc_decoder(self):
        """Lazily build CTC decoder with optional LM on first use."""
        if self._decoder_build_attempted:
            return
        self._decoder_build_attempted = True

        # Build decoder (with or without LM)
        try:
            self.ctc_decoder = self.vocab.build_ctc_decoder(
                kenlm_model_path=self.kenlm_model_path,
                alpha=self.lm_alpha,
                beta=self.lm_beta,
            )
            if self.ctc_decoder is not None:
                if self.kenlm_model_path:
                    print(f"Built CTC decoder with LM from {self.kenlm_model_path}")
                    print(f"  alpha={self.lm_alpha}, beta={self.lm_beta}, beam_width={self.beam_width}")
                else:
                    print(f"Built CTC beam decoder (no LM), beam_width={self.beam_width}")
        except Exception as e:
            print(f"Failed to build CTC decoder: {e}")
            self.ctc_decoder = None

    def _log_vocoder_audio(self, writer: SummaryWriter, mel_spec: torch.Tensor, global_step: int, tag: str):
        """Convert mel spectrogram to audio using vocoder and log to TensorBoard."""
        if self.vocoder is None:
            return

        try:
            if mel_spec.dim() == 2:
                mel_spec = mel_spec.unsqueeze(0)

            mel_spec = mel_spec.float().cpu()
            self.vocoder.cpu()

            with torch.no_grad():
                outputs = self.vocoder(mel_spec)
                if isinstance(outputs, dict):
                    waveform = outputs["pred_waveform"]
                else:
                    waveform = outputs

            if waveform.dim() == 3:
                waveform = waveform.squeeze(1)
            elif waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

            waveform = waveform / (waveform.abs().max() + 1e-8)

            writer.add_audio(
                tag,
                waveform[0],
                global_step=global_step,
                sample_rate=self.audio_sample_rate
            )
        except Exception as e:
            print(f"Failed to generate audio with vocoder: {e}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Skip if sync_with_eval is enabled (run viz in on_evaluate instead)
        if self.sync_with_eval and not state.global_step == 1:
            return

        if not ((state.global_step == 1) or (state.global_step % self.visualization_steps == 0)) or not state.is_world_process_zero:
            return

        self._run_visualizations(model, state, kwargs)

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Run visualizations in lockstep with eval (if enabled)."""
        if not state.is_world_process_zero:
            return

        if self.sync_with_eval:
            # Model may be passed directly or in kwargs
            actual_model = model if model is not None else kwargs.get("model")
            # Fall back to trainer's model if available
            if actual_model is None and self.trainer is not None:
                actual_model = self.trainer.model
            self._run_visualizations(actual_model, state, kwargs)

    def _run_visualizations(self, model, state, kwargs):
        """Run all visualizations and log to TensorBoard."""
        if model is None:
            print("  [Visualization] Skipping: model is None")
            return

        writer = get_writer(kwargs.get("trainer", self.trainer))
        if writer is None:
            print("  [Visualization] Skipping: TensorBoard writer is None")
            return

        # Lazily load vocoder and CTC decoder
        self._load_vocoder()
        self._build_ctc_decoder()

        model.eval()
        device = next(model.parameters()).device

        try:
            print(f"  Running CTC visualizations at step {state.global_step}...")
            t0 = time.time()
            self._log_tsne(model, writer, state.global_step, device)
            print(f"    PCA/t-SNE: {time.time() - t0:.1f}s")

            t0 = time.time()
            self._log_transcriptions_with_alignment(model, writer, state.global_step, device)
            print(f"    Transcriptions + alignment: {time.time() - t0:.1f}s")

            t0 = time.time()
            self._log_feature_health(model, writer, state.global_step, device)
            print(f"    Feature health: {time.time() - t0:.1f}s")

            t0 = time.time()
            self._log_audio_samples(model, writer, state.global_step, device)
            print(f"    Audio samples: {time.time() - t0:.1f}s")

            t0 = time.time()
            self._log_eval_metrics(model, writer, state.global_step, device)
            print(f"    Eval metrics: {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"Visualization error at step {state.global_step}: {e}")
            import traceback
            traceback.print_exc()

        model.train()

    @torch.no_grad()
    def _log_tsne(self, model, writer, step, device):
        """Log PCA and t-SNE visualization of features colored by speaker."""
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

            if speaker_counts.get(speaker_id, 0) >= self.num_tsne_samples // self.max_speakers_for_tsne:
                continue

            if len(speaker_counts) >= self.max_speakers_for_tsne and speaker_id not in speaker_counts:
                continue

            mel_spec = sample["mel_spec"].unsqueeze(0).to(device)
            mel_length = sample["mel_length"].unsqueeze(0)

            result = model(mel_spec, lengths=mel_length, grl_alpha=0.0)
            feat = result["features"]

            feat_pooled = feat.mean(dim=1).cpu().numpy()

            features_list.append(feat_pooled[0])
            speakers_list.append(speaker_id)
            speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1

        if len(features_list) < 10:
            return

        features = np.array(features_list)
        speakers = np.array(speakers_list)

        # Map speakers to color indices
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
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

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

    @torch.no_grad()
    def _log_transcriptions_with_alignment(self, model, writer, step, device):
        """Log sample transcriptions with CER/WER and CTC alignment visualization."""
        transcriptions = []
        total_cer = 0
        total_wer = 0

        indices = np.random.choice(
            len(self.eval_dataset),
            min(self.num_transcription_samples, len(self.eval_dataset)),
            replace=False,
        )

        for i, idx in enumerate(indices):
            sample = self.eval_dataset[idx]

            mel_spec = sample["mel_spec"].unsqueeze(0).to(device)
            mel_length = sample["mel_length"].unsqueeze(0)
            actual_mel_len = sample["mel_length"].item()
            text_tokens = sample["text_tokens"]
            text_length = sample["text_length"].item()

            result = model(mel_spec, lengths=mel_length, grl_alpha=0.0)
            asr_logits = result["asr_logits"]  # [1, T, vocab]
            features = result["features"]  # [1, T, D]
            feature_length = result["feature_lengths"][0].item() if result["feature_lengths"] is not None else asr_logits.size(1)

            # Get probabilities
            asr_probs = F.softmax(asr_logits[0, :feature_length], dim=-1).cpu().numpy()  # [T, vocab]

            # Get ground truth
            target_text = self.vocab.decode(
                text_tokens[:text_length].tolist(),
                remove_blanks=True,
                collapse_repeats=False,
            )

            # Always compute greedy decode for base metrics (maintains consistency with previous runs)
            pred_text_greedy = self.vocab.ctc_decode_greedy(asr_logits)[0]
            wer_greedy, cer_greedy = compute_wer_cer(pred_text_greedy, target_text)
            length_ratio_greedy = len(pred_text_greedy) / max(len(target_text), 1)
            total_wer += wer_greedy
            total_cer += cer_greedy

            # Optionally compute beam+LM decode for separate metrics
            if self.ctc_decoder is not None:
                try:
                    pred_text_lm = self.vocab.ctc_decode_beam(
                        asr_logits[0, :feature_length],
                        decoder=self.ctc_decoder,
                        beam_width=self.beam_width,
                    )[0]
                    wer_lm, cer_lm = compute_wer_cer(pred_text_lm, target_text)
                    length_ratio_lm = len(pred_text_lm) / max(len(target_text), 1)
                except Exception as e:
                    print(f"Warning: Beam+LM decode failed: {e}")
                    pred_text_lm = None
                    wer_lm, cer_lm, length_ratio_lm = None, None, None
            else:
                pred_text_lm = None
                wer_lm, cer_lm, length_ratio_lm = None, None, None

            transcriptions.append({
                "target": target_text,
                "pred_greedy": pred_text_greedy,
                "pred_lm": pred_text_lm,
                "cer": cer_greedy,
                "wer": wer_greedy,
                "length_ratio": length_ratio_greedy,
                "cer_lm": cer_lm,
                "wer_lm": wer_lm,
                "length_ratio_lm": length_ratio_lm,
            })

            # Create alignment visualization (mel + CTC probs + blank prob + text comparison)
            if i < 4:  # Only visualize first 4
                fig, axes = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 2, 1.5, 0.8]})

                # 1. Mel spectrogram
                mel_np = sample["mel_spec"][:, :actual_mel_len].numpy()
                im0 = axes[0].imshow(mel_np, aspect="auto", origin="lower", cmap="viridis")
                axes[0].set_title(f"Mel Spectrogram (T={actual_mel_len})")
                axes[0].set_ylabel("Mel Bin")
                plt.colorbar(im0, ax=axes[0])

                # 2. Feature heatmap
                feat_np = features[0, :feature_length].cpu().numpy().T  # [D, T]
                im1 = axes[1].imshow(feat_np, aspect="auto", origin="lower", cmap="viridis")
                axes[1].set_title(f"GuBERT Features (T'={feature_length}, D={feat_np.shape[0]})")
                axes[1].set_ylabel("Feature Dim")
                plt.colorbar(im1, ax=axes[1])

                # 3. CTC probability heatmap (top-k characters)
                # Get top-k most likely chars at each timestep
                top_k = 10
                top_indices = np.argsort(asr_probs, axis=-1)[:, -top_k:][:, ::-1]  # [T, top_k]
                top_probs = np.take_along_axis(asr_probs, top_indices, axis=-1)  # [T, top_k]

                im2 = axes[2].imshow(top_probs.T, aspect="auto", origin="lower", cmap="hot", vmin=0, vmax=1)
                axes[2].set_title("CTC Probabilities (top-10 chars per frame)")
                axes[2].set_ylabel("Char Rank")
                axes[2].set_yticks(range(top_k))
                plt.colorbar(im2, ax=axes[2])

                # 4. Blank probability over time
                blank_probs = asr_probs[:, self.vocab.blank_idx]  # [T]
                axes[3].plot(blank_probs, color='blue', linewidth=1)
                axes[3].fill_between(range(len(blank_probs)), blank_probs, alpha=0.3)
                axes[3].set_xlim(0, feature_length)
                axes[3].set_ylim(0, 1)
                axes[3].set_ylabel("P(blank)")
                axes[3].set_xlabel("Frame")
                axes[3].set_title(f"Blank Probability (mean={np.mean(blank_probs):.3f})")
                axes[3].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

                # Add CER/WER, length ratio in upper right (greedy metrics, with LM if available)
                # Length ratio > 1 means prediction is too long, < 1 means too short
                metrics_text = f"Greedy: CER={cer_greedy:.3f} WER={wer_greedy:.3f} Len={length_ratio_greedy:.2f}x"
                if cer_lm is not None:
                    metrics_text += f"\nBeam+LM: CER={cer_lm:.3f} WER={wer_lm:.3f} Len={length_ratio_lm:.2f}x"
                fig.text(0.98, 0.98, metrics_text, fontsize=9,
                        ha='right', verticalalignment='top', fontweight='bold', family='monospace')

                # Add text comparison below figure with manual line wrapping
                def wrap_text(text, max_chars=120):
                    """Wrap text to multiple lines."""
                    lines = []
                    while len(text) > max_chars:
                        # Find last space before max_chars
                        wrap_at = text.rfind(' ', 0, max_chars)
                        if wrap_at == -1:
                            wrap_at = max_chars
                        lines.append(text[:wrap_at])
                        text = text[wrap_at:].lstrip()
                    lines.append(text)
                    return '\n'.join(lines)

                target_wrapped = wrap_text(f"Target:  {target_text}")
                greedy_wrapped = wrap_text(f"Greedy:  {pred_text_greedy}")

                # Count lines needed for proper spacing
                target_lines = target_wrapped.count('\n') + 1
                greedy_lines = greedy_wrapped.count('\n') + 1
                total_lines = target_lines + greedy_lines + 1

                # Add LM prediction if available
                if pred_text_lm is not None:
                    lm_wrapped = wrap_text(f"Beam+LM: {pred_text_lm}")
                    lm_lines = lm_wrapped.count('\n') + 1
                    total_lines += lm_lines + 1
                else:
                    lm_wrapped = None
                    lm_lines = 0

                # Calculate bottom margin based on number of lines
                line_height = 0.018
                bottom_margin = max(0.12, total_lines * line_height + 0.02)

                y_pos = bottom_margin - 0.01
                fig.text(0.02, y_pos, target_wrapped, fontsize=8, family='monospace', verticalalignment='top')
                y_pos -= (target_lines * line_height) + 0.01
                fig.text(0.02, y_pos, greedy_wrapped, fontsize=8, family='monospace', verticalalignment='top')
                if lm_wrapped is not None:
                    y_pos -= (greedy_lines * line_height) + 0.01
                    fig.text(0.02, y_pos, lm_wrapped, fontsize=8, family='monospace', verticalalignment='top', color='blue')

                # rect=[left, bottom, right, top] - leave room at top for metrics, bottom for text
                top_margin = 0.94 if cer_lm is not None else 0.96  # More room if LM metrics shown
                plt.tight_layout(rect=[0, bottom_margin, 1, top_margin])
                writer.add_figure(f"ctc_alignment/sample_{i}", fig, step)
                plt.close(fig)

        # Log greedy metrics (consistent with previous runs)
        avg_cer = total_cer / len(transcriptions) if transcriptions else 0
        avg_wer = total_wer / len(transcriptions) if transcriptions else 0
        avg_length_ratio = sum(t['length_ratio'] for t in transcriptions) / len(transcriptions) if transcriptions else 1.0
        writer.add_scalar("eval/cer", avg_cer, step)
        writer.add_scalar("eval/wer", avg_wer, step)
        writer.add_scalar("eval/length_ratio", avg_length_ratio, step)

        # Log LM metrics if available
        if self.ctc_decoder is not None:
            lm_cers = [t['cer_lm'] for t in transcriptions if t['cer_lm'] is not None]
            lm_wers = [t['wer_lm'] for t in transcriptions if t['wer_lm'] is not None]
            lm_ratios = [t['length_ratio_lm'] for t in transcriptions if t['length_ratio_lm'] is not None]
            if lm_cers:
                avg_cer_lm = sum(lm_cers) / len(lm_cers)
                avg_wer_lm = sum(lm_wers) / len(lm_wers)
                avg_length_ratio_lm = sum(lm_ratios) / len(lm_ratios)
                writer.add_scalar("eval/cer_lm", avg_cer_lm, step)
                writer.add_scalar("eval/wer_lm", avg_wer_lm, step)
                writer.add_scalar("eval/length_ratio_lm", avg_length_ratio_lm, step)

        # Log sample transcriptions as text
        text_summary = f"**Step {step} Sample Transcriptions**\n\n"
        text_summary += f"**Greedy:** CER={avg_cer:.3f} | WER={avg_wer:.3f} | Len={avg_length_ratio:.2f}x\n"
        if self.ctc_decoder is not None and lm_cers:
            text_summary += f"**Beam+LM:** CER={avg_cer_lm:.3f} | WER={avg_wer_lm:.3f} | Len={avg_length_ratio_lm:.2f}x\n"
        text_summary += "\n"
        for i, t in enumerate(transcriptions[:4]):
            text_summary += f"**Sample {i + 1}**\n"
            text_summary += f"  Target:  `{t['target'][:100]}`\n"
            text_summary += f"  Greedy:  `{t['pred_greedy'][:100]}` (CER={t['cer']:.3f})\n"
            if t['pred_lm'] is not None:
                text_summary += f"  Beam+LM: `{t['pred_lm'][:100]}` (CER={t['cer_lm']:.3f})\n"
            text_summary += "\n"

        writer.add_text("transcriptions/samples", text_summary, step)

    @torch.no_grad()
    def _log_feature_health(self, model, writer, step, device, num_samples: int = 50):
        """Log feature health metrics and visualization (global stats, temporal smoothness, etc.)."""
        all_features = []
        temporal_sims_norm = []
        temporal_sims_unnorm = []

        indices = list(range(min(len(self.eval_dataset), num_samples)))

        for idx in indices:
            sample = self.eval_dataset[idx]
            mel_spec = sample["mel_spec"].unsqueeze(0).to(device)
            mel_length = sample["mel_length"].unsqueeze(0)

            result = model(mel_spec, lengths=mel_length, grl_alpha=0.0)
            features_norm = result["features"][0].cpu()  # [T, D]
            features_unnorm = result["features_unnorm"][0].cpu()  # [T, D]

            all_features.append(features_norm)

            # Temporal smoothness
            if features_norm.shape[0] > 1:
                feat_n = F.normalize(features_unnorm, dim=-1)
                sim = (feat_n[:-1] * feat_n[1:]).sum(dim=-1)
                temporal_sims_unnorm.append(sim.mean().item())

                feat_n = F.normalize(features_norm, dim=-1)
                sim = (feat_n[:-1] * feat_n[1:]).sum(dim=-1)
                temporal_sims_norm.append(sim.mean().item())

        all_features = torch.cat(all_features, dim=0).numpy()
        feat_dim = all_features.shape[1]

        # Global statistics
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

        # Effective dimensionality (via explained variance ratio using entropy)
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
            writer.add_scalar("feature_health/temporal_smoothness", mean_smoothness_unnorm, step)
        if temporal_sims_norm:
            mean_smoothness_norm = np.mean(temporal_sims_norm)
            writer.add_scalar("feature_health/temporal_smoothness_norm", mean_smoothness_norm, step)

        # Debug print to console
        if temporal_sims_unnorm and temporal_sims_norm:
            print(f"      [Smoothness] pre-norm={mean_smoothness_unnorm:.4f}, post-norm={mean_smoothness_norm:.4f}")

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
        print(f"      Feature health: mean={global_mean:.3f}, std={global_std:.3f}, "
              f"norm={np.mean(feature_norms):.3f}, dead={dead_dims}/{feat_dim}, "
              f"effective_dim={effective_dim:.1f}")

    @torch.no_grad()
    def _log_audio_samples(self, model, writer, step, device):
        """Log audio samples with vocoder output aligned with transcription."""
        if self.vocoder is None:
            return

        num_samples = min(self.num_audio_samples, len(self.eval_dataset))

        for i in range(num_samples):
            sample = self.eval_dataset[i]

            mel_spec = sample["mel_spec"]  # [n_mels, T]
            mel_length = sample["mel_length"].item()
            text_tokens = sample["text_tokens"]
            text_length = sample["text_length"].item()

            # Crop to actual length
            mel_cropped = mel_spec[:, :mel_length]

            # Get ground truth text
            target_text = self.vocab.decode(
                text_tokens[:text_length].tolist(),
                remove_blanks=True,
                collapse_repeats=False,
            )

            # Generate audio from mel spectrogram
            self._log_vocoder_audio(
                writer, mel_cropped, step,
                tag=f"audio_samples/sample_{i}"
            )

            # Log the transcription for this sample
            writer.add_text(
                f"audio_samples/sample_{i}_text",
                f"**Sample {i}**: {target_text}",
                step
            )

    @torch.no_grad()
    def _log_eval_metrics(self, model, writer, step, device, num_samples: int = 100):
        """Compute and log speaker accuracy (GRL effectiveness) over eval samples."""
        speaker_accs = []

        # Use sequential indices for shard locality
        indices = list(range(min(len(self.eval_dataset), num_samples)))

        for idx in tqdm(indices, desc="Eval metrics", leave=False):
            sample = self.eval_dataset[idx]

            mel_spec = sample["mel_spec"].unsqueeze(0).to(device)
            mel_length = sample["mel_length"].unsqueeze(0)
            speaker_id = sample["speaker_id"].to(device)

            result = model(mel_spec, lengths=mel_length, grl_alpha=0.0)

            # Compute speaker accuracy
            speaker_logits = result["speaker_logits"]
            speaker_pred = speaker_logits.argmax(dim=-1)
            speaker_acc = (speaker_pred == speaker_id).float().item()
            speaker_accs.append(speaker_acc)

        # Log aggregated metrics
        if speaker_accs:
            mean_speaker_acc = np.mean(speaker_accs)
            writer.add_scalar("eval/speaker_accuracy", mean_speaker_acc, step)
            print(f"    Eval speaker accuracy: {mean_speaker_acc:.4f} (lower is better for GRL)")



# ============================================================================
# Custom Trainer
# ============================================================================

class GuBERTTrainer(Trainer):
    """
    Custom trainer for GuBERT with CTC + GRL losses.

    Supports:
    - Separate optimizer/LR for speaker classifier (grl_lr)
    - GRL pre-training phase (grl_start_step) where classifier learns without adversarial pressure
    """

    def __init__(
        self,
        *args,
        vocab: CTCVocab,
        grl_alpha_scheduler: GRLAlphaScheduler,
        ctc_weight: float = 1.0,
        grl_weight: float = 0.1,
        grl_start_step: int = 0,  # Step at which GRL kicks in (before this, classifier trains freely)
        grl_lr: float = None,  # Separate LR for speaker classifier (None = use base LR)
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
        self.grl_start_step = grl_start_step
        self.grl_lr = grl_lr
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

    def create_optimizer(self):
        """
        Override to create separate parameter groups with different learning rates.
        Speaker classifier gets grl_lr (or base_lr if None).
        """
        if self.optimizer is not None:
            return self.optimizer

        model = self.model
        base_lr = self.args.learning_rate
        speaker_lr = self.grl_lr if self.grl_lr is not None else base_lr

        # Separate speaker classifier parameters
        speaker_classifier_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'speaker_classifier' in name:
                speaker_classifier_params.append(param)
            else:
                other_params.append(param)

        # Create parameter groups
        optimizer_grouped_parameters = [
            {
                "params": other_params,
                "lr": base_lr,
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": speaker_classifier_params,
                "lr": speaker_lr,
                "weight_decay": self.args.weight_decay,
            },
        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, model)

        # Remove lr from kwargs since we set it per-group
        optimizer_kwargs.pop("lr", None)

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

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

        # GRL pre-training phase:
        # Before grl_start_step, classifier trains freely (no gradient reversal)
        # After grl_start_step, GRL kicks in with alpha ramping from that point
        in_pretraining = global_step < self.grl_start_step
        if in_pretraining:
            # Pre-training phase: classifier learns without adversarial pressure
            grl_alpha = 0.0
        else:
            # GRL active: alpha scheduler starts from grl_start_step
            effective_step = global_step - self.grl_start_step
            grl_alpha = self.grl_alpha_scheduler.get_alpha(effective_step)

        # Forward pass
        result = model(mel_specs, lengths=mel_lengths, grl_alpha=grl_alpha)

        asr_logits = result["asr_logits"]  # [B, T, vocab]
        speaker_logits = result["speaker_logits"]  # [B, num_speakers]
        # Use ctc_lengths for CTC loss (accounts for upsampling if enabled)
        ctc_lengths = result.get("ctc_lengths", result["feature_lengths"])  # [B]

        # CTC loss
        # CTC expects [T, B, vocab] and log probabilities
        log_probs = F.log_softmax(asr_logits, dim=-1).permute(1, 0, 2)  # [T, B, vocab]
        ctc_loss = self.ctc_criterion(log_probs, text_tokens, ctc_lengths, text_lengths)

        # GRL speaker classification loss
        # We want the classifier to FAIL (be at chance level)
        # But we train it normally - GRL reverses gradients to encoder
        speaker_loss = self.speaker_criterion(speaker_logits, speaker_ids)

        # Speaker accuracy and diagnostics (for logging)
        with torch.no_grad():
            speaker_preds = speaker_logits.argmax(dim=-1)
            speaker_acc = (speaker_preds == speaker_ids).float().mean().item()

            # Diagnostic: check for mode collapse
            pred_probs = F.softmax(speaker_logits, dim=-1)
            pred_entropy = -(pred_probs * torch.log(pred_probs + 1e-8)).sum(dim=-1).mean().item()
            unique_preds = speaker_preds.unique().numel()

            # Max probability (confidence) - high values with low accuracy = overconfident
            max_prob = pred_probs.max(dim=-1).values.mean().item()

        # Combined loss
        # During pre-training phase, speaker loss still contributes but doesn't affect encoder
        # (because grl_alpha=0 means no gradient reversal, but classifier still learns)
        total_loss = self.ctc_weight * ctc_loss + self.grl_weight * speaker_loss

        # Log to TensorBoard
        if self.writer is not None and global_step % self.args.logging_steps == 0:
            self.writer.add_scalar("train/ctc_loss", ctc_loss.item(), global_step)
            self.writer.add_scalar("train/speaker_loss", speaker_loss.item(), global_step)
            self.writer.add_scalar("train/speaker_accuracy", speaker_acc, global_step)
            self.writer.add_scalar("train/grl_alpha", grl_alpha, global_step)
            self.writer.add_scalar("train/total_loss", total_loss.item(), global_step)
            self.writer.add_scalar("train/grl_pretraining", float(in_pretraining), global_step)
            # Diagnostics for speaker classifier behavior
            self.writer.add_scalar("train/speaker_pred_entropy", pred_entropy, global_step)
            self.writer.add_scalar("train/speaker_unique_preds", unique_preds, global_step)
            self.writer.add_scalar("train/speaker_max_prob", max_prob, global_step)

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
            # Use ctc_lengths for CTC loss (accounts for upsampling if enabled)
            ctc_lengths = result.get("ctc_lengths", result["feature_lengths"])

            # CTC loss
            log_probs = F.log_softmax(asr_logits, dim=-1).permute(1, 0, 2)
            ctc_loss = self.ctc_criterion(log_probs, text_tokens, ctc_lengths, text_lengths)

            # Speaker loss
            speaker_loss = self.speaker_criterion(speaker_logits, speaker_ids)

            # Combined loss
            total_loss = self.ctc_weight * ctc_loss + self.grl_weight * speaker_loss

        return (total_loss, None, None)



# ============================================================================
# Model Configuration
# ============================================================================

def create_model(
    config_name: str,
    num_speakers: int,
    vocab_size: int = 30,
    **overrides,
):
    """Create GuBERT model from config name with overrides."""
    if config_name not in GUBERT_CONFIGS:
        config_name = "small"
    return GuBERTEncoder.from_config(
        config_name,
        num_speakers=num_speakers,
        vocab_size=vocab_size,
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

    # Dataset settings
    train_cache_dir = unk_dict.get("train_cache_dir", "./cached_datasets/gubert_ctc_train")
    val_cache_dir = unk_dict.get("val_cache_dir", "./cached_datasets/gubert_ctc_val")

    # Model settings
    config_name = args.config if args.config in GUBERT_CONFIGS else "small"

    # Audio settings
    n_mels = int(unk_dict.get("n_mels", 80))
    max_mel_frames = int(unk_dict.get("max_mel_frames", 1875))

    # GRL settings
    grl_warmup_steps = int(unk_dict.get("grl_warmup_steps", 5000))
    grl_max_alpha = float(unk_dict.get("grl_max_alpha", 1.0))
    grl_weight = float(unk_dict.get("grl_weight", 0.1))
    grl_start_step = int(unk_dict.get("grl_start_step", 0))  # Pre-training phase before GRL kicks in
    grl_lr_str = unk_dict.get("grl_lr", None)  # Separate LR for speaker classifier
    grl_lr = float(grl_lr_str) if grl_lr_str is not None else None
    speaker_pooling = unk_dict.get("speaker_pooling", "attentive_statistics")  # Pooling strategy for speaker classifier

    # CTC-specific settings
    ctc_weight = float(unk_dict.get("ctc_weight", 1.0))

    # Dropout settings for regularization (helps prevent memorization)
    conv_dropout = float(unk_dict.get("conv_dropout", 0.05))  # Dropout1d in conv frontend
    feature_dropout = float(unk_dict.get("feature_dropout", 0.0))
    head_dropout = float(unk_dict.get("head_dropout", 0.0))
    attention_head_drop = float(unk_dict.get("attention_head_drop", 0.0))  # DropHead on attention

    # Architectural options
    use_rotary_embedding = unk_dict.get("use_rotary_embedding", "false").lower() in ("true", "1", "yes")
    use_conformer_conv = unk_dict.get("use_conformer_conv", "false").lower() in ("true", "1", "yes")
    conformer_kernel_size = int(unk_dict.get("conformer_kernel_size", 31))
    use_macaron = unk_dict.get("use_macaron", "false").lower() in ("true", "1", "yes")
    activation = unk_dict.get("activation", "gelu")  # "gelu" or "swiglu"

    # Speaker normalization (strips speaker-specific statistics)
    # Instance norm normalizes each sample across time, removing per-utterance mean/variance
    # This is more direct than GRL for speaker removal
    use_instance_norm = unk_dict.get("use_instance_norm", "false").lower() in ("true", "1", "yes")
    instance_norm_affine = unk_dict.get("instance_norm_affine", "false").lower() in ("true", "1", "yes")

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

    # CTC upsampling (relaxes CTC length constraint without increasing transformer cost)
    ctc_upsample_factor = int(unk_dict.get("ctc_upsample_factor", 1))

    # Vocoder settings (for audio generation in TensorBoard)
    vocoder_checkpoint_path = unk_dict.get("vocoder_checkpoint_path", None)
    vocoder_config = unk_dict.get("vocoder_config", "tiny_attention_freq_domain_vocoder")
    audio_sample_rate = int(unk_dict.get("audio_sample_rate", 16000))
    audio_n_fft = int(unk_dict.get("audio_n_fft", 1024))
    audio_hop_length = int(unk_dict.get("audio_hop_length", 256))
    num_audio_samples = int(unk_dict.get("num_audio_samples", 4))

    # LM decoder settings (for CTC mode - beam search with optional language model)
    kenlm_model_path = unk_dict.get("kenlm_model_path", "./pretrained_models/KenLM-4-gram/4-gram.arpa")
    lm_alpha = float(unk_dict.get("lm_alpha", 0.5))  # LM weight
    lm_beta = float(unk_dict.get("lm_beta", 1.0))    # Word insertion bonus
    beam_width = int(unk_dict.get("beam_width", 100))

    # Max samples (for testing)
    max_train_samples = int(unk_dict.get("max_train_samples", 0)) or None
    max_val_samples = int(unk_dict.get("max_val_samples", 0)) or None

    print(f"GuBERT Pretraining")
    print(f"==================")
    print(f"Config: {config_name}")
    print(f"Run dir: {run_dir}")
    print(f"Train cache: {train_cache_dir}")
    print(f"Val cache: {val_cache_dir}")
    if ctc_upsample_factor > 1:
        print(f"CTC upsampling: ENABLED")
        print(f"  ctc_upsample_factor: {ctc_upsample_factor} ({ctc_upsample_factor}x more CTC frames)")
    if conv_dropout > 0 or feature_dropout > 0 or head_dropout > 0 or attention_head_drop > 0:
        print(f"Dropout regularization: ENABLED")
        print(f"  conv_dropout: {conv_dropout} (Dropout1d in conv frontend)")
        print(f"  feature_dropout: {feature_dropout}")
        print(f"  head_dropout: {head_dropout} (prediction head)")
        print(f"  attention_head_drop: {attention_head_drop} (DropHead on attention)")
    if use_rotary_embedding or use_conformer_conv or activation != "gelu":
        print(f"Architectural options:")
        if use_rotary_embedding:
            print(f"  RoPE: ENABLED (rotary position embeddings)")
        if use_conformer_conv:
            print(f"  Conformer conv: ENABLED (kernel_size={conformer_kernel_size})")
        if activation != "gelu":
            print(f"  Activation: {activation}")
    if use_augmentation:
        print(f"Data augmentation: ENABLED")
        print(f"  noise: prob={aug_noise_prob}, std=[{aug_noise_std_min}, {aug_noise_std_max}]")
        print(f"  gain: prob={aug_gain_prob}, range=[{aug_gain_min}, {aug_gain_max}]")
        print(f"  freq_shift: prob={aug_freq_shift_prob}, max_bins={aug_freq_shift_max}")
        print(f"  time_warp: prob={aug_time_warp_prob}, range=[{aug_time_warp_min}, {aug_time_warp_max}]")
    if vocoder_checkpoint_path:
        print(f"Vocoder (for audio visualization): {vocoder_config}")
        print(f"  checkpoint: {vocoder_checkpoint_path}")
        print(f"  sample_rate: {audio_sample_rate}, n_fft: {audio_n_fft}, hop_length: {audio_hop_length}")
        print(f"  num_audio_samples: {num_audio_samples}")
    print(f"CTC decoding: beam_width={beam_width}")
    if kenlm_model_path:
        print(f"  LM: {kenlm_model_path}")
        print(f"  alpha={lm_alpha}, beta={lm_beta}")
    else:
        print(f"  No language model (greedy fallback or beam search without LM)")
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
    train_dataset = GuBERTShardedDataset(train_cache_dir, max_samples=max_train_samples, mode="ctc", augmentation=augmentation)
    val_dataset = GuBERTShardedDataset(val_cache_dir, max_samples=max_val_samples, mode="ctc")

    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples: {len(val_dataset):,}")
    print(f"  Num speakers: {train_dataset.num_speakers}")
    print(f"  Vocab size: {train_dataset.vocab.vocab_size}")
    print(f"")

    # Create model
    print(f"Creating model ({config_name})...")
    model = create_model(
        config_name=config_name,
        num_speakers=train_dataset.num_speakers,
        vocab_size=train_dataset.vocab.vocab_size,
        n_mels=n_mels,
        # CTC upsampling (relaxes CTC length constraint)
        ctc_upsample_factor=ctc_upsample_factor,
        # Dropout regularization
        conv_dropout=conv_dropout,
        feature_dropout=feature_dropout,
        head_dropout=head_dropout,
        attention_head_drop=attention_head_drop,
        # Architectural options
        use_rotary_embedding=use_rotary_embedding,
        use_conformer_conv=use_conformer_conv,
        conformer_kernel_size=conformer_kernel_size,
        use_macaron=use_macaron,
        activation=activation,
        # Speaker normalization (strips speaker statistics)
        use_instance_norm=use_instance_norm,
        instance_norm_affine=instance_norm_affine,
        # Speaker classifier pooling strategy
        speaker_pooling=speaker_pooling,
    )

    num_params = model.get_num_params()
    print(f"Model: {model}")
    print(f"Total Parameters: {num_params:,}")

    conv_upsample_params = sum(p.numel() for p in model.conv_subsample.parameters())
    encoder_blocks_params = sum(p.numel() for p in model.encoder_blocks.parameters())
    final_norm_params = sum(p.numel() for p in model.final_norm.parameters())
    head_params = sum(p.numel() for p in model.asr_head.parameters())
    print(f"GuBERT Parameters: {conv_upsample_params + encoder_blocks_params + final_norm_params + head_params:,}")
    print(f"GRL Parameters: {sum(p.numel() for p in model.speaker_classifier.parameters()):,}")

    # Create data collator
    collator = GuBERTDataCollator(n_mels=n_mels, max_mel_frames=max_mel_frames, mode="ctc")

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
        report_to="tensorboard",
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

    # Visualization callback
    viz_callback = GuBERTVisualizationCallback(
        eval_dataset=val_dataset,
        vocab=train_dataset.vocab,
        visualization_steps=visualization_steps,
        sync_with_eval=sync_viz_with_eval,
        audio_sample_rate=audio_sample_rate,
        audio_n_fft=audio_n_fft,
        audio_hop_length=audio_hop_length,
        vocoder_checkpoint_path=vocoder_checkpoint_path,
        vocoder_config=vocoder_config,
        num_audio_samples=num_audio_samples,
        # LM decoder settings
        kenlm_model_path=kenlm_model_path,
        lm_alpha=lm_alpha,
        lm_beta=lm_beta,
        beam_width=beam_width,
    )
    callbacks.append(viz_callback)

    # Build command line string for logging
    import sys
    cmdline = " ".join(sys.argv)

    # Create trainer
    step_offset = args.start_step or 0
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
        grl_start_step=grl_start_step,
        grl_lr=grl_lr,
        callbacks=callbacks,
        cmdline=cmdline,
        git_commit_hash=args.commit_hash or "",
        step_offset=step_offset,
    )

    # Set trainer reference for visualization callback
    viz_callback.trainer = trainer

    # Log configuration
    print("Training configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  CTC weight: {ctc_weight}")
    print(f"  GRL weight: {grl_weight}")
    print(f"  GRL warmup steps: {grl_warmup_steps}")
    print(f"  GRL max alpha: {grl_max_alpha}")
    print(f"  GRL start step: {grl_start_step}" + (" (pre-training phase)" if grl_start_step > 0 else ""))
    print(f"  GRL LR: {grl_lr if grl_lr is not None else 'same as base LR'}")
    print(f"  Speaker pooling: {speaker_pooling}")
    print(f"")

    # Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    final_path = os.path.join(run_dir, "final")
    trainer.save_model(final_path)

    # Save config
    config_out = {
        "config_name": config_name,
        "model_config": model.config.__dict__,
        "num_speakers": train_dataset.num_speakers,
        "vocab_size": train_dataset.vocab.vocab_size,
        "ctc_weight": ctc_weight,
        "grl_weight": grl_weight,
        "grl_warmup_steps": grl_warmup_steps,
        "grl_max_alpha": grl_max_alpha,
        "grl_start_step": grl_start_step,
        "grl_lr": grl_lr,
        "speaker_pooling": speaker_pooling,
    }

    with open(os.path.join(run_dir, "training_config.json"), "w") as f:
        json.dump(config_out, f, indent=2, default=str)

    print(f"\nTraining complete!")
    print(f"Model saved to: {final_path}")


if __name__ == "__main__":
    main()
