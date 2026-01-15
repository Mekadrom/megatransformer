"""
Speech Reconstruction Model Training

Trains the speech reconstruction model that converts GuBERT features + speaker embeddings
back to mel spectrograms. The model has two components:
1. SpeakerEncoder: Extracts speaker identity from mel spectrograms (attention-pooled, no structural info)
2. MelReconstructor: Reconstructs mel from GuBERT features + speaker embedding

Training objectives:
- Mel reconstruction loss (L1 + MSE)
- Multi-scale mel spectrogram loss (multi-resolution STFT-like)
- Speaker embedding similarity loss (cosine similarity between GT and reconstructed)
- Optional ArcFace loss for speaker embedding separation

Usage:
    python pretrain_speech_reconstruction.py \\
        --run_name speech_recon_v1 \\
        --config small \\
        --learning_rate 1e-4 \\
        --batch_size 32 \\
        --train_cache_dir cached_datasets/speech_recon_train \\
        --val_cache_dir cached_datasets/speech_recon_val

With ArcFace:
    python pretrain_speech_reconstruction.py \\
        --run_name speech_recon_arcface_v1 \\
        --config small \\
        --use_arcface \\
        --arcface_weight 0.1 \\
        --train_cache_dir cached_datasets/speech_recon_train \\
        --val_cache_dir cached_datasets/speech_recon_val

With GE2E (for content-invariant speaker embeddings):
    python pretrain_speech_reconstruction.py \\
        --run_name speech_recon_ge2e_v1 \\
        --config small \\
        --use_ge2e true \\
        --ge2e_weight 0.1 \\
        --ge2e_n_speakers 8 \\
        --ge2e_n_utterances 4 \\
        --train_cache_dir cached_datasets/speech_recon_train \\
        --val_cache_dir cached_datasets/speech_recon_val

    Note: GE2E requires structured batching (N speakers × M utterances).
    The batch size will be ge2e_n_speakers × ge2e_n_utterances = 32.
    Ensure your dataset has enough speakers with >= ge2e_n_utterances samples each.

DeepSpeed (recommended for larger configs):
    deepspeed --num_gpus=2 pretrain_speech_reconstruction.py \\
        --use_deepspeed \\
        --bf16 \\
        --run_name speech_recon_v1 \\
        --config medium \\
        --deepspeed_config ds_config_zero-2.json
"""

import os

os.environ["DEEPSPEED_UNIT_TEST"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ['NCCL_TIMEOUT'] = '1200000'

import json
import math
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.manifold import TSNE
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers.integrations import TensorBoardCallback

from model.audio.criteria import MultiScaleMelSpectrogramLoss
from model.audio.speech_reconstruction import (
    create_speech_reconstruction_model,
    SPEAKER_ENCODER_CONFIGS,
    MEL_RECONSTRUCTOR_CONFIGS,
    GE2ELoss,
)
from model.audio.vocoders.vocoders import model_config_lookup as vocoder_config_lookup
from utils import megatransformer_utils
from utils.audio_utils import SharedWindowBuffer
from utils.model_loading_utils import load_model
from utils.speaker_encoder import (
    get_speaker_encoder,
    extract_speaker_embedding,
    SpeakerEncoderType,
    SPEAKER_EMBEDDING_DIMS,
)
from utils.training_utils import setup_int8_training


# =============================================================================
# Dataset
# =============================================================================

class SpeechReconShardedDataset(Dataset):
    """
    Dataset for loading preprocessed speech reconstruction shards.

    Loads shards containing:
    - mel_specs: [B, n_mels, T] mel spectrograms
    - mel_lengths: [B] mel spectrogram lengths
    - gubert_features: [B, T', D] GuBERT features
    - gubert_lengths: [B] GuBERT feature lengths
    - speaker_ids: [B] speaker IDs (for ArcFace)

    Features:
    - Lazy loading of shards
    - LRU cache for recently accessed shards
    """

    SHARD_INDEX_FILE = "shard_index.json"

    def __init__(
        self,
        shard_dir: str,
        cache_size: int = 8,
    ):
        self.shard_dir = shard_dir
        self.cache_size = cache_size

        # Load or build shard index
        index_path = os.path.join(shard_dir, self.SHARD_INDEX_FILE)
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            self.shard_files = index_data["shard_files"]
            self.shard_sizes = index_data["shard_sizes"]
            self.cumulative_sizes = index_data["cumulative_sizes"]
        else:
            self._build_and_save_index(shard_dir, index_path)

        self.total_samples = self.cumulative_sizes[-1] if self.cumulative_sizes else 0

        # Load config if available
        config_path = os.path.join(shard_dir, "config.json")
        self.config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)

        self.num_speakers = self.config.get("num_speakers", 0)
        self.gubert_dim = self.config.get("gubert_dim", 256)
        self.n_mels = self.config.get("n_mels", 80)

    def _build_and_save_index(self, shard_dir: str, index_path: str):
        """Build shard index by scanning all shards."""
        shard_files = sorted([
            f for f in os.listdir(shard_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        ])

        shard_sizes = []
        cumulative_sizes = []
        cumsum = 0

        for shard_file in shard_files:
            shard_path = os.path.join(shard_dir, shard_file)
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
            size = shard.get("num_samples", shard["mel_specs"].shape[0])
            shard_sizes.append(size)
            cumsum += size
            cumulative_sizes.append(cumsum)

        self.shard_files = shard_files
        self.shard_sizes = shard_sizes
        self.cumulative_sizes = cumulative_sizes

        # Save index for faster startup
        index_data = {
            "shard_files": shard_files,
            "shard_sizes": shard_sizes,
            "cumulative_sizes": cumulative_sizes,
        }
        with open(index_path, 'w') as f:
            json.dump(index_data, f)

    @lru_cache(maxsize=8)
    def _load_shard(self, shard_idx: int) -> Dict[str, torch.Tensor]:
        """Load a shard from disk with caching."""
        shard_path = os.path.join(self.shard_dir, self.shard_files[shard_idx])
        return torch.load(shard_path, map_location="cpu", weights_only=True)

    def _find_shard_and_idx(self, global_idx: int) -> tuple:
        """Find which shard contains the global index and the local index within it."""
        shard_idx = 0
        for i, cumsum in enumerate(self.cumulative_sizes):
            if global_idx < cumsum:
                shard_idx = i
                break

        local_idx = global_idx
        if shard_idx > 0:
            local_idx = global_idx - self.cumulative_sizes[shard_idx - 1]

        return shard_idx, local_idx

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shard_idx, local_idx = self._find_shard_and_idx(idx)
        shard = self._load_shard(shard_idx)

        mel_spec = shard["mel_specs"][local_idx]  # [n_mels, T]
        mel_length = shard["mel_lengths"][local_idx].item()
        gubert_features = shard["gubert_features"][local_idx]  # [T', D]
        gubert_length = shard["gubert_lengths"][local_idx].item()
        speaker_id = shard["speaker_ids"][local_idx].item()

        return {
            "mel_spec": mel_spec,
            "mel_length": mel_length,
            "gubert_features": gubert_features,
            "gubert_length": gubert_length,
            "speaker_id": speaker_id,
        }

    def get_sampler(self, shuffle: bool = True, seed: int = 42):
        """Get a shard-aware sampler for efficient training."""
        from shard_utils import ShardAwareSampler
        return ShardAwareSampler(
            shard_sizes=self.shard_sizes,
            shuffle=shuffle,
            seed=seed,
        )

    def build_speaker_to_indices(self) -> Dict[int, List[int]]:
        """
        Build mapping from speaker ID to list of sample indices.
        Required for GE2E batching.
        """
        speaker_to_indices: Dict[int, List[int]] = {}

        for global_idx in range(self.total_samples):
            shard_idx, local_idx = self._find_shard_and_idx(global_idx)
            shard = self._load_shard(shard_idx)
            speaker_id = shard["speaker_ids"][local_idx].item()

            if speaker_id not in speaker_to_indices:
                speaker_to_indices[speaker_id] = []
            speaker_to_indices[speaker_id].append(global_idx)

        return speaker_to_indices


class GE2ESampler(torch.utils.data.Sampler):
    """
    Sampler for GE2E (Generalized End-to-End) loss training.

    Creates batches with exactly N speakers × M utterances per speaker.
    This structured batching is required for GE2E loss computation.

    Each batch contains:
    - N different speakers
    - M utterances from each speaker
    - Total batch size = N × M

    The indices are ordered so that utterances from the same speaker are grouped:
    [spk0_utt0, spk0_utt1, ..., spk0_uttM-1, spk1_utt0, ..., spkN-1_uttM-1]
    """

    def __init__(
        self,
        speaker_to_indices: Dict[int, List[int]],
        n_speakers: int,
        n_utterances: int,
        num_batches_per_epoch: int = None,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
    ):
        """
        Args:
            speaker_to_indices: Mapping from speaker ID to list of sample indices
            n_speakers: Number of speakers per batch (N)
            n_utterances: Number of utterances per speaker per batch (M)
            num_batches_per_epoch: Number of batches per epoch. If None, uses all data once.
            shuffle: Whether to shuffle speakers and utterances
            seed: Random seed for reproducibility
            drop_last: Whether to drop speakers with fewer than n_utterances samples
        """
        self.speaker_to_indices = speaker_to_indices
        self.n_speakers = n_speakers
        self.n_utterances = n_utterances
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        # Filter speakers with enough utterances
        self.valid_speakers = [
            spk for spk, indices in speaker_to_indices.items()
            if len(indices) >= n_utterances
        ]

        if len(self.valid_speakers) < n_speakers:
            raise ValueError(
                f"Not enough speakers with >= {n_utterances} utterances. "
                f"Found {len(self.valid_speakers)}, need {n_speakers}. "
                f"Try reducing n_speakers or n_utterances."
            )

        # Calculate number of batches
        if num_batches_per_epoch is not None:
            self._num_batches = num_batches_per_epoch
        else:
            # Estimate based on available data
            total_utterances = sum(
                len(speaker_to_indices[spk]) for spk in self.valid_speakers
            )
            self._num_batches = total_utterances // (n_speakers * n_utterances)

        self.epoch = 0
        self._rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return self._num_batches * self.n_speakers * self.n_utterances

    def __iter__(self):
        # Set seed for this epoch
        if self.shuffle:
            self._rng = np.random.RandomState(self.seed + self.epoch)

        # Create a copy of indices for this epoch
        speaker_indices_pool = {
            spk: list(indices) for spk, indices in self.speaker_to_indices.items()
            if spk in self.valid_speakers
        }

        # Shuffle within each speaker
        if self.shuffle:
            for spk in speaker_indices_pool:
                self._rng.shuffle(speaker_indices_pool[spk])

        # Track position within each speaker's utterances
        speaker_positions = {spk: 0 for spk in self.valid_speakers}

        for _ in range(self._num_batches):
            # Find speakers with enough remaining utterances
            available_speakers = [
                spk for spk in self.valid_speakers
                if len(speaker_indices_pool[spk]) - speaker_positions[spk] >= self.n_utterances
            ]

            # Reset if not enough speakers available
            if len(available_speakers) < self.n_speakers:
                # Reset all positions
                speaker_positions = {spk: 0 for spk in self.valid_speakers}
                if self.shuffle:
                    for spk in speaker_indices_pool:
                        self._rng.shuffle(speaker_indices_pool[spk])
                available_speakers = self.valid_speakers

            # Select N speakers
            if self.shuffle:
                selected_speakers = self._rng.choice(
                    available_speakers, size=self.n_speakers, replace=False
                ).tolist()
            else:
                selected_speakers = available_speakers[:self.n_speakers]

            # Collect M utterances from each speaker
            batch_indices = []
            for spk in selected_speakers:
                start = speaker_positions[spk]
                end = start + self.n_utterances
                utterance_indices = speaker_indices_pool[spk][start:end]
                batch_indices.extend(utterance_indices)
                speaker_positions[spk] = end

            yield from batch_indices

        self.epoch += 1

    def set_epoch(self, epoch: int):
        """Set the epoch for distributed training synchronization."""
        self.epoch = epoch


class SpeechReconDataCollator:
    """Collate function for speech reconstruction batches."""

    def __init__(
        self,
        n_mels: int = 80,
        audio_max_frames: int = 1875,
        gubert_max_frames: int = 470,  # Approx audio_max_frames / 4 (GuBERT downsampling)
    ):
        self.n_mels = n_mels
        self.audio_max_frames = audio_max_frames
        self.gubert_max_frames = gubert_max_frames

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)

        # Find max lengths in batch for efficient padding
        max_mel_length = min(max(s["mel_length"] for s in batch), self.audio_max_frames)
        max_gubert_length = min(max(s["gubert_length"] for s in batch), self.gubert_max_frames)

        # Collect and pad
        mel_specs = []
        mel_lengths = []
        gubert_features = []
        gubert_lengths = []
        speaker_ids = []

        for sample in batch:
            mel = sample["mel_spec"]
            mel_len = min(sample["mel_length"], self.audio_max_frames)
            gubert = sample["gubert_features"]
            gubert_len = min(sample["gubert_length"], self.gubert_max_frames)

            # Truncate or pad mel
            if mel.shape[-1] > max_mel_length:
                mel = mel[..., :max_mel_length]
            elif mel.shape[-1] < max_mel_length:
                mel = F.pad(mel, (0, max_mel_length - mel.shape[-1]), value=0)

            # Truncate or pad GuBERT
            if gubert.shape[0] > max_gubert_length:
                gubert = gubert[:max_gubert_length]
            elif gubert.shape[0] < max_gubert_length:
                gubert = F.pad(gubert, (0, 0, 0, max_gubert_length - gubert.shape[0]), value=0)

            mel_specs.append(mel)
            mel_lengths.append(mel_len)
            gubert_features.append(gubert)
            gubert_lengths.append(gubert_len)
            speaker_ids.append(sample["speaker_id"])

        return {
            "mel_spec": torch.stack(mel_specs),  # [B, n_mels, T]
            "mel_lengths": torch.tensor(mel_lengths, dtype=torch.long),
            "gubert_features": torch.stack(gubert_features),  # [B, T', D]
            "gubert_lengths": torch.tensor(gubert_lengths, dtype=torch.long),
            "speaker_ids": torch.tensor(speaker_ids, dtype=torch.long),
        }


# =============================================================================
# Visualization Callback
# =============================================================================

def get_writer(trainer: Trainer) -> Optional[SummaryWriter]:
    """Get TensorBoard writer from trainer callbacks."""
    if hasattr(trainer, 'callback_handler'):
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback):
                if callback.tb_writer is not None:
                    return callback.tb_writer
    return None


class SpeechReconVisualizationCallback(TrainerCallback):
    """
    Callback for logging speech reconstruction visualizations during training.

    Logs:
    - Mel spectrogram reconstructions (original vs reconstructed)
    - Speaker embedding t-SNE visualization
    - Speaker embedding similarity metrics (using external speaker encoder)
    - Audio reconstructions via vocoder (if available)
    """

    def __init__(
        self,
        shared_window_buffer: SharedWindowBuffer,
        step_offset: int = 0,
        generation_steps: int = 1000,
        audio_sample_rate: int = 16000,
        audio_n_mels: int = 80,
        vocoder_checkpoint_path: Optional[str] = None,
        vocoder_config: str = "tiny_attention_freq_domain_vocoder",
        num_eval_samples: int = 8,
        num_tsne_samples: int = 256,
        speaker_encoder_type: SpeakerEncoderType = "ecapa_tdnn",
        compute_speaker_similarity: bool = True,
    ):
        self.shared_window_buffer = shared_window_buffer
        self.trainer: Optional[Trainer] = None
        self.step_offset = step_offset or 0
        self.generation_steps = generation_steps
        self.audio_sample_rate = audio_sample_rate
        self.audio_n_mels = audio_n_mels
        self.num_eval_samples = num_eval_samples
        self.num_tsne_samples = num_tsne_samples

        # Vocoder settings
        self.vocoder_checkpoint_path = vocoder_checkpoint_path
        self.vocoder_config = vocoder_config
        self.vocoder = None
        self._vocoder_load_attempted = False

        # Speaker encoder settings for similarity metrics
        self.speaker_encoder_type = speaker_encoder_type
        self.compute_speaker_similarity = compute_speaker_similarity
        self._speaker_encoder = None
        self._speaker_encoder_loaded = False

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

    def _load_speaker_encoder(self, device):
        """Lazily load speaker encoder for similarity metrics."""
        if self._speaker_encoder_loaded:
            return self._speaker_encoder
        self._speaker_encoder_loaded = True

        if not self.compute_speaker_similarity:
            return None

        try:
            print(f"Loading speaker encoder ({self.speaker_encoder_type}) for eval similarity metrics...")
            self._speaker_encoder = get_speaker_encoder(
                encoder_type=self.speaker_encoder_type,
                device=device,
            )
            print(f"Speaker encoder loaded (embedding_dim={self._speaker_encoder.embedding_dim})")
        except Exception as e:
            print(f"Failed to load speaker encoder: {e}")
            self._speaker_encoder = None

        return self._speaker_encoder

    def _get_device(self):
        """Determine the device to use for inference."""
        if torch.distributed.is_initialized():
            return torch.device(f"cuda:{torch.distributed.get_rank()}")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _visualize_mel_spec(self, mel: np.ndarray) -> np.ndarray:
        """Convert mel spectrogram to RGB image."""
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(
            mel,
            y_axis='mel',
            x_axis='time',
            sr=self.audio_sample_rate,
            hop_length=256,
            ax=ax,
            cmap='viridis',
        )
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('Mel Spectrogram')

        fig.tight_layout()
        fig.canvas.draw()

        # Convert to numpy array
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)

        # Transpose to [C, H, W] for TensorBoard
        return data.transpose(2, 0, 1)

    def _log_mel_comparison(
        self,
        writer: SummaryWriter,
        recon: np.ndarray,
        original: np.ndarray,
        global_step: int,
        tag: str,
    ):
        """Log side-by-side comparison of original and reconstructed mels."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 4))

        librosa.display.specshow(
            original,
            y_axis='mel',
            x_axis='time',
            sr=self.audio_sample_rate,
            hop_length=256,
            ax=axes[0],
            cmap='viridis',
        )
        axes[0].set_title('Original')

        librosa.display.specshow(
            recon,
            y_axis='mel',
            x_axis='time',
            sr=self.audio_sample_rate,
            hop_length=256,
            ax=axes[1],
            cmap='viridis',
        )
        axes[1].set_title('Reconstructed')

        fig.tight_layout()
        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)

        writer.add_image(tag, data.transpose(2, 0, 1), global_step)

    def _log_vocoder_audio(
        self,
        writer: SummaryWriter,
        mel_spec: torch.Tensor,
        global_step: int,
        tag: str,
    ):
        """Convert mel spectrogram to audio using vocoder and log to TensorBoard."""
        if self.vocoder is None:
            return

        try:
            if mel_spec.dim() == 2:
                mel_spec = mel_spec.unsqueeze(0)

            mel_spec = mel_spec.float()

            with torch.no_grad():
                outputs = self.vocoder(mel_spec)
                if isinstance(outputs, dict):
                    waveform = outputs["pred_waveform"]
                else:
                    waveform = outputs

            if waveform.dim() > 1:
                waveform = waveform.squeeze()

            waveform = waveform / (waveform.abs().max() + 1e-8)

            writer.add_audio(
                tag,
                waveform.numpy(),
                global_step,
                sample_rate=self.audio_sample_rate
            )
        except Exception as e:
            print(f"Failed to generate audio with vocoder: {e}")

    def _log_tsne(
        self,
        writer: SummaryWriter,
        embeddings: np.ndarray,
        speaker_ids: np.ndarray,
        global_step: int,
        tag: str,
    ):
        """Log t-SNE visualization of speaker embeddings."""
        if embeddings.shape[0] < 10:
            return

        try:
            # Run t-SNE
            tsne = TSNE(n_components=2, perplexity=min(30, embeddings.shape[0] - 1), random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings)

            # Create scatter plot
            fig, ax = plt.subplots(figsize=(10, 10))

            # Use different colors for different speakers
            unique_speakers = np.unique(speaker_ids)
            colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(unique_speakers))))

            for i, speaker_id in enumerate(unique_speakers[:20]):  # Limit to 20 for readability
                mask = speaker_ids == speaker_id
                ax.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    c=[colors[i % len(colors)]],
                    label=f"Speaker {speaker_id}",
                    alpha=0.7,
                    s=50,
                )

            ax.set_title("Speaker Embedding t-SNE")
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

            fig.tight_layout()
            fig.canvas.draw()

            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            plt.close(fig)

            writer.add_image(tag, data.transpose(2, 0, 1), global_step)
        except Exception as e:
            print(f"Failed to generate t-SNE: {e}")

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Generate and log visualizations during evaluation."""
        global_step = state.global_step + self.step_offset

        if not state.is_world_process_zero:
            return

        writer = get_writer(self.trainer)
        if writer is None:
            return

        eval_dataset = self.trainer.eval_dataset
        if eval_dataset is None or len(eval_dataset) == 0:
            return

        print(f"Generating speech reconstruction visualizations at step {global_step}...")

        self._load_vocoder()
        device = self._get_device()
        model.eval()

        num_samples = min(self.num_eval_samples, len(eval_dataset))
        indices = torch.randperm(len(eval_dataset))[:num_samples].tolist()

        # Collect data for t-SNE
        all_embeddings = []
        all_speaker_ids = []

        with torch.no_grad():
            dtype = torch.bfloat16 if bool(args.bf16) else torch.float16 if args.fp16 else torch.float32

            with autocast(device.type, dtype=dtype, enabled=args.bf16 or args.fp16):
                for i, idx in enumerate(indices):
                    sample = eval_dataset[idx]

                    mel_spec = sample["mel_spec"]
                    mel_length = sample["mel_length"]
                    gubert_features = sample["gubert_features"]
                    gubert_length = sample["gubert_length"]
                    speaker_id = sample["speaker_id"]

                    if mel_spec.dim() == 2:
                        mel_spec = mel_spec.unsqueeze(0)
                    if gubert_features.dim() == 2:
                        gubert_features = gubert_features.unsqueeze(0)

                    mel_spec = mel_spec[..., :mel_length].to(device)
                    gubert_features = gubert_features[:gubert_length].unsqueeze(0).to(device)

                    # Forward pass
                    result = model(
                        mel_spec.unsqueeze(0),
                        gubert_features,
                    )

                    mel_recon = result["mel_recon"]
                    speaker_embedding = result["speaker_embedding"]

                    # Collect embeddings for t-SNE
                    all_embeddings.append(speaker_embedding.cpu().numpy())
                    all_speaker_ids.append(speaker_id)

                    # Prepare for visualization
                    mel_original = mel_spec.squeeze(0).float().cpu().numpy()
                    mel_reconstructed = mel_recon.squeeze().float().cpu().numpy()[..., :mel_length]

                    # Log individual samples
                    self._log_mel_comparison(
                        writer, mel_reconstructed, mel_original, global_step,
                        tag=f"eval_recon/comparison/{i}"
                    )

                    # Log audio if vocoder available
                    if self.vocoder is not None:
                        self._log_vocoder_audio(
                            writer, torch.tensor(mel_original), global_step,
                            tag=f"eval_recon/original_audio/{i}"
                        )
                        self._log_vocoder_audio(
                            writer, torch.tensor(mel_reconstructed), global_step,
                            tag=f"eval_recon/reconstructed_audio/{i}"
                        )

                    # Log reconstruction metrics
                    mse = F.mse_loss(
                        torch.tensor(mel_reconstructed),
                        torch.tensor(mel_original)
                    ).item()
                    l1 = F.l1_loss(
                        torch.tensor(mel_reconstructed),
                        torch.tensor(mel_original)
                    ).item()

                    writer.add_scalar(f"eval_recon/mse/{i}", mse, global_step)
                    writer.add_scalar(f"eval_recon/l1/{i}", l1, global_step)

        # Compute speaker similarity using external speaker encoder
        all_gt_speaker_embs = []
        all_recon_speaker_embs = []
        speaker_encoder = self._load_speaker_encoder(device)

        if speaker_encoder is not None:
            print("Computing speaker similarity metrics...")
            with torch.no_grad():
                with autocast(device.type, dtype=dtype, enabled=args.bf16 or args.fp16):
                    for i, idx in enumerate(indices):
                        sample = eval_dataset[idx]

                        mel_spec = sample["mel_spec"]
                        mel_length = sample["mel_length"]
                        gubert_features = sample["gubert_features"]
                        gubert_length = sample["gubert_length"]

                        if mel_spec.dim() == 2:
                            mel_spec = mel_spec.unsqueeze(0)
                        if gubert_features.dim() == 2:
                            gubert_features = gubert_features.unsqueeze(0)

                        mel_spec_input = mel_spec[..., :mel_length].to(device)
                        gubert_features_input = gubert_features[:gubert_length].unsqueeze(0).to(device)

                        # Get reconstruction
                        result = model(mel_spec_input.unsqueeze(0), gubert_features_input)
                        mel_recon = result["mel_recon"]

                        # Extract speaker embeddings using external encoder
                        gt_speaker_emb = extract_speaker_embedding(
                            mel_spec=mel_spec_input.unsqueeze(0),
                            encoder=speaker_encoder,
                        )
                        recon_speaker_emb = extract_speaker_embedding(
                            mel_spec=mel_recon[..., :mel_length],
                            encoder=speaker_encoder,
                        )

                        all_gt_speaker_embs.append(gt_speaker_emb)
                        all_recon_speaker_embs.append(recon_speaker_emb)

                        # Per-sample cosine similarity
                        cos_sim = F.cosine_similarity(
                            recon_speaker_emb, gt_speaker_emb, dim=-1
                        ).item()
                        writer.add_scalar(f"eval_recon/speaker_cosine_sim/{i}", cos_sim, global_step)

            # Aggregate speaker similarity metrics
            if all_gt_speaker_embs:
                all_gt_speaker_embs = torch.cat(all_gt_speaker_embs, dim=0)
                all_recon_speaker_embs = torch.cat(all_recon_speaker_embs, dim=0)

                # Mean cosine similarity
                mean_cos_sim = F.cosine_similarity(
                    all_recon_speaker_embs, all_gt_speaker_embs, dim=-1
                ).mean().item()
                writer.add_scalar("eval_recon/mean_speaker_cosine_sim", mean_cos_sim, global_step)

                # L2 distance
                mean_l2_dist = (all_recon_speaker_embs - all_gt_speaker_embs).norm(dim=-1).mean().item()
                writer.add_scalar("eval_recon/mean_speaker_l2_dist", mean_l2_dist, global_step)

                print(f"  Mean speaker cosine similarity: {mean_cos_sim:.4f}")
                print(f"  Mean speaker L2 distance: {mean_l2_dist:.4f}")

        # Collect more embeddings for t-SNE if we have more data
        if len(eval_dataset) > num_samples:
            extra_indices = torch.randperm(len(eval_dataset))[:self.num_tsne_samples].tolist()

            with torch.no_grad():
                with autocast(device.type, dtype=dtype, enabled=args.bf16 or args.fp16):
                    for idx in extra_indices:
                        if idx in indices:
                            continue

                        sample = eval_dataset[idx]
                        mel_spec = sample["mel_spec"]
                        mel_length = sample["mel_length"]
                        speaker_id = sample["speaker_id"]

                        if mel_spec.dim() == 2:
                            mel_spec = mel_spec.unsqueeze(0)

                        mel_spec = mel_spec[..., :mel_length].unsqueeze(0).to(device)

                        # Get speaker embedding only
                        speaker_emb = model.encode_speaker(mel_spec)
                        all_embeddings.append(speaker_emb.cpu().numpy())
                        all_speaker_ids.append(speaker_id)

        # Log t-SNE
        if len(all_embeddings) >= 10:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
            all_speaker_ids = np.array(all_speaker_ids)
            self._log_tsne(
                writer, all_embeddings, all_speaker_ids, global_step,
                tag="eval_recon/speaker_tsne"
            )

        print(f"Visualization complete: {num_samples} samples logged")
        writer.flush()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Periodic visualization during training."""
        global_step = state.global_step + self.step_offset

        if ((global_step == 1) or (global_step % self.generation_steps == 0)) and state.is_world_process_zero:
            # Minimal logging during training steps - full visualization on eval
            pass


# =============================================================================
# Trainer
# =============================================================================

class SpeechReconstructionTrainer(Trainer):
    """
    Custom trainer for speech reconstruction model.

    Handles:
    - Mel reconstruction loss (L1 + MSE)
    - Multi-scale mel spectrogram loss (multi-resolution STFT-like)
    - Speaker embedding similarity loss (cosine similarity)
    - Optional ArcFace loss for speaker separation
    - Optional GE2E loss for content-invariant speaker embeddings
    """

    def __init__(
        self,
        *args,
        mse_weight: float = 1.0,
        l1_weight: float = 1.0,
        multi_scale_mel_weight: float = 0.0,
        speaker_similarity_weight: float = 0.0,
        speaker_encoder_type: SpeakerEncoderType = "ecapa_tdnn",
        use_arcface: bool = False,
        arcface_weight: float = 0.1,
        use_ge2e: bool = False,
        ge2e_weight: float = 0.1,
        ge2e_n_speakers: int = 8,
        ge2e_n_utterances: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.multi_scale_mel_weight = multi_scale_mel_weight
        self.speaker_similarity_weight = speaker_similarity_weight
        self.speaker_encoder_type = speaker_encoder_type
        self.use_arcface = use_arcface
        self.arcface_weight = arcface_weight
        self.use_ge2e = use_ge2e
        self.ge2e_weight = ge2e_weight
        self.ge2e_n_speakers = ge2e_n_speakers
        self.ge2e_n_utterances = ge2e_n_utterances

        # Multi-scale mel loss
        self.multi_scale_mel_loss = None
        if multi_scale_mel_weight > 0:
            self.multi_scale_mel_loss = MultiScaleMelSpectrogramLoss(
                scales=[1, 2, 4, 8],
                use_log=True,
            )

        # GE2E loss
        self.ge2e_loss_fn = None
        if use_ge2e:
            self.ge2e_loss_fn = GE2ELoss(init_w=10.0, init_b=-5.0)

        # Speaker encoder for similarity loss (lazy loaded)
        self._speaker_encoder = None
        self._speaker_encoder_loaded = False

    def _get_speaker_encoder(self, device):
        """Lazily load speaker encoder for similarity loss."""
        if not self._speaker_encoder_loaded:
            self._speaker_encoder_loaded = True
            if self.speaker_similarity_weight > 0:
                print(f"Loading speaker encoder ({self.speaker_encoder_type}) for similarity loss...")
                self._speaker_encoder = get_speaker_encoder(
                    encoder_type=self.speaker_encoder_type,
                    device=device,
                )
        return self._speaker_encoder

    def set_ge2e_sampler(self, sampler: GE2ESampler):
        """Set a custom GE2E sampler for structured batching."""
        self._ge2e_sampler = sampler

    def _get_train_sampler(self):
        """Override to use GE2E sampler if configured."""
        if hasattr(self, '_ge2e_sampler') and self._ge2e_sampler is not None:
            return self._ge2e_sampler
        return super()._get_train_sampler()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute combined loss for speech reconstruction."""
        mel_spec = inputs["mel_spec"]
        mel_lengths = inputs["mel_lengths"]
        gubert_features = inputs["gubert_features"]
        gubert_lengths = inputs["gubert_lengths"]
        speaker_ids = inputs.get("speaker_ids", None)

        device = mel_spec.device

        # Forward pass
        result = model(
            mel_spec=mel_spec,
            gubert_features=gubert_features,
            mel_lengths=mel_lengths,
            gubert_lengths=gubert_lengths,
            speaker_ids=speaker_ids if self.use_arcface else None,
        )

        mel_recon = result["mel_recon"]
        speaker_embedding = result["speaker_embedding"]

        # Compute masked reconstruction loss
        B, n_mels, T = mel_spec.shape
        T_recon = mel_recon.shape[-1]

        # Create mask for valid positions
        lengths_for_mask = mel_lengths.clamp(max=min(T, T_recon))
        mask = torch.arange(T, device=device).unsqueeze(0) < lengths_for_mask.unsqueeze(1)
        mask = mask.unsqueeze(1).expand(-1, n_mels, -1)  # [B, n_mels, T]

        # Align shapes for loss computation
        mel_recon_aligned = mel_recon
        mel_spec_aligned = mel_spec
        mask_aligned = mask

        if T_recon > T:
            mel_recon_aligned = mel_recon[..., :T]
        elif T > T_recon:
            mel_spec_aligned = mel_spec[..., :T_recon]
            mask_aligned = mask[..., :T_recon]

        # Masked L1/MSE losses
        total_valid = mask_aligned.sum().clamp(min=1)

        mse_loss = ((mel_recon_aligned - mel_spec_aligned) ** 2 * mask_aligned).sum() / total_valid
        l1_loss = ((mel_recon_aligned - mel_spec_aligned).abs() * mask_aligned).sum() / total_valid

        recon_loss = self.mse_weight * mse_loss + self.l1_weight * l1_loss

        # Total loss
        total_loss = recon_loss

        # Multi-scale mel loss
        ms_mel_loss = torch.tensor(0.0, device=device)
        if self.multi_scale_mel_loss is not None:
            # Move loss module to correct device if needed
            if next(self.multi_scale_mel_loss.parameters(), None) is None:
                self.multi_scale_mel_loss = self.multi_scale_mel_loss.to(device)
            else:
                self.multi_scale_mel_loss = self.multi_scale_mel_loss.to(device)

            ms_mel_loss = self.multi_scale_mel_loss(mel_recon_aligned, mel_spec_aligned)
            total_loss = total_loss + self.multi_scale_mel_weight * ms_mel_loss

        # Speaker embedding similarity loss
        speaker_sim_loss = torch.tensor(0.0, device=device)
        speaker_cosine_sim = torch.tensor(0.0, device=device)
        if self.speaker_similarity_weight > 0:
            speaker_encoder = self._get_speaker_encoder(device)
            if speaker_encoder is not None:
                with torch.no_grad():
                    # Extract speaker embeddings from ground truth mel
                    gt_speaker_emb = extract_speaker_embedding(
                        mel_spec=mel_spec,
                        encoder=speaker_encoder,
                        lengths=mel_lengths,
                    )  # [B, embedding_dim]

                # Extract speaker embeddings from reconstructed mel
                # Use no_grad for the speaker encoder but allow gradients for mel_recon
                recon_speaker_emb = extract_speaker_embedding(
                    mel_spec=mel_recon,
                    encoder=speaker_encoder,
                    lengths=mel_lengths,
                )  # [B, embedding_dim]

                # Cosine similarity loss (1 - cos_sim to minimize)
                # Higher cosine similarity = lower loss
                cosine_sim = F.cosine_similarity(recon_speaker_emb, gt_speaker_emb, dim=-1)
                speaker_cosine_sim = cosine_sim.mean()
                speaker_sim_loss = 1.0 - speaker_cosine_sim

                total_loss = total_loss + self.speaker_similarity_weight * speaker_sim_loss

        # ArcFace loss
        arcface_loss = torch.tensor(0.0, device=device)
        if self.use_arcface and "speaker_logits" in result and speaker_ids is not None:
            speaker_logits = result["speaker_logits"]
            arcface_loss = F.cross_entropy(speaker_logits, speaker_ids)
            total_loss = total_loss + self.arcface_weight * arcface_loss

        # GE2E loss for content-invariant speaker embeddings
        ge2e_loss = torch.tensor(0.0, device=device)
        if self.use_ge2e and self.ge2e_loss_fn is not None:
            # Move GE2E loss to correct device if needed
            if next(self.ge2e_loss_fn.parameters()).device != device:
                self.ge2e_loss_fn = self.ge2e_loss_fn.to(device)

            # GE2E requires structured batches: [N speakers × M utterances]
            expected_batch_size = self.ge2e_n_speakers * self.ge2e_n_utterances
            actual_batch_size = speaker_embedding.shape[0]

            if actual_batch_size == expected_batch_size:
                # Batch is correctly structured for GE2E
                ge2e_loss = self.ge2e_loss_fn(
                    speaker_embedding,
                    n_speakers=self.ge2e_n_speakers,
                    n_utterances=self.ge2e_n_utterances,
                )
                total_loss = total_loss + self.ge2e_weight * ge2e_loss
            # If batch size doesn't match, skip GE2E for this batch
            # This can happen during evaluation or at end of epoch

        # Log losses
        if self.state.global_step % self.args.logging_steps == 0:
            log_dict = {
                "loss/total": total_loss.item(),
                "loss/recon": recon_loss.item(),
                "loss/mse": mse_loss.item(),
                "loss/l1": l1_loss.item(),
                "loss/multi_scale_mel": ms_mel_loss.item(),
                "loss/speaker_similarity": speaker_sim_loss.item(),
                "loss/arcface": arcface_loss.item() if self.use_arcface else 0.0,
                "loss/ge2e": ge2e_loss.item() if self.use_ge2e else 0.0,
                "stats/speaker_emb_norm": speaker_embedding.norm(dim=-1).mean().item(),
            }
            if self.speaker_similarity_weight > 0:
                log_dict["stats/speaker_cosine_similarity"] = speaker_cosine_sim.item()
            self.log(log_dict)

        if return_outputs:
            return total_loss, result
        return total_loss


# =============================================================================
# Main
# =============================================================================

def main():
    args, unk = megatransformer_utils.parse_args()
    run_dir = os.path.join(args.logging_base_dir, args.run_name)

    # Parse extra arguments
    unk_dict = {}
    for i in range(0, len(unk), 2):
        if i + 1 < len(unk):
            unk_dict[unk[i].lstrip('-')] = unk[i + 1]

    # Dataset settings
    train_cache_dir = unk_dict.get("train_cache_dir", "./cached_datasets/speech_recon_train")
    val_cache_dir = unk_dict.get("val_cache_dir", "./cached_datasets/speech_recon_val")

    # Audio settings
    audio_sample_rate = args.audio_sample_rate if args.audio_sample_rate is not None else 16000
    n_mels = int(unk_dict.get("n_mels", 80))
    audio_max_frames = int(unk_dict.get("audio_max_frames", 1875))
    gubert_max_frames = int(unk_dict.get("gubert_max_frames", 470))

    # Model settings
    config = unk_dict.get("config", args.config if hasattr(args, 'config') else "small")
    gubert_dim = int(unk_dict.get("gubert_dim", 256))

    # Loss settings
    mse_weight = float(unk_dict.get("mse_weight", 1.0))
    l1_weight = float(unk_dict.get("l1_weight", 1.0))
    multi_scale_mel_weight = float(unk_dict.get("multi_scale_mel_weight", 0.5))
    speaker_similarity_weight = float(unk_dict.get("speaker_similarity_weight", 0.1))
    speaker_encoder_type = unk_dict.get("speaker_encoder_type", "ecapa_tdnn")

    # ArcFace settings
    use_arcface = unk_dict.get("use_arcface", "false").lower() == "true"
    arcface_weight = float(unk_dict.get("arcface_weight", 0.1))
    arcface_scale = float(unk_dict.get("arcface_scale", 30.0))
    arcface_margin = float(unk_dict.get("arcface_margin", 0.5))

    # GE2E settings for content-invariant speaker embeddings
    use_ge2e = unk_dict.get("use_ge2e", "false").lower() == "true"
    ge2e_weight = float(unk_dict.get("ge2e_weight", 0.1))
    ge2e_n_speakers = int(unk_dict.get("ge2e_n_speakers", 8))
    ge2e_n_utterances = int(unk_dict.get("ge2e_n_utterances", 4))

    # Vocoder settings
    vocoder_checkpoint_path = unk_dict.get("vocoder_checkpoint_path", None)
    vocoder_config = unk_dict.get("vocoder_config", "tiny_attention_freq_domain_vocoder")

    # Visualization settings
    generation_steps = int(unk_dict.get("generation_steps", 1000))
    num_eval_samples = int(unk_dict.get("num_eval_samples", 8))

    # Create shared window buffer
    shared_window_buffer = SharedWindowBuffer()

    # Load datasets
    print(f"Loading datasets...")
    print(f"  Train: {train_cache_dir}")
    print(f"  Val: {val_cache_dir}")

    train_dataset = SpeechReconShardedDataset(
        shard_dir=train_cache_dir,
        cache_size=8,
    )
    eval_dataset = SpeechReconShardedDataset(
        shard_dir=val_cache_dir,
        cache_size=8,
    )

    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples: {len(eval_dataset):,}")
    print(f"  Num speakers: {train_dataset.num_speakers}")
    print(f"  GuBERT dim: {train_dataset.gubert_dim}")

    # Update config from dataset
    gubert_dim = train_dataset.gubert_dim or gubert_dim
    num_speakers = train_dataset.num_speakers

    # Create model
    print(f"\nCreating model with config: {config}")

    if config not in SPEAKER_ENCODER_CONFIGS:
        raise ValueError(f"Unknown config: {config}. Available: {list(SPEAKER_ENCODER_CONFIGS.keys())}")

    model = create_speech_reconstruction_model(
        config=config,
        gubert_dim=gubert_dim,
        n_mels=n_mels,
        use_arcface=use_arcface,
        num_speakers=num_speakers,
        arcface_scale=arcface_scale,
        arcface_margin=arcface_margin,
    )

    # Try to load existing checkpoint
    model, model_loaded = load_model(False, model, run_dir)

    # Print model info
    if args.local_rank == 0 or not args.use_deepspeed:
        print(f"\nModel structure: {model}")
        print(f"Model parameters: {model.get_num_params():,}")
        print(f"  Speaker encoder: {model.speaker_encoder.get_num_params():,}")
        print(f"  Mel reconstructor: {model.mel_reconstructor.get_num_params():,}")
        if model.arcface_head is not None:
            arcface_params = sum(p.numel() for p in model.arcface_head.parameters())
            print(f"  ArcFace head: {arcface_params:,}")
        print(f"\nTraining settings:")
        print(f"  Config: {config}")
        print(f"  GuBERT dim: {gubert_dim}")
        print(f"  N mels: {n_mels}")
        print(f"  Audio max frames: {audio_max_frames}")
        print(f"  GuBERT max frames: {gubert_max_frames}")
        print(f"  MSE weight: {mse_weight}")
        print(f"  L1 weight: {l1_weight}")
        print(f"  Multi-scale mel weight: {multi_scale_mel_weight}")
        print(f"  Speaker similarity weight: {speaker_similarity_weight}")
        if speaker_similarity_weight > 0:
            print(f"    Speaker encoder type: {speaker_encoder_type}")
            print(f"    Embedding dim: {SPEAKER_EMBEDDING_DIMS.get(speaker_encoder_type, 'unknown')}")
        if use_arcface:
            print(f"  ArcFace: enabled (weight={arcface_weight}, scale={arcface_scale}, margin={arcface_margin})")
            print(f"  Num speakers: {num_speakers}")
        else:
            print(f"  ArcFace: disabled")
        if use_ge2e:
            print(f"  GE2E: enabled (weight={ge2e_weight}, n_speakers={ge2e_n_speakers}, n_utterances={ge2e_n_utterances})")
            print(f"    Effective batch size: {ge2e_n_speakers * ge2e_n_utterances}")
        else:
            print(f"  GE2E: disabled")
        if vocoder_checkpoint_path:
            print(f"  Vocoder: {vocoder_config} from {vocoder_checkpoint_path}")
        else:
            print(f"  Vocoder: disabled (no audio generation)")

    model = setup_int8_training(args, model)

    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        tpu_num_cores=8 if args.use_xla else None,
        output_dir=run_dir,
        overwrite_output_dir=True,
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs if args.num_train_epochs > 0 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        weight_decay=args.weight_decay,
        report_to="tensorboard",
        logging_dir=run_dir,
        logging_steps=args.logging_steps,
        save_safetensors=False,
        save_steps=args.save_steps,
        gradient_checkpointing=args.use_gradient_checkpointing,
        bf16=args.bf16,
        fp16=args.fp16,
        max_grad_norm=args.max_grad_norm,
        torch_compile=args.compile_model and not args.use_deepspeed and not args.use_xla,
        deepspeed=args.deepspeed_config if args.use_deepspeed and not args.use_xla else None,
        use_cpu=args.cpu,
        log_level=args.log_level,
        logging_first_step=True,
        local_rank=args.local_rank,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        ignore_data_skip=False,
        remove_unused_columns=False,
        eval_strategy="steps" if args.eval_steps > 0 else "no",
        eval_steps=args.eval_steps,
    )

    # Create data collator
    data_collator = SpeechReconDataCollator(
        n_mels=n_mels,
        audio_max_frames=audio_max_frames,
        gubert_max_frames=gubert_max_frames,
    )

    # Create visualization callback
    visualization_callback = SpeechReconVisualizationCallback(
        shared_window_buffer=shared_window_buffer,
        generation_steps=generation_steps,
        audio_sample_rate=audio_sample_rate,
        audio_n_mels=n_mels,
        vocoder_checkpoint_path=vocoder_checkpoint_path,
        vocoder_config=vocoder_config,
        num_eval_samples=num_eval_samples,
        speaker_encoder_type=speaker_encoder_type,
        compute_speaker_similarity=True,
    )

    # Create GE2E sampler if needed
    train_sampler = None
    if use_ge2e:
        print("\nBuilding speaker-to-indices mapping for GE2E...")
        speaker_to_indices = train_dataset.build_speaker_to_indices()
        print(f"  Found {len(speaker_to_indices)} speakers in training data")

        try:
            train_sampler = GE2ESampler(
                speaker_to_indices=speaker_to_indices,
                n_speakers=ge2e_n_speakers,
                n_utterances=ge2e_n_utterances,
                shuffle=True,
                seed=42,
            )
            print(f"  GE2E sampler created: {len(train_sampler)} samples per epoch")
            print(f"  Valid speakers (>= {ge2e_n_utterances} utterances): {len(train_sampler.valid_speakers)}")
        except ValueError as e:
            print(f"  Warning: Could not create GE2E sampler: {e}")
            print(f"  Falling back to standard sampling (GE2E loss will be skipped)")
            use_ge2e = False
            train_sampler = None

    # Create trainer
    trainer = SpeechReconstructionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[visualization_callback],
        mse_weight=mse_weight,
        l1_weight=l1_weight,
        multi_scale_mel_weight=multi_scale_mel_weight,
        speaker_similarity_weight=speaker_similarity_weight,
        speaker_encoder_type=speaker_encoder_type,
        use_arcface=use_arcface,
        arcface_weight=arcface_weight,
        use_ge2e=use_ge2e,
        ge2e_weight=ge2e_weight,
        ge2e_n_speakers=ge2e_n_speakers,
        ge2e_n_utterances=ge2e_n_utterances,
    )

    # Set GE2E sampler if using GE2E
    if train_sampler is not None:
        trainer.set_ge2e_sampler(train_sampler)

    # Store trainer reference in callback
    visualization_callback.trainer = trainer

    # Train
    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    trainer.save_model(os.path.join(run_dir, "final"))
    print(f"\nTraining complete. Model saved to {run_dir}/final")


if __name__ == "__main__":
    main()
