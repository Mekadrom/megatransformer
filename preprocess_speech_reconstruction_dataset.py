"""
Speech Reconstruction Dataset Preprocessing

Extracts GuBERT features and mel spectrograms for training the speech reconstruction model.

Produces sharded datasets containing:
- mel_spec: Mel spectrogram [n_mels, T]
- mel_length: Original mel length before padding
- gubert_features: GuBERT features [T', D]
- gubert_length: GuBERT feature length
- speaker_id: Speaker ID (optional, for ArcFace training)

Designed for multi-GPU parallel preprocessing:
    GPU 0: python preprocess_speech_reconstruction_dataset.py --gpu_id 0 --total_gpus 4 ...
    GPU 1: python preprocess_speech_reconstruction_dataset.py --gpu_id 1 --total_gpus 4 ...
    etc.

Then merge with:
    python shard_utils.py merge-speech-recon --input_dir cached_datasets/speech_recon_raw --output_dir cached_datasets/speech_recon

Usage:
    python preprocess_speech_reconstruction_dataset.py \\
        --gubert_checkpoint runs/gubert/my_run/final \\
        --gubert_config small \\
        --gubert_mode masked \\
        --output_dir cached_datasets/speech_recon_train \\
        --dataset_name openslr/librispeech_asr \\
        --split train.360
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from model.audio.feature_extractors import SharedWindowBuffer, extract_mels
from model.audio.gubert import (
    GuBERTEncoder,
    MaskedGuBERTEncoder,
    GUBERT_CONFIGS,
    MASKED_GUBERT_CONFIGS,
)


class SpeechReconBatchProcessor:
    """Batched processing for mel spectrograms and GuBERT feature extraction."""

    def __init__(
        self,
        gubert_model: torch.nn.Module,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        max_audio_seconds: int = 30,
        device: str = "cuda",
    ):
        self.gubert_model = gubert_model
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_audio_seconds = max_audio_seconds
        self.device = device

        self.audio_max_frames = (max_audio_seconds * sample_rate) // hop_length
        self.shared_window_buffer = SharedWindowBuffer()

        # Put model in eval mode
        self.gubert_model.eval()

    @torch.no_grad()
    def process_batch(
        self,
        waveforms: List[torch.Tensor],
        speaker_ids: List[int],
    ) -> Dict[str, torch.Tensor]:
        """
        Process batch of waveforms.

        Args:
            waveforms: List of [T] waveform tensors
            speaker_ids: List of speaker IDs

        Returns:
            Dict with:
                - mel_specs: [B, n_mels, max_frames] padded mel spectrograms
                - mel_lengths: [B] original lengths before padding
                - gubert_features: [B, T', D] GuBERT features
                - gubert_lengths: [B] GuBERT feature lengths
                - speaker_ids: [B] speaker IDs
        """
        batch_size = len(waveforms)

        # Process mel spectrograms
        mel_specs = []
        mel_lengths = []

        for waveform in waveforms:
            # Extract mel spectrogram
            mel = extract_mels(
                self.shared_window_buffer,
                waveform,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )

            # Store original length
            mel_length = min(mel.shape[-1], self.audio_max_frames)
            mel_lengths.append(mel_length)

            # Pad or truncate to max frames
            if mel.shape[-1] < self.audio_max_frames:
                mel = F.pad(mel, (0, self.audio_max_frames - mel.shape[-1]), value=0)
            elif mel.shape[-1] > self.audio_max_frames:
                mel = mel[..., :self.audio_max_frames]

            mel_specs.append(mel)

        # Stack mel specs: [B, n_mels, T]
        mel_specs = torch.stack(mel_specs).to(self.device)
        mel_lengths_tensor = torch.tensor(mel_lengths, dtype=torch.long, device=self.device)

        # Extract GuBERT features
        gubert_result = self.gubert_model(mel_specs, lengths=mel_lengths_tensor)
        gubert_features = gubert_result["features"]  # [B, T', D]
        gubert_lengths = gubert_result["feature_lengths"]  # [B]

        return {
            "mel_specs": mel_specs.cpu(),
            "mel_lengths": mel_lengths_tensor.cpu(),
            "gubert_features": gubert_features.cpu(),
            "gubert_lengths": gubert_lengths.cpu(),
            "speaker_ids": torch.tensor(speaker_ids, dtype=torch.long),
        }


def load_gubert_model(
    checkpoint_path: str,
    config_name: str,
    mode: str = "masked",
    device: str = "cuda",
) -> torch.nn.Module:
    """Load a pre-trained GuBERT model."""
    if mode == "ctc":
        configs = GUBERT_CONFIGS
        model_class = GuBERTEncoder
    else:
        configs = MASKED_GUBERT_CONFIGS
        model_class = MaskedGuBERTEncoder

    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")

    # Create model
    config = configs[config_name]
    model = model_class(config)

    # Load checkpoint
    if os.path.isdir(checkpoint_path):
        # HuggingFace trainer format
        model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if not os.path.exists(model_path):
            model_path = os.path.join(checkpoint_path, "model.safetensors")
    else:
        model_path = checkpoint_path

    if os.path.exists(model_path):
        if model_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(model_path)
        else:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded GuBERT weights from {model_path}")
    else:
        print(f"  Warning: No checkpoint found at {model_path}, using random initialization")

    model = model.to(device)
    model.eval()

    # Freeze model
    for param in model.parameters():
        param.requires_grad = False

    return model


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio dataset for speech reconstruction training")

    # GuBERT settings
    parser.add_argument("--gubert_checkpoint", type=str, required=True,
                        help="Path to GuBERT checkpoint")
    parser.add_argument("--gubert_config", type=str, default="small",
                        help="GuBERT config name")
    parser.add_argument("--gubert_mode", type=str, choices=["ctc", "masked"], default="masked",
                        help="GuBERT mode")

    # Dataset
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for shards")
    parser.add_argument("--dataset_name", type=str, default="openslr/librispeech_asr",
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, default="clean",
                        help="Dataset configuration")
    parser.add_argument("--split", type=str, default="train.360",
                        help="Dataset split")

    # Audio settings
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--max_audio_seconds", type=int, default=30)

    # Column names
    parser.add_argument("--audio_column", type=str, default="audio")
    parser.add_argument("--speaker_id_column", type=str, default="speaker_id")

    # Processing settings
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--shard_size", type=int, default=1000)
    parser.add_argument("--min_audio_energy", type=float, default=1e-4)
    parser.add_argument("--min_audio_std", type=float, default=1e-4)
    parser.add_argument("--min_audio_seconds", type=float, default=1.0)

    # Multi-GPU
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--total_gpus", type=int, default=1)

    args = parser.parse_args()

    # Create output directory
    if args.total_gpus > 1:
        output_dir = os.path.join(args.output_dir, f"gpu_{args.gpu_id}")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Speech Reconstruction Dataset Preprocessing")
    print(f"============================================")
    print(f"GuBERT checkpoint: {args.gubert_checkpoint}")
    print(f"GuBERT config: {args.gubert_config}")
    print(f"GuBERT mode: {args.gubert_mode}")
    print(f"GPU {args.gpu_id}/{args.total_gpus}")
    print(f"Output: {output_dir}")
    print(f"")

    # Load GuBERT model
    print("Loading GuBERT model...")
    gubert_model = load_gubert_model(
        args.gubert_checkpoint,
        args.gubert_config,
        args.gubert_mode,
        device,
    )
    gubert_dim = gubert_model.config.encoder_dim
    print(f"  GuBERT dim: {gubert_dim}")
    print(f"")

    # Load dataset
    print(f"Loading dataset {args.dataset_name}/{args.dataset_config} split {args.split}...")
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
        trust_remote_code=True,
    )
    print(f"  Total samples: {len(dataset):,}")

    # Count samples for this GPU
    samples_for_this_gpu = len([
        i for i in range(len(dataset))
        if i % args.total_gpus == args.gpu_id
    ])
    print(f"  Samples for this GPU: {samples_for_this_gpu:,}")

    # Build speaker ID mapping
    print("Building speaker ID mapping...")
    speaker_id_to_idx = {}
    for i in range(len(dataset)):
        if i % args.total_gpus != args.gpu_id:
            continue
        speaker_id = dataset[i].get(args.speaker_id_column, None)
        if speaker_id is not None and speaker_id not in speaker_id_to_idx:
            speaker_id_to_idx[speaker_id] = len(speaker_id_to_idx)

    num_speakers = len(speaker_id_to_idx)
    print(f"  Found {num_speakers} unique speakers")

    # Initialize processor
    processor = SpeechReconBatchProcessor(
        gubert_model=gubert_model,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        max_audio_seconds=args.max_audio_seconds,
        device=device,
    )

    # Stats
    stats = {
        "processed": 0,
        "saved": 0,
        "skipped_silent": 0,
        "skipped_short": 0,
        "skipped_no_speaker": 0,
        "skipped_error": 0,
    }

    # Shard accumulators
    shard_mel_specs = []
    shard_mel_lengths = []
    shard_gubert_features = []
    shard_gubert_lengths = []
    shard_speaker_ids = []
    shard_idx = 0

    def flush_shard():
        nonlocal shard_mel_specs, shard_mel_lengths, shard_gubert_features
        nonlocal shard_gubert_lengths, shard_speaker_ids, shard_idx

        if not shard_mel_specs:
            return

        # Find max GuBERT length for padding
        max_gubert_len = max(f.shape[0] for f in shard_gubert_features)

        # Pad GuBERT features
        padded_gubert = []
        for features in shard_gubert_features:
            if features.shape[0] < max_gubert_len:
                pad_len = max_gubert_len - features.shape[0]
                features = F.pad(features, (0, 0, 0, pad_len), value=0)
            padded_gubert.append(features)

        shard_data = {
            "mel_specs": torch.cat(shard_mel_specs, dim=0),
            "mel_lengths": torch.cat(shard_mel_lengths, dim=0),
            "gubert_features": torch.stack(padded_gubert, dim=0),
            "gubert_lengths": torch.cat(shard_gubert_lengths, dim=0),
            "speaker_ids": torch.cat(shard_speaker_ids, dim=0),
            "num_samples": len(shard_mel_specs),
        }

        shard_path = os.path.join(output_dir, f"shard_{shard_idx:06d}.pt")
        torch.save(shard_data, shard_path)

        print(f"  Saved shard {shard_idx} ({shard_data['num_samples']} samples)")

        shard_mel_specs = []
        shard_mel_lengths = []
        shard_gubert_features = []
        shard_gubert_lengths = []
        shard_speaker_ids = []
        shard_idx += 1

    # Batch accumulators
    batch_waveforms = []
    batch_speaker_ids = []

    def process_and_accumulate():
        nonlocal batch_waveforms, batch_speaker_ids
        nonlocal shard_mel_specs, shard_mel_lengths, shard_gubert_features
        nonlocal shard_gubert_lengths, shard_speaker_ids

        if not batch_waveforms:
            return

        try:
            result = processor.process_batch(batch_waveforms, batch_speaker_ids)

            # Add to shard (unbatch)
            for i in range(len(batch_waveforms)):
                shard_mel_specs.append(result["mel_specs"][i:i+1])
                shard_mel_lengths.append(result["mel_lengths"][i:i+1])
                shard_gubert_features.append(result["gubert_features"][i])  # [T', D]
                shard_gubert_lengths.append(result["gubert_lengths"][i:i+1])
                shard_speaker_ids.append(result["speaker_ids"][i:i+1])

            stats["saved"] += len(batch_waveforms)

            # Flush shard if full
            if len(shard_mel_specs) >= args.shard_size:
                flush_shard()

        except Exception as e:
            print(f"Batch processing error: {e}")
            stats["skipped_error"] += len(batch_waveforms)

        batch_waveforms = []
        batch_speaker_ids = []

    # Main processing loop
    start_time = time.time()
    pbar = tqdm(total=samples_for_this_gpu, desc=f"GPU {args.gpu_id}")

    min_audio_samples = int(args.min_audio_seconds * args.sample_rate)

    for idx in range(len(dataset)):
        if idx % args.total_gpus != args.gpu_id:
            continue

        stats["processed"] += 1

        try:
            example = dataset[idx]

            # Extract waveform
            audio = example[args.audio_column]
            waveform = torch.tensor(audio["array"], dtype=torch.float32)

            # Skip short audio
            if len(waveform) < min_audio_samples:
                stats["skipped_short"] += 1
                pbar.update(1)
                continue

            # Get speaker ID
            original_speaker_id = example.get(args.speaker_id_column, None)
            if original_speaker_id is None:
                stats["skipped_no_speaker"] += 1
                pbar.update(1)
                continue

            # Map to contiguous index
            speaker_id = speaker_id_to_idx[original_speaker_id]

            # Skip silent/near-silent audio
            if waveform.abs().max() < args.min_audio_energy or waveform.std() < args.min_audio_std:
                stats["skipped_silent"] += 1
                pbar.update(1)
                continue

            # Add to batch
            batch_waveforms.append(waveform)
            batch_speaker_ids.append(speaker_id)

            # Process batch if full
            if len(batch_waveforms) >= args.batch_size:
                process_and_accumulate()

        except Exception as e:
            stats["skipped_error"] += 1
            print(f"Error processing sample {idx}: {e}")

        pbar.update(1)

    pbar.close()

    # Process remaining batch
    process_and_accumulate()

    # Flush remaining shard
    flush_shard()

    elapsed = time.time() - start_time

    # Save config
    stats["gpu_id"] = args.gpu_id
    stats["total_gpus"] = args.total_gpus
    stats["num_shards"] = shard_idx

    config = {
        "gubert_checkpoint": args.gubert_checkpoint,
        "gubert_config": args.gubert_config,
        "gubert_mode": args.gubert_mode,
        "gubert_dim": gubert_dim,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "sample_rate": args.sample_rate,
        "n_mels": args.n_mels,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "max_audio_seconds": args.max_audio_seconds,
        "num_speakers": num_speakers,
        "speaker_id_to_idx": speaker_id_to_idx,
        "shard_size": args.shard_size,
        "stats": stats,
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*50}")
    print(f"GPU {args.gpu_id} complete!")
    print(f"  Processed: {stats['processed']:,}")
    print(f"  Saved: {stats['saved']:,}")
    print(f"  Skipped (silent): {stats['skipped_silent']:,}")
    print(f"  Skipped (short): {stats['skipped_short']:,}")
    print(f"  Skipped (no speaker): {stats['skipped_no_speaker']:,}")
    print(f"  Skipped (error): {stats['skipped_error']:,}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Speed: {stats['saved']/elapsed:.1f} samples/sec")
    print(f"  Shards: {shard_idx}")
    print(f"  Unique speakers: {num_speakers}")
    print(f"  GuBERT dim: {gubert_dim}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
