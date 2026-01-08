#!/usr/bin/env python3
"""
Optimized Audio VAE Dataset Preprocessing with Sharding.

Produces sharded datasets containing:
- mel_spec: Mel spectrogram labels for VAE reconstruction [1, n_mels, T]
- mel_spec_length: Original length before padding
- speaker_embedding: ECAPA-TDNN speaker embeddings for decoder conditioning [192]

Designed for multi-GPU parallel preprocessing:
    GPU 0: python preprocess_audio_vae_dataset.py --gpu_id 0 --total_gpus 4 ...
    GPU 1: python preprocess_audio_vae_dataset.py --gpu_id 1 --total_gpus 4 ...
    etc.

Then merge with:
    python shard_utils.py merge --input_dir cached_datasets/audio_vae_raw --output_dir cached_datasets/audio_vae

Or use AudioVAEShardedDataset directly with per-GPU output directories.
"""

import os
import json
import argparse
import time
from typing import List, Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset, Audio

from dataset_loading.audio_loading import extract_mels, remove_mains_hum
from utils.audio_utils import SharedWindowBuffer


def load_speaker_encoder(device: str = "cuda"):
    """Load ECAPA-TDNN speaker encoder from SpeechBrain."""
    try:
        from speechbrain.inference.speaker import EncoderClassifier

        encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb-mel-spec",
            savedir="pretrained_models/spkrec-ecapa-voxceleb-mel-spec",
            run_opts={"device": device},
        )
        return encoder
    except ImportError:
        print("Warning: speechbrain not available. Speaker embeddings will be zeros.")
        return None


class AudioBatchProcessor:
    """Batched GPU processing for mel spectrograms and speaker embeddings."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        max_audio_seconds: int = 30,
        compute_speaker_embeddings: bool = True,
        device: str = "cuda",
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_audio_seconds = max_audio_seconds
        self.device = device

        self.audio_max_frames = (max_audio_seconds * sample_rate) // hop_length

        self.shared_window_buffer = SharedWindowBuffer()

        # Speaker encoder
        self.speaker_encoder = None
        if compute_speaker_embeddings:
            print("Loading ECAPA-TDNN speaker encoder...")
            self.speaker_encoder = load_speaker_encoder(device)
            if self.speaker_encoder is not None:
                print("  Speaker encoder loaded successfully")

    @torch.no_grad()
    def process_batch(
        self,
        waveforms: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Process batch of waveforms to mel specs and speaker embeddings.

        Args:
            waveforms: List of [T] waveform tensors

        Returns:
            Dict with:
                - mel_specs: [B, 1, n_mels, max_frames] padded mel spectrograms
                - mel_lengths: [B] original lengths before padding
                - speaker_embeddings: [B, 192] speaker embeddings
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

            # Add channel dimension: [n_mels, T] -> [1, n_mels, T]
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)

            mel_specs.append(mel)

        # Stack mel specs: [B, 1, n_mels, T]
        mel_specs = torch.stack(mel_specs)
        mel_lengths = torch.tensor(mel_lengths, dtype=torch.long)

        # Compute speaker embeddings from mel specs
        if self.speaker_encoder is not None:
            # mel_specs is [B, 1, n_mels, T], ECAPA-TDNN mel-spec model expects [B, n_mels, T]
            mel_for_ecapa = mel_specs.squeeze(1)  # [B, n_mels, T]
            speaker_embeddings = self.speaker_encoder.encode_batch(
                mel_for_ecapa.to(self.device)
            ).squeeze(1).cpu()  # [B, 192]
        else:
            # Zeros if no speaker encoder
            speaker_embeddings = torch.zeros(batch_size, 192)

        return {
            "mel_specs": mel_specs,
            "mel_lengths": mel_lengths,
            "speaker_embeddings": speaker_embeddings,
        }


def save_shard(shard_data: Dict[str, torch.Tensor], shard_path: str):
    """Save a shard to disk."""
    torch.save(shard_data, shard_path)


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio dataset for VAE training")

    # Dataset
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for shards")
    parser.add_argument("--dataset_name", type=str, default="openslr/librispeech_asr",
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, default="clean",
                        help="Dataset configuration")
    parser.add_argument("--split", type=str, default="train.360",
                        help="Dataset split")

    # Multi-GPU
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="This GPU's ID (0-indexed)")
    parser.add_argument("--total_gpus", type=int, default=1,
                        help="Total number of GPUs preprocessing in parallel")

    # Audio settings
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--max_audio_seconds", type=int, default=30,
                        help="Maximum audio length in seconds")

    # Processing
    parser.add_argument("--gpu_batch_size", type=int, default=32,
                        help="Batch size for GPU processing")
    parser.add_argument("--shard_size", type=int, default=5000,
                        help="Number of samples per shard")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")

    # Speaker embeddings
    parser.add_argument("--compute_speaker_embeddings", action="store_true", default=True)
    parser.add_argument("--no_speaker_embeddings", action="store_false",
                        dest="compute_speaker_embeddings")

    # Filtering
    parser.add_argument("--min_audio_energy", type=float, default=0.05,
                        help="Minimum audio energy (skip silent samples)")
    parser.add_argument("--min_audio_std", type=float, default=0.02,
                        help="Minimum audio std (skip near-silent samples)")
    parser.add_argument("--remove_mains_hum", action="store_true", default=True,
                        help="Remove 50/60Hz mains hum from audio")
    parser.add_argument("--no_remove_mains_hum", action="store_false",
                        dest="remove_mains_hum")

    # Limits
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to process (for testing)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting index in dataset")

    args = parser.parse_args()

    # Create output directory with GPU suffix for parallel processing
    if args.total_gpus > 1:
        output_dir = os.path.join(args.output_dir, f"gpu_{args.gpu_id}")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Audio VAE Dataset Preprocessing")
    print(f"================================")
    print(f"GPU {args.gpu_id}/{args.total_gpus}")
    print(f"Processing every {args.total_gpus}th sample starting at offset {args.gpu_id}")
    print(f"Output: {output_dir}")
    print(f"")

    # Load dataset
    print(f"Loading dataset {args.dataset_name}/{args.dataset_config} split {args.split}...")
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
        trust_remote_code=True,
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=args.sample_rate))
    print(f"  Total samples in dataset: {len(dataset):,}")

    # Calculate samples for this GPU
    samples_for_this_gpu = len([
        i for i in range(args.start_idx, len(dataset))
        if i % args.total_gpus == args.gpu_id
    ])
    print(f"  Samples for this GPU: {samples_for_this_gpu:,}")

    # Initialize processor
    processor = AudioBatchProcessor(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        max_audio_seconds=args.max_audio_seconds,
        compute_speaker_embeddings=args.compute_speaker_embeddings,
        device=device,
    )

    # Stats
    stats = {
        "processed": 0,
        "saved": 0,
        "skipped_silent": 0,
        "skipped_too_long": 0,
        "skipped_error": 0,
    }

    # Shard accumulators
    shard_mel_specs = []
    shard_mel_lengths = []
    shard_speaker_embeddings = []
    shard_idx = 0

    def flush_shard():
        nonlocal shard_mel_specs, shard_mel_lengths, shard_speaker_embeddings, shard_idx

        if not shard_mel_specs:
            return

        shard_data = {
            "mel_specs": torch.cat(shard_mel_specs, dim=0),
            "mel_lengths": torch.cat(shard_mel_lengths, dim=0),
            "speaker_embeddings": torch.cat(shard_speaker_embeddings, dim=0),
            "num_samples": sum(x.shape[0] for x in shard_mel_specs),
        }

        shard_path = os.path.join(output_dir, f"shard_{shard_idx:06d}.pt")
        torch.save(shard_data, shard_path)

        print(f"  Saved shard {shard_idx} ({shard_data['num_samples']} samples)")

        shard_mel_specs = []
        shard_mel_lengths = []
        shard_speaker_embeddings = []
        shard_idx += 1

    # Batch accumulators
    batch_waveforms = []
    batch_indices = []

    def process_and_accumulate():
        nonlocal batch_waveforms, batch_indices
        nonlocal shard_mel_specs, shard_mel_lengths, shard_speaker_embeddings

        if not batch_waveforms:
            return

        try:
            result = processor.process_batch(batch_waveforms)

            # Add to shard
            shard_mel_specs.append(result["mel_specs"])
            shard_mel_lengths.append(result["mel_lengths"])
            shard_speaker_embeddings.append(result["speaker_embeddings"])

            stats["saved"] += len(batch_waveforms)

            # Flush shard if full
            current_size = sum(x.shape[0] for x in shard_mel_specs)
            if current_size >= args.shard_size:
                flush_shard()

        except Exception as e:
            print(f"Batch processing error: {e}")
            stats["skipped_error"] += len(batch_waveforms)

        batch_waveforms = []
        batch_indices = []

    # Main processing loop
    start_time = time.time()
    pbar = tqdm(total=samples_for_this_gpu, desc=f"GPU {args.gpu_id}")

    for idx in range(args.start_idx, len(dataset)):
        # Skip if not our sample (multi-GPU distribution)
        if idx % args.total_gpus != args.gpu_id:
            continue

        # Check max samples limit
        if args.max_samples and stats["saved"] >= args.max_samples:
            break

        try:
            example = dataset[idx]

            # Extract waveform
            audio = example["audio"]
            waveform = torch.tensor(audio["array"], dtype=torch.float32)

            # Skip silent/near-silent audio
            if waveform.abs().max() < args.min_audio_energy or waveform.std() < args.min_audio_std:
                stats["skipped_silent"] += 1
                pbar.update(1)
                continue

            # Remove mains hum if enabled
            if args.remove_mains_hum:
                waveform = remove_mains_hum(waveform.unsqueeze(0), args.sample_rate).squeeze(0)

            # Check length (max_frames corresponds to ~30 sec at default settings)
            max_samples = args.max_audio_seconds * args.sample_rate
            if len(waveform) > max_samples:
                # Truncate to max length
                waveform = waveform[:max_samples]

            # Add to batch
            batch_waveforms.append(waveform)
            batch_indices.append(idx)

            # Process batch when full
            if len(batch_waveforms) >= args.gpu_batch_size:
                process_and_accumulate()

            stats["processed"] += 1
            pbar.update(1)

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            stats["skipped_error"] += 1
            pbar.update(1)
            continue

    # Final flush
    process_and_accumulate()
    flush_shard()

    pbar.close()
    elapsed = time.time() - start_time

    # Save stats and config
    stats["elapsed_seconds"] = elapsed
    stats["samples_per_second"] = stats["saved"] / elapsed if elapsed > 0 else 0
    stats["gpu_id"] = args.gpu_id
    stats["total_gpus"] = args.total_gpus
    stats["num_shards"] = shard_idx

    config = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "sample_rate": args.sample_rate,
        "n_mels": args.n_mels,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "max_audio_seconds": args.max_audio_seconds,
        "compute_speaker_embeddings": args.compute_speaker_embeddings,
        "remove_mains_hum": args.remove_mains_hum,
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
    print(f"  Skipped (error): {stats['skipped_error']:,}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Speed: {stats['saved']/elapsed:.1f} samples/sec")
    print(f"  Shards: {shard_idx}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()