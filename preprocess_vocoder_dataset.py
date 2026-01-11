#!/usr/bin/env python3
"""
Vocoder Dataset Preprocessing with Sharding.

Produces sharded datasets containing:
- mel_specs: [B, n_mels, T] mel spectrograms (input)
- waveform_labels: [B, T_audio] waveform targets
- target_complex_stfts: [B, n_fft//2+1, T_stft] complex STFT targets
- speaker_ids: [B] speaker IDs
- texts: List[str] text transcriptions (for ASR evaluation)

Designed for multi-GPU parallel preprocessing:
    GPU 0: python preprocess_vocoder_dataset.py --gpu_id 0 --total_gpus 4 ...
    GPU 1: python preprocess_vocoder_dataset.py --gpu_id 1 --total_gpus 4 ...
    etc.

Then merge with:
    python shard_utils.py merge-vocoder --input_dir cached_datasets/vocoder_raw --output_dir cached_datasets/vocoder

Or use VocoderShardedDataset directly with per-GPU output directories.
"""

import os
import random
import json
import time
import argparse
from typing import List, Dict

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from tqdm import tqdm
from datasets import load_dataset, Audio

from dataset_loading.audio_loading import extract_mels, remove_mains_hum
from utils.audio_utils import SharedWindowBuffer


def apply_augmentations_fast(
    waveform: torch.Tensor,
    sample_rate: int,
    speed_perturb: bool = False,
    speed_range: tuple = (0.9, 1.1),
    gain_perturb: bool = False,
    gain_range_db: tuple = (-6, 6),
    pitch_shift: bool = False,
    pitch_range_semitones: tuple = (-2, 2),
) -> torch.Tensor:
    """
    Apply audio augmentations to waveform (optimized version).

    Uses discrete speed/pitch values to enable kernel caching, and avoids
    redundant resampling operations.

    Args:
        waveform: 1D tensor of audio samples
        sample_rate: Sample rate of audio
        speed_perturb: Whether to apply speed perturbation (changes duration)
        speed_range: (min, max) speed factor
        gain_perturb: Whether to apply random gain
        gain_range_db: (min, max) gain in dB
        pitch_shift: Whether to apply pitch shifting
        pitch_range_semitones: (min, max) pitch shift in semitones

    Returns:
        Augmented waveform
    """
    # Gain perturbation first (cheapest operation)
    if gain_perturb:
        gain_db = random.uniform(*gain_range_db)
        gain_linear = 10 ** (gain_db / 20)
        waveform = waveform * gain_linear

    # Combined speed + pitch: compute single effective ratio to minimize resampling
    effective_ratio = 1.0

    if speed_perturb:
        # Use discrete steps for better caching (5% increments)
        min_speed, max_speed = speed_range
        speed_steps = [0.90, 0.95, 1.0, 1.05, 1.10]
        speed_steps = [s for s in speed_steps if min_speed <= s <= max_speed]
        if speed_steps:
            speed_factor = random.choice(speed_steps)
            effective_ratio *= speed_factor

    if pitch_shift:
        # Use discrete semitone steps (integers only for speed)
        min_semi, max_semi = pitch_range_semitones
        semitone_steps = list(range(int(min_semi), int(max_semi) + 1))
        if semitone_steps:
            semitones = random.choice(semitone_steps)
            if semitones != 0:
                pitch_ratio = 2 ** (semitones / 12)
                effective_ratio *= pitch_ratio

    # Single resample operation if ratio changed
    if effective_ratio != 1.0:
        # Round to reduce unique ratios (helps with any internal caching)
        # Use GCD-friendly sample rates
        if effective_ratio > 1.0:
            # Speed up / pitch up: resample from higher sr to target
            source_sr = int(sample_rate * effective_ratio)
            waveform = AF.resample(waveform, source_sr, sample_rate)
        else:
            # Speed down / pitch down: resample from lower sr to target
            target_sr = int(sample_rate / effective_ratio)
            waveform = AF.resample(waveform, sample_rate, target_sr)
            waveform = AF.resample(waveform, target_sr, sample_rate)

    # Clip at the end
    if gain_perturb:
        waveform = waveform.clamp(-1.0, 1.0)

    return waveform


class VocoderBatchProcessor:
    """Batched processing for vocoder training data."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        max_audio_seconds: int = 10,
        max_waveform_samples: int = 160000,
        device: str = "cpu",
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_audio_seconds = max_audio_seconds
        self.audio_max_frames = (sample_rate * max_audio_seconds) // hop_length
        self.max_waveform_samples = max_waveform_samples  # Global max for consistent padding
        self.device = device

        self.shared_window_buffer = SharedWindowBuffer()

    @torch.no_grad()
    def process_batch(
        self,
        waveforms: List[torch.Tensor],
        speaker_ids: List[int],
        texts: List[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process batch of waveforms to mel specs, waveform labels, and STFTs.

        Args:
            waveforms: List of [T] waveform tensors
            speaker_ids: List of speaker IDs
            texts: Optional list of text transcriptions

        Returns:
            Dict with:
                - mel_specs: [B, n_mels, max_frames] mel spectrograms
                - waveform_labels: [B, max_samples] padded waveform targets
                - target_complex_stfts: [B, n_fft//2+1, max_stft_frames] complex STFT
                - speaker_ids: [B] speaker IDs
                - mel_lengths: [B] original mel lengths
                - texts: List[str] text transcriptions (if provided)
        """
        batch_size = len(waveforms)

        mel_specs = []
        mel_lengths = []
        processed_waveforms = []
        stfts = []

        # Use global max for consistent padding across batches
        max_waveform_len = self.max_waveform_samples
        max_stft_frames = (max_waveform_len // self.hop_length) + 1

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

            # Pad or truncate mel to max frames
            if mel.shape[-1] < self.audio_max_frames:
                mel = F.pad(mel, (0, self.audio_max_frames - mel.shape[-1]), value=0)
            elif mel.shape[-1] > self.audio_max_frames:
                mel = mel[..., :self.audio_max_frames]

            mel_specs.append(mel)

            # Pad waveform
            padded_waveform = F.pad(
                waveform,
                (0, max_waveform_len - waveform.shape[0]),
                value=0,
            )
            processed_waveforms.append(padded_waveform)

            # Compute STFT
            stft = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.shared_window_buffer.get_window(self.n_fft, waveform.device),
                return_complex=True,
            )

            # Pad STFT to max frames
            if stft.shape[-1] < max_stft_frames:
                stft = F.pad(stft, (0, max_stft_frames - stft.shape[-1]), value=0)
            stfts.append(stft)

        # Stack tensors
        mel_specs = torch.stack(mel_specs)  # [B, n_mels, T]
        mel_lengths = torch.tensor(mel_lengths, dtype=torch.long)
        waveform_labels = torch.stack(processed_waveforms)  # [B, T_audio]
        target_complex_stfts = torch.stack(stfts)  # [B, n_fft//2+1, T_stft]
        speaker_ids_tensor = torch.tensor(speaker_ids, dtype=torch.long)

        result = {
            "mel_specs": mel_specs,
            "waveform_labels": waveform_labels,
            "target_complex_stfts": target_complex_stfts,
            "speaker_ids": speaker_ids_tensor,
            "mel_lengths": mel_lengths,
        }

        if texts is not None:
            result["texts"] = texts

        return result


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio dataset for vocoder training")

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
    parser.add_argument("--max_audio_seconds", type=int, default=10,
                        help="Maximum audio length in seconds")

    # Segmentation
    parser.add_argument("--segment_length_sec", type=float, default=10.0,
                        help="Segment length in seconds for long audio")
    parser.add_argument("--segment_overlap_sec", type=float, default=1.0,
                        help="Overlap between segments in seconds")

    # Processing
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing")
    parser.add_argument("--shard_size", type=int, default=2000,
                        help="Number of samples per shard")

    # Augmentation
    parser.add_argument("--num_augmented_copies", type=int, default=0,
                        help="Number of augmented copies per segment (0 = no augmentation)")
    parser.add_argument("--speed_perturb", action="store_true",
                        help="Apply random speed perturbation")
    parser.add_argument("--speed_range", type=float, nargs=2, default=[0.9, 1.1],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--gain_perturb", action="store_true",
                        help="Apply random gain perturbation")
    parser.add_argument("--gain_range_db", type=float, nargs=2, default=[-6, 6],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--pitch_shift", action="store_true",
                        help="Apply random pitch shifting")
    parser.add_argument("--pitch_range_semitones", type=float, nargs=2, default=[-2, 2],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--augmentation_seed", type=int, default=42)

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

    args = parser.parse_args()

    random.seed(args.augmentation_seed)

    # Create output directory with GPU suffix for parallel processing
    if args.total_gpus > 1:
        output_dir = os.path.join(args.output_dir, f"gpu_{args.gpu_id}")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    augment_enabled = (args.speed_perturb or args.gain_perturb or args.pitch_shift) and args.num_augmented_copies > 0

    print(f"Vocoder Dataset Preprocessing")
    print(f"==============================")
    print(f"GPU {args.gpu_id}/{args.total_gpus}")
    print(f"Processing every {args.total_gpus}th sample starting at offset {args.gpu_id}")
    print(f"Output: {output_dir}")
    if augment_enabled:
        print(f"Augmentation: {args.num_augmented_copies} copies per segment")
    print()

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
        i for i in range(len(dataset))
        if i % args.total_gpus == args.gpu_id
    ])
    print(f"  Samples for this GPU: {samples_for_this_gpu:,}")

    max_samples = int(args.segment_length_sec * args.sample_rate)

    # Initialize processor with global max for consistent padding
    processor = VocoderBatchProcessor(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        max_audio_seconds=args.max_audio_seconds,
        max_waveform_samples=max_samples,
    )
    overlap_samples = int(args.segment_overlap_sec * args.sample_rate)
    stride = max_samples - overlap_samples

    # Stats
    stats = {
        "processed": 0,
        "saved": 0,
        "segments": 0,
        "skipped_silent": 0,
        "skipped_too_long": 0,
        "skipped_error": 0,
        "no_speaker_id": 0,
        "augmented": 0,
    }

    # Shard accumulators
    shard_mel_specs = []
    shard_waveform_labels = []
    shard_stfts = []
    shard_speaker_ids = []
    shard_mel_lengths = []
    shard_texts = []
    shard_idx = 0

    def flush_shard():
        nonlocal shard_mel_specs, shard_waveform_labels, shard_stfts
        nonlocal shard_speaker_ids, shard_mel_lengths, shard_texts, shard_idx

        if not shard_mel_specs:
            return

        shard_data = {
            "mel_specs": torch.cat(shard_mel_specs, dim=0),
            "waveform_labels": torch.cat(shard_waveform_labels, dim=0),
            "target_complex_stfts": torch.cat(shard_stfts, dim=0),
            "speaker_ids": torch.cat(shard_speaker_ids, dim=0),
            "mel_lengths": torch.cat(shard_mel_lengths, dim=0),
            "texts": shard_texts,  # List of strings - can't be tensors
            "num_samples": sum(x.shape[0] for x in shard_mel_specs),
        }

        shard_path = os.path.join(output_dir, f"shard_{shard_idx:06d}.pt")
        torch.save(shard_data, shard_path)

        print(f"  Saved shard {shard_idx} ({shard_data['num_samples']} samples)")

        shard_mel_specs = []
        shard_waveform_labels = []
        shard_stfts = []
        shard_speaker_ids = []
        shard_mel_lengths = []
        shard_texts = []
        shard_idx += 1

    # Batch accumulators
    batch_waveforms = []
    batch_speaker_ids = []
    batch_texts = []

    def process_and_accumulate():
        nonlocal batch_waveforms, batch_speaker_ids, batch_texts
        nonlocal shard_mel_specs, shard_waveform_labels, shard_stfts
        nonlocal shard_speaker_ids, shard_mel_lengths, shard_texts

        if not batch_waveforms:
            return

        try:
            result = processor.process_batch(batch_waveforms, batch_speaker_ids, batch_texts)

            # Add to shard
            shard_mel_specs.append(result["mel_specs"])
            shard_waveform_labels.append(result["waveform_labels"])
            shard_stfts.append(result["target_complex_stfts"])
            shard_speaker_ids.append(result["speaker_ids"])
            shard_mel_lengths.append(result["mel_lengths"])
            if "texts" in result:
                shard_texts.extend(result["texts"])

            stats["saved"] += len(batch_waveforms)

            # Flush shard if full
            current_size = sum(x.shape[0] for x in shard_mel_specs)
            if current_size >= args.shard_size:
                flush_shard()

        except Exception as e:
            print(f"Batch processing error: {e}")
            stats["skipped_error"] += len(batch_waveforms)

        batch_waveforms = []
        batch_speaker_ids = []
        batch_texts = []

    # Main processing loop
    start_time = time.time()
    pbar = tqdm(total=samples_for_this_gpu, desc=f"GPU {args.gpu_id}")

    for idx in range(len(dataset)):
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

            # Get speaker ID
            speaker_id = example.get("speaker_id", None)
            if speaker_id is None:
                stats["no_speaker_id"] += 1
                speaker_id = -1

            # Get text transcription
            text = example.get("text", "")

            # Segment long audio
            if len(waveform) > max_samples:
                segments = []
                for start in range(0, len(waveform) - max_samples + 1, stride):
                    segments.append(waveform[start:start + max_samples])
                # Include final segment if there's leftover
                if len(waveform) % stride != 0:
                    segments.append(waveform[-max_samples:])
            else:
                segments = [waveform]

            # Process segments (with optional augmentation)
            for seg_idx, segment in enumerate(segments):
                num_versions = 1 + (args.num_augmented_copies if augment_enabled else 0)

                for aug_idx in range(num_versions):
                    if aug_idx == 0:
                        current_segment = segment
                    else:
                        current_segment = apply_augmentations_fast(
                            segment.clone(),
                            args.sample_rate,
                            speed_perturb=args.speed_perturb,
                            speed_range=tuple(args.speed_range),
                            gain_perturb=args.gain_perturb,
                            gain_range_db=tuple(args.gain_range_db),
                            pitch_shift=args.pitch_shift,
                            pitch_range_semitones=tuple(args.pitch_range_semitones),
                        )
                        stats["augmented"] += 1

                    # Add to batch
                    batch_waveforms.append(current_segment)
                    batch_speaker_ids.append(speaker_id)
                    batch_texts.append(text)

                    # Process batch when full
                    if len(batch_waveforms) >= args.batch_size:
                        process_and_accumulate()

                    stats["segments"] += 1

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
        "segment_length_sec": args.segment_length_sec,
        "segment_overlap_sec": args.segment_overlap_sec,
        "remove_mains_hum": args.remove_mains_hum,
        "shard_size": args.shard_size,
        "augmentation": {
            "enabled": augment_enabled,
            "speed_perturb": args.speed_perturb,
            "gain_perturb": args.gain_perturb,
            "pitch_shift": args.pitch_shift,
            "num_copies": args.num_augmented_copies,
        } if augment_enabled else None,
        "stats": stats,
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*50}")
    print(f"GPU {args.gpu_id} complete!")
    print(f"  Processed: {stats['processed']:,}")
    print(f"  Segments: {stats['segments']:,}")
    print(f"  Saved: {stats['saved']:,}")
    if augment_enabled:
        print(f"  Augmented: {stats['augmented']:,}")
    print(f"  Skipped (silent): {stats['skipped_silent']:,}")
    print(f"  Skipped (error): {stats['skipped_error']:,}")
    print(f"  No speaker ID: {stats['no_speaker_id']:,}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Speed: {stats['saved']/elapsed:.1f} samples/sec")
    print(f"  Shards: {shard_idx}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()