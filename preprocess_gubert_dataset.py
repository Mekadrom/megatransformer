#!/usr/bin/env python3
"""
GuBERT Dataset Preprocessing

Supports two modes:
1. CTC mode (--mode ctc): Requires transcriptions, produces text tokens for ASR training
2. Masked mode (--mode masked): Self-supervised, no transcriptions needed

Produces sharded datasets containing:
- mel_spec: Mel spectrogram inputs [n_mels, T]
- mel_spec_length: Original length before padding
- text_tokens: CTC target token indices [L] (CTC mode only)
- text_length: Original text length (CTC mode only)
- speaker_id: Speaker ID for GRL training

Designed for multi-GPU parallel preprocessing:
    GPU 0: python preprocess_gubert_dataset.py --gpu_id 0 --total_gpus 4 --mode masked ...
    GPU 1: python preprocess_gubert_dataset.py --gpu_id 1 --total_gpus 4 --mode masked ...
    etc.

Then merge with:
    python shard_utils.py merge-gubert --input_dir cached_datasets/gubert_raw --output_dir cached_datasets/gubert
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
from model.audio.gubert import CTCVocab


class GuBERTBatchProcessor:
    """Batched processing for mel spectrograms and optional text tokenization."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        max_audio_seconds: int = 30,
        device: str = "cuda",
        mode: str = "ctc",  # "ctc" or "masked"
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_audio_seconds = max_audio_seconds
        self.device = device
        self.mode = mode

        self.audio_max_frames = (max_audio_seconds * sample_rate) // hop_length
        self.shared_window_buffer = SharedWindowBuffer()

        # CTC vocabulary for text tokenization (only needed in ctc mode)
        self.vocab = CTCVocab() if mode == "ctc" else None

    @torch.no_grad()
    def process_batch(
        self,
        waveforms: List[torch.Tensor],
        texts: List[str],
        speaker_ids: List[int],
    ) -> Dict[str, torch.Tensor]:
        """
        Process batch of waveforms and optionally texts.

        Args:
            waveforms: List of [T] waveform tensors
            texts: List of text transcriptions (can be None/empty in masked mode)
            speaker_ids: List of speaker IDs

        Returns:
            Dict with:
                - mel_specs: [B, n_mels, max_frames] padded mel spectrograms
                - mel_lengths: [B] original lengths before padding
                - text_tokens: [B, max_text_len] padded token indices (CTC mode only)
                - text_lengths: [B] original text lengths (CTC mode only)
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
        mel_specs = torch.stack(mel_specs)
        mel_lengths = torch.tensor(mel_lengths, dtype=torch.long)
        speaker_ids_tensor = torch.tensor(speaker_ids, dtype=torch.long)

        result = {
            "mel_specs": mel_specs,
            "mel_lengths": mel_lengths,
            "speaker_ids": speaker_ids_tensor,
        }

        # Tokenize texts only in CTC mode
        if self.mode == "ctc" and texts is not None:
            text_tokens_list = []
            text_lengths = []
            max_text_len = 0

            for text in texts:
                tokens = self.vocab.encode(text)
                text_tokens_list.append(tokens)
                text_lengths.append(len(tokens))
                max_text_len = max(max_text_len, len(tokens))

            # Pad text tokens
            text_tokens = torch.zeros(batch_size, max_text_len, dtype=torch.long)
            for i, tokens in enumerate(text_tokens_list):
                text_tokens[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)

            result["text_tokens"] = text_tokens
            result["text_lengths"] = torch.tensor(text_lengths, dtype=torch.long)

        return result


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio dataset for GuBERT training")

    # Mode
    parser.add_argument("--mode", type=str, choices=["ctc", "masked"], default="ctc",
                        help="Training mode: 'ctc' requires transcriptions, 'masked' is self-supervised")

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

    # Dataset columns
    parser.add_argument("--text_column", type=str, default="text",
                        help="Column name for text transcription")
    parser.add_argument("--speaker_id_column", type=str, default="speaker_id",
                        help="Column name for speaker ID")

    # Filtering
    parser.add_argument("--min_audio_energy", type=float, default=0.05,
                        help="Minimum audio energy (skip silent samples)")
    parser.add_argument("--min_audio_std", type=float, default=0.02,
                        help="Minimum audio std (skip near-silent samples)")
    parser.add_argument("--min_text_length", type=int, default=2,
                        help="Minimum text length (skip very short transcriptions)")
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

    print(f"GuBERT Dataset Preprocessing")
    print(f"=============================")
    print(f"Mode: {args.mode.upper()}")
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
    processor = GuBERTBatchProcessor(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        max_audio_seconds=args.max_audio_seconds,
        device=device,
        mode=args.mode,
    )

    # Stats
    stats = {
        "processed": 0,
        "saved": 0,
        "skipped_silent": 0,
        "skipped_short_text": 0,
        "skipped_no_speaker": 0,
        "skipped_error": 0,
    }

    # Track unique speaker IDs for remapping
    speaker_id_set = set()
    speaker_id_to_idx = {}

    # First pass: collect all unique speaker IDs for remapping
    print("First pass: collecting speaker IDs...")
    for idx in tqdm(range(len(dataset)), desc="Collecting speakers"):
        try:
            example = dataset[idx]
            speaker_id = example.get(args.speaker_id_column, None)
            if speaker_id is not None:
                speaker_id_set.add(speaker_id)
        except Exception:
            pass

    # Create mapping from original speaker ID to contiguous indices
    for i, sid in enumerate(sorted(speaker_id_set)):
        speaker_id_to_idx[sid] = i

    num_speakers = len(speaker_id_to_idx)
    print(f"  Found {num_speakers} unique speakers")

    # Shard accumulators
    shard_mel_specs = []
    shard_mel_lengths = []
    shard_text_tokens = []  # CTC mode only
    shard_text_lengths = []  # CTC mode only
    shard_speaker_ids = []
    shard_raw_texts = []  # Keep original texts for debugging (CTC mode)
    shard_idx = 0

    def flush_shard():
        nonlocal shard_mel_specs, shard_mel_lengths, shard_text_tokens, shard_text_lengths
        nonlocal shard_speaker_ids, shard_raw_texts, shard_idx

        if not shard_mel_specs:
            return

        shard_data = {
            "mel_specs": torch.cat(shard_mel_specs, dim=0),
            "mel_lengths": torch.cat(shard_mel_lengths, dim=0),
            "speaker_ids": torch.cat(shard_speaker_ids, dim=0),
            "num_samples": sum(x.shape[0] for x in shard_mel_specs),
            "mode": args.mode,
        }

        # Add text data only in CTC mode
        if args.mode == "ctc" and shard_text_tokens:
            # Find max text length in this shard
            max_text_len = max(t.shape[1] for t in shard_text_tokens)

            # Pad all text tokens to same length
            padded_text_tokens = []
            for tokens in shard_text_tokens:
                if tokens.shape[1] < max_text_len:
                    tokens = F.pad(tokens, (0, max_text_len - tokens.shape[1]), value=0)
                padded_text_tokens.append(tokens)

            shard_data["text_tokens"] = torch.cat(padded_text_tokens, dim=0)
            shard_data["text_lengths"] = torch.cat(shard_text_lengths, dim=0)
            shard_data["raw_texts"] = shard_raw_texts

        shard_path = os.path.join(output_dir, f"shard_{shard_idx:06d}.pt")
        torch.save(shard_data, shard_path)

        print(f"  Saved shard {shard_idx} ({shard_data['num_samples']} samples)")

        shard_mel_specs = []
        shard_mel_lengths = []
        shard_text_tokens = []
        shard_text_lengths = []
        shard_speaker_ids = []
        shard_raw_texts = []
        shard_idx += 1

    # Batch accumulators
    batch_waveforms = []
    batch_texts = []
    batch_speaker_ids = []

    def process_and_accumulate():
        nonlocal batch_waveforms, batch_texts, batch_speaker_ids
        nonlocal shard_mel_specs, shard_mel_lengths, shard_text_tokens, shard_text_lengths
        nonlocal shard_speaker_ids, shard_raw_texts

        if not batch_waveforms:
            return

        try:
            result = processor.process_batch(batch_waveforms, batch_texts, batch_speaker_ids)

            # Add to shard
            shard_mel_specs.append(result["mel_specs"])
            shard_mel_lengths.append(result["mel_lengths"])
            shard_speaker_ids.append(result["speaker_ids"])

            # Add text data only in CTC mode
            if args.mode == "ctc":
                shard_text_tokens.append(result["text_tokens"])
                shard_text_lengths.append(result["text_lengths"])
                shard_raw_texts.extend(batch_texts)

            stats["saved"] += len(batch_waveforms)

            # Flush shard if full
            current_size = sum(x.shape[0] for x in shard_mel_specs)
            if current_size >= args.shard_size:
                flush_shard()

        except Exception as e:
            print(f"Batch processing error: {e}")
            stats["skipped_error"] += len(batch_waveforms)

        batch_waveforms = []
        batch_texts = []
        batch_speaker_ids = []

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

            # Get text transcription (only required in CTC mode)
            text = example.get(args.text_column, "")
            if args.mode == "ctc" and len(text) < args.min_text_length:
                stats["skipped_short_text"] += 1
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

            # Remove mains hum if enabled
            if args.remove_mains_hum:
                waveform = remove_mains_hum(waveform.unsqueeze(0), args.sample_rate).squeeze(0)

            # Check length
            max_samples = args.max_audio_seconds * args.sample_rate
            if len(waveform) > max_samples:
                waveform = waveform[:max_samples]

            # Add to batch
            batch_waveforms.append(waveform)
            batch_texts.append(text)
            batch_speaker_ids.append(speaker_id)

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
        "mode": args.mode,
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
        "remove_mains_hum": args.remove_mains_hum,
        "shard_size": args.shard_size,
        "stats": stats,
    }

    # Add vocab size only in CTC mode
    if args.mode == "ctc":
        config["vocab_size"] = processor.vocab.vocab_size

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*50}")
    print(f"GPU {args.gpu_id} complete! (Mode: {args.mode.upper()})")
    print(f"  Processed: {stats['processed']:,}")
    print(f"  Saved: {stats['saved']:,}")
    print(f"  Skipped (silent): {stats['skipped_silent']:,}")
    if args.mode == "ctc":
        print(f"  Skipped (short text): {stats['skipped_short_text']:,}")
    print(f"  Skipped (no speaker): {stats['skipped_no_speaker']:,}")
    print(f"  Skipped (error): {stats['skipped_error']:,}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Speed: {stats['saved']/elapsed:.1f} samples/sec")
    print(f"  Shards: {shard_idx}")
    print(f"  Unique speakers: {num_speakers}")
    if args.mode == "ctc":
        print(f"  Vocab size: {processor.vocab.vocab_size}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
