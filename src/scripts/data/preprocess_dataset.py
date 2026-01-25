#!/usr/bin/env python3
import argparse
import json
import os
import time
from collections import defaultdict
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm

from scripts.data.audio.vae.preprocess import SIVEFeatureDatasetPreprocessor
from scripts.data.image.preprocess import ImageDatasetPreprocessor
from scripts.data.preprocessor import Preprocessor


preprocessor_clss = [
    SIVEFeatureDatasetPreprocessor,
    ImageDatasetPreprocessor,
]


def get_preprocessor(command: str, args, dataset, output_dir, shard_fields, batch_accumulators, stats, device: str) -> Preprocessor:
    if command == "audio-cvae":
        return SIVEFeatureDatasetPreprocessor(args, dataset, output_dir, shard_fields, batch_accumulators, stats, device=device)
    elif command == "image-vae" or command == "image":
        return ImageDatasetPreprocessor(args, dataset, output_dir, shard_fields, batch_accumulators, stats, device=device)
    else:
        raise ValueError(f"Unknown command: {command}. Available: audio-vae, image-vae")


def main():
    parser = argparse.ArgumentParser(description="Extract SIVE features for VAE training")
    
    subparsers = parser.add_subparsers(dest="command")
    for preprocessor_cls in preprocessor_clss:
        sub_parser = preprocessor_cls.add_cli_args(subparsers)
        # Dataset
        sub_parser.add_argument("--output_dir", type=str, required=True,
                            help="Output directory for shards")
        sub_parser.add_argument("--dataset_name", type=str, required=True,
                            help="HuggingFace dataset name")
        sub_parser.add_argument("--dataset_config", type=str,
                            help="Dataset configuration")
        sub_parser.add_argument("--split", type=str, required=True,
                            help="Dataset split")
        # Multi-GPU
        sub_parser.add_argument("--gpu_id", type=int, default=0,
                            help="This GPU's ID (0-indexed)")
        sub_parser.add_argument("--total_gpus", type=int, default=1,
                            help="Total number of GPUs preprocessing in parallel")
        # Processing
        sub_parser.add_argument("--gpu_batch_size", type=int, default=32,
                            help="Batch size for GPU processing")
        sub_parser.add_argument("--shard_size", type=int, default=2000,
                            help="Number of samples per shard")
        # Limits
        sub_parser.add_argument("--max_samples", type=int, default=None,
                            help="Maximum samples to process (for testing)")
        sub_parser.add_argument("--start_idx", type=int, default=0,
                            help="Starting index in dataset")

    args = parser.parse_args()

    # Create output directory with GPU suffix for parallel processing
    if args.total_gpus > 1:
        output_dir = os.path.join(args.output_dir, f"gpu_{args.gpu_id}")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"GPU {args.gpu_id}/{args.total_gpus}")
    print(f"Processing every {args.total_gpus}th sample starting at offset {args.gpu_id}")
    print(f"Output: {output_dir}")

    # Load dataset
    print(f"Loading dataset {args.dataset_name}/{args.dataset_config} split {args.split}...")
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
        trust_remote_code=True,
    )

    # Shard accumulators
    shard_fields = {
        "shard_idx": 0,
    }

    # Batch accumulators
    batch_accumulators: dict[Any, Any] = {}

    # Stats - use defaultdict for skipped reasons so preprocessors can add their own
    stats = {
        "processed": 0,
        "saved": 0,
        "skipped": defaultdict(int),
    }

    preprocessor: Preprocessor = get_preprocessor(args.command, args, dataset, output_dir, shard_fields, batch_accumulators, stats, device)

    # Calculate samples for this GPU
    samples_for_this_gpu = len([
        i for i in range(args.start_idx, len(dataset))
        if i % args.total_gpus == args.gpu_id
    ])
    print(f"  Samples for this GPU: {samples_for_this_gpu:,}")

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

            if preprocessor.preprocess_example(example):
                stats["processed"] += 1
                pbar.update(1)
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            stats["skipped"]["error"] += 1
            pbar.update(1)
            continue

    # Final flush
    preprocessor.process_and_accumulate()
    preprocessor.flush_shard()

    pbar.close()
    elapsed = time.time() - start_time

    # Save stats and config
    stats["elapsed_seconds"] = elapsed
    stats["samples_per_second"] = stats["saved"] / elapsed if elapsed > 0 else 0
    stats["gpu_id"] = args.gpu_id
    stats["total_gpus"] = args.total_gpus
    stats["num_shards"] = shard_fields['shard_idx']

    config = preprocessor.parse_config()

    # Convert defaultdict to regular dict for JSON serialization
    if "stats" in config and "skipped" in config["stats"]:
        config["stats"]["skipped"] = dict(config["stats"]["skipped"])

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*50}")
    print(f"GPU {args.gpu_id} complete!")
    print(f"  Processed: {stats['processed']:,}")
    print(f"  Saved: {stats['saved']:,}")
    for reason in stats['skipped']:
        print(f"  Skipped ({reason}): {stats['skipped'][reason]:,}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Speed: {stats['saved']/elapsed:.1f} samples/sec")
    print(f"  Shards: {shard_fields['shard_idx']}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
