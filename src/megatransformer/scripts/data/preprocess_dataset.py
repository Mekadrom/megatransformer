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

from megatransformer.scripts.data.voice.preprocess import VoiceDatasetPreprocessor
from megatransformer.scripts.data.image.preprocess import ImageDatasetPreprocessor
from megatransformer.scripts.data.image.vae.preprocess import ImageVAEDatasetPreprocessor
from megatransformer.scripts.data.text.preprocess import TextDatasetPreprocessor
from megatransformer.scripts.data.preprocessor import Preprocessor


preprocessor_clss: list[type[Preprocessor]] = [
    VoiceDatasetPreprocessor,
    ImageDatasetPreprocessor,
    ImageVAEDatasetPreprocessor,
    TextDatasetPreprocessor,
]


def _collect_shard_info(shard_dir, col=None):
    """Scan a shard directory: return (shard_files, sample_counts, unique_speaker_ids)."""
    shard_files = sorted([
        f for f in os.listdir(shard_dir)
        if f.startswith("shard_") and f.endswith(".pt")
    ])
    sample_counts = []
    speaker_ids = set()
    for shard_file in shard_files:
        shard = torch.load(os.path.join(shard_dir, shard_file), map_location="cpu", weights_only=False)
        sample_counts.append(shard["num_samples"])
        if col is not None and col in shard:
            ids = shard[col]
            if isinstance(ids, torch.Tensor):
                speaker_ids.update(ids.tolist())
            elif isinstance(ids, list):
                speaker_ids.update(ids)
    return shard_files, sample_counts, speaker_ids


def _remap_shards(shard_dir, shard_files, col, id_to_seq):
    """Rewrite speaker IDs in shards using the global mapping."""
    for shard_file in tqdm(shard_files, desc=f"Remapping {shard_dir}"):
        shard_path = os.path.join(shard_dir, shard_file)
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        if col in shard:
            ids = shard[col]
            if isinstance(ids, torch.Tensor):
                remapped = torch.tensor([id_to_seq[x.item()] for x in ids], dtype=torch.long)
            elif isinstance(ids, list):
                remapped = torch.tensor([id_to_seq[x] for x in ids], dtype=torch.long)
            else:
                continue
            shard[col] = remapped
            torch.save(shard, shard_path)


def _write_index(shard_dir, shard_files, sample_counts, num_speakers, command):
    """Write shard_index.json for a directory."""
    shard_offsets = []
    total = 0
    for count in sample_counts:
        shard_offsets.append(total)
        total += count

    index_data = {
        "shard_files": shard_files,
        "shard_offsets": shard_offsets,
        "total_samples": total,
        "dataset_type": command,
    }
    if num_speakers > 0:
        index_data["num_speakers"] = num_speakers

    index_path = os.path.join(shard_dir, "shard_index.json")
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)
    print(f"  {shard_dir}: {len(shard_files)} shards, {total:,} samples → {index_path}")


def _stat_shards(args):
    """Build shard index for existing shards, and optionally remap speaker IDs
    to dense sequential integers across all directories (train + val)."""
    col = args.speaker_id_column
    all_dirs = [args.output_dir] + (args.additional_shard_dirs or [])

    # Pass 1: scan all directories, collect unique speaker IDs globally
    dir_info = {}  # dir → (shard_files, sample_counts)
    all_speaker_ids = set()

    for d in all_dirs:
        if not os.path.isdir(d):
            print(f"Warning: {d} not found, skipping")
            continue
        print(f"Scanning {d}...")
        shard_files, sample_counts, spk_ids = _collect_shard_info(d, col)
        if not shard_files:
            print(f"  No shard files found, skipping")
            continue
        dir_info[d] = (shard_files, sample_counts)
        all_speaker_ids.update(spk_ids)

    if not dir_info:
        print("No shard files found in any directory.")
        exit(1)

    # Build global dense mapping
    num_speakers = 0
    if col is not None and all_speaker_ids:
        sorted_ids = sorted(all_speaker_ids)
        id_to_seq = {native: seq for seq, native in enumerate(sorted_ids)}
        num_speakers = len(sorted_ids)
        is_already_dense = (
            all(isinstance(x, (int, float)) for x in sorted_ids)
            and len(sorted_ids) == (max(sorted_ids) - min(sorted_ids) + 1)
            and min(sorted_ids) == 0
        )

        if is_already_dense:
            print(f"Speaker IDs already dense: {num_speakers} speakers, range [0, {num_speakers - 1}]")
        else:
            print(f"Remapping {num_speakers} unique speaker IDs to dense [0, {num_speakers - 1}] across {len(dir_info)} dirs...")
            # Pass 2: rewrite speaker IDs in all directories
            for d, (shard_files, _) in dir_info.items():
                _remap_shards(d, shard_files, col, id_to_seq)

    # Write index for each directory
    print()
    for d, (shard_files, sample_counts) in dir_info.items():
        _write_index(d, shard_files, sample_counts, num_speakers, args.command)
    print(f"\nDone.")


def get_preprocessor(command: str, args, dataset, output_dir, shard_fields, batch_accumulators, stats, device: str) -> Preprocessor:
    if command == "voice":
        return VoiceDatasetPreprocessor(args, dataset, output_dir, shard_fields, batch_accumulators, stats, device=device)
    elif command == "image":
        return ImageDatasetPreprocessor(args, dataset, output_dir, shard_fields, batch_accumulators, stats, device=device)
    elif command == "image-vae":
        return ImageVAEDatasetPreprocessor(args, dataset, output_dir, shard_fields, batch_accumulators, stats, device=device)
    elif command == "text":
        return TextDatasetPreprocessor(args, dataset, output_dir, shard_fields, batch_accumulators, stats, device=device)
    else:
        raise ValueError(f"Unknown command: {command}. Available: voice, image, image-vae, text")


def main():
    parser = argparse.ArgumentParser(description="Extract SIVE features for VAE training")
    
    subparsers = parser.add_subparsers(dest="command")
    sub_parser = subparsers.add_parser("stat-shards", help="Produces the index file for existing shards. Combine manually for multi-gpu processing.")
    sub_parser.add_argument("--output_dir", type=str, required=True, help="Output directory for shards")
    sub_parser.add_argument("--speaker_id_column", type=str, default=None, help="If specified, remap speaker IDs to dense sequential integers")
    sub_parser.add_argument("--additional_shard_dirs", type=str, nargs="*", default=[], help="Extra shard dirs (e.g. val split) to include in global speaker ID mapping. These dirs also get remapped and their own shard_index.json written.")
    for preprocessor_cls in preprocessor_clss:
        # Dataset
        sub_parser = preprocessor_cls.add_cli_args(subparsers)
        sub_parser.add_argument("--output_dir", type=str, required=True,
                            help="Output directory for shards")
        sub_parser.add_argument("--dataset_name", type=str,
                            help="HuggingFace dataset name")
        sub_parser.add_argument("--dataset_config", type=str,
                            help="Dataset configuration")
        sub_parser.add_argument("--data_dir", type=str, default=None,
                            help="Optional subdirectory within the dataset to load from. "
                                 "Use for datasets that organize content as language/category "
                                 "subdirectories without proper HF configs (e.g. "
                                 "bigcode/starcoderdata --data_dir python). When set, only "
                                 "files under that subdirectory are loaded.")
        sub_parser.add_argument("--split", type=str,
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
        # Streaming
        sub_parser.add_argument("--streaming", action="store_true", default=False,
                            help="Stream dataset instead of downloading entirely (avoids disk caching)")

    args = parser.parse_args()

    if args.command == "stat-shards":
        _stat_shards(args)
        return

    # Create output directory with GPU suffix for parallel processing
    if args.total_gpus > 1:
        output_dir = os.path.join(args.output_dir, f"gpu_{args.gpu_id}")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"GPU {args.gpu_id}/{args.total_gpus}")
    print(f"Processing every {args.total_gpus}th sample starting at offset {args.gpu_id}")
    print(f"Output: {output_dir}")

    # Load dataset
    streaming = getattr(args, 'streaming', False)
    data_dir = getattr(args, 'data_dir', None)
    print(f"Loading dataset {args.dataset_name}/{args.dataset_config} split {args.split} "
          f"(streaming={streaming}, data_dir={data_dir})...")
    load_kwargs = dict(
        path=args.dataset_name,
        name=args.dataset_config,
        split=args.split,
        trust_remote_code=True,
        streaming=streaming,
    )
    if data_dir:
        load_kwargs["data_dir"] = data_dir
    dataset = load_dataset(**load_kwargs)

    # Shard accumulators
    shard_fields = {
        "shard_idx": args.gpu_id,
    }

    # Batch accumulators
    batch_accumulators: dict[Any, Any] = {}

    # Stats - use defaultdict for skipped reasons so preprocessors can add their own.
    # `tokens_saved` is maintained by the text preprocessor; `seconds_saved` by
    # the voice preprocessor. Both let the main loop enforce size budgets
    # (--max_tokens / --max_hours). For subcommands that don't emit tokens or
    # audio, the counters stay at 0 and the corresponding checks are no-ops.
    stats = {
        "processed": 0,
        "saved": 0,
        "tokens_saved": 0,
        "seconds_saved": 0.0,
        "skipped": defaultdict(int),
    }

    # Corpus-size caps. Only one is meaningful per subcommand (text → tokens,
    # voice → hours). For subcommands that don't track the relevant counter,
    # the budget resolves to None and the check is a no-op.
    max_tokens_budget = getattr(args, "max_tokens", None)
    max_hours_budget = getattr(args, "max_hours", None)
    max_seconds_budget = max_hours_budget * 3600 if max_hours_budget else None

    preprocessor: Preprocessor = get_preprocessor(args.command, args, dataset, output_dir, shard_fields, batch_accumulators, stats, device)

    # preprocessor may modify dataset (e.g., map speaker IDs)
    dataset = preprocessor.dataset if hasattr(preprocessor, 'dataset') else dataset

    assert args.dataset_name is not None, "--dataset_name is required"
    assert args.split is not None, "--split is required"

    # Main processing loop
    start_time = time.time()

    if streaming:
        # Streaming mode: iterate directly, no len() or indexing
        total = args.max_samples or None
        pbar = tqdm(total=total, desc=f"GPU {args.gpu_id}")
        for idx, example in enumerate(dataset):
            if idx < args.start_idx:
                continue
            if idx % args.total_gpus != args.gpu_id:
                continue
            if args.max_samples and stats["saved"] >= args.max_samples:
                break
            if max_tokens_budget and stats["tokens_saved"] >= max_tokens_budget:
                break
            if max_seconds_budget and stats["seconds_saved"] >= max_seconds_budget:
                break
            try:
                if preprocessor.preprocess_example(example):
                    stats["processed"] += 1
                pbar.update(1)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error processing sample {idx}: {e}")
                stats["skipped"]["error"] += 1
                pbar.update(1)
                continue
    else:
        # Standard mode: random access by index
        samples_for_this_gpu = len([
            i for i in range(args.start_idx, len(dataset))
            if i % args.total_gpus == args.gpu_id
        ])
        print(f"  Samples for this GPU: {samples_for_this_gpu:,}")

        pbar = tqdm(total=samples_for_this_gpu, desc=f"GPU {args.gpu_id}")
        for idx in range(args.start_idx, len(dataset)):
            if idx % args.total_gpus != args.gpu_id:
                continue
            if args.max_samples and stats["saved"] >= args.max_samples:
                break
            if max_tokens_budget and stats["tokens_saved"] >= max_tokens_budget:
                break
            if max_seconds_budget and stats["seconds_saved"] >= max_seconds_budget:
                break
            try:
                example = dataset[idx]
                if preprocessor.preprocess_example(example):
                    stats["processed"] += 1
                pbar.update(1)
            except Exception as e:
                import traceback
                traceback.print_exc()
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
    if stats.get("tokens_saved", 0) > 0:
        print(f"  Tokens saved: {stats['tokens_saved']:,}")
    if stats.get("seconds_saved", 0) > 0:
        print(f"  Audio saved: {stats['seconds_saved']:,.1f} s ({stats['seconds_saved'] / 3600:.2f} h)")
    for reason in stats['skipped']:
        print(f"  Skipped ({reason}): {stats['skipped'][reason]:,}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Speed: {stats['saved']/elapsed:.1f} samples/sec")

    # Image preprocessing: surface the download failure breakdown so 100%-fail
    # runs don't look like a mystery ("all 420k skipped as download_failed").
    try:
        from megatransformer.scripts.data.image.preprocess import print_download_error_summary
        print_download_error_summary()
    except ImportError:
        pass
    print(f"  Shards: {shard_fields['shard_idx']}")
    print(f"  Output: {output_dir}")

    # Streaming mode leaves background prefetch threads alive inside the HF
    # datasets library. Their HTTP retries race with Python interpreter
    # teardown and trigger "Bad file descriptor" / "PyGILState_Release"
    # crashes (SIGABRT) AFTER preprocessing has already succeeded. All shards
    # and config.json have been flushed by this point, so bypassing Python
    # finalizers via os._exit is safe — it skips the teardown race entirely.
    if streaming:
        import sys as _sys
        _sys.stdout.flush()
        _sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
