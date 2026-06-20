"""Offset speaker IDs in shards to avoid collisions when merging datasets.

Before merging shards from a new dataset into an existing shard directory,
run this to offset the new shards' speaker IDs above the existing max.
Then copy them in and run stat-shards to make everything dense.

Usage:
    # Check existing max speaker ID
    python -m megatransformer.scripts.data.offset_speaker_ids --shard_dir ../cached_datasets/voice_existing_train --dry_run

    # Offset new shards so they don't collide
    python -m megatransformer.scripts.data.offset_speaker_ids --shard_dir ../cached_datasets/voice_new_train --offset_above ../cached_datasets/voice_existing_train

    # Then copy new shards into existing dir and run stat-shards
"""

import argparse
import os

import torch
from tqdm import tqdm


def get_max_speaker_id(shard_dir, col="speaker_ids"):
    """Find the highest speaker ID across all shards in a directory."""
    max_id = -1
    shard_files = sorted(f for f in os.listdir(shard_dir) if f.startswith("shard_") and f.endswith(".pt"))
    for shard_file in shard_files:
        shard = torch.load(os.path.join(shard_dir, shard_file), map_location="cpu", weights_only=False)
        if col in shard:
            ids = shard[col]
            if isinstance(ids, torch.Tensor):
                max_id = max(max_id, ids.max().item())
            elif isinstance(ids, list):
                max_id = max(max_id, max(ids))
    return max_id


def main():
    p = argparse.ArgumentParser(description="Offset speaker IDs in shards to avoid collisions")
    p.add_argument("--shard_dir", type=str, required=True, help="Directory containing shards to offset")
    p.add_argument("--offset_above", type=str, default=None, help="Existing shard directory — offset above its max speaker ID")
    p.add_argument("--offset", type=int, default=None, help="Explicit offset to add (alternative to --offset_above)")
    p.add_argument("--speaker_id_column", type=str, default="speaker_ids")
    p.add_argument("--dry_run", action="store_true", help="Just print max speaker ID, don't modify anything")
    args = p.parse_args()

    col = args.speaker_id_column

    if args.dry_run:
        max_id = get_max_speaker_id(args.shard_dir, col)
        print(f"Max speaker ID in {args.shard_dir}: {max_id}")
        return

    if args.offset is not None:
        offset = args.offset
    elif args.offset_above is not None:
        existing_max = get_max_speaker_id(args.offset_above, col)
        if existing_max < 0:
            print(f"No speaker IDs found in {args.offset_above}")
            return
        offset = existing_max + 1
        print(f"Existing max speaker ID: {existing_max}")
        print(f"Offsetting by: {offset}")
    else:
        print("ERROR: specify either --offset_above or --offset")
        return

    shard_files = sorted(f for f in os.listdir(args.shard_dir) if f.startswith("shard_") and f.endswith(".pt"))
    if not shard_files:
        print(f"No shards found in {args.shard_dir}")
        return

    for shard_file in tqdm(shard_files, desc="Offsetting"):
        shard_path = os.path.join(args.shard_dir, shard_file)
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        if col in shard:
            ids = shard[col]
            if isinstance(ids, torch.Tensor):
                shard[col] = ids + offset
            elif isinstance(ids, list):
                shard[col] = [x + offset for x in ids]
            torch.save(shard, shard_path)

    new_max = get_max_speaker_id(args.shard_dir, col)
    print(f"Done. New speaker ID range in {args.shard_dir}: [{offset}, {new_max}]")


if __name__ == "__main__":
    main()
