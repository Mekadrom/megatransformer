"""Slice existing shards into smaller ones without rerunning preprocessing.

Walks a shard directory, splits each shard into N smaller shards based on
`--samples_per_shard`, and writes them to a new output directory. Works for
any shard format — per-sample fields (whose length matches `num_samples`) are
sliced; everything else (configs, metadata, global stats) is copied verbatim
into every output shard.

After resharding, run stat-shards to regenerate `shard_index.json`:
    python -m megatransformer.scripts.data.preprocess_dataset stat-shards --output_dir <new_dir>

Usage:
    python -m megatransformer.scripts.data.reshard --input_dir ../cached_datasets/voice_sive_train --output_dir ../cached_datasets/voice_sive_train_small --samples_per_shard 1000
"""

import argparse
import math
import os
import shutil

import torch
from tqdm import tqdm


def _detect_num_samples(shard: dict) -> int:
    """Figure out how many samples are in this shard.

    Prefer the explicit `num_samples` field; fall back to the max length
    across any field that supports len(). Lists of strings, tensors, and
    numpy arrays all have __len__.
    """
    if "num_samples" in shard and isinstance(shard["num_samples"], int):
        return shard["num_samples"]
    lens = []
    for v in shard.values():
        try:
            lens.append(len(v))
        except TypeError:
            pass
    if not lens:
        raise ValueError("Cannot determine num_samples: no sized fields in shard")
    return max(lens)


def _is_per_sample(value, n: int) -> bool:
    """A field is 'per-sample' if len(value) == n. Anything else (scalars,
    dicts, differently-sized tensors) is treated as metadata."""
    try:
        return len(value) == n
    except TypeError:
        return False


def _slice_value(value, start: int, end: int):
    """Slice tensors, lists, and numpy arrays, detaching the slice from its
    source storage. Critical for tensors: slicing creates a VIEW that still
    references the full original storage, and torch.save() would then save
    the entire original storage with every output shard (easy way to 10x
    the dataset size). .clone() forces a new compact storage for just the
    slice. Same issue applies to numpy arrays (slicing returns a view)."""
    if isinstance(value, torch.Tensor):
        return value[start:end].clone()
    # Lists slice cleanly (new list object with references to per-item objects).
    # Numpy arrays need .copy() to detach from source storage.
    sliced = value[start:end]
    if hasattr(sliced, "copy") and type(sliced).__module__.startswith("numpy"):
        return sliced.copy()
    return sliced


def split_shard(shard_path: str, output_dir: str, samples_per_shard: int,
                base_name: str, start_index: int) -> int:
    """Split one shard into multiple smaller ones. Returns the number of
    output shards written."""
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    n = _detect_num_samples(shard)

    per_sample_keys = [k for k, v in shard.items() if _is_per_sample(v, n) and k != "num_samples"]
    metadata = {k: v for k, v in shard.items() if k not in per_sample_keys and k != "num_samples"}

    n_new = math.ceil(n / samples_per_shard)
    for i in range(n_new):
        start = i * samples_per_shard
        end = min(start + samples_per_shard, n)
        new_shard = dict(metadata)
        for k in per_sample_keys:
            new_shard[k] = _slice_value(shard[k], start, end)
        new_shard["num_samples"] = end - start

        out_name = f"{base_name}_{start_index + i:06d}.pt"
        torch.save(new_shard, os.path.join(output_dir, out_name))
    return n_new


def main():
    p = argparse.ArgumentParser(description="Slice shards into smaller ones")
    p.add_argument("--input_dir", type=str, required=True, help="Directory containing shards to split")
    p.add_argument("--output_dir", type=str, required=True, help="Directory to write sliced shards (must not exist or be empty)")
    p.add_argument("--samples_per_shard", type=int, required=True, help="Target number of samples per output shard")
    p.add_argument("--base_name", type=str, default="shard", help="Prefix for output shard filenames")
    p.add_argument("--copy_index", action="store_true", help="If set, also copy shard_index.json (you'll want to rerun stat-shards after)")
    args = p.parse_args()

    if os.path.isdir(args.output_dir) and os.listdir(args.output_dir):
        raise FileExistsError(f"{args.output_dir} exists and is not empty")
    os.makedirs(args.output_dir, exist_ok=True)

    shard_files = sorted(f for f in os.listdir(args.input_dir)
                         if f.startswith("shard_") and f.endswith(".pt"))
    if not shard_files:
        raise FileNotFoundError(f"No shard_*.pt files in {args.input_dir}")

    print(f"Splitting {len(shard_files)} shards → {args.samples_per_shard} samples each → {args.output_dir}")

    total_out = 0
    for sf in tqdm(shard_files, desc="Splitting"):
        in_path = os.path.join(args.input_dir, sf)
        total_out += split_shard(in_path, args.output_dir, args.samples_per_shard,
                                 args.base_name, total_out)

    print(f"Wrote {total_out} output shards")

    if args.copy_index:
        idx = os.path.join(args.input_dir, "shard_index.json")
        if os.path.exists(idx):
            shutil.copy2(idx, args.output_dir)
            print("Copied shard_index.json (rerun stat-shards to regenerate for new layout)")

    print("\nNext step: rerun stat-shards to build the shard index for the new directory:")
    print(f"    python -m megatransformer.scripts.data.preprocess_dataset stat-shards --output_dir {args.output_dir}")


if __name__ == "__main__":
    main()
