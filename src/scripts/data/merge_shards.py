"""Merge shards from multiple input directories into uniformly-sized output shards.

Walks one or more shard directories, builds a global index of every sample across
every input shard, optionally shuffles, optionally caps, and emits output shards
each containing exactly `--samples_per_shard` samples (last one may be smaller).

Output shards can contain samples drawn from any input directory — boundary
shards will mix datasets freely, which is the point when preparing a combined
training corpus.

After merging, run stat-shards to regenerate `shard_index.json`:
    python -m scripts.data.preprocess_dataset stat-shards --output_dir <merged_dir>

Usage:
    python -m scripts.data.merge_shards --input_dirs cached_datasets/voice_sive/librispeech cached_datasets/voice_sive/libritts-r cached_datasets/voice_sive/mls --output_dir cached_datasets/voice_sive_train --samples_per_shard 1000 --shuffle
"""

import argparse
import math
import os
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm


def _detect_num_samples(shard: dict) -> int:
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
    try:
        return len(value) == n
    except TypeError:
        return False


def _extract_one(value, idx: int):
    """Pull a single sample out of a per-sample field, detaching tensor storage."""
    if isinstance(value, torch.Tensor):
        return value[idx].clone()
    return value[idx]


def _max_shape(tensors: list[torch.Tensor]) -> tuple:
    """Element-wise max shape across a list of same-ndim tensors."""
    max_shape = list(tensors[0].shape)
    for t in tensors[1:]:
        if t.ndim != len(max_shape):
            raise RuntimeError(
                f"Tensor ndim mismatch: {t.ndim} vs {len(max_shape)}. "
                "Shards have incompatible tensor rank for this field — "
                "check preprocessor consistency."
            )
        for i, d in enumerate(t.shape):
            if d > max_shape[i]:
                max_shape[i] = d
    return tuple(max_shape)


def _pad_to_shape(t: torch.Tensor, target: tuple) -> torch.Tensor:
    """Zero-pad a tensor on the right of each dim up to target shape. Never truncates."""
    if tuple(t.shape) == target:
        return t
    # F.pad takes pads in reverse-dim order: (last_dim_left, last_dim_right, ...)
    pads = []
    for i in reversed(range(len(target))):
        pad_amount = target[i] - t.shape[i]
        if pad_amount < 0:
            raise RuntimeError(
                f"Negative pad requested (shape {t.shape} > target {target}). "
                "This shouldn't happen since target is the element-wise max."
            )
        pads.extend([0, pad_amount])
    return F.pad(t, pads)


def _stack_with_padding(samples: list[torch.Tensor]) -> torch.Tensor:
    """Stack tensors that may have different shapes, padding shorter ones to max with zeros."""
    target = _max_shape(samples)
    padded = [_pad_to_shape(s, target) for s in samples]
    return torch.stack(padded)


def _discover_shards(input_dirs: list[str]) -> list[str]:
    """Return absolute paths of all shard_*.pt files across the input dirs,
    sorted within each dir, dirs processed in the order provided."""
    out = []
    for d in input_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Input dir does not exist: {d}")
        names = sorted(
            f for f in os.listdir(d)
            if f.startswith("shard_") and f.endswith(".pt")
        )
        if not names:
            raise FileNotFoundError(f"No shard_*.pt files in {d}")
        for name in names:
            out.append(os.path.join(d, name))
    return out


def merge(
    input_dirs: list[str],
    output_dir: str,
    samples_per_shard: int,
    shuffle: bool,
    seed: int,
    max_samples: Optional[int],
    base_name: str,
):
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f"{output_dir} exists and is not empty")
    os.makedirs(output_dir, exist_ok=True)

    input_shards = _discover_shards(input_dirs)
    print(f"Discovered {len(input_shards)} input shards across {len(input_dirs)} directories")

    # First pass: count samples per input shard (mmap keeps this cheap).
    input_sizes = []
    for path in tqdm(input_shards, desc="Counting"):
        s = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        input_sizes.append(_detect_num_samples(s))
        del s

    total = sum(input_sizes)
    print(f"Total samples across all inputs: {total:,}")

    # Build flat global index: [(input_shard_idx, local_idx), ...]
    flat = [(i, j) for i, n in enumerate(input_sizes) for j in range(n)]

    if shuffle:
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(total, generator=rng).tolist()
        flat = [flat[p] for p in perm]
        print(f"Shuffled with seed={seed}")

    if max_samples is not None and max_samples > 0 and max_samples < total:
        flat = flat[:max_samples]
        print(f"Capped to {max_samples:,} samples")

    total = len(flat)
    n_out = math.ceil(total / samples_per_shard)
    print(f"Writing {n_out} output shards × up to {samples_per_shard} samples → {output_dir}")

    # Detect per-sample keys + metadata from the first input shard.
    first = torch.load(input_shards[0], map_location="cpu", weights_only=False, mmap=True)
    n_first = _detect_num_samples(first)
    per_sample_keys = [
        k for k, v in first.items()
        if _is_per_sample(v, n_first) and k != "num_samples"
    ]
    metadata = {
        k: v for k, v in first.items()
        if k not in per_sample_keys and k != "num_samples"
    }
    del first

    if not per_sample_keys:
        raise ValueError("Could not detect any per-sample fields in the first input shard")

    for out_idx in tqdm(range(n_out), desc="Writing"):
        start = out_idx * samples_per_shard
        end = min(start + samples_per_shard, total)
        needed = flat[start:end]  # list of (in_idx, local_idx)

        # Group needed samples by input shard so we load each at most once per output.
        by_input: dict[int, list[tuple[int, int]]] = {}
        for out_pos, (in_idx, local_idx) in enumerate(needed):
            by_input.setdefault(in_idx, []).append((local_idx, out_pos))

        # Output buffers: one slot per destination position, per field.
        out_buf: dict[str, list] = {k: [None] * len(needed) for k in per_sample_keys}

        for in_idx, pairs in by_input.items():
            shard = torch.load(
                input_shards[in_idx],
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            missing = [k for k in per_sample_keys if k not in shard]
            if missing:
                raise KeyError(
                    f"Input shard {input_shards[in_idx]} is missing per-sample keys {missing}"
                )
            for local_idx, out_pos in pairs:
                for k in per_sample_keys:
                    out_buf[k][out_pos] = _extract_one(shard[k], local_idx)
            del shard

        # Assemble final shard: stack tensor fields (auto-pad to the max shape
        # among samples in this output shard), keep list fields as lists.
        # Per-sample length fields (mel_lengths, feature_lengths, ctc_lengths, etc.)
        # are stacked as-is — they track real-signal lengths within the padded
        # tensors, and the collator uses them to mask padding at training time.
        final = dict(metadata)
        for k in per_sample_keys:
            samples = out_buf[k]
            if samples and isinstance(samples[0], torch.Tensor):
                try:
                    final[k] = _stack_with_padding(samples)
                except RuntimeError as e:
                    raise RuntimeError(
                        f"Failed to stack field '{k}' across inputs: {e}"
                    )
            else:
                final[k] = samples
        final["num_samples"] = len(needed)

        out_name = f"{base_name}_{out_idx:06d}.pt"
        torch.save(final, os.path.join(output_dir, out_name))

    print(f"\nWrote {n_out} shards totaling {total:,} samples to {output_dir}")
    print("\nNext step: rerun stat-shards to build the shard index:")
    print(f"    python -m scripts.data.preprocess_dataset stat-shards --output_dir {output_dir}")


def main():
    p = argparse.ArgumentParser(description="Merge shards from multiple dirs into uniform output shards")
    p.add_argument("--input_dirs", type=str, nargs="+", required=True,
                   help="One or more input directories, each containing shard_*.pt files")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to write merged shards (must not exist or be empty)")
    p.add_argument("--samples_per_shard", type=int, required=True,
                   help="Target number of samples per output shard")
    p.add_argument("--shuffle", action="store_true", default=False,
                   help="Shuffle samples across all inputs before chunking into output shards")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap total output samples (applied after shuffle)")
    p.add_argument("--base_name", type=str, default="shard",
                   help="Prefix for output shard filenames")
    args = p.parse_args()

    merge(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        samples_per_shard=args.samples_per_shard,
        shuffle=args.shuffle,
        seed=args.seed,
        max_samples=args.max_samples,
        base_name=args.base_name,
    )


if __name__ == "__main__":
    main()
