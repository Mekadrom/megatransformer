"""Merge multiple preprocessed shard dirs into one, with sequential renumbering.

Preprocess writes shard_000000.pt.. per run, so two runs collide on names. This
gathers shard_*.pt from each --src dir (in the order given) and re-lays them into
--output_dir as a single contiguous shard_000000.. sequence. Hardlinks by default
(instant, no extra disk on the same filesystem); falls back to copy across
filesystems or with --copy.

After merging, run stat-shards on --output_dir to build shard_index.json.

Usage:
    python scripts_local/merge_shard_dirs.py \
        --src ./cached_datasets/.../jackyhate ./cached_datasets/.../progamergov \
        --output_dir ./cached_datasets/.../train_merged
"""

import argparse
import glob
import os
import shutil


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", nargs="+", required=True, help="Source shard dirs, in merge order.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--copy", action="store_true", help="Force copy instead of hardlink.")
    p.add_argument("--max_shards", type=int, default=0, help="Cap total shards written (0=all).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    idx = 0
    total_samples = 0
    for src in args.src:
        shards = sorted(glob.glob(os.path.join(src, "shard_*.pt")))
        if not shards:
            print(f"WARNING: no shards in {src}")
            continue
        for s in shards:
            if args.max_shards and idx >= args.max_shards:
                print(f"reached --max_shards {args.max_shards}, stopping")
                break
            dst = os.path.join(args.output_dir, f"shard_{idx:06d}.pt")
            if os.path.exists(dst):
                os.remove(dst)
            if args.copy:
                shutil.copy2(s, dst)
            else:
                try:
                    os.link(s, dst)          # hardlink: instant, no extra disk
                except OSError:
                    shutil.copy2(s, dst)     # cross-filesystem fallback
            idx += 1
        print(f"  {src}: added {len(shards)} shards")
    print(f"\nmerged {idx} shards into {args.output_dir}")
    print(f"NEXT: python -m megatransformer.scripts.data.preprocess_dataset stat-shards --output_dir {args.output_dir}")


if __name__ == "__main__":
    main()
