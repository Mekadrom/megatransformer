"""Compute per-channel latent scales for --image_latent_channel_scales.

The DiT normalizes clean latents to ~unit variance per channel via
`x_0_scaled = x_0 * latent_scale` (diffusion_decoder.py). To bring channel c to
unit std the scale must be **1/std_c**, so this script measures the per-channel
std of the cached latents across ALL images and spatial positions and prints the
reciprocals, ready to paste into `--image_latent_channel_scales`.

The scales are an empirical statistic of the EXACT cache/mixture, so recompute
them whenever the training mixture changes or the cache is rebuilt (e.g. after
fixing the datasets-2.21 blank-latent corruption). Point it at the MERGED cache
you actually train on, not a single dataset.

Streams shard-by-shard with a running sum/sumsq accumulator, so it handles a
full multi-million-image merged cache without loading everything into memory.

Usage:
    python scripts_local/compute_latent_channel_scales.py --cache_dir ../cached_datasets/world_image_merged
    # optional: --max_shards N (quick estimate), --latents_key latents
"""

import argparse
import glob
import os

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Per-channel latent scale (1/std) for --image_latent_channel_scales")
    p.add_argument("--cache_dir", type=str, required=True,
                   help="Merged cache dir containing shard_*.pt files")
    p.add_argument("--latents_key", type=str, default="latents",
                   help="Key holding the (N,C,H,W) latent tensor in each shard (default: latents)")
    p.add_argument("--max_shards", type=int, default=0,
                   help="If >0, only scan this many shards (quick estimate; 0 = all)")
    p.add_argument("--precision", type=int, default=4, help="Decimal places in the printed list")
    return p.parse_args()


def main():
    args = parse_args()
    shards = sorted(glob.glob(os.path.join(args.cache_dir, "shard_*.pt")))
    if not shards:
        raise SystemExit(f"No shard_*.pt found in {args.cache_dir}")
    if args.max_shards > 0:
        shards = shards[:args.max_shards]

    # Per-channel running accumulators in float64 for numerical stability.
    count = 0            # number of (image x spatial) elements per channel
    ch_sum = None        # (C,)
    ch_sumsq = None      # (C,)

    for i, f in enumerate(shards):
        sh = torch.load(f, map_location="cpu", weights_only=False)
        if args.latents_key not in sh:
            raise SystemExit(f"{f} has no key '{args.latents_key}' (keys: {list(sh.keys())})")
        lat = sh[args.latents_key].to(torch.float64)  # (N,C,H,W)
        if lat.ndim != 4:
            raise SystemExit(f"{f}: expected (N,C,H,W) latents, got shape {tuple(lat.shape)}")
        n, c, h, w = lat.shape
        # collapse everything except channel -> (C, N*H*W)
        flat = lat.permute(1, 0, 2, 3).reshape(c, -1)
        if ch_sum is None:
            ch_sum = torch.zeros(c, dtype=torch.float64)
            ch_sumsq = torch.zeros(c, dtype=torch.float64)
        ch_sum += flat.sum(dim=1)
        ch_sumsq += (flat * flat).sum(dim=1)
        count += flat.shape[1]
        print(f"  [{i + 1}/{len(shards)}] {os.path.basename(f)}: {n} imgs  (cum elems/ch={count})", flush=True)

    mean = ch_sum / count
    var = ch_sumsq / count - mean * mean
    # unbiased (N-1) correction to match torch.std default; negligible at scale.
    var = var * (count / max(count - 1, 1))
    std = var.clamp(min=1e-12).sqrt()
    inv = (1.0 / std)

    fmt = lambda t: ",".join(f"{v:.{args.precision}f}" for v in t.tolist())
    print()
    print(f"channels: {std.numel()}   elements/channel: {count}   shards: {len(shards)}")
    print(f"per-channel std : {fmt(std)}")
    print(f"std range       : {std.min():.3f}-{std.max():.3f}  ({(std.max() / std.min()):.2f}x)")
    print()
    print("--image_latent_channel_scales (1/std, paste this):")
    print(fmt(inv))


if __name__ == "__main__":
    main()
