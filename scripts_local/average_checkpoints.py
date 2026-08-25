"""Uniformly average world-model checkpoints (SWA / model-soup style).

WHY, HERE. Per-checkpoint eval variance on this project's 8-prompt probe is ~+-0.05 while the
repeat-measurement noise is ~0.005 -- the model is wandering in a basin, which is exactly what
weight averaging smooths. `--use_ema` exists in the trainer but was NOT passed to the from-scratch
runs, so this is the available post-hoc substitute.

⭐ PICK THE WINDOW BY DIVERSITY, NOT RECENCY. Averaging buys variance reduction by combining
DIVERSE points in one basin. Under cosine decay to ~0 the tail checkpoints are nearly identical
(measured: score sd 0.0020 over 96k-100k vs ~0.0070 over 60k-62k), so averaging the last-K -- the
textbook SWA choice -- is close to a no-op here. Prefer a high-scoring window that still has
spread.

⚠️ The score of an average is NOT predictable from its members' scores. Measure it.
⚠️ Judge the result by RENDER (CLIPScore + high-frequency energy), never by training loss: in this
project those are anti-correlated along the dispersion axis. And do NOT finetune afterwards --
that re-settles toward the loss that points the wrong way, and walks off the averaged point.

No BatchNorm anywhere in the image path, so there are no running statistics to recompute (the
standard SWA post-step does not apply). Verified: these checkpoints contain 581 tensors, all
floating point.

Usage:
    python scripts_local/average_checkpoints.py \
        --checkpoints runs/world/<run>/checkpoint-{59000,60000,61000,62000,63000} \
        --output runs/world/<run>/avg-59k-63k
"""
import argparse
import json
import os
import shutil

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="checkpoint DIRS (each containing pytorch_model.bin), >=2")
    p.add_argument("--output", required=True, help="output checkpoint dir to create")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="optional per-checkpoint weights (default: uniform). Normalised.")
    p.add_argument("--copy_aux", action="store_true",
                   help="also copy trainer_state.json/scheduler.pt from the LAST checkpoint. Off "
                        "by default: an averaged model is not a resumable training state, and "
                        "copying optimizer/scheduler files invites resuming from one.")
    return p.parse_args()


def main():
    args = parse_args()
    if len(args.checkpoints) < 2:
        raise SystemExit("need >=2 checkpoints to average")
    w = args.weights or [1.0] * len(args.checkpoints)
    if len(w) != len(args.checkpoints):
        raise SystemExit(f"--weights has {len(w)} entries for {len(args.checkpoints)} checkpoints")
    tot = sum(w)
    w = [x / tot for x in w]

    paths = []
    for d in args.checkpoints:
        f = os.path.join(d, "pytorch_model.bin")
        if not os.path.isfile(f):
            raise SystemExit(f"missing {f}")
        paths.append(f)

    acc, ref_keys, n_nonfloat = None, None, 0
    for i, (f, wi) in enumerate(zip(paths, w)):
        sd = torch.load(f, map_location="cpu", weights_only=True)
        if ref_keys is None:
            ref_keys = set(sd)
        elif set(sd) != ref_keys:
            only_a = sorted(ref_keys - set(sd))[:5]
            only_b = sorted(set(sd) - ref_keys)[:5]
            raise SystemExit(f"key mismatch at {f}\n  missing here: {only_a}\n  extra here: {only_b}")
        if acc is None:
            acc = {}
            for k, v in sd.items():
                if v.dtype.is_floating_point:
                    acc[k] = v.to(torch.float64) * wi
                else:
                    # counters / pointers / ids: averaging them is meaningless. Take the LAST
                    # checkpoint's value (handled after the loop) and report the count.
                    acc[k] = v.clone()
                    n_nonfloat += 1
        else:
            for k, v in sd.items():
                if v.dtype.is_floating_point:
                    acc[k] += v.to(torch.float64) * wi
                else:
                    acc[k] = v.clone()          # last one wins
        print(f"  [{i+1}/{len(paths)}] {os.path.basename(os.path.dirname(f))}  weight={wi:.4f}",
              flush=True)
        del sd

    ref = torch.load(paths[0], map_location="cpu", weights_only=True)
    out = {}
    max_dev, max_dev_key = 0.0, None
    for k, v in acc.items():
        if ref[k].dtype.is_floating_point:
            out[k] = v.to(ref[k].dtype)
            d = float((out[k].to(torch.float32) - ref[k].to(torch.float32)).abs().max())
            if d > max_dev:
                max_dev, max_dev_key = d, k
        else:
            out[k] = v
    os.makedirs(args.output, exist_ok=True)
    torch.save(out, os.path.join(args.output, "pytorch_model.bin"))

    if args.copy_aux:
        for fn in ("trainer_state.json", "scheduler.pt"):
            src = os.path.join(args.checkpoints[-1], fn)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(args.output, fn))

    meta = {"sources": args.checkpoints, "weights": w, "n_tensors": len(out),
            "n_nonfloat_taken_from_last": n_nonfloat,
            "max_abs_deviation_from_first": max_dev, "max_dev_key": max_dev_key}
    with open(os.path.join(args.output, "average_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {args.output}/pytorch_model.bin")
    print(f"  {len(out)} tensors, {n_nonfloat} non-float taken from the last checkpoint")
    print(f"  max |avg - first| = {max_dev:.6g} at {max_dev_key}")
    print("  (a near-zero max deviation means the window had no diversity to average)")


if __name__ == "__main__":
    main()
