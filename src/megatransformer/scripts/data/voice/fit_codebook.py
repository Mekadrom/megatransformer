"""Fit a k-means codebook over cached content features (ContentVec / SIVE).

Fit on TRAIN only, then point training at the saved codebook with --voice_codebook_path.
Quantization happens on the fly in the dataset, so trying a different K is a re-fit plus a
flag -- no re-preprocessing.

    python -m megatransformer.scripts.data.voice.fit_codebook \
        --cache_dir ../cached_datasets/voice_contentvec_train \
        --k 500 --output ./codebooks/contentvec_k500.pt

Reported diagnostics and how to read them:
  quant_l1 vs trivial   how much of the signal the centroids discard. It is EXPECTED to be
                        large (~65% of trivial at K=500 on 256-dim ContentVec): most of the
                        discarded variance is prosody/nuisance, not phonetics. Measured:
                        quantized speech stays intelligible and goes flat.
  frames_repeat         fraction of adjacent frames sharing a unit. This is the number that
                        decides whether discretization helps: if it were high, "predict the
                        same token again" would win and the model would ignore its
                        conditioning exactly as it does on continuous features. ~30% on
                        ContentVec@50Hz, so the next-unit task carries real entropy and no
                        run-length dedup (AudioLM/SPEAR-TTS style) is needed -- which also
                        keeps frame alignment with the SMG intact.
"""
import argparse
import glob
import os

import numpy as np
import torch

from megatransformer.utils.codebook import save_codebook


def load_frames(cache_dir: str, max_frames: int):
    """Sample up to max_frames real (non-padded) frames from the shards."""
    shards = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    if not shards:
        raise SystemExit(f"no .pt shards under {cache_dir}")
    chunks, total, runs = [], 0, []
    for p in shards:
        sh = torch.load(p, map_location="cpu", weights_only=False)
        if "features" not in sh:
            raise SystemExit(f"{p} has no 'features' column")
        feats, lens = sh["features"], sh.get("feature_lengths")
        for i in range(len(feats)):
            f = feats[i].float()
            if lens is not None:
                f = f[..., :int(lens[i])]
            if f.shape[-1] < 8:
                continue
            chunks.append(f.T.numpy())          # (T, D)
            total += f.shape[-1]
        if total >= max_frames:
            break
    X = np.concatenate(chunks, 0)
    return X[:max_frames], chunks


def main():
    ap = argparse.ArgumentParser(description="Fit a k-means codebook over cached content features")
    ap.add_argument("--cache_dir", required=True, help="TRAIN shard dir (fit on train only)")
    ap.add_argument("--k", type=int, required=True, help="Codebook size (units)")
    ap.add_argument("--output", required=True, help="Where to write the codebook .pt")
    ap.add_argument("--max_frames", type=int, default=400_000,
                    help="Frames sampled for fitting. 400k is ample for K<=2000.")
    ap.add_argument("--batch_size", type=int, default=8192, help="MiniBatchKMeans batch size")
    ap.add_argument("--max_iter", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sklearn.cluster import MiniBatchKMeans

    X, utts = load_frames(args.cache_dir, args.max_frames)
    print(f"fitting k={args.k} on {X.shape[0]} frames x {X.shape[1]} dims from {args.cache_dir}")

    km = MiniBatchKMeans(n_clusters=args.k, batch_size=args.batch_size, n_init=3,
                         max_iter=args.max_iter, random_state=args.seed).fit(X)
    C = torch.from_numpy(km.cluster_centers_).float()

    trivial = float(np.abs(X).mean())
    quant_l1 = float(np.abs(X - km.cluster_centers_[km.predict(X)]).mean())

    reps = tot = 0
    for u in utts[:400]:
        lab = km.predict(u)
        reps += int((lab[1:] == lab[:-1]).sum())
        tot += len(lab) - 1
    repeat = reps / tot if tot else float("nan")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    save_codebook(args.output, C, meta={
        "source_cache_dir": args.cache_dir, "max_frames": int(X.shape[0]),
        "seed": args.seed, "quant_l1": quant_l1, "trivial_l1": trivial,
        "frames_repeat": repeat,
    })

    print(f"  quant_l1      = {quant_l1:.4f}  ({100 * quant_l1 / trivial:.0f}% of trivial {trivial:.4f})")
    print(f"  frames_repeat = {100 * repeat:.1f}%   (low => next-unit prediction needs the "
          f"conditioning, which is the whole point; high => dedup needed)")
    print(f"  wrote {args.output}  ({args.k} x {C.shape[1]})")


if __name__ == "__main__":
    main()
