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


def fit_f0_stats(cache_dirs):
    """Per-speaker log-F0 mean/std over the whole TRAIN split.

    Shipped inside the codebook so there is ONE artifact both models share and neither can
    half-update: unit ids are meaningless without the centroids that define them, and a
    contour is meaningless without the speaker statistics that denormalize it.

    Per SPEAKER, not per utterance -- see utils.codebook.normalize_f0 for why.

    Scan EVERY split, not just train. Speakers are typically disjoint across splits (0/40
    overlap on LibriTTS-R), so a train-only table sends every val speaker to the global
    fallback: their contour then carries the very speaker offset normalization exists to
    remove, and eval F0 metrics measure how far that speaker sits from the corpus mean --
    something the model cannot and should not predict.

    This is not meaningful leakage. Per-speaker F0 mean/std is a preprocessing constant
    used only to CONSTRUCT training targets; the model never sees it, and nothing looks it
    up at inference (the SMG denormalizes from ECAPA, which it learned on train speakers
    and which generalizes to unseen voices).
    """
    from collections import defaultdict
    sums, sqs, ns = defaultdict(float), defaultdict(float), defaultdict(float)
    paths = []
    for d in cache_dirs:
        paths.extend(sorted(glob.glob(os.path.join(d, "*.pt"))))
    for path in paths:
        sh = torch.load(path, map_location="cpu", weights_only=False)
        if "f0" not in sh or "speaker_ids" not in sh:
            return None
        f0s, vuvs, sids, lens = sh["f0"], sh["vuv"], sh["speaker_ids"], sh["feature_lengths"]
        for i in range(len(f0s)):
            L = int(lens[i])
            s_id = int(sids[i])
            f, v = f0s[i][:L].double(), vuvs[i][:L].double()
            # Weight by voicing: F0 is undefined where unvoiced, and those frames are
            # stored as 0, which would drag every mean toward zero if counted.
            w = float(v.sum())
            if w < 5:
                continue
            sums[s_id] += float((f * v).sum())
            sqs[s_id] += float(((f ** 2) * v).sum())
            ns[s_id] += w
    if not ns:
        return None

    S = max(ns) + 1
    g_mean = sum(sums.values()) / sum(ns.values())
    g_var = sum(sqs.values()) / sum(ns.values()) - g_mean ** 2
    g_std = max(1e-3, g_var ** 0.5)

    mean = torch.full((S,), g_mean, dtype=torch.float32)
    std = torch.full((S,), g_std, dtype=torch.float32)
    n_spk = 0
    for s_id, n in ns.items():
        # Threshold deliberately low. Falling back to the global mean is not a neutral
        # default -- it injects the speaker's whole offset into the contour (measured: a
        # 187-frame val speaker got 1.6 sigma of error that way). A mean over ~50 voiced
        # frames is worth roughly +-0.1 by comparison, since pitch varies slowly and
        # adjacent frames are highly correlated. A noisy estimate beats a biased one.
        if n < 50:
            continue
        mu = sums[s_id] / n
        var = max(1e-8, sqs[s_id] / n - mu * mu)
        mean[s_id] = mu
        std[s_id] = max(1e-3, var ** 0.5)
        n_spk += 1
    return mean, std, g_mean, g_std, n_spk, len(ns)


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
    ap.add_argument("--f0_stats_cache_dirs", nargs="*", default=None, metavar="DIR",
                    help="Extra shard dirs to include when computing per-speaker F0 stats (e.g. "
                         "the val split). k-means still fits on --cache_dir ALONE. Splits usually "
                         "have disjoint speakers, so without this every val speaker falls back to "
                         "the global mean and eval F0 metrics measure the speaker offset rather "
                         "than the model. Not meaningful leakage: the stats only build training "
                         "targets and are never consulted at inference.")
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

    meta = {
        "source_cache_dir": args.cache_dir, "max_frames": int(X.shape[0]),
        "seed": args.seed, "quant_l1": quant_l1, "trivial_l1": trivial,
        "frames_repeat": repeat,
    }

    stats_dirs = [args.cache_dir] + list(args.f0_stats_cache_dirs or [])
    stats = fit_f0_stats(stats_dirs)
    if stats is not None:
        mean, std, g_mean, g_std, n_spk, n_seen = stats
        meta.update({"speaker_f0_mean": mean, "speaker_f0_std": std,
                     "global_f0_mean": g_mean, "global_f0_std": g_std})
        import numpy as _np
        est = mean[mean != g_mean]
        print(f"  f0 stats from : {', '.join(stats_dirs)}")
        print(f"  f0 stats      = {n_spk}/{n_seen} speakers estimated "
              f"(rest fall back to global mu={g_mean:.3f} sd={g_std:.3f})")
        if len(est):
            print(f"                  between-speaker spread of mean log-F0 = {float(est.std()):.3f} "
                  f"(this is the component the world model cannot know, and that "
                  f"normalization removes)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    save_codebook(args.output, C, meta=meta)

    print(f"  quant_l1      = {quant_l1:.4f}  ({100 * quant_l1 / trivial:.0f}% of trivial {trivial:.4f})")
    print(f"  frames_repeat = {100 * repeat:.1f}%   (low => next-unit prediction needs the "
          f"conditioning, which is the whole point; high => dedup needed)")
    print(f"  wrote {args.output}  ({args.k} x {C.shape[1]})")


if __name__ == "__main__":
    main()
