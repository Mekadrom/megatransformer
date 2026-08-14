"""Backfill train/vq_perplexity_frac into existing SIVE run dirs.

Reads each run's already-logged train/vq_perplexity scalar series, divides by the codebook
size K, and writes the normalized series (train/vq_perplexity_frac) to a NEW events file in
the SAME run directory. TensorBoard merges all events files in a dir, so the normalized
curve shows up retroactively alongside the original — no retraining, no touching existing
files. (Matches the metric now logged live by training.py.)

Usage:
  python scripts_local/backfill_vq_perplexity_frac.py <run_dir>=<K> [<run_dir>=<K> ...]
    e.g. ... runs/sive/..._k250_...=250 runs/sive/..._k384_...=384
"""
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

SRC = "train/vq_perplexity"
DST = "train/vq_perplexity_frac"


def backfill(run_dir, K):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if SRC not in ea.Tags().get("scalars", []):
        print(f"  SKIP {run_dir}: no {SRC}")
        return
    if DST in ea.Tags().get("scalars", []):
        print(f"  SKIP {run_dir}: {DST} already present (not double-writing)")
        return
    pts = ea.Scalars(SRC)
    mx = max(p.value for p in pts)
    if mx > K:
        raise SystemExit(f"  {run_dir}: max {SRC}={mx:.1f} > K={K} — wrong K, aborting")
    w = SummaryWriter(log_dir=run_dir)
    for p in pts:
        w.add_scalar(DST, p.value / K, p.step)
    w.close()
    print(f"  {run_dir}\n    wrote {len(pts)} {DST} pts | K={K} | "
          f"frac range {min(p.value for p in pts)/K:.3f}–{mx/K:.3f} "
          f"| steps {pts[0].step}–{pts[-1].step}")


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    for a in args:
        if "=" not in a:
            raise SystemExit(f"expected <run_dir>=<K>, got: {a}")
        run_dir, k = a.rsplit("=", 1)
        backfill(run_dir, int(k))


if __name__ == "__main__":
    main()
