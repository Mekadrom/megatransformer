"""Is anything in this run still MOVING, or has it converged?

Eyeballing a noisy TensorBoard curve answers "what is the number", not "is it still
improving" -- and on a long fine-tune those are different questions. Adjacent eval points can
swing more than the entire trend, so a single before/after comparison invents trends that are
not there (and misses slow real ones).

For each scalar this fits an OLS slope over the TRAILING fraction of the run and compares the
total drift across that window against the series' OWN short-timescale noise (std of
consecutive diffs / sqrt(2)):

  UP / DOWN   |drift| exceeds the noise floor => a real trend
  --          noise-dominated => plateaued; do NOT read its wiggles as signal

Use it to decide when to stop training: when the eval-side metrics go '--' while the
train-side ones already have, the run has converged and later checkpoints are just noise.

⚠️ A RESUMED run (crash restart, fine-tune continuation) writes a new event file per process,
and the default is NEWEST PROCESS ONLY -- so tags logged before the last resume go MISSING and
the fit covers only the tail segment. Pass --merge for anything that has ever been resumed.

    python -m megatransformer.scripts.tb_trend runs/sive/<run> --merge
    python -m megatransformer.scripts.tb_trend runs/sive/<run> --merge --frac 0.4 --tags eval/,perplexity
"""
import argparse
from collections import defaultdict

import numpy as np

from megatransformer.scripts.read_tensorboard import _load, _resolve_files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="a run dir or a specific .tfevents file")
    ap.add_argument("--frac", type=float, default=0.5,
                    help="trailing fraction of the run to fit (0.5 = second half)")
    ap.add_argument("--tags", default="", help="comma-separated substrings to filter tags")
    ap.add_argument("--min_points", type=int, default=6, help="skip series shorter than this")
    ap.add_argument("--clip", type=float, default=1e6,
                    help="drop |value| above this (step-0 init spikes wreck the fit)")
    ap.add_argument("--merge", action="store_true",
                    help="read ALL event files in the dir (default: newest process only)")
    ap.add_argument("--file", dest="file_substr", default="",
                    help="only read event files whose name contains this")
    args = ap.parse_args()

    scalars, _ = _load(_resolve_files(args.path, args.file_substr, args.merge))
    series = defaultdict(list)
    for tag, pts in scalars.items():
        series[tag] = sorted(set(pts))

    wanted = [w for w in args.tags.split(",") if w]
    rows = []
    for tag, pts in sorted(series.items()):
        if wanted and not any(w in tag for w in wanted):
            continue
        s = np.array([p[0] for p in pts], float)
        v = np.array([p[1] for p in pts], float)
        keep = np.isfinite(v) & (np.abs(v) < args.clip)
        s, v = s[keep], v[keep]
        if len(s) < args.min_points:
            continue
        cut = s.min() + (1 - args.frac) * (s.max() - s.min())
        m = s >= cut
        if m.sum() < 5:
            continue
        sw, vw = s[m], v[m]
        slope_per_10k = np.polyfit(sw, vw, 1)[0] * 10000.0
        drift = slope_per_10k * (sw.max() - sw.min()) / 10000.0
        noise = float(np.std(np.diff(vw)) / np.sqrt(2)) if len(vw) > 2 else 0.0
        k = max(1, len(vw) // 8)
        direction = "--" if abs(drift) <= max(noise, 1e-12) else ("UP" if drift > 0 else "DOWN")
        rows.append((tag, len(s), vw.mean(), vw[:k].mean(), vw[-k:].mean(), drift, noise, direction))

    if not rows:
        raise SystemExit("no series matched (try --merge, or loosen --tags/--min_points)")
    print(f"\n=== trend over trailing {args.frac:.0%} of run: {args.path.rstrip('/').split('/')[-1]} ===")
    print(f"{'tag':<42}{'n':>6}{'mean':>11}{'start':>11}{'end':>11}{'drift':>11}{'noise':>10}  dir")
    for r in rows:
        print(f"{r[0]:<42}{r[1]:>6}{r[2]:>11.4g}{r[3]:>11.4g}{r[4]:>11.4g}"
              f"{r[5]:>+11.3g}{r[6]:>10.3g}  {r[7]}")
    print("\ndir: UP/DOWN = drift exceeds short-timescale noise (real trend); "
          "-- = plateaued/noise-dominated")


if __name__ == "__main__":
    main()
