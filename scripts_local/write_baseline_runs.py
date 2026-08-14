"""Write flat-line baseline "runs" beside a training run so voice_unit_accuracy
reference lines OVERLAY on the SAME TensorBoard chart.

TensorBoard overlays scalars that share a tag ACROSS runs. So each baseline is written as
its own sibling run directory logging the SAME tag (default train/voice_unit_accuracy) as a
constant at step 0 and step max_step -> a flat horizontal line spanning the accuracy chart,
labeled by the run's name. Point TB at the level that contains BOTH the training run and
these siblings (here runs/world, per tensorboard.sh) and toggle the "__baseline_*" runs on.

Values come from scripts_local/ngram_unit_baseline.py (codebook-specific).

Usage:
  python scripts_local/write_baseline_runs.py runs/world/<run> \
      --repeat 0.117 --ngram_ceiling 0.211 --asymptote 0.229 [--max_step 100000]
"""
import argparse
import os

from torch.utils.tensorboard import SummaryWriter


def write_line(parent, base, suffix, tags, value, max_step):
    d = os.path.join(parent, f"{base}__baseline_{suffix}")
    os.makedirs(d, exist_ok=True)
    w = SummaryWriter(log_dir=d)
    # Log the constant under EVERY requested tag so it overlays on each accuracy chart
    # (train AND held-out eval -- the eval chart is the rigorous held-out-vs-held-out read).
    # Points at 0, mid, max render a flat line across the whole x-range even under TB
    # down-sampling.
    for tag in tags:
        for step in (0, max_step // 2, max_step):
            w.add_scalar(tag, value, step)
    w.close()
    print(f"  {d}   {value}  on {', '.join(tags)}  over [0, {max_step}]")


def main():
    ap = argparse.ArgumentParser(description="Overlay baseline reference lines as sibling TB runs")
    ap.add_argument("run_dir", help="training run dir, e.g. runs/world/world_vqsive_k250_20hz_ar_test_0")
    ap.add_argument("--tag", nargs="+",
                    default=["train/voice_unit_accuracy", "eval/voice_synthesis/voice_unit_accuracy"],
                    help="Scalar tag(s) to overlay on (must match the training run's tags exactly). "
                         "Defaults to both the train and held-out eval unit-accuracy charts.")
    ap.add_argument("--repeat", type=float, default=None, help="repeat-previous-unit AR-crutch baseline")
    ap.add_argument("--ngram_ceiling", type=float, default=None, help="best backed-off n-gram top-1 (held out)")
    ap.add_argument("--asymptote", type=float, default=None, help="highest-order n-gram seen-only top-1")
    ap.add_argument("--max_step", type=int, default=100000, help="span the line to this step (= training max_steps)")
    args = ap.parse_args()

    run_dir = args.run_dir.rstrip("/")
    if not os.path.isdir(run_dir):
        raise SystemExit(f"run_dir not found: {run_dir}")
    parent, base = os.path.dirname(run_dir), os.path.basename(run_dir)
    lines = [("repeat", args.repeat), ("ngram_ceiling", args.ngram_ceiling), ("ngram_asymptote", args.asymptote)]
    lines = [(s, v) for s, v in lines if v is not None]
    if not lines:
        raise SystemExit("Nothing to write -- pass at least one of --repeat/--ngram_ceiling/--asymptote.")

    print(f"Writing {len(lines)} baseline overlay run(s) beside {run_dir}:")
    for suffix, value in lines:
        write_line(parent, base, suffix, args.tag, value, args.max_step)
    print(f"\nDone. In TB (logdir {parent}), the '{base}__baseline_*' runs overlay flat lines")
    print(f"on {' and '.join(args.tag)}. Toggle them on in the run selector.")


if __name__ == "__main__":
    main()
