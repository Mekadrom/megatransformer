"""Read scalars / text / tags straight from TensorBoard .tfevents files.

A small living helper so we don't hand-write an EventAccumulator snippet every time.
Handles the annoying cases: a run dir with MULTIPLE event files (multiple processes,
resumes, or copied-in history) and huge stale files that make a whole-dir load time out.

By default it reads only the NEWEST process's event files (all files sharing the newest
`<unixtime>` in the filename), which is almost always the run you care about and avoids
merging gigabytes of old logs. Use --merge to read every file in the dir, or --file
SUBSTR to target specific files (e.g. a PID).

Examples:
  # list scalar tags matching a substring
  python -m megatransformer.scripts.read_tensorboard runs/smg/<run> --list d_

  # latest value of every GAN scalar
  python -m megatransformer.scripts.read_tensorboard runs/smg/<run> --last d_ g_

  # dump a subsampled series (step,value) for one or more tags
  python -m megatransformer.scripts.read_tensorboard runs/smg/<run> --scalar train/d_real_mean train/d_fake_mean --points 20

  # step range + all event files merged
  python -m megatransformer.scripts.read_tensorboard runs/smg/<run> --scalar train/loss --steps 50000:52000 --merge

  # print a text summary (e.g. the launch command)
  python -m megatransformer.scripts.read_tensorboard runs/smg/<run> --text
"""
import argparse
import glob
import os
import re

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

_TS = re.compile(r"tfevents\.(\d+)\.")


def _resolve_files(path, file_substr, merge):
    """Return the list of .tfevents files to read for `path` (a file or a dir)."""
    if os.path.isfile(path):
        return [path]
    files = sorted(glob.glob(os.path.join(path, "events.out.tfevents.*")))
    if not files:
        raise SystemExit(f"no .tfevents files under {path}")
    if file_substr:
        files = [f for f in files if file_substr in os.path.basename(f)]
        if not files:
            raise SystemExit(f"no event files matching '{file_substr}' in {path}")
        return files
    if merge:
        return files
    # default: newest process only (all files sharing the max <unixtime> in the name)
    def ts(f):
        m = _TS.search(os.path.basename(f))
        return int(m.group(1)) if m else 0
    newest = max(ts(f) for f in files)
    return [f for f in files if ts(f) == newest]


def _load(files):
    """Merge scalars+tensors across the given event files. Returns (scalars, tensors, ea0)."""
    scalars, tensors = {}, {}
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0, "tensors": 0})  # 0 = keep all
        ea.Reload()
        avail = ea.Tags()
        for t in avail.get("scalars", []):
            scalars.setdefault(t, []).extend((s.step, s.value) for s in ea.Scalars(t))
        for t in avail.get("tensors", []):
            tensors.setdefault(t, ea)  # keep an accessor; text is small
    for t in scalars:
        scalars[t].sort(key=lambda sv: sv[0])
    return scalars, tensors


def _fmt_series(series, points):
    if not points or len(series) <= points:
        return series
    stride = max(1, len(series) // points)
    out = series[::stride]
    if out[-1] != series[-1]:
        out.append(series[-1])
    return out


def main():
    ap = argparse.ArgumentParser(description="Read TensorBoard .tfevents (scalars/text/tags)")
    ap.add_argument("path", help="A run dir or a specific .tfevents file")
    ap.add_argument("--list", nargs="?", const="", metavar="SUBSTR",
                    help="List scalar tags (optionally filter by substring)")
    ap.add_argument("--scalar", nargs="+", metavar="TAG", help="Print (step,value) series for tag(s)")
    ap.add_argument("--last", nargs="*", metavar="SUBSTR",
                    help="Print the latest value of each scalar tag (optional substring filters)")
    ap.add_argument("--text", nargs="?", const="__all__", metavar="TAG",
                    help="Print text summaries (default: all text tags)")
    ap.add_argument("--points", type=int, default=20, help="Max points per series (subsample; 0=all)")
    ap.add_argument("--steps", metavar="A:B", help="Filter to step range A:B (either side optional)")
    ap.add_argument("--merge", action="store_true", help="Read ALL event files in the dir (default: newest process only)")
    ap.add_argument("--file", metavar="SUBSTR", help="Only read event files whose name contains SUBSTR (e.g. a PID)")
    args = ap.parse_args()

    files = _resolve_files(args.path, args.file, args.merge)
    print(f"# reading {len(files)} event file(s): {', '.join(os.path.basename(f) for f in files)}")
    scalars, tensors = _load(files)

    lo, hi = None, None
    if args.steps:
        a, _, b = args.steps.partition(":")
        lo = int(a) if a else None
        hi = int(b) if b else None

    def in_range(step):
        return (lo is None or step >= lo) and (hi is None or step <= hi)

    did = False
    # default action if nothing specified: list all tags
    if not any([args.list is not None, args.scalar, args.last is not None, args.text]):
        args.list = ""

    if args.list is not None:
        did = True
        tags = sorted(t for t in scalars if args.list in t)
        print(f"# {len(tags)} scalar tag(s)" + (f" matching '{args.list}'" if args.list else "") + ":")
        for t in tags:
            n = len(scalars[t])
            s0, s1 = scalars[t][0][0], scalars[t][-1][0]
            print(f"  {t}   (n={n}, steps {s0}..{s1})")
        txt = sorted(tensors)
        if txt:
            print(f"# {len(txt)} text/tensor tag(s): {', '.join(txt)}")

    if args.last is not None:
        did = True
        subs = args.last or [""]
        tags = sorted(t for t in scalars if any(s in t for s in subs))
        print(f"# latest value ({len(tags)} tags):")
        for t in tags:
            series = [sv for sv in scalars[t] if in_range(sv[0])]
            if series:
                st, v = series[-1]
                print(f"  {t:40s} step {st:>8} = {v:.5f}")

    if args.scalar:
        did = True
        for t in args.scalar:
            if t not in scalars:
                cand = [x for x in scalars if t in x]
                print(f"  [tag '{t}' not found]" + (f" did you mean: {cand[:5]}" if cand else ""))
                continue
            series = [sv for sv in scalars[t] if in_range(sv[0])]
            if not series:
                print(f"  [{t}: no points in range]"); continue
            vals = [v for _, v in series]
            print(f"# {t}: n={len(series)} steps {series[0][0]}..{series[-1][0]} "
                  f"min={min(vals):.5f} max={max(vals):.5f} last={vals[-1]:.5f}")
            for st, v in _fmt_series(series, args.points):
                print(f"  {st:>8}  {v:.6f}")

    if args.text:
        did = True
        from tensorboard.util import tensor_util
        want = sorted(tensors) if args.text == "__all__" else [t for t in tensors if args.text in t]
        for t in want:
            ea = tensors[t]
            try:
                ev = ea.Tensors(t)
                raw = tensor_util.make_ndarray(ev[-1].tensor_proto)
                s = raw.item()
                s = s.decode() if isinstance(s, bytes) else str(s)
                print(f"# {t} (step {ev[-1].step}):\n{s}\n")
            except Exception as e:
                print(f"  [{t}: could not decode ({e})]")

    if not did:
        print("# nothing to do")


if __name__ == "__main__":
    main()
