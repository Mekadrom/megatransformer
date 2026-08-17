"""Strip summary values whose tag matches a regex from a TensorBoard .tfevents file.

TB event files are append-only, CRC-framed TFRecords -- there is no in-place delete, so
this REWRITES the file: parse every Event, drop the matching summary values (e.g. the
black dummy-latent images the adapter viz used to log), keep everything else (scalars,
text, other images), and re-frame via tensorboard's RecordWriter. The original is backed
up to <file>.bak.

Reads/writes with the `tensorboard` package only (no tensorflow).

Examples:
  # see what WOULD be dropped (no writes)
  python scripts_local/strip_tb_images.py <event_file> --pattern '^text_to_image/\\d+/target_latent$' --dry-run
  # actually strip
  python scripts_local/strip_tb_images.py <event_file> --pattern '^text_to_image/\\d+/target_latent$'
"""
import argparse
import os
import re
import struct

from tensorboard.compat.proto import event_pb2
from tensorboard.summary.writer.record_writer import RecordWriter


def read_events(path):
    """Yield Event protos from a .tfevents file (TFRecord framing, CRCs skipped)."""
    with open(path, "rb") as f:
        buf = f.read()
    i, n = 0, len(buf)
    while i + 12 <= n:
        length = struct.unpack("<Q", buf[i:i + 8])[0]
        i += 12  # 8-byte length + 4-byte length-CRC
        if i + length + 4 > n:
            break
        data = buf[i:i + length]
        i += length + 4  # payload + 4-byte data-CRC
        ev = event_pb2.Event()
        try:
            ev.ParseFromString(data)
        except Exception:
            break
        yield ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a single .tfevents file")
    ap.add_argument("--pattern", required=True, help="regex; summary values whose tag matches are dropped")
    ap.add_argument("--dry-run", action="store_true", help="report only; do not write")
    args = ap.parse_args()
    pat = re.compile(args.pattern)

    kept, total, dropped_vals, dropped_events, multi = [], 0, 0, 0, 0
    hit_tags = {}
    for ev in read_events(args.path):
        total += 1
        if ev.HasField("summary") and len(ev.summary.value):
            drop = [v for v in ev.summary.value if pat.search(v.tag)]
            if drop:
                for v in drop:
                    hit_tags[v.tag] = hit_tags.get(v.tag, 0) + 1
                dropped_vals += len(drop)
                keep = [v for v in ev.summary.value if not pat.search(v.tag)]
                if len(ev.summary.value) > 1:
                    multi += 1
                if not keep:
                    dropped_events += 1
                    continue  # whole event was just the dropped value(s)
                new = event_pb2.Event()
                new.CopyFrom(ev)
                del new.summary.value[:]
                new.summary.value.extend(keep)
                kept.append(new)
                continue
        kept.append(ev)

    print(f"file: {args.path}")
    print(f"events: {total} read, {len(kept)} kept, {dropped_events} dropped")
    print(f"values dropped: {dropped_vals}  (multi-value events touched: {multi})")
    print("dropped tags:")
    for t, c in sorted(hit_tags.items()):
        print(f"  {t}  x{c}")
    if args.dry_run:
        print("\n[dry-run] no changes written.")
        return
    if dropped_vals == 0:
        print("\nnothing matched; leaving file untouched.")
        return

    # Backup name MUST NOT contain "tfevents" -- TB loads any basename with that
    # substring, so a "<file>.bak" backup would re-surface the stripped images.
    d, base = os.path.split(args.path)
    bak = os.path.join(d, base.replace("tfevents", "stripped_bak") + ".bak")
    tmp = args.path + ".tmp"
    with open(tmp, "wb") as f:
        w = RecordWriter(f)
        for ev in kept:
            w.write(ev.SerializeToString())
    os.rename(args.path, bak)     # back up original
    os.replace(tmp, args.path)    # cleaned file takes the original name (TB finds it)
    print(f"\nwrote cleaned file: {args.path}\nbackup: {bak}")


if __name__ == "__main__":
    main()
