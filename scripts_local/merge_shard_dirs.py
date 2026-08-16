"""Merge multiple preprocessed shard dirs into one.

Two modes:

  Default (hardlink): gather shard_*.pt from each --src dir (in order) and re-lay
  them into --output_dir as a contiguous shard_000000.. sequence, hardlinked
  (instant, no extra disk on the same filesystem; --copy forces a copy). Shard
  CONTENTS are untouched — this does NOT resize shards.

  Repack (--samples_per_shard N): actually open the shards, concatenate their
  samples across all sources, and re-emit new shards of ~N samples each (token_ids
  are re-padded per new shard since each source shard pads to its own width). Use
  this to turn many tiny lean caption shards into fewer larger ones in one pass.
  Assumes the sources share a schema (fine for the lean image-caption sets).

After either, run stat-shards on --output_dir to build shard_index.json.

Usage:
    # merge only (relink)
    python scripts_local/merge_shard_dirs.py --src A B C --output_dir OUT
    # merge + resize to ~16k samples/shard
    python scripts_local/merge_shard_dirs.py --src A B C --output_dir OUT --samples_per_shard 16000
"""

import argparse
import glob
import os
import shutil


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", nargs="+", required=True, help="Source shard dirs, in merge order.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--copy", action="store_true", help="Force copy instead of hardlink (relink mode).")
    p.add_argument("--max_shards", type=int, default=0, help="Cap total shards written (relink mode; 0=all).")
    p.add_argument("--samples_per_shard", type=int, default=0,
                   help="If >0, REPACK: re-emit new shards of this many samples (merge + resize).")
    p.add_argument("--pad_id", type=int, default=0, help="Pad id for re-padding token_ids in repack mode.")
    return p.parse_args()


def relink(args):
    os.makedirs(args.output_dir, exist_ok=True)
    idx = 0
    for src in args.src:
        shards = sorted(glob.glob(os.path.join(src, "shard_*.pt")))
        if not shards:
            print(f"WARNING: no shards in {src}"); continue
        for s in shards:
            if args.max_shards and idx >= args.max_shards:
                print(f"reached --max_shards {args.max_shards}, stopping"); break
            dst = os.path.join(args.output_dir, f"shard_{idx:06d}.pt")
            if os.path.exists(dst):
                os.remove(dst)
            if args.copy:
                shutil.copy2(s, dst)
            else:
                try:
                    os.link(s, dst)
                except OSError:
                    shutil.copy2(s, dst)
            idx += 1
        print(f"  {src}: added {len(shards)} shards")
    print(f"\nmerged {idx} shards into {args.output_dir}")


def repack(args):
    import torch
    import torch.nn.functional as F
    N = args.samples_per_shard
    os.makedirs(args.output_dir, exist_ok=True)
    buf_text, buf_ids, buf_lens = [], [], []
    extra = {}                # per-sample tensor fields (latents/mu/logvar/images)
    state = {"dummy": None, "out": 0, "total": 0}

    def emit(count):
        rows, lens, texts = buf_ids[:count], buf_lens[:count], buf_text[:count]
        mx = max(lens) if lens else 1
        padded = torch.stack([r if r.numel() == mx else F.pad(r, (0, mx - r.numel()), value=args.pad_id)
                              for r in rows])
        shard = {"num_samples": count, "text": list(texts),
                 "token_ids": padded, "text_lengths": torch.tensor(lens, dtype=torch.long)}
        if state["dummy"] is not None:
            shard["dummy_latent_shape"] = state["dummy"]
        for k, rowsk in extra.items():
            shard[k] = torch.stack(rowsk[:count])
        dst = os.path.join(args.output_dir, f"shard_{state['out']:06d}.pt")
        tmp = dst + ".tmp"; torch.save(shard, tmp); os.replace(tmp, dst)
        state["out"] += 1; state["total"] += count
        del buf_text[:count]; del buf_ids[:count]; del buf_lens[:count]
        for k in extra:
            del extra[k][:count]

    for src in args.src:
        shards = sorted(glob.glob(os.path.join(src, "shard_*.pt")))
        if not shards:
            print(f"WARNING: no shards in {src}"); continue
        for s in shards:
            sh = torch.load(s, map_location="cpu", weights_only=False)
            n = sh["num_samples"]
            texts = sh.get("text", [""] * n)
            ids, lens = sh["token_ids"], sh["text_lengths"]
            psf = [k for k, v in sh.items()
                   if hasattr(v, "shape") and getattr(v, "ndim", 0) >= 1
                   and v.shape[0] == n and k not in ("token_ids", "text_lengths")]
            for k in psf:
                extra.setdefault(k, [])
            if "dummy_latent_shape" in sh:
                state["dummy"] = sh["dummy_latent_shape"]
            for i in range(n):
                L = int(lens[i])
                buf_text.append(texts[i]); buf_ids.append(ids[i][:L].clone().long()); buf_lens.append(L)
                for k in psf:
                    extra[k].append(sh[k][i].clone())
            while len(buf_ids) >= N:
                emit(N)
        print(f"  {src}: {len(shards)} shards read")
    if buf_ids:
        emit(len(buf_ids))
    print(f"\nrepacked {state['total']:,} samples into {state['out']} shards (~{N}/shard) in {args.output_dir}")


def main():
    args = parse_args()
    if args.samples_per_shard > 0:
        repack(args)
    else:
        relink(args)
    print(f"NEXT: python -m megatransformer.scripts.data.preprocess_dataset stat-shards --output_dir {args.output_dir}")


if __name__ == "__main__":
    main()
