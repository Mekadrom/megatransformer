"""Write a checkpoint's EMA shadow weights into a NEW checkpoint dir, loadable by any eval.

Why: `ema_state.pt` holds `shadow_params` for the TRAINABLE tensors only (186 of 484 on the
world-voice runs -- the rest are frozen SmolLM2 / the voice codebook), so it is not a
drop-in state dict. This copies the source checkpoint, overwrites exactly the shadowed
tensors in `pytorch_model.bin`, and leaves everything else (frozen weights, config) alone.

Use: EMA is a noise-reduced view of the SAME weights, so an EMA-vs-raw comparison on the
free-running diagnostics is a cheap predictor of what an LR anneal would buy. If removing
optimizer noise already improves adj_repeat_rate / len_mean, the anneal should help; if EMA
looks identical to raw, the pathology is structural and annealing will not rescue it.

Non-destructive: refuses to write into an existing non-empty directory.

  uv run python scripts_local/materialize_ema_checkpoint.py \
      --checkpoint runs/world_voice/<run>/checkpoint-40000 \
      --out_dir    runs/world_voice/<run>/checkpoint-40000-ema
"""
import argparse, os, shutil
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    if os.path.isdir(a.out_dir) and os.listdir(a.out_dir):
        raise SystemExit(f"{a.out_dir} exists and is not empty -- refusing to overwrite")

    ema_path = os.path.join(a.checkpoint, "ema_state.pt")
    raw_path = os.path.join(a.checkpoint, "pytorch_model.bin")
    for p in (ema_path, raw_path):
        if not os.path.isfile(p):
            raise SystemExit(f"missing {p}")

    ema = torch.load(ema_path, map_location="cpu", weights_only=False)
    shadow = ema["shadow_params"]
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)

    missing = [k for k in shadow if k not in raw]
    if missing:
        raise SystemExit(f"{len(missing)} shadow keys absent from pytorch_model.bin, e.g. {missing[:3]}")
    mismatch = [k for k in shadow if tuple(shadow[k].shape) != tuple(raw[k].shape)]
    if mismatch:
        raise SystemExit(f"{len(mismatch)} shape mismatches, e.g. {mismatch[:3]}")

    n_repl = 0
    for k, v in shadow.items():
        raw[k] = v.clone()          # clone: a shadow slice would pickle its whole storage
        n_repl += 1

    print(f"EMA step={ema.get('step')} decay={ema.get('decay')}")
    print(f"replacing {n_repl}/{len(raw)} tensors ({n_repl/len(raw):.1%} of the state dict; "
          f"the remainder are frozen)")
    if a.dry_run:
        print("(dry run -- nothing written)")
        return

    os.makedirs(a.out_dir, exist_ok=True)
    for f in os.listdir(a.checkpoint):
        if f in ("pytorch_model.bin", "optimizer.pt", "ema_state.pt", "rng_state.pth"):
            continue                # skip the big optimizer/RNG state: eval never reads it
        src = os.path.join(a.checkpoint, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(a.out_dir, f))
    torch.save(raw, os.path.join(a.out_dir, "pytorch_model.bin"))
    print(f"wrote {a.out_dir}")


if __name__ == "__main__":
    main()
