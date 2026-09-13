"""Widen a unistream checkpoint's voice unit head so a BISTREAM run can warm-start from it.

Bistream adds `fill_token` (id K+1) to end a chunk, so the head goes K+1 -> K+2 rows
(6562 -> 6563 for CosyVoice 2). Nothing in the loader widens it: load_model's
allow_size_mismatch DROPS shape-mismatched keys, which would silently reinitialise the ENTIRE
unit head and discard everything the model learned about unit prediction -- a several-day
retrain disguised as a warm start, with no error.

This copies the checkpoint and appends one row. The new row is NOT random: it is initialised
from the EOV row, scaled down. fill_token and EOV play the same structural part (both end a
span and hand control back), so EOV's direction is a far better prior than noise -- while the
scale keeps it from outcompeting EOV before it has learned anything of its own.

  uv run python scripts_local/widen_unit_head_for_bistream.py \
      --checkpoint runs/world_voice/<run>/checkpoint-78000 \
      --out_dir    runs/world_voice/<run>/checkpoint-78000-bistream
"""
import argparse, os, shutil
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--head_prefix", default="voice_generator.unit_head")
    ap.add_argument("--init_scale", type=float, default=0.1,
                    help="Scale on the copied EOV row. 0 = zeros, 1 = an exact EOV duplicate "
                         "(which would make fill and EOV indistinguishable at init).")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    src = os.path.join(a.checkpoint, "pytorch_model.bin")
    if not os.path.isfile(src):
        raise SystemExit(f"no pytorch_model.bin in {a.checkpoint}")
    if os.path.isdir(a.out_dir) and os.listdir(a.out_dir):
        raise SystemExit(f"{a.out_dir} exists and is not empty -- refusing to overwrite")

    sd = torch.load(src, map_location="cpu", weights_only=False)
    wk, bk = a.head_prefix + ".weight", a.head_prefix + ".bias"
    for k in (wk, bk):
        if k not in sd:
            avail = sorted(x for x in sd if "unit_head" in x)
            raise SystemExit(f"{k} not in checkpoint. unit_head keys present: {avail}")

    W, B = sd[wk], sd[bk]
    K_plus_1 = W.shape[0]
    print(f"unit head: {tuple(W.shape)} / bias {tuple(B.shape)}  (EOV id = {K_plus_1 - 1})")
    if K_plus_1 % 2 == 0 and K_plus_1 >= 6562:
        pass  # informational only; no assumption about K
    eov_w, eov_b = W[-1:].clone(), B[-1:].clone()
    new_w = torch.cat([W, eov_w * a.init_scale], dim=0)
    new_b = torch.cat([B, eov_b * a.init_scale], dim=0)
    print(f"      -> {tuple(new_w.shape)} / bias {tuple(new_b.shape)}  "
          f"(fill_token id = {K_plus_1}, init = EOV row x {a.init_scale})")
    print(f"      new row norm {new_w[-1].norm():.4f} vs EOV row {W[-1].norm():.4f}")

    if a.dry_run:
        print("(dry run -- nothing written)")
        return

    sd[wk], sd[bk] = new_w, new_b
    os.makedirs(a.out_dir, exist_ok=True)
    # Carry the trainer/RNG state so the warm start resumes cleanly; the optimizer state for
    # the head is now the wrong shape, so DROP it -- a stale moment for a resized tensor is
    # worse than starting the head's moments fresh.
    for f in os.listdir(a.checkpoint):
        if f in ("pytorch_model.bin", "optimizer.pt", "ema_state.pt"):
            continue
        p = os.path.join(a.checkpoint, f)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(a.out_dir, f))
    torch.save(sd, os.path.join(a.out_dir, "pytorch_model.bin"))

    # The EMA shadow holds its OWN copy of every trainable tensor, so it needs the same
    # surgery -- otherwise --use_ema shape-mismatches the moment it applies the shadow, or
    # (worse) silently skips the head and evaluates a half-averaged model.
    ema_src = os.path.join(a.checkpoint, "ema_state.pt")
    if os.path.isfile(ema_src):
        ema = torch.load(ema_src, map_location="cpu", weights_only=False)
        shadow = ema.get("shadow_params", {})
        done = []
        for k, grow in ((wk, new_w), (bk, new_b)):
            if k in shadow and shadow[k].shape[0] == K_plus_1:
                tail = shadow[k][-1:].clone() * a.init_scale
                shadow[k] = torch.cat([shadow[k], tail], dim=0)
                done.append(f"{k} -> {tuple(shadow[k].shape)}")
        torch.save(ema, os.path.join(a.out_dir, "ema_state.pt"))
        print("      EMA shadow widened: " + (", ".join(done) if done else "nothing to do"))
    print(f"wrote {a.out_dir}  (optimizer.pt intentionally omitted -- head was resized)")


if __name__ == "__main__":
    main()
