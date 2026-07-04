"""Quantify how 'dead' the SMG's speaker-conditioning (FiLM) path is, by weight magnitude.

The cross-speaker collapse (embedding ignored — see embedding_control.py) should leave a
fingerprint in the weights: the FiLM / speaker-embedding path either (a) never grew from
its small init (never learned to modulate on the embedding) or (b) grew but is overridden
downstream. This script separates the model's parameters into the speaker/FiLM path, the
F0 path, and the backbone, and reports each group's weight RMS against BOTH the rest of the
model and a freshly-initialized model of the same config — then a per-tensor trained-vs-init
breakdown so you can see WHICH sub-layers are stuck at init (dead) vs which moved.

FiLM output layers init at std=0.02 (smg.py `_init_film_projection`), so a trained/init
ratio near 1.0 on those = the FiLM never learned to condition on the embedding.

No forward pass / data needed — just loads the checkpoint + a fresh model and compares
weights, so it's fast and runs on CPU (leaves the GPUs for training).
"""
import argparse
import torch

from megatransformer.model.smg.smg import SMG
from megatransformer.utils.model_loading_utils import load_model

# Substrings that identify each path in named_parameters().
SPEAKER_KEYS = ("speaker_embedding_projection", "early_film_projection", "speaker_projections_2d")
F0_KEYS = ("f0_to_2d_projection", "f0_pred", "f0_predictor", "f0_conditioning", "f0_embedding")


def _group(name: str) -> str:
    if any(k in name for k in SPEAKER_KEYS):
        return "speaker_film"
    if any(k in name for k in F0_KEYS):
        return "f0"
    return "backbone"


def _rms(t: torch.Tensor) -> float:
    return float(t.detach().float().pow(2).mean().sqrt())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="SMG speaker-conditioning (FiLM) weight-health / what's-dying analysis")
    ap.add_argument("--checkpoint", required=True, help="SMG checkpoint dir")
    ap.add_argument("--config", default="medium_decoder_only_1d_3x")
    ap.add_argument("--sive_encoder_dim", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--near_init_tol", type=float, default=0.15,
                    help="|trained/init - 1| <= tol counts a tensor as 'at init' (dead)")
    args = ap.parse_args()

    ov = {"sive_encoder_dim": args.sive_encoder_dim}
    trained = load_model(SMG, args.config, checkpoint_path=args.checkpoint, overrides=ov, device=args.device).eval()
    fresh = load_model(SMG, args.config, checkpoint_path=None, overrides=ov, device=args.device).eval()
    tp = dict(trained.named_parameters())
    fp = dict(fresh.named_parameters())

    # Group-level aggregate RMS (trained + init).
    groups: dict[str, dict] = {}
    for name, p in tp.items():
        d = groups.setdefault(_group(name), {"sumsq": 0.0, "n": 0, "sumsq_init": 0.0})
        d["sumsq"] += float(p.detach().float().pow(2).sum())
        d["n"] += p.numel()
        if name in fp:
            d["sumsq_init"] += float(fp[name].detach().float().pow(2).sum())

    def grp_rms(g, key="sumsq"):
        d = groups[g]
        return (d[key] / max(d["n"], 1)) ** 0.5

    print(f"\n=== SMG conditioning-path weight health @ {args.checkpoint} ===")
    print(f"{'group':14s} {'params':>11s} {'RMS(trained)':>13s} {'RMS(init)':>11s} {'trained/init':>13s}")
    for g in ("speaker_film", "f0", "backbone"):
        if g not in groups:
            continue
        rt, ri = grp_rms(g), grp_rms(g, "sumsq_init")
        print(f"{g:14s} {groups[g]['n']:>11,d} {rt:>13.4f} {ri:>11.4f} {rt / max(ri, 1e-12):>13.3f}")

    if "speaker_film" in groups and "backbone" in groups:
        print(f"\nspeaker_film RMS / backbone RMS (trained) = {grp_rms('speaker_film') / max(grp_rms('backbone'), 1e-12):.3f}   "
              f"(how big the conditioning weights are vs the rest of the model)")

    # Per-tensor breakdown of the speaker/FiLM path — the "what exactly is dying" view.
    print(f"\n--- speaker/FiLM path, per-tensor (trained vs init RMS; x = trained/init) ---")
    rows = []
    for name, p in tp.items():
        if _group(name) != "speaker_film":
            continue
        rt = _rms(p)
        ri = _rms(fp[name]) if name in fp else float("nan")
        ratio = rt / ri if (ri == ri and ri > 0) else float("nan")
        rows.append((name, tuple(p.shape), rt, ri, ratio))
    rows.sort(key=lambda r: (r[4] if r[4] == r[4] else 1e9))  # most-dead (near 1.0) sorted by ratio
    for name, shape, rt, ri, ratio in rows:
        dead = ratio == ratio and abs(ratio - 1.0) <= args.near_init_tol
        flag = "  <-- AT INIT (dead)" if dead else ("  (grew)" if ratio == ratio and ratio > 1 + args.near_init_tol else "")
        print(f"  {name:54s} {str(shape):>16s}  trained {rt:.4f}  init {ri:.4f}  x{ratio:.2f}{flag}")

    ratios = [r[4] for r in rows if r[4] == r[4]]
    near = sum(1 for x in ratios if abs(x - 1.0) <= args.near_init_tol)
    sf = grp_rms("speaker_film") / max(grp_rms("speaker_film", "sumsq_init"), 1e-12) if "speaker_film" in groups else float("nan")
    bb = grp_rms("backbone") / max(grp_rms("backbone", "sumsq_init"), 1e-12) if "backbone" in groups else float("nan")
    print(f"\nsummary: {near}/{len(ratios)} speaker/FiLM tensors within {args.near_init_tol:.0%} of init; "
          f"speaker_film grew x{sf:.2f} vs backbone x{bb:.2f} (both trained/init).")
    if sf >= bb:
        print("Verdict: conditioning path is ALIVE — it grew at least as much as the backbone. So the "
              "cross-speaker collapse is FUNCTIONAL, not structural: the FiLM is trained and DOES modulate "
              "the output, but that modulation doesn't carry SPEAKER identity (the leaky SIVE features "
              "supply identity regardless). The lever is features-side (less leakage) or a contrastive loss "
              "that forces emb-swaps to change IDENTITY — not reviving weights, which are already grown. "
              "NB: this means a purely output-diff contrastive loss risks amplifying the (already large) "
              "non-identity modulation instead of fixing identity.")
    else:
        print("Verdict: conditioning path UNDER-grew the backbone (weights closer to init) => the decoder "
              "learned to reconstruct largely WITHOUT the embedding; the path atrophied toward init.")


if __name__ == "__main__":
    main()
