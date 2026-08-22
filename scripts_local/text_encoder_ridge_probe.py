"""Is the frozen SmolLM2-135M spine contributing PRETRAINING, or just random features?

The world-image path's whole job is prompt -> Qwen3-4B conditioning. So fit the best possible
LINEAR map from each candidate text-feature source to the whitened Qwen3 target and compare R².
Ridge is closed form -- one matrix solve, seconds on CPU -- so this costs a single forward pass of
each encoder, NOT a training run.

ARMS (the middle one is the whole point):
  pretrained   frozen SmolLM2-135M, last hidden state          <- what the model uses today
  random       SAME architecture, RANDOM weights, frozen       <- controls for random-feature
                                                                  expansion: if this ties
                                                                  `pretrained`, the PRETRAINING is
                                                                  inert and the spine is a crutch
  embed_only   SmolLM2 embedding table, no transformer         <- controls for token identity
                                                                  alone (~what a from-scratch
                                                                  `wte` gives you at init)

READING IT. R²(pretrained) ~= R²(random) is a real null: it says the 135M frozen params buy
nothing a random projection of the same width would not. R²(pretrained) >> R²(random) is only
SUGGESTIVE of necessity -- a linear probe on FROZEN features is a lower bound, and the
from-scratch prelude gets to LEARN its features over 100k steps rather than accept fixed ones.
This probe can falsify necessity cleanly; it cannot establish it.

Feature-space R² is also not the final arbiter -- in this project MSE has already pointed the
wrong way down the gain axis (it said 0.607, the render said 1.45). Use --render to push the
ridge predictions through Z-Image and score them, which is the honest comparison.

Usage (R² only, ~3GB, safe under GPU contention):
    CUDA_VISIBLE_DEVICES=3 python scripts_local/text_encoder_ridge_probe.py --cache_dir ./cached_datasets/Mekadrom/image_gen_captions_only/train --whiten_stats ./cached_datasets/qwen_whiten_stats_k64.pt --n_samples 2048 --output_dir eval_output/text_dependency/ridge
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--whiten_stats", type=str, required=True,
                   help="qwen_whiten_stats_k64.pt -- regress in the SAME whitened space the "
                        "adapter is trained in, or per-dim scale dominates R².")
    p.add_argument("--text_encoder_model", type=str, default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--n_samples", type=int, default=2048)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--holdout", type=float, default=0.25)
    p.add_argument("--ridge_lambdas", type=float, nargs="+",
                   default=[1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],
                   help="swept on the holdout; the BEST per arm is reported, so no arm loses on "
                        "a bad regularization choice")
    p.add_argument("--no_4bit", action="store_true")
    p.add_argument("--pairs_reference", action="store_true",
                   help="ALSO compute the minimal-pair distances from text_dependency_probe.py on "
                        "the GROUND-TRUTH Qwen3 targets. Without this reference the model's "
                        "swap-vs-different-prompt ratio is uninterpretable: the pairs share a "
                        "scene, so SOME similarity is correct. Qwen3 is the competent encoder the "
                        "adapter is trying to match, so ITS ratio is the achievable bar.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    from megatransformer.utils.zimage_text_encoder import Qwen3TextTargetEncoder
    from megatransformer.scripts.data.world.dataset import MultimodalShardedDataset
    from megatransformer.scripts.data.world.data_collator import MultimodalDataCollator

    def resolve(d, s):
        for c in (d + "_" + s, d):
            if os.path.isdir(c):
                return c
        return None

    tdir = resolve(args.cache_dir, "train")
    dataset = MultimodalShardedDataset(text_shard_dir=tdir, image_shard_dir=tdir,
                                       cache_size=8, max_samples=args.n_samples)
    collator = MultimodalDataCollator(special_token_base=49152, eos_token_id=0)
    collator.force_direction = "synthesis"
    caps = []
    for i in range(len(dataset)):
        b = collator([dataset[i]])
        c = (b.get("text_texts") or [""])[0]
        if c:
            caps.append(c)
        if len(caps) >= args.n_samples:
            break
    print(f"gathered {len(caps)} captions", flush=True)

    # ── targets: whitened Qwen3 conditioning, K slots, exactly as the trainer builds them ──
    st = torch.load(args.whiten_stats, map_location="cpu")
    w_mean, w_std = st["mean"].float(), st["std"].float().clamp_min(1e-6)
    enc = Qwen3TextTargetEncoder(seq_len=args.seq_len, device=args.device,
                                 load_in_4bit=not args.no_4bit)
    Ys = []
    for i in range(0, len(caps), args.batch_size):
        t = enc.encode(caps[i:i + args.batch_size]).float().cpu()      # (B, K, 2560)
        Ys.append((t - w_mean) / w_std)
        if (i // args.batch_size) % 10 == 0:
            print(f"  targets {i}/{len(caps)}", flush=True)
    Y = torch.cat(Ys).reshape(-1, 2560).double()                        # (N*K, 2560)
    del Ys

    if args.pairs_reference:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from text_dependency_probe import PAIRS

        def gt(prompt):
            t = enc.encode([prompt]).float().cpu()[0]
            return ((t - w_mean) / w_std).flatten()

        def d(a, b):
            return float(1.0 - torch.nn.functional.cosine_similarity(a, b, dim=0))

        ref_rows = []
        for i, (base, swapped) in enumerate(PAIRS):
            other = PAIRS[(i + 1) % len(PAIRS)][0]
            cb, cs, co = gt(base), gt(swapped), gt(other)
            ref_rows.append({"prompt": base, "d_swap": d(cb, cs), "d_other": d(cb, co)})
            print(f"  [GT {i}] d_swap={ref_rows[-1]['d_swap']:.4f} "
                  f"d_other={ref_rows[-1]['d_other']:.4f}", flush=True)
        ms = sum(r["d_swap"] for r in ref_rows) / len(ref_rows)
        mo = sum(r["d_other"] for r in ref_rows) / len(ref_rows)
        gt_ref = {"rows": ref_rows, "mean_d_swap": ms, "mean_d_other": mo,
                  "swap_over_other": ms / max(mo, 1e-9)}
        print(f"\n[GT Qwen3] mean d_swap={ms:.4f} d_other={mo:.4f} "
              f"ratio={gt_ref['swap_over_other']:.3f}  <- the bar the adapter is chasing", flush=True)
        with open(os.path.join(args.output_dir, "pairs_gt_reference.json"), "w") as f:
            json.dump(gt_ref, f, indent=2)
    del enc
    torch.cuda.empty_cache()

    tok = AutoTokenizer.from_pretrained(args.text_encoder_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    @torch.no_grad()
    def features(kind):
        """(N*K, d) features, resampled to the SAME K slots the target uses."""
        cfg = AutoConfig.from_pretrained(args.text_encoder_model)
        if kind == "pretrained":
            m = AutoModelForCausalLM.from_pretrained(args.text_encoder_model,
                                                     dtype=torch.float32).model
        else:
            m = AutoModelForCausalLM.from_config(cfg).model          # random init
        m = m.to(args.device).eval()
        embed = m.get_input_embeddings()
        out = []
        for i in range(0, len(caps), args.batch_size):
            enc_in = tok(caps[i:i + args.batch_size], return_tensors="pt",
                         padding=True, truncation=True, max_length=128).to(args.device)
            if kind == "embed_only":
                h = embed(enc_in.input_ids)
            else:
                h = m(**enc_in).last_hidden_state
            mask = enc_in.attention_mask.unsqueeze(-1).to(h.dtype)
            h = h * mask                                              # zero the pads before resample
            # (B, T, d) -> (B, K, d): same K-slot resample the Qwen target path uses
            h = F.interpolate(h.transpose(1, 2), size=args.seq_len,
                              mode="linear", align_corners=False).transpose(1, 2)
            out.append(h.float().cpu())
        del m, embed
        torch.cuda.empty_cache()
        return torch.cat(out).reshape(-1, cfg.hidden_size).double()

    n_hold = int(len(Y) * args.holdout)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(Y), generator=g)
    tr_idx, te_idx = perm[n_hold:], perm[:n_hold]
    Ytr, Yte = Y[tr_idx], Y[te_idx]
    # R² is measured against the HOLDOUT's own mean, so "predict the mean" scores exactly 0.
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum()

    results = {}
    for kind in ("pretrained", "random", "embed_only"):
        X = features(kind)
        X = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], dim=1)   # bias column
        Xtr, Xte = X[tr_idx], X[te_idx]
        XtX, XtY = Xtr.T @ Xtr, Xtr.T @ Ytr
        eye = torch.eye(XtX.shape[0], dtype=XtX.dtype)
        best = None
        for lam in args.ridge_lambdas:
            beta = torch.linalg.solve(XtX + lam * eye, XtY)
            r2 = float(1.0 - ((Yte - Xte @ beta) ** 2).sum() / ss_tot)
            print(f"  [{kind}] lambda={lam:<8g} holdout R²={r2:.4f}", flush=True)
            if best is None or r2 > best[1]:
                best = (lam, r2)
        results[kind] = {"best_lambda": best[0], "r2": best[1], "dim": X.shape[1] - 1}
        print(f"[{kind}] BEST holdout R²={best[1]:.4f} (lambda={best[0]:g})", flush=True)
        del X, Xtr, Xte, XtX, XtY

    gap = results["pretrained"]["r2"] - results["random"]["r2"]
    summary = {"n_captions": len(caps), "seq_len": args.seq_len,
               "text_encoder_model": args.text_encoder_model,
               "arms": results, "pretrained_minus_random": gap}
    with open(os.path.join(args.output_dir, "ridge.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n| arm | holdout R² | best lambda |")
    print("|---|---|---|")
    for k, v in results.items():
        print(f"| {k} | {v['r2']:.4f} | {v['best_lambda']:g} |")
    print(f"\npretrained - random = {gap:+.4f}")
    print("~0 => SmolLM2's PRETRAINING is inert for this path (crutch confirmed).")
    print(">0 => pretraining carries linearly-usable structure; NOT proof of necessity, since a")
    print("      from-scratch prelude LEARNS its features instead of accepting frozen ones.")
    print(f"\nwrote {os.path.join(args.output_dir, 'ridge.json')}")


if __name__ == "__main__":
    main()
