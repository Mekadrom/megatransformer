"""Is the Z-Image adapter's error SHRINKAGE toward the mean?

Motivating observation (t1q run): with a 4096-entry InfoNCE memory queue the adapter
solves 4104-way caption retrieval to loss ~0.04 (chance = ln 4104 = 8.3) -- caption
IDENTITY is preserved almost perfectly -- yet the rendered CLIPScore sits frozen at
~78% of the Z-Image GT ceiling while the MSE keeps falling.

Both facts are explained by a shrunk point estimate:

    pred ~= mean + alpha * (target - mean),   alpha < 1

Shrinkage preserves relative geometry exactly (so InfoNCE is happy at ANY number of
negatives) while pulling every prediction into a low-amplitude shell the DiT reads as a
washed-out, generic version of the caption. It is also what a whitened MSE PROVABLY
converges to under residual uncertainty: the conditional-mean estimator shrinks by the
fraction of variance it cannot explain, so alpha ~= R^2 is the *optimal* MSE behaviour,
not a bug -- which is exactly why more MSE training cannot escape it.

This probe measures alpha directly in WHITENED space (where the loss lives; whitening
also makes the target mean 0, so "shrinkage toward the mean" is just "smaller norm").

Reported:
  alpha_global   least-squares slope of pred on target (1.0 = no shrinkage)
  std_ratio      std(pred)/std(target)  (a shrunk estimate is under-dispersed)
  R2             variance explained = 1 - ||p-t||^2/||t||^2
  alpha per-dim  median/IQR across the 2560 dims (is shrinkage uniform or selective?)
  between/within variance decomposition: does the adapter keep the caption-to-caption
                 spread (gist) but crush the per-token spread (detail), or shrink both?
  retrieval      rank-1 accuracy + mean rank of the correct target (reconciles InfoNCE)
  gain-corrected MSE: whitened MSE after rescaling pred by 1/alpha. If this is much
                 lower than the raw MSE, shrinkage is the DOMINANT error mode and a
                 variance-matching term (--image_var_loss_weight, already in the world
                 trainer for other modalities) is a cheap fix. If it barely moves, the
                 error is directional/structural and needs T2/T3 instead.

No Z-Image DiT/VAE is loaded (this is a pure feature-space measurement): world model +
4-bit Qwen3 target encoder only, ~4GB. Use --render to additionally render pred,
pred/alpha and target through Z-Image and CLIPScore them -- the arbiter for whether a
gain fix actually buys image quality (needs ~14GB more).

Example:
  uv run python scripts_local/zimage_shrinkage_probe.py --checkpoint_path runs/world/world_image_zimage_qwen_t1q_0/checkpoint-1000 --config small_sum_zimage_whiten_t1q --text_encoder_model HuggingFaceTB/SmolLM2-135M --bf16 --cache_dir ./cached_datasets/Mekadrom/image_gen_captions_only/val/ --max_samples 128
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--config", type=str, default="small_sum_zimage_whiten_t1q")
    p.add_argument("--include_modes", type=str, default="text,image")
    p.add_argument("--text_encoder_model", type=str, default=None,
                   help="MUST match training (e.g. HuggingFaceTB/SmolLM2-135M).")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="Val shard dir to pull captions from. More captions = a far better "
                        "between-caption variance estimate than the 8-prompt probe.")
    p.add_argument("--max_samples", type=int, default=128)
    p.add_argument("--prompts_file", type=str, default=None,
                   help="Use these captions instead of the dataset (one per line).")
    p.add_argument("--flow_bypass", action="store_true",
                   help="Bypass the T3 flow head and surface the auxiliary POINT head instead. Diagnostic: if point-head renders are healthy while sampled ones are not, the shared Q-Former/trunk is intact and only the flow head is undertrained/broken; if BOTH are bad, the fresh head's gradients damaged the warm-started trunk.")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--target_device", type=str, default=None,
                   help="Put the Qwen3 target encoder on a different GPU.")
    p.add_argument("--target_bf16", action="store_true")
    p.add_argument("--render", action="store_true",
                   help="ALSO render pred / pred-rescaled-by-1/alpha / target through Z-Image and "
                        "CLIPScore them. The arbiter: does undoing the shrinkage fix the image?")
    p.add_argument("--render_n", type=int, default=8)
    p.add_argument("--gain_sweep", type=str, default=None,
                   help="With --render: comma-separated whitened-space output gains to sweep "
                        "(e.g. '1.0,1.2,1.35,1.5,1.75,2.0'). 1/alpha exactly undoes the shrinkage, "
                        "but the DiT's preferred dispersion is an empirical question -- this finds "
                        "the CLIPScore-optimal gain. Saves a montage (rows=prompts, cols=gains+GT).")
    p.add_argument("--gen_steps", type=int, default=8)
    p.add_argument("--guidance", type=float, default=0.0)
    p.add_argument("--zimage_model", type=str, default="Tongyi-MAI/Z-Image-Turbo")
    p.add_argument("--output_dir", type=str, default="eval_output/zimage_shrinkage")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--log_dir", type=str, default=None, help="Also log scalars to this run's TB.")
    return p.parse_args()


def main():
    from megatransformer.model.world.world_model import MegaTransformerWorldModel, ZImageConditioningAdapter
    from megatransformer.utils import model_loading_utils, constants
    from megatransformer.utils.zimage_text_encoder import Qwen3TextTargetEncoder

    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)
    include_modes = [m.strip() for m in args.include_modes.split(",")]

    # ── world model ──
    overrides = {"include_modes": include_modes}
    if args.text_encoder_model:
        from transformers import AutoConfig, AutoTokenizer
        _llm = AutoConfig.from_pretrained(args.text_encoder_model)
        _eos = _llm.eos_token_id
        if _eos is None:
            _eos = AutoTokenizer.from_pretrained(args.text_encoder_model).eos_token_id
        overrides["special_token_base"] = int(_llm.vocab_size)
        overrides["eos_token_id"] = int(_eos)
        overrides["text_encoder"] = {
            "model": args.text_encoder_model, "freeze": True,
            "translator_hidden_mult": 2.0, "n_special_tokens": constants.N_SPECIAL_TOKENS,
        }
    model = model_loading_utils.load_model(
        MegaTransformerWorldModel, args.config,
        checkpoint_path=args.checkpoint_path, overrides=overrides, device=device)
    model.to(device).eval()
    gen = getattr(model, "image_generator", None)
    if not isinstance(gen, ZImageConditioningAdapter):
        raise SystemExit(f"--config {args.config} does not use the Z-Image adapter.")
    seq_len = int(gen.config.seq_len)
    if not gen.whiten:
        print("[warn] adapter has whitening OFF; stats below are in raw Qwen3 space, where the "
              "555x per-dim std spread makes alpha dominated by a few massive dims.", flush=True)
    w_mean = gen.whiten_mean.detach().float().cpu()      # (2560,)
    if getattr(args, "flow_bypass", False) and getattr(model.image_generator, "flow_head", None) is not None:
        model.image_generator.flow_head = None
        print("[t3] flow head BYPASSED -> surfacing the auxiliary point head", flush=True)
    w_std = gen.whiten_std.detach().float().cpu()

    _mcfg = model.module.config if hasattr(model, "module") else model.config
    _stb = int(getattr(_mcfg, "special_token_base", 32000))
    _sptok = constants.special_token_ids(_stb)
    _eos_id = int(getattr(_mcfg, "eos_token_id", 2))
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(args.text_encoder_model)

    # ── captions ──
    captions = []
    if args.prompts_file:
        captions = [l.strip() for l in open(args.prompts_file) if l.strip()][:args.max_samples]
    elif args.cache_dir:
        from megatransformer.scripts.data.world.dataset import MultimodalShardedDataset
        from megatransformer.scripts.data.world.data_collator import MultimodalDataCollator
        d = args.cache_dir
        d = next((c for c in (d + "_val", d) if os.path.isdir(c)), d)
        ds = MultimodalShardedDataset(text_shard_dir=d, image_shard_dir=d, cache_size=8,
                                      max_samples=args.max_samples * 2)
        coll = MultimodalDataCollator(special_token_base=_stb, eos_token_id=_eos_id)
        coll.force_direction = "synthesis"
        for i in range(len(ds)):
            b = coll([ds[i]])
            c = (b.get("text_texts") or [""])[0] or ""
            if c:
                captions.append(c)
            if len(captions) >= args.max_samples:
                break
    else:
        raise SystemExit("need --cache_dir or --prompts_file")
    if len(captions) < 4:
        raise SystemExit(f"only {len(captions)} captions; need >=4 for variance stats")
    print(f"[probe] {len(captions)} captions | seq_len={seq_len} whiten={gen.whiten}", flush=True)

    # ── predictions ──
    @torch.no_grad()
    def predict(caption):
        ids = _tok(caption, add_special_tokens=False).input_ids
        seq = ids + [_sptok.BOI, _sptok.IMAGE_PLACEHOLDER, _sptok.EOI, _eos_id]
        with torch.amp.autocast(device, dtype=dtype, enabled=args.bf16):
            out = model(text_input_ids=torch.tensor([seq], dtype=torch.long, device=device),
                        image_inputs=torch.zeros(1, 1, 12, 32, 32, device=device),
                        precomputed_latents=True,
                        is_synthesis=torch.tensor([True], device=device),
                        decode_outputs=False)
        sp = out.get("image_clip_seq_pred")
        return None if sp is None else sp[0].float().cpu()      # (seq_len, 2560), Qwen3 space

    preds = []
    keep = []
    for c in captions:
        p = predict(c)
        if p is not None:
            preds.append(p)
            keep.append(c)
    captions = keep
    P = torch.stack(preds)                                       # (N, L, D) Qwen3 space
    del preds

    # ── targets (frozen Qwen3, same resample the trainer uses) ──
    enc = Qwen3TextTargetEncoder(
        model_name=getattr(gen.config, "target_model", "Tongyi-MAI/Z-Image-Turbo"),
        seq_len=seq_len, device=args.target_device or device,
        load_in_4bit=(getattr(gen.config, "target_load_in_4bit", True) and not args.target_bf16))
    T = []
    B = 8
    for i in range(0, len(captions), B):
        T.append(enc.encode(captions[i:i + B]).float().cpu())
    T = torch.cat(T)                                             # (N, L, D)
    del enc
    torch.cuda.empty_cache()
    assert T.shape == P.shape, f"target {tuple(T.shape)} != pred {tuple(P.shape)}"

    # ── whitened space (where the loss lives; target mean -> 0) ──
    Pw = (P - w_mean) / w_std
    Tw = (T - w_mean) / w_std
    N, L, D = Pw.shape

    p, t = Pw.reshape(-1, D), Tw.reshape(-1, D)                  # (N*L, D)
    alpha = float((p * t).sum() / (t * t).sum().clamp_min(1e-12))
    std_ratio = float(p.std() / t.std().clamp_min(1e-12))
    mse = float(((p - t) ** 2).mean())
    r2 = float(1 - ((p - t) ** 2).sum() / (t ** 2).sum().clamp_min(1e-12))
    # Per-dim slope: does every dim shrink alike, or is it a few dims?
    a_dim = ((p * t).sum(0) / (t * t).sum(0).clamp_min(1e-12))
    q = torch.quantile(a_dim, torch.tensor([0.25, 0.5, 0.75]))

    # Best achievable MSE from a pure GAIN fix: min over g of ||g*p - t||^2.
    g_opt = float((p * t).sum() / (p * p).sum().clamp_min(1e-12))
    mse_gain = float(((g_opt * p - t) ** 2).mean())
    # Per-dim gain fix (a diagonal rescale -- what var-matching approximates).
    gd = ((p * t).sum(0) / (p * p).sum(0).clamp_min(1e-12))
    mse_gain_dim = float(((p * gd - t) ** 2).mean())

    # Between-caption (gist) vs within-caption/token (detail) variance.
    def decomp(X):
        m = X.mean(1, keepdim=True)                              # (N,1,D) per-caption mean
        return float(m.squeeze(1).var(0).mean()), float((X - m).var(1).mean())
    b_p, w_p = decomp(Pw)
    b_t, w_t = decomp(Tw)

    # Retrieval: rank of the correct target by whitened L2 on caption-mean vectors.
    pm = F.normalize(Pw.mean(1), dim=-1)
    tm = F.normalize(Tw.mean(1), dim=-1)
    sim = pm @ tm.T
    rank = (sim > sim.diag().unsqueeze(1)).sum(1)                # 0 = correct is nearest
    top1 = float((rank == 0).float().mean())
    mrank = float(rank.float().mean()) + 1

    res = {
        "checkpoint": args.checkpoint_path, "n_captions": N, "seq_len": L, "dim": D,
        "alpha_global": alpha, "std_ratio": std_ratio, "r2": r2, "whitened_mse": mse,
        "alpha_dim_p25": float(q[0]), "alpha_dim_median": float(q[1]), "alpha_dim_p75": float(q[2]),
        "gain_opt": g_opt, "mse_after_scalar_gain": mse_gain, "mse_after_perdim_gain": mse_gain_dim,
        "mse_reduction_scalar_gain": 1 - mse_gain / max(mse, 1e-12),
        "mse_reduction_perdim_gain": 1 - mse_gain_dim / max(mse, 1e-12),
        "between_caption_var_pred": b_p, "between_caption_var_target": b_t,
        "between_ratio": b_p / max(b_t, 1e-12),
        "within_caption_var_pred": w_p, "within_caption_var_target": w_t,
        "within_ratio": w_p / max(w_t, 1e-12),
        "retrieval_top1": top1, "retrieval_mean_rank": mrank,
    }

    print("\n== shrinkage probe (whitened Qwen3 space) ==")
    print(f"  alpha (slope pred~target) : {alpha:.4f}      (1.0 = no shrinkage)")
    print(f"  std(pred)/std(target)     : {std_ratio:.4f}")
    print(f"  R^2 / variance explained  : {r2:.4f}")
    print(f"  whitened MSE              : {mse:.4f}")
    print(f"  per-dim alpha  p25/med/p75: {float(q[0]):.3f} / {float(q[1]):.3f} / {float(q[2]):.3f}")
    print(f"  between-caption var ratio : {res['between_ratio']:.4f}   (gist retained)")
    print(f"  within-caption  var ratio : {res['within_ratio']:.4f}   (per-token detail retained)")
    print(f"  retrieval top-1 / meanrank: {top1:.3f} / {mrank:.1f}  of {N}")
    print(f"  MSE after best scalar gain: {mse_gain:.4f}  ({100*res['mse_reduction_scalar_gain']:.1f}% lower, g={g_opt:.3f})")
    print(f"  MSE after per-dim gain    : {mse_gain_dim:.4f}  ({100*res['mse_reduction_perdim_gain']:.1f}% lower)")
    verdict = ("SHRINKAGE-DOMINATED: a gain/variance-matching fix recovers most of the error"
               if res["mse_reduction_scalar_gain"] > 0.25 else
               "NOT shrinkage-dominated: error is directional/structural, gain fix won't help")
    print(f"  -> {verdict}\n", flush=True)

    with open(os.path.join(args.output_dir, "shrinkage.json"), "w") as f:
        json.dump(res, f, indent=2)

    if args.log_dir:
        from megatransformer.utils import metrics
        from megatransformer.utils.metrics_backend import TensorBoardBackend
        step = args.step
        if step is None:
            b = os.path.basename(args.checkpoint_path.rstrip("/"))
            step = int(b.split("-")[-1]) if b.startswith("checkpoint-") else 0
        metrics.init_metrics(TensorBoardBackend(log_dir=args.log_dir))  # 1st positional is `writer`
        for k, v in res.items():
            if isinstance(v, (int, float)):
                metrics.log_scalar(f"shrinkage/{k}", v, step, skip_zero=False)
        metrics.flush()

    # ── optional: does undoing the shrinkage fix the IMAGE? ──
    if args.render:
        import numpy as np
        from PIL import Image
        from diffusers import ZImagePipeline
        import open_clip
        pipe = ZImagePipeline.from_pretrained(args.zimage_model, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        cm, _, cp = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        ct = open_clip.get_tokenizer("ViT-B-32")
        cm = cm.to(device).eval()

        @torch.no_grad()
        def render(seq_qwen):
            g = torch.Generator(device=device).manual_seed(1234)
            return pipe(prompt_embeds=[seq_qwen.to(device, torch.bfloat16)],
                        num_inference_steps=args.gen_steps, guidance_scale=args.guidance,
                        height=1024, width=1024, generator=g).images[0]

        @torch.no_grad()
        def score(img, txt):
            i = cm.encode_image(cp(img).unsqueeze(0).to(device))
            x = cm.encode_text(ct([txt]).to(device))
            return float(F.cosine_similarity(F.normalize(i, dim=-1), F.normalize(x, dim=-1)).item())

        n = min(args.render_n, N)

        if args.gain_sweep:
            # Sweep the whitened-space output gain. The training loss is MINIMIZED at g=1
            # by construction, so any g>1 that scores better is direct evidence of the
            # proxy gap (the DiT wants dispersion more than it wants L2 accuracy).
            gains = [float(g) for g in args.gain_sweep.split(",") if g.strip()]
            scores = {g: [] for g in gains}
            gt_scores, rows = [], []
            for i in range(n):
                imgs = []
                for g in gains:
                    seq = (Pw[i] * g) * w_std + w_mean
                    im = render(seq)
                    scores[g].append(score(im, captions[i]))
                    imgs.append(im)
                gt_im = render(T[i])
                gt_scores.append(score(gt_im, captions[i]))
                imgs.append(gt_im)
                rows.append(np.concatenate([np.asarray(im.resize((320, 320))) for im in imgs], 1))
                print(f"  [{i}] " + "  ".join(f"g{g}={scores[g][-1]:.3f}" for g in gains)
                      + f"  gt={gt_scores[-1]:.3f} | {captions[i][:50]}", flush=True)
            Image.fromarray(np.concatenate(rows, 0)).save(
                os.path.join(args.output_dir, "gain_sweep.png"))
            mg = float(np.mean(gt_scores))
            means = {g: float(np.mean(v)) for g, v in scores.items()}
            best = max(means, key=means.get)
            print(f"\n  == gain sweep (cols: {gains} + GT) ==")
            for g in gains:
                closed = 100 * (means[g] - means[gains[0]]) / max(mg - means[gains[0]], 1e-9)
                print(f"    gain {g:>5}: CLIPScore {means[g]:.4f}   ({closed:+.1f}% of the gap closed)")
            print(f"    GT ceiling : {mg:.4f}")
            print(f"    -> BEST gain = {best} (1/alpha = {1/max(alpha,1e-6):.3f})\n", flush=True)
            res.update({"gain_sweep": means, "gain_sweep_best": best, "clip_target": mg})
            with open(os.path.join(args.output_dir, "shrinkage.json"), "w") as f:
                json.dump(res, f, indent=2)
            return

        rows, s_raw, s_fix, s_gt = [], [], [], []
        for i in range(n):
            # rescale in WHITENED space by 1/alpha, then de-whiten back to Qwen3 space
            fixed_w = Pw[i] / max(alpha, 1e-6)
            fixed = fixed_w * w_std + w_mean
            imgs = [render(P[i]), render(fixed), render(T[i])]
            s_raw.append(score(imgs[0], captions[i]))
            s_fix.append(score(imgs[1], captions[i]))
            s_gt.append(score(imgs[2], captions[i]))
            rows.append(np.concatenate([np.asarray(im.resize((384, 384))) for im in imgs], 1))
            print(f"  [{i}] raw={s_raw[-1]:.4f} rescaled={s_fix[-1]:.4f} gt={s_gt[-1]:.4f} | {captions[i][:60]}",
                  flush=True)
        Image.fromarray(np.concatenate(rows, 0)).save(
            os.path.join(args.output_dir, "pred_vs_rescaled_vs_target.png"))
        mr, mf, mg = float(np.mean(s_raw)), float(np.mean(s_fix)), float(np.mean(s_gt))
        print(f"\n  mean CLIPScore  pred={mr:.4f}  rescaled(1/alpha)={mf:.4f}  target={mg:.4f}")
        print(f"  ceiling closed by rescale: {100*(mf-mr)/max(mg-mr,1e-9):.1f}%\n", flush=True)
        res.update({"clip_pred": mr, "clip_rescaled": mf, "clip_target": mg})
        with open(os.path.join(args.output_dir, "shrinkage.json"), "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
