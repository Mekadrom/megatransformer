"""Does the SDXL adapter show the SAME shrinkage as the Z-Image one -- and does a gain fix it?

WHY THIS EXISTS (it is a claim-test, not a decoder reconsideration -- Z-Image Turbo remains
the chosen decoder): the SDXL/CLIP arm plateaued at ~78% of ITS GT ceiling and the Z-Image/Qwen
arm plateaued at ~78% of ITS ceiling, and that coincidence was recorded as evidence for a
CONDITIONER-level ceiling (SmolLM2 + recurrent trunk + adapter) rather than anything
decoder-specific. But we then showed the Z-Image plateau was ~26% recoverable by an
inference-time dispersion gain -- an MSE-optimal point estimate is forced to shrink toward the
target mean by 1-R^2, and the frozen decoder reads that under-dispersion as washed-out/generic.

If SDXL shows the same alpha and the same gain response, then "78% on both" was substantially
measuring the SAME shrinkage artifact twice, and the conditioner-ceiling conclusion is weaker
than recorded. If SDXL does NOT respond, the shared 78% really is a conditioner property.

TWO DIFFERENCES from the Z-Image probe:
  1. SDXL conditioning is TWO tensors -- seq (77x2048, CLIP-L 768 (+) bigG 1280) and pooled
     (1280 bigG). They are measured and swept separately AND jointly: pooled is known to carry
     disproportionate scene/style control in SDXL, so a gain that helps one may hurt the other.
  2. The SDXL adapter has NO whitening buffers (Tier-0 whitening was built for the Z-Image
     arm). Shrinkage is only meaningful relative to the target distribution's mean/std, so this
     probe estimates them EMPIRICALLY from the encoded targets (per-dim over captions x tokens).
     With a few hundred captions that is a fine estimate for a diagnostic.

Example:
  uv run python scripts_local/sdxl_shrinkage_probe.py --checkpoint_path runs/world/world_image_sdxl_clip_baseline_0/checkpoint-21000 --config small_sum_sdxl --text_encoder_model HuggingFaceTB/SmolLM2-135M --bf16 --cache_dir ./cached_datasets/Mekadrom/image_gen_captions_only/val/ --max_samples 128
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--config", type=str, default="small_sum_sdxl")
    p.add_argument("--include_modes", type=str, default="text,image")
    p.add_argument("--text_encoder_model", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=128)
    p.add_argument("--prompts_file", type=str, default=None)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--render", action="store_true",
                   help="Render through SDXL and CLIPScore. Without --gain_sweep: pred vs "
                        "pred-rescaled-by-1/alpha vs target.")
    p.add_argument("--render_n", type=int, default=8)
    p.add_argument("--gain_sweep", type=str, default=None,
                   help="Comma-separated gains applied to BOTH seq and pooled (see --gain_mode).")
    p.add_argument("--gain_mode", type=str, default="both", choices=["both", "seq", "pooled"],
                   help="Which conditioning tensor the swept gain scales. 'both' is the direct "
                        "analogue of the Z-Image result; 'seq'/'pooled' isolate which one carries it.")
    p.add_argument("--sdxl_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--gen_steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.0)
    p.add_argument("--output_dir", type=str, default="eval_output/sdxl_shrinkage")
    return p.parse_args()


def stats(pred, tgt, name):
    """alpha / std-ratio / R^2 / best-gain MSE recovery in EMPIRICALLY whitened space."""
    D = tgt.shape[-1]
    t_flat = tgt.reshape(-1, D)
    mean, std = t_flat.mean(0), t_flat.std(0).clamp_min(1e-6)
    p = ((pred.reshape(-1, D) - mean) / std)
    t = ((t_flat - mean) / std)
    alpha = float((p * t).sum() / (t * t).sum().clamp_min(1e-12))
    std_ratio = float(p.std() / t.std().clamp_min(1e-12))
    mse = float(((p - t) ** 2).mean())
    r2 = float(1 - ((p - t) ** 2).sum() / (t ** 2).sum().clamp_min(1e-12))
    g_opt = float((p * t).sum() / (p * p).sum().clamp_min(1e-12))
    mse_gain = float(((g_opt * p - t) ** 2).mean())
    rho = alpha / max(std_ratio, 1e-9)
    out = {f"{name}_alpha": alpha, f"{name}_std_ratio": std_ratio, f"{name}_r2": r2,
           f"{name}_corr": rho, f"{name}_whitened_mse": mse, f"{name}_gain_opt": g_opt,
           f"{name}_mse_reduction_gain": 1 - mse_gain / max(mse, 1e-12)}
    print(f"  [{name:6s}] alpha={alpha:.4f}  std_ratio={std_ratio:.4f}  rho={rho:.4f}  "
          f"R^2={r2:.4f}  1/alpha={1/max(alpha,1e-6):.3f}  g_opt={g_opt:.3f} "
          f"(recovers {100*out[f'{name}_mse_reduction_gain']:.1f}% of MSE)", flush=True)
    return out, mean, std


def main():
    import numpy as np
    from megatransformer.model.world.world_model import MegaTransformerWorldModel
    from megatransformer.model.image.sdxl_adapter import SDXLConditioningAdapter
    from megatransformer.utils import model_loading_utils, constants
    from megatransformer.utils.sdxl_text_encoder import SDXLTextTargetEncoder

    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)
    include_modes = [m.strip() for m in args.include_modes.split(",")]

    overrides = {"include_modes": include_modes}
    if args.text_encoder_model:
        from transformers import AutoConfig, AutoTokenizer
        _llm = AutoConfig.from_pretrained(args.text_encoder_model)
        _eos = _llm.eos_token_id or AutoTokenizer.from_pretrained(args.text_encoder_model).eos_token_id
        overrides["special_token_base"] = int(_llm.vocab_size)
        overrides["eos_token_id"] = int(_eos)
        overrides["text_encoder"] = {"model": args.text_encoder_model, "freeze": True,
                                     "translator_hidden_mult": 2.0,
                                     "n_special_tokens": constants.N_SPECIAL_TOKENS}
    model = model_loading_utils.load_model(
        MegaTransformerWorldModel, args.config,
        checkpoint_path=args.checkpoint_path, overrides=overrides, device=device)
    model.to(device).eval()
    if not isinstance(getattr(model, "image_generator", None), SDXLConditioningAdapter):
        raise SystemExit(f"--config {args.config} does not use the SDXL adapter.")

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
            c = (coll([ds[i]]).get("text_texts") or [""])[0] or ""
            if c:
                captions.append(c)
            if len(captions) >= args.max_samples:
                break
    else:
        raise SystemExit("need --cache_dir or --prompts_file")
    print(f"[probe] {len(captions)} captions", flush=True)

    # ── predictions ──
    @torch.no_grad()
    def predict(caption):
        ids = _tok(caption, add_special_tokens=False).input_ids
        seq = ids + [_sptok.BOI, _sptok.IMAGE_PLACEHOLDER, _sptok.EOI, _eos_id]
        with torch.amp.autocast(device, dtype=dtype, enabled=args.bf16):
            out = model(text_input_ids=torch.tensor([seq], dtype=torch.long, device=device),
                        # dummy synthesis trigger; shape = prelude's LiteVAE latent (12ch, 256/8)
                        image_inputs=torch.zeros(1, 1, 12, 32, 32, device=device),
                        precomputed_latents=True,
                        is_synthesis=torch.tensor([True], device=device), decode_outputs=False)
        s, p = out.get("image_clip_seq_pred"), out.get("image_clip_pooled_pred")
        return (None, None) if s is None else (s[0].float().cpu(), p[0].float().cpu())

    Ps, Pp, keep = [], [], []
    for c in captions:
        s, p = predict(c)
        if s is not None:
            Ps.append(s); Pp.append(p); keep.append(c)
    captions = keep
    Ps, Pp = torch.stack(Ps), torch.stack(Pp)               # (N,77,2048), (N,1280)

    # ── targets ──
    enc = SDXLTextTargetEncoder(device=device, dtype=torch.float16)
    Ts, Tp = [], []
    for i in range(0, len(captions), 8):
        s, p = enc.encode(captions[i:i + 8])
        Ts.append(s.float().cpu()); Tp.append(p.float().cpu())
    Ts, Tp = torch.cat(Ts), torch.cat(Tp)
    del enc
    torch.cuda.empty_cache()

    print("\n== SDXL shrinkage probe (empirically whitened CLIP space) ==")
    res, s_mean, s_std = stats(Ps, Ts, "seq")
    r2_, p_mean, p_std = stats(Pp, Tp, "pooled")
    res.update(r2_)
    res.update({"checkpoint": args.checkpoint_path, "n_captions": len(captions)})
    with open(os.path.join(args.output_dir, "shrinkage.json"), "w") as f:
        json.dump(res, f, indent=2)

    if not args.render:
        return

    # ── render ──
    from PIL import Image
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    import open_clip
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.sdxl_model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    with torch.no_grad():
        neg_pe, _, neg_pp, _ = pipe.encode_prompt(prompt="", device=device,
                                                  num_images_per_prompt=1,
                                                  do_classifier_free_guidance=False)
    cm, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    ctok = open_clip.get_tokenizer("ViT-B-32")
    cm = cm.to(device).eval()
    from torchvision import transforms as _T
    cnorm = _T.Compose([_T.Resize(224), _T.CenterCrop(224),
                        _T.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711))])

    @torch.no_grad()
    def render(seq, pool):
        g = torch.Generator(device=device).manual_seed(1234)
        return pipe(prompt_embeds=seq.unsqueeze(0).to(device).half(),
                    pooled_prompt_embeds=pool.unsqueeze(0).to(device).half(),
                    negative_prompt_embeds=neg_pe.half(),
                    negative_pooled_prompt_embeds=neg_pp.half(),
                    num_inference_steps=args.gen_steps, guidance_scale=args.guidance,
                    height=1024, width=1024, generator=g).images[0]

    @torch.no_grad()
    def score(img, txt):
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
        im = F.normalize(cm.encode_image(cnorm(x)), dim=-1)
        tx = F.normalize(cm.encode_text(ctok([txt[:77]]).to(device)), dim=-1)
        return float((im @ tx.T).item())

    def scaled(i, g):
        """Scale deviation-from-target-mean by g, on whichever tensor(s) --gain_mode selects."""
        gs = g if args.gain_mode in ("both", "seq") else 1.0
        gp = g if args.gain_mode in ("both", "pooled") else 1.0
        return ((Ps[i] - s_mean) * gs + s_mean, (Pp[i] - p_mean) * gp + p_mean)

    n = min(args.render_n, len(captions))
    gains = [float(x) for x in args.gain_sweep.split(",")] if args.gain_sweep \
        else [1.0, 1 / max(res["seq_alpha"], 1e-6)]
    scores = {g: [] for g in gains}
    gt, rows = [], []
    for i in range(n):
        imgs = []
        for g in gains:
            im = render(*scaled(i, g))
            scores[g].append(score(im, captions[i]))
            imgs.append(im)
        gi = render(Ts[i], Tp[i])
        gt.append(score(gi, captions[i]))
        imgs.append(gi)
        rows.append(np.concatenate([np.asarray(im.resize((320, 320))) for im in imgs], 1))
        print(f"  [{i}] " + "  ".join(f"g{g:.2f}={scores[g][-1]:.3f}" for g in gains)
              + f"  gt={gt[-1]:.3f} | {captions[i][:50]}", flush=True)
    Image.fromarray(np.concatenate(rows, 0)).save(
        os.path.join(args.output_dir, f"gain_sweep_{args.gain_mode}.png"))
    mg = float(np.mean(gt))
    means = {g: float(np.mean(v)) for g, v in scores.items()}
    # Baseline at gain 1.0 (as-trained) when it was swept, NOT gains[0] -- a sweep that
    # starts below 1.0 would otherwise report every gain as an improvement over the worst.
    base = means.get(1.0, means[gains[0]])
    print(f"\n  == SDXL gain sweep (mode={args.gain_mode}) ==")
    for g in gains:
        print(f"    gain {g:>5.2f}: CLIPScore {means[g]:.4f}   "
              f"({100*(means[g]-base)/max(mg-base,1e-9):+.1f}% of the gap closed)")
    print(f"    GT ceiling : {mg:.4f}")
    print(f"    -> BEST gain = {max(means, key=means.get):.2f}\n", flush=True)
    res.update({"gain_sweep": means, "clip_target": mg, "gain_mode": args.gain_mode})
    with open(os.path.join(args.output_dir, "shrinkage.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
