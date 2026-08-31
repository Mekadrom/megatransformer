"""Does recurrent DEPTH buy anything? CLIPScore vs recurrent iteration count.

MOTIVATION. `trunk_compression_probe.py` measured that ~96% of the prompt conditioning reaching
the image gen queries arrives in recurrent iteration 0, and 23 further iterations add ~5%
(cond/const 0.0409 -> 0.0428 at ckpt-11000; flat 0.1076 -> 0.1085 across all 13 iterations at
ckpt-65000). That is a measurement of the REPRESENTATION. This measures the thing that actually
matters: whether the RENDER improves with depth.

DESIGN
  - iteration counts are POWERS OF 2 (1,2,4,8,16,32): if depth matters at all it should show as a
    monotone trend across octaves; a linear sweep wastes renders in a region the representation
    probe says is flat.
  - PAIRED SEEDS: sample s of prompt p uses the same flow-sampler seed at every iteration count,
    so the comparison is within-draw and does not re-roll the sampler noise that dominates this
    head at w=1 (d_seed 0.88 vs d_other 0.13 unguided).
  - UNSEEN PROMPTS by default (zimage_unseen_prompts.txt, 20 prompts). The committed 8-prompt
    probe has been used for every decision in this project and is a selection risk.
  - GT REFERENCE per prompt: the same prompt rendered from the TRUE Qwen3 conditioning, so each
    prompt's curve is read against its own ceiling rather than a global mean (per-prompt GT
    ranges 0.32-0.42 on the old probe).

⚠️ EXIT CRITERIA ARE DISABLED. The recurrent block has a KL early-exit, so `mean_thinking_steps`
is normally a CAP, not a count -- at ckpt-65000 it exits at 13 of 32. Without disabling it, "32
iterations" would silently be "13 iterations" and the sweep would flatten by construction. The
actual per-forward count is recorded and asserted against the request.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts_local/iteration_sweep.py \
        --checkpoint_path runs/world_image/zimage_qwen_.../checkpoint-66000 \
        --config small_sum_zimage_t3_xskip \
        --text_encoder_model HuggingFaceTB/SmolLM2-135M --bf16 \
        --output_dir eval_output/iter_sweep/ckpt66000
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--text_encoder_model", default=None)
    p.add_argument("--include_modes", default="text,image")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--prompts_file", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "src", "megatransformer",
        "scripts", "eval", "world", "zimage_unseen_prompts.txt"))
    p.add_argument("--iters", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--n_samples", type=int, default=2, help="draws per prompt per iteration count")
    p.add_argument("--seed_base", type=int, default=1234)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--gen_steps", type=int, default=8)
    p.add_argument("--max_prompts", type=int, default=None)
    p.add_argument("--no_save_images", action="store_true",
                   help="skip writing renders. Default SAVES them: a depth sweep is about whether "
                        "the IMAGE changes, and CLIPScore moving by 0.02 does not tell you "
                        "whether the failure mode changed. Renders are the arbiter here.")
    p.add_argument("--zimage_model", default="Tongyi-MAI/Z-Image-Turbo")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    import numpy as np
    from megatransformer.model.world.world_model import MegaTransformerWorldModel
    from megatransformer.utils import model_loading_utils, constants
    from transformers import AutoTokenizer, AutoConfig

    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)

    prompts = [l.strip() for l in open(args.prompts_file) if l.strip()]
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"{len(prompts)} prompts, iters={args.iters}, n_samples={args.n_samples}", flush=True)

    overrides = {"include_modes": [m.strip() for m in args.include_modes.split(",")]}
    if args.text_encoder_model:
        llm = AutoConfig.from_pretrained(args.text_encoder_model)
        eos = llm.eos_token_id or AutoTokenizer.from_pretrained(args.text_encoder_model).eos_token_id
        overrides["special_token_base"] = int(llm.vocab_size)
        overrides["eos_token_id"] = int(eos)
        overrides["text_encoder"] = {"model": args.text_encoder_model, "freeze": True,
                                     "translator_hidden_mult": 2.0,
                                     "n_special_tokens": constants.N_SPECIAL_TOKENS}
    model = model_loading_utils.load_model(MegaTransformerWorldModel, args.config,
                                           checkpoint_path=args.checkpoint_path,
                                           overrides=overrides, device=device)
    model.to(device).eval()
    adapter = model.image_generator
    if getattr(adapter, "flow_head", None) is not None:
        adapter.flow_head.guidance = float(args.guidance)

    rb = model.recurrent_block
    # See the header: without this, mean_thinking_steps is a CAP and the sweep flattens itself.
    saved_exit, saved_steps = rb.exit_criteria, rb.mean_thinking_steps
    rb.exit_criteria = None

    tok = AutoTokenizer.from_pretrained(args.text_encoder_model or "mistralai/Mistral-7B-v0.1")
    mcfg = model.config
    sptok = constants.special_token_ids(int(getattr(mcfg, "special_token_base", 32000)))
    eos_id = int(getattr(mcfg, "eos_token_id", 2))
    seq_len = int(adapter.seq_len)

    from diffusers import ZImagePipeline
    pipe = ZImagePipeline.from_pretrained(args.zimage_model, torch_dtype=torch.bfloat16)
    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    import open_clip
    from torchvision import transforms
    cm, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
    cm.eval()
    ctok = open_clip.get_tokenizer("ViT-B-32")
    cnorm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))])

    @torch.no_grad()
    def clipscore(img, txt):
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div(255)
        x = x.unsqueeze(0).to(device)
        im = F.normalize(cm.encode_image(cnorm(x)), dim=-1)
        tx = F.normalize(cm.encode_text(ctok([txt[:77]]).to(device)), dim=-1)
        return float((im @ tx.T).item())

    @torch.no_grad()
    def render(seq, seed):
        g = torch.Generator(device=device).manual_seed(seed)
        return pipe(prompt_embeds=[seq], num_inference_steps=args.gen_steps,
                    guidance_scale=0.0, height=1024, width=1024, generator=g).images[0]

    @torch.no_grad()
    def gt_cond(caption):
        s = pipe.tokenizer.apply_chat_template([{"role": "user", "content": caption}],
                                               tokenize=False, add_generation_prompt=True,
                                               enable_thinking=True)
        ti = pipe.tokenizer(s, padding="max_length", max_length=512, truncation=True,
                            return_tensors="pt")
        ids = ti.input_ids.to(pipe.text_encoder.device)
        attn = ti.attention_mask.to(pipe.text_encoder.device)
        hs = pipe.text_encoder(input_ids=ids, attention_mask=attn,
                              output_hidden_states=True).hidden_states[-2][0]
        real = hs[attn.bool()[0]]
        if real.shape[0] == 0:
            real = hs[:1]
        x = real.transpose(0, 1).unsqueeze(0).float()
        return F.interpolate(x, size=seq_len, mode="linear",
                             align_corners=False)[0].transpose(0, 1).to(hs.dtype)

    @torch.no_grad()
    def pred_cond(prompt, seed):
        if getattr(adapter, "flow_head", None) is not None:
            g = torch.Generator(device=device); g.manual_seed(int(seed))
            adapter.flow_generator = g
        ids = tok(prompt, add_special_tokens=False).input_ids
        seq = ids + [sptok.BOI, sptok.IMAGE_PLACEHOLDER, sptok.EOI, eos_id]
        with torch.amp.autocast(device, dtype=dtype, enabled=args.bf16):
            out = model(text_input_ids=torch.tensor([seq], dtype=torch.long, device=device),
                        image_inputs=torch.zeros(1, 1, 12, 32, 32, device=device),
                        precomputed_latents=True,
                        is_synthesis=torch.tensor([True], device=device),
                        decode_outputs=False)
        return out["image_clip_seq_pred"][0].float(), int(out.get("recurrent_num_iterations", -1))

    save_imgs = not args.no_save_images
    rdir = os.path.join(args.output_dir, "renders")
    if save_imgs:
        os.makedirs(rdir, exist_ok=True)

    # ── GT reference per prompt (deterministic; one render each) ──
    gt, gt_img = {}, {}
    for i, p in enumerate(prompts):
        im = render(gt_cond(p).float(), 10_000 + i)
        gt[p] = clipscore(im, p)
        if save_imgs:
            gt_img[i] = im
            im.save(os.path.join(rdir, f"p{i:02d}_GT.png"))
        print(f"  [GT {i+1}/{len(prompts)}] {gt[p]:.3f}  {p[:52]}", flush=True)
    print(f"GT mean {np.mean(list(gt.values())):.4f}", flush=True)

    rows = []
    imgs_by_prompt = {}
    for it in args.iters:
        rb.mean_thinking_steps = int(it)
        for pi, p in enumerate(prompts):
            for s in range(args.n_samples):
                seed = args.seed_base + 1000 * pi + s     # PAIRED across iteration counts
                seq, used = pred_cond(p, seed)
                im = render(seq, seed)
                sc = clipscore(im, p)
                if save_imgs:
                    im.save(os.path.join(rdir, f"p{pi:02d}_s{s}_it{it:02d}.png"))
                    imgs_by_prompt.setdefault((pi, s), {})[it] = im
                rows.append({"iters_req": it, "iters_used": used, "prompt": p, "prompt_i": pi,
                             "sample": s, "seed": seed, "clip": sc, "gt": gt[p],
                             "frac_of_gt": sc / max(gt[p], 1e-9)})
        got = sorted({r["iters_used"] for r in rows if r["iters_req"] == it})
        m = np.mean([r["clip"] for r in rows if r["iters_req"] == it])
        print(f"[iters={it:>2}] mean CLIP {m:.4f}   actual iterations observed: {got}", flush=True)

    if save_imgs and imgs_by_prompt:
        from PIL import Image, ImageDraw
        TH = 320
        for (pi, s_i), byit in sorted(imgs_by_prompt.items()):
            cols = [byit[i] for i in args.iters if i in byit] + [gt_img.get(pi)]
            cols = [c for c in cols if c is not None]
            labels = [f"it={i}" for i in args.iters if i in byit] + ["GT"]
            canvas = Image.new("RGB", (TH * len(cols), TH + 22), "white")
            d = ImageDraw.Draw(canvas)
            for k, (c, lab) in enumerate(zip(cols, labels)):
                canvas.paste(c.resize((TH, TH)), (k * TH, 22))
                d.text((k * TH + 6, 6), lab, fill="black")
            canvas.save(os.path.join(rdir, f"montage_p{pi:02d}_s{s_i}.png"))
        print(f"wrote {len(imgs_by_prompt)} per-prompt montages (columns = iteration counts, "
              f"rightmost = GT) to {rdir}", flush=True)

    rb.exit_criteria, rb.mean_thinking_steps = saved_exit, saved_steps
    with open(os.path.join(args.output_dir, "iteration_sweep.json"), "w") as f:
        json.dump({"checkpoint": args.checkpoint_path, "guidance": args.guidance,
                   "gt": gt, "rows": rows}, f, indent=2)

    # ── plot ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for pi, p in enumerate(prompts):
        ys = [np.mean([r["clip"] for r in rows if r["prompt_i"] == pi and r["iters_req"] == it])
              for it in args.iters]
        ax.plot(args.iters, ys, color="#888", alpha=0.35, lw=1)
    mean = [np.mean([r["clip"] for r in rows if r["iters_req"] == it]) for it in args.iters]
    sem = [np.std([r["clip"] for r in rows if r["iters_req"] == it]) /
           np.sqrt(max(1, len([r for r in rows if r["iters_req"] == it]))) for it in args.iters]
    ax.errorbar(args.iters, mean, yerr=sem, color="#1f77b4", lw=2.5, marker="o",
                capsize=4, label="mean CLIPScore")
    ax.axhline(np.mean(list(gt.values())), color="#d62728", ls="--", lw=1.5,
               label=f"GT mean ({np.mean(list(gt.values())):.3f})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(args.iters); ax.set_xticklabels([str(i) for i in args.iters])
    ax.set_xlabel("recurrent iterations (KL early-exit DISABLED)")
    ax.set_ylabel("CLIPScore (raw cosine)")
    ax.set_title(f"CLIPScore vs recurrent depth — {os.path.basename(args.checkpoint_path)}\n"
                 f"{len(prompts)} unseen prompts x {args.n_samples} paired draws, w={args.guidance}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(args.output_dir, "iteration_sweep.png")
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}\nwrote {os.path.join(args.output_dir, 'iteration_sweep.json')}")
    print("faint grey lines are per-prompt means; if depth mattered they would rise together.")


if __name__ == "__main__":
    main()
