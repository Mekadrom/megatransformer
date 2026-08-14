"""Image-synthesis diagnostics for world-image runs — the DiT analog of the
world-voice text-conditioning probes (early_text_delta / cfg_guidance_probe).

The recurring question for the image path is the SAME one that dominated
world-tts: **is the DiT actually conditioning on the caption, or just modelling
the marginal image distribution?** This script answers it two ways, from
cheapest/most-sensitive to most-perceptual:

  PROBE 1 — cond  (teacher-forced conditioning-loss delta; the WORKHORSE)
    Runs the model's own teacher-forced flow-matching loss (velocity-MSE) with
    the TRUE caption vs a MISMATCHED caption on the same image latent, averaged
    over many random-t draws (the loss is ~16% noisy per draw — see memory
    `feedback_diagnose_checkpoint_noise`, so we repeat). If the caption matters,
    the mismatched caption should predict velocity WORSE:
        cond_delta      = loss_shuf - loss_true       (>0 ⇒ caption helps)
        cond_rel        = cond_delta / loss_shuf      (fractional, scale-free)
    This is the image `early_text_delta`: it works even when generations are
    garbage (early training), because it reads the loss gradient directly and
    needs no sampling. No model edits — reads outputs["image_diffusion_loss_raw"].

  PROBE 2 — clip  (caption-shuffle CLIPScore delta; the PERCEPTUAL check)
    Generates an image from the TRUE caption and from a MISMATCHED caption, and
    CLIP-scores BOTH against the true caption:
        clip_true       = CLIP(gen_from_true_caption, true_caption)
        clip_shuf       = CLIP(gen_from_shuffled_caption, true_caption)
        clip_delta      = clip_true - clip_shuf       (>0 ⇒ caption steers output)
        clip_ceiling    = CLIP(decoded REAL latent, true_caption)   (VAE+CLIP ceiling)
    Plus a retrieval read on the true-caption generations (free once we have the
    features): does gen_i match caption_i better than the other captions?
        retrieval@1     = fraction where own caption is the argmax
        retrieval_margin= mean(matched sim) - mean(mismatched sim)
    CAVEAT: on an under-trained DiT, CLIPScore is dominated by raw image quality,
    not conditioning — clip_delta can read ~0 while cond_delta already shows a
    real signal. Trust `cond` earlier in training; `clip` confirms it perceptually
    once images are legible. clip_ceiling tells you how much of a low clip_true is
    the DiT vs the VAE/CLIP.

Reading the numbers (rough guidance; establish your own baselines per run):
  - cond_rel  ~0.00      : no conditioning (marginal model).  ~0.05-0.15+ : real.
  - clip_delta ~0.00     : output ignores the caption.        >~0.02 : steering.
  - clip_true well below clip_ceiling : DiT undertrained OR conditioning weak
                                        (use cond to disambiguate).

Usage (image-isolation run; text rides inline in the image shards, so
--include_modes image alone self-conditions):
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 -u scripts_local/image_synthesis_diagnostics.py \
    --checkpoint_path runs/world/world_image_litevae_0/checkpoint-20000 \
    --config small_sum --include_modes image \
    --image_cache_dir ./cached_datasets/image_dit_val_merged \
    --image_vae_decoder_config litevae --bf16 \
    --probes cond,clip --n_samples 64 --cond_repeats 8 \
    --output_dir ./eval_output/image_diag/world_image_litevae_0_20000

cond alone is cheap and needs no VAE decoder; clip needs --image_vae_decoder_config.
"""

import argparse
import os
import random
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.amp import autocast

# Reuse the stable loaders from the shipped eval script (no argparse at import).
from megatransformer.scripts.eval.world.eval_image_synthesis import (
    load_world_model,
    load_dataset,
    load_image_decoder,
    decode_latent_to_pixels,
)
from megatransformer.scripts.data.world.data_collator import MultimodalDataCollator


def parse_args():
    p = argparse.ArgumentParser(description="World-image synthesis diagnostics")
    # model / data (superset compatible with eval_image_synthesis loaders)
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--config", type=str, default="small_sum")
    p.add_argument("--include_modes", type=str, default="image")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--text_cache_dir", type=str, default=None)
    p.add_argument("--image_cache_dir", type=str, default=None)
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--use_memorization_dataset", action="store_true")
    p.add_argument("--tie_word_embeddings", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--image_no_bridge", action="store_true", default=False,
                   help="Match a checkpoint trained with --image_no_bridge: null the DiT's Q-Former "
                        "bridge after load so the DiT cross-attends the trunk outputs directly. "
                        "REQUIRED for nobridge runs — otherwise a random-init bridge is used and the "
                        "read is garbage. DiT/trunk weights are identical either way.")
    p.add_argument("--device", type=str, default=None)
    # decoder (needed for clip probe only)
    p.add_argument("--image_vae_decoder_config", type=str, default=None)
    p.add_argument("--image_vae_decoder_path", type=str, default=None)
    # clip
    p.add_argument("--clip_model", type=str, default="ViT-B-32")
    p.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    # probe control
    p.add_argument("--probes", type=str, default="cond,clip",
                   help="Comma list: cond,clip (default both)")
    p.add_argument("--n_samples", type=int, default=64,
                   help="Number of (image,caption) pairs to probe")
    p.add_argument("--cond_repeats", type=int, default=8,
                   help="Random-t draws per sample for the teacher-forced loss (averages the ~16%% noise)")
    p.add_argument("--cond_batch_size", type=int, default=16,
                   help="Batch size for the teacher-forced forward passes")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    # output
    p.add_argument("--output_dir", type=str, default=None,
                   help="Dir for markdown report + example gen/real triptychs")
    p.add_argument("--num_save_images", type=int, default=8)
    p.add_argument("--log_dir", type=str, default=None, help="TensorBoard log dir")
    p.add_argument("--step", type=int, default=None)
    return p.parse_args()


def extract_caption(sample):
    """Return the caption string for an image sample (shards store raw text)."""
    t = sample.get("text_text", None)
    if isinstance(t, list):
        t = t[0] if t else ""
    if t is None:
        t = ""
    return str(t).strip()


def to_int(x):
    return int(x.item()) if isinstance(x, torch.Tensor) else int(x)


def collect_samples(dataset, n, seed):
    """Pick up to n image samples that have both a latent and a non-empty caption."""
    idxs = list(range(len(dataset)))
    random.Random(seed).shuffle(idxs)
    out = []
    for i in idxs:
        s = dataset[i]
        if "image_image" not in s:
            continue
        cap = extract_caption(s)
        if not cap:
            continue
        if "text_token_ids" not in s or "text_text_length" not in s:
            continue
        out.append({
            "image_image": s["image_image"],
            "text_token_ids": s["text_token_ids"],
            "text_text_length": s["text_text_length"],
            "text_text": cap,
        })
        if len(out) >= n:
            break
    return out


def make_shuffle(n, seed):
    """Derangement-ish mismatch: pair index i with a different caption index."""
    if n <= 1:
        return [0] * n
    perm = list(range(n))
    random.Random(seed + 1).shuffle(perm)
    # fix any fixed points so every image gets a genuinely different caption
    for i in range(n):
        if perm[i] == i:
            perm[i], perm[(i + 1) % n] = perm[(i + 1) % n], perm[i]
    return perm


# ─────────────────────────────── PROBE 1: cond ───────────────────────────────

def teacher_forced_loss(model, examples, collator, device, dtype, bf16):
    """Mean raw velocity-MSE (image_diffusion_loss_raw) over a batch of examples,
    forced to the synthesis direction. Each call draws a fresh random t/noise."""
    batch = collator(examples)
    text_input_ids = batch["text_token_ids"][:, :-1].contiguous().to(device)
    image_data = batch["image_images"].to(device)
    image_inputs = image_data.unsqueeze(1)               # (B,1,C,H,W)
    image_latent_labels = image_data.clone()
    is_synthesis = batch["is_synthesis"].to(device)
    with torch.no_grad():
        with autocast(device, dtype=dtype, enabled=bf16):
            out = model(
                text_input_ids=text_input_ids,
                image_inputs=image_inputs,
                image_latent_labels=image_latent_labels,
                precomputed_latents=True,
                decode_outputs=False,
                is_synthesis=is_synthesis,
            )
    raw = out.get("image_diffusion_loss_raw")
    if raw is None:
        raw = out.get("image_diffusion_loss")
    return float(raw.item())


def run_cond_probe(model, samples, shuffle, device, dtype, args):
    collator = MultimodalDataCollator(max_seq_len=args.max_seq_len)
    collator.force_direction = "synthesis"

    n = len(samples)
    bs = max(1, args.cond_batch_size)
    true_losses, shuf_losses = [], []
    # Per-repeat delta estimates: each repeat re-draws t/noise over the SAME n samples,
    # so the std across repeats / sqrt(R) is a proper standard error for the t/noise
    # variance — exactly the right error bar for a same-samples checkpoint-vs-checkpoint
    # comparison (the sample set is fixed by --seed, so it cancels between checkpoints).
    per_repeat_true, per_repeat_shuf, per_repeat_delta = [], [], []

    for r in range(args.cond_repeats):
        torch.manual_seed(args.seed + 1000 * r)  # reproducible; not relied on for pairing
        rt, rs = [], []
        for start in range(0, n, bs):
            idx = list(range(start, min(start + bs, n)))
            true_ex = [samples[i] for i in idx]
            shuf_ex = [{
                "image_image": samples[i]["image_image"],
                "text_token_ids": samples[shuffle[i]]["text_token_ids"],
                "text_text_length": samples[shuffle[i]]["text_text_length"],
                "text_text": samples[shuffle[i]]["text_text"],
            } for i in idx]
            # weight each batch loss by its sample count so the repeat mean is the
            # true per-sample mean even when the last batch is short
            w = len(idx)
            tl = teacher_forced_loss(model, true_ex, collator, device, dtype, args.bf16)
            sl = teacher_forced_loss(model, shuf_ex, collator, device, dtype, args.bf16)
            rt.append((tl, w)); rs.append((sl, w))
            true_losses.append(tl); shuf_losses.append(sl)
        tw = sum(w for _, w in rt)
        true_r = sum(v * w for v, w in rt) / tw
        shuf_r = sum(v * w for v, w in rs) / tw
        per_repeat_true.append(true_r)
        per_repeat_shuf.append(shuf_r)
        per_repeat_delta.append(shuf_r - true_r)
        print(f"  [cond] repeat {r+1}/{args.cond_repeats}  "
              f"true={sum(per_repeat_true)/len(per_repeat_true):.4f}  "
              f"shuf={sum(per_repeat_shuf)/len(per_repeat_shuf):.4f}  "
              f"delta_r={per_repeat_delta[-1]:+.4f}", flush=True)

    import statistics as st
    loss_true = st.mean(per_repeat_true)
    loss_shuf = st.mean(per_repeat_shuf)
    delta = st.mean(per_repeat_delta)
    R = len(per_repeat_delta)
    delta_se = (st.stdev(per_repeat_delta) / (R ** 0.5)) if R > 1 else float("nan")
    rel = delta / loss_shuf if loss_shuf > 0 else 0.0
    rel_se = delta_se / loss_shuf if loss_shuf > 0 else float("nan")
    return {
        "cond_loss_true": loss_true,
        "cond_loss_shuf": loss_shuf,
        "cond_delta": delta,
        "cond_delta_se": delta_se,
        "cond_rel": rel,
        "cond_rel_se": rel_se,
        "cond_repeats": R,
        "cond_n_forward": len(true_losses) + len(shuf_losses),
    }


# ─────────────────────────────── PROBE 2: clip ───────────────────────────────

def build_clip(args, device):
    import open_clip
    from torchvision import transforms
    model, _, _ = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained, device=device)
    model.eval()
    tok = open_clip.get_tokenizer(args.clip_model)
    normalize = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    return model, tok, normalize


@torch.no_grad()
def clip_image_feat(clip_model, normalize, pixels, device):
    x = normalize(pixels).unsqueeze(0).to(device)
    f = clip_model.encode_image(x)
    return F.normalize(f, dim=-1)


@torch.no_grad()
def clip_text_feat(clip_model, tok, text, device):
    t = tok([text[:77]]).to(device)
    f = clip_model.encode_text(t)
    return F.normalize(f, dim=-1)


def generate_latent(model, caption, tokenizer, device, dtype, args):
    from megatransformer.utils.constants import BOI_TOKEN_ID
    ids = tokenizer.encode(caption[:500], add_special_tokens=True)
    max_prompt = args.max_seq_len - args.max_new_tokens - 1
    ids = ids[:max(1, max_prompt)] + [BOI_TOKEN_ID]
    prompt = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        with autocast(device, dtype=dtype, enabled=args.bf16):
            out = model.generate(text_input_ids=prompt,
                                 max_new_tokens=args.max_new_tokens,
                                 temperature=args.temperature)
    preds = out.get("image_latent_preds")
    if preds is None or preds.numel() == 0:
        return None
    return preds[0, 0]  # (C,H,W)


def run_clip_probe(model, samples, shuffle, decoder, device, dtype, args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    clip_model, clip_tok, clip_norm = build_clip(args, device)

    save_dir = None
    if args.output_dir and args.num_save_images > 0:
        save_dir = os.path.join(args.output_dir, "images")
        os.makedirs(save_dir, exist_ok=True)

    clip_true, clip_shuf, clip_ceiling = [], [], []
    gen_true_feats, cap_feats = [], []

    for i, s in enumerate(samples):
        cap = s["text_text"]
        shuf_cap = samples[shuffle[i]]["text_text"]

        gen_t = generate_latent(model, cap, tokenizer, device, dtype, args)
        gen_s = generate_latent(model, shuf_cap, tokenizer, device, dtype, args)
        if gen_t is None or gen_s is None:
            print(f"  [clip] {i}: generation failed, skipping", flush=True)
            continue

        px_t = decode_latent_to_pixels(decoder, gen_t, device)
        px_s = decode_latent_to_pixels(decoder, gen_s, device)
        px_real = decode_latent_to_pixels(decoder, s["image_image"], device)

        cap_f = clip_text_feat(clip_model, clip_tok, cap, device)
        ft = clip_image_feat(clip_model, clip_norm, px_t, device)
        fs = clip_image_feat(clip_model, clip_norm, px_s, device)
        fr = clip_image_feat(clip_model, clip_norm, px_real, device)

        ct = (ft @ cap_f.T).item()
        cs = (fs @ cap_f.T).item()
        cr = (fr @ cap_f.T).item()
        clip_true.append(ct); clip_shuf.append(cs); clip_ceiling.append(cr)
        gen_true_feats.append(ft.squeeze(0)); cap_feats.append(cap_f.squeeze(0))

        if save_dir and i < args.num_save_images:
            from torchvision.utils import save_image
            trip = torch.stack([px_real, px_t, px_s], dim=0)  # real | gen(true) | gen(shuf)
            save_image(trip, os.path.join(save_dir, f"{i:03d}_real_genTrue_genShuf.png"), nrow=3)

        print(f"  [clip] {i}: true={ct:.4f} shuf={cs:.4f} ceil={cr:.4f}  '{cap[:60]}'", flush=True)

    if not clip_true:
        return {}

    import statistics as st
    res = {
        "clip_true": st.mean(clip_true),
        "clip_shuf": st.mean(clip_shuf),
        "clip_delta": st.mean(clip_true) - st.mean(clip_shuf),
        "clip_ceiling": st.mean(clip_ceiling),
        "clip_n": len(clip_true),
    }
    # retrieval read on the true-caption generations
    if len(gen_true_feats) >= 3:
        I = torch.stack(gen_true_feats)   # (N,D)
        Tt = torch.stack(cap_feats)       # (N,D)
        sim = I @ Tt.T                    # (N,N): row=gen image, col=caption
        n = sim.shape[0]
        r1 = (sim.argmax(dim=1) == torch.arange(n)).float().mean().item()
        matched = sim.diag().mean().item()
        eye = torch.eye(n, dtype=torch.bool)
        mismatched = sim[~eye].mean().item()
        res["clip_retrieval@1"] = r1
        res["clip_retrieval_margin"] = matched - mismatched
    return res


# ─────────────────────────────── report ───────────────────────────────

def write_report(args, cond_res, clip_res, out_dir):
    lines = ["# Image-synthesis diagnostics", ""]
    lines.append(f"- checkpoint: `{args.checkpoint_path}`")
    lines.append(f"- config: `{args.config}`  split: `{args.split}`  n_samples: {args.n_samples}")
    lines.append("")
    if cond_res:
        lines += [
            "## Probe: teacher-forced conditioning-loss delta (cond)",
            "",
            f"- loss_true (true caption):     {cond_res['cond_loss_true']:.4f}",
            f"- loss_shuf (mismatched caption): {cond_res['cond_loss_shuf']:.4f}",
            f"- **cond_delta = shuf - true:   {cond_res['cond_delta']:+.4f} ± {cond_res.get('cond_delta_se', float('nan')):.4f}** (±1 SE)",
            f"- **cond_rel  = delta / shuf:   {cond_res['cond_rel']:+.4f} ± {cond_res.get('cond_rel_se', float('nan')):.4f}**   (>0 ⇒ caption reduces velocity error)",
            f"- forward passes: {cond_res['cond_n_forward']} ({cond_res.get('cond_repeats', args.cond_repeats)} repeats)",
            "",
        ]
    if clip_res:
        lines += [
            "## Probe: caption-shuffle CLIPScore delta (clip)",
            "",
            f"- clip_true (gen from true cap vs true cap):   {clip_res['clip_true']:.4f}",
            f"- clip_shuf (gen from shuffled cap vs true cap): {clip_res['clip_shuf']:.4f}",
            f"- **clip_delta = true - shuf:   {clip_res['clip_delta']:+.4f}**   (>0 ⇒ caption steers output)",
            f"- clip_ceiling (real decoded vs true cap):     {clip_res['clip_ceiling']:.4f}   (VAE+CLIP ceiling)",
        ]
        if "clip_retrieval@1" in clip_res:
            lines += [
                f"- retrieval@1:      {clip_res['clip_retrieval@1']:.3f}   (chance = 1/{clip_res['clip_n']})",
                f"- retrieval_margin: {clip_res['clip_retrieval_margin']:+.4f}   (matched - mismatched sim)",
            ]
        lines += [f"- samples scored: {clip_res['clip_n']}", ""]
    txt = "\n".join(lines)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "report.md"), "w") as f:
            f.write(txt + "\n")
    return txt


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    probes = [p.strip() for p in args.probes.split(",") if p.strip()]
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading world model from {args.checkpoint_path} ...", flush=True)
    model = load_world_model(args, device)
    if args.image_no_bridge and getattr(model, "image_generator", None) is not None:
        # Match a --image_no_bridge checkpoint: discard the (checkpoint-absent, random-init)
        # bridge so _compute_conditioning cross-attends the trunk outputs directly.
        model.image_generator.config.use_bridge = False
        model.image_generator.bridge = None
        print("  [image_no_bridge] bridge nulled — DiT cross-attends trunk outputs directly", flush=True)
    model.to(device)
    model.eval()

    print("Loading dataset ...", flush=True)
    dataset = load_dataset(args, split=args.split)
    print(f"Dataset: {len(dataset)} samples", flush=True)

    samples = collect_samples(dataset, args.n_samples, args.seed)
    if len(samples) < 2:
        print("ERROR: fewer than 2 usable (image+caption) samples found.")
        sys.exit(1)
    print(f"Probing {len(samples)} samples", flush=True)
    shuffle = make_shuffle(len(samples), args.seed)

    cond_res, clip_res = {}, {}

    if "cond" in probes:
        print("\n=== PROBE: teacher-forced conditioning-loss delta ===", flush=True)
        cond_res = run_cond_probe(model, samples, shuffle, device, dtype, args)

    if "clip" in probes:
        print("\n=== PROBE: caption-shuffle CLIPScore delta ===", flush=True)
        decoder = load_image_decoder(args)
        if decoder is None:
            print("  SKIP clip probe: need --image_vae_decoder_config litevae (or --image_vae_decoder_path)")
        else:
            clip_res = run_clip_probe(model, samples, shuffle, decoder, device, dtype, args)

    print("\n" + "=" * 60)
    print(write_report(args, cond_res, clip_res, args.output_dir))
    print("=" * 60)

    if args.log_dir:
        from megatransformer.scripts.eval.world.eval_utils import (
            infer_step_from_checkpoint, init_eval_metrics, log_eval_scalars)
        step = args.step if args.step is not None else infer_step_from_checkpoint(args.checkpoint_path)
        init_eval_metrics(args.log_dir, args.checkpoint_path)
        md = {}
        if cond_res:
            md["eval/image_cond_delta"] = cond_res["cond_delta"]
            md["eval/image_cond_rel"] = cond_res["cond_rel"]
        if clip_res:
            md["eval/image_clip_delta"] = clip_res["clip_delta"]
            md["eval/image_clip_true"] = clip_res["clip_true"]
            md["eval/image_clip_ceiling"] = clip_res["clip_ceiling"]
            if "clip_retrieval_margin" in clip_res:
                md["eval/image_clip_retrieval_margin"] = clip_res["clip_retrieval_margin"]
        if md:
            log_eval_scalars(md, step)


if __name__ == "__main__":
    main()
