"""Z-Image (Turbo) conditioning-tolerance probe — step 1 of the LLM-decoder upgrade.

Mirror of sdxl_conditioning_tolerance.py, but for the frozen Z-Image-Turbo decoder,
whose text conditioning is Qwen3-4B penultimate hidden states: a VARIABLE-length
[L x 2560] sequence, NO pooled vector, consumed by the S3-DiT via cross-attention.
Turbo = 8 NFEs, guidance_scale=0.0 (no CFG, no negatives).

The linchpin question (same as SDXL): how close must a learned adapter's predicted
Qwen3 conditioning be to the true embedding before Z-Image's output falls apart? That
tolerance = the adapter's regression-error budget. Unlike CLIP, Qwen3 states are
causal-LM features (NOT contrastive), so we do NOT trust cosine as the metric — we
read the CLIPScore-vs-perturbation curve on the rendered pixels.

Perturbations (fixed latent seed per caption, only conditioning varies):
  - additive Gaussian noise at increasing fractions of the embedding std
  - linear interpolation toward a DIFFERENT caption's embedding (truncated to the
    shorter sequence, since lengths vary)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts_local/zimage_conditioning_tolerance.py
"""

import os
import csv
import torch
import torch.nn.functional as F
from PIL import Image

OUT = "eval_output/zimage_cond_tolerance"
MODEL = "Tongyi-MAI/Z-Image-Turbo"
STEPS = 8            # Turbo
GUIDANCE = 0.0       # Turbo: CFG disabled
SEED = 1234
NOISE_LEVELS = [0.05, 0.1, 0.2, 0.4, 0.8]     # * per-tensor std
INTERP_LEVELS = [0.15, 0.35, 0.6]             # lerp toward another caption
CAPTIONS = [
    "a red fox curled up asleep in fresh snow, soft morning light",
    "a bustling night market street with neon signs and steam rising from food stalls",
    "a lighthouse on a rocky cliff during a violent storm, huge crashing waves",
    "a still life of lemons and a ceramic jug on a wooden table, oil painting",
    "an astronaut riding a horse across a desert under a pink sky",
    "a cozy bookstore interior with tall shelves and a cat sleeping on a chair",
]


def clip_scorer(device):
    import open_clip
    from torchvision import transforms
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
    model.eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    norm = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])

    @torch.no_grad()
    def score(pil_img, text):
        import numpy as np
        x = torch.from_numpy(np.asarray(pil_img).copy()).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
        x = norm(x)
        t = tok([text[:77]]).to(device)
        im = F.normalize(model.encode_image(x), dim=-1)
        tx = F.normalize(model.encode_text(t), dim=-1)
        return float((im @ tx.T).item())

    return score


def main():
    os.makedirs(OUT, exist_ok=True)
    device = "cuda"
    dtype = torch.bfloat16

    from diffusers import ZImagePipeline
    print(f"Loading {MODEL} ...", flush=True)
    pipe = ZImagePipeline.from_pretrained(MODEL, torch_dtype=dtype)
    # ~20GB stack (6B DiT + Qwen3-4B + Flux VAE) — offload to fit a 24GB card with
    # room for the CLIP scorer. Turbo is only 8 steps so offload overhead is small.
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    score = clip_scorer(device)

    @torch.no_grad()
    def encode(caption):
        # Turbo: no CFG -> encode without negatives. Returns (list[[L,2560]], _).
        pe_list, _ = pipe.encode_prompt(
            prompt=caption, device=device, do_classifier_free_guidance=False,
            max_sequence_length=512)
        return pe_list[0]  # [L, 2560]

    @torch.no_grad()
    def gen(seq, seed):
        g = torch.Generator(device=device).manual_seed(seed)
        img = pipe(prompt_embeds=[seq], num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                   height=1024, width=1024, generator=g).images[0]
        return img

    def cos(a, b):
        m = min(a.shape[0], b.shape[0])
        return float(F.cosine_similarity(a[:m].flatten().float(), b[:m].flatten().float(), dim=0).item())

    # Precompute embeddings for all captions (interp needs a partner).
    embeds = [encode(c) for c in CAPTIONS]
    print("Encoded conditioning shapes:", [tuple(e.shape) for e in embeds], flush=True)

    rows = []
    noise_grid = []
    interp_grid = []

    for i, cap in enumerate(CAPTIONS):
        pe = embeds[i]
        seed_i = SEED + i
        base_img = gen(pe, seed_i)
        base_score = score(base_img, cap)
        base_img.save(os.path.join(OUT, f"c{i}_baseline.png"))
        rows.append(dict(cap=i, kind="baseline", level=0.0, seq_cos=1.0, seq_len=pe.shape[0],
                         clip=base_score, clip_drop=0.0))
        print(f"[c{i}] baseline CLIP={base_score:.3f}  (L={pe.shape[0]}) | {cap[:50]}", flush=True)
        noise_row = [base_img]
        interp_row = [base_img]

        # --- Gaussian noise sweep ---
        std = pe.float().std()
        for lv in NOISE_LEVELS:
            gseed = torch.Generator(device=device).manual_seed(9000 + i * 10 + int(lv * 100))
            pe_n = pe + torch.randn(pe.shape, generator=gseed, device=device, dtype=pe.dtype) * (std * lv)
            img = gen(pe_n, seed_i)
            sc = score(img, cap)
            rows.append(dict(cap=i, kind="noise", level=lv, seq_cos=cos(pe_n, pe), seq_len=pe.shape[0],
                             clip=sc, clip_drop=base_score - sc))
            img.save(os.path.join(OUT, f"c{i}_noise{lv}.png"))
            noise_row.append(img)
            print(f"[c{i}] noise {lv:>4}: seq_cos={cos(pe_n,pe):.3f} CLIP={sc:.3f} (drop {base_score-sc:+.3f})", flush=True)
        noise_grid.append(noise_row)

        # --- interpolation toward another caption (truncate to shorter length) ---
        j = (i + 1) % len(CAPTIONS)
        pe_o = embeds[j]
        m = min(pe.shape[0], pe_o.shape[0])
        for a in INTERP_LEVELS:
            pe_m = (1 - a) * pe[:m] + a * pe_o[:m]
            img = gen(pe_m, seed_i)
            sc = score(img, cap)
            rows.append(dict(cap=i, kind="interp", level=a, seq_cos=cos(pe_m, pe), seq_len=m,
                             clip=sc, clip_drop=base_score - sc))
            img.save(os.path.join(OUT, f"c{i}_interp{a}.png"))
            interp_row.append(img)
            print(f"[c{i}] interp {a:>4} ->c{j}: seq_cos={cos(pe_m,pe):.3f} CLIP={sc:.3f} (drop {base_score-sc:+.3f})", flush=True)
        interp_grid.append(interp_row)

    # --- montages ---
    def montage(grid, path, thumb=256):
        rows_n = len(grid); cols_n = max(len(r) for r in grid)
        canvas = Image.new("RGB", (cols_n * thumb, rows_n * thumb), (20, 20, 20))
        for r, row in enumerate(grid):
            for c, im in enumerate(row):
                canvas.paste(im.resize((thumb, thumb)), (c * thumb, r * thumb))
        canvas.save(path)
        print(f"saved {path}")

    montage(noise_grid, os.path.join(OUT, "montage_noise.png"))
    montage(interp_grid, os.path.join(OUT, "montage_interp.png"))

    # --- csv + summary ---
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    import statistics as st
    print("\n=== TOLERANCE SUMMARY (aggregate over captions) ===")
    print("Noise sweep: seq_cos -> mean CLIP drop")
    for lv in NOISE_LEVELS:
        rs = [r for r in rows if r["kind"] == "noise" and r["level"] == lv]
        print(f"  noise {lv:>4}: seq_cos~{st.mean(r['seq_cos'] for r in rs):.3f}  CLIP drop {st.mean(r['clip_drop'] for r in rs):+.3f}")
    print("Interp sweep:")
    for a in INTERP_LEVELS:
        rs = [r for r in rows if r["kind"] == "interp" and r["level"] == a]
        print(f"  interp {a:>4}: seq_cos~{st.mean(r['seq_cos'] for r in rs):.3f}  CLIP drop {st.mean(r['clip_drop'] for r in rs):+.3f}")
    print(f"\nbaseline mean CLIP = {st.mean(r['clip'] for r in rows if r['kind']=='baseline'):.3f}")
    print(f"outputs in {OUT}/  (montage_noise.png, montage_interp.png, results.csv)")


if __name__ == "__main__":
    main()
