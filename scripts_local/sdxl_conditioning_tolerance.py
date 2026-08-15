"""SDXL conditioning-tolerance probe.

De-risks the Option-B image plan (frozen SDXL + a learned adapter that regresses
CLIP conditioning from the world-model trunk). The linchpin question: how CLOSE
must the adapter's predicted conditioning be to the true CLIP embedding before
SDXL's output quality/adherence falls apart? That tolerance = the adapter's
regression-error budget.

Method: for each caption, get the true SDXL conditioning
(prompt_embeds 77x2048 + pooled 1280), then generate with GRADED perturbations:
  - additive Gaussian noise at increasing fractions of the embedding std
  - linear interpolation toward a DIFFERENT caption's embedding
For every perturbation we record the embedding cosine-to-truth (sequence + pooled)
and the CLIPScore of the generated image vs the ORIGINAL caption, and save montage
grids for the eye. The CLIPScore-vs-cosine curve is the tolerance calibration.

Fixed latent seed per caption, so ONLY the conditioning varies.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts_local/sdxl_conditioning_tolerance.py
"""

import os
import csv
import torch
import torch.nn.functional as F
from PIL import Image

OUT = "eval_output/sdxl_cond_tolerance"
MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
STEPS = 22
GUIDANCE = 7.0
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
        x = torch.from_numpy(
            __import__("numpy").asarray(pil_img).copy()
        ).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
        x = norm(x)
        t = tok([text[:77]]).to(device)
        im = F.normalize(model.encode_image(x), dim=-1)
        tx = F.normalize(model.encode_text(t), dim=-1)
        return float((im @ tx.T).item())

    return score


def main():
    os.makedirs(OUT, exist_ok=True)
    device = "cuda"
    dtype = torch.float16

    from diffusers import StableDiffusionXLPipeline
    print(f"Loading {MODEL} ...", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, torch_dtype=dtype, use_safetensors=True)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    score = clip_scorer(device)

    @torch.no_grad()
    def encode(caption):
        pe, npe, ppe, nppe = pipe.encode_prompt(
            prompt=caption, device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=True, negative_prompt="")
        return pe, npe, ppe, nppe

    @torch.no_grad()
    def gen(pe, ppe, npe, nppe, seed):
        g = torch.Generator(device=device).manual_seed(seed)
        img = pipe(prompt_embeds=pe, pooled_prompt_embeds=ppe,
                   negative_prompt_embeds=npe, negative_pooled_prompt_embeds=nppe,
                   num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                   height=1024, width=1024, generator=g).images[0]
        return img

    def cos(a, b):
        return float(F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item())

    # Precompute embeddings for all captions (interp needs a partner).
    embeds = [encode(c) for c in CAPTIONS]

    rows = []
    noise_grid = []   # list of rows (each row = list of PIL imgs) for the noise montage
    interp_grid = []

    for i, cap in enumerate(CAPTIONS):
        pe, npe, ppe, nppe = embeds[i]
        seed_i = SEED + i
        # --- baseline (true conditioning) ---
        base_img = gen(pe, ppe, npe, nppe, seed_i)
        base_score = score(base_img, cap)
        base_img.save(os.path.join(OUT, f"c{i}_baseline.png"))
        rows.append(dict(cap=i, kind="baseline", level=0.0, seq_cos=1.0, pooled_cos=1.0,
                         clip=base_score, clip_drop=0.0))
        print(f"[c{i}] baseline CLIP={base_score:.3f}  | {cap[:50]}", flush=True)
        noise_row = [base_img]
        interp_row = [base_img]

        # --- Gaussian noise sweep ---
        for lv in NOISE_LEVELS:
            gseed = torch.Generator(device=device).manual_seed(9000 + i * 10 + int(lv * 100))
            pe_n = pe + torch.randn(pe.shape, generator=gseed, device=device, dtype=pe.dtype) * (pe.float().std() * lv)
            ppe_n = ppe + torch.randn(ppe.shape, generator=gseed, device=device, dtype=ppe.dtype) * (ppe.float().std() * lv)
            img = gen(pe_n, ppe_n, npe, nppe, seed_i)
            sc = score(img, cap)
            rows.append(dict(cap=i, kind="noise", level=lv,
                             seq_cos=cos(pe_n, pe), pooled_cos=cos(ppe_n, ppe),
                             clip=sc, clip_drop=base_score - sc))
            img.save(os.path.join(OUT, f"c{i}_noise{lv}.png"))
            noise_row.append(img)
            print(f"[c{i}] noise {lv:>4}: seq_cos={cos(pe_n,pe):.3f} pooled_cos={cos(ppe_n,ppe):.3f} CLIP={sc:.3f} (drop {base_score-sc:+.3f})", flush=True)
        noise_grid.append(noise_row)

        # --- interpolation toward another caption ---
        j = (i + 1) % len(CAPTIONS)
        pe_o, _, ppe_o, _ = embeds[j]
        for a in INTERP_LEVELS:
            pe_m = (1 - a) * pe + a * pe_o
            ppe_m = (1 - a) * ppe + a * ppe_o
            img = gen(pe_m, ppe_m, npe, nppe, seed_i)
            sc = score(img, cap)
            rows.append(dict(cap=i, kind="interp", level=a,
                             seq_cos=cos(pe_m, pe), pooled_cos=cos(ppe_m, ppe),
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

    print("\n=== TOLERANCE SUMMARY (aggregate over captions) ===")
    print("Noise sweep: seq_cos -> mean CLIP drop")
    for lv in NOISE_LEVELS:
        rs = [r for r in rows if r["kind"] == "noise" and r["level"] == lv]
        import statistics as st
        print(f"  noise {lv:>4}: seq_cos~{st.mean(r['seq_cos'] for r in rs):.3f} pooled_cos~{st.mean(r['pooled_cos'] for r in rs):.3f}  CLIP drop {st.mean(r['clip_drop'] for r in rs):+.3f}")
    print("Interp sweep:")
    for a in INTERP_LEVELS:
        rs = [r for r in rows if r["kind"] == "interp" and r["level"] == a]
        import statistics as st
        print(f"  interp {a:>4}: seq_cos~{st.mean(r['seq_cos'] for r in rs):.3f}  CLIP drop {st.mean(r['clip_drop'] for r in rs):+.3f}")
    import statistics as st
    print(f"\nbaseline mean CLIP = {st.mean(r['clip'] for r in rows if r['kind']=='baseline'):.3f}")
    print(f"outputs in {OUT}/  (montage_noise.png, montage_interp.png, results.csv)")


if __name__ == "__main__":
    main()
