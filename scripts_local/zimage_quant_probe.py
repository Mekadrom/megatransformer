"""Measure 4-bit vs bf16 Qwen3 conditioning error for the Z-Image adapter target.

The adapter trains toward the 4-bit Qwen3 penultimate features, but Z-Image's DiT was
trained on full-precision features. This quantifies the gap, two ways:
  (1) feature-space: flat cosine, per-token cosine, relative-L2 between the resampled-K
      target computed in bf16 vs 4-bit nf4 (the ACTUAL training-target code path).
  (2) downstream: render Z-Image from each (same seed) and CLIPScore -- does the gap
      actually move the image? (the honest test, per the tolerance-probe lesson).

Reference for "safe" = the tolerance probe's noise sweep: cos~0.995 renders identically,
cos~0.98 is where content starts to drift.

Usage: CUDA_VISIBLE_DEVICES=3 python scripts_local/zimage_quant_probe.py
"""
import os, numpy as np, torch, torch.nn.functional as F
from PIL import Image

DEVICE = "cuda"; K = 64
OUT = "eval_output/zimage_cond_tolerance/quant"; os.makedirs(OUT, exist_ok=True)
CAPS = [
    "a red fox curled up asleep in fresh snow, soft morning light",
    "a bustling night market street with neon signs and steam rising from food stalls",
    "an astronaut riding a horse across a desert under a pink sky",
    "a cozy bookstore interior with tall shelves and a cat sleeping on a chair",
    "a lighthouse on a rocky cliff during a violent storm, huge crashing waves",
]

from megatransformer.utils.zimage_text_encoder import Qwen3TextTargetEncoder

# ── Phase 1: features (bf16 reference vs 4-bit) ──
print("loading bf16 + 4-bit Qwen3 encoders...", flush=True)
enc_bf16 = Qwen3TextTargetEncoder(seq_len=K, device=DEVICE, load_in_4bit=False)
enc_q4 = Qwen3TextTargetEncoder(seq_len=K, device=DEVICE, load_in_4bit=True)
fb = enc_bf16.encode(CAPS).float().cpu()   # (B,K,2560)
fq = enc_q4.encode(CAPS).float().cpu()
print("bf16 4bit unavailable?" , enc_q4.model.dtype if hasattr(enc_q4.model,'dtype') else "?", flush=True)

print("\n=== FEATURE-SPACE ERROR (bf16 vs 4-bit target) ===")
flat_cos, tok_cos, rel_l2 = [], [], []
for i, cap in enumerate(CAPS):
    fc = float(F.cosine_similarity(fb[i].flatten(), fq[i].flatten(), dim=0))
    tc = float(F.cosine_similarity(fb[i], fq[i], dim=-1).mean())   # mean over K tokens
    rl = float((fb[i] - fq[i]).norm() / fb[i].norm())
    flat_cos.append(fc); tok_cos.append(tc); rel_l2.append(rl)
    print(f"[c{i}] flat_cos={fc:.4f} per_token_cos={tc:.4f} rel_L2={rl:.4f} | {cap[:40]}", flush=True)
print(f"MEAN flat_cos={np.mean(flat_cos):.4f}  per_token_cos={np.mean(tok_cos):.4f}  rel_L2={np.mean(rel_l2):.4f}")

del enc_bf16, enc_q4
torch.cuda.empty_cache()

# ── Phase 2: downstream render (bf16-target vs 4bit-target through Z-Image) ──
print("\nloading Z-Image for downstream render...", flush=True)
from diffusers import ZImagePipeline
pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload(); pipe.set_progress_bar_config(disable=True)

import open_clip
from torchvision import transforms
cm, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=DEVICE)
cm.eval(); ctok = open_clip.get_tokenizer("ViT-B-32")
cnorm = transforms.Compose([transforms.Resize((224,224), antialias=True),
    transforms.Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))])
@torch.no_grad()
def clipscore(img, txt):
    x = torch.from_numpy(np.asarray(img).copy()).permute(2,0,1).float().div(255).unsqueeze(0).to(DEVICE)
    return float((F.normalize(cm.encode_image(cnorm(x)),dim=-1) @ F.normalize(cm.encode_text(ctok([txt[:77]]).to(DEVICE)),dim=-1).T).item())
@torch.no_grad()
def render(seq, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(prompt_embeds=[seq.to(torch.bfloat16).to(DEVICE)], num_inference_steps=8,
                guidance_scale=0.0, height=1024, width=1024, generator=g).images[0]

print("\n=== DOWNSTREAM RENDER (CLIPScore) ===")
grid, sc_b, sc_q = [], [], []
for i, cap in enumerate(CAPS):
    ib = render(fb[i], 500 + i); iq = render(fq[i], 500 + i)
    sb, sq = clipscore(ib, cap), clipscore(iq, cap)
    sc_b.append(sb); sc_q.append(sq); grid.append((ib, iq))
    print(f"[c{i}] bf16 CLIP={sb:.3f}  4bit CLIP={sq:.3f}  (delta {sq-sb:+.3f}) | {cap[:40]}", flush=True)
print(f"MEAN CLIP bf16={np.mean(sc_b):.3f}  4bit={np.mean(sc_q):.3f}  (delta {np.mean(sc_q)-np.mean(sc_b):+.3f})")

thumb = 320
canvas = Image.new("RGB", (2*thumb, len(CAPS)*thumb), (18,18,18))
for r,(ib,iq) in enumerate(grid):
    canvas.paste(ib.resize((thumb,thumb)), (0, r*thumb))
    canvas.paste(iq.resize((thumb,thumb)), (thumb, r*thumb))
p = os.path.join(OUT, "bf16_vs_4bit_render.png"); canvas.save(p)
print(f"\nmontage: {p}  (left=bf16 target, right=4bit target)")
