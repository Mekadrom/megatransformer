"""Toy adapter: does a contrastive term prevent the basin-collapse the tolerance
probe predicted?

The probe (sdxl_conditioning_tolerance.py) showed SDXL tolerates large RANDOM
conditioning error (cos ~0.93) but not STRUCTURED/semantic error (a move toward
another caption at cos 0.992 flips the scene). So an adapter that mode-collapses
toward the caption-cloud centroid — high average cosine, wrong basin — will look
"good" by L2/cosine yet generate the wrong image. Prediction: an L2-only
regression under-separates the basins; adding an InfoNCE (discriminability) term
fixes it.

This toy stands in a proxy (frozen T5-small caption embedding) for the eventual
trunk gen-query state, and learns a small adapter proxy -> SDXL conditioning
(77x2048 seq + 1280 pooled) through a K=16 latent bottleneck (the gen-query
bandwidth analog). Two adapters, identical but for the loss:
  A: L2 (MSE on seq + pooled)
  B: L2 + InfoNCE (on pooled and seq-mean)
Then: held-out retrieval@1, pred collapse (mean off-diagonal cosine), and — the
real test — feed both to frozen SDXL and CLIPScore + montage vs the TRUE-condition
reference.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts_local/sdxl_adapter_toy.py
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = "eval_output/sdxl_adapter_toy"
MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
PROXY = "google/t5-v1_1-small"
N_TRAIN = 1500
N_EVAL = 8
K = 16
DMODEL = 512
STEPS = 2500
BATCH = 64
LR = 3e-4
GEN_STEPS = 22
GUIDANCE = 7.0
SEED = 7
device = "cuda"


# ---------------- data: captions -> (proxy emb, CLIP seq target, CLIP pooled target) -------------
def get_captions(n):
    from datasets import load_dataset
    ds = load_dataset("common-canvas/commoncatalog-cc-by",
                      data_files="*/least_dim_range=512-768/**/*.parquet",  # 0% junk partition
                      split="train", streaming=True)
    caps, seen = [], set()
    for ex in ds:
        c = (ex.get("blip2_caption") or "").strip()
        if c and c not in seen and len(c) > 8:
            seen.add(c); caps.append(c)
        if len(caps) >= n:
            break
    return caps


@torch.no_grad()
def t5_proxy(captions, tok, enc, bs=64):
    out = []
    for i in range(0, len(captions), bs):
        b = captions[i:i + bs]
        e = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        h = enc(**e).last_hidden_state                       # (B,L,512)
        m = e["attention_mask"].unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)      # mean-pool
        out.append(pooled.float().cpu())
    return torch.cat(out)


@torch.no_grad()
def clip_targets(captions, pipe, bs=16):
    seqs, pools = [], []
    for i in range(0, len(captions), bs):
        b = captions[i:i + bs]
        pe, _, ppe, _ = pipe.encode_prompt(prompt=b, device=device, num_images_per_prompt=1,
                                           do_classifier_free_guidance=False)
        seqs.append(pe.float().cpu()); pools.append(ppe.float().cpu())
    return torch.cat(seqs), torch.cat(pools)                 # (N,77,2048),(N,1280)


# ---------------- adapter ----------------
class Adapter(nn.Module):
    def __init__(self, proxy_dim=512, d=DMODEL, k=K, seq_len=77, seq_dim=2048, pool_dim=1280):
        super().__init__()
        self.k, self.d = k, d
        self.in_proj = nn.Linear(proxy_dim, k * d)
        enc = nn.TransformerEncoderLayer(d, 8, d * 4, batch_first=True, activation="gelu")
        self.mem_enc = nn.TransformerEncoder(enc, 2)
        self.out_q = nn.Parameter(torch.randn(seq_len, d) * 0.02)
        self.cross = nn.MultiheadAttention(d, 8, batch_first=True)
        self.cross_ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
        self.seq_head = nn.Linear(d, seq_dim)
        self.pool_head = nn.Linear(d, pool_dim)

    def forward(self, proxy):                                # proxy (B,proxy_dim)
        B = proxy.shape[0]
        mem = self.in_proj(proxy).view(B, self.k, self.d)    # (B,K,d) latent bottleneck
        mem = self.mem_enc(mem)
        q = self.out_q.unsqueeze(0).expand(B, -1, -1)        # (B,77,d)
        a, _ = self.cross(q, mem, mem)
        h = a + self.cross_ff(a)
        return self.seq_head(h), self.pool_head(mem.mean(1)) # (B,77,2048),(B,1280)


def info_nce(pred, tgt, temp=0.07):
    p = F.normalize(pred, dim=-1); t = F.normalize(tgt, dim=-1)
    logits = p @ t.T / temp
    labels = torch.arange(p.shape[0], device=p.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def train(proxy, tgt_seq, tgt_pool, contrastive, steps=STEPS):
    torch.manual_seed(0)
    net = Adapter().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    N = proxy.shape[0]
    for s in range(steps):
        idx = torch.randint(0, N, (BATCH,))
        px = proxy[idx].to(device)
        ts = tgt_seq[idx].to(device); tp = tgt_pool[idx].to(device)
        ps, pp = net(px)
        loss = F.mse_loss(ps, ts) + F.mse_loss(pp, tp)
        if contrastive:
            loss = loss + info_nce(pp, tp) + info_nce(ps.mean(1), ts.mean(1))
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 500 == 0 or s == steps - 1:
            print(f"    step {s:>4} loss {loss.item():.4f}", flush=True)
    return net.eval()


@torch.no_grad()
def discriminability(net, proxy, tgt_pool):
    ps, pp = net(proxy.to(device))
    pp = pp.cpu()
    # retrieval@1: does pred_i's nearest target (by pooled cosine) = target_i?
    P = F.normalize(pp, dim=-1); T = F.normalize(tgt_pool, dim=-1)
    sim = P @ T.T
    ret1 = (sim.argmax(1) == torch.arange(pp.shape[0])).float().mean().item()
    # collapse: mean off-diagonal cosine of DISTINCT preds (high = collapsed)
    Pn = F.normalize(pp, dim=-1); cc = Pn @ Pn.T
    n = pp.shape[0]
    offdiag = (cc.sum() - cc.diag().sum()) / (n * n - n)
    return ret1, float(offdiag)


def main():
    os.makedirs(OUT, exist_ok=True)
    from diffusers import StableDiffusionXLPipeline
    from transformers import T5EncoderModel, T5Tokenizer

    print("captions...", flush=True)
    caps = get_captions(N_TRAIN + N_EVAL)
    train_caps, eval_caps = caps[:N_TRAIN], caps[N_TRAIN:N_TRAIN + N_EVAL]

    print(f"proxy encoder {PROXY}...", flush=True)
    ttok = T5Tokenizer.from_pretrained(PROXY)
    tenc = T5EncoderModel.from_pretrained(PROXY).to(device).eval()

    print(f"SDXL {MODEL} (text encoders + unet + vae)...", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, torch_dtype=torch.float16, use_safetensors=True).to(device)
    pipe.set_progress_bar_config(disable=True)

    print("precompute proxy + CLIP targets...", flush=True)
    proxy_tr = t5_proxy(train_caps, ttok, tenc)
    seq_tr, pool_tr = clip_targets(train_caps, pipe)
    proxy_ev = t5_proxy(eval_caps, ttok, tenc)
    seq_ev, pool_ev = clip_targets(eval_caps, pipe)

    print("train A (L2 only)...", flush=True)
    net_a = train(proxy_tr, seq_tr, pool_tr, contrastive=False)
    print("train B (L2 + contrastive)...", flush=True)
    net_b = train(proxy_tr, seq_tr, pool_tr, contrastive=True)

    ra, ca = discriminability(net_a, proxy_ev, pool_ev)
    rb, cb = discriminability(net_b, proxy_ev, pool_ev)
    print(f"\n[discriminability on {N_EVAL} held-out]")
    print(f"  A (L2):          retrieval@1={ra:.3f}  pred-collapse(off-diag cos)={ca:.3f}")
    print(f"  B (L2+contrast): retrieval@1={rb:.3f}  pred-collapse(off-diag cos)={cb:.3f}")

    # ---- generate: TRUE cond vs A-pred vs B-pred, held-out ----
    import open_clip
    from torchvision import transforms
    import numpy as np
    from PIL import Image
    cm, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
    cm.eval(); ctok = open_clip.get_tokenizer("ViT-B-32")
    cnorm = transforms.Compose([transforms.Resize((224, 224), antialias=True),
                                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                     (0.26862954, 0.26130258, 0.27577711))])

    @torch.no_grad()
    def clipscore(img, txt):
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
        im = F.normalize(cm.encode_image(cnorm(x)), dim=-1)
        tx = F.normalize(cm.encode_text(ctok([txt[:77]]).to(device)), dim=-1)
        return float((im @ tx.T).item())

    neg_pe, _, neg_pp, _ = pipe.encode_prompt(prompt="", device=device, num_images_per_prompt=1,
                                              do_classifier_free_guidance=False)

    @torch.no_grad()
    def gen(pe, pp, seed):
        g = torch.Generator(device=device).manual_seed(seed)
        return pipe(prompt_embeds=pe.half(), pooled_prompt_embeds=pp.half(),
                    negative_prompt_embeds=neg_pe.half(), negative_pooled_prompt_embeds=neg_pp.half(),
                    num_inference_steps=GEN_STEPS, guidance_scale=GUIDANCE,
                    height=1024, width=1024, generator=g).images[0]

    scores = {"true": [], "A": [], "B": []}
    grid = []
    with torch.no_grad():
        pa_s, pa_p = net_a(proxy_ev.to(device))
        pb_s, pb_p = net_b(proxy_ev.to(device))
    for i, cap in enumerate(eval_caps):
        sd = SEED + i
        im_t = gen(seq_ev[i:i+1].to(device), pool_ev[i:i+1].to(device), sd)
        im_a = gen(pa_s[i:i+1], pa_p[i:i+1], sd)
        im_b = gen(pb_s[i:i+1], pb_p[i:i+1], sd)
        st, sa, sb = clipscore(im_t, cap), clipscore(im_a, cap), clipscore(im_b, cap)
        scores["true"].append(st); scores["A"].append(sa); scores["B"].append(sb)
        grid.append([im_t, im_a, im_b])
        print(f"  [{i}] CLIP true={st:.3f} A(L2)={sa:.3f} B(contrast)={sb:.3f} | {cap[:44]}", flush=True)

    thumb = 320
    canvas = Image.new("RGB", (3 * thumb, N_EVAL * thumb), (20, 20, 20))
    for r, row in enumerate(grid):
        for c, im in enumerate(row):
            canvas.paste(im.resize((thumb, thumb)), (c * thumb, r * thumb))
    canvas.save(os.path.join(OUT, "montage_true_vs_A_vs_B.png"))

    import statistics as st_
    print(f"\n=== SUMMARY (cols: TRUE-cond | A=L2 | B=L2+contrastive) ===")
    print(f"  mean CLIPScore: true={st_.mean(scores['true']):.3f}  A={st_.mean(scores['A']):.3f}  B={st_.mean(scores['B']):.3f}")
    print(f"  retrieval@1:    A={ra:.3f}  B={rb:.3f}   (higher=better)")
    print(f"  pred-collapse:  A={ca:.3f}  B={cb:.3f}   (lower=better; high=all preds alike)")
    print(f"outputs in {OUT}/ (montage_true_vs_A_vs_B.png)")


if __name__ == "__main__":
    main()
