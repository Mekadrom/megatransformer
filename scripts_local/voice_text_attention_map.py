"""Voice->text attention maps from the recurrent trunk.

CAVEAT (agreed with the user): attention weights are NOT proof of what a model uses --
information also flows through values and the residual stream. This test is ASYMMETRIC: a
clean monotonic ridge would be strong evidence that alignment exists; a diffuse map is weak
evidence of its absence. The decisive test is the causal one
(scripts_local/text_substitution_probe.py), which found first-divergence at 0.8% of the
utterance for a substitution 57% through the text = no positional correspondence. This is
here to SEE that.

Mechanics: the trunk takes SDPA (which exposes no probabilities) unless a score-rewriting
option is set, so we set attn_logit_cap to a huge value -- cap*tanh(s/cap) ~= s, numerically
near-identity -- to force the manual path, then capture F.softmax outputs.
"""
import argparse, os, sys
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_voice_ar_diagnostics import build_args, make_collator
from megatransformer.scripts.eval.world.visualize import load_world_model, load_dataset
from megatransformer.utils.codebook import load_codebook
from megatransformer.utils import constants

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint_path", required=True)
ap.add_argument("--step", type=int, required=True)
ap.add_argument("--cache_dir", required=True)
ap.add_argument("--codebook", required=True)
ap.add_argument("--config", default="small_sum")
ap.add_argument("--text_encoder_model", default="HuggingFaceTB/SmolLM2-135M")
ap.add_argument("--voice_max_frames", type=int, default=250)
ap.add_argument("--voice_predict_f0", dest="voice_predict_f0", action="store_true", default=False)
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--device", default="cuda:2")
ap.add_argument("--out_dir", default="eval_output/world_tts_attention_maps")
ap.add_argument("--per_head", action="store_true",
                help="Report per (block/iteration, head) sharpness instead of the head-average. "
                     "The average can hide ONE sharp aligner head under many diffuse ones — a "
                     "different diagnosis (signal diluted downstream) than 'no head aligns'.")
a = ap.parse_args()

cb = load_codebook(a.codebook); K, D = int(cb.shape[0]), int(cb.shape[1])
args = build_args(a, D)
model = load_world_model(args, a.device); model.set_voice_codebook(cb)
model.to(a.device).eval()
sp_base = getattr(model.config, "special_token_base", constants.SPECIAL_TOKEN_BASE)
sp = constants.special_token_ids(sp_base)
ds = load_dataset(args, "val")
coll = make_collator(K, a.voice_max_frames, special_token_base=sp_base); coll.force_direction = "synthesis"
os.makedirs(a.out_dir, exist_ok=True)

# force the manual attention path in the recurrent trunk
n_forced = 0
for blk in model.recurrent_block.recurrent_blocks:
    if hasattr(blk, "self_attn") and hasattr(blk.self_attn, "config"):
        blk.self_attn.config.attn_logit_cap = 1e6
        n_forced += 1
print(f"forced manual attention in {n_forced} recurrent blocks", flush=True)

CAP = []
_orig_softmax = F.softmax
def _cap_softmax(x, *args_, **kw):
    out = _orig_softmax(x, *args_, **kw)
    if out.dim() == 4 and out.shape[-1] == out.shape[-2]:
        CAP.append(out.detach().float().cpu())
    return out

done = 0
for i in range(len(ds)):
    s = ds[i]
    if not any(k.startswith("voice_") for k in s):
        continue
    b = coll([s])
    text_ids = b["text_token_ids"][0]
    # The collator emits ONE placeholder per voice EXAMPLE; the interleaver expands it to T
    # frames inside the model. So the voice block in the INTERLEAVED sequence is
    # [p, p+T) where p is that placeholder's index -- it cannot be read off text_token_ids.
    ph = (text_ids == sp.VOICE_PLACEHOLDER).nonzero(as_tuple=True)[0]
    if len(ph) != 1:
        continue
    p_idx = int(ph[0])
    T_v = int(b["voice_feature_lengths"].reshape(-1)[0])
    if p_idx < 5 or T_v < 20:
        continue
    CAP.clear()
    F.softmax = _cap_softmax
    try:
        with torch.no_grad():
            model(text_input_ids=b["text_token_ids"].to(a.device),
                  voice_inputs=b["voice_features"].unsqueeze(1).to(a.device),
                  voice_lengths=b["voice_feature_lengths"].unsqueeze(1).to(a.device),
                  voice_latent_labels=b["voice_features"].to(a.device),
                  is_synthesis=b["is_synthesis"].to(a.device), decode_outputs=False)
    finally:
        F.softmax = _orig_softmax
    if not CAP:
        print("no attention captured — the manual path did not run"); break
    S = max(x.shape[-1] for x in CAP)
    full = [x for x in CAP if x.shape[-1] == S]
    if a.per_head:
        import math as _m
        stats = []
        v_hi_ = min(p_idx + T_v, S)
        vp_ = torch.arange(p_idx, v_hi_); tp_ = torch.arange(0, min(p_idx, S))
        if len(vp_) < 10 or len(tp_) < 3:
            continue
        uni = _m.log(len(tp_))
        for ci, cap_t in enumerate(full):
            for h in range(cap_t.shape[1]):
                sub_h = cap_t[0, h][vp_][:, tp_]
                rn = sub_h / sub_h.sum(-1, keepdim=True).clamp_min(1e-9)
                e = float(-(rn * (rn + 1e-9).log()).sum(-1).mean())
                am_ = sub_h.argmax(-1).float(); ix = torch.arange(len(am_)).float()
                c = float(((am_ - am_.mean()) * (ix - ix.mean())).sum() /
                          (am_.std().clamp_min(1e-6) * ix.std().clamp_min(1e-6) * len(am_)))
                stats.append((e, c, ci, h))
        stats.sort()
        print(f"\n  utt {i}: uniform entropy {uni:.3f} nats over {len(tp_)} text tokens")
        print(f"  {'rank':>4} {'entropy':>8} {'ent/uni':>8} {'corr':>7}  (capture, head)")
        for r, (e, c, ci, h) in enumerate(stats[:6]):
            print(f"  {r:>4} {e:>8.3f} {e/uni:>8.3f} {c:>+7.3f}  ({ci}, {h})")
        e_all = sum(x[0] for x in stats) / len(stats)
        print(f"  mean over {len(stats)} (capture,head) pairs: entropy {e_all:.3f} "
              f"({e_all/uni:.3f} of uniform)")
        done += 1
        if done >= a.n:
            break
        continue
    A = torch.stack([x[0].mean(0) for x in full]).mean(0)      # avg heads + all blocks/iters
    # interleaved layout: text [0, p_idx) | voice [p_idx, p_idx+T_v) | trailing text
    v_hi = min(p_idx + T_v, S)
    vp = torch.arange(p_idx, v_hi)
    tp = torch.arange(0, min(p_idx, S))
    if len(vp) < 10 or len(tp) < 3:
        print(f"  [skip] utt {i}: interleaved S={S} p={p_idx} T={T_v}"); continue
    sub = A[vp][:, tp]                                          # (voice, text)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(sub.numpy(), aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("text token index"); ax.set_ylabel("voice frame index")
    ax.set_title(f"trunk attention: voice->text  step {a.step}  utt {i}\n"
                 f"{str(s.get('voice_voice_text',''))[:64]}")
    # a monotonic aligner would follow this line
    ax.plot([0, len(tp) - 1], [0, len(vp) - 1], color="cyan", lw=1, ls="--", alpha=.7,
            label="monotonic alignment (reference)")
    ax.legend(loc="upper left", fontsize=7)
    fig.colorbar(im, ax=ax); fig.tight_layout()
    p = os.path.join(a.out_dir, f"attn_{done:02d}_utt{i}.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    # quantify: how concentrated is each voice frame's text attention, and does it advance?
    am = sub.argmax(-1).float()
    rows_n = sub / sub.sum(-1, keepdim=True).clamp_min(1e-9)
    ent = -(rows_n * (rows_n + 1e-9).log()).sum(-1).mean()
    import math as _m
    idx = torch.arange(len(am)).float()
    corr = float(((am - am.mean()) * (idx - idx.mean())).sum() /
                 (am.std().clamp_min(1e-6) * idx.std().clamp_min(1e-6) * len(am)))
    print(f"  [{done}] utt {i}: argmax-vs-frame corr {corr:+.3f} | row entropy {ent:.3f} nats "
          f"(uniform={_m.log(len(tp)):.3f}) -> {p}", flush=True)
    done += 1
    if done >= a.n:
        break
print("\nA monotonic aligner: corr -> +1 and row entropy << uniform.")
print("Bag-of-text conditioning: corr ~ 0 and row entropy ~ uniform.")
