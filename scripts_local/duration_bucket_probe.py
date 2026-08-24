"""How accurate is the duration head, and would a bucket-derived M-RoPE rate beat a constant?

M-RoPE uses ONE global voice_rate for every utterance, but the corpus rate varies 2x
(n=6102: mean 6.015, std 1.239, p5 4.19, p95 8.22). So a constant is wrong by ~20.6% RMS
relative on every example. The duration token gives L before any voice position is allocated,
so the rate could instead be duration_bucket_frames(bucket) / n_text_tokens -- per utterance.

Whether that is better depends entirely on the duration head's error distribution, which two
summary statistics (exact accuracy, +-1 accuracy) do not pin down: a tight cluster at +-2 and
a long tail give very different answers. This measures the distribution and computes both RMS
errors directly, so the comparison is arithmetic rather than argument.

  uv run python scripts_local/duration_bucket_probe.py --checkpoint_path <ckpt> --step N \
      --cache_dir <cache> --codebook <cb> --device cuda:3
"""
import argparse, json, math, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_voice_ar_diagnostics import build_args, make_collator, add_mrope_args
from megatransformer.scripts.eval.world.visualize import load_world_model, load_dataset
from megatransformer.utils.codebook import load_codebook
from megatransformer.utils import constants as C

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint_path", required=True)
ap.add_argument("--step", type=int, required=True)
ap.add_argument("--cache_dir", required=True)
ap.add_argument("--codebook", required=True)
ap.add_argument("--config", default="small_sum")
ap.add_argument("--text_encoder_model", default="HuggingFaceTB/SmolLM2-135M")
ap.add_argument("--voice_max_frames", type=int, default=250)
ap.add_argument("--voice_predict_f0", action="store_true", default=False)
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--device", default="cuda:3")
ap.add_argument("--out_dir", default="eval_output/duration_bucket")
add_mrope_args(ap)
a = ap.parse_args()

cb = load_codebook(a.codebook); K, D = int(cb.shape[0]), int(cb.shape[1])
args = build_args(a, D)
model = load_world_model(args, a.device); model.set_voice_codebook(cb)
model.to(a.device).eval()
sp = C.special_token_ids(getattr(model.config, "special_token_base", C.SPECIAL_TOKEN_BASE))
dur_lo, dur_hi = sp.base + 9, sp.base + 9 + C.N_DURATION_BUCKETS
ds = load_dataset(args, "val")
coll = make_collator(K, a.voice_max_frames, special_token_base=sp.base)
coll.force_direction = "synthesis"

errs, rate_pred, rate_true, n_done = [], [], [], 0
with torch.no_grad():
    for i in range(len(ds)):
        if n_done >= a.n:
            break
        s = ds[i]
        if not any(k.startswith("voice_") for k in s):
            continue
        b = coll([s])
        text = b["text_token_ids"][0]
        bov = (text == sp.BOV).nonzero(as_tuple=True)[0]
        if len(bov) == 0:
            continue
        prompt = text[:bov[0].item() + 1].unsqueeze(0).to(a.device)
        out = model(text_input_ids=prompt, decode_outputs=False)
        lg = out.get("logits")
        if lg is None or lg.shape[-1] < dur_hi:
            raise SystemExit("no duration head on this checkpoint (logits too narrow)")
        pred_b = int(torch.argmax(lg[0, -1, dur_lo:dur_hi]).item())
        L = int(s["voice_feature_length"]); N = int(bov[0].item())
        if N <= 0:
            continue
        true_b = C.duration_bucket(L)
        errs.append(pred_b - true_b)
        rate_pred.append(C.duration_bucket_frames(pred_b) / N)
        rate_true.append(L / N)
        n_done += 1

import numpy as np
e = np.array(errs); rp = np.array(rate_pred); rt = np.array(rate_true)
CONST = 6.0
rms_const = float(np.sqrt(np.mean(((CONST - rt) / rt) ** 2)))
rms_bucket = float(np.sqrt(np.mean(((rp - rt) / rt) ** 2)))
print(f"\n=== duration head @ step {a.step} (n={len(e)}) ===")
print(f"exact bucket      {float((e == 0).mean()):.3f}")
print(f"within +-1        {float((np.abs(e) <= 1).mean()):.3f}")
print(f"within +-2        {float((np.abs(e) <= 2).mean()):.3f}")
print(f"|error| mean {float(np.abs(e).mean()):.2f} buckets | median {float(np.median(np.abs(e))):.0f} | p90 {float(np.percentile(np.abs(e),90)):.0f} | max {int(np.abs(e).max())}")
print(f"signed error mean {float(e.mean()):+.2f} (bias)")
print(f"\n=== M-RoPE rate error, relative RMS ===")
print(f"global constant {CONST}    {rms_const:.4f}  ({rms_const*100:.1f}%)")
print(f"bucket-derived per-utt   {rms_bucket:.4f}  ({rms_bucket*100:.1f}%)")
print(f"-> bucket-derived is {'BETTER' if rms_bucket < rms_const else 'WORSE'} by "
      f"{abs(rms_const-rms_bucket)/rms_const*100:.0f}% relative")
os.makedirs(a.out_dir, exist_ok=True)
json.dump({"step": a.step, "n": len(e), "exact": float((e == 0).mean()),
           "pm1": float((np.abs(e) <= 1).mean()), "pm2": float((np.abs(e) <= 2).mean()),
           "abs_err_mean_buckets": float(np.abs(e).mean()),
           "rms_rel_const": rms_const, "rms_rel_bucket": rms_bucket},
          open(os.path.join(a.out_dir, f"step{a.step}.json"), "w"), indent=2)
print(f"\nwrote {a.out_dir}/step{a.step}.json")
