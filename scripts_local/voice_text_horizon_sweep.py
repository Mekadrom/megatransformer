"""Track the TEXT HORIZON across checkpoints of a run.

The ear at ~30k: "the onset of the output is very good, but it falls off quickly ... aligned
completely with the GT before breaking down on 'heaven'" (the 5th word). The bucketed decay
confirms it — text_delta roughly HALVES every 8 frames and is gone past ~11 words.

The distinction that matters for judging progress: a rising early_text_delta means the ONSET
is getting stronger, which is NOT the same as the model getting better at speech. What would
count as real progress is the horizon moving RIGHT — text conditioning holding further into
the utterance. This sweeps checkpoints and reports that trajectory.

Loads the model once per checkpoint and calls run_tf_and_ablation directly (no subprocess),
teacher-forced only — no generation, so it is comparatively cheap.

  uv run python scripts_local/voice_text_horizon_sweep.py \
      --run_dir runs/world/world_tts_cosyvoice2_smollm2_mrope_scale_text_0 \
      --steps 10000,20000,30000 --n 1024 --device cuda:3
"""
import argparse, glob, json, os, re, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_voice_ar_diagnostics import (build_args, make_collator, run_tf_and_ablation,
                                        text_horizon, add_mrope_args)
from megatransformer.scripts.eval.world.visualize import load_world_model, load_dataset
from megatransformer.utils.codebook import load_codebook
from megatransformer.utils import constants

ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True)
ap.add_argument("--steps", default=None, help="comma-separated; default = every --every-th checkpoint")
ap.add_argument("--every", type=int, default=10000)
ap.add_argument("--cache_dir", default="cached_datasets/Mekadrom/libritts_r_cosyvoice2_smollm2")
ap.add_argument("--codebook", default="cached_datasets/Mekadrom/libritts_r_cosyvoice2_smollm2/val/cosyvoice2_codebook.pt")
ap.add_argument("--config", default="small_sum")
ap.add_argument("--text_encoder_model", default="HuggingFaceTB/SmolLM2-135M")
ap.add_argument("--voice_max_frames", type=int, default=250)
ap.add_argument("--voice_predict_f0", dest="voice_predict_f0", action="store_true", default=False)
ap.add_argument("--n", type=int, default=1024)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--threshold", type=float, default=0.01)
ap.add_argument("--device", default="cuda:3")
ap.add_argument("--out_dir", default="eval_output/world_tts_horizon")
add_mrope_args(ap)
a = ap.parse_args()

avail = sorted(int(m.group(1)) for d in glob.glob(os.path.join(a.run_dir, "checkpoint-*"))
               if (m := re.search(r"checkpoint-(\d+)$", d)))
if a.steps:
    want = [int(x) for x in a.steps.split(",")]
    steps = [min(avail, key=lambda s: abs(s - w)) for w in want]
else:
    steps = [s for s in avail if s % a.every == 0] or avail[-1:]
steps = sorted(set(steps))
print(f"{os.path.basename(a.run_dir)}: probing steps {steps}", flush=True)

cb = load_codebook(a.codebook); K, D = int(cb.shape[0]), int(cb.shape[1])
os.makedirs(a.out_dir, exist_ok=True)
rows = []
for st in steps:
    a.checkpoint_path = os.path.join(a.run_dir, f"checkpoint-{st}")
    args = build_args(a, D)
    model = load_world_model(args, a.device); model.set_voice_codebook(cb)
    model.to(a.device).eval()
    sp_base = getattr(model.config, "special_token_base", constants.SPECIAL_TOKEN_BASE)
    ds = load_dataset(args, "val")
    coll = make_collator(K, a.voice_max_frames, special_token_base=sp_base)
    r = run_tf_and_ablation(model, ds, coll, a.device, a.n, K, bs=a.bs)
    h = text_horizon(r.get("buckets", []), a.threshold)
    rows.append({"step": st, "horizon_frames": h, "horizon_words": h / 6 if h != float("inf") else None,
                 "early_text_delta": r["early_text_delta"], "early_ci": r["early_text_delta_ci"],
                 "acc_real": r["acc_real"], "text_delta": r["text_delta"],
                 "text_attributed_fraction": r["text_delta"] / max(r["acc_real"], 1e-9),
                 "buckets": r.get("buckets", [])})
    hw = "never" if h == float("inf") else f"{h:.1f}f/~{h/6:.1f}w"
    print(f"  step {st:>6}: horizon {hw:>12}  early_delta {r['early_text_delta']:+.4f}  "
          f"acc {r['acc_real']:.4f}  text-frac {rows[-1]['text_attributed_fraction']:.3f}", flush=True)
    del model
    torch.cuda.empty_cache()

print(f"\n| step | horizon (frames) | ~words | early_text_delta | acc_real | text-frac |")
print(f"|---|---|---|---|---|---|")
for r in rows:
    hw = "never" if r["horizon_frames"] == float("inf") else f"{r['horizon_frames']:.1f}"
    ww = "-" if r["horizon_words"] is None else f"{r['horizon_words']:.1f}"
    print(f"| {r['step']} | {hw} | {ww} | {r['early_text_delta']:+.4f} | "
          f"{r['acc_real']:.4f} | {r['text_attributed_fraction']:.3f} |")
print("\nHORIZON RISING = conditioning holds further in (real progress).")
print("HORIZON FLAT while early_text_delta rises = the ONSET is getting taller only.")
print("Teacher reference: early_text_delta +0.0538, text-attributed fraction 0.463.")

name = os.path.basename(a.run_dir.rstrip("/"))
json.dump(rows, open(os.path.join(a.out_dir, f"{name}_horizon.json"), "w"), indent=2)

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for r in rows:
        xs = [(b["lo"] + min(b["hi"], 256)) / 2 for b in r["buckets"]]
        ax[0].plot(xs, [b["delta"] for b in r["buckets"]], marker="o", label=f"step {r['step']}")
    ax[0].axhline(a.threshold, ls="--", c="grey", lw=1)
    ax[0].set_xlabel("voice frame index"); ax[0].set_ylabel("text_delta")
    ax[0].set_title("text conditioning vs position"); ax[0].legend(fontsize=7)
    hs = [r["horizon_frames"] for r in rows]
    ax[1].plot([r["step"] for r in rows], [None if h == float("inf") else h for h in hs], marker="o")
    ax[1].set_xlabel("training step"); ax[1].set_ylabel("horizon (frames)")
    ax[1].set_title(f"text horizon (delta < {a.threshold})")
    fig.tight_layout(); p = os.path.join(a.out_dir, f"{name}_horizon.png")
    fig.savefig(p, dpi=120); print(f"plot -> {p}")
except Exception as e:
    print(f"(plot skipped: {type(e).__name__})")
