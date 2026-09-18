"""Activation-side max attention logit per checkpoint -- the number qk_clip_tau is set from.

The weight-side probe (attn_qk_growth_probe.py) gives sigma(Wq)*sigma(Wk)/sqrt(d), an UPPER
BOUND that is fine for detecting a trend but useless for picking a threshold: q_i.k_j almost
never realizes the worst-case alignment, so the bound can be orders of magnitude above the
real logits. This runs actual batches through the actual model and reads the real maxima out
of the qk-clip forward probe.

⚠️ Depth dependence. The trunk's iteration count is Poisson-sampled per step
(recurrent.py:273), and the probe accumulates a max over every iteration in a forward pass, so
a deeper forward samples more attention maps and reports a higher max for identical weights.
--trunk_iters pins it so the series is comparable across checkpoints.
"""
import argparse, glob, json, os, re, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_voice_ar_diagnostics import build_args, add_mrope_args
from megatransformer.scripts.eval.world.visualize import load_world_model, load_dataset
from megatransformer.scripts.data.world.data_collator import MultimodalDataCollator
from megatransformer.model.qk_clip import QKClipController
from megatransformer.utils.codebook import load_codebook
from megatransformer.utils import constants

ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True)
ap.add_argument("--pattern", default="checkpoint-*")
ap.add_argument("--cache_dir", required=True)
ap.add_argument("--codebook", required=True)
ap.add_argument("--config", default="small_sum")
ap.add_argument("--text_encoder_model", default="HuggingFaceTB/SmolLM2-135M")
ap.add_argument("--batches", type=int, default=4)
ap.add_argument("--batch_size", type=int, default=8)
ap.add_argument("--trunk_iters", type=int, default=32,
                help="PIN the recurrent depth so the series is comparable across checkpoints.")
ap.add_argument("--top", type=int, default=5)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--voice_predict_f0", action="store_true", default=False)
ap.add_argument("--out", default=None)
add_mrope_args(ap)
a = ap.parse_args()


def step_of(p):
    m = re.search(r"checkpoint-(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else -1


ckpts = sorted([c for c in glob.glob(os.path.join(a.run_dir, a.pattern))
                if os.path.isdir(c) and step_of(c) >= 0], key=step_of)
if not ckpts:
    sys.exit(f"no checkpoints under {a.run_dir}/{a.pattern}")

cb = load_codebook(a.codebook)
K, D = int(cb.shape[0]), int(cb.shape[1])
a.checkpoint_path = ckpts[0]
a.n = a.batch_size * a.batches
margs = build_args(a, D)
ds = load_dataset(margs, "val")

rows = []
for ck in ckpts:
    margs.checkpoint_path = ck
    model = load_world_model(margs, a.device)
    model.set_voice_codebook(cb)
    model.to(a.device).eval()
    model.recurrent_block.mean_thinking_steps = a.trunk_iters
    sp_base = getattr(model.config, "special_token_base", constants.SPECIAL_TOKEN_BASE)
    coll = MultimodalDataCollator(
        max_seq_len=2048, max_waveforms=160000, max_mel_spec_frames=625,
        max_sive_feature_frames=250, voice_eov_id=K, special_token_base=sp_base,
        bistream_text_chunk=0, bistream_voice_chunk=0, bistream_prob=0.0, voice_fill_id=None)
    coll.force_direction = "synthesis"

    # tau=inf: we only want the controller's probe, never its clip.
    ctrl = QKClipController(model, tau=float("inf"), probe_every=1)
    ctrl.maybe_arm(0)

    picked, i = [], 0
    while len(picked) < a.batch_size * a.batches and i < len(ds):
        s = ds[i]; i += 1
        if any(k.startswith("voice_") for k in s):
            picked.append(s)
    with torch.no_grad():
        for b in range(a.batches):
            batch = coll(picked[b * a.batch_size:(b + 1) * a.batch_size])
            batch = {k: (v.to(a.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            # Mirror the trainer's batch -> forward mapping (world/training.py:657,771-779).
            # The collator emits unit ids; the dataset/trainer turn them into features by
            # indexing the codebook, exactly as the cache was built (features = centroids[ids]).
            uid = batch["voice_unit_ids"].clamp(0, K - 1)
            feats = cb.to(a.device)[uid].transpose(1, 2)          # [B, T, C] -> [B, C, T]
            vlen = batch["voice_feature_lengths"]
            model(text_input_ids=batch["text_token_ids"],
                  voice_inputs=feats.unsqueeze(1),
                  voice_lengths=vlen.unsqueeze(1),
                  is_synthesis=batch.get("is_synthesis"),
                  decode_outputs=False)

    per = [(float(m._qk_max_logit.max()), n) for n, m in ctrl.modules
           if getattr(m, "_qk_max_logit", None) is not None]
    allv = torch.cat([m._qk_max_logit.flatten().float().cpu() for _, m in ctrl.modules
                      if getattr(m, "_qk_max_logit", None) is not None])
    per.sort(reverse=True)
    row = {"ckpt": os.path.basename(ck), "step": step_of(ck),
           "max": float(allv.max()), "p999": float(allv.quantile(0.999)),
           "p99": float(allv.quantile(0.99)), "p95": float(allv.quantile(0.95)),
           "median": float(allv.median()), "n_heads": int(allv.numel()),
           "argmax": per[0][1] if per else ""}
    rows.append(row)
    print(f"  {row['ckpt']:20s} max={row['max']:8.2f}  p99={row['p99']:7.2f}  "
          f"p95={row['p95']:7.2f}  med={row['median']:6.2f}   {row['argmax']}", flush=True)
    del model
    torch.cuda.empty_cache()

print()
if len(rows) >= 2:
    f, l = rows[0], rows[-1]
    print(f"TREND steps {f['step']} -> {l['step']} (trunk_iters pinned at {a.trunk_iters}):")
    print(f"  max  {f['max']:.2f} -> {l['max']:.2f}   ({(l['max']/f['max']-1)*100:+.1f}%)")
    print(f"  p99  {f['p99']:.2f} -> {l['p99']:.2f}   ({(l['p99']/f['p99']-1)*100:+.1f}%)")
    print()
    print(f"  tau SUGGESTION: a threshold should sit ABOVE the bulk and bite only the tail.")
    print(f"  p99 at the last checkpoint = {l['p99']:.1f}; max = {l['max']:.1f}.")
    print(f"  Starting tau in [{l['p99']:.0f}, {l['max']:.0f}] clips only the worst heads.")

if a.out:
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {a.out}")
