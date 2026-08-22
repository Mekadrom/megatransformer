"""How peaked is the voice unit head's PREDICTIVE distribution?

WHY THIS EXISTS: a lot of today's interpretation rested on "the model's distribution is so
peaked that sampling is effectively argmax" -- inferred twice from indirect evidence (runs
that looked identical, sampler interventions that did little) and never measured. Both
inferences turned out to rest on bugs. This measures it directly.

⚠️ THE NUMBER TO NOT CONFUSE IT WITH: the free-running reports quote unit_entropy_bits ~4.8
against a GT marginal of 10.5. That is the entropy of the MARGINAL distribution of emitted
units over a whole corpus -- it says the model uses few distinct units overall. It says
nothing about how confident any single step is. A model can be uncertain at every step
(high predictive entropy) and still emit a low-diversity corpus, or the reverse. What
governs whether sampling behaves like argmax is the PER-STEP predictive distribution, which
is what this reports.

Reports, over content positions:
  top1_prob        - mean/median probability of the argmax unit
  entropy_bits     - mean per-step predictive entropy (max log2(K+1) ~ 12.7)
  top5 / top10     - cumulative mass in the top k
  frac_gt_{0.5,0.9} - fraction of steps where the argmax already holds that much mass;
                     high here means multinomial sampling is argmax in practice
Each at the raw distribution and at the decode temperature, since tempering is what actually
runs at inference.

For a NAR checkpoint --nar_mask_ratio is REQUIRED (mask ratio 0 lets it copy its input); 1.0
is the inference start condition. AR checkpoints use ordinary teacher forcing.

  uv run python scripts_local/unit_confidence_probe.py --checkpoint_path <ckpt> --step N \
      --cache_dir <cache> --codebook <cb> --nar_mask_ratio 1.0 --device cuda:3
"""
import argparse, json, os, sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_voice_ar_diagnostics import build_args, make_collator, add_mrope_args
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
ap.add_argument("--voice_predict_f0", action="store_true", default=False)
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--temperature", type=float, default=0.6, help="decode temperature to also report at")
ap.add_argument("--nar_mask_ratio", type=float, default=None)
ap.add_argument("--device", default="cuda:3")
ap.add_argument("--out_dir", default="eval_output/unit_confidence")
add_mrope_args(ap)
a = ap.parse_args()

cb = load_codebook(a.codebook); K, D = int(cb.shape[0]), int(cb.shape[1])
args = build_args(a, D)
model = load_world_model(args, a.device); model.set_voice_codebook(cb)
model.to(a.device).eval()
is_nar = getattr(model, "voice_mask_feature", None) is not None
if is_nar and a.nar_mask_ratio is None:
    raise SystemExit("NAR checkpoint: --nar_mask_ratio is required (1.0 = inference start condition)")
sp_base = getattr(model.config, "special_token_base", constants.SPECIAL_TOKEN_BASE)
ds = load_dataset(args, "val")
coll = make_collator(K, a.voice_max_frames, special_token_base=sp_base)
coll.force_direction = "synthesis"
print(f"{'NAR' if is_nar else 'AR'} checkpoint | mask_ratio={a.nar_mask_ratio} | T={a.temperature}",
      flush=True)

stats = {}
for label in ("raw", f"T={a.temperature}"):
    stats[label] = {"top1": [], "ent": [], "top5": [], "top10": [], "n": 0,
                    "gt50": 0, "gt90": 0}

with torch.no_grad():
    for s in range(0, min(a.n, len(ds)), a.bs):
        samples = [ds[i] for i in range(s, min(s + a.bs, min(a.n, len(ds))))]
        samples = [x for x in samples if any(k.startswith("voice_") for k in x)]
        if not samples:
            continue
        b = coll(samples)
        text = b["text_token_ids"].to(a.device)
        vin = b["voice_features"].unsqueeze(1).to(a.device)
        vlen = b["voice_feature_lengths"].unsqueeze(1).to(a.device)
        tgt = b["voice_unit_ids"].to(a.device)
        syn = b["is_synthesis"].to(a.device)
        masked = None
        if a.nar_mask_ratio is not None:
            mf = model.voice_mask_feature
            _b, _n, _c, _t = vin.shape
            masked = (torch.rand(_b, _n, 1, _t, device=a.device) < float(a.nar_mask_ratio))
            vin = torch.where(masked, mf.view(1, 1, _c, 1).to(vin.dtype), vin)
            masked = masked.squeeze(2).reshape(_b * _n, _t)
        out = model(text_input_ids=text, voice_inputs=vin, voice_lengths=vlen,
                    is_synthesis=syn, decode_outputs=False)
        logits = out.get("voice_unit_logits")
        if logits is None:
            continue
        B, T, V = logits.shape
        t = tgt[:, :T]
        if t.shape[1] < T:
            t = F.pad(t, (0, T - t.shape[1]), value=-100)
        keep = (t >= 0) & (t < K)                 # content positions only
        if masked is not None and masked.shape[0] == keep.shape[0]:
            keep = keep & masked[:, :T].to(keep.device)
        if not bool(keep.any()):
            continue
        lg = logits[keep].float()                 # (M, V)
        for label, temp in (("raw", 1.0), (f"T={a.temperature}", max(1e-3, a.temperature))):
            p = torch.softmax(lg / temp, dim=-1)
            top = p.topk(10, dim=-1).values
            st = stats[label]
            st["top1"].append(top[:, 0].cpu())
            st["top5"].append(top[:, :5].sum(-1).cpu())
            st["top10"].append(top.sum(-1).cpu())
            st["ent"].append((-(p.clamp_min(1e-12).log2() * p).sum(-1)).cpu())
            st["gt50"] += int((top[:, 0] > 0.5).sum().item())
            st["gt90"] += int((top[:, 0] > 0.9).sum().item())
            st["n"] += int(top.shape[0])

os.makedirs(a.out_dir, exist_ok=True)
report = {"checkpoint": a.checkpoint_path, "step": a.step, "is_nar": is_nar,
          "nar_mask_ratio": a.nar_mask_ratio, "max_entropy_bits": float(torch.log2(torch.tensor(float(K + 1))))}
print(f"\n=== unit-head predictive distribution (step {a.step}, "
      f"{'NAR r=%s' % a.nar_mask_ratio if is_nar else 'AR teacher-forced'}) ===")
print(f"{'':<12} {'top1':>8} {'median':>8} {'entropy':>9} {'top5':>7} {'top10':>7} "
      f"{'>0.5':>7} {'>0.9':>7}")
for label, st in stats.items():
    if not st["n"]:
        continue
    t1 = torch.cat(st["top1"]); en = torch.cat(st["ent"])
    row = {"top1_mean": float(t1.mean()), "top1_median": float(t1.median()),
           "entropy_bits": float(en.mean()),
           "top5_mass": float(torch.cat(st["top5"]).mean()),
           "top10_mass": float(torch.cat(st["top10"]).mean()),
           "frac_top1_gt_0.5": st["gt50"] / st["n"], "frac_top1_gt_0.9": st["gt90"] / st["n"],
           "n_positions": st["n"]}
    report[label] = row
    print(f"{label:<12} {row['top1_mean']:>8.4f} {row['top1_median']:>8.4f} "
          f"{row['entropy_bits']:>9.3f} {row['top5_mass']:>7.3f} {row['top10_mass']:>7.3f} "
          f"{row['frac_top1_gt_0.5']:>7.3f} {row['frac_top1_gt_0.9']:>7.3f}")
print(f"\nmax entropy = {report['max_entropy_bits']:.2f} bits (log2 of {K+1} classes)")
print("READ: frac>0.9 near 1.0 => multinomial IS argmax in practice and every sampler knob is "
      "cosmetic. frac>0.9 near 0 with low corpus diversity => the model is uncertain per step "
      "but keeps landing in the same small set, which is a different failure.")
name = os.path.basename(a.checkpoint_path.rstrip('/'))
json.dump(report, open(os.path.join(a.out_dir, f"{name}_r{a.nar_mask_ratio}.json"), "w"), indent=2)
print(f"wrote {a.out_dir}/{name}_r{a.nar_mask_ratio}.json")
