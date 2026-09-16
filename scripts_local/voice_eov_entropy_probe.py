"""Is premature EOV a CONFIDENT decision or an uncertain one?

Decides whether confidence-adaptive sampling (min-p, eta, entropy-scaled temperature) can
possibly help the collapse failure mode. If the model fires EOV with high p_max / low
entropy, min-p will behave like greedy exactly there and change nothing. If EOV fires from a
flat distribution, adaptive truncation has something to work with.

Compares, at the step EOV is emitted:
  COLLAPSED utterances (gen < 50% of reference frames)  vs  OK utterances.
Also reports the distribution shape at ordinary (non-EOV) steps as a baseline.
"""
import argparse, glob, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
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
ap.add_argument("--n", type=int, default=48)
ap.add_argument("--voice_temperature", type=float, default=0.0)
ap.add_argument("--ras_win", type=int, default=10)
ap.add_argument("--ras_tau", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out_dir", default="eval_output/world_voice_eov_entropy")
add_mrope_args(ap)
a = ap.parse_args()

cb = load_codebook(a.codebook); K, D = int(cb.shape[0]), int(cb.shape[1])
args = build_args(a, D)
model = load_world_model(args, a.device); model.set_voice_codebook(cb)
model.to(a.device).eval()
sp_base = getattr(model.config, "special_token_base", constants.SPECIAL_TOKEN_BASE)
sp = constants.special_token_ids(sp_base)
ds = load_dataset(args, "val")
coll = make_collator(K, a.voice_max_frames, special_token_base=sp_base)
coll.force_direction = "synthesis"
torch.manual_seed(a.seed)
os.makedirs(a.out_dir, exist_ok=True)

rows, done = [], 0
for i in range(len(ds)):
    s = ds[i]
    if not any(k.startswith("voice_") for k in s):
        continue
    b = coll([s])
    text = b["text_token_ids"][0]
    bov = (text == sp.BOV).nonzero(as_tuple=True)[0]
    if len(bov) == 0:
        continue
    prompt = text[:bov[0].item() + 1].unsqueeze(0).to(a.device)
    stats = []
    with torch.no_grad():
        model.generate(text_input_ids=prompt, max_new_tokens=512,
                       voice_token_budget=a.voice_max_frames,
                       voice_temperature=a.voice_temperature,
                       voice_ras_win=a.ras_win, voice_ras_tau=a.ras_tau,
                       voice_step_stats=stats, decode_outputs=False)
    L_ref = int(s["voice_feature_length"])
    gen_len = len([x for x in stats if not x["is_eov"]])
    eov = [x for x in stats if x["is_eov"]]
    rows.append({"idx": i, "ref_frames": L_ref, "gen_frames": gen_len,
                 "collapsed": gen_len < 0.5 * L_ref,
                 "eov_step": eov[0] if eov else None,
                 "n_steps": len(stats),
                 "mid_entropy": statistics.median([x["entropy"] for x in stats]) if stats else None,
                 "mid_pmax": statistics.median([x["p_max"] for x in stats]) if stats else None,
                 "max_p_eov_before_end": max([x["p_eov"] for x in stats[:-1]], default=0.0)})
    done += 1
    print(f"  [{done}] gen {gen_len:>3}/{L_ref:<3} {'COLLAPSED' if rows[-1]['collapsed'] else ''}", flush=True)
    if done >= a.n:
        break

json.dump(rows, open(os.path.join(a.out_dir, f"eov_entropy_step{a.step}.json"), "w"), indent=1)


def summ(name, vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        print(f"  {name:34s} (none)"); return
    vals = sorted(vals)
    print(f"  {name:34s} n={len(vals):>3}  median {statistics.median(vals):.4f}  "
          f"p10 {vals[max(0,int(.1*len(vals))-0)]:.4f}  p90 {vals[min(len(vals)-1,int(.9*len(vals)))]:.4f}")


col = [r for r in rows if r["collapsed"]]
ok = [r for r in rows if not r["collapsed"]]
print(f"\n=== EOV DECISION SHAPE — step {a.step}, T={a.voice_temperature}, n={len(rows)} ===")
print(f"collapsed: {len(col)}   ok: {len(ok)}\n")
print("AT THE STEP EOV FIRED:")
summ("collapsed  entropy (nats)", [r["eov_step"]["entropy"] for r in col if r["eov_step"]])
summ("ok         entropy (nats)", [r["eov_step"]["entropy"] for r in ok if r["eov_step"]])
summ("collapsed  p_max", [r["eov_step"]["p_max"] for r in col if r["eov_step"]])
summ("ok         p_max", [r["eov_step"]["p_max"] for r in ok if r["eov_step"]])
summ("collapsed  p(EOV)", [r["eov_step"]["p_eov"] for r in col if r["eov_step"]])
summ("ok         p(EOV)", [r["eov_step"]["p_eov"] for r in ok if r["eov_step"]])
print("\nBASELINE — median over ALL steps of each utterance:")
summ("collapsed  entropy", [r["mid_entropy"] for r in col])
summ("ok         entropy", [r["mid_entropy"] for r in ok])
print(f"\nmax entropy = {torch.log(torch.tensor(float(K+1))).item():.3f} nats (uniform over K+1)")
print("""
READ: if collapsed-EOV p_max is HIGH / entropy LOW, the model is CONFIDENTLY stopping and
min-p will be greedy exactly there -- adaptive truncation cannot help. If EOV fires from a
FLAT distribution, the collapse is a sampling accident and adaptive methods have purchase.""")
