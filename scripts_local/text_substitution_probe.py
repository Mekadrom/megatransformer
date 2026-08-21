"""Does the trunk TRACK ITS POSITION in the transcript? A causal, behavioural test.

Observation motivating it (user, by ear, distill@44k): word ONSETS are right and word
INTERIORS devolve -- "What do you moy of it, gerineybry?" for "What do you make of it,
Gryce?" -- and the model overruns short utterances 4-8x. Hypothesis: it knows WHAT the text
says but not WHERE IT IS in it, so it re-latches at each onset and never learns it has
reached the end.

Test: substitute ONE text token mid-prompt, regenerate GREEDILY (deterministic), and see
WHERE the unit sequence diverges from the unmodified run.

  tracking      -> divergence LOCALIZED near that word's time; prefix identical
  no tracking   -> divergence immediate/global; one word perturbs everything

Unlike attention maps this is falsifiable in the direction of my own hypothesis: it predicts
global divergence, so a localized result kills it outright.
"""
import argparse, os, sys, json
import torch

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
ap.add_argument("--voice_predict_f0", dest="voice_predict_f0", action="store_true", default=False)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--frac", type=float, default=0.6, help="substitute at this fraction through the text")
ap.add_argument("--device", default="cuda:2")
ap.add_argument("--out_dir", default="eval_output/world_tts_position_probe")
add_mrope_args(ap)
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


def gen(prompt):
    with torch.no_grad():
        out = model.generate(text_input_ids=prompt, max_new_tokens=512,
                             voice_token_budget=a.voice_max_frames,
                             voice_temperature=0.0,            # greedy => deterministic
                             voice_ras_win=0, decode_outputs=False)
    return [int(x) for x in out.get("voice_unit_id_trace", [[]])[0]]


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
    cut = int(bov[0].item())
    if cut < 8:
        continue
    base = text[:cut + 1].unsqueeze(0).to(a.device)
    # substitute one REAL text token (below the special-token base) at ~frac through
    pos = max(1, min(cut - 1, int(cut * a.frac)))
    orig_id = int(base[0, pos])
    new_id = (orig_id + 1000) % sp_base          # a different, still-valid text token
    alt = base.clone(); alt[0, pos] = new_id

    t0, t1 = gen(base), gen(alt)
    L = min(len(t0), len(t1))
    if L < 10:
        continue
    diff = [j for j in range(L) if t0[j] != t1[j]]
    first = diff[0] if diff else None
    rows.append({
        "idx": i, "text": s.get("voice_voice_text", "")[:70],
        "sub_token_pos": pos, "text_len": cut, "sub_frac": pos / cut,
        "len_base": len(t0), "len_alt": len(t1),
        "first_divergence": first,
        "first_divergence_frac": (first / L) if first is not None else None,
        "frac_positions_differing": len(diff) / L,
    })
    print(f"  [{done}] sub@{pos}/{cut} ({pos/cut:.2f}) | first_div={first}/{L} "
          f"({(first/L if first is not None else float('nan')):.2f}) "
          f"| differ={len(diff)/L:.2f} | {rows[-1]['text'][:40]}", flush=True)
    done += 1
    if done >= a.n:
        break

fd = [r["first_divergence_frac"] for r in rows if r["first_divergence_frac"] is not None]
sf = [r["sub_frac"] for r in rows]
pd = [r["frac_positions_differing"] for r in rows]
print(f"\nn={len(rows)}")
print(f"  substitution at        mean {sum(sf)/len(sf):.3f} through the TEXT")
print(f"  first divergence at    mean {sum(fd)/len(fd):.3f} through the VOICE" if fd else "  no divergence")
print(f"  positions differing    mean {sum(pd)/len(pd):.3f}")
print("\nTRACKING  => first divergence tracks the substitution point (~equal fractions), few positions differ.")
print("NO TRACK  => first divergence ~0.0 and most positions differ, regardless of where the word was.")
json.dump(rows, open(os.path.join(a.out_dir, "substitution.json"), "w"), indent=2)
