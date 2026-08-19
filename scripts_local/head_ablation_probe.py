"""Is the sharp alignment head LOAD-BEARING, or decorative?

Per-head attention analysis found head 4 of recurrent block 0 is a sharp monotonic aligner
(entropy ~9% of uniform, argmax-vs-frame corr +0.85) while the other 11 heads are diffuse.
But attention sharpness is not proof of use -- information also flows through values and the
residual stream. This ablates the head and measures the damage.

Ablation = zero the head's slice of o_proj (columns [h*d_v, (h+1)*d_v)), which removes that
head's contribution exactly, without touching the attention computation itself.

  aligner ablation >> random-head ablation  -> the head IS load-bearing; amplify/sharpen it
  aligner ablation ~= random-head ablation  -> decorative; the position scheme is the lever
"""
import argparse, os, sys, json, random
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_voice_ar_diagnostics import build_args, make_collator, run_tf_and_ablation
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
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--block", type=int, default=0, help="recurrent block holding the aligner")
ap.add_argument("--head", type=int, default=4, help="the sharp aligner head")
ap.add_argument("--n_random", type=int, default=3, help="control heads to ablate for comparison")
ap.add_argument("--device", default="cuda:2")
ap.add_argument("--out_dir", default="eval_output/world_tts_head_ablation")
a = ap.parse_args()

cb = load_codebook(a.codebook); K, D = int(cb.shape[0]), int(cb.shape[1])
args = build_args(a, D)
model = load_world_model(args, a.device); model.set_voice_codebook(cb)
model.to(a.device).eval()
sp_base = getattr(model.config, "special_token_base", constants.SPECIAL_TOKEN_BASE)
ds = load_dataset(args, "val")
coll = make_collator(K, a.voice_max_frames, special_token_base=sp_base)
os.makedirs(a.out_dir, exist_ok=True)

attn = model.recurrent_block.recurrent_blocks[a.block].self_attn
d_v, n_heads = attn.d_values, attn.n_heads
W = attn.o_proj.weight            # (d_model, n_heads*d_v)
W0 = W.detach().clone()
print(f"block {a.block}: {n_heads} heads x d_v {d_v}; o_proj {tuple(W.shape)}", flush=True)


def measure(tag):
    r = run_tf_and_ablation(model, ds, coll, a.device, a.n, K, bs=a.bs)
    print(f"  {tag:22s} acc_real {r['acc_real']:.4f}  early_delta {r['early_text_delta']:+.4f}  "
          f"ppl {r['ppl_real']:.1f}", flush=True)
    return {"tag": tag, "acc_real": r["acc_real"], "early_text_delta": r["early_text_delta"],
            "text_delta": r["text_delta"], "ppl": r["ppl_real"]}


def ablate(h):
    with torch.no_grad():
        W.copy_(W0)
        W[:, h * d_v:(h + 1) * d_v] = 0.0


results = []
with torch.no_grad():
    W.copy_(W0)
results.append(measure("baseline"))

ablate(a.head)
results.append(measure(f"ablate ALIGNER h{a.head}"))

rng = random.Random(0)
others = [h for h in range(n_heads) if h != a.head]
for h in rng.sample(others, min(a.n_random, len(others))):
    ablate(h)
    results.append(measure(f"ablate random h{h}"))

with torch.no_grad():
    W.copy_(W0)

base = results[0]
print(f"\n{'condition':24s} {'d acc_real':>11} {'d early_delta':>14}")
for r in results[1:]:
    print(f"{r['tag']:24s} {r['acc_real']-base['acc_real']:>+11.4f} "
          f"{r['early_text_delta']-base['early_text_delta']:>+14.4f}")
rand = [r for r in results if r["tag"].startswith("ablate random")]
if rand:
    ma = sum(r["acc_real"] for r in rand) / len(rand) - base["acc_real"]
    md = sum(r["early_text_delta"] for r in rand) / len(rand) - base["early_text_delta"]
    al = results[1]
    print(f"\naligner drop / mean random drop:  acc {al['acc_real']-base['acc_real']:+.4f} vs {ma:+.4f}"
          f"   early_delta {al['early_text_delta']-base['early_text_delta']:+.4f} vs {md:+.4f}")
json.dump(results, open(os.path.join(a.out_dir, "ablation.json"), "w"), indent=2)
