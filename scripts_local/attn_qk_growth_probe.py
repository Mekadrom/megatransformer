"""Is MuonClip (qk-clip) indicated for this model? Weight-only trend over checkpoints.

WHY THIS FORM: MuonClip fires on max pre-softmax attention logit growing without bound, and
ACTS by rescaling W_q / W_k. The logits themselves are never materialized here -- every run
has attn_logit_cap=None so attention takes the SDPA path (transformer.py:271) -- but the
quantity qk-clip controls is available directly from the weights:

    max_ij (q_i . k_j) / sqrt(d_q)  <=  max_i||q_i|| * max_j||k_j|| / sqrt(d_q)
                                    <=  sigma_max(W_q) * sigma_max(W_k) * max_i||x_i||^2 / sqrt(d_q)

RoPE is norm-preserving per rotated pair, so it does not affect any of these. GQA repetition
does not change the SET of key vectors. So sigma_max(W_q)*sigma_max(W_k)/sqrt(d_q) is both a
valid data-free bound on the logit scale AND exactly the product qk-clip shrinks.

This is an UPPER BOUND, deliberately. A flat bound is a sound negative (the true max is
bounded above by it). A growing bound is NOT proof of a real spike -- escalate to an
activation-side measurement in that case.

Per-head where possible: W_q is (n_heads*d_q, d_model), so it splits into per-head blocks and
qk-clip is applied per head. GQA means W_k has fewer heads; each q head is paired with its
group's k head.
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True, help="runs/<dir>/<run> containing checkpoint-*/")
ap.add_argument("--pattern", default="checkpoint-*", help="glob under --run_dir")
ap.add_argument("--d_queries", type=int, default=64)
ap.add_argument("--top", type=int, default=8, help="how many worst heads to name")
ap.add_argument("--exclude", default="llm_body",
                help="comma-separated name substrings to skip. Default llm_body = the FROZEN\n                     SmolLM2 spine, whose weights never move and whose heads otherwise\n                     dominate the max and hide the trainable trend entirely.")
ap.add_argument("--out", default=None, help="optional jsonl of per-checkpoint rows")
a = ap.parse_args()


def sigma_max(W):
    """Largest singular value. float32 on CPU; these are at most 3072x768."""
    return torch.linalg.matrix_norm(W.float(), ord=2).item()


def load_sd(ckpt):
    for name in ("pytorch_model.bin", "model.safetensors"):
        p = os.path.join(ckpt, name)
        if os.path.exists(p):
            if name.endswith(".bin"):
                return torch.load(p, map_location="cpu", weights_only=False)
            from safetensors.torch import load_file
            return load_file(p)
    return None


def step_of(path):
    m = re.search(r"checkpoint-(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


ckpts = sorted(glob.glob(os.path.join(a.run_dir, a.pattern)), key=step_of)
ckpts = [c for c in ckpts if step_of(c) >= 0 and os.path.isdir(c)]
if not ckpts:
    sys.exit(f"no checkpoints under {a.run_dir}/{a.pattern}")
print(f"{len(ckpts)} checkpoints: {[step_of(c) for c in ckpts]}\n")

rows = []
for ck in ckpts:
    sd = load_sd(ck)
    if sd is None:
        print(f"  {os.path.basename(ck)}: no weight file, skipped")
        continue
    ema = "-ema" in os.path.basename(ck)
    pairs = {}
    for k in sd:
        if k.endswith("q_proj.weight"):
            base = k[: -len("q_proj.weight")]
            kk = base + "k_proj.weight"
            if kk in sd:
                pairs[base] = (sd[k], sd[kk])
    skip = [x.strip() for x in a.exclude.split(",") if x.strip()]
    per_head = []
    for base, (Wq, Wk) in pairs.items():
        if any(x in base for x in skip):
            continue
        dq = a.d_queries
        nq, nk = Wq.shape[0] // dq, Wk.shape[0] // dq
        if nq == 0 or nk == 0:
            continue
        group = max(nq // max(nk, 1), 1)
        for h in range(nq):
            q_blk = Wq[h * dq:(h + 1) * dq]
            kh = min(h // group, nk - 1)
            k_blk = Wk[kh * dq:(kh + 1) * dq]
            per_head.append((sigma_max(q_blk) * sigma_max(k_blk) / (dq ** 0.5), f"{base}h{h}"))
    if not per_head:
        continue
    per_head.sort(reverse=True)
    vals = torch.tensor([v for v, _ in per_head])
    row = {"ckpt": os.path.basename(ck), "step": step_of(ck), "ema": ema,
           "n_heads": len(per_head), "max": vals.max().item(),
           "p95": vals.quantile(0.95).item(), "mean": vals.mean().item(),
           "argmax": per_head[0][1]}
    rows.append(row)
    print(f"  {row['ckpt']:24s} heads={row['n_heads']:3d}  "
          f"max={row['max']:8.3f}  p95={row['p95']:8.3f}  mean={row['mean']:8.3f}   {row['argmax']}")

print()
raw = [r for r in rows if not r["ema"]]
series = raw if len(raw) >= 2 else rows
if len(series) >= 2:
    f, l = series[0], series[-1]
    d = (l["max"] / f["max"] - 1) * 100 if f["max"] else float("nan")
    dm = (l["mean"] / f["mean"] - 1) * 100 if f["mean"] else float("nan")
    print(f"TREND over steps {f['step']} -> {l['step']} ({'raw' if series is raw else 'all'} ckpts):")
    print(f"  max  sigma(Wq)*sigma(Wk)/sqrt(d):  {f['max']:.3f} -> {l['max']:.3f}   ({d:+.1f}%)")
    print(f"  mean sigma(Wq)*sigma(Wk)/sqrt(d):  {f['mean']:.3f} -> {l['mean']:.3f}   ({dm:+.1f}%)")
    print()
    print("  READING: a flat or shrinking bound means uncapped logits are NOT diverging and")
    print("  qk-clip has nothing to act on. Sustained growth is a reason to measure the")
    print("  activation-side max logit before implementing anything.")

if a.out:
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {a.out}")
