"""Paired bootstrap CIs over the WER-eval arms, so decode/arm rankings stop being read off
point estimates.

WHY: at n=64 utterances the LCS-recall differences between arms have been 4-6% relative
(0.1104 vs 0.1058). Nothing in cosyvoice_wer_eval reports an interval, so those gaps were
being ranked by eye. Every arm decodes THE SAME val utterances, so the comparison can be
PAIRED -- which removes between-utterance variance (by far the largest source here: per-utt
LCS recall ranges 0 to ~0.9) and is much tighter than comparing two independent CIs.

Reads every wer_step*_ras*.json under a directory tree, recomputes per-utterance LCS recall
with the same normalization/LCS the eval used, and reports:
  - per-arm mean + 95% CI (bootstrap over utterances)
  - paired differences vs a chosen baseline arm, with 95% CI and the fraction of bootstrap
    resamples favouring each arm (a directional read that survives small n better than a
    significance verdict)

  uv run python scripts_local/wer_arm_compare.py <root_dir> [--key generated|truncated]
                                                 [--baseline <arm>] [--iters 10000]
"""
import argparse, glob, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from megatransformer.scripts.eval.world.tts_intelligibility import normalize_text


def lcs_recall(ref_words, hyp_words):
    """Fraction of REFERENCE words appearing in order in the hypothesis (LCS / len(ref)).

    Duplicated from cosyvoice_wer_eval rather than imported: that module parses argv at
    import time, so importing it here would demand its full CLI. Keep the two in sync.
    """
    n, m = len(ref_words), len(hyp_words)
    if n == 0:
        return float("nan")
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        for j in range(1, m + 1):
            cur[j] = prev[j - 1] + 1 if ref_words[i - 1] == hyp_words[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[m] / n

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--key", default="both", choices=["generated", "truncated", "both"],
                help="'generated' = full hypothesis (favours verbose arms); 'truncated' = "
                     "hypothesis cut to reference length (length-fair). Default reports both, "
                     "because the two disagree whenever arms differ in length.")
ap.add_argument("--baseline", default=None, help="arm name to compare against (default: best mean)")
ap.add_argument("--iters", type=int, default=10000)
a = ap.parse_args()

ROWKEY = {"generated": "hyp", "truncated": "hyp_trunc"}


def load(root):
    arms = {}
    for f in sorted(glob.glob(os.path.join(root, "**", "wer_step*_ras*.json"), recursive=True)):
        d = json.load(open(f))
        # arm name = parent dir + ras suffix, so ras/non-ras of one config stay distinct
        name = os.path.basename(os.path.dirname(f))
        if d.get("ras_win"):
            name += f"+ras{d['ras_win']}"
        arms[name] = d
    return arms


def per_utt(d, rowkey):
    """{utterance idx: lcs_recall} for one arm."""
    out = {}
    for r in d["rows"]:
        ref = normalize_text(r["ref"])
        if not ref:
            continue
        hyp = normalize_text(r.get(rowkey, ""))
        out[r["idx"]] = lcs_recall(ref.split(), hyp.split())
    return out


def boot_mean(vals, iters, rng):
    n = len(vals)
    s = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return s[int(0.025 * iters)], s[int(0.975 * iters)]


def boot_paired(dx, iters, rng):
    """CI on the mean paired difference + fraction of resamples > 0."""
    n = len(dx)
    s, wins = [], 0
    for _ in range(iters):
        m = sum(dx[rng.randrange(n)] for _ in range(n)) / n
        s.append(m)
        wins += m > 0
    s.sort()
    return s[int(0.025 * iters)], s[int(0.975 * iters)], wins / iters


arms = load(a.root)
if not arms:
    sys.exit(f"no wer_step*_ras*.json under {a.root}")

for key in (["generated", "truncated"] if a.key == "both" else [a.key]):
    rowkey = ROWKEY[key]
    scores = {k: per_utt(d, rowkey) for k, d in arms.items()}
    rng = random.Random(0)
    print(f"\n{'='*78}\nLCS recall — {key}"
          f"{'  (length-fair: hypothesis truncated to reference length)' if key=='truncated' else '  (full hypothesis; a longer arm has more chances to contain the reference)'}"
          f"\n{'='*78}")
    means = {k: sum(v.values()) / len(v) for k, v in scores.items()}
    print(f"{'arm':<28} {'n':>4} {'mean':>8}   95% CI")
    for k in sorted(means, key=means.get, reverse=True):
        lo, hi = boot_mean(list(scores[k].values()), a.iters, rng)
        print(f"{k:<28} {len(scores[k]):>4} {means[k]:>8.4f}   [{lo:.4f}, {hi:.4f}]")

    base = a.baseline or max(means, key=means.get)
    if base not in scores:
        sys.exit(f"baseline {base} not among {list(scores)}")
    print(f"\npaired differences vs {base} (same utterances; positive = arm is better)")
    print(f"{'arm':<28} {'n_pair':>6} {'delta':>9}   95% CI                P(arm better)")
    for k in sorted(means, key=means.get, reverse=True):
        if k == base:
            continue
        shared = sorted(set(scores[k]) & set(scores[base]))
        if not shared:
            continue
        dx = [scores[k][i] - scores[base][i] for i in shared]
        lo, hi, p = boot_paired(dx, a.iters, rng)
        sig = "" if lo <= 0 <= hi else "  <-- CI excludes 0"
        print(f"{k:<28} {len(shared):>6} {sum(dx)/len(dx):>+9.4f}   [{lo:+.4f}, {hi:+.4f}]   {p:5.2f}{sig}")
