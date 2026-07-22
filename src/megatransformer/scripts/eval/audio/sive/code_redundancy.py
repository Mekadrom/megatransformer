"""VQ-code redundancy / dedup / n-gram structure -- how much INFORMATION the codes carry.

code_diversity.py counts the alphabet; this measures whether the sequence is actually
informative or merely long. Reported per region (all / 1st half / 2nd half):

  repeat              frac of frames identical to the previous frame
  dedup_ratio         run-length-collapsed length / original length
  uni_H (ppl, used)   unigram entropy in bits, its perplexity = EFFECTIVE codes, and how
                      many distinct codes appear at all
  bigram_top1         next-code top-1 predictability given the previous code
  deduped_bigram_top1 same after removing trivial repeats = AR-cheatability

Two findings this probe exists to preserve:
  1. REPEAT RATE IS NOT THE COLLAPSE SIGNAL. A collapsed run measured a perfectly FLAT
     7.7%/7.8% repeat across halves while its tail fell to 4.15 bits (17.8 effective codes)
     and 68% next-code predictability. A dedup-only check calls that healthy.
  2. It catches tails that code_diversity.py calls fixed: a "+16% drop = FIXED" checkpoint
     had tail entropy 7.22 -> 6.64 bits and DOUBLED predictability (11.6% -> 22.3%).

Low deduped_bigram_top1 (vs 1/K chance) also means the codes are not trivially predictable,
which is what the world-model voice-AR path needs.

    python -m megatransformer.scripts.eval.audio.sive.code_redundancy \
        --checkpoint run=runs/sive/<run>/checkpoint-45000 --vq_cosine --n_utts 150
"""
import argparse
from collections import Counter

import numpy as np

from megatransformer.scripts.eval.audio.sive.vq_code_common import (
    add_common_args, code_sequences, load_sive, parse_checkpoints)


def dedup(c):
    return c[np.insert(np.diff(c) != 0, 0, True)]


def _cond_top1(bigrams):
    """Sum over previous-codes of the most common successor / total = top-1 hit rate."""
    by_prev = {}
    for (a, b), n in bigrams.items():
        by_prev.setdefault(a, []).append(n)
    total = sum(sum(v) for v in by_prev.values())
    return sum(max(v) for v in by_prev.values()) / max(total, 1)


def region_stats(seqs, label, num_codes):
    if not seqs:
        return
    rep = np.mean([np.mean(np.diff(c) == 0) if len(c) > 1 else 0.0 for c in seqs])
    ddr = np.mean([len(dedup(c)) / len(c) for c in seqs])
    uni, big, big_d = Counter(), Counter(), Counter()
    for c in seqs:
        uni.update(c.tolist())
        big.update(zip(c[:-1].tolist(), c[1:].tolist()))
        d = dedup(c)
        big_d.update(zip(d[:-1].tolist(), d[1:].tolist()))
    p = np.array(list(uni.values()), float)
    p /= p.sum()
    H = float(-(p * np.log2(p)).sum())
    print(f"  {label:<10} repeat={rep:6.1%}  dedup_ratio={ddr:5.2f}  uni_H={H:5.2f}b "
          f"(ppl={2**H:6.1f}/{len(uni):3d} used of {num_codes})  "
          f"bigram_top1={_cond_top1(big):5.1%}  deduped_bigram_top1={_cond_top1(big_d):5.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args = ap.parse_args()

    for name, path in parse_checkpoints(args):
        print(f"\n{'='*70}\n[{name}] {path}\n{'='*70}")
        model = load_sive(args, path)
        K = int(getattr(model.config, "vq_num_codes", 0))
        seqs = [e["codes"].numpy() for e in code_sequences(model, args)]
        print(f"\n[{name}] {len(seqs)} utts   (chance top-1 = 1/{K} = {1.0/max(K,1):.3%})")
        region_stats(seqs, "ALL", K)
        region_stats([c[:len(c) // 2] for c in seqs], "1st-half", K)
        region_stats([c[len(c) // 2:] for c in seqs], "2nd-half", K)
    print("\n  a 2nd-half entropy drop with RISING predictability = tail degradation, even if")
    print("  code_diversity.py called the alphabet healthy")


if __name__ == "__main__":
    main()
