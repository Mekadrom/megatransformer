"""Per-position VQ-code diversity -- the first-pass tail-collapse screen.

Splits each utterance's code sequence in half and reports the fraction of frames carrying a
DISTINCT code in each half. A VQ-SIVE whose CTC front-loads the transcript blanks the tail
over real speech; those content-less frames snap to a handful of codes and the SMG trained on
them flatlines partway through every utterance.

    drop = 1 - (2nd-half diversity / 1st-half diversity)      >20% => TAIL-COLLAPSE

⚠️ This is a SCREEN, not a verdict. It measures ALPHABET SIZE only and is blind to how
PREDICTABLE the sequence is -- a checkpoint can pass here while its tail entropy has already
fallen and its next-code predictability doubled (this is exactly how a "+16% = FIXED" reading
once hid a genuinely degraded tail). ALWAYS pair it with code_redundancy.py, and use
code_tail_profile.py to locate the damage and check it against the audio.

    python -m megatransformer.scripts.eval.audio.sive.code_diversity \
        --checkpoint run=runs/sive/<run>/checkpoint-45000 --vq_cosine --n_utts 150
"""
import argparse

import numpy as np

from megatransformer.scripts.eval.audio.sive.vq_code_common import (
    add_common_args, code_sequences, load_sive, parse_checkpoints)


def analyze(seqs):
    first, second, runs = [], [], []
    for e in seqs:
        c = e["codes"].numpy()
        fl = len(c)
        h = fl // 2
        first.append(len(np.unique(c[:h])) / max(h, 1))
        second.append(len(np.unique(c[h:])) / max(fl - h, 1))
        # longest single-code run, as a fraction of the utterance
        best = cur = 1
        for t in range(1, fl):
            cur = cur + 1 if c[t] == c[t - 1] else 1
            best = max(best, cur)
        runs.append(best / fl)
    return np.array(first), np.array(second), np.array(runs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args = ap.parse_args()

    for name, path in parse_checkpoints(args):
        print(f"\n{'='*70}\n[{name}] {path}\n{'='*70}")
        model = load_sive(args, path)
        seqs = code_sequences(model, args)
        first, second, runs = analyze(seqs)
        drop = 1 - second.mean() / max(first.mean(), 1e-6)
        verdict = "TAIL-COLLAPSE" if drop > 0.20 else "uniform=FIXED"
        print(f"\n[{name}] {len(first)} utts")
        print(f"  code diversity  1st-half={first.mean():.2f}  2nd-half={second.mean():.2f}  "
              f"drop={100*drop:+.0f}%   [{verdict}]")
        print(f"  longest single-code run (frac of utt): mean={runs.mean():.2f}  max={runs.max():.2f}")
    print("\nreminder: alphabet size only -- confirm with code_redundancy.py + code_tail_profile.py")


if __name__ == "__main__":
    main()
