"""Positional DECILE profile of code information vs the ACOUSTIC content that justifies it.

The probe that settles "is the tail actually monotonous, or does it just contain silence?"

An utterance tail genuinely holds trailing silence and breath, so SOME decline in code
entropy toward the end is FAITHFUL, not collapse. Judging the tail without an acoustic
reference therefore over-reports. This bins each utterance into deciles and compares the
code-side decline against a mel-side one over the SAME bins:

  code_H      unigram entropy of the codes in that decile (bits)
  uniq        distinct codes in that decile
  repeat      frac of frames identical to the previous
  mel_energy  mean log-mel = how loud the region is
  mel_delta   mean |frame-to-frame mel change| = how much ACOUSTIC ACTIVITY is there

Both curves are then normalized to decile 0 and subtracted:

  gap = rel(code_H) - rel(mel_delta)
  gap <  0  code information fell FASTER than the audio justifies = real monotony
  gap >= 0  the decline tracks the audio = faithful, not collapse

Also runs a per-utterance PAIRED test (1st vs 2nd half unique-code rate) so the effect is not
judged from pooled means: |t| > 2 means the drop is systematic across utterances.

Validated behaviour: on a healthy run the worst region is 70-90% (full-loudness speech coded
with a shrinking alphabet), NOT the very end -- at the very end low energy legitimately
explains part of it. Residual monotony that resolves with training shows the paired t rising
toward 0 and the tail deciles' gaps closing together.

    python -m megatransformer.scripts.eval.audio.sive.code_tail_profile \
        --checkpoint run=runs/sive/<run>/checkpoint-62000 --vq_cosine --n_utts 150
"""
import argparse
from collections import Counter

import numpy as np

from megatransformer.scripts.eval.audio.sive.vq_code_common import (
    add_common_args, code_sequences, load_sive, parse_checkpoints)


def _entropy(counter):
    p = np.array(list(counter.values()), float)
    if p.sum() == 0:
        return 0.0
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def profile(seqs, nbins):
    bins = [Counter() for _ in range(nbins)]
    rep, energy, delta = [[] for _ in range(nbins)], [[] for _ in range(nbins)], [[] for _ in range(nbins)]
    pair_first, pair_second = [], []

    for e in seqs:
        c, m = e["codes"].numpy(), e["mel"]
        fl = len(c)
        h = fl // 2
        pair_first.append(len(np.unique(c[:h])) / h)
        pair_second.append(len(np.unique(c[h:])) / (fl - h))

        cedge = np.linspace(0, fl, nbins + 1).astype(int)
        medge = np.linspace(0, m.shape[1], nbins + 1).astype(int)
        for b in range(nbins):
            seg = c[cedge[b]:cedge[b + 1]]
            if len(seg):
                bins[b].update(seg.tolist())
                if len(seg) > 1:
                    rep[b].append(float(np.mean(np.diff(seg) == 0)))
            ms = m[:, medge[b]:medge[b + 1]]
            if ms.shape[1] > 1:
                energy[b].append(float(ms.mean()))
                delta[b].append(float(np.abs(np.diff(ms, axis=1)).mean()))

    agg = lambda x: np.array([np.mean(v) if v else 0.0 for v in x])
    return (np.array([_entropy(b) for b in bins]), np.array([len(b) for b in bins], float),
            agg(rep), agg(energy), agg(delta), np.array(pair_first), np.array(pair_second))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--nbins", type=int, default=10, help="positional bins (10 = deciles)")
    args = ap.parse_args()

    for name, path in parse_checkpoints(args):
        print(f"\n{'='*70}\n[{name}] {path}\n{'='*70}")
        model = load_sive(args, path)
        seqs = code_sequences(model, args, want_mel=True)
        H, U, R, E, D, a, b = profile(seqs, args.nbins)
        step = 100 // args.nbins

        print(f"\n[{name}] {len(a)} utts")
        print(f"{'bin %':<10}{'code_H(b)':>11}{'uniq':>7}{'repeat':>9}{'  |':>4}"
              f"{'mel_energy':>12}{'mel_delta':>11}")
        for i in range(args.nbins):
            print(f"{i*step:>3}-{(i+1)*step:<6}{H[i]:>11.2f}{U[i]:>7.0f}{R[i]:>9.1%}{'  |':>4}"
                  f"{E[i]:>12.2f}{D[i]:>11.3f}")

        rel = lambda x: x / x[0] if x[0] != 0 else x
        rH, rD = rel(H), rel(D)
        print(f"\n  relative to bin 0 -- does code info fall FASTER than the acoustics justify?")
        print(f"{'bin %':<10}{'code_H':>9}{'mel_delta':>11}{'gap':>9}")
        for i in range(args.nbins):
            print(f"{i*step:>3}-{(i+1)*step:<6}{rH[i]:>9.3f}{rD[i]:>11.3f}{rH[i]-rD[i]:>+9.3f}")
        print("    gap<0 => code entropy fell MORE than acoustic activity = residual monotony")
        print("    gap>=0 => decline is justified by the audio (faithful, not collapse)")

        d = b - a
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 and d.std(ddof=1) > 0 else 0.0
        print(f"\n  paired 1st vs 2nd half (unique-code rate), n={len(d)}:")
        print(f"    1st={a.mean():.3f}  2nd={b.mean():.3f}  mean_diff={d.mean():+.4f}  t={t:+.2f}  "
              f"utts_with_lower_tail={(d < 0).mean():.1%}")
        print("    |t|>2 => systematic across utterances, not a pooled-average artifact")


if __name__ == "__main__":
    main()
