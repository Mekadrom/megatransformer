"""N-gram next-unit prediction baselines over the training VQ unit sequences.

The "text-free ceiling": the best a pure unit-statistics model can do at predicting the
next SIVE-VQ unit from the previous (n-1) units, with NO access to the text. Compare the
world model's voice_unit_accuracy against these:

  model >  best n-gram   => it is using MORE than local unit history (text / long-range) -- the snap.
  model == best n-gram   => it is only doing n-gram statistics (text being ignored -- the AR crutch).
  model <  best n-gram   => it hasn't even captured local unit statistics yet (still warming up).

Held-out BY UTTERANCE (fit on one split, eval on a disjoint split) so the numbers are honest
generalization -- an in-sample n-gram trivially memorizes and its top-1 climbs toward 100% as n
grows, which is meaningless. Each order-n model backs off to lower orders on unseen contexts;
coverage + seen-only accuracy show how much backoff is happening. Also reports the
repeat-previous-unit baseline (the pure AR crutch) and the unigram floor.

CPU only (no model, no GPU). Units are quantized EXACTLY as the dataset does it
(codebook.quantize == L2 nearest), so they match what the model trains on. EOV/padding are
excluded -- content units only (EOV is <1% of tokens; negligible vs the ~15% figure).

Usage:
  python scripts_local/ngram_unit_baseline.py <train_shard_dir> [codebook.pt] \
      [--max_utts 0] [--heldout 0.1] [--max_n 5] [--seed 0]
  (codebook defaults to <train_shard_dir>/sive_vq_codebook.pt; --max_utts 0 = all)
"""
import argparse
import glob
import math
import os
import random
from collections import Counter, defaultdict

import torch

from megatransformer.utils.codebook import load_codebook


def load_sequences(shard_dir, codebook, max_utts, heldout_frac, seed):
    """Quantize every utterance to its content-unit sequence, split fit/heldout by utterance."""
    shards = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
    if not shards:
        raise SystemExit(f"No shard_*.pt in {shard_dir}")
    rng = random.Random(seed)
    fit, held = [], []
    n_utts = n_units = 0
    for si, path in enumerate(shards):
        s = torch.load(path, map_location="cpu", weights_only=False)
        lens = s["feature_lengths"]      # (N,)
        # PRE-QUANTIZED cache (e.g. Mimi cb0): units are stored directly as `unit_ids`, there is
        # no continuous `features` tensor to quantize. Use the ids verbatim (they already ARE the
        # dataset's quantization) -- codebook is only needed for K. Slice each utt to its valid
        # length so padding (and any trailing EOV, which is not stored) is excluded.
        if "unit_ids" in s:
            uids = s["unit_ids"]         # (N, T) int
            N = uids.shape[0]
            for j in range(N):
                L = int(lens[j])
                if L <= 0:
                    continue
                seq = uids[j, :L].tolist()
                (held if rng.random() < heldout_frac else fit).append(seq)
                n_utts += 1
                n_units += L
                if max_utts and n_utts >= max_utts:
                    print(f"  [{si+1}/{len(shards)}] reached --max_utts {max_utts}", flush=True)
                    return fit, held, n_utts, n_units
            if (si + 1) % 8 == 0:
                print(f"  shard {si+1}/{len(shards)}: {n_utts} utts, {n_units} units", flush=True)
            continue
        feats = s["features"]            # (N, D, T)
        N = feats.shape[0]
        # Vectorized quantize for the whole shard: stack all valid frames, one cdist.
        chunks, offs = [], []
        for j in range(N):
            L = int(lens[j])
            if L <= 0:
                offs.append((0, 0)); continue
            fr = feats[j, :, :L].T.float()   # (L, D)
            offs.append((sum(c.shape[0] for c in chunks), L))
            chunks.append(fr)
        if not chunks:
            continue
        allfr = torch.cat(chunks, 0)                       # (M, D)
        ids = torch.cdist(allfr, codebook).argmin(1)       # (M,) L2-nearest == dataset.quantize
        for j in range(N):
            start, L = offs[j]
            if L <= 0:
                continue
            seq = ids[start:start + L].tolist()
            (held if rng.random() < heldout_frac else fit).append(seq)
            n_utts += 1
            n_units += L
            if max_utts and n_utts >= max_utts:
                print(f"  [{si+1}/{len(shards)}] reached --max_utts {max_utts}", flush=True)
                return fit, held, n_utts, n_units
        if (si + 1) % 8 == 0:
            print(f"  shard {si+1}/{len(shards)}: {n_utts} utts, {n_units} units", flush=True)
    return fit, held, n_utts, n_units


def build_counts(sequences, max_n):
    """counts[o][ctx] = Counter(next_unit), for order o (context length o-1), o=1..max_n."""
    counts = [None] + [defaultdict(Counter) for _ in range(max_n)]  # 1-indexed
    for seq in sequences:
        for t in range(len(seq)):
            nxt = seq[t]
            for o in range(1, max_n + 1):
                if t >= o - 1:
                    ctx = tuple(seq[t - (o - 1):t])   # length o-1 (() for unigram)
                    counts[o][ctx][nxt] += 1
    return counts


def eval_order(counts, sequences, n, K):
    """Backed-off order-n model: top-1 acc (with backoff), seen-only acc, coverage, ppl, ce_norm."""
    hits = seen_hits = seen = total = 0
    ce = 0.0
    logK = math.log(K)
    for seq in sequences:
        for t in range(len(seq)):
            actual = seq[t]
            total += 1
            dist = None
            full_seen = False
            for o in range(n, 0, -1):               # backoff n -> n-1 -> ... -> 1
                if t < o - 1:
                    continue
                ctx = tuple(seq[t - (o - 1):t])
                d = counts[o].get(ctx)
                if d:
                    dist = d
                    if o == n:
                        full_seen = True
                    break
            if dist is None:                        # unseen even as unigram (shouldn't happen)
                ce += logK
                continue
            tot = sum(dist.values())
            # top-1 (argmax; deterministic tie-break by lowest id)
            best = min(dist.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            if best == actual:
                hits += 1
            # add-1 smoothed prob of the actual unit
            ce += -math.log((dist.get(actual, 0) + 1) / (tot + K))
            if full_seen:
                seen += 1
                if best == actual:
                    seen_hits += 1
    return {
        "top1": hits / max(total, 1),
        "seen_only_top1": seen_hits / max(seen, 1),
        "coverage": seen / max(total, 1),
        "ppl": math.exp(ce / max(total, 1)),
        "ce_norm": (ce / max(total, 1)) / logK,   # comparable to voice_unit_ce_loss_norm
        "total": total,
    }


def repeat_baseline(sequences):
    """Predict u[t] = u[t-1] (the pure AR crutch). t=0 has no prior -> skipped."""
    hits = total = 0
    for seq in sequences:
        for t in range(1, len(seq)):
            total += 1
            if seq[t] == seq[t - 1]:
                hits += 1
    return hits / max(total, 1), total


def main():
    ap = argparse.ArgumentParser(description="N-gram unit-prediction baselines (text-free ceiling)")
    ap.add_argument("shard_dir")
    ap.add_argument("codebook", nargs="?", default=None)
    ap.add_argument("--max_utts", type=int, default=0, help="0 = all utterances")
    ap.add_argument("--heldout", type=float, default=0.1)
    ap.add_argument("--max_n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cb_path = args.codebook or os.path.join(args.shard_dir, "sive_vq_codebook.pt")
    codebook = load_codebook(cb_path).float()
    K = codebook.shape[0]
    print(f"Codebook: {tuple(codebook.shape)} (K={K}) from {cb_path}")
    print(f"Loading + quantizing shards from {args.shard_dir} ...", flush=True)

    fit, held, n_utts, n_units = load_sequences(
        args.shard_dir, codebook, args.max_utts, args.heldout, args.seed
    )
    print(f"\n{n_utts} utterances, {n_units} content units | fit={len(fit)} heldout={len(held)}")
    used = len({u for seq in fit for u in seq})
    print(f"distinct units used (fit): {used}/{K}")
    if not held:
        raise SystemExit("Heldout split empty -- raise --heldout or --max_utts.")

    print("Building n-gram counts on the fit split ...", flush=True)
    counts = build_counts(fit, args.max_n)

    rep_acc, _ = repeat_baseline(held)
    print(f"\nEvaluating on {len(held)} held-out utterances "
          f"({sum(len(s) for s in held)} positions)\n")

    hdr = f"{'model':<16}{'top1':>9}{'seen-only':>11}{'coverage':>10}{'ppl':>9}{'ce_norm':>9}"
    print(hdr); print("-" * len(hdr))
    print(f"{'repeat u[t-1]':<16}{rep_acc:>9.4f}{'-':>11}{'-':>10}{'-':>9}{'-':>9}")
    best_top1 = 0.0
    for n in range(1, args.max_n + 1):
        r = eval_order(counts, held, n, K)
        name = "unigram" if n == 1 else f"{n}-gram"
        print(f"{name:<16}{r['top1']:>9.4f}{r['seen_only_top1']:>11.4f}"
              f"{r['coverage']:>10.4f}{r['ppl']:>9.2f}{r['ce_norm']:>9.4f}")
        best_top1 = max(best_top1, r["top1"])

    print("\n" + "=" * 60)
    print(f"TEXT-FREE CEILING (best n-gram top-1): {best_top1:.4f}")
    print(f"AR-crutch (repeat) baseline          : {rep_acc:.4f}")
    print("Compare voice_unit_accuracy: below ceiling = still warming up;")
    print("at ceiling = doing n-gram stats (text ignored); above = using text.")


if __name__ == "__main__":
    main()
