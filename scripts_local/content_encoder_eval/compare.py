"""Stage 2: three-axis comparison of content encoders from their extract.py dumps.

Axes (all on the SAME utterances / protocol, each encoder at its NATIVE rate except
where noted):
  CONTENT   — greedy CER/WER of a small CTC probe (feats -> chars). Lower = more
              phonetic content is (shallowly) decodable. Probe feats are upsampled
              to a common 50 Hz so the CTC alignment budget doesn't penalize low-rate
              encoders (interp adds no info, only frames). The number to beat: SIVE ~0.60.
  REDUNDANCY— how much of the sequence is deduplicable. Continuous: mean adjacent-frame
              cosine + dedup ratio at cos>=0.95/0.99. Discrete (native codes, else common-K
              k-means): repeat / dedup_ratio / unigram-entropy / bigram predictability.
              This is the "ContentVec-at-50Hz is redundant" hypothesis, quantified.
  LEAKAGE   — speaker identifiability from the mean-pooled feature (reuses the SIVE
              per_speaker_leakage probe: macro@1 x chance). Higher = more speaker leaks
              = worse for the voice-conversion SMG. SIVE/ContentVec are built to be low.

Usage (after extracting each encoder):
  CUDA_VISIBLE_DEVICES=3 uv run python -m scripts_local.content_encoder_eval.compare \
      --dumps sive mimi contentvec --dump_dir ./eval_output/content_encoder_eval
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatransformer.model.voice.sive.ctc_vocab import CTCVocab
from megatransformer.scripts.eval.audio.sive.code_redundancy import dedup, region_stats
from megatransformer.scripts.eval.audio.sive.per_speaker_leakage import (
    stratified_split, train_probe, per_speaker_metrics)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_dump(dump_dir, name):
    d = torch.load(os.path.join(dump_dir, f"{name}.pt"), weights_only=False)
    return d["meta"], d["records"]


# --------------------------------------------------------------------------- #
# REDUNDANCY
# --------------------------------------------------------------------------- #
def continuous_redundancy(records):
    adj, ded95, ded99 = [], [], []
    for r in records:
        f = r["feats"].float()
        if f.shape[0] < 3:
            continue
        fn = F.normalize(f, dim=-1)
        cos = (fn[1:] * fn[:-1]).sum(-1)                 # [L-1] adjacent cosine
        adj.append(cos.mean().item())
        # dedup ratio: fraction of frames RETAINED after collapsing near-duplicates
        ded95.append(1.0 - (cos >= 0.95).float().mean().item())
        ded99.append(1.0 - (cos >= 0.99).float().mean().item())
    return {"adj_cos": float(np.mean(adj)),
            "dedup_ratio_cos95": float(np.mean(ded95)),
            "dedup_ratio_cos99": float(np.mean(ded99))}


def discrete_redundancy(records, meta, common_k=512, max_fit_frames=60000):
    """Native codes if the encoder is discrete; else common-K k-means on feats so
    every encoder gets a comparable run-length/entropy read."""
    if records[0]["codes"] is not None:
        seqs = [r["codes"].numpy().astype(np.int64) for r in records]
        K = int(max(s.max() for s in seqs)) + 1
        src = f"native (K={K})"
    else:
        from sklearn.cluster import KMeans
        pool = np.concatenate([r["feats"].float().numpy() for r in records], 0)
        if pool.shape[0] > max_fit_frames:
            sel = np.random.RandomState(0).choice(pool.shape[0], max_fit_frames, replace=False)
            fit = pool[sel]
        else:
            fit = pool
        km = KMeans(n_clusters=common_k, n_init=4, random_state=0).fit(fit)
        seqs = [km.predict(r["feats"].float().numpy()).astype(np.int64) for r in records]
        K = common_k
        src = f"kmeans (K={K})"
    rep = float(np.mean([np.mean(np.diff(c) == 0) if len(c) > 1 else 0.0 for c in seqs]))
    ddr = float(np.mean([len(dedup(c)) / len(c) for c in seqs]))
    from collections import Counter
    uni = Counter()
    for c in seqs:
        uni.update(c.tolist())
    p = np.array(list(uni.values()), float)
    p /= p.sum()
    H = float(-(p * np.log2(p)).sum())
    return {"src": src, "repeat": rep, "dedup_ratio": ddr, "uni_H": H,
            "ppl": 2 ** H, "used": len(uni), "K": K}


# --------------------------------------------------------------------------- #
# LEAKAGE (reuse the SIVE per-speaker-leakage probe machinery)
# --------------------------------------------------------------------------- #
class MLPProbe(nn.Module):
    def __init__(self, d, k, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, k))

    def forward(self, x):
        return self.net(x)


def leakage_metric(records, min_utts=5, test_split=0.3, seed=7):
    X = np.stack([r["pooled"].numpy() for r in records]).astype(np.float32)
    spk_raw = np.array([r["speaker_id"] for r in records])
    tr, te, kept, ndspk, ndutt = stratified_split(spk_raw, test_split, min_utts, seed)
    if len(kept) < 5 or len(te) < 5:
        return {"error": f"too few speakers with >={min_utts} utts (kept {len(kept)})"}
    remap = {int(s): i for i, s in enumerate(kept)}
    y = np.array([remap.get(int(s), -1) for s in spk_raw])
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mu) / sd
    K = len(kept)
    probe = MLPProbe(X.shape[1], K)
    res = train_probe(probe, Xn[tr], y[tr], Xn[te], y[te], K,
                      max_epochs=300, patience=30, lr=1e-3, batch_size=256, device=DEVICE)
    m = per_speaker_metrics(res["top5_preds"], y[te], K)
    return {"macro_top1": m["macro_top1"], "ratio_macro_top1": m["ratio_macro_top1"],
            "ratio_micro_top1": m["ratio_micro_top1"], "K_spk": K,
            "chance": m["chance_top1"], "plateaued": res["plateaued"]}


# --------------------------------------------------------------------------- #
# CONTENT — CTC probe (feats -> chars), CER/WER
# --------------------------------------------------------------------------- #
class CTCProbe(nn.Module):
    def __init__(self, d, vocab, h=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d, h, 5, padding=2), nn.GELU(),
            nn.Conv1d(h, h, 5, padding=2), nn.GELU())
        self.out = nn.Linear(h, vocab)

    def forward(self, x):  # x [B, L, D]
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.out(h)  # [B, L, V]


def _upsample_to_rate(f, src_rate, dst_rate):
    if abs(src_rate - dst_rate) < 1e-6:
        return f
    L2 = max(2, int(round(f.shape[0] * dst_rate / src_rate)))
    return F.interpolate(f.transpose(0, 1).unsqueeze(0), size=L2, mode="linear",
                         align_corners=False)[0].transpose(0, 1)


def content_cer(records, meta, probe_rate=50.0, steps=3000, bs=16, seed=7,
                eval_frac=0.2):
    vocab = CTCVocab()
    rng = np.random.RandomState(seed)
    src_rate = meta["frame_rate"]
    feats, tgts = [], []
    for r in records:
        f = _upsample_to_rate(r["feats"].float(), src_rate, probe_rate)
        t = r["ctc_tokens"]
        t = t[t != vocab.blank_idx].long()  # strip blank-padding -> real transcript
        if t.numel() < 2 or f.shape[0] < t.numel():   # need L >= U for CTC
            continue
        mu, sd = f.mean(0, keepdim=True), f.std(0, keepdim=True) + 1e-6
        feats.append(((f - mu) / sd))
        tgts.append(t)
    n = len(feats)
    idx = rng.permutation(n)
    n_eval = max(4, int(n * eval_frac))
    ev, trn = idx[:n_eval], idx[n_eval:]

    d = feats[0].shape[1]
    probe = CTCProbe(d, vocab.vocab_size).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=3e-4)
    ctc = nn.CTCLoss(blank=vocab.blank_idx, zero_infinity=True)

    probe.train()
    for step in range(steps):
        bi = rng.choice(trn, size=min(bs, len(trn)), replace=False)
        maxL = max(feats[i].shape[0] for i in bi)
        xb = torch.zeros(len(bi), maxL, d)
        il = torch.zeros(len(bi), dtype=torch.long)
        tl = torch.zeros(len(bi), dtype=torch.long)
        tcat = []
        for j, i in enumerate(bi):
            L = feats[i].shape[0]
            xb[j, :L] = feats[i]
            il[j] = L
            tl[j] = tgts[i].numel()
            tcat.append(tgts[i])
        logp = probe(xb.to(DEVICE)).log_softmax(-1).transpose(0, 1)  # [L,B,V]
        loss = ctc(logp, torch.cat(tcat).to(DEVICE), il.to(DEVICE), tl.to(DEVICE))
        opt.zero_grad(); loss.backward(); opt.step()

    import jiwer
    probe.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for i in ev:
            logits = probe(feats[i].unsqueeze(0).to(DEVICE))[0]
            hyp = vocab.ctc_decode_greedy(logits,
                    torch.tensor([logits.shape[0]], device=DEVICE))[0].lower().strip()
            ref = vocab.decode(tgts[i].tolist(), remove_blanks=True,
                               collapse_repeats=False).lower().strip()
            if ref:
                refs.append(ref); hyps.append(hyp)
    return {"cer": float(jiwer.cer(refs, hyps)), "wer": float(jiwer.wer(refs, hyps)),
            "n_probe": n, "n_eval": len(refs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--dump_dir", default="./eval_output/content_encoder_eval")
    ap.add_argument("--common_k", type=int, default=512)
    ap.add_argument("--probe_rate", type=float, default=50.0)
    ap.add_argument("--ctc_steps", type=int, default=3000)
    ap.add_argument("--skip_content", action="store_true")
    ap.add_argument("--leakage_only", action="store_true",
                    help="pooled-only dumps: run just the speaker-leakage probe")
    ap.add_argument("--min_utts", type=int, default=5)
    ap.add_argument("--suffix", default="", help="dump filename suffix, e.g. _full")
    a = ap.parse_args()

    if a.leakage_only:
        print(f"{'encoder':<16} {'macro@1':>8} {'x chance':>9} {'K spk':>6} {'chance':>8} {'plat':>5}")
        print("-" * 60)
        for name in a.dumps:
            meta, recs = load_dump(a.dump_dir, name + a.suffix)
            leak = leakage_metric(recs, min_utts=a.min_utts)
            if "error" in leak:
                print(f"{name:<16} {leak['error']}")
            else:
                print(f"{name:<16} {leak['macro_top1']:>8.3f} {leak['ratio_macro_top1']:>8.1f}x "
                      f"{leak['K_spk']:>6} {leak['chance']:>8.4f} {str(leak['plateaued']):>5}  "
                      f"(n={meta['n']})")
        return

    rows = []
    for name in a.dumps:
        meta, recs = load_dump(a.dump_dir, name)
        print(f"\n[{name}] n={meta['n']} dim={meta['dim']} rate={meta['frame_rate']:.2f}Hz "
              f"discrete={meta['is_discrete']}")
        cont = continuous_redundancy(recs)
        disc = discrete_redundancy(recs, meta, common_k=a.common_k)
        leak = leakage_metric(recs, min_utts=a.min_utts)
        con = None if a.skip_content else content_cer(recs, meta, probe_rate=a.probe_rate,
                                                      steps=a.ctc_steps)
        print(f"  redundancy(cont): adj_cos={cont['adj_cos']:.3f}  "
              f"dedup@.95={cont['dedup_ratio_cos95']:.2f}  dedup@.99={cont['dedup_ratio_cos99']:.2f}")
        print(f"  redundancy(disc {disc['src']}): repeat={disc['repeat']:.1%}  "
              f"dedup_ratio={disc['dedup_ratio']:.2f}  uni_H={disc['uni_H']:.2f}b "
              f"ppl={disc['ppl']:.0f}/{disc['used']} used")
        if "error" in leak:
            print(f"  leakage: {leak['error']}")
        else:
            print(f"  leakage: macro@1={leak['macro_top1']:.3f} = {leak['ratio_macro_top1']:.1f}x chance "
                  f"({leak['K_spk']} spk, plateaued={leak['plateaued']})")
        if con:
            print(f"  content: CER={con['cer']:.3f}  WER={con['wer']:.3f}  "
                  f"(probe n={con['n_probe']}, eval={con['n_eval']})")
        rows.append((name, cont, disc, leak, con))

    # summary table
    print("\n" + "=" * 92)
    print(f"{'encoder':<12} {'rate':>6} {'CER':>6} {'WER':>6} | {'adj_cos':>7} {'dd@.95':>6} "
          f"{'dd_disc':>7} {'uni_H':>6} | {'leak x':>7}")
    print("-" * 92)
    for name, cont, disc, leak, con in rows:
        meta, _ = load_dump(a.dump_dir, name)
        cer = f"{con['cer']:.3f}" if con else "  -  "
        wer = f"{con['wer']:.3f}" if con else "  -  "
        lk = f"{leak['ratio_macro_top1']:.1f}" if "error" not in leak else "err"
        print(f"{name:<12} {meta['frame_rate']:>5.1f}H {cer:>6} {wer:>6} | "
              f"{cont['adj_cos']:>7.3f} {cont['dedup_ratio_cos95']:>6.2f} "
              f"{disc['dedup_ratio']:>7.2f} {disc['uni_H']:>6.2f} | {lk:>6}")
    print("=" * 92)
    print("CER lower=more content | dd=dedup ratio (lower=more deduplicable/redundant) | "
          "leak x=macro speaker id over chance (lower=more invariant)")


if __name__ == "__main__":
    main()
