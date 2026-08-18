"""Calibrate early_text_delta: what does the metric read for a model that WORKS?

We have been steering by early_text_delta without knowing its achievable range on this
data. The CosyVoice 2 teacher demonstrably does the task, so running the SAME shuffled-text
ablation on it gives the ceiling.

Reads either way:
  teacher delta >> student's ~0.024  -> the metric has headroom; keep optimizing against it
  teacher delta ~= student's         -> text moves little even for a working model, the
                                        metric has no dynamic range here, stop steering by it

Reports bootstrap 95% CIs over UTTERANCES so we stop narrating differences smaller than the
error bar (the student trajectory -0.0005/+0.0132/+0.0283/+0.0244 was read at n=256, where
~0.005 moves are noise).

  uv run python scripts_local/teacher_text_ablation.py --model_dir <cv2> \
      --cache_dir cached_datasets/.../val --n 1024 --device cuda:2
"""
import argparse, glob, os, random
import torch

from megatransformer.model.voice.cosyvoice2_teacher import CosyVoice2Teacher


def bootstrap_ci(per_utt, iters=10000, seed=0, alpha=0.05):
    """per_utt: list of (real_hits, shuf_hits, n). Returns (delta, lo, hi) as rates."""
    rng = random.Random(seed)
    n = len(per_utt)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    R = sum(x[0] for x in per_utt); S = sum(x[1] for x in per_utt); N = sum(x[2] for x in per_utt)
    point = (R - S) / max(N, 1)
    samples = []
    for _ in range(iters):
        r = s = c = 0
        for _ in range(n):
            a, b, k = per_utt[rng.randrange(n)]
            r += a; s += b; c += k
        samples.append((r - s) / max(c, 1))
    samples.sort()
    return point, samples[int(alpha / 2 * iters)], samples[int((1 - alpha / 2) * iters)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--early_k", type=int, default=8, help="match the student's early window")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    t = CosyVoice2Teacher.from_pretrained(a.model_dir, device=a.device, dtype=torch.bfloat16)

    texts, units, lens = [], [], []
    for f in sorted(glob.glob(os.path.join(a.cache_dir, "shard_*.pt"))):
        s = torch.load(f, map_location="cpu", weights_only=False)
        for i in range(len(s["text"])):
            L = int(s["feature_lengths"][i])
            if L >= a.early_k + 2:
                texts.append(s["text"][i]); units.append(s["unit_ids"][i, :L]); lens.append(L)
            if len(texts) >= a.n:
                break
        if len(texts) >= a.n:
            break
    print(f"{len(texts)} utterances, mean {sum(lens)/len(lens):.1f} frames", flush=True)

    all_utt, early_utt = [], []
    for st in range(0, len(texts), a.bs):
        bt = texts[st:st + a.bs]
        bu = units[st:st + a.bs]
        bl = torch.tensor(lens[st:st + a.bs])
        if len(bt) < 2:
            continue
        T = int(bl.max())
        pad = torch.zeros(len(bt), T, dtype=torch.long)
        for r, u in enumerate(bu):
            pad[r, :len(u)] = u
        pad = pad.to(a.device)
        # real text vs text rolled by one across the batch (each voice + a WRONG transcript)
        shuf = bt[-1:] + bt[:-1]
        lg_r, mk = t(bt, pad, bl, T)
        lg_s, _ = t(shuf, pad, bl, T)
        for r in range(len(bt)):
            L = int(bl[r])
            m = mk[r, :L]                                  # content frames only (drop EOV)
            if not bool(m.any()):
                continue
            gt = pad[r, :L]
            pr = lg_r[r, :L].argmax(-1); ps = lg_s[r, :L].argmax(-1)
            hr = ((pr == gt) & m); hs = ((ps == gt) & m)
            all_utt.append((int(hr.sum()), int(hs.sum()), int(m.sum())))
            ek = min(a.early_k, L)
            me = m[:ek]
            early_utt.append((int(hr[:ek].sum()), int(hs[:ek].sum()), int(me.sum())))
        if (st // a.bs) % 10 == 0:
            print(f"  {st + len(bt)}/{len(texts)} ...", flush=True)

    for name, per in (("all-position", all_utt), (f"early (first {a.early_k})", early_utt)):
        d, lo, hi = bootstrap_ci(per)
        R = sum(x[0] for x in per); S = sum(x[1] for x in per); N = sum(x[2] for x in per)
        print(f"\n{name}: n_utt={len(per)} positions={N}")
        print(f"  teacher acc real      {R/N:.4f}")
        print(f"  teacher acc shuffled  {S/N:.4f}")
        print(f"  TEXT DELTA            {d:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")

    print("\nStudent reference (nocurric @20k, n=256): early_text_delta +0.0244, acc_real 0.0814")
    print("If the teacher's early delta is close to the student's, early_text_delta has little "
          "dynamic range on this data and should not be the steering metric.")


if __name__ == "__main__":
    main()
