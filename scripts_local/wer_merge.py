"""Pool per-shard tts_intelligibility JSONs (<out>/shard_*/tts_intelligibility.json) into one
result and recompute aggregate WER/CER/coverage over the union of utterances."""
import glob
import json
import os
import sys

import numpy as np


def main(out_dir):
    shard_jsons = sorted(glob.glob(os.path.join(out_dir, "shard_*", "tts_intelligibility.json")))
    if not shard_jsons:
        raise SystemExit(f"no shard JSONs under {out_dir}/shard_*/")
    samples, ckpt = [], None
    for p in shard_jsons:
        d = json.load(open(p))
        ckpt = ckpt or d.get("checkpoint")
        samples.extend(d.get("samples", []))
    # dedup by idx (stripes are disjoint, but be safe)
    seen, uniq = set(), []
    for s in samples:
        k = s.get("idx")
        if k in seen:
            continue
        seen.add(k)
        uniq.append(s)
    gen = [s for s in uniq if not s.get("no_voice")]
    wers = np.array([s["wer"] for s in gen], dtype=float)
    cers = np.array([s["cer"] for s in gen], dtype=float)
    out = {
        "checkpoint": ckpt, "n": len(uniq), "n_shards": len(shard_jsons),
        "coverage": len(gen) / max(1, len(uniq)),
        "wer_mean": float(wers.mean()) if len(wers) else None,
        "wer_median": float(np.median(wers)) if len(wers) else None,
        "cer_mean": float(cers.mean()) if len(cers) else None,
        "cer_median": float(np.median(cers)) if len(cers) else None,
        "pct_intelligible_wer_le_0.2": float((wers <= 0.2).mean() * 100) if len(wers) else None,
        "samples": uniq,
    }
    merged_path = os.path.join(out_dir, "tts_intelligibility_merged.json")
    json.dump(out, open(merged_path, "w"), indent=2)
    print(f"=== merged TTS intelligibility ({out['n']} utts, {out['n_shards']} shards) ===")
    print(f"  coverage: {out['coverage']*100:.1f}%")
    print(f"  WER  mean={out['wer_mean']:.3f}  median={out['wer_median']:.3f}")
    print(f"  CER  mean={out['cer_mean']:.3f}  median={out['cer_median']:.3f}")
    print(f"  %% WER<=0.2 (intelligible): {out['pct_intelligible_wer_le_0.2']:.1f}%")
    print(f"  -> {merged_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
