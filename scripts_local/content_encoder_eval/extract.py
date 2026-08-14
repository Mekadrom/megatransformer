"""Stage 1: run ONE content encoder over a fixed val subset and dump per-utterance
records the metric harness (compare.py) consumes. Decoupling extraction (per-model,
possibly per-venv, messy deps) from metrics (uniform, one venv) keeps dependency
hell out of the comparison.

Dump (torch .pt): {"meta": {...}, "records": [ {feats fp16 [L,D], codes long [L] or
None, pooled fp32 [D], ctc_tokens long, speaker_id int}, ... ]}.

Usage:
  CUDA_VISIBLE_DEVICES=3 uv run python -m scripts_local.content_encoder_eval.extract \
      --encoder sive --checkpoint runs/sive/<stdhinge11>/checkpoint-300000 --n 1000
  CUDA_VISIBLE_DEVICES=3 uv run python -m scripts_local.content_encoder_eval.extract --encoder mimi --n 1000
  CUDA_VISIBLE_DEVICES=3 uv run python -m scripts_local.content_encoder_eval.extract --encoder contentvec --n 1000
"""
import argparse
import os
import random

import torch

from megatransformer.model.voice.sive.ctc_vocab import CTCVocab
from megatransformer.scripts.data.voice.dataset import VoiceShardedDataset
from scripts_local.content_encoder_eval.encoders import build_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--checkpoint", default=None, help="required for --encoder sive")
    ap.add_argument("--dim", type=int, default=256, help="contentvec width (256|768)")
    ap.add_argument("--val_cache_dir", default="./cached_datasets/voice_sive_gender_val_merged/")
    ap.add_argument("--out_dir", default="./eval_output/content_encoder_eval")
    ap.add_argument("--n", type=int, default=1000, help="<=0 => all utterances")
    ap.add_argument("--pooled_only", action="store_true",
                    help="store only mean-pooled vec + speaker_id (for full-val leakage)")
    ap.add_argument("--sequential", action="store_true",
                    help="iterate in dataset (shard) order for fast sequential I/O")
    ap.add_argument("--out_name", default=None, help="override output basename")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    enc = build_encoder(a.encoder, device=a.device, checkpoint=a.checkpoint, dim=a.dim)
    print(f"[{enc.name}] dim={enc.dim} frame_rate={enc.frame_rate:.2f}Hz discrete={enc.is_discrete}")

    ds = VoiceShardedDataset(a.val_cache_dir, columns=["waveforms", "ctc_tokens", "speaker_ids"])
    vocab = CTCVocab()

    idxs = list(range(len(ds)))
    if not a.sequential:
        random.Random(a.seed).shuffle(idxs)
    target = len(ds) if a.n <= 0 else a.n

    records = []
    seen = 0
    for i in idxs:
        if len(records) >= target:
            break
        s = ds[i]
        if "waveform" not in s or "ctc_tokens" not in s:
            continue
        ctc = s["ctc_tokens"]
        if int((ctc != vocab.blank_idx).sum()) < 2:
            continue
        wav = s["waveform"].float().reshape(-1)
        wlen = int(s.get("waveform_length", wav.shape[0]))
        wav = wav[:wlen]  # trim padding
        if wav.numel() < 1600:  # <0.1s, skip degenerate
            continue
        out = enc.encode(wav)
        if out.feats.shape[0] < 2:
            continue
        spk = s.get("speaker_id", s.get("speaker_ids", -1))
        spk = int(spk.item()) if torch.is_tensor(spk) else int(spk)
        # .clone() drops shared/view storage so pickle stores only the live tensor
        # (a sliced view otherwise serializes its full backing buffer). See
        # feedback_torch_slicing_pickle.
        if a.pooled_only:
            records.append({"pooled": out.feats.mean(0).float().clone(),
                            "speaker_id": spk})
        else:
            records.append({
                "feats": out.feats.half().clone(),               # [L, D] fp16
                "codes": None if out.codes is None else out.codes.cpu().clone(),
                "pooled": out.feats.mean(0).float().clone(),     # [D]
                "ctc_tokens": ctc.long().cpu().clone(),
                "speaker_id": spk,
            })
        seen += 1
        if seen % 500 == 0:
            print(f"  {len(records)} kept / {seen} seen (target {target})")

    meta = {"encoder": enc.name, "dim": enc.dim, "frame_rate": enc.frame_rate,
            "is_discrete": enc.is_discrete, "n": len(records),
            "pooled_only": a.pooled_only}
    out_path = os.path.join(a.out_dir, f"{a.out_name or enc.name}.pt")
    torch.save({"meta": meta, "records": records}, out_path)
    print(f"\nsaved {len(records)} records -> {out_path}")
    print(f"  meta: {meta}")


if __name__ == "__main__":
    main()
