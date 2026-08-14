"""Dump the SAME matched val subset as extract.py, but as raw 16 kHz waveforms +
targets (no model), so an ISOLATED-venv encoder (e.g. GLM-4-Voice, pinned to
transformers 4.44) can extract on the identical utterances without importing
megatransformer. Selection mirrors extract.py exactly (seed 7 shuffle, same
filters) so the resulting glm.pt lines up index-for-index with sive/mimi/etc.

  uv run python -m scripts_local.content_encoder_eval.dump_raw_subset --n 1000
"""
import argparse
import os
import random

import torch

from megatransformer.model.voice.sive.ctc_vocab import CTCVocab
from megatransformer.scripts.data.voice.dataset import VoiceShardedDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_cache_dir", default="./cached_datasets/voice_sive_gender_val_merged/")
    ap.add_argument("--out", default="./eval_output/content_encoder_eval/raw_subset.pt")
    ap.add_argument("--n", type=int, default=1000, help="<=0 => all")
    ap.add_argument("--sequential", action="store_true", help="dataset order (fast I/O)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    ds = VoiceShardedDataset(a.val_cache_dir, columns=["waveforms", "ctc_tokens", "speaker_ids"])
    vocab = CTCVocab()
    idxs = list(range(len(ds)))
    if not a.sequential:
        random.Random(a.seed).shuffle(idxs)
    target = len(ds) if a.n <= 0 else a.n

    recs = []
    for i in idxs:
        if len(recs) >= target:
            break
        s = ds[i]
        if "waveform" not in s or "ctc_tokens" not in s:
            continue
        ctc = s["ctc_tokens"]
        if int((ctc != vocab.blank_idx).sum()) < 2:
            continue
        wav = s["waveform"].float().reshape(-1)
        wlen = int(s.get("waveform_length", wav.shape[0]))
        wav = wav[:wlen]
        if wav.numel() < 1600:
            continue
        spk = s.get("speaker_id", s.get("speaker_ids", -1))
        spk = int(spk.item()) if torch.is_tensor(spk) else int(spk)
        recs.append({"waveform": wav.half(), "ctc_tokens": ctc.long().cpu(), "speaker_id": spk})

    torch.save({"sr": 16000, "records": recs}, a.out)
    print(f"dumped {len(recs)} raw utts -> {a.out}")


if __name__ == "__main__":
    main()
