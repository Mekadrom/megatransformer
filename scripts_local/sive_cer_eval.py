"""Standalone greedy CER/WER eval for a SIVE checkpoint (reproduces the trainer's eval/cer).

The trainer computes eval/cer via greedy CTC decode of asr_logits vs the ctc_token targets.
This does the same for an arbitrary checkpoint, so we can read CER off a checkpoint whose TB
metrics are missing/corrupted (e.g. stdhinge11's crash-truncated 200k-300k tail). VALIDATE it
by first running a checkpoint with a known TB eval/cer (const@87k = 0.474); if it reproduces,
the number for the base is trustworthy.
"""
import argparse
import random

import jiwer
import torch

from megatransformer.model.voice.sive.sive import SpeakerInvariantVoiceEncoder
from megatransformer.model.voice.sive.ctc_vocab import CTCVocab
from megatransformer.utils.model_loading_utils import detect_sive_variant, load_model
from megatransformer.utils.audio_utils import SharedWindowBuffer, extract_mels
from megatransformer.scripts.data.voice.dataset import VoiceShardedDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="small_deep_3xdownsample_conv2d_attentive")
    ap.add_argument("--num_speakers", type=int, default=3610)
    ap.add_argument("--val_cache_dir", default="./cached_datasets/voice_sive_gender_val_merged/")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--activation", default="swiglu")
    ap.add_argument("--use_std_hinge", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    vq = detect_sive_variant(a.checkpoint)
    if vq.get("use_vq"):
        vq["vq_cosine"] = True
    overrides = {"num_speakers": a.num_speakers, "activation": a.activation,
                 "use_std_hinge": a.use_std_hinge, **vq}
    model = load_model(SpeakerInvariantVoiceEncoder, a.config,
                       checkpoint_path=a.checkpoint, overrides=overrides)
    model.to(a.device).eval()
    print(f"loaded {a.checkpoint}\n  VQ={vq}  ctc_upsample={model.config.ctc_upsample_factor}")

    ds = VoiceShardedDataset(a.val_cache_dir, columns=["waveforms", "ctc_tokens", "text"])
    vocab = CTCVocab()
    swb = SharedWindowBuffer()

    idxs = list(range(len(ds)))
    random.Random(a.seed).shuffle(idxs)
    idxs = idxs[:a.n]

    preds, refs = [], []
    with torch.no_grad():
        for i in idxs:
            s = ds[i]
            if "waveform" not in s or "ctc_tokens" not in s:
                continue
            # Reference = the CTC target decoded (no CTC collapse) — exactly what the
            # trainer scores greedy decodes against.
            ref = vocab.decode(s["ctc_tokens"].tolist(), remove_blanks=True,
                               collapse_repeats=False).lower().strip()
            if not ref:
                continue
            wav = s["waveform"].to(a.device).float().reshape(-1)  # 1D [T], padded
            wlen = int(s.get("waveform_length", wav.shape[0]))
            wav = wav[:wlen]                          # trim padding -> real speech only
            mel = extract_mels(swb, wav, sr=16000, n_mels=80, n_fft=1024, hop_length=256)
            if mel.dim() == 2:                       # [n_mels, T] -> [1, n_mels, T]
                mel = mel.unsqueeze(0)
            mlen = torch.tensor([mel.shape[-1]], device=a.device)
            out = model(mel.to(a.device), lengths=mlen, grl_alpha=0.0)
            logits = out["asr_logits"]
            clen = out.get("ctc_lengths", out.get("feature_lengths"))
            hyp = vocab.ctc_decode_greedy(logits, clen)[0].lower().strip()
            preds.append(hyp)
            refs.append(ref)

    cer = jiwer.cer(refs, preds)
    wer = jiwer.wer(refs, preds)
    print(f"\n==> n={len(preds)}  greedy CER={cer:.4f}  WER={wer:.4f}")
    for j in range(min(4, len(preds))):
        print(f"  REF: {refs[j][:90]}")
        print(f"  HYP: {preds[j][:90]}\n")


if __name__ == "__main__":
    main()
