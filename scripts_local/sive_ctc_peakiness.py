"""Measure CTC peakiness / front-loading of a SIVE checkpoint's asr_logits.

Question this answers: is the CTC head (a) peaky (sparse spikes, blank-dominated)
and (b) FRONT-LOADED (emissions crammed into the head of the sequence, tail left
blank)? These are the two structural CTC failure modes behind the SIVE dying-tail.

Metrics (per valid CTC frame = feature_lengths * ctc_upsample_factor, padding trimmed):
- blank_frac      : fraction of frames whose argmax is <blank> (idx 0). High => peaky.
- entropy_nats    : mean per-frame posterior entropy. Low => confident/peaky.
- spikes_per_char : # non-blank emission EVENTS (runs) / transcript char count.
                    ~1 = one spike per char (classic peaky); >>1 = smeared; <1 = dropped.
- com             : center-of-mass of non-blank frames / valid_len. 0.5 = centered;
                    <0.5 = front-loaded.
- last_pos        : position of the LAST non-blank frame / valid_len. <1.0 => a blank
                    tail (the dying-tail signature). Trailing silence justifies SOME
                    gap; a large gap is front-loading.
- first_half_frac : fraction of non-blank emissions in the first half of the sequence.
                    0.5 = balanced; >0.5 = front-loaded.
- decile hist     : avg fraction of a utterance's emissions falling in each 10% bin of
                    the (normalized) sequence. Flat ~0.10 each = healthy; a decaying
                    profile = front-loading.

Usage:
  CUDA_VISIBLE_DEVICES=3 uv run python scripts_local/sive_ctc_peakiness.py \
      --checkpoint runs/sive/<run>/checkpoint-300000 --use_std_hinge --n 400
"""
import argparse
import random

import torch
import torch.nn.functional as F

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
    ap.add_argument("--n", type=int, default=400)
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

    NB = 10  # deciles
    agg = {k: [] for k in ["blank_frac", "entropy", "spikes_per_char",
                           "com", "last_pos", "first_half_frac"]}
    decile = torch.zeros(NB)
    decile_n = 0

    with torch.no_grad():
        for i in idxs:
            s = ds[i]
            if "waveform" not in s or "ctc_tokens" not in s:
                continue
            n_chars = int((s["ctc_tokens"] != vocab.blank_idx).sum())  # transcript length U
            if n_chars < 2:
                continue
            wav = s["waveform"].to(a.device).float().reshape(-1)
            wlen = int(s.get("waveform_length", wav.shape[0]))
            wav = wav[:wlen]
            mel = extract_mels(swb, wav, sr=16000, n_mels=80, n_fft=1024, hop_length=256)
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)
            mlen = torch.tensor([mel.shape[-1]], device=a.device)
            out = model(mel.to(a.device), lengths=mlen, grl_alpha=0.0)
            logits = out["asr_logits"][0]                       # [T', V]
            clen = out.get("ctc_lengths", out.get("feature_lengths"))
            clen = int(clen[0]) if clen is not None else logits.shape[0]
            clen = min(clen, logits.shape[0])
            logits = logits[:clen]                              # trim CTC padding

            probs = F.softmax(logits, dim=-1)
            ent = -(probs * torch.log(probs + 1e-9)).sum(-1).mean().item()
            argmax = logits.argmax(-1)                          # [clen]
            is_blank = argmax == vocab.blank_idx
            nonblank = ~is_blank
            agg["blank_frac"].append(is_blank.float().mean().item())
            agg["entropy"].append(ent)

            # emission events = # of runs of consecutive non-blank frames
            prev = torch.cat([torch.zeros(1, dtype=torch.bool, device=nonblank.device),
                              nonblank[:-1]])
            starts = nonblank & (~prev)
            n_events = int(starts.sum())
            agg["spikes_per_char"].append(n_events / n_chars)

            nb_idx = nonblank.nonzero().flatten().float()
            if nb_idx.numel() == 0:
                continue
            agg["com"].append(((nb_idx.mean() + 0.5) / clen).item())
            agg["last_pos"].append(((nb_idx.max() + 1) / clen).item())
            agg["first_half_frac"].append((nb_idx < clen / 2).float().mean().item())

            # per-utt decile histogram of emission positions (normalized), then avg
            bins = torch.clamp((nb_idx / clen * NB).long(), 0, NB - 1)
            h = torch.bincount(bins.cpu(), minlength=NB).float()
            decile += h / h.sum()
            decile_n += 1

    def ms(k):
        v = torch.tensor(agg[k])
        return v.mean().item(), v.std().item()

    print(f"\n==> n={len(agg['blank_frac'])} utts")
    print(f"  blank_frac      = {ms('blank_frac')[0]:.3f} ± {ms('blank_frac')[1]:.3f}   (peaky if high)")
    print(f"  entropy_nats    = {ms('entropy')[0]:.3f} ± {ms('entropy')[1]:.3f}   (peaky if low; max={torch.log(torch.tensor(float(logits.shape[-1]))).item():.2f})")
    print(f"  spikes_per_char = {ms('spikes_per_char')[0]:.3f} ± {ms('spikes_per_char')[1]:.3f}   (~1 = one spike/char)")
    print(f"  com             = {ms('com')[0]:.3f} ± {ms('com')[1]:.3f}   (0.5=centered, <0.5=front-loaded)")
    print(f"  last_pos        = {ms('last_pos')[0]:.3f} ± {ms('last_pos')[1]:.3f}   (<1.0 => blank tail)")
    print(f"  first_half_frac = {ms('first_half_frac')[0]:.3f} ± {ms('first_half_frac')[1]:.3f}   (>0.5=front-loaded)")
    if decile_n:
        prof = (decile / decile_n).tolist()
        print("\n  emission decile profile (frac of emissions per 10% of sequence; flat 0.10=healthy):")
        print("   pos: " + " ".join(f"{(j+1)*10:4d}%" for j in range(NB)))
        print("   frac: " + " ".join(f"{p:5.3f}" for p in prof))


if __name__ == "__main__":
    main()
