"""Vocoder MOS ceiling: GT mel -> vocoder -> MOS predictor.

The best an SMG can sound is bounded by the vocoder: even a perfect mel prediction
is only as good as GT-mel copy-synthesis. This scores that ceiling directly — take
the stored GT Vocos mels, run them through the Vocos vocoder, and score with a neural
MOS predictor. Compare an SMG's own `mos_recon` (from mos.py) against this to get the
naturalness gap the SMG itself is responsible for vs. what the vocoder caps.

⚠️ UTMOS is a 16 kHz model: its forward() resamples the input to 16 kHz, so it is BLIND
to everything above 8 kHz — it CANNOT see the 8-12 kHz detail that is the whole point of
a 24 kHz vocoder, and will under-report Vocos's true quality. It is still the right meter
for SMG-vs-ceiling COMPARISON (the SMG recon is scored by the same UTMOS). For the absolute
24 kHz picture use a wideband metric (--wideband tries torchaudio SQUIM MOS if installed)
or the ear.

  PYTHONPATH=src python3 -m megatransformer.scripts.eval.smg.vocoder_mos \
      --cache_dir ./cached_datasets/Mekadrom/libritts_r_clean_mimi_wavlm/val --n 400 --device cpu
"""
import argparse
import glob
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description="Vocoder MOS ceiling (GT mel -> Vocos -> UTMOS)")
    ap.add_argument("--cache_dir", required=True, help="shard dir with GT Vocos mel_specs")
    ap.add_argument("--n", type=int, default=400, help="number of clips to score")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--vocos_id", default="charactr/vocos-mel-24khz")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--wideband", action="store_true",
                    help="also try torchaudio SQUIM subjective MOS (a different predictor) if installed")
    a = ap.parse_args()

    from megatransformer.utils.vocos_features import load_vocos, vocos_frame_rate
    dev = a.device
    vocos = load_vocos(dev, a.vocos_id)
    sr = 24000
    print(f"Vocos {a.vocos_id} @ {sr} Hz, mel rate {vocos_frame_rate(a.vocos_id):.2f} Hz")

    utmos = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong", trust_repo=True).to(dev).eval()

    squim = None
    if a.wideband:
        try:
            from torchaudio.pipelines import SQUIM_SUBJECTIVE
            squim = SQUIM_SUBJECTIVE.get_model().to(dev).eval()
            print("wideband: torchaudio SQUIM subjective MOS loaded (note: also 16 kHz internally)")
        except Exception as e:
            print(f"wideband: SQUIM unavailable ({e}); UTMOS only")

    shards = sorted(glob.glob(os.path.join(a.cache_dir, "shard_*.pt")))
    if not shards:
        raise SystemExit(f"no shards in {a.cache_dir}")

    mos_utmos = []
    mos_squim = []
    n_done = 0
    for sp in shards:
        if n_done >= a.n:
            break
        d = torch.load(sp, map_location="cpu")
        mels = d["mel_specs"]
        lens = d.get("mel_lengths")
        for i in range(mels.shape[0]):
            if n_done >= a.n:
                break
            L = int(lens[i]) if lens is not None else mels.shape[-1]
            mel = mels[i:i + 1, :, :L].float().to(dev)   # [1, 100, L] trimmed, no padding
            with torch.no_grad():
                wav = vocos.decode(mel)                   # [1, samples] @24k
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                # UTMOS: pass native sr; it resamples to 16 kHz internally
                mos_utmos.append(float(utmos(wav, sr).reshape(-1)[0]))
                if squim is not None:
                    # SQUIM subjective needs a non-matching reference; use the same wav as its own
                    # reference proxy (self-MOS) — a rough wideband cross-check, not calibrated NR.
                    try:
                        mos_squim.append(float(squim(wav, wav).reshape(-1)[0]))
                    except Exception:
                        pass
            n_done += 1

    u = np.array(mos_utmos)
    print(f"\n=== Vocos vocoder MOS ceiling  (n={len(u)})  cache={a.cache_dir} ===")
    print(f"UTMOS (16 kHz-band): mean {u.mean():.3f}  std {u.std():.3f}  "
          f"p10 {np.percentile(u,10):.3f}  p50 {np.percentile(u,50):.3f}  p90 {np.percentile(u,90):.3f}")
    print(f"  min {u.min():.3f}  max {u.max():.3f}")
    if mos_squim:
        s = np.array(mos_squim)
        print(f"SQUIM subj (wideband-ish): mean {s.mean():.3f}  std {s.std():.3f}  p50 {np.percentile(s,50):.3f}")
    print("\nInterpretation: an SMG's mos_recon (mos.py, same UTMOS) can't exceed the UTMOS ceiling above;"
          "\nmos_gap = ceiling - mos_recon is the SMG's own naturalness cost. UTMOS is 8 kHz-blind, so the"
          "\ntrue 24 kHz quality is higher than the number shown — use the ear for the absolute read.")


if __name__ == "__main__":
    main()
