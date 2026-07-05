"""F0-contour expressiveness (semitone-std ratio) on already-rendered synthesis WAVs.

Objectifies "flat delivery" / "less acting in the voice": the semitone-std of the voiced
F0 contour is register-invariant intonation dynamics. ratio = recon_true / target on the
SAME utterance (controls for content, the way GV does for texture) — **<1 = the recon
FLATTENS the prosody** (monotone). Scores the saved render WAVs, so NO probe re-train
(the tiny probe isn't checkpointed). Same metric synthesis_usability --> reports inline.

Usage:
  python -m megatransformer.scripts.eval.audio.sive.expressiveness_wavs --device cuda \
    --dir stdhinge11=<render_dir> --dir groupnormfrontend=<render_dir> --dir covreg=<render_dir>
"""
import argparse
import glob
import os
import re

import numpy as np
import torch

FNAME = re.compile(r"sample(\d+)_spk(\d+)_([mf])_(.+)\.wav$")


def _semitone_std(path, device):
    import torchcrepe
    from scipy.io import wavfile
    _sr, wav = wavfile.read(path)
    w = torch.tensor(np.clip(wav.astype(np.float32) / 32768.0, -1, 1)).unsqueeze(0)
    pitch, period = torchcrepe.predict(
        w, 16000, hop_length=256, fmin=50.0, fmax=550.0, model="full",
        decoder=torchcrepe.decode.viterbi, return_periodicity=True,
        batch_size=512, device=device, pad=True)
    voiced = period[0] > 0.5
    if int(voiced.sum()) < 2:
        return float("nan")
    vp = pitch[0][voiced]
    return float((12.0 * torch.log2(vp.clamp(min=1e-3))).std().item())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="F0-contour expressiveness (intonation semitone-std ratio) on rendered WAVs")
    ap.add_argument("--dir", action="append", required=True, metavar="name=render_dir", help="Repeatable.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"\n{'variant':>18} {'intonation(recon)':>18} {'intonation(GT)':>16} {'ratio':>7}   n   (<1 = flatter delivery)")
    for spec in args.dir:
        name, d = spec.split("=", 1)
        samples = {}
        for p in glob.glob(os.path.join(d, "*.wav")):
            m = FNAME.search(os.path.basename(p))
            if m:
                samples.setdefault(int(m.group(1)), {})[m.group(4)] = p
        sr_, st_, ratios = [], [], []
        for s in samples.values():
            if "recon_true" in s and "target" in s:
                sr = _semitone_std(s["recon_true"], args.device)
                st = _semitone_std(s["target"], args.device)
                if not (np.isnan(sr) or np.isnan(st)) and st > 0:
                    sr_.append(sr); st_.append(st); ratios.append(sr / st)
        if ratios:
            print(f"{name:>18} {np.median(sr_):>18.2f} {np.median(st_):>16.2f} {np.median(ratios):>7.2f} {len(ratios):>4}")
        else:
            print(f"{name:>18}   (no voiced recon_true/target pairs)")


if __name__ == "__main__":
    main()
