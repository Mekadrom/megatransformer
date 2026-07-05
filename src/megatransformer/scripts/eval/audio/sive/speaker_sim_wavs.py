"""Objective speaker-similarity (ECAPA cosine) on already-rendered synthesis WAVs.

Scores the identity axis the SIVE base decision hinges on — the axis leakage / recon-L1 /
disentangle / UTMOS are all blind to (e.g. the speaker-1420 male->feminine flip). Works on
the render set synthesis_usability already saves, so NO probe re-training is needed (the tiny
probe is not checkpointed):

  identity = cos(ECAPA(recon_true), ECAPA(target))                  did recon keep the SOURCE speaker
  convert  = cos(ECAPA(recon_wrong_<g>), ECAPA(xref_spk<j>_<g>))    did conversion land on the TARGET

Low identity on a same-speaker recon = the decoder produced the wrong person even WITH the
correct embedding (the 1420 failure). Same ECAPA encoder as the stored FiLM embeddings.

Usage:
  python -m megatransformer.scripts.eval.audio.sive.speaker_sim_wavs \
    --dir stdhinge11=<render_dir> --dir groupnormfrontend=<render_dir> --dir covreg=<render_dir>
"""
import argparse
import glob
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile

from megatransformer.utils.speaker_encoder import SpeakerEncoderWrapper
from megatransformer.utils.audio_utils import SharedWindowBuffer, extract_mels

FNAME = re.compile(r"sample(\d+)_spk(\d+)_([mf])_(.+)\.wav$")


@torch.no_grad()
def _emb(enc, buf, path, dev):
    _sr, wav = wavfile.read(path)
    wav = torch.tensor(wav.astype(np.float32) / 32768.0)
    mel = extract_mels(buf, wav, sr=16000, n_mels=80, n_fft=1024, hop_length=256).unsqueeze(0).to(dev)  # [1,80,T]
    e = enc(mel_spec=mel).reshape(-1)
    return F.normalize(e, dim=-1)


@torch.no_grad()
def score_dir(enc, buf, d, dev):
    samples = {}
    for p in glob.glob(os.path.join(d, "*.wav")):
        m = FNAME.search(os.path.basename(p))
        if not m:
            continue
        k = int(m.group(1))
        s = samples.setdefault(k, {})
        s["spk"], s["g"], s[m.group(4)] = int(m.group(2)), m.group(3), p
    rows = []
    for k in sorted(samples):
        s = samples[k]
        ident = conv = float("nan")
        if "recon_true" in s and "target" in s:
            ident = float((_emb(enc, buf, s["recon_true"], dev) * _emb(enc, buf, s["target"], dev)).sum())
        wrong = next((t for t in s if t.startswith("recon_wrong_")), None)
        xref = next((t for t in s if t.startswith("xref_spk")), None)
        if wrong and xref:
            conv = float((_emb(enc, buf, s[wrong], dev) * _emb(enc, buf, s[xref], dev)).sum())
        rows.append((k, s["spk"], s["g"], ident, conv))
    return rows


def main():
    ap = argparse.ArgumentParser(description="ECAPA speaker-similarity on rendered synthesis WAVs")
    ap.add_argument("--dir", action="append", required=True, metavar="name=render_dir", help="Repeatable.")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    enc = SpeakerEncoderWrapper(encoder_type="ecapa_tdnn", device=args.device)
    buf = SharedWindowBuffer()
    summary = []
    for spec in args.dir:
        name, d = spec.split("=", 1)
        rows = score_dir(enc, buf, d, args.device)
        idents = [r[3] for r in rows if r[3] == r[3]]
        convs = [r[4] for r in rows if r[4] == r[4]]
        print(f"\n=== {name} ===  ({d})")
        print(f"  {'sample':>7} {'spk':>7} {'g':>2}  {'identity':>9} {'convert':>8}")
        for k, spk, g, ident, conv in rows:
            flag = "  <-- 1420 (the flip case)" if spk == 1420 else ""
            print(f"  {('s'+str(k)):>7} {spk:>7} {g:>2}  {ident:>9.3f} {conv:>8.3f}{flag}")
        print(f"  MEAN identity={np.mean(idents):.3f}  convert={np.mean(convs):.3f}  (higher=better; identity=kept source, convert=landed on target)")
        summary.append((name, np.mean(idents), np.mean(convs)))

    print("\n=== SUMMARY (mean over render set) ===")
    print(f"  {'variant':>18} {'identity':>9} {'convert':>8}")
    for name, i, c in summary:
        print(f"  {name:>18} {i:>9.3f} {c:>8.3f}")


if __name__ == "__main__":
    main()
