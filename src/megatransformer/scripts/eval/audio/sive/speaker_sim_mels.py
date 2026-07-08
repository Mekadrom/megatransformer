"""Mel-domain ECAPA speaker-similarity — the vocoder-free counterpart to
speaker_sim_wavs.py.

speaker_sim_wavs runs ECAPA on the VOCODED wavs (tiny-probe mel -> HiFiGAN ->
re-extract mel -> ECAPA), so its cosine includes a lossy round-trip that can be
OOD for HiFiGAN/ECAPA in a NORM-DEPENDENT way (a run whose decoded mels have a
different spectral character — e.g. batchnorm-final vs layernorm-final — gets
scored on a shifted scale). This script feeds the tiny-probe's PRE-vocoder mel
straight into ECAPA (which takes mel natively), isolating the feature/decoder
identity from the vocoder step. Run both and compare: if a run's cosine JUMPS in
the mel domain, the vocoder round-trip — not the features — was sandbagging it.

Requires a render produced with `synthesis_usability --save_render_mels` (writes
<render_dir>/mels/sample*.mels.pt with target/recon_true/recon_wrong/xref mels).

Usage:
    python -m megatransformer.scripts.eval.audio.sive.speaker_sim_mels \
        --dir stdhinge11=eval_output/.../synth/stdhinge11 \
        --dir batchnorm=eval_output/.../synth/batchnorm
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

from megatransformer.utils.speaker_encoder import SpeakerEncoderWrapper


def _ecapa(enc, mel, device):
    """ECAPA embedding of a [n_mels, T] mel fed DIRECTLY (no vocoder)."""
    m = mel.to(device).float()
    if m.dim() == 2:
        m = m.unsqueeze(0)  # [1, n_mels, T]
    emb = enc._forward_ecapa_tdnn(m, lengths=None)  # [1, 192]
    return emb.squeeze(0)


@torch.no_grad()
def score_dir(enc, d, device):
    mel_dir = os.path.join(d, "mels")
    rows = []
    for p in sorted(glob.glob(os.path.join(mel_dir, "sample*.mels.pt"))):
        b = torch.load(p, map_location="cpu")
        k = int(os.path.basename(p).split("_")[0].replace("sample", ""))
        e_true = _ecapa(enc, b["recon_true"], device)
        e_tgt = _ecapa(enc, b["target"], device)
        e_wrong = _ecapa(enc, b["recon_wrong"], device)
        e_xref = _ecapa(enc, b["xref"], device)
        ident = float(F.cosine_similarity(e_true, e_tgt, dim=-1))
        conv = float(F.cosine_similarity(e_wrong, e_xref, dim=-1))
        rows.append((k, int(b["spk"]), b["g"], ident, conv))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Mel-domain (vocoder-free) ECAPA speaker-similarity")
    ap.add_argument("--dir", action="append", required=True, metavar="name=render_dir",
                    help="Render dir with a mels/ subdir (from --save_render_mels). Repeatable.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--spk", type=int, default=None, help="If set, also print only rows for this speaker id.")
    args = ap.parse_args()

    enc = SpeakerEncoderWrapper(encoder_type="ecapa_tdnn", device=args.device)
    summary = []
    for spec in args.dir:
        name, d = spec.split("=", 1)
        rows = score_dir(enc, d, args.device)
        if not rows:
            print(f"\n=== {name} ===  ({d})  NO mels/ found (render with --save_render_mels)")
            continue
        idents = [r[3] for r in rows]
        convs = [r[4] for r in rows]
        print(f"\n=== {name} ===  ({d})   mel-domain, n={len(rows)}")
        print(f"  {'sample':>7} {'spk':>7} {'g':>2}  {'identity':>9} {'convert':>8}")
        for k, spk, g, ident, conv in rows:
            if args.spk is not None and spk != args.spk:
                continue
            flag = f"  <-- spk{spk}" if (args.spk is not None and spk == args.spk) else ""
            print(f"  {('s'+str(k)):>7} {spk:>7} {g:>2}  {ident:>9.3f} {conv:>8.3f}{flag}")
        print(f"  MEAN identity={np.mean(idents):.3f}  convert={np.mean(convs):.3f}")
        summary.append((name, float(np.mean(idents)), float(np.mean(convs))))

    print("\n=== SUMMARY (mel-domain mean) ===")
    print(f"  {'variant':>18} {'identity':>9} {'convert':>8}")
    for name, i, c in summary:
        print(f"  {name:>18} {i:>9.3f} {c:>8.3f}")


if __name__ == "__main__":
    main()
