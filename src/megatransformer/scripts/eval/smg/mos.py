"""Perceptual naturalness of SMG reconstructions via UTMOS (a neural MOS predictor).

GV/over-smoothing misses fidelity-driven "roboticness" (early recons can have full
variance but wrong harmonic placement). UTMOS is trained on human MOS ratings, so it
tracks perceived naturalness directly. This vocodes the SMG recon AND the GT mel
(copy-synthesis) through the same HiFi-GAN and scores both with UTMOS:

  mos_recon   UTMOS of the SMG's true-emb reconstruction
  mos_gt_voc  UTMOS of the GT mel vocoded (the vocoder CEILING — best achievable here)
  mos_gap     mos_gt_voc - mos_recon (how much naturalness the SMG itself costs)

Vocoding through the same vocoder for both isolates the SMG's contribution from the
vocoder's. UTMOS model is torch.hub 'tarepan/SpeechMOS' (utmos22_strong), 16 kHz, MOS ~1-5.
"""
import argparse
import numpy as np
import torch

from megatransformer.model.smg.smg import SMG
from megatransformer.utils.model_loading_utils import load_model, load_vocoder
from megatransformer.utils.audio_utils import SharedWindowBuffer
from megatransformer.utils import visualization
from megatransformer.scripts.data.voice.dataset import VoiceShardedDataset


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="SMG perceptual naturalness via UTMOS")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="medium_decoder_only_1d_3x")
    ap.add_argument("--sive_encoder_dim", type=int, default=256)
    ap.add_argument("--cache_dir", default="./cached_datasets/smg_libritts_r_clean_stdhinge11-300k_val")
    ap.add_argument("--vocoder_config", default="hifigan")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = args.device
    model = load_model(SMG, args.config, checkpoint_path=args.checkpoint,
                       overrides={"sive_encoder_dim": args.sive_encoder_dim}).to(dev).eval()
    vocoder = load_vocoder(None, args.vocoder_config, SharedWindowBuffer()).to(dev).eval()
    utmos = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong", trust_repo=True).to(dev).eval()
    ds = VoiceShardedDataset(args.cache_dir,
                             columns=["features", "mel_specs", "speaker_embeddings", "speaker_ids", "f0", "vuv"])
    idxs = sorted(np.random.choice(len(ds), size=min(args.n, len(ds)), replace=False).tolist())

    mos_r, mos_g = [], []
    for i in idxs:
        s = ds[i]
        T = int(s["mel_length"])
        if T < 8:
            continue
        feat = s["features"].float().unsqueeze(0).to(dev)
        mel = s["mel_spec"].float().to(dev)
        emb = s["speaker_embedding"].float().unsqueeze(0).to(dev)
        recon = model.decode(feat, speaker_embedding=emb, features=feat)[0]
        rec = recon[:, :min(recon.shape[-1], T)].unsqueeze(0)   # [1, 80, T]
        gt = mel[:, :T].unsqueeze(0)
        wr = torch.from_numpy(visualization.render_vocoder_audio(vocoder, rec)).float().unsqueeze(0).to(dev)  # [1, Tw]
        wg = torch.from_numpy(visualization.render_vocoder_audio(vocoder, gt)).float().unsqueeze(0).to(dev)
        mos_r.append(float(utmos(wr, 16000).reshape(-1)[0]))
        mos_g.append(float(utmos(wg, 16000).reshape(-1)[0]))

    mos_r, mos_g = np.array(mos_r), np.array(mos_g)
    print(f"\n=== SMG UTMOS naturalness @ {args.checkpoint} (n={len(mos_r)}) ===")
    print(f"  mos_recon    = {mos_r.mean():.3f} ± {mos_r.std():.3f}   (SMG true-emb recon, vocoded)")
    print(f"  mos_gt_voc   = {mos_g.mean():.3f} ± {mos_g.std():.3f}   (GT mel vocoded = vocoder ceiling)")
    print(f"  mos_gap      = {(mos_g - mos_r).mean():+.3f}   (ceiling - recon; the SMG's naturalness cost)")


if __name__ == "__main__":
    main()
