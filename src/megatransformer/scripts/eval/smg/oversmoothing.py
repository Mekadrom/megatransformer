"""Objectively quantify how "robotic" / over-smoothed an SMG's reconstructions are.

Robotic quality comes from the L1/MSE objective predicting the conditional MEAN of the
mel, which flattens harmonics and kills frame-to-frame variation. That has a classic
signature: reduced variance. On a val subset, reconstruct each utterance (TRUE emb) and
compare its mel statistics to the GT mel, on VALID frames:

  gv_ratio      mean_d var_t(recon) / mean_d var_t(GT)   over ALL mel bins.
                <1 => over-smoothed/robotic. GAN pushes this back toward 1, so it is the
                ideal before/after metric for the GAN arm.
  hf_gv_ratio   same variance ratio restricted to HIGH mel bins (>= --hf_bin) — HF spectral
                detail is smoothed first, so this is usually lower than gv_ratio.
  hf_mean_gap   mean(recon HF band) - mean(GT HF band), in mel units. Negative => recon is
                quieter/duller in the high band (muffled).

All from mels (no vocoder), so it's cheap and isolates the SMG's contribution.
"""
import argparse
import numpy as np
import torch

from megatransformer.model.smg.smg import SMG
from megatransformer.utils.model_loading_utils import load_model
from megatransformer.scripts.data.voice.dataset import VoiceShardedDataset


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="SMG over-smoothing / naturalness (GV + HF) metric")
    ap.add_argument("--checkpoint", required=True, help="SMG checkpoint dir")
    ap.add_argument("--config", default="medium_decoder_only_1d_3x")
    ap.add_argument("--sive_encoder_dim", type=int, default=256)
    ap.add_argument("--cache_dir", default="./cached_datasets/smg_libritts_r_clean_stdhinge11-300k_val")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--hf_bin", type=int, default=53, help="First mel bin of the HF band (of 80; ~3.3kHz)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--wrong_emb", action="store_true",
                    help="Also compute GV on each utterance decoded with a DIFFERENT speaker's "
                         "embedding (the conversion regime). gv_ratio_wrong below gv_ratio => the "
                         "conversion over-smooths more than same-speaker recon (GAN's target). "
                         "Collapsed baseline => ~= gv_ratio (wrong-emb output == source).")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = load_model(SMG, args.config, checkpoint_path=args.checkpoint,
                       overrides={"sive_encoder_dim": args.sive_encoder_dim}).to(args.device).eval()
    ds = VoiceShardedDataset(args.cache_dir,
                             columns=["features", "mel_specs", "speaker_embeddings", "speaker_ids", "f0", "vuv"])
    idxs = sorted(np.random.choice(len(ds), size=min(args.n, len(ds)), replace=False).tolist())
    samples = [ds[i] for i in idxs]
    spks = [int(s["speaker_id"]) for s in samples]
    embs = [s["speaker_embedding"].float() for s in samples]
    rng = np.random.RandomState(args.seed)

    gv_r, gv_g, hfgv_r, hfgv_g, hf_r, hf_g = [], [], [], [], [], []
    gvw_r, gvw_g = [], []
    hb = args.hf_bin
    for k, s in enumerate(samples):
        feat = s["features"].float().unsqueeze(0).to(args.device)
        mel = s["mel_spec"].float().to(args.device)
        emb = embs[k].unsqueeze(0).to(args.device)
        rec = model.decode(feat, speaker_embedding=emb, features=feat)[0]     # [80, To]
        T = min(rec.shape[-1], int(s["mel_length"]))
        rec, gt = rec[:, :T], mel[:, :T]                                       # [80, T]
        if T < 4:
            continue
        vr = rec.var(dim=1); vg = gt.var(dim=1)                               # per-mel-dim var over time
        gv_r.append(vr.mean().item());  gv_g.append(vg.mean().item())
        hfgv_r.append(vr[hb:].mean().item()); hfgv_g.append(vg[hb:].mean().item())
        hf_r.append(rec[hb:].mean().item()); hf_g.append(gt[hb:].mean().item())
        if args.wrong_emb:
            # Variance of the DIFFERENT-speaker conversion vs natural (source GT)
            # variance — the over-smoothing gauge on the extrapolative regime.
            cand = [m for m in range(len(samples)) if spks[m] != spks[k]]
            if cand:
                emb_w = embs[int(rng.choice(cand))].unsqueeze(0).to(args.device)
                rw = model.decode(feat, speaker_embedding=emb_w, features=feat)[0][:, :T]
                gvw_r.append(rw.var(dim=1).mean().item()); gvw_g.append(vg.mean().item())

    gv_r, gv_g, hfgv_r, hfgv_g, hf_r, hf_g = map(np.array, (gv_r, gv_g, hfgv_r, hfgv_g, hf_r, hf_g))
    print(f"\n=== SMG over-smoothing @ {args.checkpoint} (n={len(gv_r)}, true-emb recon vs GT) ===")
    print(f"  gv_ratio     = {gv_r.mean()/gv_g.mean():.3f}   (recon var / GT var, all bins; <1 => over-smoothed/robotic, GAN -> 1)")
    print(f"  hf_gv_ratio  = {hfgv_r.mean()/hfgv_g.mean():.3f}   (variance ratio, HF bins >= {hb}; HF detail loss)")
    print(f"  hf_mean_gap  = {hf_r.mean()-hf_g.mean():+.3f}   (recon HF mean - GT HF mean, mel units; <0 => duller/muffled HF)")
    if args.wrong_emb and gvw_r:
        gvw_r, gvw_g = np.array(gvw_r), np.array(gvw_g)
        print(f"  gv_ratio_wrong = {gvw_r.mean()/gvw_g.mean():.3f}   (n={len(gvw_r)}; DIFFERENT-speaker conversion var / GT var; "
              f"below gv_ratio => conversion over-smooths more than same-speaker recon => GAN's target)")


if __name__ == "__main__":
    main()
