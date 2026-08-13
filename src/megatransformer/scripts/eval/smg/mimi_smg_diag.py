"""One-pass diagnostics for the Mimi/Vocos/15-2 SMG (the SIVE-era eval scripts can't
load this pipeline: WavLM-768 speaker, integer Mimi unit-ids, Vocos 24kHz, 15/2 decoder).

Per val clip: decode with the TRUE embedding (recon) and a DIFFERENT-speaker embedding
(swap), then report the standard sweep adapted to this pipeline:
  embedding-control : l1_true, l1_wrong, disentangle, output_diff, rel_influence
  over-smoothing    : gv_ratio (true), gv_ratio_wrong  (var(recon)/var(GT), valid frames)
  MOS (Vocos->UTMOS): mos_recon, mos_wrong, mos_gt_voc (vocoder ceiling), mos_gap, conv_gap
  swap-content-drift: mimi_content CE on the swap (the A-vs-C decider -- does the
                      content-cycle keep conversion words? lower = better content preserved)

UTMOS is 16kHz-band (blind >8kHz), so absolute MOS undervalues the 24kHz path; it is still
right for A-vs-C comparison (same meter on both). Ear stays the arbiter.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python3 -m megatransformer.scripts.eval.smg.mimi_smg_diag \
    --checkpoint runs/smg/<run>/checkpoint-50000 --n 200 --device cuda
"""
import argparse, glob, os
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="medium_decoder_only_1d_15x2_mimicontour_vocos")
    ap.add_argument("--sive_encoder_dim", type=int, default=256)
    ap.add_argument("--speaker_embedding_dim", type=int, default=768)
    ap.add_argument("--cache_dir", default="cached_datasets/Mekadrom/libritts_r_clean_mimi_wavlm/val")
    ap.add_argument("--codebook", default="cached_datasets/Mekadrom/libritts_r_clean_mimi_wavlm/val/mimi_semantic_codebook.pt")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(a.seed)

    from megatransformer.model.smg.smg import SMG
    from megatransformer.utils import model_loading_utils as mlu
    from megatransformer.utils.codebook import load_f0_stats, normalize_f0
    from megatransformer.utils.vocos_features import load_vocos
    from megatransformer.utils.mimi_content import load_mimi, mimi_content_cycle_loss

    ov = dict(sive_encoder_dim=a.sive_encoder_dim, speaker_embedding_dim=a.speaker_embedding_dim,
              hop_length=256, sample_rate=24000, code_embed_init="learned_random")
    model = mlu.load_model(SMG, a.config, checkpoint_path=a.checkpoint, overrides=ov).to(dev).eval()
    f0_stats = load_f0_stats(a.codebook)
    if f0_stats is None:
        raise SystemExit(f"no per-speaker F0 stats in {a.codebook} (contour mode needs them)")
    vocos = load_vocos(dev)
    mimi = load_mimi(dev)
    utmos = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong", trust_repo=True).to(dev).eval()
    # WavLM speaker encoder for the DIRECT conversion metric: does the swap output's speaker
    # embedding land on the TARGET vs the SOURCE? (mel-L1 disentangle is a poor speaker proxy --
    # identity is a small fraction of mel-L1, dominated by content + recon error.)
    import torchaudio.functional as AF
    from megatransformer.utils.speaker_encoder import SpeakerEncoderWrapper
    spk_enc = SpeakerEncoderWrapper(encoder_type="wavlm", device=dev).eval()
    def spk_emb_of(wav24):  # [1, T] @24k -> WavLM 768 embedding
        w16 = AF.resample(wav24, 24000, 16000)
        return spk_enc(waveform=w16, sample_rate=16000).reshape(-1)

    # gather clips (cap at n)
    clips = []
    for sp in sorted(glob.glob(os.path.join(a.cache_dir, "shard_*.pt"))):
        d = torch.load(sp, map_location="cpu")
        for i in range(d["mel_specs"].shape[0]):
            clips.append((d["unit_ids"][i], d["speaker_embeddings"][i], d["mel_specs"][i],
                          int(d["mel_lengths"][i]), int(d["feature_lengths"][i]),
                          d["f0"][i], int(d["speaker_ids"][i])))
            if len(clips) >= a.n:
                break
        if len(clips) >= a.n:
            break

    def voc_utmos(mel_1):  # mel [100,L] -> UTMOS scalar
        wav = vocos.decode(mel_1.unsqueeze(0))
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        return float(utmos(wav, 24000).reshape(-1)[0]), wav

    acc = {k: [] for k in ["l1_true", "l1_wrong", "output_diff", "gv", "gv_wrong",
                           "mos_recon", "mos_wrong", "mos_gt", "swap_ce",
                           "conv_target", "conv_source", "recon_source"]}
    cos = torch.nn.functional.cosine_similarity
    with torch.no_grad():
        for idx, (uid, spk, gtmel, mlen, flen, f0, sid) in enumerate(clips):
            feats = uid[:flen].unsqueeze(0).to(dev)
            spk_t = spk.unsqueeze(0).to(dev)
            # different-speaker embedding: pick a clip with a different speaker id
            j = (idx + 1 + int(torch.randint(0, max(1, len(clips) - 1), (1,)))) % len(clips)
            tries = 0
            while clips[j][6] == sid and tries < 20:
                j = (j + 1) % len(clips); tries += 1
            spk_wrong = clips[j][1].unsqueeze(0).to(dev)
            contour = normalize_f0(f0.float(), sid, f0_stats)[:mlen].unsqueeze(0).to(dev)
            gt = gtmel[:, :mlen].to(dev)

            recon = model.decode(feats, speaker_embedding=spk_t, features=feats, f0_contour=contour)
            swap = model.decode(feats, speaker_embedding=spk_wrong, features=feats, f0_contour=contour)
            if recon.dim() == 4: recon = recon.squeeze(1)
            if swap.dim() == 4: swap = swap.squeeze(1)
            L = min(recon.shape[-1], swap.shape[-1], mlen)
            r, w, g = recon[0, :, :L], swap[0, :, :L], gt[:, :L]

            acc["l1_true"].append((r - g).abs().mean().item())
            acc["l1_wrong"].append((w - g).abs().mean().item())
            acc["output_diff"].append((r - w).abs().mean().item())
            gvt = (r.var(dim=1) / g.var(dim=1).clamp(min=1e-6)).mean().item()
            gvw = (w.var(dim=1) / g.var(dim=1).clamp(min=1e-6)).mean().item()
            acc["gv"].append(gvt); acc["gv_wrong"].append(gvw)

            mr, r_wav = voc_utmos(r)
            mw, sw_wav = voc_utmos(w)
            mg, _ = voc_utmos(g)
            acc["mos_recon"].append(mr); acc["mos_wrong"].append(mw); acc["mos_gt"].append(mg)
            # swap-content drift: re-encode the swapped waveform, CE vs the SOURCE units
            ce = mimi_content_cycle_loss(mimi, sw_wav, feats, loss_type="ce", temperature=0.1)
            acc["swap_ce"].append(float(ce))
            # DIRECT conversion metric: swap output's speaker embedding vs target/source.
            src_emb = spk.to(dev); tgt_emb = clips[j][1].to(dev)
            swap_spk = spk_emb_of(sw_wav)
            recon_spk = spk_emb_of(r_wav)
            acc["conv_target"].append(float(cos(swap_spk, tgt_emb, dim=0)))   # want HIGH
            acc["conv_source"].append(float(cos(swap_spk, src_emb, dim=0)))   # want LOW
            acc["recon_source"].append(float(cos(recon_spk, src_emb, dim=0))) # sanity: recon=source, HIGH

    m = {k: float(np.mean(v)) for k, v in acc.items()}
    disentangle = m["l1_wrong"] - m["l1_true"]
    rel_influence = m["output_diff"] / max(m["l1_true"], 1e-6)
    print(f"\n=== {a.checkpoint}  (n={len(clips)}) ===")
    print(f"  embedding-control: l1_true={m['l1_true']:.3f}  l1_wrong={m['l1_wrong']:.3f}  "
          f"disentangle={disentangle:+.3f}  rel_influence={rel_influence:.3f}")
    print(f"  over-smoothing   : gv_ratio={m['gv']:.3f}  gv_ratio_wrong={m['gv_wrong']:.3f}")
    print(f"  MOS (Vocos/UTMOS): mos_recon={m['mos_recon']:.3f}  mos_wrong={m['mos_wrong']:.3f}  "
          f"mos_gt_voc(ceiling)={m['mos_gt']:.3f}")
    print(f"                     mos_gap(ceil-recon)={m['mos_gt']-m['mos_recon']:+.3f}  "
          f"conv_gap(recon-wrong)={m['mos_recon']-m['mos_wrong']:+.3f}")
    print(f"  swap-content CE  : {m['swap_ce']:.3f}   (lower = conversion keeps source content)")
    conv_margin = m["conv_target"] - m["conv_source"]
    print(f"  conversion (WavLM speaker cos):")
    print(f"    swap->target={m['conv_target']:.3f}  swap->source={m['conv_source']:.3f}  "
          f"margin={conv_margin:+.3f}  (>0 => swap lands on TARGET, not source)")
    print(f"    recon->source={m['recon_source']:.3f}  (sanity: recon should match its own speaker)")


if __name__ == "__main__":
    main()
