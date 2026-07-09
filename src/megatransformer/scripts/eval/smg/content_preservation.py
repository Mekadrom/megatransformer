"""Separate the two axes embedding_control's `disentangle` confounds: when the SMG
decodes with a DIFFERENT speaker's embedding, does it (a) actually convert the SPEAKER,
and (b) PRESERVE the content? `disentangle = l1_wrong - l1_true` rises for BOTH real
conversion and content-breakage, so it can't tell a clean conversion from a garbled one.

This decodes each val utterance with its TRUE emb and a WRONG emb, then measures:

  CONTENT (frozen SIVE on the decoded mel; SIVE is speaker-clean):
    content_true   L1(SIVE(recon_true),  source_features)   same-emb content error (ref)
    content_wrong  L1(SIVE(recon_wrong), source_features)   converted-output content error
    content_drift  content_wrong - content_true             SWAP-INDUCED content damage
                                                             (~0 = content preserved; high = garbled)

  SPEAKER (ECAPA on the decoded mel):
    convert   cos(ECAPA(recon_wrong), target_emb)   did it BECOME the target  (higher better)
    residual  cos(ECAPA(recon_wrong), source_emb)   does it still sound SOURCE (lower better)
    true_id   cos(ECAPA(recon_true),  source_emb)   same-emb identity faithfulness (ref)

A healthy converter wants convert high, residual low, AND content_drift ~0. The failure
this tool catches (content-blind identity loss): convert high + residual low BUT
content_drift large = right speaker, wrong words.
"""
import argparse
import numpy as np
import torch

from megatransformer.model.smg.smg import SMG
from megatransformer.model.voice.sive.sive import SpeakerInvariantVoiceEncoder
from megatransformer.utils.model_loading_utils import load_model
from megatransformer.utils.speaker_encoder import get_speaker_encoder
from megatransformer.scripts.data.voice.dataset import VoiceShardedDataset


def _sive_feats(sive, mel, lengths, layer):
    """mel [B, n_mels, T] -> SIVE features [B, T', C] at `layer` (-1 = final)."""
    out = sive(mel, lengths=lengths, return_all_hiddens=(layer != -1))
    feats = out["features"] if layer == -1 else out["all_hiddens"][layer]
    return feats, out["feature_lengths"]


def _masked_l1(pred, target, lengths):
    """pred/target [B, T', C]; mean L1 over valid frames (lengths in SIVE frames)."""
    min_t = min(pred.shape[1], target.shape[1])
    pred, target = pred[:, :min_t], target[:, :min_t]
    if lengths is not None:
        m = (torch.arange(min_t, device=pred.device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)
        return ((pred - target).abs() * m).sum() / (m.sum() * pred.shape[-1]).clamp(min=1)
    return (pred - target).abs().mean()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="SMG content-preservation vs speaker-conversion on the swap")
    ap.add_argument("--checkpoint", required=True, help="SMG checkpoint dir")
    ap.add_argument("--config", default="medium_decoder_only_1d_3x")
    ap.add_argument("--sive_encoder_dim", type=int, default=256)
    ap.add_argument("--cache_dir", default="./cached_datasets/smg_libritts_r_clean_stdhinge11-300k_val")
    ap.add_argument("--sive_checkpoint", required=True, help="Frozen SIVE checkpoint (the SMG's content encoder)")
    ap.add_argument("--sive_config", default="small_deep_3xdownsample_conv2d_attentive")
    ap.add_argument("--sive_num_speakers", type=int, default=3610, help="Speaker-head size the SIVE checkpoint was trained with (must match to load)")
    ap.add_argument("--sive_layer", type=int, default=10, help="SIVE layer for the content probe (match the SMG tap; 10)")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = args.device

    model = load_model(SMG, args.config, checkpoint_path=args.checkpoint,
                       overrides={"sive_encoder_dim": args.sive_encoder_dim}).to(dev).eval()
    sive = load_model(SpeakerInvariantVoiceEncoder, args.sive_config,
                      checkpoint_path=args.sive_checkpoint, device=dev,
                      overrides={"num_speakers": args.sive_num_speakers}).eval()
    for p in sive.parameters():
        p.requires_grad = False
    ecapa = get_speaker_encoder(encoder_type="ecapa_tdnn", device=dev)

    ds = VoiceShardedDataset(args.cache_dir,
                             columns=["features", "mel_specs", "speaker_embeddings", "speaker_ids", "f0", "vuv"])
    idxs = sorted(np.random.choice(len(ds), size=min(args.n, len(ds)), replace=False).tolist())
    samples = [ds[i] for i in idxs]
    spks = [int(s["speaker_id"]) for s in samples]
    embs = [s["speaker_embedding"].float() for s in samples]

    c_true, c_wrong, convert, residual, true_id = [], [], [], [], []
    for k, s in enumerate(samples):
        feat = s["features"].float().unsqueeze(0).to(dev)           # [1, C, T'] source content
        emb_true = embs[k].unsqueeze(0).to(dev)
        j = next((m for m in range(len(samples)) if spks[m] != spks[k]), k)
        emb_wrong = embs[j].unsqueeze(0).to(dev)

        rt = model.decode(feat, speaker_embedding=emb_true, features=feat)
        rw = model.decode(feat, speaker_embedding=emb_wrong, features=feat)
        if rt.dim() == 4:
            rt, rw = rt.squeeze(1), rw.squeeze(1)                   # [1, n_mels, T]
        T = min(rt.shape[-1], rw.shape[-1], int(s["mel_length"]))
        rt, rw = rt[..., :T], rw[..., :T]
        mel_len = torch.tensor([T], device=dev)

        # content: SIVE on decoded mels vs the SOURCE features
        st, lt = _sive_feats(sive, rt, mel_len, args.sive_layer)
        sw, lw = _sive_feats(sive, rw, mel_len, args.sive_layer)
        tgt = feat.permute(0, 2, 1)                                 # [1, T', C]
        c_true.append(_masked_l1(st, tgt, lt).item())
        c_wrong.append(_masked_l1(sw, tgt, lw).item())

        # speaker: ECAPA on decoded mels vs target / source embeddings
        et = ecapa(mel_spec=rt).float()
        ew = ecapa(mel_spec=rw).float()
        convert.append(torch.cosine_similarity(ew, emb_wrong, dim=-1).item())
        residual.append(torch.cosine_similarity(ew, emb_true, dim=-1).item())
        true_id.append(torch.cosine_similarity(et, emb_true, dim=-1).item())

    c_true, c_wrong = np.array(c_true), np.array(c_wrong)
    convert, residual, true_id = map(np.array, (convert, residual, true_id))
    print(f"\n=== SMG content-preservation vs conversion @ {args.checkpoint} (n={len(c_true)}, SIVE layer {args.sive_layer}) ===")
    print("  CONTENT (SIVE on decoded mel vs source features; speaker-clean):")
    print(f"    content_true   = {c_true.mean():.4f}   (same-emb content error, ref)")
    print(f"    content_wrong  = {c_wrong.mean():.4f}   (converted-output content error)")
    print(f"    content_drift  = {(c_wrong - c_true).mean():+.4f}   <- SWAP-induced content damage (~0 good; high = garbled)")
    print("  SPEAKER (ECAPA on decoded mel):")
    print(f"    convert (->target) = {convert.mean():.3f}   (higher = became the target)")
    print(f"    residual(->source) = {residual.mean():.3f}   (lower = less source identity left)")
    print(f"    true_id (->source) = {true_id.mean():.3f}   (same-emb identity faithfulness, ref)")
    good_spk = convert.mean() - residual.mean()
    print(f"  conversion margin (convert - residual) = {good_spk:+.3f}   (higher = cleaner speaker swap)")
    print("  READ: want conversion margin HIGH and content_drift ~0. High margin + high")
    print("        content_drift = right speaker, wrong words (the content-blind-identity-loss failure).")


if __name__ == "__main__":
    main()
