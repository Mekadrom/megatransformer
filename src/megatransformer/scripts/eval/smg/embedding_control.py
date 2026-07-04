"""Measure how much a trained SMG's output is controlled by the speaker embedding
(the cross-speaker / voice-conversion "collapse" metric).

For a val subset, decode each utterance twice — with its TRUE embedding and with a
DIFFERENT-speaker embedding — and report, on VALID (unpadded) frames only:

  l1_true       L1(recon_true, GT mel)        normal recon quality
  l1_wrong      L1(recon_wrong, GT mel)        recon using a wrong speaker's emb
  disentangle   l1_wrong - l1_true             ~0  => wrong emb reconstructs the source
                                               just as well => embedding IGNORED
  output_diff   L1(recon_true, recon_wrong)    how much swapping the emb changes output
  rel_influence output_diff / l1_true          <~0.3 => emb barely matters (collapse);
                                               healthy conversion wants >~0.5

Swapping the embedding changes BOTH the predicted F0 and the FiLM conditioning (decode()
predicts F0 from the given emb), so this is the full conversion signal. Recon-only SMG
training gives no pressure for the embedding to control speaker identity (features + emb
are always the same speaker at train time), so this metric collapses as the decoder
converges and learns to read speaker identity straight from the (leaky) SIVE features.
The FiLM contrastive loss is the intended fix; watch disentangle/rel_influence rise when
it is on.
"""
import argparse
import numpy as np
import torch

from megatransformer.model.smg.smg import SMG
from megatransformer.utils.model_loading_utils import load_model
from megatransformer.scripts.data.voice.dataset import VoiceShardedDataset


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="SMG speaker-embedding control / cross-speaker-collapse metric")
    ap.add_argument("--checkpoint", required=True, help="SMG checkpoint dir")
    ap.add_argument("--config", default="medium_decoder_only_1d_3x", help="SMG config (must match the checkpoint)")
    ap.add_argument("--sive_encoder_dim", type=int, default=256, help="SIVE feature width the SMG was trained with")
    ap.add_argument("--cache_dir", default="./cached_datasets/smg_libritts_r_clean_stdhinge11-300k_val",
                    help="Shard dir with features/mel_specs/speaker_embeddings/speaker_ids")
    ap.add_argument("--n", type=int, default=400, help="Number of val utterances to sample")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = load_model(SMG, args.config, checkpoint_path=args.checkpoint,
                       overrides={"sive_encoder_dim": args.sive_encoder_dim}).to(args.device).eval()
    ds = VoiceShardedDataset(args.cache_dir,
                             columns=["features", "mel_specs", "speaker_embeddings", "speaker_ids", "f0", "vuv"])
    idxs = sorted(np.random.choice(len(ds), size=min(args.n, len(ds)), replace=False).tolist())
    samples = [ds[i] for i in idxs]
    spks = [int(s["speaker_id"]) for s in samples]
    embs = [s["speaker_embedding"].float() for s in samples]

    l1_t, l1_w, odiff = [], [], []
    for k, s in enumerate(samples):
        feat = s["features"].float().unsqueeze(0).to(args.device)   # [1, D, T']
        mel = s["mel_spec"].float().to(args.device)                 # [80, T]
        emb_true = embs[k].unsqueeze(0).to(args.device)
        j = next((m for m in range(len(samples)) if spks[m] != spks[k]), k)  # a different speaker
        emb_wrong = embs[j].unsqueeze(0).to(args.device)
        rt = model.decode(feat, speaker_embedding=emb_true, features=feat)[0]
        rw = model.decode(feat, speaker_embedding=emb_wrong, features=feat)[0]
        T = min(rt.shape[-1], rw.shape[-1], int(s["mel_length"]))    # valid frames only
        rt, rw, mg = rt[..., :T], rw[..., :T], mel[..., :T]
        l1_t.append((rt - mg).abs().mean().item())
        l1_w.append((rw - mg).abs().mean().item())
        odiff.append((rt - rw).abs().mean().item())

    l1_t, l1_w, odiff = map(np.array, (l1_t, l1_w, odiff))
    print(f"\n=== SMG embedding-control @ {args.checkpoint} (n={len(l1_t)}, different-speaker swaps) ===")
    print(f"  l1_true      = {l1_t.mean():.4f}")
    print(f"  l1_wrong     = {l1_w.mean():.4f}")
    print(f"  disentangle  = {(l1_w - l1_t).mean():+.4f}   (~0 => wrong emb reconstructs the source => IGNORED)")
    print(f"  output_diff  = {odiff.mean():.4f}")
    print(f"  rel_influence= {(odiff / (l1_t + 1e-8)).mean():.3f}   (output_diff / l1_true; <~0.3 => emb barely matters)")


if __name__ == "__main__":
    main()
