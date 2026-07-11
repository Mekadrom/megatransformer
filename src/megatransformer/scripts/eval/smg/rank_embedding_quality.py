"""Rank speaker embeddings by the OUTPUT QUALITY they produce from the SMG.

Goal: find "high quality" speaker embeddings — points that decode to the cleanest /
most natural audio (independent of which voice). Decodes a FIXED set of content
utterances with each candidate embedding, vocodes (with the 50->62.5Hz resample when
needed), and scores naturalness with UTMOS (the metric that tracks the ear). Also
reports a mel-domain variance ratio (higher = less over-smoothed) as a resample-free
secondary. Ranks candidates by mean UTMOS so you listen only to the winners; a PCA-grid
input reveals whether the clean region is systematic (a direction) or a lone point.

  python -m megatransformer.scripts.eval.smg.rank_embedding_quality \
    --embeddings_dir eval_outputs/probe_grid \
    --checkpoint runs/smg/smg_libritts_r_1d1x_contentvec_baseline_nogan_0/checkpoint-27000 \
    --config medium_decoder_only_1d_1x --cache_dir ./cached_datasets/smg_libritts_r_clean_contentvec_val \
    --mel_hop_length 320 --n_content 3 --render_topk 3
"""
import argparse, glob, json, os
import numpy as np
import torch

from megatransformer.model.smg.smg import SMG
from megatransformer.utils.model_loading_utils import load_model, load_vocoder
from megatransformer.utils.audio_utils import SharedWindowBuffer
from megatransformer.utils import visualization
from megatransformer.scripts.data.voice.dataset import VoiceShardedDataset


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Rank speaker embeddings by SMG output quality (UTMOS)")
    ap.add_argument("--embeddings_dir", required=True, help="Dir of candidate [dim] .pt embeddings")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="medium_decoder_only_1d_1x")
    ap.add_argument("--sive_encoder_dim", type=int, default=256)
    ap.add_argument("--cache_dir", default="./cached_datasets/smg_libritts_r_clean_contentvec_val")
    ap.add_argument("--vocoder_config", default="hifigan")
    ap.add_argument("--mel_hop_length", type=int, default=320)
    ap.add_argument("--n_content", type=int, default=3, help="Fixed content utterances to average over")
    ap.add_argument("--render_topk", type=int, default=0, help="Render top-K (+bottom-2) to WAVs")
    ap.add_argument("--output_dir", default=None, help="Where to write ranking.json / WAVs (default: embeddings_dir)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    dev = args.device
    out_dir = args.output_dir or args.embeddings_dir
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    model = load_model(SMG, args.config, checkpoint_path=args.checkpoint,
                       overrides={"sive_encoder_dim": args.sive_encoder_dim}).to(dev).eval()
    vocoder = load_vocoder(None, args.vocoder_config, SharedWindowBuffer()).to(dev).eval()
    utmos = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong", trust_repo=True).to(dev).eval()
    voc_hop = getattr(getattr(vocoder, "config", None), "hop_length", args.mel_hop_length)

    ds = VoiceShardedDataset(args.cache_dir, columns=["features", "mel_specs", "speaker_ids"])
    cidx = [i for i in range(len(ds)) if int(ds[i]["mel_length"]) > 150][:200]
    cidx = sorted(np.random.RandomState(args.seed).choice(cidx, min(args.n_content, len(cidx)), replace=False).tolist())
    contents = []
    for i in cidx:
        s = ds[i]; T = int(s["mel_length"])
        contents.append((s["features"].float().unsqueeze(0).to(dev),
                         s["mel_spec"].float()[:, :T].to(dev), T))

    cands = {}
    for p in sorted(glob.glob(os.path.join(args.embeddings_dir, "*.pt"))):
        cands[os.path.splitext(os.path.basename(p))[0]] = torch.load(p, map_location="cpu").float().reshape(1, -1).to(dev)

    def utmos_of(mel_1c):
        wav = torch.from_numpy(visualization.render_vocoder_audio(
            vocoder, mel_1c, mel_hop_length=args.mel_hop_length, vocoder_hop_length=voc_hop)).float().unsqueeze(0).to(dev)
        return float(utmos(wav, 16000).reshape(-1)[0])

    rows = []
    for name, emb in cands.items():
        us, gvs = [], []
        for feat, gt, T in contents:
            mel = model.decode(feat, speaker_embedding=emb, features=feat)[0][:, :T]
            us.append(utmos_of(mel.unsqueeze(0)))
            gvs.append(float((mel.var(1).mean() / (gt.var(1).mean() + 1e-8)).clamp(0, 3)))
        rows.append((name, float(np.mean(us)), float(np.std(us)), float(np.mean(gvs))))
    rows.sort(key=lambda r: -r[1])

    print(f"\n=== embedding quality ranking (n_content={len(contents)}, UTMOS ↑) @ {os.path.basename(args.checkpoint)} ===")
    print(f"{'rank':>4} {'embedding':26s} {'UTMOS':>7s} {'±':>5s} {'var_ratio':>9s}")
    for r, (name, u, sd, gv) in enumerate(rows):
        print(f"{r+1:>4} {name:26s} {u:7.3f} {sd:5.3f} {gv:9.3f}")

    with open(os.path.join(out_dir, "ranking.json"), "w") as f:
        json.dump([{"name": n, "utmos": u, "utmos_std": s, "var_ratio": g} for n, u, s, g in rows], f, indent=2)

    if args.render_topk > 0:
        import torchaudio
        feat, _, T = contents[0]
        picks = rows[:args.render_topk] + rows[-2:]
        rd = os.path.join(out_dir, "ranked_wavs"); os.makedirs(rd, exist_ok=True)
        for rank_i, (name, u, sd, gv) in enumerate(picks):
            mel = model.decode(feat, speaker_embedding=cands[name], features=feat)[0][:, :T]
            wav = visualization.render_vocoder_audio(vocoder, mel.unsqueeze(0),
                                                     mel_hop_length=args.mel_hop_length, vocoder_hop_length=voc_hop)
            tag = f"top{rank_i+1}" if rank_i < args.render_topk else "bottom"
            torchaudio.save(os.path.join(rd, f"{tag}_{name}_utmos{u:.2f}.wav"),
                            torch.from_numpy(wav).float().reshape(1, -1), 16000)
        print(f"\nRendered {len(picks)} WAVs -> {rd}")


if __name__ == "__main__":
    main()
