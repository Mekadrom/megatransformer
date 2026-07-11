"""Track SMG roboticness across checkpoints — the GAN before/after curve.

Runs the same metrics as oversmoothing.py (GV / HF-GV, mel domain) and mos.py (UTMOS,
vocoded) on a BASE checkpoint plus successive checkpoints, on a FIXED seeded content
subset (so numbers are comparable across steps), and emits a trajectory table + CSV +
figure. Watch gv_ratio / hf_gv_ratio climb toward 1.0 and mos_recon / mos_wrong rise as
the GAN de-robotifies the decoder.

Columns per checkpoint (means over --n utterances, valid frames):
  gv_ratio        var(recon_true)/var(GT), all mel bins   (↑ toward 1 = less over-smoothed)
  hf_gv_ratio     same, HF bins >= --hf_bin               (HF detail; the robotic signature)
  gv_ratio_wrong  var(recon_wrong_emb)/var(GT)            (the conversion regime)
  mos_recon       UTMOS of vocoded true-emb recon         (perceptual naturalness)
  mos_wrong       UTMOS of vocoded wrong-emb (conversion)
  mos_gap         mos_gt_voc - mos_recon                  (SMG cost vs the vocoder ceiling)
  conv_gap        mos_recon - mos_wrong                   (conversion naturalness cost)

Example (base nogan-50k vs the GAN run's checkpoints, mapped onto the offset step axis):
  python -m megatransformer.scripts.eval.smg.roboticness_over_time \
    --run runs/smg/smg_libritts_r_1d1x_contentvec_baseline_gan_0 --start_step 50000 --stride 2 \
    --base_checkpoint runs/smg/smg_libritts_r_1d1x_contentvec_baseline_nogan_0/checkpoint-50000 --base_step 50000 \
    --config medium_decoder_only_1d_1x --sive_encoder_dim 256 \
    --cache_dir ./cached_datasets/smg_libritts_r_clean_contentvec_val --mel_hop_length 320 --n 48
"""
import argparse, csv, glob, os, re
import numpy as np
import torch

from megatransformer.model.smg.smg import SMG
from megatransformer.utils.model_loading_utils import load_model, load_vocoder
from megatransformer.utils.audio_utils import SharedWindowBuffer
from megatransformer.utils import visualization
from megatransformer.scripts.data.voice.dataset import VoiceShardedDataset


def discover(run_dir, steps, stride):
    ck = {}
    for p in glob.glob(os.path.join(run_dir, "checkpoint-*")):
        m = re.search(r"checkpoint-(\d+)$", p)
        if m:
            ck[int(m.group(1))] = p
    order = sorted(ck)
    if steps:
        order = [s for s in steps if s in ck]
    elif stride > 1:
        order = order[::stride]
    return [(s, ck[s]) for s in order]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="SMG roboticness (GV+UTMOS) across checkpoints")
    ap.add_argument("--run", help="GAN run dir to discover checkpoint-* in")
    ap.add_argument("--checkpoint", nargs="+", default=[], metavar="STEP=PATH",
                    help="Explicit checkpoints as step=path (alternative/addition to --run)")
    ap.add_argument("--steps", help="Comma list of checkpoint steps to eval from --run (default: all)")
    ap.add_argument("--stride", type=int, default=1, help="Subsample discovered checkpoints by this stride")
    ap.add_argument("--start_step", type=int, default=0, help="Offset added to --run checkpoint numbers for the x-axis (e.g. 50000 for a --start_step run)")
    ap.add_argument("--base_checkpoint", help="A 'before' checkpoint (e.g. the nogan base)")
    ap.add_argument("--base_step", type=int, default=None, help="X-axis step for --base_checkpoint (default: parse from path)")
    ap.add_argument("--config", default="medium_decoder_only_1d_1x")
    ap.add_argument("--sive_encoder_dim", type=int, default=256)
    ap.add_argument("--cache_dir", default="./cached_datasets/smg_libritts_r_clean_contentvec_val")
    ap.add_argument("--vocoder_config", default="hifigan")
    ap.add_argument("--mel_hop_length", type=int, default=320)
    ap.add_argument("--hf_bin", type=int, default=53)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--output_dir", default="eval_outputs/roboticness_over_time_0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dev = args.device
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # Build the checkpoint list: base first, then run/explicit (step, path, label)
    entries = []
    if args.base_checkpoint:
        bs = args.base_step
        if bs is None:
            m = re.search(r"checkpoint-(\d+)", args.base_checkpoint); bs = int(m.group(1)) if m else 0
        entries.append((bs, args.base_checkpoint, "base"))
    if args.run:
        req = [int(x) for x in args.steps.split(",")] if args.steps else None
        for s, p in discover(args.run, req, args.stride):
            entries.append((args.start_step + s, p, ""))
    for kv in args.checkpoint:
        name, _, path = kv.partition("=")
        m = re.search(r"checkpoint-(\d+)", path)
        entries.append((int(name) if name.isdigit() else (int(m.group(1)) if m else 0), path, name if not name.isdigit() else ""))
    entries.sort(key=lambda e: e[0])
    if not entries:
        raise SystemExit("no checkpoints — pass --run and/or --base_checkpoint/--checkpoint")
    print(f"Evaluating {len(entries)} checkpoints: {[e[0] for e in entries]}")

    vocoder = load_vocoder(None, args.vocoder_config, SharedWindowBuffer()).to(dev).eval()
    utmos = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong", trust_repo=True).to(dev).eval()
    voc_hop = getattr(getattr(vocoder, "config", None), "hop_length", args.mel_hop_length)

    ds = VoiceShardedDataset(args.cache_dir, columns=["features", "mel_specs", "speaker_embeddings", "speaker_ids", "f0", "vuv"])
    good = [i for i in range(len(ds)) if int(ds[i]["mel_length"]) > 150]
    idxs = sorted(np.random.RandomState(args.seed).choice(good, min(args.n, len(good)), replace=False).tolist())
    samples = [ds[i] for i in idxs]
    spks = [int(s["speaker_id"]) for s in samples]
    embs = [s["speaker_embedding"].float() for s in samples]
    rng = np.random.RandomState(args.seed)
    wrong_of = []  # fixed wrong-emb partner per sample (comparable across checkpoints)
    for k in range(len(samples)):
        cand = [m for m in range(len(samples)) if spks[m] != spks[k]]
        wrong_of.append(int(rng.choice(cand)) if cand else k)

    def utmos_of(mel_1c):
        wav = torch.from_numpy(visualization.render_vocoder_audio(
            vocoder, mel_1c, mel_hop_length=args.mel_hop_length, vocoder_hop_length=voc_hop)).float().unsqueeze(0).to(dev)
        return float(utmos(wav, 16000).reshape(-1)[0])

    # Precompute GT (checkpoint-independent): per-bin var + vocoded UTMOS ceiling
    gt_var, gt_mos = [], []
    for s in samples:
        T = int(s["mel_length"]); gt = s["mel_spec"].float()[:, :T].to(dev)
        gt_var.append(gt.var(dim=1) + 1e-8)
        gt_mos.append(utmos_of(gt.unsqueeze(0)))
    mos_gt_voc = float(np.mean(gt_mos))

    rows = []
    for step, path, label in entries:
        model = load_model(SMG, args.config, checkpoint_path=path, overrides={"sive_encoder_dim": args.sive_encoder_dim}).to(dev).eval()
        gv, hfgv, gvw, mr, mw = [], [], [], [], []
        for k, s in enumerate(samples):
            T = int(s["mel_length"])
            feat = s["features"].float().unsqueeze(0).to(dev)
            emb_t = embs[k].unsqueeze(0).to(dev)
            emb_w = embs[wrong_of[k]].unsqueeze(0).to(dev)
            mel_t = model.decode(feat, speaker_embedding=emb_t, features=feat)[0][:, :T]
            mel_w = model.decode(feat, speaker_embedding=emb_w, features=feat)[0][:, :T]
            rv = (mel_t.var(dim=1) / gt_var[k]).clamp(0, 3)
            gv.append(float(rv.mean())); hfgv.append(float(rv[args.hf_bin:].mean()))
            gvw.append(float((mel_w.var(dim=1) / gt_var[k]).clamp(0, 3).mean()))
            mr.append(utmos_of(mel_t.unsqueeze(0))); mw.append(utmos_of(mel_w.unsqueeze(0)))
        row = dict(step=step, label=label, gv_ratio=np.mean(gv), hf_gv_ratio=np.mean(hfgv),
                   gv_ratio_wrong=np.mean(gvw), mos_recon=np.mean(mr), mos_wrong=np.mean(mw),
                   mos_gt_voc=mos_gt_voc, mos_gap=mos_gt_voc - np.mean(mr), conv_gap=np.mean(mr) - np.mean(mw))
        rows.append(row)
        del model; torch.cuda.empty_cache()
        print(f"  step {step:>7}{' ('+label+')' if label else '':10} gv={row['gv_ratio']:.3f} hf_gv={row['hf_gv_ratio']:.3f} "
              f"gv_wrong={row['gv_ratio_wrong']:.3f} mos_recon={row['mos_recon']:.3f} mos_wrong={row['mos_wrong']:.3f} conv_gap={row['conv_gap']:+.3f}")

    # CSV
    csv_path = os.path.join(args.output_dir, "roboticness_over_time.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [r["step"] for r in rows]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
        a1.axhline(1.0, color="gray", ls=":", lw=1)
        for key, lab in [("gv_ratio", "gv_ratio"), ("hf_gv_ratio", "hf_gv"), ("gv_ratio_wrong", "gv_wrong")]:
            a1.plot(xs, [r[key] for r in rows], marker="o", ms=3, label=lab)
        a1.set_title("Variance ratio (↑ toward 1 = less over-smoothed)"); a1.set_xlabel("step"); a1.legend(); a1.grid(alpha=.3)
        a2.axhline(mos_gt_voc, color="gray", ls=":", lw=1, label=f"GT ceiling {mos_gt_voc:.2f}")
        for key, lab in [("mos_recon", "mos_recon"), ("mos_wrong", "mos_wrong (conv)")]:
            a2.plot(xs, [r[key] for r in rows], marker="o", ms=3, label=lab)
        a2.set_title("UTMOS naturalness (↑ = better)"); a2.set_xlabel("step"); a2.legend(); a2.grid(alpha=.3)
        fig.tight_layout()
        fig_path = os.path.join(args.output_dir, "roboticness_over_time.png")
        fig.savefig(fig_path, dpi=110); plt.close(fig)
        print(f"\nFigure -> {fig_path}")
    except Exception as e:
        print(f"[figure skipped: {e}]")
    print(f"CSV -> {csv_path}")


if __name__ == "__main__":
    main()
