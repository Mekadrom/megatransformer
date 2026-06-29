"""SIVE synthesis-usability probe: how decodable are SIVE features back to mel?

CTC measures phonetic content but is blind to prosody and reconstructability —
the things SMG actually needs. This probe trains a small SMG-shaped decoder
(FiLM-conditioned 1D conv, ~0.5-1.5M params) to reconstruct the mel from frozen
SIVE features + the stored ECAPA speaker embedding, on a fixed budget (not to
convergence) so the number is comparable across checkpoints. It reports:

  - recon L1 with the TRUE embedding  -> content sufficiency (lower = better;
    a better-than-CTC usability signal).
  - recon L1 with a SHUFFLED embedding vs the original mel -> disentanglement:
    clean features move the output to the wrong speaker (L1 HIGH); leaky
    features keep the source speaker because identity is in the features, not
    the embedding (L1 stays LOW). Score = L1(shuffled) - L1(true); large = good.

It also saves mel-comparison figures and vocoded audio (target / recon-true /
recon-shuffled) through HiFi-GAN so you can see and hear what the tiny decoder
produces.

Mirrors per_speaker_leakage.py conventions (frozen SIVE, mel-from-waveform,
multi-checkpoint, markdown report). Pair the two for a 2D usability map:
high retention (low true-L1) + low leakage.

Usage:
    python -m megatransformer.scripts.eval.audio.sive.synthesis_usability \
        --config small_deep_3xdownsample_conv2d_attentive --num_speakers 3610 \
        --val_cache_dir ./cached_datasets/voice_sive_gender_val_merged/ \
        --output_dir ./eval_output/synthesis_usability --vocoder_config hifigan \
        --checkpoint stdhinge11=runs/sive/stdhinge_..._1_1/checkpoint-224000
"""

import argparse
import hashlib
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile

from megatransformer.model.voice.sive.sive import SpeakerInvariantVoiceEncoder
from megatransformer.scripts.data.voice.dataset import VoiceShardedDataset
from megatransformer.utils.audio_utils import SharedWindowBuffer, extract_mels
from megatransformer.utils.model_loading_utils import load_model, load_vocoder
from megatransformer.utils import visualization


# ---------------------------------------------------------------------------
# Subset extraction: per-utterance (SIVE feature seq upsampled to mel rate, mel, emb)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_subset(model, dataset, indices, device, batch_size,
                   sr, n_mels, n_fft, hop_length):
    """For each chosen utterance, return (feat_up [256,T], mel [n_mels,T], emb [E],
    speaker_id). Features are linear-interpolated to the mel frame rate so the
    decoder sees frame-aligned inputs (the SMG upsamples too)."""
    model.eval()
    buf = SharedWindowBuffer()
    feats, mels, embs, spks, gens = [], [], [], [], []

    for start in tqdm(range(0, len(indices), batch_size), desc="Extracting subset"):
        chunk = indices[start:start + batch_size]
        mel_list, lengths, emb_list, spk_list, gen_list = [], [], [], [], []
        for i in chunk:
            s = dataset[int(i)]
            wav_len = int(s["waveform_length"])
            wav = s["waveform"][:wav_len].to(torch.float32)
            mel = extract_mels(buf, wav, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)  # [n_mels, T]
            mel_list.append(mel)
            lengths.append(mel.shape[-1])
            emb_list.append(s["speaker_embedding"].float().reshape(-1))
            spk_list.append(int(s["speaker_id"]))
            gg = s.get("gender_id", None)
            gen_list.append(-1 if gg is None else int(gg if not isinstance(gg, torch.Tensor) else gg.item()))

        max_t = max(m.shape[-1] for m in mel_list)
        mel_batch = torch.zeros(len(mel_list), n_mels, max_t)
        for j, m in enumerate(mel_list):
            mel_batch[j, :, :m.shape[-1]] = m
        mel_batch = mel_batch.to(device)
        len_batch = torch.tensor(lengths, dtype=torch.long, device=device)

        result = model(mel_batch, lengths=len_batch, grl_alpha=0.0)
        feat = result["features"]            # [B, T', D]
        feat_len = result["feature_lengths"]  # [B]

        for j in range(feat.shape[0]):
            tprime = max(int(feat_len[j].item()), 1)
            tmel = lengths[j]
            f = feat[j, :tprime, :].transpose(0, 1).unsqueeze(0)        # [1, D, T']
            f_up = F.interpolate(f, size=tmel, mode="linear", align_corners=False)[0]  # [D, T]
            feats.append(f_up.cpu())
            mels.append(mel_list[j])          # [n_mels, T]
            embs.append(emb_list[j])
            spks.append(spk_list[j])
            gens.append(gen_list[j])

    return (feats, mels, torch.stack(embs),
            torch.tensor(spks, dtype=torch.long), torch.tensor(gens, dtype=torch.long))


# ---------------------------------------------------------------------------
# Decoder: SMG-shaped — FiLM-conditioned 1D conv (frame-parallel, ~0.5-1.5M params)
# ---------------------------------------------------------------------------

class FiLMConvDecoder(nn.Module):
    def __init__(self, feat_dim=256, emb_dim=192, width=192, n_blocks=4, kernel=5, n_mels=80):
        super().__init__()
        self.stem = nn.Conv1d(feat_dim, width, 1)
        self.blocks = nn.ModuleList([
            nn.Conv1d(width, width, kernel, padding=kernel // 2) for _ in range(n_blocks)
        ])
        self.film = nn.Sequential(
            nn.Linear(emb_dim, width * 2), nn.GELU(), nn.Linear(width * 2, n_blocks * 2 * width)
        )
        self.head = nn.Conv1d(width, n_mels, 1)
        self.n_blocks, self.width = n_blocks, width

    def forward(self, feat_up, emb):
        # feat_up [B, feat_dim, T], emb [B, emb_dim]
        h = self.stem(feat_up)
        film = self.film(emb).view(emb.size(0), self.n_blocks, 2, self.width)
        for i, conv in enumerate(self.blocks):
            g = film[:, i, 0].unsqueeze(-1)
            b = film[:, i, 1].unsqueeze(-1)
            h = h + F.gelu(g * conv(h) + b)   # FiLM speaker conditioning per block
        return self.head(h)


def _collate(feats, mels, idxs, device):
    """Pad a set of variable-length (feat_up, mel) to batch max; return tensors + mask."""
    maxt = max(feats[i].shape[-1] for i in idxs)
    D, n_mels = feats[idxs[0]].shape[0], mels[idxs[0]].shape[0]
    fb = torch.zeros(len(idxs), D, maxt)
    mb = torch.zeros(len(idxs), n_mels, maxt)
    mask = torch.zeros(len(idxs), 1, maxt)
    for k, i in enumerate(idxs):
        t = feats[i].shape[-1]
        fb[k, :, :t] = feats[i]
        mb[k, :, :t] = mels[i]
        mask[k, 0, :t] = 1.0
    return fb.to(device), mb.to(device), mask.to(device)


def _masked_l1(pred, tgt, mask):
    n_mels = pred.shape[1]
    return (F.l1_loss(pred, tgt, reduction="none") * mask).sum() / (mask.sum() * n_mels + 1e-8)


def train_decoder(feats, mels, embs, train_idx, args, device):
    torch.manual_seed(args.seed)
    dec = FiLMConvDecoder(feat_dim=feats[0].shape[0], emb_dim=embs.shape[1],
                          width=args.dec_width, n_blocks=args.dec_blocks,
                          kernel=args.dec_kernel, n_mels=mels[0].shape[0]).to(device)
    nparams = sum(p.numel() for p in dec.parameters())
    opt = optim.Adam(dec.parameters(), lr=args.probe_lr)
    g = torch.Generator().manual_seed(args.seed)
    dec.train()
    for step in range(args.probe_steps):
        sel = [train_idx[i] for i in torch.randint(0, len(train_idx), (args.probe_batch,), generator=g).tolist()]
        fb, mb, mask = _collate(feats, mels, sel, device)
        eb = embs[sel].to(device)
        opt.zero_grad()
        loss = _masked_l1(dec(fb, eb), mb, mask)
        loss.backward()
        opt.step()
    return dec, nparams


@torch.no_grad()
def eval_decoder(dec, feats, mels, embs, spks, eval_idx, device, batch=32):
    """Per-utterance recon L1 with the true embedding and with a different-speaker
    (shuffled) embedding (recon always compared to the ORIGINAL mel). Returns
    (order, l1_true[order], l1_shuf[order], shuffled_emb_idx) so callers can
    stratify by gender."""
    dec.eval()
    order = list(eval_idx)
    shuffled_emb_idx = order[len(order) // 2:] + order[:len(order) // 2]
    for k in range(len(order)):
        if spks[order[k]] == spks[shuffled_emb_idx[k]]:
            shuffled_emb_idx[k] = shuffled_emb_idx[(k + 1) % len(order)]

    n_mels = mels[order[0]].shape[0]
    l1_true = np.zeros(len(order), dtype=np.float64)
    l1_shuf = np.zeros(len(order), dtype=np.float64)
    for start in range(0, len(order), batch):
        sel = order[start:start + batch]
        sel_sh = shuffled_emb_idx[start:start + batch]
        fb, mb, mask = _collate(feats, mels, sel, device)
        denom = mask.sum(dim=(1, 2)) * n_mels + 1e-8
        pt = (F.l1_loss(dec(fb, embs[sel].to(device)), mb, reduction="none") * mask).sum(dim=(1, 2)) / denom
        ps = (F.l1_loss(dec(fb, embs[sel_sh].to(device)), mb, reduction="none") * mask).sum(dim=(1, 2)) / denom
        l1_true[start:start + len(sel)] = pt.cpu().numpy()
        l1_shuf[start:start + len(sel)] = ps.cpu().numpy()
    return order, l1_true, l1_shuf, shuffled_emb_idx


# ---------------------------------------------------------------------------
# Rendering: mel comparison figures + vocoded audio
# ---------------------------------------------------------------------------

@torch.no_grad()
def render_samples(dec, feats, mels, embs, spks, gens, eval_idx, vocoder, args, name, device):
    dec.eval()
    out = os.path.join(args.output_dir, name)
    os.makedirs(out, exist_ok=True)
    gmap = {0: "m", 1: "f", -1: "u"}
    # Render a gender-balanced spread so female targets are always represented.
    males = [int(o) for o in eval_idx if int(gens[o]) == 0]
    females = [int(o) for o in eval_idx if int(gens[o]) == 1]
    h = args.num_render // 2
    order = (females[:h] + males[:args.num_render - len(females[:h])])[:args.num_render]
    if not order:
        order = list(eval_idx)[:args.num_render]
    # one CROSS-gender wrong-speaker embedding per sample where possible (most revealing)
    for k, i in enumerate(order):
        f = feats[i].unsqueeze(0).to(device)
        tgt = mels[i]                                   # [n_mels, T]
        pred_true = dec(f, embs[i:i + 1].to(device))[0].cpu()   # [n_mels, T]
        gi = int(gens[i])
        j = next((o for o in eval_idx if int(gens[o]) >= 0 and int(gens[o]) != gi), None)  # cross-gender
        if j is None:
            j = next((o for o in eval_idx if spks[o] != spks[i]), i)
        pred_shuf = dec(f, embs[j:j + 1].to(device))[0].cpu()
        gj = gmap[int(gens[j])]

        fig = visualization.render_mel_comparison(pred_true.numpy(), tgt.numpy())
        fig.savefig(os.path.join(out, f"sample{k}_spk{int(spks[i])}_{gmap[gi]}_mel.png"), dpi=100)
        plt.close(fig)

        if vocoder is not None:
            for tag, mel in [("target", tgt), ("recon_true", pred_true), (f"recon_wrong_{gj}", pred_shuf)]:
                try:
                    wav = visualization.render_vocoder_audio(vocoder, mel.to(torch.float32))
                    wav = np.clip(np.asarray(wav), -1, 1)
                    wavfile.write(os.path.join(out, f"sample{k}_spk{int(spks[i])}_{gmap[gi]}_{tag}.wav"),
                                  args.voice_sample_rate, (wav * 32767).astype(np.int16))
                except Exception as e:
                    print(f"  [{name}] vocode {tag} failed: {e}")
    print(f"  [{name}] wrote {len(order)} mel figures + audio to {out}/")


# ---------------------------------------------------------------------------
# Pitch (F0) + spectral-centroid gender analysis
# ---------------------------------------------------------------------------

def _mel_centroid(mel_np):
    """Spectral centroid as a mel-bin index, linear-energy weighted, averaged over
    frames. Higher = brighter/higher (female-leaning); lower = darker ('masculine
    undertone'). Robust — computed directly from the predicted mel, no vocoding."""
    lin = np.exp(mel_np)                       # log-mel -> ~linear energy
    bins = np.arange(mel_np.shape[0])[:, None]
    return float(((bins * lin).sum(0) / (lin.sum(0) + 1e-8)).mean())


def _plot_f0_contours(by_gender, out_dir, name):
    """Per-frame F0 contour for one sample per source gender: target vs recon(true emb)
    vs recon(cross-gender emb). The cross-gender curve should move toward the OTHER
    gender's range if the decoder re-pitches; if it tracks the source/true curve, it
    failed to re-gender."""
    rows = [(g, by_gender[g]) for g in ("male", "female")
            if g in by_gender and by_gender[g].get("_contours")]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 3 * len(rows)), squeeze=False)
    for r, (g, d) in enumerate(rows):
        ax = axes[r][0]
        ct, cr, cc = d["_contours"][0]
        ax.plot(ct, label="target", color="black", lw=1.3)
        ax.plot(cr, label="recon (true emb)", color="tab:blue", lw=1)
        ax.plot(cc, label="recon (cross-gender emb)", color="tab:red", lw=1)
        ax.set_title(f"[{name}] {g} source — F0 contour (cross-gender should leave the source range)")
        ax.set_ylabel("F0 (Hz)")
        ax.set_ylim(50, 320)
        ax.legend(fontsize=8)
    axes[-1][0].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "f0_contours.png"), dpi=100)
    plt.close(fig)


@torch.no_grad()
def spectral_gender_analysis(dec, feats, mels, embs, gens, eval_idx, vocoder, args, name, device):
    """Per-gender median F0 + mel centroid, AND the cross-gender re-pitch test: feed a
    source utterance an OPPOSITE-gender embedding and check whether the output F0 moves
    toward the target gender's range. repitch fraction: 1.0 = fully re-pitched to the
    other gender, 0 = stayed at source gender, <0 = moved further from target."""
    import torchcrepe
    dec.eval()
    out_dir = os.path.join(args.output_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    def _f0(mel):
        wav = np.asarray(visualization.render_vocoder_audio(vocoder, mel.to(torch.float32)))
        w = torch.tensor(np.clip(wav, -1, 1), dtype=torch.float32).unsqueeze(0)
        pitch, period = torchcrepe.predict(
            w, 16000, hop_length=256, fmin=50.0, fmax=550.0, model="full",
            decoder=torchcrepe.decode.viterbi, return_periodicity=True,
            batch_size=512, device=device, pad=True)
        voiced = period[0] > 0.5
        med = float(pitch[0][voiced].median().item()) if voiced.any() else float("nan")
        contour = pitch[0].cpu().numpy().copy()
        contour[~voiced.cpu().numpy()] = np.nan
        return med, contour

    by_gender = {}
    for gid, gname in [(0, "male"), (1, "female")]:
        src = [int(o) for o in eval_idx if int(gens[o]) == gid][:args.f0_samples]
        cross_pool = [int(o) for o in eval_idx if int(gens[o]) == (1 - gid)]
        if not src or not cross_pool:
            continue
        f0_t, f0_r, f0_c, c_t, c_r, contours, vr = [], [], [], [], [], [], 0
        for n, i in enumerate(src):
            tgt = mels[i]
            rt = dec(feats[i].unsqueeze(0).to(device), embs[i:i + 1].to(device))[0].cpu()
            j = cross_pool[n % len(cross_pool)]
            rc = dec(feats[i].unsqueeze(0).to(device), embs[j:j + 1].to(device))[0].cpu()
            c_t.append(_mel_centroid(tgt.numpy()))
            c_r.append(_mel_centroid(rt.numpy()))
            try:
                mt, ct_ = _f0(tgt)
                mr, cr_ = _f0(rt)
                mc, cc_ = _f0(rc)
                if not np.isnan(mt): f0_t.append(mt)
                if not np.isnan(mr): f0_r.append(mr); vr += 1
                if not np.isnan(mc): f0_c.append(mc)
                if len(contours) < 1:
                    contours.append((ct_, cr_, cc_))
            except Exception:
                pass
        by_gender[gname] = {
            "n": len(src),
            "f0_target": float(np.median(f0_t)) if f0_t else float("nan"),
            "f0_recon": float(np.median(f0_r)) if f0_r else float("nan"),
            "f0_recon_cross": float(np.median(f0_c)) if f0_c else float("nan"),
            "f0_voiced_frac_recon": vr / max(len(src), 1),
            "centroid_target": float(np.median(c_t)) if c_t else float("nan"),
            "centroid_recon": float(np.median(c_r)) if c_r else float("nan"),
            "_contours": contours,
        }

    # Re-pitch fraction, anchored to the real per-gender target F0 medians.
    if "male" in by_gender and "female" in by_gender:
        m, f = by_gender["male"]["f0_target"], by_gender["female"]["f0_target"]
        span = f - m
        if span and not (np.isnan(m) or np.isnan(f)):
            # male source + female emb should rise toward f; female source + male emb toward m
            by_gender["male"]["repitch"] = (by_gender["male"]["f0_recon_cross"] - m) / span
            by_gender["female"]["repitch"] = (by_gender["female"]["f0_recon_cross"] - f) / (-span)

    _plot_f0_contours(by_gender, out_dir, name)
    for v in by_gender.values():
        v.pop("_contours", None)
    return by_gender


# ---------------------------------------------------------------------------
# Per-checkpoint
# ---------------------------------------------------------------------------

def analyze_checkpoint(args, name, ckpt_path, vocoder, dataset, subset_indices, device):
    print(f"\n{'='*70}\n[{name}] {ckpt_path}\n{'='*70}")
    model = load_model(SpeakerInvariantVoiceEncoder, args.config, checkpoint_path=ckpt_path,
                       device=device,
                       overrides={"num_speakers": args.num_speakers,
                                  **({"final_norm_type": args.final_norm_type} if args.final_norm_type else {})},
                       strict=False, allow_size_mismatch=True)
    feats, mels, embs, spks, gens = extract_subset(
        model, dataset, subset_indices, device, args.batch_size,
        args.voice_sample_rate, args.voice_n_mels, args.voice_n_fft, args.voice_hop_length)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    n = len(feats)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)
    n_eval = max(1, int(n * args.eval_frac))
    eval_idx, train_idx = list(perm[:n_eval]), list(perm[n_eval:])

    dec, nparams = train_decoder(feats, mels, embs, train_idx, args, device)
    order, l1t, l1s, sh_idx = eval_decoder(dec, feats, mels, embs, spks, eval_idx, device)
    render_samples(dec, feats, mels, embs, spks, gens, eval_idx, vocoder, args, name, device)
    spectral = (spectral_gender_analysis(dec, feats, mels, embs, gens, eval_idx, vocoder, args, name, device)
                if (vocoder is not None and not args.no_f0) else {})

    l1_true, l1_shuf = float(l1t.mean()), float(l1s.mean())
    delta = l1_shuf - l1_true

    # Gender stratification
    g = gens.numpy()
    tgt_g = g[order]                       # gender of the source/true speaker per eval utt
    swap_g = g[np.array(sh_idx)]           # gender of the swapped-in embedding
    def _mean(arr, mask):
        return float(arr[mask].mean()) if mask.any() else float("nan")
    l1_true_m = _mean(l1t, tgt_g == 0)
    l1_true_f = _mean(l1t, tgt_g == 1)
    dper = l1s - l1t
    cross = (tgt_g >= 0) & (swap_g >= 0) & (tgt_g != swap_g)
    same = (tgt_g >= 0) & (swap_g >= 0) & (tgt_g == swap_g)
    d_same, d_cross = _mean(dper, same), _mean(dper, cross)
    d_to_f = _mean(dper, cross & (swap_g == 1))   # asked to become female
    d_to_m = _mean(dper, cross & (swap_g == 0))   # asked to become male

    print(f"[{name}] decoder {nparams/1e6:.2f}M params | train={len(train_idx)} eval={len(eval_idx)} utts")
    print(f"[{name}] recon L1 true={l1_true:.4f}  shuffled={l1_shuf:.4f}  "
          f"disentangle Δ={delta:+.4f}  (Δ/true={delta/max(l1_true,1e-6):.2f})")
    print(f"[{name}]   by target gender — L1 male={l1_true_m:.4f} female={l1_true_f:.4f} "
          f"(female−male={l1_true_f - l1_true_m:+.4f})")
    print(f"[{name}]   swap Δ same-gender={d_same:.4f} cross-gender={d_cross:.4f} | "
          f"→female={d_to_f:.4f} →male={d_to_m:.4f}")
    for gname in ("male", "female"):
        s = spectral.get(gname)
        if s:
            other = "female" if gname == "male" else "male"
            rp = s.get("repitch", float("nan"))
            print(f"[{name}]   F0 {gname}: target={s['f0_target']:.0f}Hz true-emb={s['f0_recon']:.0f}Hz "
                  f"→{other}-emb={s['f0_recon_cross']:.0f}Hz (re-pitch={rp:+.2f}) | "
                  f"centroid t={s['centroid_target']:.1f} r={s['centroid_recon']:.1f}")
    return {"name": name, "params_M": nparams / 1e6, "l1_true": l1_true,
            "l1_shuffled": l1_shuf, "delta": delta, "n_eval": len(eval_idx),
            "l1_true_male": l1_true_m, "l1_true_female": l1_true_f,
            "delta_same": d_same, "delta_cross": d_cross,
            "delta_to_female": d_to_f, "delta_to_male": d_to_m,
            "n_male": int((tgt_g == 0).sum()), "n_female": int((tgt_g == 1).sum()),
            "spectral": spectral}


def write_report(args, results):
    lines = ["# SIVE synthesis-usability (mel reconstruction) report", "",
             f"- Decoder: FiLM-conditioned 1D conv ({args.dec_blocks} blocks, width {args.dec_width}, "
             f"k{args.dec_kernel}), fixed-budget {args.probe_steps} steps — NOT to convergence, so numbers "
             f"are comparable across checkpoints, not absolute.",
             f"- Subset {args.subset_size} val utts, {int((1-args.eval_frac)*100)}/{int(args.eval_frac*100)} "
             f"probe-train/eval split (seed {args.seed}); recon on the held-out split.",
             f"- L1 true = content sufficiency (lower better). Δ = L1(shuffled emb) − L1(true) = "
             f"disentanglement (higher better: clean features move output to the wrong speaker; leaky "
             f"features keep the source speaker so swapping the embedding barely changes recon).",
             "",
             "| run | recon L1 (true) | Δ | Δ/true |",
             "|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['name']} | {r['l1_true']:.4f} | {r['delta']:+.4f} | "
                     f"{r['delta']/max(r['l1_true'],1e-6):.2f} |")
    lines += ["", "## Gender stratification",
              "L1 by target gender (lower=better recon of that gender). female−male gap = how much "
              "worse female reconstructs. Δ→female = how much a cross-gender swap toward a female "
              "embedding changes the output (larger = embedding more in control of gender).", "",
              "| run | L1 male | L1 female | female−male | Δ same-gen | Δ cross-gen | Δ→female | Δ→male |",
              "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['name']} | {r['l1_true_male']:.4f} | {r['l1_true_female']:.4f} | "
                     f"{r['l1_true_female']-r['l1_true_male']:+.4f} | {r['delta_same']:.4f} | "
                     f"{r['delta_cross']:.4f} | {r['delta_to_female']:.4f} | {r['delta_to_male']:.4f} |")
    if any(r.get("spectral") for r in results):
        lines += ["", "## Pitch (F0) by gender, incl. cross-gender re-pitch",
                  "F0 via torchcrepe on vocoded audio (median over voiced frames). "
                  "`true-emb` = recon with the source speaker's own embedding; "
                  "`cross-emb` = source utterance + an OPPOSITE-gender embedding. "
                  "`re-pitch` = how far cross-emb F0 moved from the source gender toward the "
                  "other gender's median (1.0 = fully re-pitched, 0 = stayed at source, <0 = "
                  "moved further away). Low/negative re-pitch = cross-gender cloning fails to "
                  "move the pitch (the 'masculine female' effect). Per-run `f0_contours.png` "
                  "shows the trajectories.", "",
                  "| run | source gender | F0 target | true-emb | cross-emb | re-pitch | centroid t→r |",
                  "|---|---|---|---|---|---|---|"]
        for r in results:
            for gname in ("female", "male"):
                s = r.get("spectral", {}).get(gname)
                if s:
                    lines.append(f"| {r['name']} | {gname} | {s['f0_target']:.0f} | {s['f0_recon']:.0f} | "
                                 f"{s['f0_recon_cross']:.0f} | {s.get('repitch', float('nan')):+.2f} | "
                                 f"{s['centroid_target']:.1f}→{s['centroid_recon']:.1f} |")
    lines += ["", "Per-run mel figures + target/recon_true/recon_wrong_<g> WAVs (filenames tagged "
              "with source gender m/f) are under `<output_dir>/<run>/`."]
    path = os.path.join(args.output_dir, "synthesis_usability_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", action="append", default=[], metavar="name=path", help="Repeatable.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--num_speakers", type=int, default=3610)
    ap.add_argument("--final_norm_type", default=None,
                    help="Override the model's final_norm_type to MATCH a norm-variant checkpoint "
                         "(layernorm/rmsnorm/none). REQUIRED for rmsnorm/none runs — without it the "
                         "eval model builds a LayerNorm final and the features load wrong (garbage). "
                         "Default: use the config preset's value (layernorm).")
    ap.add_argument("--val_cache_dir", required=True)
    ap.add_argument("--output_dir", default="./eval_output/synthesis_usability")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--shard_cache_size", type=int, default=8)
    # subset / split
    ap.add_argument("--subset_size", type=int, default=1024)
    ap.add_argument("--eval_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    # decoder + probe budget
    ap.add_argument("--dec_width", type=int, default=192)
    ap.add_argument("--dec_blocks", type=int, default=4)
    ap.add_argument("--dec_kernel", type=int, default=5)
    ap.add_argument("--probe_steps", type=int, default=600)
    ap.add_argument("--probe_lr", type=float, default=3e-4)
    ap.add_argument("--probe_batch", type=int, default=16)
    # rendering / vocoder
    ap.add_argument("--num_render", type=int, default=6)
    ap.add_argument("--f0_samples", type=int, default=30, help="utts per gender for F0/centroid analysis")
    ap.add_argument("--no_f0", action="store_true", help="skip the F0/centroid gender analysis")
    ap.add_argument("--vocoder_config", default="hifigan")
    ap.add_argument("--vocoder_checkpoint_path", default=None)
    ap.add_argument("--no_audio", action="store_true", help="skip vocoder/audio rendering")
    # mel params (match training)
    ap.add_argument("--voice_sample_rate", type=int, default=16000)
    ap.add_argument("--voice_n_mels", type=int, default=80)
    ap.add_argument("--voice_n_fft", type=int, default=1024)
    ap.add_argument("--voice_hop_length", type=int, default=256)
    args = ap.parse_args()

    if not args.checkpoint:
        ap.error("pass at least one --checkpoint name=path")
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = VoiceShardedDataset(
        shard_dir=args.val_cache_dir, cache_size=args.shard_cache_size,
        columns=["waveforms", "mel_specs", "speaker_embeddings", "speaker_ids", "gender_ids"])
    n_total = len(dataset)
    rng = np.random.RandomState(args.seed)
    subset_indices = sorted(rng.choice(n_total, size=min(args.subset_size, n_total), replace=False).tolist())

    vocoder = None
    if not args.no_audio:
        try:
            vocoder = load_vocoder(args.vocoder_checkpoint_path, args.vocoder_config, SharedWindowBuffer())
        except Exception as e:
            print(f"Vocoder load failed ({e}); proceeding without audio.")

    results = []
    for spec in args.checkpoint:
        if "=" not in spec:
            ap.error(f"--checkpoint must be name=path, got: {spec}")
        nm, pth = spec.split("=", 1)
        results.append(analyze_checkpoint(args, nm, pth, vocoder, dataset, subset_indices, args.device))

    if results:
        write_report(args, results)


if __name__ == "__main__":
    main()
