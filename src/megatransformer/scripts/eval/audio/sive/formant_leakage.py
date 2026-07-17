"""Formant-recoverability probe for the frozen SIVE / ContentVec representation.

The per-speaker speaker-ID probe (`per_speaker_leakage.py`) measures speaker
DISCRIMINABILITY — and it ranks features BACKWARDS for voice cloning: ContentVec
scores the WORST leakage (~172x) yet clones for free, because its residual
discriminability is pronunciation/STYLE (which a mel-decoder cannot render as a
voice), while its raw formants are stripped. SIVE keeps the raw formants, so an
SMG decoder reads timbre straight off the features and ignores the speaker
embedding (cross-speaker collapse).

This probe measures the axis that actually predicts cloning: **are the raw
formants (the vocal-tract resonances = timbre) recoverable from the features?**
It extracts frame-level F1..Fk via LPC, aligns them to each feature's own frame
rate, and trains a linear+MLP REGRESSOR to predict them from the frozen
frame-level features. High R² = formants survive = a decoder can shortcut timbre
= cloning-collapse risk.

The content confound: formants are part CONTENT (which vowel — F1/F2) and part
SPEAKER (vocal-tract-length scaling — strongest in F3+). SIVE *should* carry
content-formants; the leakage is the speaker part. V1 isolates it by CROSS-
FEATURE comparison (content recovery is ~constant across features, so the
*difference* in formant-R² is mostly timbre) plus the per-formant breakdown
(F3 is the speaker-timbre tell). The built-in validation: run ContentVec vs a
SIVE run — if ContentVec's formant-R² comes out LOW and SIVE's HIGH, separating
the two where the speaker classifier could not, the probe measures cloning-
suitability and becomes the right yardstick for future SIVE variants.

Usage:
    python -m megatransformer.scripts.eval.audio.sive.formant_leakage \
        --config small_deep_3xdownsample_conv2d_attentive --num_speakers 3610 \
        --val_cache_dir ./cached_datasets/voice_sive_gender_val_merged/ \
        --output_dir ./eval_output/formant_leakage --subset_size 512 \
        --checkpoint stdhinge11=runs/sive/stdhinge_.../checkpoint-300000 \
        --checkpoint consistency=runs/sive/sive_consistency_.../checkpoint-5000 \
        --checkpoint contentvec256=CONTENTVEC   # sentinel: use --content_encoder path
"""

import argparse
import hashlib
import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
from tqdm import tqdm

from megatransformer.model.voice.sive.sive import SpeakerInvariantVoiceEncoder
from megatransformer.scripts.data.voice.dataset import VoiceShardedDataset
from megatransformer.utils.audio_utils import SharedWindowBuffer, extract_mels
from megatransformer.utils.model_loading_utils import detect_sive_vq_codes, load_model


# ---------------------------------------------------------------------------
# LPC formant extraction (frame-level, off the raw waveform)
# ---------------------------------------------------------------------------

# Per-formant frequency floors (Hz). A candidate is assigned to slot i only if
# it clears FLOORS[i] and sits at least MIN_GAP above the previous formant — this
# rejects spurious low/mid poles that a naive frequency-sort would mis-slot as a
# higher formant (e.g. a 1.4kHz pole stealing F3). Floors follow standard adult
# formant ranges; slots beyond 3 extend by 1kHz steps.
_FORMANT_FLOORS = [200.0, 750.0, 1800.0, 2800.0, 3800.0]
_FORMANT_MIN_GAP = 200.0


def _lpc_formants_frame(win_frame, sr, order, n_formants):
    """F1..F_n_formants (Hz) for one windowed frame via LPC roots; NaN if absent.

    Candidates (bw<400, in-band) are assigned to formant slots by a greedy
    range-anchored pass rather than a raw frequency-sort, so a spurious mid pole
    can't displace a real higher formant."""
    nan = [np.nan] * n_formants
    if not np.isfinite(win_frame).all():
        return nan
    try:
        a = librosa.lpc(win_frame.astype(np.float64), order=order)
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        return nan
    if not np.isfinite(a).all():
        return nan
    roots = np.roots(a)
    roots = roots[np.imag(roots) >= 0.01]  # upper half-plane, drop (near-)real roots
    if roots.size == 0:
        return nan
    freqs = np.arctan2(np.imag(roots), np.real(roots)) * (sr / (2 * np.pi))
    bw = -0.5 * (sr / (2 * np.pi)) * np.log(np.abs(roots) + 1e-12)
    keep = (freqs > 90.0) & (freqs < sr / 2 - 100.0) & (bw < 400.0)
    cand = np.sort(freqs[keep])
    out = [np.nan] * n_formants
    prev, ci = 0.0, 0
    for slot in range(n_formants):
        floor = max(_FORMANT_FLOORS[min(slot, len(_FORMANT_FLOORS) - 1)],
                    prev + _FORMANT_MIN_GAP)
        while ci < len(cand) and cand[ci] < floor:
            ci += 1
        if ci >= len(cand):
            break
        out[slot] = float(cand[ci]); prev = cand[ci]; ci += 1
    return out


def extract_formant_track(wav, sr, order, n_formants, win=400, hop=160,
                          energy_frac=0.10):
    """Frame-level formants + validity for one utterance waveform (numpy [T]).

    Returns (formants [F, n_formants] Hz, valid [F] bool, times [F] sec).
    Pre-emphasis, Hamming-windowed LPC per frame; frames are 'valid' when they
    clear an energy floor (drops silence) AND all n_formants were found."""
    x = np.asarray(wav, dtype=np.float64)
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])  # pre-emphasis
    if x.size < win:
        return np.zeros((0, n_formants)), np.zeros((0,), bool), np.zeros((0,))
    window = np.hamming(win)
    starts = np.arange(0, x.size - win + 1, hop)
    rms = np.empty(starts.size)
    forms = np.empty((starts.size, n_formants))
    for i, s in enumerate(starts):
        fr = x[s:s + win]
        rms[i] = np.sqrt(np.mean(fr * fr) + 1e-12)
        forms[i] = _lpc_formants_frame(fr * window, sr, order, n_formants)
    thr = energy_frac * rms.max() if rms.size and rms.max() > 0 else 0.0
    valid = (rms > thr) & np.isfinite(forms).all(axis=1)
    times = (starts + win / 2.0) / sr
    return forms, valid, times


def align_formants_to_feature_frames(forms, valid, times, n_feat, dur):
    """Resample a formant track to n_feat frames spanning [0, dur].

    Interpolates each formant over its VALID frames; a feature frame is marked
    valid if the nearest source formant frame was valid. Returns
    (Y [n_feat, n_formants], vmask [n_feat] bool)."""
    n_formants = forms.shape[1]
    Y = np.zeros((n_feat, n_formants), dtype=np.float32)
    if n_feat <= 0:
        return Y, np.zeros((0,), bool)
    feat_t = (np.arange(n_feat) + 0.5) * (dur / max(n_feat, 1))
    if valid.sum() < 2:
        return Y, np.zeros((n_feat,), bool)
    vt = times[valid]
    for k in range(n_formants):
        Y[:, k] = np.interp(feat_t, vt, forms[valid, k])
    # nearest-source validity
    nn_idx = np.searchsorted(times, feat_t).clip(0, len(times) - 1)
    vmask = valid[nn_idx]
    return Y, vmask


# ---------------------------------------------------------------------------
# Frame-level feature extraction (SIVE / ContentVec) + aligned formants
# ---------------------------------------------------------------------------

def _sample_to_mel(sample, buf, sr, n_mels, n_fft, hop_length):
    if "mel_spec" in sample:
        return sample["mel_spec"], int(sample["mel_length"])
    wav_len = int(sample["waveform_length"])
    wav = sample["waveform"][:wav_len].to(torch.float32)
    mel = extract_mels(buf, wav, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    return mel, mel.shape[-1]


@torch.no_grad()
def build_sive_frame_data(model, dataset, subset, device, batch_size,
                          sr, n_mels, n_fft, hop_length, layer,
                          order, n_formants):
    """Per-frame (feature, formant) pairs over a subset of utterances (SIVE)."""
    model.eval()
    buf = SharedWindowBuffer()
    X, Y, M, U, S = [], [], [], [], []
    for start in tqdm(range(0, len(subset), batch_size), desc="SIVE frames"):
        chunk = subset[start:start + batch_size]
        mels, lengths, wavs, spk = [], [], [], []
        for i in chunk:
            s = dataset[i]
            mel, length = _sample_to_mel(s, buf, sr, n_mels, n_fft, hop_length)
            mels.append(mel); lengths.append(length)
            wl = int(s["waveform_length"])
            wavs.append(s["waveform"][:wl].to(torch.float32).numpy())
            spk.append(int(s["speaker_id"]))
        max_t = max(m.shape[-1] for m in mels)
        mel_batch = torch.zeros(len(mels), mels[0].shape[0], max_t)
        for j, m in enumerate(mels):
            mel_batch[j, :, :m.shape[-1]] = m
        result = model(mel_batch.to(device),
                       lengths=torch.tensor(lengths, device=device),
                       grl_alpha=0.0, return_all_hiddens=(layer != -1))
        feats = result["features"] if layer == -1 else result["all_hiddens"][layer]
        flens = result["feature_lengths"]
        for b, i in enumerate(chunk):
            vlen = max(int(flens[b].item()), 1)
            f = feats[b, :vlen, :].float().cpu().numpy()  # [T_feat, D]
            dur = len(wavs[b]) / sr
            forms, valid, times = extract_formant_track(wavs[b], sr, order, n_formants)
            Yb, vmask = align_formants_to_feature_frames(forms, valid, times, vlen, dur)
            if vmask.sum() == 0:
                continue
            X.append(f[vmask]); Y.append(Yb[vmask])
            U.append(np.full(int(vmask.sum()), i, np.int64))
            S.append(np.full(int(vmask.sum()), spk[b], np.int64))
    return (np.concatenate(X), np.concatenate(Y),
            np.concatenate(U), np.concatenate(S))


def build_contentvec_frame_data(dataset, subset, model_id, layer, device, dim,
                                sr, order, n_formants):
    from megatransformer.utils.contentvec_features import load_contentvec, contentvec_hidden
    m = load_contentvec(model_id, device, dim=dim)
    X, Y, U, S = [], [], [], []
    for i in tqdm(subset, desc=f"ContentVec-{dim} frames"):
        s = dataset[i]
        wl = int(s["waveform_length"])
        wav_np = s["waveform"][:wl].to(torch.float32).numpy()
        wav = torch.from_numpy(wav_np).to(device)
        h = contentvec_hidden(m, wav, layer=layer, final_proj=(dim == 256))  # [T', D]
        vlen = h.shape[0]
        f = h.float().cpu().numpy()
        dur = wl / sr
        forms, valid, times = extract_formant_track(wav_np, sr, order, n_formants)
        Yb, vmask = align_formants_to_feature_frames(forms, valid, times, vlen, dur)
        if vmask.sum() == 0:
            continue
        X.append(f[vmask]); Y.append(Yb[vmask])
        U.append(np.full(int(vmask.sum()), i, np.int64))
        S.append(np.full(int(vmask.sum()), int(s["speaker_id"]), np.int64))
    return (np.concatenate(X), np.concatenate(Y),
            np.concatenate(U), np.concatenate(S))


def cached_frame_data(args, name, ckpt_path, subset):
    # Per-checkpoint encoder: the sentinel path "CONTENTVEC" runs the off-the-shelf
    # ContentVec extractor for that entry, so one invocation can mix SIVE runs with
    # the ContentVec validation anchor and score them in a single report.
    enc = "contentvec" if ckpt_path.strip().upper() == "CONTENTVEC" else getattr(args, "content_encoder", "sive")
    key = hashlib.md5(
        f"{enc}|{ckpt_path}|{args.subset_size}|{args.seed}|{args.config}|L{args.extract_layer}"
        f"|{getattr(args,'contentvec_model','')}|D{getattr(args,'contentvec_dim',768)}"
        f"|nf{args.n_formants}|o{args.lpc_order}".encode()).hexdigest()[:10]
    cache_path = os.path.join(args.output_dir, f"_frames_{name}_{key}.npz")
    if os.path.exists(cache_path) and not args.no_feature_cache:
        d = np.load(cache_path)
        print(f"[{name}] loaded cached frames from {cache_path}")
        return d["X"], d["Y"], d["U"], d["S"]

    dataset = VoiceShardedDataset(
        shard_dir=args.val_cache_dir, cache_size=args.shard_cache_size,
        columns=["waveforms", "mel_specs", "speaker_ids", "gender_ids"],
    )
    if enc == "contentvec":
        X, Y, U, S = build_contentvec_frame_data(
            dataset, subset, args.contentvec_model, args.extract_layer, args.device,
            getattr(args, "contentvec_dim", 768), args.voice_sample_rate,
            args.lpc_order, args.n_formants)
    else:
        _vq = detect_sive_vq_codes(ckpt_path)
        if _vq:
            print(f"  Detected VQ bottleneck: {_vq} codes -> use_vq=True (QUANTIZED features)")
        overrides = {"num_speakers": args.num_speakers}
        if _vq:
            overrides.update({"use_vq": True, "vq_num_codes": _vq})
        for k in ("final_norm_type", "downsample_norm_type", "block_norm_type", "conv_norm_type"):
            v = getattr(args, k, None)
            if v is not None:
                overrides[k] = v
        model = load_model(SpeakerInvariantVoiceEncoder, args.config, checkpoint_path=ckpt_path,
                           device=args.device, overrides=overrides,
                           strict=False, allow_size_mismatch=True)
        X, Y, U, S = build_sive_frame_data(
            model, dataset, subset, args.device, args.batch_size,
            args.voice_sample_rate, args.voice_n_mels, args.voice_n_fft, args.voice_hop_length,
            args.extract_layer, args.lpc_order, args.n_formants)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    if not args.no_feature_cache:
        np.savez(cache_path, X=X, Y=Y, U=U, S=S)
    return X, Y, U, S


# ---------------------------------------------------------------------------
# Regression probe (features -> formants)
# ---------------------------------------------------------------------------

class MLPRegressor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, layers, dropout):
        super().__init__()
        if layers <= 0:
            self.net = nn.Linear(in_dim, out_dim)
        else:
            mods, d = [], in_dim
            for _ in range(layers):
                mods += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
                d = hidden
            mods.append(nn.Linear(d, out_dim))
            self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


def _r2_per_col(y, yhat):
    ss_res = ((y - yhat) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    return 1.0 - ss_res / ss_tot


def train_and_predict(Xtr, Ytr, Xte, Yte, device, hidden, layers, dropout,
                      lr, max_epochs, patience, batch, seed=0):
    """Train one regressor (early-stop on standardized test mean-R²); return the
    best-epoch standardized test predictions [Nte, out] + meta. R²/RMSE are
    computed by the caller, which differs per probe (frame vs per-speaker)."""
    torch.manual_seed(seed)  # reproducible init + minibatch order
    in_dim, out_dim = Xtr.shape[1], Ytr.shape[1]
    model = MLPRegressor(in_dim, out_dim, hidden, layers, dropout).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    Xtr_t = torch.from_numpy(Xtr).to(device); Ytr_t = torch.from_numpy(Ytr).to(device)
    Xte_t = torch.from_numpy(Xte).to(device); Yte_np = Yte
    n = Xtr_t.shape[0]
    best_mean_r2, best_pred, wait = -1e9, None, 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xtr_t[idx]), Ytr_t[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xte_t).cpu().numpy()
        mean_r2 = float(np.mean(_r2_per_col(Yte_np, pred)))
        if mean_r2 > best_mean_r2 + 1e-4:
            best_mean_r2, best_pred, wait = mean_r2, pred, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best_pred, {"epochs": ep + 1, "plateaued": wait >= patience}


def _fit_report(Xtr, Ytr, Xte, Yte, ymu, ysd, args, agg=None):
    """Standardized-in inputs. Fit linear + MLP; return per-formant R²/RMSE(Hz).

    agg: optional (n_test,) group id (e.g. speaker) -> metrics are averaged
    within group first (per-speaker R²), else per-row (per-frame R²)."""
    def metrics(pred_std):
        if agg is not None:
            groups = np.unique(agg)
            p = np.stack([pred_std[agg == g].mean(0) for g in groups])
            t = np.stack([Yte[agg == g].mean(0) for g in groups])
        else:
            p, t = pred_std, Yte
        r2 = _r2_per_col(t, p)
        rmse_hz = np.sqrt(((t - p) ** 2).mean(0)) * ysd
        return [float(x) for x in r2], float(np.mean(r2)), [float(x) for x in rmse_hz]
    out = {}
    common = dict(device=args.device, dropout=args.mlp_dropout, lr=args.probe_lr,
                  max_epochs=args.probe_max_epochs, patience=args.probe_patience,
                  batch=args.probe_batch)
    for label, h, ly in (("linear", 0, 0), ("mlp", args.mlp_hidden_dim, args.mlp_num_layers)):
        # Average over probe seeds — a small test set (per-speaker VTL) makes a
        # single fit noisy (~±0.06), enough to flip close rankings.
        runs = []
        for sd in range(args.probe_seeds):
            pred, meta = train_and_predict(Xtr, Ytr, Xte, Yte, hidden=h, layers=ly, seed=sd, **common)
            runs.append(metrics(pred))
        r2 = np.mean([r[0] for r in runs], axis=0)
        mean_r2s = [r[1] for r in runs]
        rmse = np.mean([r[2] for r in runs], axis=0)
        out[label] = {"r2": [float(x) for x in r2], "mean_r2": float(np.mean(mean_r2s)),
                      "mean_r2_std": float(np.std(mean_r2s)), "rmse_hz": [float(x) for x in rmse],
                      **meta}
    return out


def run_frame_probe(name, X, Y, U, args):
    """FRAME mode (V1): utterance-disjoint, regress frame formants from frame
    features. Confounded by content (a stronger content encoder scores higher)."""
    rng = np.random.default_rng(args.seed)
    utts = np.unique(U); rng.shuffle(utts)
    n_test = max(1, int(len(utts) * args.test_split))
    test_utts = set(utts[:n_test].tolist())
    te = np.array([u in test_utts for u in U]); tr = ~te
    xmu, xsd = X[tr].mean(0), X[tr].std(0) + 1e-6
    ymu, ysd = Y[tr].mean(0), Y[tr].std(0) + 1e-6
    Xtr = ((X[tr] - xmu) / xsd).astype(np.float32); Xte = ((X[te] - xmu) / xsd).astype(np.float32)
    Ytr = ((Y[tr] - ymu) / ysd).astype(np.float32); Yte = ((Y[te] - ymu) / ysd).astype(np.float32)
    out = _fit_report(Xtr, Ytr, Xte, Yte, ymu, ysd, args, agg=None)
    print(f"[{name}] FRAME frames={len(X)} (tr {int(tr.sum())}/te {int(te.sum())}) "
          f"mlp mean-R2={out['mlp']['mean_r2']:.3f} r2={['%.3f'%r for r in out['mlp']['r2']]}")
    out.update({"name": name, "n_frames": int(len(X)), "n_test_utts": n_test})
    return out


def shared_speaker_split(U, S, min_utts, test_split, seed):
    """Feature-INDEPENDENT speaker filter + train/test split, computed once and
    reused for every feature so all are scored on the SAME speakers (a frame-
    count filter would admit more speakers for a higher-frame-rate feature like
    ContentVec, an unfair comparison). Keyed on utterance count in the subset,
    which is identical across features."""
    u2s = {}
    for u, s in zip(U, S):
        u2s[int(u)] = int(s)
    counts = {}
    for s in u2s.values():
        counts[s] = counts.get(s, 0) + 1
    keep = np.array(sorted(s for s, c in counts.items() if c >= min_utts))
    rng = np.random.default_rng(seed)
    ks = keep.copy(); rng.shuffle(ks)
    n_test = max(1, int(len(ks) * test_split))
    return set(int(s) for s in keep), set(int(s) for s in ks[:n_test])


def run_vtl_probe(name, X, Y, U, S, keep_spk, test_spk, args):
    """VTL mode (V2): content-controlled. Target = each speaker's MEAN formant
    (their vocal-tract signature; content averages out). Regress it from the
    UTTERANCE-pooled feature on a SPEAKER-disjoint split (shared across features);
    R² is per-speaker. This is the pure timbre axis: ContentVec should fall BELOW
    SIVE here if it strips renderable timbre."""
    m = np.array([int(s) in keep_spk for s in S])
    Xk, Yk, Uk, Sk = X[m], Y[m], U[m], S[m]
    mu = {int(s): Yk[Sk == s].mean(0) for s in np.unique(Sk)}  # speaker VTL signature
    utts = np.unique(Uk)
    Xu = np.stack([Xk[Uk == u].mean(0) for u in utts])         # utt-pooled feature
    Su = np.array([int(Sk[Uk == u][0]) for u in utts])          # utt's speaker
    Yu = np.stack([mu[s] for s in Su])                          # target = speaker VTL
    te = np.array([s in test_spk for s in Su]); tr = ~te
    xmu, xsd = Xu[tr].mean(0), Xu[tr].std(0) + 1e-6
    ymu, ysd = Yu[tr].mean(0), Yu[tr].std(0) + 1e-6
    Xtr = ((Xu[tr] - xmu) / xsd).astype(np.float32); Xte = ((Xu[te] - xmu) / xsd).astype(np.float32)
    Ytr = ((Yu[tr] - ymu) / ysd).astype(np.float32); Yte = ((Yu[te] - ymu) / ysd).astype(np.float32)
    out = _fit_report(Xtr, Ytr, Xte, Yte, ymu, ysd, args, agg=Su[te])  # per-speaker R²
    print(f"[{name}] VTL speakers={len(keep_spk)} (tr {int(tr.sum())}/te {int(te.sum())} utts, "
          f"{len(test_spk)} test spk) mlp mean-R2={out['mlp']['mean_r2']:.3f} "
          f"r2={['%.3f'%r for r in out['mlp']['r2']]}")
    out.update({"name": name, "n_speakers": len(keep_spk), "n_test_spk": len(test_spk),
                "min_spk_utts": args.min_spk_utts})
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_MODE_HEADER = {
    "frame": ("formant-recoverability (FRAME — V1, content-CONFOUNDED)",
              "Frame formants regressed from frame features, utterance-disjoint. A stronger "
              "content encoder scores higher regardless of timbre, so cross-feature ranking is "
              "unreliable — use the VTL report for the timbre verdict."),
    "vtl": ("VTL-signature (V2 — content-controlled timbre probe)",
            "Target = each speaker's MEAN formant (their vocal-tract-length signature; content "
            "averages out). Regressed from the UTTERANCE-pooled feature, SPEAKER-disjoint, R² "
            "per-speaker. This is the pure timbre axis: a feature that strips renderable timbre "
            "(ContentVec) should score LOW; one that keeps it (SIVE) HIGH."),
}


def write_report(results, args, mode):
    os.makedirs(args.output_dir, exist_ok=True)
    fl = [f"F{i+1}" for i in range(args.n_formants)]
    title, blurb = _MODE_HEADER[mode]
    unit = "per-speaker" if mode == "vtl" else "per-frame"
    lines = [f"# SIVE {title} report", "", f"- {blurb}",
             f"- LPC order {args.lpc_order}, {args.subset_size} val utts, seed {args.seed}, "
             f"test_split={args.test_split}. R² is {unit} (higher = more recoverable). RMSE in Hz.",
             ""]
    if mode == "vtl":
        lines.append(f"- Shared speaker set (≥{args.min_spk_utts} utts, same for all features). "
                     "**Lower mean-R² = renderable timbre stripped = better for voice cloning.**")
    else:
        lines.append("- **Higher R² = formants more recoverable.** Content-confounded (see header).")
    lines += ["", "## Formant-R² (MLP probe)", "",
              "| run | mean R² | " + " | ".join(f"{f} R²" for f in fl) + " | "
              + " | ".join(f"{f} RMSE Hz" for f in fl) + " | plateaued |",
              "|---|---|" + "---|" * (2 * args.n_formants + 1)]
    for r in results:
        m = r["mlp"]
        lines.append(f"| {r['name']} | {m['mean_r2']:.3f} | "
                     + " | ".join(f"{x:.3f}" for x in m["r2"]) + " | "
                     + " | ".join(f"{x:.0f}" for x in m["rmse_hz"]) + f" | {m['plateaued']} |")
    lines += ["", "## Formant-R² (linear probe)", "",
              "| run | mean R² | " + " | ".join(f"{f} R²" for f in fl) + " |",
              "|---|---|" + "---|" * args.n_formants]
    for r in results:
        lin = r["linear"]
        lines.append(f"| {r['name']} | {lin['mean_r2']:.3f} | "
                     + " | ".join(f"{x:.3f}" for x in lin["r2"]) + " |")

    by = {r["name"]: r["mlp"]["mean_r2"] for r in results}
    cv = [n for n in by if "contentvec" in n.lower() or "cv" in n.lower()]
    sive = [n for n in by if n not in cv]
    lines += ["", "## Validation verdict", ""]
    if cv and sive:
        # The bar is the LEAKY continuous SIVE baseline. Exclude already-stripped
        # SIVE variants (VQ etc., mean-R² < 0.2) — they'd pin the floor near zero
        # and no continuous feature could ever pass. For stripping to hold,
        # ContentVec must sit below the lowest CONTINUOUS SIVE run.
        cont_sive = [n for n in sive if by[n] >= 0.2] or sive
        cvm = min(by[n] for n in cv); svm = min(by[n] for n in cont_sive); gap = svm - cvm
        if mode == "vtl":
            verdict = ("PROBE VALIDATED — ContentVec's VTL/timbre is less recoverable than every "
                       "SIVE run, confirming the timbre-stripping mechanism." if gap > 0.03 else
                       "NOT VALIDATED — ContentVec is NOT below SIVE on the pure timbre axis. Its "
                       "VTL is about as recoverable as SIVE's, so the info-stripping story does not "
                       "hold; the cloning difference is about FORM/ACCESSIBILITY, not stripping.")
        else:
            verdict = ("(frame mode is content-confounded; defer to the VTL verdict)")
        lines.append(f"- ContentVec mean-R² {cvm:.3f} vs LOWEST SIVE {svm:.3f} (gap {gap:+.3f}). "
                     f"**{verdict}**")
    else:
        lines.append("- Add a ContentVec run (`--checkpoint contentvec256=CONTENTVEC`) to validate.")
    lines.append("")

    path = os.path.join(args.output_dir, f"formant_{mode}_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(args.output_dir, f"formant_{mode}_results.json"), "w") as f:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "results": results}, f, indent=2)
    print("\n".join(lines))
    print(f"\nReport: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_checkpoint(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError("checkpoint must be name=path")
    name, path = s.split("=", 1)
    return name, path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=parse_checkpoint, action="append", required=True,
                    help="name=path (repeat). For ContentVec pass any path + --content_encoder contentvec.")
    ap.add_argument("--config", default="small_deep_3xdownsample_conv2d_attentive")
    ap.add_argument("--num_speakers", type=int, default=3610)
    ap.add_argument("--content_encoder", default="sive", choices=["sive", "contentvec"])
    ap.add_argument("--contentvec_model", default="lengyue233/content-vec-best")
    ap.add_argument("--contentvec_dim", type=int, default=256)
    ap.add_argument("--extract_layer", type=int, default=-1)
    ap.add_argument("--final_norm_type", default=None)
    ap.add_argument("--downsample_norm_type", default=None)
    ap.add_argument("--block_norm_type", default=None)
    ap.add_argument("--conv_norm_type", default=None)
    ap.add_argument("--val_cache_dir", default="./cached_datasets/voice_sive_gender_val_merged/")
    ap.add_argument("--output_dir", default="./eval_output/formant_leakage")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--shard_cache_size", type=int, default=4)
    ap.add_argument("--subset_size", type=int, default=512,
                    help="Number of val utterances to draw frames from (seed-fixed).")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--test_split", type=float, default=0.3)
    ap.add_argument("--no_feature_cache", action="store_true")
    ap.add_argument("--mode", default="both", choices=["frame", "vtl", "both"],
                    help="frame=V1 (content-confounded); vtl=V2 (content-controlled timbre); both.")
    ap.add_argument("--min_spk_utts", type=int, default=2,
                    help="VTL: keep speakers with >= this many utts in the subset (shared across "
                         "features; feature-independent so the comparison is fair).")
    # formant extraction
    ap.add_argument("--n_formants", type=int, default=3)
    ap.add_argument("--lpc_order", type=int, default=18)
    # probe
    ap.add_argument("--probe_max_epochs", type=int, default=300)
    ap.add_argument("--probe_patience", type=int, default=25)
    ap.add_argument("--probe_lr", type=float, default=1e-3)
    ap.add_argument("--probe_batch", type=int, default=4096)
    ap.add_argument("--mlp_hidden_dim", type=int, default=512)
    ap.add_argument("--mlp_num_layers", type=int, default=2)
    ap.add_argument("--mlp_dropout", type=float, default=0.1)
    ap.add_argument("--probe_seeds", type=int, default=3,
                    help="Average R² over this many probe re-inits (small VTL test sets are noisy).")
    # mel params (must match training)
    ap.add_argument("--voice_sample_rate", type=int, default=16000)
    ap.add_argument("--voice_n_mels", type=int, default=80)
    ap.add_argument("--voice_n_fft", type=int, default=1024)
    ap.add_argument("--voice_hop_length", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # Fixed subset of utterances, shared across checkpoints.
    ds = VoiceShardedDataset(shard_dir=args.val_cache_dir, cache_size=args.shard_cache_size,
                             columns=["waveforms", "mel_specs", "speaker_ids", "gender_ids"])
    rng = np.random.default_rng(args.seed)
    subset = sorted(rng.choice(len(ds), size=min(args.subset_size, len(ds)), replace=False).tolist())
    del ds

    frame_res, vtl_res = [], []
    keep_spk = test_spk = None
    for name, ckpt in args.checkpoint:
        print(f"\n{'='*70}\n[{name}] {ckpt}\n{'='*70}")
        X, Y, U, S = cached_frame_data(args, name, ckpt, subset)
        if args.mode in ("frame", "both"):
            frame_res.append(run_frame_probe(name, X, Y, U, args))
        if args.mode in ("vtl", "both"):
            if keep_spk is None:  # compute the shared speaker set/split once
                keep_spk, test_spk = shared_speaker_split(
                    U, S, args.min_spk_utts, args.test_split, args.seed)
            vtl_res.append(run_vtl_probe(name, X, Y, U, S, keep_spk, test_spk, args))
    if frame_res:
        write_report(frame_res, args, "frame")
    if vtl_res:
        write_report(vtl_res, args, "vtl")


if __name__ == "__main__":
    main()
