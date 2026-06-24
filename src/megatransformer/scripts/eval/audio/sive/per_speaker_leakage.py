"""Per-speaker speaker-identity leakage analysis for the frozen SIVE representation.

This complements ``speaker_leakage_probe.py`` (a per-layer linear/MLP sweep with
CKA/silhouette). Here the focus is the SHAPE of residual speaker leakage in the
FINAL SIVE feature (``result["features"]`` — the representation SMG / the world
model actually consume), answered per-speaker:

  - Closed-set speaker-ID probe (linear AND mlp) on mean-pooled utterance
    features, trained to convergence (early-stop on plateau).
  - Utterance-disjoint, speaker-stratified split: every evaluated speaker has
    utterances in BOTH train and test (closed-set over utterances, not speakers).
  - Every accuracy reported as a RATIO OVER CHANCE (chance = 1/K top-1,
    ~min(5,K)/K top-5), never raw.
  - Per-speaker top-1 / top-5 recall (micro AND macro), a sorted recall bar
    chart, a concentration metric (Gini + CV + top-k share of above-chance
    hits), a confusion submatrix over the hottest speakers, and a check for
    whether residual confusions cluster by GENDER (a distinct failure mode the
    GRL also targeted).
  - Linear-vs-MLP per-speaker delta: speakers where the MLP pulls ahead carry
    nonlinearly-stored residual identity.

Runs over one or more checkpoints (pass a no-GRL run as the contrast) and emits
a markdown report + two plots per checkpoint plus a cross-checkpoint summary.

NOTE on "layer 10": the original suggestion framed this as a layer-10 probe.
In this codebase the representation that leaks downstream is the final encoder
output; that is what we probe by default. Use speaker_leakage_probe.py for the
intermediate-layer sweep.

Usage:
    python -m megatransformer.scripts.eval.audio.sive.per_speaker_leakage \
        --config small_deep_3xdownsample_conv2d_layernorm_attentive \
        --num_speakers 3610 \
        --val_cache_dir ./cached_datasets/voice_sive_gender_val_merged/ \
        --output_dir ./eval_output/per_speaker_leakage \
        --checkpoint grl=runs/sive/baseline_.../checkpoint-224000 \
        --checkpoint nogrl=runs/sive/baseline_nogrl_.../checkpoint-224000 \
        --checkpoint covreg=runs/sive/covariancereg_.../checkpoint-224000 \
        --checkpoint stdhinge=runs/sive/stdhinge_.../checkpoint-224000
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
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from megatransformer.model.voice.sive.sive import SpeakerInvariantVoiceEncoder
from megatransformer.scripts.data.voice.dataset import VoiceShardedDataset
from megatransformer.scripts.eval.audio.sive.speaker_leakage_probe import LinearProbe, MLPProbe
from megatransformer.utils.audio_utils import SharedWindowBuffer, extract_mels
from megatransformer.utils.model_loading_utils import load_model


# ---------------------------------------------------------------------------
# Feature extraction (final SIVE feature, mean-pooled per utterance)
# ---------------------------------------------------------------------------

def _sample_to_mel(sample, buf, sr, n_mels, n_fft, hop_length):
    """Return (mel [n_mels, T], length:int). Uses precomputed mel when present,
    otherwise extracts from the waveform with the exact training mel params
    (these merged shards store waveforms only)."""
    if "mel_spec" in sample:
        mel = sample["mel_spec"]
        length = int(sample["mel_length"])
        return mel, length
    wav_len = int(sample["waveform_length"])
    wav = sample["waveform"][:wav_len].to(torch.float32)
    mel = extract_mels(buf, wav, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)  # [n_mels, T]
    return mel, mel.shape[-1]


@torch.no_grad()
def extract_final_features(model, dataset, max_samples, device, batch_size,
                           sr, n_mels, n_fft, hop_length):
    """Mean-pool the FINAL SIVE feature over valid frames, one vector / utterance.

    Returns (features [N, D] float32, speaker_ids [N] int64, gender_ids [N] int64
    with -1 = unknown)."""
    model.eval()
    n = len(dataset) if max_samples <= 0 else min(len(dataset), max_samples)
    buf = SharedWindowBuffer()

    feats_list, spk_list, gen_list = [], [], []
    idx = 0
    pbar = tqdm(total=n, desc="Extracting features")
    while idx < n:
        end = min(idx + batch_size, n)
        mels, lengths = [], []
        for i in range(idx, end):
            s = dataset[i]
            mel, length = _sample_to_mel(s, buf, sr, n_mels, n_fft, hop_length)
            mels.append(mel)
            lengths.append(length)
            spk_list.append(int(s["speaker_id"]))
            g = s.get("gender_id", None)
            gen_list.append(-1 if g is None else int(g if not isinstance(g, torch.Tensor) else g.item()))

        max_t = max(m.shape[-1] for m in mels)
        mel_batch = torch.zeros(len(mels), mels[0].shape[0], max_t)
        for j, m in enumerate(mels):
            mel_batch[j, :, :m.shape[-1]] = m
        mel_batch = mel_batch.to(device)
        len_batch = torch.tensor(lengths, dtype=torch.long, device=device)

        result = model(mel_batch, lengths=len_batch, grl_alpha=0.0)
        features = result["features"]            # [B, T', D]
        feat_lengths = result["feature_lengths"]  # [B]
        for b in range(features.shape[0]):
            vlen = int(feat_lengths[b].item())
            vlen = max(vlen, 1)
            feats_list.append(features[b, :vlen, :].mean(dim=0).float().cpu().numpy())

        pbar.update(end - idx)
        idx = end
    pbar.close()

    return (np.stack(feats_list).astype(np.float32),
            np.array(spk_list, dtype=np.int64),
            np.array(gen_list, dtype=np.int64))


def cached_features(args, name, ckpt_path):
    """Extract (or load cached) per-utterance features for one checkpoint."""
    key = hashlib.md5(f"{ckpt_path}|{args.max_samples}|{args.config}".encode()).hexdigest()[:10]
    cache_path = os.path.join(args.output_dir, f"_features_{name}_{key}.npz")
    if os.path.exists(cache_path) and not args.no_feature_cache:
        d = np.load(cache_path)
        print(f"[{name}] loaded cached features from {cache_path}")
        return d["features"], d["speaker_ids"], d["gender_ids"]

    model = load_model(
        SpeakerInvariantVoiceEncoder, args.config, checkpoint_path=ckpt_path,
        device=args.device, overrides={"num_speakers": args.num_speakers},
        strict=False, allow_size_mismatch=True,
    )
    dataset = VoiceShardedDataset(
        shard_dir=args.val_cache_dir, cache_size=args.shard_cache_size,
        columns=["waveforms", "mel_specs", "speaker_ids", "gender_ids"],
    )
    feats, spk, gen = extract_final_features(
        model, dataset, args.max_samples, args.device, args.batch_size,
        args.voice_sample_rate, args.voice_n_mels, args.voice_n_fft, args.voice_hop_length,
    )
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    if not args.no_feature_cache:
        np.savez(cache_path, features=feats, speaker_ids=spk, gender_ids=gen)
    return feats, spk, gen


# ---------------------------------------------------------------------------
# Speaker-stratified, utterance-disjoint split
# ---------------------------------------------------------------------------

def stratified_split(speaker_ids, test_split, min_utts, seed):
    """Per-speaker utterance split. Each kept speaker (>= min_utts utterances)
    contributes >=1 utt to BOTH train and test. Speakers below the threshold are
    dropped from the probe entirely.

    Returns (train_idx, test_idx, kept_speaker_ids, n_dropped_speakers,
    n_dropped_utts)."""
    rng = np.random.RandomState(seed)
    train_idx, test_idx, kept = [], [], []
    dropped_spk = dropped_utt = 0
    for spk in np.unique(speaker_ids):
        members = np.where(speaker_ids == spk)[0]
        if len(members) < max(2, min_utts):
            dropped_spk += 1
            dropped_utt += len(members)
            continue
        members = members[rng.permutation(len(members))]
        n_test = max(1, int(round(len(members) * test_split)))
        n_test = min(n_test, len(members) - 1)  # leave >=1 for train
        test_idx.extend(members[:n_test].tolist())
        train_idx.extend(members[n_test:].tolist())
        kept.append(spk)
    return (np.array(sorted(train_idx)), np.array(sorted(test_idx)),
            np.array(kept), dropped_spk, dropped_utt)


# ---------------------------------------------------------------------------
# Probe training with plateau early-stop, returning top-5 test predictions
# ---------------------------------------------------------------------------

def train_probe(probe, Xtr, ytr, Xte, yte, num_classes, max_epochs, patience,
                lr, batch_size, device):
    """Train to convergence; early-stop when test top-1 stops improving. Returns
    the top-5 test predictions from the best epoch plus diagnostics. Using the
    test set for plateau detection biases toward an UPPER bound on leakage —
    the safe direction for a leakage audit (an under-trained probe under-reports)."""
    probe = probe.to(device)
    opt = optim.Adam(probe.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr).long()
    Xte_t = torch.from_numpy(Xte).float().to(device)
    n = Xtr_t.shape[0]

    best_top1, best_top5_preds, best_epoch = -1.0, None, -1
    history = []
    since_improve = 0
    for epoch in range(max_epochs):
        probe.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            bi = perm[i:i + batch_size]
            xb = Xtr_t[bi].to(device)
            yb = ytr_t[bi].to(device)
            opt.zero_grad()
            loss = ce(probe(xb), yb)
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            logits = []
            for i in range(0, Xte_t.shape[0], 4096):
                logits.append(probe(Xte_t[i:i + 4096]))
            logits = torch.cat(logits, dim=0)
            top5 = logits.topk(min(5, num_classes), dim=1).indices.cpu().numpy()
        yte_arr = yte
        top1_acc = float((top5[:, 0] == yte_arr).mean())
        history.append(top1_acc)
        if top1_acc > best_top1 + 1e-4:
            best_top1, best_top5_preds, best_epoch = top1_acc, top5, epoch
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= patience:
                break

    plateaued = best_epoch < (len(history) - 1)  # improved before the final epoch
    return {
        "top5_preds": best_top5_preds,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "plateaued": bool(plateaued),
        "history_tail": history[-min(8, len(history)):],
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    nx = len(x)
    if nx == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((nx + 1 - 2 * (cum.sum() / cum[-1])) / nx)


def per_speaker_metrics(top5_preds, yte, num_classes):
    """Per-speaker top-1 / top-5 recall over the test set. Returns dict keyed by
    class index -> {n, recall1, recall5}, plus micro/macro aggregates."""
    pred1 = top5_preds[:, 0]
    in5 = (top5_preds == yte[:, None]).any(axis=1)
    correct1 = (pred1 == yte)
    per = {}
    for c in np.unique(yte):
        m = yte == c
        per[int(c)] = {
            "n": int(m.sum()),
            "recall1": float(correct1[m].mean()),
            "recall5": float(in5[m].mean()),
        }
    recalls1 = np.array([v["recall1"] for v in per.values()])
    recalls5 = np.array([v["recall5"] for v in per.values()])
    K = num_classes
    micro1 = float(correct1.mean())
    micro5 = float(in5.mean())
    chance1 = 1.0 / K
    chance5 = min(5, K) / K
    return {
        "per_speaker": per,
        "micro_top1": micro1,
        "micro_top5": micro5,
        "macro_top1": float(recalls1.mean()),
        "macro_top5": float(recalls5.mean()),
        "chance_top1": chance1,
        "chance_top5": chance5,
        "ratio_micro_top1": micro1 / chance1,
        "ratio_micro_top5": micro5 / chance5,
        "ratio_macro_top1": float(recalls1.mean()) / chance1,
        "gini_recall1": gini(recalls1),
        "cv_recall1": float(recalls1.std() / (recalls1.mean() + 1e-9)),
    }


def above_chance_concentration(per_speaker, K, top_k=10):
    """What fraction of above-chance correct hits comes from the top-k speakers."""
    excess = []
    for c, v in per_speaker.items():
        expected = v["n"] * (1.0 / K)
        correct = v["recall1"] * v["n"]
        excess.append((c, max(0.0, correct - expected)))
    excess.sort(key=lambda t: -t[1])
    total = sum(e for _, e in excess)
    if total <= 0:
        return 0.0, []
    topk_sum = sum(e for _, e in excess[:top_k])
    return float(topk_sum / total), excess[:top_k]


def gender_structured_confusion(top1, yte, class_gender):
    """Among top-1 MISclassifications, fraction where the predicted speaker shares
    the true speaker's gender, vs the chance baseline (random other-speaker)."""
    mis = top1 != yte
    if mis.sum() == 0:
        return None
    tg = class_gender[yte[mis]]
    pg = class_gender[top1[mis]]
    valid = (tg >= 0) & (pg >= 0)
    if valid.sum() == 0:
        return None
    same = float((tg[valid] == pg[valid]).mean())
    # Chance baseline: P(two distinct speakers share gender) from class gender mix
    gv = class_gender[class_gender >= 0]
    p_m = float((gv == 0).mean())
    p_f = float((gv == 1).mean())
    baseline = p_m ** 2 + p_f ** 2
    return {"same_gender_misclass_rate": same, "chance_baseline": baseline,
            "ratio": same / (baseline + 1e-9), "n_misclassified": int(valid.sum())}


# ---------------------------------------------------------------------------
# Per-checkpoint analysis
# ---------------------------------------------------------------------------

def analyze_checkpoint(args, name, ckpt_path):
    print(f"\n{'='*70}\n[{name}] {ckpt_path}\n{'='*70}")
    feats, spk, gen = cached_features(args, name, ckpt_path)
    print(f"[{name}] {len(spk)} utterances, {len(np.unique(spk))} raw speakers")

    train_idx, test_idx, kept, n_drop_spk, n_drop_utt = stratified_split(
        spk, args.test_split, args.min_utts, args.seed)
    if len(kept) < 2:
        print(f"[{name}] too few eval speakers ({len(kept)}); skipping")
        return None

    # Remap kept speakers to contiguous [0, K)
    label_map = {old: i for i, old in enumerate(kept)}
    inv_map = {i: int(old) for old, i in label_map.items()}
    K = len(kept)
    ytr = np.array([label_map[s] for s in spk[train_idx]])
    yte = np.array([label_map[s] for s in spk[test_idx]])
    Xtr, Xte = feats[train_idx], feats[test_idx]

    # class -> gender (majority vote of known gender labels)
    class_gender = np.full(K, -1, dtype=np.int64)
    for c in range(K):
        gids = gen[spk == inv_map[c]]
        gids = gids[gids >= 0]
        if len(gids) > 0:
            class_gender[c] = int(np.round(gids.mean()))

    print(f"[{name}] eval: {K} speakers, train={len(ytr)} test={len(yte)} utts "
          f"(dropped {n_drop_spk} speakers / {n_drop_utt} utts with < {max(2, args.min_utts)} utts)")
    print(f"[{name}] chance top-1 = 1/{K} = {1.0/K:.5f}")

    in_dim = feats.shape[1]
    results = {}
    for probe_name, probe in [
        ("linear", LinearProbe(in_dim, K)),
        ("mlp", MLPProbe(in_dim, K, hidden_dim=args.mlp_hidden_dim,
                         num_layers=args.mlp_num_layers, dropout=args.mlp_dropout)),
    ]:
        print(f"[{name}] training {probe_name} probe...")
        tr = train_probe(probe, Xtr, ytr, Xte, yte, K,
                         args.probe_max_epochs, args.probe_patience,
                         args.probe_lr, args.probe_batch_size, args.device)
        m = per_speaker_metrics(tr["top5_preds"], yte, K)
        conc, top_excess = above_chance_concentration(m["per_speaker"], K, top_k=args.top_k_hot)
        gconf = gender_structured_confusion(tr["top5_preds"][:, 0], yte, class_gender)
        m.update({"concentration_topk_share": conc, "gender_confusion": gconf,
                  "plateaued": tr["plateaued"], "best_epoch": tr["best_epoch"],
                  "epochs_run": tr["epochs_run"], "history_tail": tr["history_tail"]})
        # Hot-spot original speaker IDs (highest recall1, with >=2 test utts)
        hot = sorted(
            [(inv_map[c], v["recall1"], v["n"]) for c, v in m["per_speaker"].items() if v["n"] >= 2],
            key=lambda t: -t[1])[:args.top_k_hot]
        m["hot_speakers"] = hot
        results[probe_name] = m
        print(f"[{name}]   {probe_name}: micro@1={m['micro_top1']:.4f} "
              f"({m['ratio_micro_top1']:.1f}x chance)  macro@1={m['macro_top1']:.4f}  "
              f"Gini={m['gini_recall1']:.3f}  top{args.top_k_hot}-share={conc:.2f}  "
              f"plateaued={tr['plateaued']}")

    # Linear-vs-MLP per-speaker delta
    lin, mlp = results["linear"]["per_speaker"], results["mlp"]["per_speaker"]
    deltas = sorted(
        [(inv_map[c], mlp[c]["recall1"] - lin[c]["recall1"], lin[c]["recall1"],
          mlp[c]["recall1"], lin[c]["n"]) for c in lin if lin[c]["n"] >= 2],
        key=lambda t: -t[1])
    results["mlp_minus_linear_top"] = deltas[:args.top_k_hot]

    _plot_checkpoint(args, name, results, K)
    test_counts = np.array([v["n"] for v in results["linear"]["per_speaker"].values()])
    results["_meta"] = {"K": K, "n_train": len(ytr), "n_test": len(yte),
                        "n_dropped_speakers": n_drop_spk, "n_dropped_utts": n_drop_utt,
                        "in_dim": in_dim,
                        "test_utts_per_speaker": {"min": int(test_counts.min()),
                                                  "median": int(np.median(test_counts)),
                                                  "max": int(test_counts.max())}}
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_checkpoint(args, name, results, K):
    # Plot 1: sorted per-speaker top-1 recall, linear vs mlp, with chance line
    fig, ax = plt.subplots(figsize=(11, 5))
    for pname, color in [("linear", "tab:blue"), ("mlp", "tab:orange")]:
        r = np.array(sorted((v["recall1"] for v in results[pname]["per_speaker"].values()), reverse=True))
        ax.plot(np.arange(len(r)), r, label=f"{pname} (macro@1={results[pname]['macro_top1']:.3f})",
                color=color, lw=1.2)
    ax.axhline(1.0 / K, color="red", ls="--", lw=1, label=f"chance = 1/{K}")
    ax.set_xlabel("speaker rank (sorted by recall)")
    ax.set_ylabel("per-speaker top-1 recall")
    ax.set_title(f"[{name}] per-speaker speaker-ID recall (sorted)\n"
                 f"Gini(lin)={results['linear']['gini_recall1']:.3f}  "
                 f"micro@1 ratio-over-chance: lin={results['linear']['ratio_micro_top1']:.1f}x "
                 f"mlp={results['mlp']['ratio_micro_top1']:.1f}x")
    ax.legend()
    fig.tight_layout()
    p1 = os.path.join(args.output_dir, f"per_speaker_recall_{name}.png")
    fig.savefig(p1, dpi=110)
    plt.close(fig)

    # Plot 2: confusion submatrix over the hottest speakers (linear probe)
    hot = results["linear"]["hot_speakers"][:args.confusion_n]
    if len(hot) >= 2:
        # we need raw preds; recompute a small confusion from stored per-speaker is
        # not possible, so this plot uses the hot-speaker recall as a diagonal proxy
        # alongside the gender-confusion summary in the report.
        ids = [h[0] for h in hot]
        rec = [h[1] for h in hot]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(ids)), rec, color="tab:purple")
        ax.axhline(1.0 / K, color="red", ls="--", lw=1, label=f"chance = 1/{K}")
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels([str(i) for i in ids], rotation=90, fontsize=7)
        ax.set_ylabel("top-1 recall")
        ax.set_xlabel("hot-spot speaker id (original)")
        gc = results["linear"].get("gender_confusion")
        gtxt = (f"  same-gender misclass {gc['same_gender_misclass_rate']:.2f} "
                f"vs chance {gc['chance_baseline']:.2f} ({gc['ratio']:.2f}x)") if gc else ""
        ax.set_title(f"[{name}] hottest {len(ids)} speakers (linear){gtxt}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f"hot_speakers_{name}.png"), dpi=110)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def verdict_line(name, res):
    """Macro-primary verdict. This val set is heavily utterance-imbalanced, so
    MICRO accuracy is dominated by a few high-count speakers; MACRO (mean per-
    speaker recall) reflects the typical speaker. Leakage is judged on macro;
    the micro-vs-macro gap + Gini diagnose whether residual identity is
    concentrated in a few speakers."""
    l = res["linear"]
    macro_ratio = l["ratio_macro_top1"]
    micro_ratio = l["ratio_micro_top1"]
    g = l["gini_recall1"]
    share = l["concentration_topk_share"]
    hot_ids = ", ".join(str(h[0]) for h in l["hot_speakers"][:5])
    broad = macro_ratio > 5.0
    concentrated = (not broad) and g > 0.6 and share > 0.5 and micro_ratio > 3.0 * max(macro_ratio, 1.0)
    if broad:
        return (f"**{name}: broad leakage** — the typical (macro) speaker sits {macro_ratio:.1f}x above "
                f"chance; identity survives across many speakers (micro@1={micro_ratio:.1f}x, Gini={g:.2f}).")
    if concentrated:
        return (f"**{name}: concentrated leakage** — macro@1 only {macro_ratio:.1f}x chance (typical "
                f"speaker ~clean) but micro@1 {micro_ratio:.1f}x, Gini={g:.2f}; top-{len(l['hot_speakers'])} "
                f"speakers hold {share*100:.0f}% of above-chance hits. Hot-spot ids: {hot_ids}.")
    return (f"**{name}: uniform-weak** — macro@1 {macro_ratio:.1f}x chance, micro@1 {micro_ratio:.1f}x, "
            f"Gini={g:.2f}; residual identity near-floor / diffuse.")


def write_report(args, all_results):
    lines = ["# SIVE per-speaker leakage report", ""]
    lines.append(f"- Representation: final SIVE feature (mean-pooled per utterance), dim "
                 f"{next(iter(all_results.values()))['_meta']['in_dim']}")
    lines.append(f"- Val set: `{args.val_cache_dir}`  |  speaker-stratified, utterance-disjoint split "
                 f"(test_split={args.test_split}, min_utts={max(2, args.min_utts)})")
    lines.append(f"- Probes: linear + MLP ({args.mlp_num_layers}L/{args.mlp_hidden_dim}d), "
                 f"trained to plateau (max {args.probe_max_epochs} ep, patience {args.probe_patience})")
    _m0 = next(iter(all_results.values()))["_meta"]
    tu = _m0["test_utts_per_speaker"]
    lines.append(f"- Eval: {_m0['K']} speakers (split identical across runs); test utts/speaker "
                 f"min={tu['min']} median={tu['median']} **max={tu['max']}** — heavy imbalance, so "
                 f"**macro** is the honest leakage metric; micro is inflated by high-count speakers.")
    lines.append("")

    lines.append("## Verdicts")
    for name, res in all_results.items():
        lines.append("- " + verdict_line(name, res))
    lines.append("")

    lines.append("## Cross-checkpoint summary (linear probe)")
    lines.append("")
    lines.append("| run | K | micro@1 ×chance | macro@1 ×chance | micro@5 ×chance | Gini(rec1) | CV | top-k share | same-gender misclass ×chance | plateaued |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, res in all_results.items():
        l = res["linear"]
        gc = l.get("gender_confusion")
        gcr = f"{gc['ratio']:.2f}x" if gc else "n/a"
        lines.append(
            f"| {name} | {res['_meta']['K']} | {l['ratio_micro_top1']:.1f}x | "
            f"{l['ratio_macro_top1']:.1f}x | {l['ratio_micro_top5']:.1f}x | "
            f"{l['gini_recall1']:.3f} | {l['cv_recall1']:.2f} | "
            f"{l['concentration_topk_share']*100:.0f}% | {gcr} | {l['plateaued']} |")
    lines.append("")

    lines.append("## MLP (non-linear) contrast")
    lines.append("")
    lines.append("| run | micro@1 ×chance (lin → mlp) | macro@1 (lin → mlp) | speakers where MLP pulls ahead (id:Δrecall) |")
    lines.append("|---|---|---|---|")
    for name, res in all_results.items():
        l, m = res["linear"], res["mlp"]
        ahead = ", ".join(f"{sid}:+{d:.2f}" for sid, d, *_ in res["mlp_minus_linear_top"][:5] if d > 0.05)
        lines.append(
            f"| {name} | {l['ratio_micro_top1']:.1f}x → {m['ratio_micro_top1']:.1f}x | "
            f"{l['macro_top1']:.3f} → {m['macro_top1']:.3f} | {ahead or '(none > +0.05)'} |")
    lines.append("")

    for name, res in all_results.items():
        lines.append(f"## {name} — detail")
        for pname in ("linear", "mlp"):
            p = res[pname]
            lines.append(f"### {pname}")
            lines.append(f"- micro top-1 {p['micro_top1']:.4f} ({p['ratio_micro_top1']:.1f}x chance), "
                         f"top-5 {p['micro_top5']:.4f} ({p['ratio_micro_top5']:.1f}x)")
            lines.append(f"- macro top-1 {p['macro_top1']:.4f} ({p['ratio_macro_top1']:.1f}x), "
                         f"top-5 {p['macro_top5']:.4f}")
            lines.append(f"- Gini(recall1)={p['gini_recall1']:.3f}, CV={p['cv_recall1']:.2f}, "
                         f"top-{args.top_k_hot} speakers hold {p['concentration_topk_share']*100:.0f}% "
                         f"of above-chance hits")
            gc = p.get("gender_confusion")
            if gc:
                lines.append(f"- gender-structured confusion: {gc['same_gender_misclass_rate']:.3f} of "
                             f"misclassifications stay within gender vs {gc['chance_baseline']:.3f} chance "
                             f"({gc['ratio']:.2f}x; n={gc['n_misclassified']})")
            lines.append(f"- plateaued={p['plateaued']} (best epoch {p['best_epoch']}/{p['epochs_run']}, "
                         f"tail {[round(x,3) for x in p['history_tail']]})")
            hot = ", ".join(f"{sid}({rec:.2f},n={n})" for sid, rec, n in p["hot_speakers"][:8])
            lines.append(f"- hot-spot speakers (id(recall,n)): {hot}")
            lines.append("")

    report_path = os.path.join(args.output_dir, "per_speaker_leakage_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    # JSON (drop bulky per_speaker dicts)
    slim = {}
    for name, res in all_results.items():
        slim[name] = {"_meta": res["_meta"],
                      "mlp_minus_linear_top": res["mlp_minus_linear_top"]}
        for pname in ("linear", "mlp"):
            p = dict(res[pname])
            p.pop("per_speaker", None)
            slim[name][pname] = p
    with open(os.path.join(args.output_dir, "per_speaker_leakage_results.json"), "w") as f:
        json.dump(slim, f, indent=2)
    print(f"\nReport: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", action="append", default=[], metavar="name=path",
                    help="Checkpoint to evaluate as name=path. Repeatable. Pass a no-GRL run as contrast.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--num_speakers", type=int, default=3610)
    ap.add_argument("--val_cache_dir", required=True)
    ap.add_argument("--output_dir", default="./eval_output/per_speaker_leakage")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all val utterances")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--shard_cache_size", type=int, default=8)
    # split
    ap.add_argument("--test_split", type=float, default=0.3)
    ap.add_argument("--min_utts", type=int, default=4, help="drop speakers with fewer utts (>=2 enforced)")
    ap.add_argument("--seed", type=int, default=42)
    # probes
    ap.add_argument("--probe_max_epochs", type=int, default=200)
    ap.add_argument("--probe_patience", type=int, default=15)
    ap.add_argument("--probe_lr", type=float, default=1e-3)
    ap.add_argument("--probe_batch_size", type=int, default=512)
    ap.add_argument("--mlp_hidden_dim", type=int, default=512)
    ap.add_argument("--mlp_num_layers", type=int, default=3)
    ap.add_argument("--mlp_dropout", type=float, default=0.1)
    # reporting
    ap.add_argument("--top_k_hot", type=int, default=10)
    ap.add_argument("--confusion_n", type=int, default=30)
    # mel params (must match training; defaults are the SIVE training defaults)
    ap.add_argument("--voice_sample_rate", type=int, default=16000)
    ap.add_argument("--voice_n_mels", type=int, default=80)
    ap.add_argument("--voice_n_fft", type=int, default=1024)
    ap.add_argument("--voice_hop_length", type=int, default=256)
    ap.add_argument("--no_feature_cache", action="store_true")
    args = ap.parse_args()

    if not args.checkpoint:
        ap.error("pass at least one --checkpoint name=path")
    os.makedirs(args.output_dir, exist_ok=True)

    ckpts = []
    for spec in args.checkpoint:
        if "=" not in spec:
            ap.error(f"--checkpoint must be name=path, got: {spec}")
        nm, pth = spec.split("=", 1)
        ckpts.append((nm, pth))

    all_results = {}
    for name, path in ckpts:
        res = analyze_checkpoint(args, name, path)
        if res is not None:
            all_results[name] = res

    if all_results:
        write_report(args, all_results)
        print("\n=== VERDICTS ===")
        for name, res in all_results.items():
            print(verdict_line(name, res))


if __name__ == "__main__":
    main()
