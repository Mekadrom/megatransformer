"""
Speaker information leakage analysis across SIVE encoder layers.

For each encoder layer (including conv_subsample), this script:
  1. Extracts utterance-level features (mean-pooled over valid frames).
  2. Trains a linear probe to classify speaker identity — higher accuracy
     means more speaker information survives at that layer.
  3. Computes congruency metrics:
     - Silhouette score:  How tightly features cluster by speaker.
     - Speaker CKA:       Representational alignment between features and
                          speaker-identity one-hot (higher = more speaker info).
     - CTC CKA:           Representational alignment between features and
                          CTC bag-of-tokens (higher = more phonetic content).
     - Pairwise cosine ratio:  Mean intra-speaker cosine similarity divided
                               by mean inter-speaker cosine similarity.

  The ideal extraction layer has low Speaker CKA / high CTC CKA — it
  retains phonetic content while discarding speaker identity.

Usage:
    # All layers:
    python -m src.scripts.eval.audio.sive.speaker_leakage_probe \
        --checkpoint_path ./checkpoints/sive \
        --config tiny_deep \
        --cache_dir ../cached_datasets/audio_sive_val \
        --num_speakers 921 \
        --device cuda \
        --max_samples 2000 \
        --probe_epochs 50 \
        --output_dir ./eval_output/speaker_leakage

    # Specific layers only (0=conv_subsample, 1..N=encoder blocks):
    python -m src.scripts.eval.audio.sive.speaker_leakage_probe \
        --checkpoint_path ./checkpoints/sive \
        --config small_deep \
        --cache_dir ../cached_datasets/audio_sive_val \
        --num_speakers 921 \
        --layers 0,4,6,8,10,12
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model.audio.sive.sive import SpeakerInvariantVoiceEncoder
from scripts.data.audio.dataset import AudioShardedDataset
from utils.model_loading_utils import load_model


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_all_layer_features(
    model: SpeakerInvariantVoiceEncoder,
    dataset: AudioShardedDataset,
    max_samples: int,
    device: str,
    batch_size: int = 32,
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """
    Run the SIVE encoder over the dataset and collect mean-pooled
    utterance-level features from every hidden layer.

    Returns:
        layer_features: list[np.ndarray]  - one (N, D) array per layer
        speaker_ids:    np.ndarray        - (N,) speaker labels
        ctc_tokens_list: list[np.ndarray] - per-utterance CTC token arrays
    """
    model.eval()
    num_samples = min(len(dataset), max_samples)

    # Accumulate per-layer lists
    layer_accum: list[list[np.ndarray]] | None = None
    speaker_ids_list: list[int] = []
    ctc_tokens_list: list[np.ndarray] = []

    idx = 0
    pbar = tqdm(total=num_samples, desc="Extracting features")
    while idx < num_samples:
        end = min(idx + batch_size, num_samples)

        mel_specs = []
        lengths = []
        for i in range(idx, end):
            sample = dataset[i]
            mel_specs.append(sample["mel_spec"])        # [n_mels, T]
            lengths.append(sample["mel_length"])
            speaker_ids_list.append(sample["speaker_id"])
            ctc_len = sample["ctc_length"]
            ctc_tokens_list.append(
                sample["ctc_tokens"][:ctc_len].numpy()
                if isinstance(sample["ctc_tokens"], torch.Tensor)
                else np.asarray(sample["ctc_tokens"][:ctc_len])
            )

        # Pad to common length
        max_t = max(m.shape[-1] for m in mel_specs)
        mel_batch = torch.zeros(len(mel_specs), mel_specs[0].shape[0], max_t)
        for j, m in enumerate(mel_specs):
            mel_batch[j, :, :m.shape[-1]] = m
        length_batch = torch.tensor(lengths, dtype=torch.long)

        mel_batch = mel_batch.to(device)
        length_batch = length_batch.to(device)

        result = model(
            mel_batch,
            lengths=length_batch,
            grl_alpha=0.0,
            return_all_hiddens=True,
        )

        all_hiddens = result["all_hiddens"]   # list of [B, T', D]
        feat_lengths = result["feature_lengths"]  # [B]

        if layer_accum is None:
            layer_accum = [[] for _ in range(len(all_hiddens))]

        for layer_idx, h in enumerate(all_hiddens):
            # h: [B, T', D]
            for b in range(h.shape[0]):
                valid_len = feat_lengths[b].item()
                # Mean pool over valid frames → (D,)
                pooled = h[b, :valid_len, :].mean(dim=0).cpu().numpy()
                layer_accum[layer_idx].append(pooled)

        pbar.update(end - idx)
        idx = end

    pbar.close()

    layer_features = [np.stack(feats, axis=0) for feats in layer_accum]
    speaker_ids = np.array(speaker_ids_list, dtype=np.int64)
    return layer_features, speaker_ids, ctc_tokens_list


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_linear_probe(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_test: np.ndarray,
    labels_test: np.ndarray,
    num_classes: int,
    epochs: int = 50,
    lr: float = 1e-2,
    batch_size: int = 256,
    device: str = "cuda",
) -> dict:
    """
    Train a single-layer linear classifier and return metrics.

    Returns dict with keys: train_acc, test_acc, test_top5_acc
    """
    in_dim = features_train.shape[1]
    probe = LinearProbe(in_dim, num_classes).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.from_numpy(features_train).float(),
        torch.from_numpy(labels_train).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # ---- train ----
    probe.train()
    for _ in range(epochs):
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = probe(x_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ---- evaluate ----
    probe.eval()
    with torch.no_grad():
        # Train accuracy (full pass)
        x_tr = torch.from_numpy(features_train).float().to(device)
        y_tr = torch.from_numpy(labels_train).long().to(device)
        train_preds = probe(x_tr).argmax(dim=-1)
        train_acc = (train_preds == y_tr).float().mean().item()

        # Test accuracy
        x_te = torch.from_numpy(features_test).float().to(device)
        y_te = torch.from_numpy(labels_test).long().to(device)
        test_logits = probe(x_te)
        test_preds = test_logits.argmax(dim=-1)
        test_acc = (test_preds == y_te).float().mean().item()

        # Top-5 accuracy
        if num_classes >= 5:
            top5 = test_logits.topk(5, dim=-1).indices
            test_top5_acc = (top5 == y_te.unsqueeze(-1)).any(dim=-1).float().mean().item()
        else:
            test_top5_acc = test_acc

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "test_top5_acc": test_top5_acc,
    }


# ---------------------------------------------------------------------------
# Congruency metrics
# ---------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute Linear CKA (Centered Kernel Alignment) between two
    representation matrices X (N, D1) and Y (N, D2).

    CKA measures how similar two representational spaces are, invariant to
    orthogonal transformations and isotropic scaling.
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    XtX = X.T @ X       # (D1, D1)
    YtY = Y.T @ Y       # (D2, D2)
    XtY = X.T @ Y       # (D1, D2)

    hsic_xy = np.sum(XtY ** 2)
    hsic_xx = np.sum(XtX ** 2)
    hsic_yy = np.sum(YtY ** 2)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def silhouette_score_sampled(
    features: np.ndarray,
    labels: np.ndarray,
    max_samples: int = 5000,
    seed: int = 42,
) -> float:
    """
    Compute silhouette score with optional subsampling for speed.
    Returns NaN if fewer than 2 unique labels.
    """
    from sklearn.metrics import silhouette_score

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return float("nan")

    if len(features) > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(features), max_samples, replace=False)
        features = features[idx]
        labels = labels[idx]

    # Filter labels with at least 2 samples (silhouette requirement)
    unique, counts = np.unique(labels, return_counts=True)
    valid_labels = set(unique[counts >= 2])
    if len(valid_labels) < 2:
        return float("nan")
    mask = np.array([l in valid_labels for l in labels])
    features = features[mask]
    labels = labels[mask]

    return float(silhouette_score(features, labels, metric="cosine"))


def pairwise_cosine_ratio(
    features: np.ndarray,
    labels: np.ndarray,
    max_pairs: int = 200_000,
    seed: int = 42,
) -> dict:
    """
    Estimate the ratio of mean intra-speaker cosine similarity to mean
    inter-speaker cosine similarity.

    A ratio >> 1 means utterances from the same speaker are much more
    similar than utterances from different speakers → speaker information
    is preserved.  A ratio near 1 means features are speaker-invariant.

    Returns dict with: intra_cos, inter_cos, ratio
    """
    from sklearn.preprocessing import normalize

    rng = np.random.RandomState(seed)
    features_norm = normalize(features, axis=1)  # L2 normalize rows
    n = len(features)

    # Sample random pairs
    num_pairs = min(max_pairs, n * (n - 1) // 2)
    i_idx = rng.randint(0, n, size=num_pairs)
    j_idx = rng.randint(0, n, size=num_pairs)
    # Avoid self-pairs
    same = i_idx == j_idx
    j_idx[same] = (j_idx[same] + 1) % n

    cos_sims = np.einsum("ij,ij->i", features_norm[i_idx], features_norm[j_idx])
    same_speaker = labels[i_idx] == labels[j_idx]

    if same_speaker.sum() == 0 or (~same_speaker).sum() == 0:
        return {"intra_cos": float("nan"), "inter_cos": float("nan"), "ratio": float("nan")}

    intra_cos = float(cos_sims[same_speaker].mean())
    inter_cos = float(cos_sims[~same_speaker].mean())
    ratio = intra_cos / inter_cos if abs(inter_cos) > 1e-8 else float("nan")

    return {"intra_cos": intra_cos, "inter_cos": inter_cos, "ratio": ratio}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def generate_plots(results: list[dict], layer_names: list[str], output_dir: str):
    """Generate summary plots for probe accuracy and congruency metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = list(range(len(results)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Speaker Information Leakage Across SIVE Encoder Layers", fontsize=14, fontweight="bold")

    # 1) Probe accuracy
    ax = axes[0, 0]
    train_accs = [r["probe"]["train_acc"] for r in results]
    test_accs = [r["probe"]["test_acc"] for r in results]
    top5_accs = [r["probe"]["test_top5_acc"] for r in results]
    ax.plot(layers, train_accs, "o-", label="Train Acc", color="tab:blue", alpha=0.7)
    ax.plot(layers, test_accs, "s-", label="Test Acc", color="tab:orange")
    ax.plot(layers, top5_accs, "^--", label="Test Top-5 Acc", color="tab:green", alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Linear Probe Speaker Classification")
    ax.set_xticks(layers)
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2) Silhouette score
    ax = axes[0, 1]
    sil_scores = [r["silhouette"] for r in results]
    ax.bar(layers, sil_scores, color="tab:purple", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Silhouette Score (cosine)")
    ax.set_title("Feature Clustering by Speaker")
    ax.set_xticks(layers)
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=7)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # 3) CKA — Speaker vs CTC
    ax = axes[1, 0]
    cka_spk = [r["cka_speaker"] for r in results]
    cka_ctc = [r["cka_ctc"] for r in results]
    x = np.array(layers)
    w = 0.35
    ax.bar(x - w / 2, cka_spk, w, label="Speaker", color="tab:red", alpha=0.8)
    ax.bar(x + w / 2, cka_ctc, w, label="CTC", color="tab:olive", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Linear CKA")
    ax.set_title("CKA: Speaker Identity vs CTC Content")
    ax.set_xticks(layers)
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4) Cosine ratio
    ax = axes[1, 1]
    intra = [r["cosine"]["intra_cos"] for r in results]
    inter = [r["cosine"]["inter_cos"] for r in results]
    ratios = [r["cosine"]["ratio"] for r in results]
    x = np.array(layers)
    width = 0.3
    ax.bar(x - width / 2, intra, width, label="Intra-speaker", color="tab:cyan", alpha=0.8)
    ax.bar(x + width / 2, inter, width, label="Inter-speaker", color="tab:pink", alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(layers, ratios, "kD-", label="Ratio", markersize=5)
    ax2.set_ylabel("Intra / Inter Ratio")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Pairwise Cosine: Intra vs Inter Speaker")
    ax.set_xticks(layers)
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=7)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "speaker_leakage_summary.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved summary plot to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_layer_names(num_encoder_blocks: int) -> list[str]:
    names = ["conv_subsample"]
    for i in range(num_encoder_blocks):
        names.append(f"encoder_block_{i + 1}")
    return names


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate speaker information leakage per SIVE encoder layer"
    )
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to SIVE checkpoint directory")
    parser.add_argument("--config", type=str, required=True,
                        help="SIVE config name (e.g. tiny_deep, small)")
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Path to cached audio dataset with mel_specs and speaker_ids")
    parser.add_argument("--num_speakers", type=int, required=True,
                        help="Number of speakers (must match training)")
    parser.add_argument("--speaker_pooling", type=str, default="attentive_statistics",
                        help="Speaker pooling method (must match training)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="Max dataset samples for feature extraction")
    parser.add_argument("--extraction_batch_size", type=int, default=32,
                        help="Batch size for feature extraction")
    parser.add_argument("--test_split", type=float, default=0.2,
                        help="Fraction held out for probe evaluation")
    parser.add_argument("--probe_epochs", type=int, default=50,
                        help="Training epochs for each linear probe")
    parser.add_argument("--probe_lr", type=float, default=1e-2,
                        help="Learning rate for linear probes")
    parser.add_argument("--probe_batch_size", type=int, default=256,
                        help="Batch size for probe training")
    parser.add_argument("--output_dir", type=str, default="./eval_output/speaker_leakage",
                        help="Directory for results and plots")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices to evaluate (0=conv_subsample, "
                             "1..N=encoder blocks). E.g. '0,4,6,8,10,12'. "
                             "Default: all layers.")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip generating plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 1. Load model ----
    print(f"Loading SIVE model (config={args.config}) from {args.checkpoint_path}...")
    model = load_model(
        SpeakerInvariantVoiceEncoder,
        args.config,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        strict=False,
        overrides={
            "num_speakers": args.num_speakers,
            "speaker_pooling": args.speaker_pooling,
        },
    )
    model.eval()

    # ---- 2. Load dataset ----
    print(f"Loading dataset from {args.cache_dir}...")
    dataset = AudioShardedDataset(
        args.cache_dir, columns=["mel_specs", "speaker_ids", "ctc_tokens"],
    )

    # ---- 3. Extract features ----
    print(f"Extracting features (max {args.max_samples} samples)...")
    layer_features, speaker_ids, ctc_tokens_list = extract_all_layer_features(
        model, dataset, args.max_samples, args.device,
        batch_size=args.extraction_batch_size,
    )
    num_total_layers = len(layer_features)
    all_layer_names = get_layer_names(num_total_layers - 1)
    num_unique_speakers = len(np.unique(speaker_ids))
    print(f"Extracted {num_total_layers} layers, {len(speaker_ids)} samples, "
          f"{num_unique_speakers} unique speakers")

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # ---- 3b. Filter to requested layers ----
    if args.layers is not None:
        eval_layer_indices = sorted(set(
            int(x.strip()) for x in args.layers.split(",")
        ))
        invalid = [i for i in eval_layer_indices if i < 0 or i >= num_total_layers]
        if invalid:
            print(f"Warning: layer indices {invalid} out of range "
                  f"[0, {num_total_layers - 1}], ignoring them.")
            eval_layer_indices = [i for i in eval_layer_indices if 0 <= i < num_total_layers]
    else:
        eval_layer_indices = list(range(num_total_layers))

    eval_layer_features = [layer_features[i] for i in eval_layer_indices]
    eval_layer_names = [all_layer_names[i] for i in eval_layer_indices]
    print(f"Evaluating {len(eval_layer_indices)} layers: "
          f"{', '.join(f'{i} ({all_layer_names[i]})' for i in eval_layer_indices)}")

    # ---- 4. Train/test split (stratified by speaker) ----
    rng = np.random.RandomState(args.seed)
    n = len(speaker_ids)
    indices = rng.permutation(n)
    split = int(n * (1 - args.test_split))
    train_idx, test_idx = indices[:split], indices[split:]

    labels_train = speaker_ids[train_idx]
    labels_test = speaker_ids[test_idx]

    # Remap labels to contiguous range for the probe (only speakers seen in train)
    all_labels = np.concatenate([labels_train, labels_test])
    unique_labels = np.unique(all_labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels_train_mapped = np.array([label_map[l] for l in labels_train])
    labels_test_mapped = np.array([label_map[l] for l in labels_test])
    num_probe_classes = len(unique_labels)

    # ---- 5. Per-layer evaluation ----
    results = []
    print(f"\n{'Layer':<22} {'Probe Test':>10} {'Top-5':>7} {'Silhouette':>11} "
          f"{'Spk CKA':>8} {'CTC CKA':>8} {'Cos Ratio':>10}")
    print("-" * 85)

    # Build speaker one-hot for CKA
    speaker_onehot = np.zeros((n, num_probe_classes), dtype=np.float32)
    all_labels_mapped = np.array([label_map[l] for l in speaker_ids])
    for i, lbl in enumerate(all_labels_mapped):
        speaker_onehot[i, lbl] = 1.0

    # Build CTC bag-of-tokens for CKA (normalized token count histogram)
    vocab_size = int(max(tok.max() for tok in ctc_tokens_list if len(tok) > 0)) + 1
    ctc_bag = np.zeros((n, vocab_size), dtype=np.float32)
    for i, toks in enumerate(ctc_tokens_list):
        if len(toks) > 0:
            np.add.at(ctc_bag[i], toks.astype(np.intp), 1.0)
            total = ctc_bag[i].sum()
            if total > 0:
                ctc_bag[i] /= total
    print(f"CTC bag-of-tokens: vocab_size={vocab_size}")

    for eval_idx, (orig_layer_idx, feats, layer_name) in enumerate(
        zip(eval_layer_indices, eval_layer_features, eval_layer_names)
    ):
        feats_train = feats[train_idx]
        feats_test = feats[test_idx]

        # Linear probe
        probe_results = train_linear_probe(
            feats_train, labels_train_mapped,
            feats_test, labels_test_mapped,
            num_classes=num_probe_classes,
            epochs=args.probe_epochs,
            lr=args.probe_lr,
            batch_size=args.probe_batch_size,
            device=args.device,
        )

        # Silhouette score (on full set, subsampled internally)
        sil = silhouette_score_sampled(feats, speaker_ids)

        # Linear CKA with speaker identity
        cka_speaker = linear_cka(feats, speaker_onehot)

        # Linear CKA with CTC bag-of-tokens
        cka_ctc = linear_cka(feats, ctc_bag)

        # Pairwise cosine ratio
        cos = pairwise_cosine_ratio(feats, speaker_ids)

        layer_result = {
            "layer_idx": orig_layer_idx,
            "layer_name": layer_name,
            "probe": probe_results,
            "silhouette": sil,
            "cka_speaker": cka_speaker,
            "cka_ctc": cka_ctc,
            "cosine": cos,
        }
        results.append(layer_result)

        print(f"{layer_name:<22} "
              f"{probe_results['test_acc']:>10.4f} "
              f"{probe_results['test_top5_acc']:>7.4f} "
              f"{sil:>11.4f} "
              f"{cka_speaker:>8.4f} "
              f"{cka_ctc:>8.4f} "
              f"{cos['ratio']:>10.4f}")

    # ---- 6. Save results ----
    results_path = os.path.join(args.output_dir, "speaker_leakage_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "config": args.config,
                "checkpoint_path": args.checkpoint_path,
                "num_samples": len(speaker_ids),
                "num_unique_speakers": num_unique_speakers,
                "num_probe_classes": num_probe_classes,
                "layers": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_path}")

    # ---- 7. Plots ----
    if not args.no_plot:
        generate_plots(results, eval_layer_names, args.output_dir)

    # ---- 8. Summary ----
    best_layer = min(results, key=lambda r: r["probe"]["test_acc"])
    worst_layer = max(results, key=lambda r: r["probe"]["test_acc"])
    print(f"\nLeast speaker leakage:  {best_layer['layer_name']} "
          f"(probe acc = {best_layer['probe']['test_acc']:.4f})")
    print(f"Most speaker leakage:   {worst_layer['layer_name']} "
          f"(probe acc = {worst_layer['probe']['test_acc']:.4f})")


if __name__ == "__main__":
    main()
