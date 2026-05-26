"""
Speaker information leakage analysis across SIVE encoder layers.

For each encoder layer (including conv_subsample), this script:
  1. Extracts utterance-level features (mean-pooled over valid frames).
  2. Trains a linear probe to classify speaker identity — higher accuracy
     means more speaker information survives at that layer. Reports
     top-1 / top-5 / top-10 test accuracy.
  3. Trains a linear binary gender probe — reports plain test accuracy and
     balanced binary accuracy (mean of male/female recall) so the metric is
     not dominated by whichever class happens to be the eval-set majority.
  4. Computes congruency metrics:
     - Silhouette score:  How tightly features cluster by speaker.
     - Speaker CKA:       Representational alignment between features and
                          speaker-identity one-hot (higher = more speaker info).
     - CTC CKA:           Representational alignment between features and
                          CTC bag-of-tokens (higher = more phonetic content).
     - Pairwise cosine ratio:  Mean intra-speaker cosine similarity divided
                               by mean inter-speaker cosine similarity.

  The ideal extraction layer has low Speaker CKA / high CTC CKA — it
  retains phonetic content while discarding speaker identity.

Additionally, a configurable non-linear (MLP) probe is trained on
mean+std time-pooled features from a single chosen layer (default: 10).
This catches non-linear speaker / gender leakage that a linear probe
misses.

Usage:
    # All layers + MLP probe at default layer 10:
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

    # Custom MLP probe architecture:
    python -m src.scripts.eval.audio.sive.speaker_leakage_probe \
        ... \
        --mlp_probe_layer 8 --mlp_probe_num_layers 6 \
        --mlp_probe_hidden_dim 1024 --mlp_probe_dropout 0.2
"""

import argparse
import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model.voice.sive.sive import SpeakerInvariantVoiceEncoder
from scripts.data.voice.dataset import VoiceShardedDataset
from utils.model_loading_utils import load_model


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_all_layer_features(
    model: SpeakerInvariantVoiceEncoder,
    dataset: VoiceShardedDataset,
    max_samples: int,
    device: str,
    batch_size: int = 32,
    mlp_probe_layer: int = 10,
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """
    Run the SIVE encoder over the dataset and collect:
      - utterance-level mean-pooled features from every hidden layer
        (linear-probe input)
      - utterance-level mean+std concat-pooled features from the chosen
        ``mlp_probe_layer`` only (MLP-probe input). Returns None if the
        layer index is out of range.
      - speaker IDs (per utterance)
      - CTC token sequences (per utterance, variable length)
      - gender IDs (per utterance, -1 if shard has no gender_ids field)

    Returns:
        layer_features:   list[np.ndarray]   one (N, D)   per layer
        speaker_ids:      np.ndarray         (N,) int64
        ctc_tokens_list:  list[np.ndarray]   per-utterance token arrays
        gender_ids:       np.ndarray         (N,) int64; -1 = unknown/missing
        mlp_meanstd:      np.ndarray | None  (N, 2*D) at mlp_probe_layer
    """
    model.eval()
    num_samples = min(len(dataset), max_samples)

    # Accumulate per-layer lists
    layer_accum: list[list[np.ndarray]] | None = None
    speaker_ids_list: list[int] = []
    ctc_tokens_list: list[np.ndarray] = []
    gender_ids_list: list[int] = []
    mlp_meanstd_accum: list[np.ndarray] = []

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
            # gender_id is optional — present only when preprocessing was
            # run with --gender_column or --gender_lookup_path.
            g = sample.get("gender_id", None)
            if g is None:
                gender_ids_list.append(-1)
            else:
                gender_ids_list.append(int(g) if not isinstance(g, torch.Tensor) else int(g.item()))

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

        # Mean+std pool at the chosen MLP-probe layer
        if 0 <= mlp_probe_layer < len(all_hiddens):
            h_mlp = all_hiddens[mlp_probe_layer]  # [B, T', D]
            for b in range(h_mlp.shape[0]):
                valid_len = feat_lengths[b].item()
                valid_h = h_mlp[b, :valid_len, :]
                if valid_h.shape[0] >= 2:
                    mean_p = valid_h.mean(dim=0)
                    std_p = valid_h.std(dim=0)
                elif valid_h.shape[0] == 1:
                    mean_p = valid_h.squeeze(0)
                    std_p = torch.zeros_like(mean_p)
                else:
                    mean_p = torch.zeros(h_mlp.shape[-1], device=h_mlp.device)
                    std_p = torch.zeros_like(mean_p)
                ms = torch.cat([mean_p, std_p], dim=0).cpu().numpy()
                mlp_meanstd_accum.append(ms)

        pbar.update(end - idx)
        idx = end

    pbar.close()

    layer_features = [np.stack(feats, axis=0) for feats in layer_accum]
    speaker_ids = np.array(speaker_ids_list, dtype=np.int64)
    gender_ids = np.array(gender_ids_list, dtype=np.int64)
    mlp_meanstd = np.stack(mlp_meanstd_accum, axis=0) if mlp_meanstd_accum else None
    return layer_features, speaker_ids, ctc_tokens_list, gender_ids, mlp_meanstd


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPProbe(nn.Module):
    """
    Configurable non-linear probe.

    Architecture (num_layers >= 2):
        in_dim → [Linear → GELU → Dropout] × (num_layers - 1) → Linear(num_classes)

    With num_layers == 1, this reduces to a single Linear (equivalent to
    LinearProbe but exercised through the same training loop).
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        layers: list[nn.Module] = []
        d_prev = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(d_prev, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_prev = hidden_dim
        layers.append(nn.Linear(d_prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_probe_loop(
    probe: nn.Module,
    features_train: np.ndarray,
    labels_train: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
) -> None:
    """Run the training loop in-place on ``probe``. Standard CE + Adam."""
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_ds = TensorDataset(
        torch.from_numpy(features_train).float(),
        torch.from_numpy(labels_train).long(),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    probe.train()
    for _ in range(epochs):
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = probe(x_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def _classification_metrics(
    probe: nn.Module,
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_test: np.ndarray,
    labels_test: np.ndarray,
    num_classes: int,
    device: str,
) -> tuple[dict, np.ndarray]:
    """Compute train/test accuracy + top-K accuracies. Returns metrics dict
    and the test prediction array (used by callers that need per-class recall).
    """
    probe.eval()
    with torch.no_grad():
        x_tr = torch.from_numpy(features_train).float().to(device)
        y_tr = torch.from_numpy(labels_train).long().to(device)
        train_acc = (probe(x_tr).argmax(dim=-1) == y_tr).float().mean().item()

        x_te = torch.from_numpy(features_test).float().to(device)
        y_te = torch.from_numpy(labels_test).long().to(device)
        test_logits = probe(x_te)
        test_preds = test_logits.argmax(dim=-1)
        test_acc = (test_preds == y_te).float().mean().item()

        def topk_acc(k: int) -> float:
            # If num_classes < k, top-k is trivially the same as top-num_classes,
            # which equals 1.0 — fall back to top-1 to keep the metric meaningful.
            if num_classes < k:
                return test_acc
            topk = test_logits.topk(k, dim=-1).indices
            return (topk == y_te.unsqueeze(-1)).any(dim=-1).float().mean().item()

        test_top5 = topk_acc(5)
        test_top10 = topk_acc(10)
        preds_np = test_preds.cpu().numpy()

    return (
        {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "test_top5_acc": test_top5,
            "test_top10_acc": test_top10,
        },
        preds_np,
    )


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
    Train a single-layer linear classifier and return train_acc, test_acc,
    test_top5_acc, test_top10_acc.
    """
    probe = LinearProbe(features_train.shape[1], num_classes).to(device)
    _train_probe_loop(
        probe, features_train, labels_train,
        epochs=epochs, lr=lr, batch_size=batch_size, device=device,
    )
    metrics, _ = _classification_metrics(
        probe, features_train, labels_train, features_test, labels_test,
        num_classes, device,
    )
    return metrics


def train_mlp_probe(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_test: np.ndarray,
    labels_test: np.ndarray,
    num_classes: int,
    hidden_dim: int = 512,
    num_layers: int = 4,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: str = "cuda",
) -> dict:
    """Train a non-linear MLP probe and return the same metrics dict shape."""
    probe = MLPProbe(
        in_dim=features_train.shape[1],
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    _train_probe_loop(
        probe, features_train, labels_train,
        epochs=epochs, lr=lr, batch_size=batch_size, device=device,
    )
    metrics, _ = _classification_metrics(
        probe, features_train, labels_train, features_test, labels_test,
        num_classes, device,
    )
    return metrics


def train_gender_probe(
    features_train: np.ndarray,
    gender_train: np.ndarray,
    features_test: np.ndarray,
    gender_test: np.ndarray,
    probe_type: str = "linear",
    hidden_dim: int = 512,
    num_layers: int = 4,
    dropout: float = 0.1,
    epochs: int = 50,
    lr: float = 1e-2,
    batch_size: int = 256,
    device: str = "cuda",
) -> Optional[dict]:
    """
    Train a binary gender classifier (0=male, 1=female). Filters out -1
    (unknown) before training/eval. Returns ``None`` if either split has no
    labeled samples.

    The reported balanced accuracy is the unweighted mean of per-class
    recall, which equals plain accuracy when the eval set is exactly
    balanced and downweights the majority class otherwise.
    """
    train_mask = gender_train != -1
    test_mask = gender_test != -1
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return None

    f_tr = features_train[train_mask]
    g_tr = gender_train[train_mask]
    f_te = features_test[test_mask]
    g_te = gender_test[test_mask]

    # Need both classes in train for a meaningful binary probe.
    if len(np.unique(g_tr)) < 2:
        return None

    in_dim = f_tr.shape[1]
    if probe_type == "linear":
        probe = LinearProbe(in_dim, 2).to(device)
    else:
        probe = MLPProbe(
            in_dim=in_dim, num_classes=2,
            hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
        ).to(device)

    _train_probe_loop(
        probe, f_tr, g_tr,
        epochs=epochs, lr=lr, batch_size=batch_size, device=device,
    )
    metrics, preds_te = _classification_metrics(
        probe, f_tr, g_tr, f_te, g_te, num_classes=2, device=device,
    )

    # Per-class recall on the test split
    recalls: list[float] = []
    for cls in (0, 1):
        cls_mask = g_te == cls
        if cls_mask.sum() == 0:
            recalls.append(float("nan"))
        else:
            recalls.append(float((preds_te[cls_mask] == cls).mean()))
    valid_recalls = [r for r in recalls if not np.isnan(r)]
    balanced_acc = float(np.mean(valid_recalls)) if valid_recalls else float("nan")

    return {
        **metrics,
        "test_balanced_acc": balanced_acc,
        "test_recall_male": recalls[0],
        "test_recall_female": recalls[1],
        "train_count_male": int((g_tr == 0).sum()),
        "train_count_female": int((g_tr == 1).sum()),
        "test_count_male": int((g_te == 0).sum()),
        "test_count_female": int((g_te == 1).sum()),
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

    # 1) Probe accuracy (top-1 / top-5 / top-10)
    ax = axes[0, 0]
    train_accs = [r["probe"]["train_acc"] for r in results]
    test_accs = [r["probe"]["test_acc"] for r in results]
    top5_accs = [r["probe"]["test_top5_acc"] for r in results]
    top10_accs = [r["probe"]["test_top10_acc"] for r in results]
    ax.plot(layers, train_accs, "o-", label="Train Acc", color="tab:blue", alpha=0.7)
    ax.plot(layers, test_accs, "s-", label="Test Top-1", color="tab:orange")
    ax.plot(layers, top5_accs, "^--", label="Test Top-5", color="tab:green", alpha=0.7)
    ax.plot(layers, top10_accs, "v--", label="Test Top-10", color="tab:brown", alpha=0.7)
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


def generate_gender_plot(results: list[dict], layer_names: list[str], output_dir: str):
    """Generate per-layer gender probe plot. Skipped if gender data absent."""
    have_gender = any(r.get("gender_probe") is not None for r in results)
    if not have_gender:
        print("Skipping gender plot — no gender labels found in shards.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = list(range(len(results)))
    test_acc = [r["gender_probe"]["test_acc"] if r.get("gender_probe") else float("nan") for r in results]
    bal_acc = [r["gender_probe"]["test_balanced_acc"] if r.get("gender_probe") else float("nan") for r in results]
    recall_m = [r["gender_probe"]["test_recall_male"] if r.get("gender_probe") else float("nan") for r in results]
    recall_f = [r["gender_probe"]["test_recall_female"] if r.get("gender_probe") else float("nan") for r in results]

    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    fig.suptitle("Gender Linear-Probe Leakage Across SIVE Layers", fontsize=13, fontweight="bold")
    ax.plot(layers, test_acc, "s-", label="Test Acc", color="tab:orange")
    ax.plot(layers, bal_acc, "o-", label="Balanced Acc", color="tab:blue")
    ax.plot(layers, recall_m, "^--", label="Recall Male", color="tab:cyan", alpha=0.7)
    ax.plot(layers, recall_f, "v--", label="Recall Female", color="tab:pink", alpha=0.7)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.6, label="chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy / Recall")
    ax.set_xticks(layers)
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "gender_leakage_summary.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved gender plot to {path}")


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

    # MLP probe (non-linear, mean+std pooled at one chosen layer)
    parser.add_argument("--mlp_probe_layer", type=int, default=10,
                        help="SIVE encoder layer index for the MLP probe input. "
                             "0=conv_subsample, 1..N=encoder blocks. Set to -1 to disable.")
    parser.add_argument("--mlp_probe_hidden_dim", type=int, default=512,
                        help="Hidden dim of MLP probe layers")
    parser.add_argument("--mlp_probe_num_layers", type=int, default=4,
                        help="Total layer count for the MLP probe (>=1; 1 ≡ linear)")
    parser.add_argument("--mlp_probe_dropout", type=float, default=0.1,
                        help="Dropout rate inside MLP probe")
    parser.add_argument("--mlp_probe_epochs", type=int, default=50,
                        help="Training epochs for the MLP probe (default: same as linear)")
    parser.add_argument("--mlp_probe_lr", type=float, default=1e-3,
                        help="Learning rate for MLP probe (default: 1e-3, lower than linear)")
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
        allow_size_mismatch=True,
        overrides={
            "num_speakers": args.num_speakers,
            "speaker_pooling": args.speaker_pooling,
        },
    )
    model.eval()

    # ---- 2. Load dataset ----
    print(f"Loading dataset from {args.cache_dir}...")
    dataset = VoiceShardedDataset(
        args.cache_dir,
        columns=["mel_specs", "speaker_ids", "ctc_tokens", "gender_ids"],
    )

    # ---- 3. Extract features ----
    print(f"Extracting features (max {args.max_samples} samples, mlp_probe_layer={args.mlp_probe_layer})...")
    layer_features, speaker_ids, ctc_tokens_list, gender_ids, mlp_meanstd = extract_all_layer_features(
        model, dataset, args.max_samples, args.device,
        batch_size=args.extraction_batch_size,
        mlp_probe_layer=args.mlp_probe_layer,
    )
    num_total_layers = len(layer_features)
    all_layer_names = get_layer_names(num_total_layers - 1)
    num_unique_speakers = len(np.unique(speaker_ids))
    n_gender_known = int((gender_ids != -1).sum())
    n_male = int((gender_ids == 0).sum())
    n_female = int((gender_ids == 1).sum())
    print(f"Extracted {num_total_layers} layers, {len(speaker_ids)} samples, "
          f"{num_unique_speakers} unique speakers")
    if n_gender_known > 0:
        print(f"Gender labels: {n_gender_known}/{len(gender_ids)} known "
              f"({n_male} male, {n_female} female, "
              f"{len(gender_ids) - n_gender_known} unknown)")
    else:
        print("Gender labels: none (shards have no gender_ids field — "
              "rerun preprocessing with --gender_column or --gender_lookup_path)")
    if mlp_meanstd is not None:
        mlp_layer_name = (
            all_layer_names[args.mlp_probe_layer]
            if 0 <= args.mlp_probe_layer < num_total_layers
            else f"layer_{args.mlp_probe_layer}"
        )
        print(f"MLP-probe input: layer {args.mlp_probe_layer} ({mlp_layer_name}), "
              f"feature dim = {mlp_meanstd.shape[1]} (mean+std concat)")
    else:
        if args.mlp_probe_layer >= 0:
            print(f"Warning: --mlp_probe_layer {args.mlp_probe_layer} is out of range "
                  f"[0, {num_total_layers - 1}]; MLP probe will be skipped.")

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

    gender_train = gender_ids[train_idx]
    gender_test = gender_ids[test_idx]

    # ---- 5. Per-layer evaluation ----
    results = []
    print(f"\n{'Layer':<22} {'Spk@1':>7} {'Spk@5':>7} {'Spk@10':>7} {'Sil':>7} "
          f"{'SpkCKA':>7} {'CTCCKA':>7} {'CosR':>6} {'GdrAcc':>7} {'GdrBal':>7}")
    print("-" * 92)

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

        # Linear speaker probe
        probe_results = train_linear_probe(
            feats_train, labels_train_mapped,
            feats_test, labels_test_mapped,
            num_classes=num_probe_classes,
            epochs=args.probe_epochs,
            lr=args.probe_lr,
            batch_size=args.probe_batch_size,
            device=args.device,
        )

        # Linear gender probe (skipped if no gender labels)
        gender_results = train_gender_probe(
            feats_train, gender_train,
            feats_test, gender_test,
            probe_type="linear",
            epochs=args.probe_epochs,
            lr=args.probe_lr,
            batch_size=args.probe_batch_size,
            device=args.device,
        )

        # Silhouette score (on full set, subsampled internally)
        sil = silhouette_score_sampled(feats, speaker_ids, max_samples=args.max_samples)

        # Linear CKA with speaker identity / CTC
        cka_speaker = linear_cka(feats, speaker_onehot)
        cka_ctc = linear_cka(feats, ctc_bag)

        # Pairwise cosine ratio
        cos = pairwise_cosine_ratio(feats, speaker_ids)

        layer_result = {
            "layer_idx": orig_layer_idx,
            "layer_name": layer_name,
            "probe": probe_results,
            "gender_probe": gender_results,
            "silhouette": sil,
            "cka_speaker": cka_speaker,
            "cka_ctc": cka_ctc,
            "cosine": cos,
        }
        results.append(layer_result)

        gdr_acc = gender_results["test_acc"] if gender_results else float("nan")
        gdr_bal = gender_results["test_balanced_acc"] if gender_results else float("nan")
        print(f"{layer_name:<22} "
              f"{probe_results['test_acc']:>7.4f} "
              f"{probe_results['test_top5_acc']:>7.4f} "
              f"{probe_results['test_top10_acc']:>7.4f} "
              f"{sil:>7.4f} "
              f"{cka_speaker:>7.4f} "
              f"{cka_ctc:>7.4f} "
              f"{cos['ratio']:>6.3f} "
              f"{gdr_acc:>7.4f} "
              f"{gdr_bal:>7.4f}")

    # ---- 6. MLP probe (single chosen layer, mean+std features) ----
    mlp_summary = None
    if mlp_meanstd is not None and 0 <= args.mlp_probe_layer < num_total_layers:
        print(f"\nTraining MLP probes at layer {args.mlp_probe_layer} "
              f"({all_layer_names[args.mlp_probe_layer]}) — "
              f"{args.mlp_probe_num_layers} layers, hidden={args.mlp_probe_hidden_dim}, "
              f"dropout={args.mlp_probe_dropout}, in_dim={mlp_meanstd.shape[1]}")
        mlp_train = mlp_meanstd[train_idx]
        mlp_test = mlp_meanstd[test_idx]

        mlp_speaker = train_mlp_probe(
            mlp_train, labels_train_mapped,
            mlp_test, labels_test_mapped,
            num_classes=num_probe_classes,
            hidden_dim=args.mlp_probe_hidden_dim,
            num_layers=args.mlp_probe_num_layers,
            dropout=args.mlp_probe_dropout,
            epochs=args.mlp_probe_epochs,
            lr=args.mlp_probe_lr,
            batch_size=args.probe_batch_size,
            device=args.device,
        )

        mlp_gender = train_gender_probe(
            mlp_train, gender_train,
            mlp_test, gender_test,
            probe_type="mlp",
            hidden_dim=args.mlp_probe_hidden_dim,
            num_layers=args.mlp_probe_num_layers,
            dropout=args.mlp_probe_dropout,
            epochs=args.mlp_probe_epochs,
            lr=args.mlp_probe_lr,
            batch_size=args.probe_batch_size,
            device=args.device,
        )

        mlp_summary = {
            "layer_idx": args.mlp_probe_layer,
            "layer_name": all_layer_names[args.mlp_probe_layer],
            "feature_dim": int(mlp_meanstd.shape[1]),
            "hidden_dim": args.mlp_probe_hidden_dim,
            "num_layers": args.mlp_probe_num_layers,
            "dropout": args.mlp_probe_dropout,
            "speaker": mlp_speaker,
            "gender": mlp_gender,
        }

        print(f"  Speaker MLP: top1={mlp_speaker['test_acc']:.4f}  "
              f"top5={mlp_speaker['test_top5_acc']:.4f}  "
              f"top10={mlp_speaker['test_top10_acc']:.4f}")
        if mlp_gender is not None:
            print(f"  Gender  MLP: acc={mlp_gender['test_acc']:.4f}  "
                  f"balanced={mlp_gender['test_balanced_acc']:.4f}  "
                  f"recall_m={mlp_gender['test_recall_male']:.4f}  "
                  f"recall_f={mlp_gender['test_recall_female']:.4f}")
        else:
            print("  Gender  MLP: skipped (no gender labels in train/test split)")

    # ---- 7. Save results ----
    results_path = os.path.join(args.output_dir, "speaker_leakage_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "config": args.config,
                "checkpoint_path": args.checkpoint_path,
                "num_samples": len(speaker_ids),
                "num_unique_speakers": num_unique_speakers,
                "num_probe_classes": num_probe_classes,
                "gender_balance": {
                    "train": {
                        "male": int((gender_train == 0).sum()),
                        "female": int((gender_train == 1).sum()),
                        "unknown": int((gender_train == -1).sum()),
                    },
                    "test": {
                        "male": int((gender_test == 0).sum()),
                        "female": int((gender_test == 1).sum()),
                        "unknown": int((gender_test == -1).sum()),
                    },
                },
                "layers": results,
                "mlp_probe": mlp_summary,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_path}")

    # ---- 8. Plots ----
    if not args.no_plot:
        generate_plots(results, eval_layer_names, args.output_dir)
        generate_gender_plot(results, eval_layer_names, args.output_dir)

    # ---- 9. Summary ----
    best_layer = min(results, key=lambda r: r["probe"]["test_acc"])
    worst_layer = max(results, key=lambda r: r["probe"]["test_acc"])
    print(f"\nLeast speaker leakage:  {best_layer['layer_name']} "
          f"(probe acc = {best_layer['probe']['test_acc']:.4f})")
    print(f"Most speaker leakage:   {worst_layer['layer_name']} "
          f"(probe acc = {worst_layer['probe']['test_acc']:.4f})")
    if mlp_summary is not None:
        delta_spk = mlp_summary["speaker"]["test_acc"] - best_layer["probe"]["test_acc"]
        print(f"Non-linear speaker gap: MLP@{mlp_summary['layer_name']} "
              f"top-1 = {mlp_summary['speaker']['test_acc']:.4f} "
              f"(Δ vs linear best = {delta_spk:+.4f})")


if __name__ == "__main__":
    main()
