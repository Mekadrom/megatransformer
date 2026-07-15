"""k-means codebook over content features (ContentVec / SIVE), and on-the-fly quantization.

Why quantize at all: trained on CONTINUOUS ContentVec, a model can predict frame t from the
audio history alone -- the features carry the phonetics, so the text (for the world model)
and F0 (for the SMG) are both redundant, earn no gradient, and get ignored. Snapping each
frame to a centroid strips the continuous variation, which measurement showed is where
prosody lives: quantized speech stays fully intelligible but goes flat. That is the point.
It forces prosody to arrive via F0 and content to arrive via the units, so neither
conditioning signal can be ignored.

Quantized features are CENTROID VECTORS, not IDs, so consumers keep their existing
(B, D, T) float interface. Unit id and centroid are bijective, so nothing is lost; a model
that wants an embedding table can use the ids instead.
"""
from typing import Optional, Tuple

import torch


def save_codebook(path: str, centroids: torch.Tensor, meta: Optional[dict] = None) -> None:
    """Persist a (K, D) centroid matrix plus provenance."""
    payload = {"centroids": centroids.cpu().float(), "k": int(centroids.shape[0]),
               "dim": int(centroids.shape[1])}
    if meta:
        payload.update(meta)
    torch.save(payload, path)


def load_codebook(path_or_tensor) -> torch.Tensor:
    """Return the (K, D) centroid matrix from a path, a payload dict, or a raw tensor."""
    if isinstance(path_or_tensor, torch.Tensor):
        return path_or_tensor.float()
    obj = path_or_tensor
    if isinstance(obj, str):
        obj = torch.load(obj, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        obj = obj["centroids"]
    return obj.float()


def quantize(
    feats: torch.Tensor,
    centroids: torch.Tensor,
    length: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Snap each frame of `feats` to its nearest centroid.

    Args:
        feats: (D, T) content features, channel-first, as stored in the shards.
        centroids: (K, D) codebook.
        length: real frame count. Frames beyond it are PADDING and are left untouched --
            zeros would otherwise snap to whatever centroid sits nearest the origin, which
            silently feeds a real unit into the pad region.

    Returns:
        (quantized (D, T) in the same dtype/device as `feats`, unit ids (T,) int64 with
        -1 at padded positions).
    """
    D, T = feats.shape
    n = T if length is None else max(0, min(int(length), T))

    q = feats.clone()
    ids = torch.full((T,), -1, dtype=torch.long, device=feats.device)
    if n == 0:
        return q, ids

    x = feats[:, :n].T.float()                       # (n, D)
    c = centroids.to(device=feats.device, dtype=torch.float32)
    idx = torch.cdist(x, c).argmin(dim=1)            # (n,)
    q[:, :n] = c[idx].T.to(feats.dtype)
    ids[:n] = idx
    return q, ids
