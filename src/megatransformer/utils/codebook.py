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


def load_f0_stats(path_or_payload):
    """Per-speaker log-F0 (mean, std) from a codebook payload, or None if absent.

    Returns (mean[S], std[S], global_mean, global_std). Speakers with too few frames to
    estimate get the global values, so indexing is always safe.
    """
    obj = path_or_payload
    if isinstance(obj, str):
        obj = torch.load(obj, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "speaker_f0_mean" not in obj:
        return None
    return (obj["speaker_f0_mean"].float(), obj["speaker_f0_std"].float(),
            float(obj["global_f0_mean"]), float(obj["global_f0_std"]))


def normalize_f0(f0: torch.Tensor, speaker_id: int, stats) -> torch.Tensor:
    """Speaker-normalize a log-F0 contour: (log_f0 - mu_spk) / sigma_spk.

    Per SPEAKER, not per utterance. Measured on LibriTTS-R, the between-speaker spread of
    mean log-F0 is 0.267 against a within-speaker spread of 0.195 (ratio 1.36) -- so the
    speaker offset is the LARGER term, and it is the one a text-only model structurally
    cannot know (ContentVec is speaker-invariant and text says nothing about who is
    talking). Removing it leaves the world model a well-posed target: the ~0.195 of
    genuinely text-driven variation. Normalizing per UTTERANCE would instead delete
    utterance-level prosody (excitement, emphasis), which is exactly what we want predicted.

    Unvoiced frames are stored as f0 == 0 and stay 0: they carry no pitch, and mapping 0
    through the normalizer would emit a large negative excursion the head would then chase.
    """
    if stats is None:
        return f0
    mean, std, g_mean, g_std = stats
    s = int(speaker_id)
    mu = float(mean[s]) if 0 <= s < len(mean) else g_mean
    sd = float(std[s]) if 0 <= s < len(std) else g_std
    sd = max(sd, 1e-3)
    return torch.where(f0 > 0, (f0 - mu) / sd, torch.zeros_like(f0))


def denormalize_f0(contour: torch.Tensor, mu: float, sd: float) -> torch.Tensor:
    """Inverse of normalize_f0 for a known speaker. Unvoiced (0) stays 0."""
    return torch.where(contour != 0, contour * sd + mu, torch.zeros_like(contour))


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


def dedup_units(unit_ids: torch.Tensor, length: Optional[int] = None):
    """Collapse consecutive-equal units into (unit, duration) segments.

    This is the coarsening that gives the world model a text gradient: at the 50Hz frame
    rate the next unit is dominated by "where am I in this phoneme", which teacher-forced
    unit history answers and text cannot (measured: text adds +0.004 accuracy over
    history). Collapsing runs removes that within-phoneme redundancy -- consecutive
    repeats are gone by construction, so "predict the same unit again" is impossible and
    the model must predict a genuinely different unit each step, which needs the text (the
    marginal jumps ~8x to +0.033 on deduped units).

    Args:
        unit_ids: (T,) int64 from quantize(), possibly -1-padded past `length`.
        length: real frame count; frames beyond it are ignored.

    Returns:
        dedup_ids:   (M,) the surviving units, no two adjacent equal.
        durations:   (M,) run length of each in ORIGINAL 50Hz frames (sums to `length`).
        seg_of_frame:(length,) segment index each original frame belongs to -- use it to
                     pool any per-frame quantity (F0, VUV) to per-segment with
                     pool_by_segment().
    """
    T = unit_ids.shape[0]
    n = T if length is None else max(0, min(int(length), T))
    dev = unit_ids.device
    if n == 0:
        z = torch.zeros(0, dtype=torch.long, device=dev)
        return z, z, torch.zeros(0, dtype=torch.long, device=dev)

    ids = unit_ids[:n]
    change = torch.ones(n, dtype=torch.bool, device=dev)
    change[1:] = ids[1:] != ids[:-1]
    starts = torch.nonzero(change, as_tuple=False).squeeze(1)   # (M,) run start indices
    dedup_ids = ids[starts]
    ends = torch.cat([starts[1:], torch.tensor([n], device=dev)])
    durations = ends - starts                                    # (M,)
    seg_of_frame = torch.cumsum(change.long(), 0) - 1           # (n,) in [0, M)
    return dedup_ids, durations, seg_of_frame


def frame_to_segment_index(durations: torch.Tensor, seg_lengths: torch.Tensor,
                           max_frames: int) -> torch.Tensor:
    """Inverse of durations: a (B, max_frames) gather index mapping each 50Hz frame to the
    segment it belongs to. Use it to expand a per-segment tensor (e.g. coda hidden states)
    to frame rate: expanded = h.gather(1, index[..., None].expand(-1, -1, D)).

    durations: (B, Md) run lengths, 0-padded past seg_lengths.
    seg_lengths: (B,) real segment count per row.
    Padded frames map to segment 0 (they are masked downstream by the frame length).
    """
    B = durations.shape[0]
    idx = torch.zeros(B, max_frames, dtype=torch.long, device=durations.device)
    for i in range(B):
        m = int(seg_lengths[i])
        reps = durations[i, :m].clamp(min=0)
        expanded = torch.repeat_interleave(torch.arange(m, device=durations.device), reps)
        idx[i, :expanded.shape[0]] = expanded[:max_frames]
    return idx


def pool_by_segment(x: torch.Tensor, seg_of_frame: torch.Tensor, num_segments: int,
                    weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Average a per-frame quantity within each dedup segment.

    Used to carry the per-frame F0 contour and VUV onto the deduped sequence (one value
    per segment). For F0, pass weights=vuv so unvoiced frames -- where the contour is 0 by
    construction -- don't drag the segment's pitch toward zero, matching how the per-speaker
    F0 stats were fit.
    """
    x = x[:seg_of_frame.shape[0]].float()
    w = torch.ones_like(x) if weights is None else weights[:seg_of_frame.shape[0]].float()
    num = torch.zeros(num_segments, device=x.device).index_add_(0, seg_of_frame, x * w)
    den = torch.zeros(num_segments, device=x.device).index_add_(0, seg_of_frame, w)
    return num / den.clamp_min(1e-8)
