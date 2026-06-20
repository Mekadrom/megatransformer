"""Pure rendering functions for training visualizations.

All functions return matplotlib Figure objects or numpy arrays — they never
log anything directly. Callers use the metrics module to log the results.
"""

from typing import Optional

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch


def render_mel_spectrogram(
    mel_spec: np.ndarray,
    hop_length: int = 256,
    sample_rate: int = 16000,
    n_fft: int = 1024,
) -> plt.Figure:
    """Render a mel spectrogram as a matplotlib Figure.

    Args:
        mel_spec: Mel spectrogram array, shape (n_mels, T) or (1, n_mels, T).
        hop_length: STFT hop length.
        sample_rate: Audio sample rate.
        n_fft: FFT size.

    Returns:
        matplotlib Figure with the rendered spectrogram.
    """
    if mel_spec.ndim == 3:
        mel_spec = mel_spec.squeeze(0)

    mel_spec_norm = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mel_spec_norm,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        sr=sample_rate,
        n_fft=n_fft,
        fmin=0,
        fmax=8000,
        ax=ax,
    )
    plt.colorbar(img, format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    return fig


def render_mel_comparison(
    pred_mel: np.ndarray,
    target_mel: np.ndarray,
) -> plt.Figure:
    """Render a 3-panel mel spectrogram comparison (target, predicted, error).

    Args:
        pred_mel: Predicted mel, shape (n_mels, T) or (1, n_mels, T).
        target_mel: Target mel, same shape.

    Returns:
        matplotlib Figure with the comparison.
    """
    if pred_mel.ndim == 3:
        pred_mel = pred_mel.squeeze(0)
    if target_mel.ndim == 3:
        target_mel = target_mel.squeeze(0)

    min_len = min(pred_mel.shape[-1], target_mel.shape[-1])
    pred_mel = pred_mel[..., :min_len]
    target_mel = target_mel[..., :min_len]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    vmin = min(pred_mel.min(), target_mel.min())
    vmax = max(pred_mel.max(), target_mel.max())

    im0 = axes[0].imshow(target_mel, aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
    axes[0].set_title('Target Mel')
    axes[0].set_ylabel('Mel bin')
    axes[0].set_xlabel('Time frame')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred_mel, aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
    axes[1].set_title('Reconstructed Mel')
    axes[1].set_ylabel('Mel bin')
    axes[1].set_xlabel('Time frame')
    plt.colorbar(im1, ax=axes[1])

    error = np.abs(pred_mel - target_mel)
    im2 = axes[2].imshow(error, aspect='auto', origin='lower', cmap='viridis')
    axes[2].set_title(f'Absolute Error (mean={error.mean():.4f})')
    axes[2].set_ylabel('Mel bin')
    axes[2].set_xlabel('Time frame')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    return fig


def render_attention_weights(
    attn_weights: torch.Tensor,
    H: int,
    W: int,
) -> dict[str, plt.Figure]:
    """Render 2D attention weight visualizations.

    Args:
        attn_weights: Attention tensor [B, n_heads, H*W, H*W].
        H: Height of the spatial grid.
        W: Width of the spatial grid.

    Returns:
        Dict mapping subtag to Figure: "global_2d", "per_head", "spatial_pattern".
    """
    weights = attn_weights[0].float().detach().cpu().numpy()
    n_heads, seq_len, _ = weights.shape

    figures = {}

    # 1. Global average attention map
    global_avg_weights = weights.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(global_avg_weights, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title(f'Attention (avg {n_heads} heads, {H}×{W}={H*W} tokens)')
    ax.set_xlabel('Key position')
    ax.set_ylabel('Query position')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    figures["global_2d"] = fig

    # 2. Per-head attention maps (first 4 heads)
    n_heads_to_show = min(4, n_heads)
    fig, axes = plt.subplots(1, n_heads_to_show, figsize=(4 * n_heads_to_show, 4))
    if n_heads_to_show == 1:
        axes = [axes]
    for head_idx, ax in enumerate(axes):
        im = ax.imshow(weights[head_idx], aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f'Head {head_idx}')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
    plt.tight_layout()
    figures["per_head"] = fig

    # 3. Spatial attention pattern from selected query positions
    query_positions = [0, seq_len // 4, seq_len // 2, 3 * seq_len // 4]
    fig, axes = plt.subplots(1, len(query_positions), figsize=(4 * len(query_positions), 4))
    for q_pos, ax in zip(query_positions, axes):
        attn_from_query = global_avg_weights[q_pos].reshape(H, W)
        im = ax.imshow(attn_from_query, aspect='auto', origin='lower', cmap='hot')
        q_h, q_w = q_pos // W, q_pos % W
        ax.set_title(f'Query ({q_h},{q_w})')
        ax.scatter([q_w], [q_h], c='cyan', s=100, marker='x')
    plt.suptitle('Attention from query positions (cyan X)')
    plt.tight_layout()
    figures["spatial_pattern"] = fig

    return figures


def render_vocoder_audio(
    vocoder: torch.nn.Module,
    mel_spec: torch.Tensor,
) -> np.ndarray:
    """Run vocoder inference and return normalized waveform.

    Args:
        vocoder: Vocoder model.
        mel_spec: Mel spectrogram, shape (n_mels, T) or (B, n_mels, T).

    Returns:
        Waveform as 1D numpy float32 array, normalized to [-1, 1].
    """
    if mel_spec.dim() == 2:
        mel_spec = mel_spec.unsqueeze(0)

    vocoder_dtype = next(vocoder.parameters()).dtype
    vocoder_device = next(vocoder.parameters()).device
    mel_spec = mel_spec.to(device=vocoder_device, dtype=vocoder_dtype)

    with torch.no_grad():
        waveform = vocoder(mel_spec)["pred_waveform"]

    if waveform.dim() > 1:
        waveform = waveform.squeeze()
    waveform = waveform.float().cpu()

    waveform = waveform / (waveform.abs().max() + 1e-8)
    return waveform.numpy()


def _resolve_stat_value(value):
    """Convert a stat value to a scalar (averages tensors)."""
    if isinstance(value, torch.Tensor):
        return value.float().mean().item()
    return value


def render_iteration_stats(
    iteration_stats: list[dict],
) -> plt.Figure:
    """Render averaged per-iteration activation stats from the recurrent block.

    Stats may be scalars or per-token tensors (batch, seq_len) — tensors
    are averaged down to scalars for the overview figure.

    Args:
        iteration_stats: List of dicts, one per iteration. Each dict has
            keys "std", "mean", "max", "min", "norm", and optionally "kl".

    Returns:
        matplotlib Figure with one subplot per stat.
    """
    stat_names = ["std", "mean", "max", "min", "norm"]
    has_kl = any("kl" in s for s in iteration_stats)
    if has_kl:
        stat_names.append("kl")

    iterations = list(range(len(iteration_stats)))

    fig, axes = plt.subplots(len(stat_names), 1, figsize=(10, 2.5 * len(stat_names)), sharex=True)
    if len(stat_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, stat_names):
        values = [_resolve_stat_value(s.get(name, 0.0)) for s in iteration_stats]
        ax.plot(iterations, values, marker='.', markersize=3, linewidth=1)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        if name == "kl":
            ax.set_yscale('symlog', linthresh=1e-6)
            ax.set_ylabel('KL divergence')

    axes[-1].set_xlabel('Iteration')
    fig.suptitle('Thought state activation stats per recurrent iteration', fontsize=12)
    plt.tight_layout()
    return fig


def render_token_iteration_stats(
    iteration_stats: list[dict],
    token_idx: int,
    batch_idx: int = 0,
    title: str = "",
) -> plt.Figure:
    """Render per-iteration stats for a single token position.

    Args:
        iteration_stats: List of dicts, one per iteration. Values are
            tensors of shape (batch, seq_len).
        token_idx: Token position in the sequence.
        batch_idx: Batch index (default 0).
        title: Optional title override.

    Returns:
        matplotlib Figure with one subplot per stat.
    """
    stat_names = ["std", "mean", "max", "min", "norm"]
    has_kl = any("kl" in s for s in iteration_stats)
    if has_kl:
        stat_names.append("kl")

    iterations = list(range(len(iteration_stats)))

    fig, axes = plt.subplots(len(stat_names), 1, figsize=(8, 2 * len(stat_names)), sharex=True)
    if len(stat_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, stat_names):
        values = []
        for s in iteration_stats:
            v = s.get(name)
            if v is None:
                values.append(0.0)
            elif isinstance(v, torch.Tensor):
                values.append(v[batch_idx, token_idx].item())
            else:
                values.append(v)
        ax.plot(iterations, values, marker='.', markersize=3, linewidth=1)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        if name == "kl":
            ax.set_yscale('symlog', linthresh=1e-6)
            ax.set_ylabel('KL divergence')

    axes[-1].set_xlabel('Iteration')
    fig.suptitle(title or f'Token {token_idx} — activation stats per iteration', fontsize=10)
    plt.tight_layout()
    return fig


def render_all_tokens_iteration_stats(
    iteration_stats: list[dict],
    batch_idx: int = 0,
    modality_map: Optional[torch.Tensor] = None,
) -> plt.Figure:
    """Render all tokens' per-iteration stats overlaid on the same axes.

    Each token is a line, colored by modality. All tokens share the same
    y-axis scale so magnitudes are directly comparable.

    Args:
        iteration_stats: List of dicts, one per iteration. Values are
            tensors of shape (batch, seq_len).
        batch_idx: Batch index (default 0).
        modality_map: Optional (batch, seq_len) tensor of modality IDs.

    Returns:
        matplotlib Figure with one subplot per stat, all tokens overlaid.
    """
    stat_names = ["std", "mean", "max", "min", "norm"]
    has_kl = any("kl" in s for s in iteration_stats)
    if has_kl:
        stat_names.append("kl")

    seq_len = iteration_stats[0]["std"].shape[1]
    iterations = list(range(len(iteration_stats)))

    modality_colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}  # text, audio, voice, image
    modality_names = {0: "text", 1: "audio", 2: "voice", 3: "image"}

    fig, axes = plt.subplots(len(stat_names), 1, figsize=(12, 3 * len(stat_names)), sharex=True)
    if len(stat_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, stat_names):
        legend_added = set()
        for tok in range(seq_len):
            values = []
            for s in iteration_stats:
                v = s.get(name)
                if v is None:
                    values.append(0.0)
                elif isinstance(v, torch.Tensor):
                    values.append(v[batch_idx, tok].item())
                else:
                    values.append(v)

            mod_id = modality_map[batch_idx, tok].item() if modality_map is not None else -1
            color = modality_colors.get(mod_id, "#999999")
            label = modality_names.get(mod_id, None) if mod_id not in legend_added else None
            if label:
                legend_added.add(mod_id)

            ax.plot(iterations, values, linewidth=0.5, alpha=0.4, color=color, label=label)

        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        if legend_added:
            ax.legend(fontsize=8, loc='upper right')
        if name == "kl":
            ax.set_yscale('symlog', linthresh=1e-6)
            ax.set_ylabel('KL divergence')

    axes[-1].set_xlabel('Iteration')
    fig.suptitle(f'All {seq_len} tokens — activation stats per iteration (colored by modality)', fontsize=12)
    plt.tight_layout()
    return fig
