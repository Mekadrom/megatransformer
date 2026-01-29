import abc

import librosa
import numpy as np
import torch


from typing import Optional

from matplotlib import pyplot as plt
from transformers import TrainerCallback

from torch.utils.tensorboard import SummaryWriter

from utils import audio_utils


class VisualizationCallback(abc.ABC, TrainerCallback):
    def _log_vocoder_audio(self, writer: SummaryWriter, mel_spec: torch.Tensor, global_step: int, tag: str):
        """Convert mel spectrogram to audio using vocoder and log to TensorBoard."""
        try:
            # Ensure mel is [B, n_mels, T] for vocoder
            if mel_spec.dim() == 2:
                mel_spec = mel_spec.unsqueeze(0)  # [1, n_mels, T]

            print(f"Generating audio with vocoder at step {global_step} for tag {tag} with mel specs {mel_spec.shape}...")

            # Match vocoder dtype (may be bfloat16 if training with bf16)
            vocoder_dtype = next(self.vocoder.parameters()).dtype
            vocoder_device = next(self.vocoder.parameters()).device
            mel_spec = mel_spec.to(device=vocoder_device, dtype=vocoder_dtype)

            with torch.no_grad():
                waveform, _ = self.vocoder(mel_spec)

            # Ensure 1D waveform and convert to float32 CPU for numpy
            if waveform.dim() > 1:
                waveform = waveform.squeeze()
            waveform = waveform.float().cpu()

            # Normalize to [-1, 1] range
            waveform = waveform / (waveform.abs().max() + 1e-8)

            # Log audio to TensorBoard
            writer.add_audio(
                tag,
                waveform.numpy(),
                global_step,
                sample_rate=self.audio_sample_rate
            )
            return waveform
        except Exception as e:
            print(f"Failed to generate audio with vocoder: {e}")
            raise e

    def _get_device(self):
        """Determine the device to use for inference."""
        if torch.distributed.is_initialized():
            return torch.device(f"cuda:{torch.distributed.get_rank()}")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _visualize_mel_spec(self, mel_spec: np.ndarray) -> np.ndarray:
        """Generate mel spectrogram visualization for TensorBoard."""
        if mel_spec.ndim == 3:
            mel_spec = mel_spec.squeeze(0)

        # Normalize to [0, 1] range for visualization
        mel_spec_norm = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(
            mel_spec_norm,
            hop_length=self.audio_hop_length,
            x_axis='time',
            y_axis='mel',
            sr=self.audio_sample_rate,
            n_fft=self.audio_n_fft,
            fmin=0,
            fmax=8000,
            ax=ax,
        )
        plt.colorbar(img, format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()

        # Convert figure to numpy array
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape((height, width, 4))[:, :, :3]  # Drop alpha channel

        plt.close(fig)

        # TensorBoard expects (C, H, W) for add_image
        data = data.transpose(2, 0, 1)

        return data

    def _log_mel_comparison(self, writer: SummaryWriter, pred_mel: np.ndarray, target_mel: np.ndarray, global_step: int, tag: str):
        """Log side-by-side comparison of predicted and target mel spectrograms."""
        if pred_mel.ndim == 3:
            pred_mel = pred_mel.squeeze(0)
        if target_mel.ndim == 3:
            target_mel = target_mel.squeeze(0)

        # Align lengths
        min_len = min(pred_mel.shape[-1], target_mel.shape[-1])
        pred_mel = pred_mel[..., :min_len]
        target_mel = target_mel[..., :min_len]

        # Create comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Normalize for visualization
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

        # Error map
        error = np.abs(pred_mel - target_mel)
        im2 = axes[2].imshow(error, aspect='auto', origin='lower', cmap='viridis')
        axes[2].set_title(f'Absolute Error (mean={error.mean():.4f})')
        axes[2].set_ylabel('Mel bin')
        axes[2].set_xlabel('Time frame')
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        writer.add_figure(tag, fig, global_step)
        plt.close(fig)

    def _log_attention_weights(
        self,
        writer: SummaryWriter,
        attn_weights: Optional[torch.Tensor],
        global_step: int,
        tag_prefix: str,
        H: int,
        W: int,
    ):
        """
        Log 2D attention weight visualizations to TensorBoard.

        Args:
            writer: TensorBoard writer
            attn_weights: Attention tensor [B, n_heads, H*W, H*W] or None
            global_step: Current training step
            tag_prefix: Tag prefix for TensorBoard (e.g., "eval_vae/example_0/encoder_attention")
            H: Height of the spatial grid
            W: Width of the spatial grid
        """
        if attn_weights is None:
            return

        # Move to CPU and convert to numpy
        # Shape: [n_heads, H*W, H*W]
        weights = attn_weights[0].float().detach().cpu().numpy()
        n_heads, seq_len, _ = weights.shape

        # 1. Global average attention map (avg across heads)
        global_avg_weights = weights.mean(axis=0)  # [H*W, H*W]

        # Log full 2D attention map
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(global_avg_weights, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f'Attention (avg {n_heads} heads, {H}×{W}={H*W} tokens)')
        ax.set_xlabel('Key position')
        ax.set_ylabel('Query position')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        writer.add_figure(f"{tag_prefix}/global_2d", fig, global_step)
        plt.close(fig)

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
        writer.add_figure(f"{tag_prefix}/per_head", fig, global_step)
        plt.close(fig)

        # 3. Spatial attention pattern - where does each position attend?
        # Reshape attention to show spatial patterns
        # For a few query positions, show attention as a 2D heatmap
        query_positions = [0, seq_len // 4, seq_len // 2, 3 * seq_len // 4]  # corners and center
        fig, axes = plt.subplots(1, len(query_positions), figsize=(4 * len(query_positions), 4))
        for i, (q_pos, ax) in enumerate(zip(query_positions, axes)):
            # Get attention from this query position to all keys
            attn_from_query = global_avg_weights[q_pos].reshape(H, W)
            im = ax.imshow(attn_from_query, aspect='auto', origin='lower', cmap='hot')
            q_h, q_w = q_pos // W, q_pos % W
            ax.set_title(f'Query ({q_h},{q_w})')
            ax.scatter([q_w], [q_h], c='cyan', s=100, marker='x')  # Mark query position
        plt.suptitle('Attention from query positions (cyan X)')
        plt.tight_layout()
        writer.add_figure(f"{tag_prefix}/spatial_pattern", fig, global_step)
        plt.close(fig)
