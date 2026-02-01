import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio


from typing import Optional

from torch.utils.tensorboard.writer import SummaryWriter
from transformers.trainer import Trainer

from scripts.train.visualization_callback import VisualizationCallback
from utils import audio_utils
from utils.audio_utils import SharedWindowBuffer
from utils.megatransformer_utils import pad_and_mask
from utils.train_utils import get_writer


class VocoderVisualizationCallback(VisualizationCallback):
    """
    Callback for logging vocoder audio reconstruction during training.
    Periodically generates audio from test mel spectrograms and logs to TensorBoard.
    """

    def __init__(
        self,
        shared_window_buffer: SharedWindowBuffer,
        step_offset: int = 0,
        generation_steps: int = 1000,
        audio_sample_rate: int = 16000,
        audio_n_mels: int = 80,
        audio_n_fft: int = 1024,
        audio_hop_length: int = 256,
        num_eval_samples: int = 8,
    ):
        self.trainer: Optional[Trainer] = None
        self.step_offset = self.step_offset = step_offset if step_offset is not None else 0
        self.generation_steps = generation_steps
        self.audio_sample_rate = audio_sample_rate
        self.audio_n_mels = audio_n_mels
        self.audio_n_fft = audio_n_fft
        self.audio_hop_length = audio_hop_length
        self.num_eval_samples = num_eval_samples

        self.shared_window_buffer = shared_window_buffer

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Generate and log vocoder reconstructions during evaluation from eval dataset."""
        global_step = state.global_step + self.step_offset

        if not state.is_world_process_zero:
            return

        writer = get_writer(self.trainer)
        if writer is None:
            print("No TensorBoard writer found, skipping eval visualization...")
            return

        # Get eval dataset from trainer
        eval_dataset = self.trainer.eval_dataset
        if eval_dataset is None or len(eval_dataset) == 0:
            print("No eval dataset available, skipping eval visualization...")
            return

        print(f"[on_evaluate] Starting eval vocoder reconstructions at step {global_step}...")

        # Determine device
        if torch.distributed.is_initialized():
            device = torch.device(f"cuda:{torch.distributed.get_rank()}")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model.eval()

        # Use sequential indices from the start of the dataset to avoid shard thrashing
        # (random access across shards causes severe I/O delays)
        num_samples = min(self.num_eval_samples, len(eval_dataset))
        indices = list(range(num_samples))  # Sequential indices from first shard

        # Collect aggregate statistics
        all_losses = {}
        all_waveform_l1 = []
        all_mel_l1 = []
        all_stft_mag_error = []

        with torch.no_grad():
            dtype = torch.bfloat16 if bool(args.bf16) else torch.float16 if args.fp16 else torch.float32

            with torch.autocast(device.type, dtype=dtype, enabled=args.bf16 or args.fp16):
                for i, idx in enumerate(indices):
                    print(f"[on_evaluate] Processing sample {i+1}/{num_samples} (idx={idx})...")

                    # dataset only has length, not mask from collator
                    sample = eval_dataset[idx]
                    mel_spec = sample["mel_spec"]
                    mel_length = sample["mel_length"]
                    waveform = sample["waveform"]
                    waveform_length = sample["waveform_length"]

                    # Ensure correct shape [B, C, T]
                    if mel_spec.dim() == 2:
                        # [C, T] -> [1, C, T]
                        mel_spec = mel_spec.unsqueeze(0)

                    # Move to device
                    mel_spec = mel_spec.to(device)
                    mel_spec_mask = pad_and_mask([mel_spec], [mel_length])[1][0].to(device)
                    waveform = waveform.to(device)
                    waveform_mask = pad_and_mask([waveform], [waveform_length])[1][0].to(device)

                    # Run vocoder
                    outputs = model(mel_spec, mel_spec_mask, waveform, waveform_mask)

                    # Collect per-sample losses
                    for loss_name in ["loss", "waveform_l1", "sc_loss", "mag_loss", "mel_recon_loss"]:
                        if loss_name in outputs:
                            if loss_name not in all_losses:
                                all_losses[loss_name] = []
                            val = outputs[loss_name]
                            if isinstance(val, torch.Tensor):
                                val = val.item()
                            all_losses[loss_name].append(val)

                    pred_waveform = outputs["pred_waveform"].detach()
                    waveform_gt = waveform.detach()

                    # Align lengths for metrics
                    min_len = min(pred_waveform.shape[-1], waveform_gt.shape[-1])
                    pred_aligned = pred_waveform[..., :min_len]
                    gt_aligned = waveform_gt[..., :min_len]

                    # Waveform L1
                    waveform_l1 = torch.abs(pred_aligned - gt_aligned).mean().item()
                    all_waveform_l1.append(waveform_l1)

                    # Mel L1 (reconstruct mel from predicted waveform)
                    pred_mel = audio_utils.extract_mels(
                        self.shared_window_buffer,
                        pred_aligned.float().cpu(),
                        sr=self.audio_sample_rate,
                        n_mels=self.audio_n_mels,
                        n_fft=self.audio_n_fft,
                        hop_length=self.audio_hop_length,
                    )
                    gt_mel = audio_utils.extract_mels(
                        self.shared_window_buffer,
                        gt_aligned.float().cpu(),
                        sr=self.audio_sample_rate,
                        n_mels=self.audio_n_mels,
                        n_fft=self.audio_n_fft,
                        hop_length=self.audio_hop_length,
                    )
                    mel_min_len = min(pred_mel.shape[-1], gt_mel.shape[-1])
                    mel_l1 = torch.abs(pred_mel[..., :mel_min_len] - gt_mel[..., :mel_min_len]).mean().item()
                    all_mel_l1.append(mel_l1)

                    # STFT magnitude error
                    pred_stft = torch.stft(
                        pred_aligned.float().cpu(),
                        n_fft=self.audio_n_fft,
                        hop_length=self.audio_hop_length,
                        window=self.shared_window_buffer.get_window(self.audio_n_fft, "cpu"),
                        return_complex=True,
                    )
                    gt_stft = torch.stft(
                        gt_aligned.float().cpu(),
                        n_fft=self.audio_n_fft,
                        hop_length=self.audio_hop_length,
                        window=self.shared_window_buffer.get_window(self.audio_n_fft, "cpu"),
                        return_complex=True,
                    )
                    stft_mag_error = torch.abs(torch.abs(pred_stft) - torch.abs(gt_stft)).mean().item()
                    all_stft_mag_error.append(stft_mag_error)

                    # Log individual samples (audio and visualizations)
                    pred_audio = torch.clamp(pred_aligned, -1.0, 1.0).float().cpu()
                    gt_audio = gt_aligned.float().cpu()

                    writer.add_audio(
                        f"eval_vocoder/audio/{i}/output",
                        pred_audio,
                        global_step,
                        sample_rate=self.audio_sample_rate
                    )
                    writer.add_audio(
                        f"eval_vocoder/audio/{i}/target",
                        gt_audio,
                        global_step,
                        sample_rate=self.audio_sample_rate
                    )

                    # Log visualizations for first few samples
                    if i < 4:
                        # Waveform comparison
                        self.log_waveform_visualization(
                            writer, pred_audio, gt_audio, global_step,
                            tag=f"eval_vocoder/waveform_comparison/{i}"
                        )

                        # Mel comparison (target vs output vs error)
                        self.log_mel_comparison(
                            writer, pred_mel, gt_mel, global_step,
                            tag=f"eval_vocoder/mel_comparison/{i}"
                        )

                        # STFT magnitude comparison
                        self.log_stft_magnitude_comparison(
                            writer, pred_stft, gt_stft, global_step,
                            tag=f"eval_vocoder/stft_magnitude_comparison/{i}"
                        )

                        # Phase error
                        self.log_phase_error(
                            writer, pred_stft, gt_stft, global_step,
                            tag=f"eval_vocoder/phase_comparison/{i}"
                        )

                        # IF comparison
                        self.log_if_comparison(
                            writer, pred_stft, gt_stft, global_step,
                            tag=f"eval_vocoder/if_comparison/{i}"
                        )

        print(f"Eval visualization complete: {num_samples} samples logged")
        writer.flush()

    def log_audio(self, writer: SummaryWriter, waveform: torch.Tensor, global_step, tag: str):
        writer.add_audio(tag, waveform, global_step, sample_rate=self.audio_sample_rate)

    def log_mel_spec_visualization(self, writer: SummaryWriter, mel_spec: torch.Tensor, global_step: int, tag: str):
        writer.add_image(tag, self._visualize_mel_spec(mel_spec.numpy(), self.audio_sample_rate), global_step)

    def log_phase_error(self, writer: SummaryWriter, pred_stft, target_stft, global_step: int, tag: str):
        """Log magnitude-weighted phase error map to tensorboard."""

        if pred_stft.dim() == 3:
            pred_stft = pred_stft.squeeze(0)
        if target_stft.dim() == 3:
            target_stft = target_stft.squeeze(0)

        pred_phase = torch.angle(pred_stft)
        target_phase = torch.angle(target_stft)
        pred_mag = torch.abs(pred_stft)

        # Match lengths
        min_t = min(pred_phase.shape[-1], target_phase.shape[-1])
        pred_phase = pred_phase[..., :min_t]
        target_phase = target_phase[..., :min_t]
        pred_mag = pred_mag[..., :min_t]

        # Wrapped phase error, 0 = perfect, π = worst
        error = torch.atan2(
            torch.sin(pred_phase - target_phase),
            torch.cos(pred_phase - target_phase)
        ).abs()

        # Weight by magnitude - only show error where there's actual signal
        mag_normalized = pred_mag / (pred_mag.max() + 1e-8)
        error_weighted = error * mag_normalized

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(error_weighted.cpu().numpy(), aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=np.pi * 0.5)
        plt.colorbar(im, ax=ax, label='Magnitude-weighted phase error')
        ax.set_ylabel('Frequency bin')
        ax.set_xlabel('Time frame')
        ax.set_title('Phase Error Map (weighted by magnitude)')

        writer.add_figure(tag, fig, global_step)
        plt.close(fig)

    def log_instantaneous_frequency(self, writer: SummaryWriter, stft, global_step: int, tag: str):
        """Log instantaneous frequency deviation to tensorboard, masked by magnitude."""

        if stft.dim() == 3:
            stft = stft.squeeze(0)

        phase = torch.angle(stft)
        mag = torch.abs(stft)

        inst_freq = torch.diff(phase, dim=-1)
        inst_freq = torch.atan2(torch.sin(inst_freq), torch.cos(inst_freq))
        inst_freq_hz = inst_freq * self.audio_sample_rate / (2 * np.pi * self.audio_hop_length)

        # Mask out low-energy regions to reduce noise
        mag_for_weight = mag[..., :-1]  # align with diff output
        mag_normalized = mag_for_weight / (mag_for_weight.max() + 1e-8)
        threshold = 0.01
        inst_freq_masked = torch.where(mag_normalized > threshold, inst_freq_hz, torch.full_like(inst_freq_hz, float('nan')))

        max_if = self.audio_sample_rate / (2 * self.audio_hop_length)

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(inst_freq_masked.cpu().numpy(), aspect='auto', origin='lower',
                       cmap='coolwarm', vmin=-max_if, vmax=max_if)
        plt.colorbar(im, ax=ax, label='Frequency deviation (Hz)')
        ax.set_ylabel('Frequency bin')
        ax.set_xlabel('Time frame')
        ax.set_title('Instantaneous Frequency Deviation (magnitude-masked)')

        writer.add_figure(tag, fig, global_step)
        plt.close(fig)

    def log_waveform_visualization(self, writer: SummaryWriter, pred_waveform: torch.Tensor, target_waveform: torch.Tensor, global_step: int, tag: str):
        """Log waveform comparison visualization with phase-invariant metrics to tensorboard."""
        # Keep tensors for metric computation
        pred_tensor = pred_waveform.cpu().flatten()
        target_tensor = target_waveform.cpu().flatten()

        # Align lengths
        min_len = min(len(pred_tensor), len(target_tensor))
        pred_tensor = pred_tensor[:min_len]
        target_tensor = target_tensor[:min_len]

        # Compute cross-correlation alignment
        aligned_pred, shift = align_waveforms_xcorr(target_tensor, pred_tensor)

        # Compute phase-invariant metrics
        si_snr = compute_si_snr(target_tensor, aligned_pred)
        mr_stft_loss = compute_multi_resolution_stft_loss(target_tensor, aligned_pred)

        # Compute envelope comparison
        target_env = compute_envelope(target_tensor, frame_size=256)
        pred_env = compute_envelope(aligned_pred, frame_size=256)
        env_min_len = min(len(target_env), len(pred_env))
        envelope_l1 = torch.abs(target_env[:env_min_len] - pred_env[:env_min_len]).mean().item()

        # Convert to numpy for plotting
        pred_wav = pred_tensor.numpy()
        target_wav = target_tensor.numpy()
        aligned_wav = aligned_pred.numpy()

        # Create time axis in seconds
        time_axis = np.arange(min_len) / self.audio_sample_rate
        env_time_axis = np.arange(env_min_len) * 256 / self.audio_sample_rate

        # Unaligned and aligned differences
        unaligned_diff = np.abs(pred_wav - target_wav)
        aligned_diff = np.abs(aligned_wav - target_wav)

        fig, axes = plt.subplots(5, 1, figsize=(12, 14))

        # Target waveform
        axes[0].plot(time_axis, target_wav, color='blue', linewidth=0.5, alpha=0.8)
        axes[0].set_title('Target Waveform')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_xlim([0, time_axis[-1]])
        axes[0].set_ylim([-1, 1])
        axes[0].grid(True, alpha=0.3)

        # Predicted waveform
        axes[1].plot(time_axis, pred_wav, color='green', linewidth=0.5, alpha=0.8)
        axes[1].set_title('Predicted Waveform')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_xlim([0, time_axis[-1]])
        axes[1].set_ylim([-1, 1])
        axes[1].grid(True, alpha=0.3)

        # Unaligned difference
        axes[2].plot(time_axis, unaligned_diff, color='red', linewidth=0.5, alpha=0.8)
        axes[2].set_title(f'Unaligned Absolute Difference (mean={unaligned_diff.mean():.4f})')
        axes[2].set_ylabel('|Error|')
        axes[2].set_xlim([0, time_axis[-1]])
        axes[2].grid(True, alpha=0.3)

        # Aligned difference
        axes[3].plot(time_axis, aligned_diff, color='orange', linewidth=0.5, alpha=0.8)
        axes[3].set_title(f'Aligned Absolute Difference (mean={aligned_diff.mean():.4f}, shift={shift} samples)')
        axes[3].set_ylabel('|Error|')
        axes[3].set_xlim([0, time_axis[-1]])
        axes[3].grid(True, alpha=0.3)

        # Envelope comparison
        axes[4].plot(env_time_axis, target_env[:env_min_len].numpy(), color='blue', linewidth=1.0, alpha=0.8, label='Target')
        axes[4].plot(env_time_axis, pred_env[:env_min_len].numpy(), color='green', linewidth=1.0, alpha=0.8, label='Predicted (aligned)')
        axes[4].set_title(f'Envelope Comparison (L1={envelope_l1:.4f})')
        axes[4].set_xlabel('Time (s)')
        axes[4].set_ylabel('RMS Amplitude')
        axes[4].set_xlim([0, env_time_axis[-1] if len(env_time_axis) > 0 else 1])
        axes[4].legend(loc='upper right')
        axes[4].grid(True, alpha=0.3)

        # Add metrics text box
        metrics_text = f'SI-SNR: {si_snr:.2f} dB\nMR-STFT Loss: {mr_stft_loss:.4f}\nEnvelope L1: {envelope_l1:.4f}'
        fig.text(0.02, 0.98, metrics_text, transform=fig.transFigure, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for metrics text
        writer.add_figure(tag, fig, global_step)
        plt.close(fig)

        # Log scalar metrics
        base_tag = tag.rsplit('/', 1)[0]  # Remove sample index from tag
        writer.add_scalar(f"{base_tag}/si_snr_db", si_snr, global_step)
        writer.add_scalar(f"{base_tag}/mr_stft_loss", mr_stft_loss, global_step)
        writer.add_scalar(f"{base_tag}/envelope_l1", envelope_l1, global_step)
        writer.add_scalar(f"{base_tag}/aligned_waveform_l1", aligned_diff.mean(), global_step)
        writer.add_scalar(f"{base_tag}/unaligned_waveform_l1", unaligned_diff.mean(), global_step)
        writer.add_scalar(f"{base_tag}/alignment_shift_samples", abs(shift), global_step)

    def log_stft_magnitude_comparison(self, writer: SummaryWriter, pred_stft: torch.Tensor, target_stft: torch.Tensor, global_step: int, tag: str):
        """Log STFT magnitude comparison visualization to tensorboard."""

        if pred_stft.dim() == 3:
            pred_stft = pred_stft.squeeze(0)
        if target_stft.dim() == 3:
            target_stft = target_stft.squeeze(0)

        pred_mag = torch.abs(pred_stft).cpu().numpy()
        target_mag = torch.abs(target_stft).cpu().numpy()

        # Align lengths
        min_t = min(pred_mag.shape[-1], target_mag.shape[-1])
        pred_mag = pred_mag[..., :min_t]
        target_mag = target_mag[..., :min_t]

        # Convert to dB scale for better visualization
        pred_mag_db = 20 * np.log10(pred_mag + 1e-8)
        target_mag_db = 20 * np.log10(target_mag + 1e-8)

        vmin = min(pred_mag_db.min(), target_mag_db.min())
        vmax = max(pred_mag_db.max(), target_mag_db.max())

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        im0 = axes[0].imshow(target_mag_db, aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
        axes[0].set_title('Target STFT Magnitude (dB)')
        axes[0].set_ylabel('Frequency bin')
        axes[0].set_xlabel('Time frame')
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(pred_mag_db, aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
        axes[1].set_title('Predicted STFT Magnitude (dB)')
        axes[1].set_ylabel('Frequency bin')
        axes[1].set_xlabel('Time frame')
        plt.colorbar(im1, ax=axes[1])

        # Error map (in dB)
        error_db = np.abs(pred_mag_db - target_mag_db)
        im2 = axes[2].imshow(error_db, aspect='auto', origin='lower', cmap='viridis')
        axes[2].set_title(f'Magnitude Error (dB) (mean={error_db.mean():.2f})')
        axes[2].set_ylabel('Frequency bin')
        axes[2].set_xlabel('Time frame')
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        writer.add_figure(tag, fig, global_step)
        plt.close(fig)

    def log_mel_comparison(self, writer: SummaryWriter, pred_mel: np.ndarray, target_mel: np.ndarray, global_step: int, tag: str):
        """Log mel spectrogram comparison (target, predicted, error) to tensorboard."""
        # Handle tensor input
        if hasattr(pred_mel, 'numpy'):
            pred_mel = pred_mel.numpy()
        if hasattr(target_mel, 'numpy'):
            target_mel = target_mel.numpy()

        # Ensure 2D
        if pred_mel.ndim == 3:
            pred_mel = pred_mel.squeeze(0)
        if target_mel.ndim == 3:
            target_mel = target_mel.squeeze(0)

        # Align lengths
        min_t = min(pred_mel.shape[-1], target_mel.shape[-1])
        pred_mel = pred_mel[..., :min_t]
        target_mel = target_mel[..., :min_t]

        vmin = min(pred_mel.min(), target_mel.min())
        vmax = max(pred_mel.max(), target_mel.max())

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

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

    def log_if_comparison(self, writer: SummaryWriter, pred_stft, target_stft, global_step: int, tag: str):
        if pred_stft.dim() == 3:
            pred_stft = pred_stft.squeeze(0)
        if target_stft.dim() == 3:
            target_stft = target_stft.squeeze(0)

        def compute_if_masked(stft, threshold=0.01):
            phase = torch.angle(stft)
            mag = torch.abs(stft)
            inst_freq = torch.diff(phase, dim=-1)
            inst_freq = torch.atan2(torch.sin(inst_freq), torch.cos(inst_freq))
            inst_freq_hz = inst_freq * self.audio_sample_rate / (2 * np.pi * self.audio_hop_length)

            # Mask by magnitude
            mag_for_weight = mag[..., :-1]
            mag_normalized = mag_for_weight / (mag_for_weight.max() + 1e-8)
            inst_freq_masked = torch.where(mag_normalized > threshold, inst_freq_hz, torch.full_like(inst_freq_hz, float('nan')))
            return inst_freq_masked, mag_normalized

        pred_if, pred_mag = compute_if_masked(pred_stft)
        target_if, target_mag = compute_if_masked(target_stft)

        pred_if = pred_if.cpu().numpy()
        target_if = target_if.cpu().numpy()
        pred_mag = pred_mag.cpu().numpy()

        min_t = min(pred_if.shape[-1], target_if.shape[-1])
        pred_if = pred_if[..., :min_t]
        target_if = target_if[..., :min_t]
        pred_mag = pred_mag[..., :min_t]

        max_if = self.audio_sample_rate / (2 * self.audio_hop_length)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        im0 = axes[0].imshow(target_if, aspect='auto', origin='lower', cmap='coolwarm', vmin=-max_if, vmax=max_if)
        axes[0].set_title('Target IF (masked)')
        axes[0].set_ylabel('Frequency bin')
        axes[0].set_xlabel('Time frame')
        plt.colorbar(im0, ax=axes[0], label='Hz')

        im1 = axes[1].imshow(pred_if, aspect='auto', origin='lower', cmap='coolwarm', vmin=-max_if, vmax=max_if)
        axes[1].set_title('Predicted IF (masked)')
        axes[1].set_ylabel('Frequency bin')
        axes[1].set_xlabel('Time frame')
        plt.colorbar(im1, ax=axes[1], label='Hz')

        # Error map: magnitude-weighted absolute difference
        if_error = np.abs(pred_if - target_if)
        # Weight error by magnitude so low-energy errors don't dominate
        if_error_weighted = if_error * pred_mag
        im2 = axes[2].imshow(if_error_weighted, aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=max_if * 0.5)
        axes[2].set_title('IF Error (mag-weighted)')
        axes[2].set_ylabel('Frequency bin')
        axes[2].set_xlabel('Time frame')
        plt.colorbar(im2, ax=axes[2], label='Hz')

        plt.tight_layout()
        writer.add_figure(tag, fig, global_step)
        plt.close(fig)

    def _save_audio_to_file(
        self,
        waveform: torch.Tensor,
        filepath: str,
        sample_rate: int,
        normalize: bool = True,
        bits_per_sample: int = 16,
    ):
        """Save waveform to audio file."""
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if normalize:
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        torchaudio.save(
            filepath,
            waveform.cpu(),
            sample_rate,
            bits_per_sample=bits_per_sample,
            format="wav",
        )
        print(f"Audio saved to {filepath}")

    def _visualize_mel_spec(self, mel_spec: np.ndarray, sample_rate: int) -> np.ndarray:
        """Generate mel spectrogram visualization for TensorBoard."""
        # Handle tensor input
        if hasattr(mel_spec, 'numpy'):
            mel_spec = mel_spec.numpy()

        # Ensure 2D
        if mel_spec.ndim == 3:
            mel_spec = mel_spec.squeeze(0)

        # Normalize to [0, 1] range
        mel_spec_norm = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)

        # Create figure with Agg backend to avoid display issues
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(
            mel_spec_norm,
            hop_length=self.audio_hop_length,
            x_axis='time',
            y_axis='mel',
            sr=sample_rate,
            n_fft=self.audio_n_fft,
            fmin=0,
            fmax=8000,
            ax=ax,
        )
        plt.colorbar(img, format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()

        # Convert figure to numpy array using Agg renderer
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape((height, width, 4))[:, :, :3]  # Drop alpha channel

        plt.close(fig)

        # TensorBoard expects (C, H, W) for add_image
        data = data.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

        return data


def align_waveforms_xcorr(target: torch.Tensor, reconstructed: torch.Tensor, max_shift: int = 1000) -> tuple[torch.Tensor, int]:
    """
    Align reconstructed waveform to target using cross-correlation.

    Args:
        target: Target waveform [T]
        reconstructed: Reconstructed waveform [T]
        max_shift: Maximum shift to search (samples)

    Returns:
        aligned: Aligned reconstructed waveform
        shift: Optimal shift (positive = reconstructed was ahead)
    """
    # Ensure 1D
    target = target.flatten()
    reconstructed = reconstructed.flatten()

    # Use shorter max_shift if waveforms are short
    max_shift = min(max_shift, len(target) // 4, len(reconstructed) // 4)

    # Compute cross-correlation using conv1d
    # Flip target for correlation (conv is cross-correlation with flipped kernel)
    target_padded = torch.nn.functional.pad(target.unsqueeze(0).unsqueeze(0), (max_shift, max_shift))
    kernel = reconstructed.unsqueeze(0).unsqueeze(0).flip(-1)

    # Use a window around center for efficiency
    window_size = min(len(reconstructed), 16000)  # Max 1 second window
    kernel_window = kernel[..., :window_size]

    correlation = torch.nn.functional.conv1d(
        target_padded.float(),
        kernel_window.float(),
        padding=0
    ).squeeze()

    # Find peak
    peak_idx = correlation.argmax().item()
    shift = peak_idx - max_shift

    # Apply shift to reconstructed
    if shift > 0:
        # Reconstructed is ahead, shift it back
        aligned = torch.nn.functional.pad(reconstructed[shift:], (0, shift))
    elif shift < 0:
        # Reconstructed is behind, shift it forward
        aligned = torch.nn.functional.pad(reconstructed[:shift], (-shift, 0))
    else:
        aligned = reconstructed

    # Ensure same length as target
    if len(aligned) > len(target):
        aligned = aligned[:len(target)]
    elif len(aligned) < len(target):
        aligned = torch.nn.functional.pad(aligned, (0, len(target) - len(aligned)))

    return aligned, shift


def compute_envelope(waveform: torch.Tensor, frame_size: int = 256) -> torch.Tensor:
    """
    Compute amplitude envelope using RMS over frames.

    This is a simple approximation that doesn't require scipy's Hilbert transform.

    Args:
        waveform: Input waveform [T]
        frame_size: Frame size for RMS computation

    Returns:
        envelope: Amplitude envelope [T // frame_size]
    """
    waveform = waveform.flatten()

    # Pad to multiple of frame_size
    pad_len = (frame_size - len(waveform) % frame_size) % frame_size
    if pad_len > 0:
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))

    # Reshape and compute RMS per frame
    frames = waveform.view(-1, frame_size)
    envelope = torch.sqrt((frames ** 2).mean(dim=1) + 1e-8)

    return envelope


def compute_si_snr(target: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Compute Scale-Invariant Signal-to-Noise Ratio (SI-SNR).

    SI-SNR is invariant to the overall scale and is a good perceptual metric.
    Higher is better (in dB).

    Args:
        target: Target waveform [T]
        estimate: Estimated waveform [T]
        eps: Small value for numerical stability

    Returns:
        si_snr: SI-SNR in dB
    """
    target = target.flatten().float()
    estimate = estimate.flatten().float()

    # Align lengths
    min_len = min(len(target), len(estimate))
    target = target[:min_len]
    estimate = estimate[:min_len]

    # Zero-mean
    target = target - target.mean()
    estimate = estimate - estimate.mean()

    # Compute s_target (projection of estimate onto target)
    dot = torch.dot(estimate, target)
    s_target = dot * target / (torch.dot(target, target) + eps)

    # Compute noise
    e_noise = estimate - s_target

    # SI-SNR in dB
    si_snr = 10 * torch.log10(
        (s_target ** 2).sum() / ((e_noise ** 2).sum() + eps) + eps
    )

    return si_snr.item()


def compute_multi_resolution_stft_loss(
    target: torch.Tensor,
    estimate: torch.Tensor,
    fft_sizes: list = [512, 1024, 2048],
    hop_sizes: list = [128, 256, 512],
    win_sizes: list = [512, 1024, 2048],
) -> float:
    """
    Compute multi-resolution STFT loss (spectral convergence + log magnitude).

    This is a phase-invariant perceptual metric.

    Args:
        target: Target waveform [T]
        estimate: Estimated waveform [T]
        fft_sizes: List of FFT sizes
        hop_sizes: List of hop sizes
        win_sizes: List of window sizes

    Returns:
        mr_stft_loss: Multi-resolution STFT loss (lower is better)
    """
    target = target.flatten().float()
    estimate = estimate.flatten().float()

    # Align lengths
    min_len = min(len(target), len(estimate))
    target = target[:min_len]
    estimate = estimate[:min_len]

    total_sc_loss = 0.0
    total_mag_loss = 0.0

    for fft_size, hop_size, win_size in zip(fft_sizes, hop_sizes, win_sizes):
        window = torch.hann_window(win_size, device=target.device)

        target_stft = torch.stft(
            target.unsqueeze(0), n_fft=fft_size, hop_length=hop_size,
            win_length=win_size, window=window, return_complex=True
        )
        estimate_stft = torch.stft(
            estimate.unsqueeze(0), n_fft=fft_size, hop_length=hop_size,
            win_length=win_size, window=window, return_complex=True
        )

        target_mag = torch.abs(target_stft)
        estimate_mag = torch.abs(estimate_stft)

        # Spectral convergence
        sc_loss = torch.norm(target_mag - estimate_mag, p='fro') / (torch.norm(target_mag, p='fro') + 1e-8)
        total_sc_loss += sc_loss.item()

        # Log magnitude loss
        log_target = torch.log(target_mag + 1e-8)
        log_estimate = torch.log(estimate_mag + 1e-8)
        mag_loss = torch.mean(torch.abs(log_target - log_estimate))
        total_mag_loss += mag_loss.item()

    return (total_sc_loss + total_mag_loss) / len(fft_sizes)
