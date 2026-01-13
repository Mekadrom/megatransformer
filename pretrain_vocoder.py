import os

os.environ["DEEPSPEED_UNIT_TEST"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ['NCCL_TIMEOUT'] = '1200000'

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from typing import Any, Mapping, Optional, Union, List

from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers.integrations import TensorBoardCallback

from dataset_loading import audio_loading
from dataset_loading.vocoder_dataset import CachedVocoderDataset, VocoderDataCollator
from shard_utils import VocoderShardedDataset
from model.audio import discriminators
from model.audio.criteria import compute_discriminator_losses, compute_generator_losses
from model.audio.discriminators import CombinedDiscriminator
from model.audio.vocoders import vocoders
from model.audio.vocoders.vocoders import VocoderWithLoss
from utils import megatransformer_utils
from utils.audio_utils import SharedWindowBuffer
from utils.model_loading_utils import load_model, load_pruned_vocoder
from utils.training_utils import EarlyStoppingCallback, setup_int8_training


def get_writer(trainer: Trainer) -> Optional[SummaryWriter]:
    """Get TensorBoard writer from trainer callbacks."""
    if hasattr(trainer, 'callback_handler'):
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback):
                if callback.tb_writer is not None:
                    return callback.tb_writer
    return None


class ShardDebugCallback(TrainerCallback):
    """
    Callback for debugging shard loading and distributed sync issues.
    Enable with SHARD_DEBUG=1 environment variable.
    """

    def __init__(self):
        self._last_epoch = -1
        self._last_step = -1
        self._step_start_time = None
        import time
        self._time = time

    def _debug_enabled(self):
        return os.environ.get("SHARD_DEBUG", "0") == "1"

    def _get_rank(self):
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self._debug_enabled():
            print(f"[Rank {self._get_rank()}] EPOCH BEGIN: epoch={state.epoch}, global_step={state.global_step}")
        self._last_epoch = state.epoch

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._debug_enabled():
            print(f"[Rank {self._get_rank()}] EPOCH END: epoch={state.epoch}, global_step={state.global_step}")

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start_time = self._time.time()
        # Log every step near the problem area (steps 180-220) and every 50 steps otherwise
        current_epoch = int(state.epoch) if state.epoch is not None else 0
        step = state.global_step
        if self._debug_enabled():
            if (180 <= step <= 220) or step % 50 == 0 or current_epoch != self._last_epoch:
                print(f"[Rank {self._get_rank()}] step_begin: step={step}, epoch={state.epoch:.4f}")
            self._last_epoch = current_epoch
            self._last_step = step

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        elapsed = self._time.time() - self._step_start_time if self._step_start_time else 0
        # Log every step near the problem area, flag slow steps (>10s), and every 50 steps
        if self._debug_enabled():
            if (180 <= step <= 220) or step % 50 == 0 or elapsed > 10:
                print(f"[Rank {self._get_rank()}] step_end: step={step}, epoch={state.epoch:.4f}, took={elapsed:.2f}s")


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


class VocoderReconstructionCallback(TrainerCallback):
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

        test_audio_paths = [
            os.path.join('inference', 'examples', 'test_alm_1.mp3'),
            os.path.join('inference', 'examples', 'test_alm_2.mp3'),
        ]
        self.test_mel_specs = []
        self.test_audio_waveforms = []

        self.shared_window_buffer = shared_window_buffer

        for test_audio_path in test_audio_paths:
            # Load test audio
            if test_audio_path is not None and os.path.exists(test_audio_path):
                test_audio_waveforms, orig_sr = torchaudio.load(test_audio_path)
                if orig_sr != audio_sample_rate:
                    test_audio_waveforms = torchaudio.transforms.Resample(
                        orig_freq=orig_sr, new_freq=audio_sample_rate
                    )(test_audio_waveforms)
                test_audio_waveforms = test_audio_waveforms[0]
                self.test_audio_waveforms.append(test_audio_waveforms)

            # Extract mel spectrogram from test audio
            self.test_mel_specs.append(audio_loading.extract_mels(
                shared_window_buffer,
                test_audio_waveforms.squeeze(0),
                sr=audio_sample_rate,
                n_mels=audio_n_mels,
                n_fft=audio_n_fft,
                hop_length=audio_hop_length,
            ))

    def on_step_end(self, args, state, control, model: VocoderWithLoss = None, **kwargs):
        global_step = state.global_step + self.step_offset

        if ((global_step == 1) or (global_step % self.generation_steps == 0)) and state.is_world_process_zero:
            writer = get_writer(self.trainer)
            if writer is None:
                print("No TensorBoard writer found, skipping audio reconstruction...")
                return

            print(f"Reconstructing audio at step {global_step}...")

            # Determine device
            if torch.distributed.is_initialized():
                device = torch.device(f"cuda:{torch.distributed.get_rank()}")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            with torch.no_grad():
                dtype = torch.bfloat16 if bool(args.bf16) else torch.float16 if args.fp16 else torch.float32
                with autocast(device.type, dtype=dtype):
                    for e, (test_mel_spec, test_audio_waveform) in enumerate(zip(self.test_mel_specs, self.test_audio_waveforms)):
                        # Move test data to device
                        test_mel = test_mel_spec.unsqueeze(0).to(device)
                        test_waveform = test_audio_waveform.to(device)

                        # Generate waveform from mel spectrogram
                        outputs = model(
                            mel_spec=test_mel,
                            waveform_labels=test_waveform[0] if test_waveform.dim() > 1 else test_waveform,
                        )
                        pred_waveform = torch.clamp(outputs["pred_waveform"], -1.0, 1.0)[0].to(torch.float64).cpu()

                        gt_waveform = self.test_audio_waveforms[e].to(torch.float64)

                        self.log_audio(writer, gt_waveform, global_step, tag=f"vocoder_reconstruction/audio/target/{e}")
                        self.log_audio(writer, pred_waveform, global_step, tag=f"vocoder_reconstruction/audio/output/{e}")

                        # Waveform visualization (shape comparison)
                        self.log_waveform_visualization(writer, pred_waveform, gt_waveform, global_step, tag=f"vocoder_reconstruction/waveform_comparison/{e}")

                        # Reconstructed waveform's mel spectrogram
                        reconstructed_mel = audio_loading.extract_mels(
                            self.shared_window_buffer,
                            pred_waveform,
                            sr=self.audio_sample_rate,
                            n_mels=self.audio_n_mels,
                            n_fft=self.audio_n_fft,
                            hop_length=self.audio_hop_length,
                        )

                        # Mel comparison (target vs output vs error)
                        self.log_mel_comparison(writer, reconstructed_mel, test_mel_spec, global_step, tag=f"vocoder_reconstruction/mel_comparison/{e}")


                        target_stft = torch.stft(
                            gt_waveform,
                            n_fft=self.audio_n_fft,
                            hop_length=self.audio_hop_length,
                            window=self.shared_window_buffer.get_window(self.audio_n_fft, pred_waveform.device),
                            return_complex=True,
                        )
                        reconstructed_stft = torch.stft(
                            pred_waveform,
                            n_fft=self.audio_n_fft,
                            hop_length=self.audio_hop_length,
                            window=self.shared_window_buffer.get_window(self.audio_n_fft, pred_waveform.device),
                            return_complex=True,
                        )
                        # STFT magnitude comparison (target vs output vs error)
                        self.log_stft_magnitude_comparison(writer, reconstructed_stft, target_stft, global_step, tag=f"vocoder_reconstruction/stft_magnitude_comparison/{e}")

                        self.log_phase_error(writer, reconstructed_stft, target_stft, global_step, tag=f"vocoder_reconstruction/phase_comparison/{e}")
                        self.log_if_comparison(writer, reconstructed_stft, target_stft, global_step, tag=f"vocoder_reconstruction/if_comparison/{e}")

                    # Log losses (uses last outputs from test examples above)
                    if "loss" in outputs:
                        writer.add_scalar("vocoder_reconstruction/loss", outputs["loss"].item(), global_step)
                    if "waveform_l1" in outputs:
                        writer.add_scalar("vocoder_reconstruction/waveform_l1", outputs["waveform_l1"].item(), global_step)
                    if "sc_loss" in outputs:
                        writer.add_scalar("vocoder_reconstruction/sc_loss", outputs["sc_loss"].item(), global_step)
                    if "mag_loss" in outputs:
                        writer.add_scalar("vocoder_reconstruction/mag_loss", outputs["mag_loss"].item(), global_step)
                    if "mel_recon_loss" in outputs:
                        writer.add_scalar("vocoder_reconstruction/mel_recon_loss", outputs["mel_recon_loss"].item(), global_step)
                    if "complex_stft_loss" in outputs:
                        writer.add_scalar("vocoder_reconstruction/complex_stft_loss", outputs["complex_stft_loss"].item(), global_step)

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

            with autocast(device.type, dtype=dtype, enabled=args.bf16 or args.fp16):
                for i, idx in enumerate(indices):
                    print(f"[on_evaluate] Processing sample {i+1}/{num_samples} (idx={idx})...")
                    sample = eval_dataset[idx]
                    mel_spec = sample["mel_spec"]
                    waveform_labels = sample["waveform_labels"]

                    # Ensure correct shape [B, C, T]
                    if mel_spec.dim() == 2:
                        # [C, T] -> [1, C, T]
                        mel_spec = mel_spec.unsqueeze(0)

                    # Move to device
                    mel_input = mel_spec.to(device)
                    waveform_gt = waveform_labels.to(device)

                    # Run vocoder
                    outputs = model(
                        mel_spec=mel_input,
                        waveform_labels=waveform_gt,
                    )
                    pred_waveform = outputs["pred_waveform"].squeeze()

                    # Collect per-sample losses
                    for loss_name in ["loss", "waveform_l1", "sc_loss", "mag_loss", "mel_recon_loss", "complex_stft_loss"]:
                        if loss_name in outputs:
                            if loss_name not in all_losses:
                                all_losses[loss_name] = []
                            val = outputs[loss_name]
                            if isinstance(val, torch.Tensor):
                                val = val.item()
                            all_losses[loss_name].append(val)

                    # Align lengths for metrics
                    min_len = min(pred_waveform.shape[-1], waveform_gt.shape[-1])
                    pred_aligned = pred_waveform[..., :min_len]
                    gt_aligned = waveform_gt[..., :min_len]

                    # Waveform L1
                    waveform_l1 = torch.abs(pred_aligned - gt_aligned).mean().item()
                    all_waveform_l1.append(waveform_l1)

                    # Mel L1 (reconstruct mel from predicted waveform)
                    pred_mel = audio_loading.extract_mels(
                        self.shared_window_buffer,
                        pred_aligned.float().cpu(),
                        sr=self.audio_sample_rate,
                        n_mels=self.audio_n_mels,
                        n_fft=self.audio_n_fft,
                        hop_length=self.audio_hop_length,
                    )
                    gt_mel = audio_loading.extract_mels(
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

                    # Log per-sample metrics
                    writer.add_scalar(f"eval_vocoder/example_{i}/waveform_l1", waveform_l1, global_step)
                    writer.add_scalar(f"eval_vocoder/example_{i}/mel_l1", mel_l1, global_step)
                    writer.add_scalar(f"eval_vocoder/example_{i}/stft_mag_error", stft_mag_error, global_step)

        # Log aggregate statistics
        for loss_name, loss_vals in all_losses.items():
            if loss_vals:
                writer.add_scalar(f"eval_vocoder/mean_{loss_name}", np.mean(loss_vals), global_step)
                writer.add_scalar(f"eval_vocoder/std_{loss_name}", np.std(loss_vals), global_step)

        if all_waveform_l1:
            writer.add_scalar("eval_vocoder/mean_waveform_l1", np.mean(all_waveform_l1), global_step)
        if all_mel_l1:
            writer.add_scalar("eval_vocoder/mean_mel_l1", np.mean(all_mel_l1), global_step)
        if all_stft_mag_error:
            writer.add_scalar("eval_vocoder/mean_stft_mag_error", np.mean(all_stft_mag_error), global_step)

        print(f"Eval visualization complete: {num_samples} samples logged")
        writer.flush()

    def log_audio(self, writer: SummaryWriter, waveform: torch.Tensor, global_step, tag: str):
        writer.add_audio(tag, waveform, global_step, sample_rate=self.audio_sample_rate)

    def log_mel_spec_visualization(self, writer: SummaryWriter, mel_spec: torch.Tensor, global_step: int, tag: str):
        writer.add_image(tag, self._visualize_mel_spec(mel_spec.numpy(), self.audio_sample_rate), global_step)

    def log_phase_error(self, writer: SummaryWriter, pred_stft, target_stft, global_step: int, tag: str):
        """Log magnitude-weighted phase error map to tensorboard."""
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




class VocoderGANTrainer(Trainer):
    """
    Custom trainer for vocoder with GAN training.
    Handles alternating generator/discriminator updates.

    Supports dynamic GAN start conditions:
    - "step": Start GAN at a specific training step
    - "reconstruction_criteria_met": Start when reconstruction loss drops below threshold

    For sharded datasets, uses ShardAwareSampler to minimize shard loading overhead.
    """

    def __init__(
        self,
        *args,
        step_offset,
        n_fft,
        hop_length,
        cmdline,
        git_commit_hash,
        discriminator: Optional[CombinedDiscriminator] = None,
        discriminator_optimizer: Optional[torch.optim.Optimizer] = None,
        gan_adv_loss_weight: float = 1.0,
        gan_feature_matching_loss_weight: float = 2.0,
        discriminator_update_frequency: int = 1,
        gan_start_condition_key: Optional[str] = None,
        gan_start_condition_value: Optional[Any] = None,
        mpd_loss_weight: float = 1.0,
        msd_loss_weight: float = 1.0,
        mrsd_loss_weight: float = 1.0,
        mpd_adv_loss_weight: float = 1.0,
        msd_adv_loss_weight: float = 1.0,
        mrsd_adv_loss_weight: float = 1.0,
        mpd_fm_loss_weight: float = 1.0,
        msd_fm_loss_weight: float = 1.0,
        mrsd_fm_loss_weight: float = 1.0,
        direct_mag_loss_weight: float = 0.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Store shard-aware sampler if available
        self._shard_sampler = None
        if hasattr(self.train_dataset, 'get_sampler'):
            self._shard_sampler = self.train_dataset.get_sampler(shuffle=True, seed=42)
            print("Using ShardAwareSampler for efficient shard loading")

        self.step_offset = step_offset if step_offset is not None else 0
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.cmdline = cmdline
        self.git_commit_hash = git_commit_hash

        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.gan_adv_loss_weight = gan_adv_loss_weight
        self.gan_feature_matching_loss_weight = gan_feature_matching_loss_weight
        self.discriminator_update_frequency = discriminator_update_frequency
        self.gan_start_condition_key = gan_start_condition_key
        self.gan_start_condition_value = gan_start_condition_value
        self.gan_already_started = False
        self.mpd_loss_weight = mpd_loss_weight
        self.msd_loss_weight = msd_loss_weight
        self.mrsd_loss_weight = mrsd_loss_weight
        self.mpd_adv_loss_weight = mpd_adv_loss_weight
        self.msd_adv_loss_weight = msd_adv_loss_weight
        self.mrsd_adv_loss_weight = mrsd_adv_loss_weight
        self.mpd_fm_loss_weight = mpd_fm_loss_weight
        self.msd_fm_loss_weight = msd_fm_loss_weight
        self.mrsd_fm_loss_weight = mrsd_fm_loss_weight
        self.direct_mag_loss_weight = direct_mag_loss_weight
        self.writer = None

        self.has_logged_cli = False

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        Override to use shard-aware sampler for sharded datasets.

        This ensures samples are grouped by shard, minimizing disk I/O
        by loading each shard only once per epoch.
        """
        import time
        start = time.time()

        if self._shard_sampler is not None:
            # Update epoch for proper shuffling reproducibility
            epoch = 0
            if self.state is not None and self.state.epoch is not None:
                epoch = int(self.state.epoch)
            self._shard_sampler.set_epoch(epoch)

            if os.environ.get("SHARD_DEBUG", "0") == "1":
                elapsed = time.time() - start
                print(f"[VocoderGANTrainer] _get_train_sampler called, epoch={epoch}, took {elapsed:.2f}s")

            return self._shard_sampler

        # Fall back to default sampler for non-sharded datasets
        if os.environ.get("SHARD_DEBUG", "0") == "1":
            print("[VocoderGANTrainer] _get_train_sampler: using default sampler (no shard sampler)")
        return super()._get_train_sampler()

    def get_train_dataloader(self):
        """Override to add diagnostic logging."""
        import time
        start = time.time()

        if os.environ.get("SHARD_DEBUG", "0") == "1":
            print(f"[VocoderGANTrainer] get_train_dataloader called, state.epoch={self.state.epoch if self.state else None}")

        dataloader = super().get_train_dataloader()

        if os.environ.get("SHARD_DEBUG", "0") == "1":
            elapsed = time.time() - start
            print(f"[VocoderGANTrainer] get_train_dataloader completed in {elapsed:.2f}s, len={len(dataloader)}")

        return dataloader

    def is_gan_enabled(self, global_step: int, recon_loss: torch.Tensor) -> bool:
        """
        Check if GAN training should be enabled based on condition.

        Supports two modes:
        - "step": Start GAN at a specific training step
        - "reconstruction_criteria_met": Start when reconstruction loss drops below threshold

        Once GAN training starts, it stays enabled (via gan_already_started flag).
        """
        if self.discriminator is None:
            return False

        if self.gan_already_started:
            return True

        if self.gan_start_condition_key is None:
            # Legacy mode: always enabled if discriminator exists
            return True

        if self.gan_start_condition_key == "step":
            return global_step >= int(self.gan_start_condition_value)

        if self.gan_start_condition_key == "reconstruction_criteria_met":
            # Start GAN when reconstruction loss drops below threshold
            threshold = float(self.gan_start_condition_value)
            return recon_loss.item() < threshold

        return False

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            if self.is_deepspeed_enabled and (torch.is_floating_point(data)):
                kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        global_step = self.state.global_step + self.step_offset

        # Debug logging for distributed sync issues
        debug_enabled = os.environ.get("SHARD_DEBUG", "0") == "1"
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        # Log batch info to detect rank divergence (different batch sizes = sync issues)
        if debug_enabled and (180 <= global_step <= 220 or global_step % 50 == 0):
            batch_size = inputs["mel_spec"].shape[0]
            print(f"[Rank {rank}] compute_loss START step={global_step}, batch_size={batch_size}")

        self._ensure_tensorboard_writer()

        if not self.has_logged_cli:
            self.writer.add_text("training/command_line", self.cmdline, global_step)
            self.writer.add_text("training/git_commit_hash", self.git_commit_hash, global_step)
            self.has_logged_cli = True

        # Forward pass through generator (vocoder)
        waveform_labels = inputs["waveform_labels"]

        if debug_enabled and (180 <= global_step <= 220):
            print(f"[Rank {rank}] step={global_step} FORWARD START")

        outputs = model(
            mel_spec=inputs["mel_spec"],
            waveform_labels=waveform_labels,
            target_complex_stfts = inputs["target_complex_stfts"],
        )

        if debug_enabled and (180 <= global_step <= 220):
            print(f"[Rank {rank}] step={global_step} FORWARD END")

        # Get reconstruction losses from model
        recon_loss = outputs["loss"]
        pred_waveform = outputs["pred_waveform"]
        pred_stft = outputs.get("pred_stft", None)

        # Ensure waveform_labels has correct shape
        if waveform_labels.dim() == 1:
            waveform_labels = waveform_labels.unsqueeze(0)

        # Align lengths
        min_len = min(pred_waveform.shape[-1], waveform_labels.shape[-1])
        pred_waveform_aligned = pred_waveform[..., :min_len]
        waveform_labels_aligned = waveform_labels[..., :min_len]

        # Check if GAN training should be enabled
        gan_enabled = self.is_gan_enabled(global_step, recon_loss)

        # Set the flag once GAN starts (stays enabled for rest of training)
        if gan_enabled and not self.gan_already_started:
            self.gan_already_started = True
            print(f"GAN training started at step {global_step}")

        g_loss_gan = torch.tensor(0.0, device=pred_waveform.device)
        g_loss_fm = torch.tensor(0.0, device=pred_waveform.device)
        d_loss = torch.tensor(0.0, device=pred_waveform.device)

        if gan_enabled:
            # Discriminator Update
            if global_step % self.discriminator_update_frequency == 0:
                self.discriminator.train()

                # Detach generator outputs for discriminator update
                with torch.no_grad():
                    pred_waveform_detached = pred_waveform_aligned.detach()

                # Get discriminator outputs for real and fake
                # autocast needs to be used here
                device_type = pred_waveform.device.type
                dtype = torch.bfloat16 if self.args.bf16 else torch.float16 if self.args.fp16 else torch.float32
                with autocast(device_type, dtype=dtype, enabled=self.args.fp16 or self.args.bf16):
                    disc_real = self.discriminator(waveform_labels_aligned)
                    disc_fake = self.discriminator(pred_waveform_detached)

                d_losses = compute_discriminator_losses(disc_real, disc_fake)

                # Compute discriminator losses
                d_loss_mpd = d_losses["d_loss_mpd"]
                d_loss_msd = d_losses["d_loss_msd"]
                d_loss_mrsd = d_losses["d_loss_mrsd"]
                
                d_loss = self.mpd_loss_weight * d_loss_mpd + self.msd_loss_weight * d_loss_msd + self.mrsd_loss_weight * d_loss_mrsd

                if global_step % self.args.logging_steps == 0 and self.writer is not None:
                    for disc_crit in disc_real.keys():
                        for o, output in enumerate(disc_real[disc_crit][0]):
                            self._log_scalar(f"train/disc_real_{disc_crit}/{o}/avg", output.mean())

                    for disc_crit in disc_fake.keys():
                        for o, output in enumerate(disc_fake[disc_crit][0]):
                            self._log_scalar(f"train/disc_fake_{disc_crit}/{o}/avg", output.mean())
                        
                    self._log_scalar("train/d_loss_mpd", d_loss_mpd)
                    self._log_scalar("train/d_loss_msd", d_loss_msd)
                    self._log_scalar("train/d_loss_mrsd", d_loss_mrsd)
                    self._log_scalar("train/d_loss_total", d_loss)

                # Update discriminator (only during training when gradients are enabled)
                if self.discriminator_optimizer is not None and self.discriminator.training and torch.is_grad_enabled():
                    self.discriminator_optimizer.zero_grad()
                    d_loss.backward()
                    self.discriminator_optimizer.step()

            # Generator GAN Loss
            # Get discriminator outputs for fake (for generator update)
            device_type = pred_waveform.device.type
            dtype = torch.bfloat16 if self.args.bf16 else torch.float16 if self.args.fp16 else torch.float32
            with autocast(device_type, dtype=dtype, enabled=self.args.fp16 or self.args.bf16):
                disc_real = self.discriminator(waveform_labels_aligned.detach())
                disc_fake = self.discriminator(pred_waveform_aligned)

            g_losses = compute_generator_losses(disc_fake, disc_real)

            # Generator adversarial loss
            g_adv_mpd = g_losses["g_adv_mpd"]
            g_fm_mpd = g_losses["g_fm_mpd"]
            g_adv_msd = g_losses["g_adv_msd"]
            g_fm_msd = g_losses["g_fm_msd"]
            g_adv_mrsd = g_losses["g_adv_mrsd"]
            g_fm_mrsd = g_losses["g_fm_mrsd"]

            g_loss_adv = self.mpd_adv_loss_weight * g_adv_mpd + self.msd_adv_loss_weight * g_adv_msd + self.mrsd_adv_loss_weight * g_adv_mrsd
            g_loss_fm = self.mpd_fm_loss_weight * g_fm_mpd + self.msd_fm_loss_weight * g_fm_msd + self.mrsd_fm_loss_weight * g_fm_mrsd
            g_loss_gan = self.gan_adv_loss_weight * g_loss_adv + self.gan_feature_matching_loss_weight * g_loss_fm

            if global_step % self.args.logging_steps == 0 and self.writer is not None:
                self._log_scalar("train/g_adv_mpd", g_adv_mpd)
                self._log_scalar("train/g_fm_mpd", g_fm_mpd)
                self._log_scalar("train/g_adv_msd", g_adv_msd)
                self._log_scalar("train/g_fm_msd", g_fm_msd)
                self._log_scalar("train/g_adv_mrsd", g_adv_mrsd)
                self._log_scalar("train/g_fm_mrsd", g_fm_mrsd)
                self._log_scalar("train/g_adv_total", g_loss_adv)
                self._log_scalar("train/g_fm_total", g_loss_fm)
                self._log_scalar("train/g_loss_total", g_loss_gan)
        # Total generator loss
        total_loss = recon_loss + g_loss_gan

        # Log individual losses
        if global_step % self.args.logging_steps == 0 and self.writer is not None:
            prefix = "train/" if model.training else "eval/"
            self._log_scalar(f"{prefix}waveform_l1", outputs.get("waveform_l1", 0))
            self._log_scalar(f"{prefix}sc_loss", outputs.get("sc_loss", 0))
            self._log_scalar(f"{prefix}mag_loss", outputs.get("mag_loss", 0))
            self._log_scalar(f"{prefix}mel_recon_loss", outputs.get("mel_recon_loss", 0))
            self._log_scalar(f"{prefix}complex_stft_loss", outputs.get("complex_stft_loss", 0))
            self._log_scalar(f"{prefix}phase_ip_loss", outputs.get("phase_ip_loss", 0))
            self._log_scalar(f"{prefix}phase_iaf_loss", outputs.get("phase_iaf_loss", 0))
            self._log_scalar(f"{prefix}phase_gd_loss", outputs.get("phase_gd_loss", 0))
            self._log_scalar(f"{prefix}phase_loss", outputs.get("phase_loss", 0))
            self._log_scalar(f"{prefix}high_freq_stft_loss", outputs.get("high_freq_stft_loss", 0))
            self._log_scalar(f"{prefix}wav2vec2_loss", outputs.get("wav2vec2_loss", 0))
            self._log_scalar(f"{prefix}recon_loss", recon_loss)
            self._log_scalar(f"{prefix}total_loss", total_loss)

        # Debug logging
        debug_enabled = os.environ.get("SHARD_DEBUG", "0") == "1"
        if debug_enabled and (180 <= global_step <= 220 or global_step % 50 == 0):
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            print(f"[Rank {rank}] compute_loss END step={global_step}, loss={total_loss.item():.4f}")

        return (total_loss, outputs) if return_outputs else total_loss

    def _log_scalar(self, tag, value):
        global_step = self.state.global_step + self.step_offset
        if self.writer is not None:
            if isinstance(value, torch.Tensor):
                value = value.item()
            if value != 0.0:
                self.writer.add_scalar(tag, value, global_step)

    def _ensure_tensorboard_writer(self):
        if hasattr(self, "writer") and self.writer is not None:
            return

        for callback in self.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback):
                self.writer = callback.tb_writer
                return

        self.writer = None

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save both generator and discriminator."""
        super().save_model(output_dir, _internal_call)

        if output_dir is None:
            output_dir = self.args.output_dir

        # Save discriminator if GAN training has started
        if self.discriminator is not None and self.gan_already_started:
            os.makedirs(output_dir, exist_ok=True)
            discriminator_path = os.path.join(output_dir, "discriminator.pt")
            torch.save({
                "discriminator_state_dict": self.discriminator.state_dict(),
                "discriminator_optimizer_state_dict": (
                    self.discriminator_optimizer.state_dict()
                    if self.discriminator_optimizer is not None else None
                ),
            }, discriminator_path)
            print(f"Discriminator saved to {discriminator_path}")


def load_discriminator(
    resume_from_checkpoint: str,
    discriminator: CombinedDiscriminator,
    discriminator_optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[CombinedDiscriminator, Optional[torch.optim.Optimizer], bool]:
    """Load discriminator from checkpoint if it exists."""

    if resume_from_checkpoint is None:
        print("No checkpoint path provided, training discriminator from scratch")
        return discriminator, discriminator_optimizer, False
    
    discriminator_path = os.path.join(resume_from_checkpoint, "discriminator.pt")
    if os.path.exists(discriminator_path):
        print(f"Loading discriminator from {discriminator_path}")
        checkpoint = torch.load(discriminator_path, map_location=device)
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

        if discriminator_optimizer is not None and checkpoint.get("discriminator_optimizer_state_dict"):
            discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer_state_dict"])

        return discriminator, discriminator_optimizer, True
    
    print("No existing discriminator checkpoint found, training from scratch")
    return discriminator, discriminator_optimizer, False


def main():
    args, unk = megatransformer_utils.parse_args()
    run_dir = os.path.join(args.logging_base_dir, args.run_name)

    # Parse extra arguments
    unk_dict = {}
    for i in range(0, len(unk), 2):
        unk_dict[unk[i].lstrip('-')] = unk[i+1]

    # Loss weights
    sc_loss_weight = float(unk_dict.get("sc_loss_weight", 1.0))
    mag_loss_weight = float(unk_dict.get("mag_loss_weight", 3.0))
    waveform_l1_loss_weight = float(unk_dict.get("waveform_l1_loss_weight", 0.1))
    mel_recon_loss_weight = float(unk_dict.get("mel_recon_loss_weight", 1.0))
    mel_recon_loss_weight_linspace_max = float(unk_dict.get("mel_recon_loss_weight_linspace_max", 1.0))
    complex_stft_loss_weight = float(unk_dict.get("complex_stft_loss_weight", 1.0))
    phase_loss_weight = float(unk_dict.get("phase_loss_weight", 0.0))
    phase_ip_loss_weight = float(unk_dict.get("phase_ip_loss_weight", 0.0))
    phase_iaf_loss_weight = float(unk_dict.get("phase_iaf_loss_weight", 0.0))
    phase_gd_loss_weight = float(unk_dict.get("phase_gd_loss_weight", 0.0))
    high_freq_stft_loss_weight = float(unk_dict.get("high_freq_stft_loss_weight", 0.0))
    high_freq_stft_cutoff_bin = int(unk_dict.get("high_freq_stft_cutoff_bin", 256))
    wav2vec2_loss_weight = float(unk_dict.get("wav2vec2_loss_weight", 0.0))
    wav2vec2_model = unk_dict.get("wav2vec2_model", "facebook/wav2vec2-base")
    input_noise_std = float(unk_dict.get("input_noise_std", 0.0))

    # GAN training settings
    use_gan = unk_dict.get("use_gan", "false").lower() == "true"
    discriminator_lr = float(unk_dict.get("discriminator_lr", 1e-4))

    # Dynamic GAN start condition
    # Options: "step" (start at specific step) or "reconstruction_criteria_met" (start when loss drops below threshold)
    # Default: Start GAN when reconstruction loss < 8.0 (generator has learned basic structure)
    gan_start_condition_key = unk_dict.get("gan_start_condition_key", "reconstruction_criteria_met")
    gan_start_condition_value = unk_dict.get("gan_start_condition_value", "8.0")

    # GAN loss weights - feature matching typically more important for stability
    gan_adv_loss_weight = float(unk_dict.get("gan_adv_loss_weight", 1.0))
    gan_feature_matching_loss_weight = float(unk_dict.get("gan_feature_matching_loss_weight", 10.0))

    # Discriminator loss weights (for D update) - balance across discriminator types
    mpd_loss_weight = float(unk_dict.get("mpd_loss_weight", 1.0))
    msd_loss_weight = float(unk_dict.get("msd_loss_weight", 1.0))
    mrsd_loss_weight = float(unk_dict.get("mrsd_loss_weight", 1.0))

    # Generator adversarial loss weights - slightly favor MPD for periodicity
    mpd_adv_loss_weight = float(unk_dict.get("mpd_adv_loss_weight", 1.0))
    msd_adv_loss_weight = float(unk_dict.get("msd_adv_loss_weight", 0.5))
    mrsd_adv_loss_weight = float(unk_dict.get("mrsd_adv_loss_weight", 0.5))

    # Feature matching weights - MSD/MRSD FM tends to be more stable
    mpd_fm_loss_weight = float(unk_dict.get("mpd_fm_loss_weight", 1.0))
    msd_fm_loss_weight = float(unk_dict.get("msd_fm_loss_weight", 2.0))
    mrsd_fm_loss_weight = float(unk_dict.get("mrsd_fm_loss_weight", 2.0))

    direct_mag_loss_weight = float(unk_dict.get("direct_mag_loss_weight", 0.0))

    discriminator_config = unk_dict.get("discriminator_config", "small_combined_disc")

    # Check for pruned checkpoint (passed via --pruned_checkpoint extra arg)
    pruned_checkpoint = unk_dict.get("pruned_checkpoint", None)

    # Sharded dataset options
    use_sharded_dataset = unk_dict.get("use_sharded_dataset", "true").lower() == "true"
    train_cache_dir = unk_dict.get("train_cache_dir", "./cached_datasets/vocoder_nfft_1024_hop_length_256_train")
    val_cache_dir = unk_dict.get("val_cache_dir", "./cached_datasets/vocoder_nfft_1024_hop_length_256_val")
    shard_cache_size = int(unk_dict.get("shard_cache_size", 20))  # Number of shards to keep in memory
    preload_shards = int(unk_dict.get("preload_shards", 0))  # Number of shards to preload (-1 for all)

    shared_window_buffer = SharedWindowBuffer()

    if pruned_checkpoint is not None:
        # Load model from pruned checkpoint (architecture is embedded in checkpoint)
        print(f"Loading pruned model from {pruned_checkpoint}")
        model = load_pruned_vocoder(pruned_checkpoint, device='cpu', shared_window_buffer=shared_window_buffer)
        model_loaded = True
    else:
        # Standard config-based model creation
        if args.config not in vocoders.model_config_lookup:
            raise ValueError(f"Unknown vocoder config: {args.config}. Available: {list(vocoders.model_config_lookup.keys())}")

        model = vocoders.model_config_lookup[args.config](
            shared_window_buffer,
            audio_n_fft=args.audio_n_fft,
            audio_hop_length=args.audio_hop_length,
            audio_sample_rate=args.audio_sample_rate,
            sc_loss_weight=sc_loss_weight,
            mag_loss_weight=mag_loss_weight,
            waveform_l1_loss_weight=waveform_l1_loss_weight,
            mel_recon_loss_weight=mel_recon_loss_weight,
            mel_recon_loss_weight_linspace_max=mel_recon_loss_weight_linspace_max,
            complex_stft_loss_weight=complex_stft_loss_weight,
            phase_loss_weight=phase_loss_weight,
            phase_ip_loss_weight=phase_ip_loss_weight,
            phase_iaf_loss_weight=phase_iaf_loss_weight,
            phase_gd_loss_weight=phase_gd_loss_weight,
            high_freq_stft_loss_weight=high_freq_stft_loss_weight,
            high_freq_stft_cutoff_bin=high_freq_stft_cutoff_bin,
            direct_mag_loss_weight=direct_mag_loss_weight,
            wav2vec2_loss_weight=wav2vec2_loss_weight,
            wav2vec2_model=wav2vec2_model,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
        )

        model, model_loaded = load_model(args.start_step is not None, model, run_dir)

    # Determine device for discriminator
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.local_rank}" if args.local_rank >= 0 else "cuda")
    else:
        device = torch.device("cpu")

    # Create discriminator if GAN training is enabled
    discriminator = None
    discriminator_optimizer = None
    if use_gan:
        discriminator = discriminators.model_config_lookup[discriminator_config](
            shared_window_buffer=shared_window_buffer
        ).to(device)
        discriminator_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=discriminator_lr,
            betas=(0.8, 0.99),
            weight_decay=0.0,  # No weight decay for discriminator
        )
        # Try to load existing discriminator checkpoint
        discriminator, discriminator_optimizer, disc_loaded = load_discriminator(
            args.resume_from_checkpoint, discriminator, discriminator_optimizer, device
        )
        if disc_loaded:
            print("Loaded discriminator from checkpoint")

    if args.local_rank == 0 or not args.use_deepspeed:
        print(f"Model structure: {model}")
        if use_gan and discriminator is not None:
            print(f"Discriminator structure: {discriminator}")
            print(f"GAN training: {'enabled' if use_gan else 'disabled'}")
            print(f"  GAN loss weight: {gan_adv_loss_weight}")
            print(f"  Feature matching loss weight: {gan_feature_matching_loss_weight}")
            print(f"  MPD adversarial loss weight: {mpd_adv_loss_weight}")
            print(f"  MSD adversarial loss weight: {msd_adv_loss_weight}")
            print(f"  MRSD adversarial loss weight: {mrsd_adv_loss_weight}")
            print(f"  MPD feature matching loss weight: {mpd_fm_loss_weight}")
            print(f"  MSD feature matching loss weight: {msd_fm_loss_weight}")
            print(f"  MRSD feature matching loss weight: {mrsd_fm_loss_weight}")
            print(f"  Discriminator LR: {discriminator_lr}")
            print(f"  GAN start condition: {gan_start_condition_key}={gan_start_condition_value}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"  Vocoder parameters: {sum(p.numel() for p in model.vocoder.parameters()):,}")
        for name, module in model.vocoder.named_children():
            print(f"    {name} parameters: {sum(p.numel() for p in module.parameters()):,}")
        if use_gan and discriminator is not None:
            print(f"  Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
            print(f"    MPD parameters: {sum(p.numel() for p in discriminator.mpd.parameters()):,}")
            print(f"    MSD parameters: {sum(p.numel() for p in discriminator.msd.parameters()):,}")
            print(f"    MRSD parameters: {sum(p.numel() for p in discriminator.mrsd.parameters()):,}")

    model = setup_int8_training(args, model)

    def check_weight_stats(model):
        """Print weight statistics to verify initialization."""
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                std = param.std().item()
                mean = param.mean().item()
                # print(f"{name}: mean={mean:.6f}, std={std:.6f}")
                if std > 1.0:
                    print(f"  WARNING: High std for {name}")
                if std < 0.001:
                    print(f"  WARNING: Very low std for {name}")

    # Call after model creation
    check_weight_stats(model.vocoder)

    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        tpu_num_cores=8 if args.use_xla else None,
        output_dir=run_dir,
        overwrite_output_dir=True,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_kwargs=args.lr_scheduler_kwargs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs if args.num_train_epochs > 0 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        weight_decay=args.weight_decay,
        report_to="tensorboard",
        logging_dir=run_dir,
        logging_steps=args.logging_steps,
        save_safetensors=False,
        save_steps=args.save_steps,
        gradient_checkpointing=args.use_gradient_checkpointing,
        bf16=args.bf16,
        fp16=args.fp16,
        max_grad_norm=args.max_grad_norm,
        torch_compile=args.compile_model and not args.use_deepspeed and not args.use_xla,
        deepspeed=args.deepspeed_config if args.use_deepspeed and not args.use_xla else None,
        use_cpu=args.cpu,
        log_level=args.log_level,
        logging_first_step=True,
        local_rank=args.local_rank,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        ignore_data_skip=False,
        remove_unused_columns=False,
        eval_strategy="no" if args.eval_steps == 0 else "steps",
        eval_steps=args.eval_steps,
    )

    # Create datasets (sharded or legacy)
    if use_sharded_dataset:
        print(f"Using sharded vocoder datasets:")
        print(f"  Train: {train_cache_dir}")
        print(f"  Val: {val_cache_dir}")

        train_dataset = VocoderShardedDataset(
            shard_dir=train_cache_dir,
            cache_size=shard_cache_size,
            audio_max_frames=model.config.audio_max_frames,
            preload_shards=preload_shards,
        )

        eval_dataset = VocoderShardedDataset(
            shard_dir=val_cache_dir,
            cache_size=shard_cache_size,
            audio_max_frames=model.config.audio_max_frames,
            preload_shards=preload_shards,
        )
    else:
        print(f"Using legacy vocoder datasets:")
        print(f"  Train: {train_cache_dir}")
        print(f"  Val: {val_cache_dir}")

        train_dataset = CachedVocoderDataset(
            cache_dir=train_cache_dir,
            audio_max_frames=model.config.audio_max_frames,
        )

        eval_dataset = CachedVocoderDataset(
            cache_dir=val_cache_dir,
            audio_max_frames=model.config.audio_max_frames,
        )

    # Create data collator
    data_collator = VocoderDataCollator(
        audio_max_frames=model.config.audio_max_frames,
        audio_max_waveform_length=model.config.audio_max_waveform_length,
        n_mels=model.config.audio_n_mels,
        input_noise_std=input_noise_std,
    )

    if input_noise_std > 0.0:
        print(f"Input noise regularization enabled: std={input_noise_std}")

    # Create trainer (with or without GAN)
    trainer = VocoderGANTrainer(
        model=model,
        step_offset=args.start_step,
        n_fft=model.config.audio_n_fft,
        hop_length=model.config.audio_hop_length,
        cmdline=args.cmdline,
        git_commit_hash=args.commit_hash,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        discriminator=discriminator if use_gan else None,
        discriminator_optimizer=discriminator_optimizer if use_gan else None,
        gan_adv_loss_weight=gan_adv_loss_weight,
        gan_feature_matching_loss_weight=gan_feature_matching_loss_weight,
        mpd_loss_weight=mpd_loss_weight,
        msd_loss_weight=msd_loss_weight,
        mrsd_loss_weight=mrsd_loss_weight,
        gan_start_condition_key=gan_start_condition_key,
        gan_start_condition_value=gan_start_condition_value,
        mpd_adv_loss_weight=mpd_adv_loss_weight,
        msd_adv_loss_weight=msd_adv_loss_weight,
        mrsd_adv_loss_weight=mrsd_adv_loss_weight,
        mpd_fm_loss_weight=mpd_fm_loss_weight,
        msd_fm_loss_weight=msd_fm_loss_weight,
        mrsd_fm_loss_weight=mrsd_fm_loss_weight,
        direct_mag_loss_weight=direct_mag_loss_weight,
    )

    # Add reconstruction callback for monitoring training progress
    reconstruction_callback = VocoderReconstructionCallback(
        shared_window_buffer,
        step_offset=args.start_step,
        generation_steps=args.generation_steps,
        audio_sample_rate=model.config.audio_sample_rate,
        audio_n_mels=model.config.audio_n_mels,
        audio_n_fft=model.config.audio_n_fft,
        audio_hop_length=model.config.audio_hop_length,
    )
    trainer.add_callback(reconstruction_callback)

    # Add debug callback for shard loading diagnostics (enable with SHARD_DEBUG=1)
    trainer.add_callback(ShardDebugCallback())

    if args.stop_step > 0:
        early_stopping_callback = EarlyStoppingCallback(stop_step=args.stop_step)
        trainer.add_callback(early_stopping_callback)

    reconstruction_callback.trainer = trainer

    if hasattr(trainer, 'deepspeed') and trainer.deepspeed is not None:
        scheduler = trainer.deepspeed.lr_scheduler
        if scheduler is not None:
            print(f"DeepSpeed scheduler step: {scheduler.last_epoch}")
            print(f"Current LR: {scheduler.get_last_lr()}")
        else:
            print("No DeepSpeed LR scheduler found.")
    elif trainer.lr_scheduler is not None:
        print(f"Scheduler last_epoch: {trainer.lr_scheduler.last_epoch}")
        print(f"Current LR: {trainer.lr_scheduler.get_last_lr()}")
    else:
        print("No LR scheduler found in trainer.")

    checkpoint_path = args.resume_from_checkpoint
    if checkpoint_path is not None:
        print(f"Rank {trainer.args.local_rank} Checkpoint exists: {os.path.exists(checkpoint_path)}")
        print(f"Rank {trainer.args.local_rank} Checkpoint contents: {os.listdir(checkpoint_path) if os.path.exists(checkpoint_path) else 'N/A'}")

    print(f"Starting vocoder training with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
