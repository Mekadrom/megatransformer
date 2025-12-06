import os

from dataset_loading.vocoder_dataset import CachedVocoderDataset, VocoderDataCollator
from model.audio import discriminators
from model.audio.criteria import compute_discriminator_losses, compute_generator_losses
from model.audio.discriminators import CombinedDiscriminator
from model.audio.shared_window_buffer import SharedWindowBuffer
from model.audio.vocoders import vocoders
from model.audio.vocoders.vocoders import VocoderWithLoss

os.environ["DEEPSPEED_UNIT_TEST"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ['NCCL_TIMEOUT'] = '1200000'

from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers.integrations import TensorBoardCallback
from typing import Any, List, Dict, Mapping, Optional, Union

from dataset_loading import audio_loading

import librosa
import librosa.display
import matplotlib.pyplot as plt
import megatransformer_utils
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


def get_writer(trainer: Trainer) -> Optional[SummaryWriter]:
    """Get TensorBoard writer from trainer callbacks."""
    if hasattr(trainer, 'callback_handler'):
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback):
                if callback.tb_writer is not None:
                    return callback.tb_writer
    return None

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
    ):
        self.trainer: Optional[Trainer] = None
        self.step_offset = step_offset
        self.generation_steps = generation_steps
        self.audio_sample_rate = audio_sample_rate
        self.audio_n_mels = audio_n_mels
        self.audio_n_fft = audio_n_fft
        self.audio_hop_length = audio_hop_length

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

                        self.log_audio(writer, gt_waveform, global_step, tag=f"vocoder_reconstruction/waveform/target/{e}")
                        self.log_audio(writer, pred_waveform, global_step, tag=f"vocoder_reconstruction/waveform/output/{e}")

                        # Save reconstructed audio to file
                        audio_filepath = os.path.join(self.trainer.args.output_dir, f"reconstructed_audio_step_{global_step}.wav")
                        self._save_audio_to_file(pred_waveform, audio_filepath, sample_rate=self.audio_sample_rate)

                        self.log_mel_spec_visualization(writer, test_mel_spec, global_step, tag=f"vocoder_reconstruction/mel_spec/target/{e}")
                        # Reconstructed waveform's mel spectrogram
                        reconstructed_mel = audio_loading.extract_mels(
                            self.shared_window_buffer,
                            pred_waveform,
                            sr=self.audio_sample_rate,
                            n_mels=self.audio_n_mels,
                            n_fft=self.audio_n_fft,
                            hop_length=self.audio_hop_length,
                        )
                        self.log_mel_spec_visualization(writer, reconstructed_mel, global_step, tag=f"vocoder_reconstruction/mel_spec/output/{e}")


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
                        self.log_phase_error(writer, reconstructed_stft, target_stft, global_step, tag=f"vocoder_reconstruction/phase_comparison/{e}")
                        self.log_instantaneous_frequency(writer, target_stft, global_step, tag=f"vocoder_reconstruction/instantaneous_frequency/target/{e}")
                        self.log_instantaneous_frequency(writer, reconstructed_stft, global_step, tag=f"vocoder_reconstruction/instantaneous_frequency/output/{e}")

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

    def log_audio(self, writer: SummaryWriter, waveform: torch.Tensor, global_step, tag: str):
        writer.add_audio(tag, waveform, global_step, sample_rate=self.audio_sample_rate)

    def log_mel_spec_visualization(self, writer: SummaryWriter, mel_spec: torch.Tensor, global_step: int, tag: str):
        writer.add_image(tag, self._visualize_mel_spec(mel_spec.numpy(), self.audio_sample_rate), global_step)

    def log_phase_error(self, writer: SummaryWriter, pred_stft, target_stft, global_step: int, tag: str):
        """Log phase error map to tensorboard."""
        pred_phase = torch.angle(pred_stft)
        target_phase = torch.angle(target_stft)
        
        # Match lengths
        min_t = min(pred_phase.shape[-1], target_phase.shape[-1])
        pred_phase = pred_phase[..., :min_t]
        target_phase = target_phase[..., :min_t]
        
        # Anti-wrapped error, 0 = perfect, π = worst
        error = torch.atan2(
            torch.sin(pred_phase - target_phase),
            torch.cos(pred_phase - target_phase)
        ).abs()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(error.cpu().numpy(), aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=np.pi)
        plt.colorbar(im, ax=ax, label='Phase error (radians)')
        ax.set_ylabel('Frequency bin')
        ax.set_xlabel('Time frame')
        ax.set_title('Phase Error Map')
        
        writer.add_figure(tag, fig, global_step)
        plt.close(fig)

    def log_instantaneous_frequency(self, writer: SummaryWriter, stft, global_step: int, tag: str):
        """Log instantaneous frequency deviation to tensorboard."""
        phase = torch.angle(stft)
        
        inst_freq = torch.diff(phase, dim=-1)
        inst_freq = torch.atan2(torch.sin(inst_freq), torch.cos(inst_freq))
        inst_freq_hz = inst_freq * self.audio_sample_rate / (2 * np.pi * self.audio_hop_length)
        
        # Correct range based on hop_length
        max_if = self.audio_sample_rate / (2 * self.audio_hop_length)  # ~31 Hz for your params
        
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(inst_freq_hz.cpu().numpy(), aspect='auto', origin='lower',
                    cmap='coolwarm', vmin=-max_if, vmax=max_if)
        plt.colorbar(im, ax=ax, label='Frequency deviation (Hz)')
        ax.set_ylabel('Frequency bin')
        ax.set_xlabel('Time frame')
        ax.set_title('Instantaneous Frequency Deviation')
        
        writer.add_figure(tag, fig, global_step)
        plt.close(fig)

    def log_if_comparison(self, writer: SummaryWriter, pred_stft, target_stft, global_step: int, tag: str):
        def compute_if(stft):
            phase = torch.angle(stft)
            inst_freq = torch.diff(phase, dim=-1)
            inst_freq = torch.atan2(torch.sin(inst_freq), torch.cos(inst_freq))
            return inst_freq * self.audio_sample_rate / (2 * np.pi * self.audio_hop_length)
        
        pred_if = compute_if(pred_stft).cpu().numpy()
        target_if = compute_if(target_stft).cpu().numpy()
        
        min_t = min(pred_if.shape[-1], target_if.shape[-1])
        pred_if = pred_if[..., :min_t]
        target_if = target_if[..., :min_t]
        
        max_if = self.audio_sample_rate / (2 * self.audio_hop_length)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        im0 = axes[0].imshow(target_if, aspect='auto', origin='lower', cmap='coolwarm', vmin=-max_if, vmax=max_if)
        axes[0].set_title('Target IF')
        axes[0].set_ylabel('Frequency bin')
        axes[0].set_xlabel('Time frame')
        plt.colorbar(im0, ax=axes[0], label='Hz')
        
        im1 = axes[1].imshow(pred_if, aspect='auto', origin='lower', cmap='coolwarm', vmin=-max_if, vmax=max_if)
        axes[1].set_title('Predicted IF')
        axes[1].set_ylabel('Frequency bin')
        axes[1].set_xlabel('Time frame')
        plt.colorbar(im1, ax=axes[1], label='Hz')
        
        # Error map: use 0 to max_if since it's absolute difference
        im2 = axes[2].imshow(np.abs(pred_if - target_if), aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=max_if)
        axes[2].set_title('IF Error')
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
    """

    def __init__(
        self,
        *args,
        n_fft,
        hop_length,
        discriminator: Optional[CombinedDiscriminator] = None,
        discriminator_optimizer: Optional[torch.optim.Optimizer] = None,
        gan_adv_loss_weight: float = 1.0,
        gan_feature_matching_loss_weight: float = 2.0,
        discriminator_update_frequency: int = 1,
        gan_start_step: int = 0,
        mpd_loss_weight: float = 1.0,
        msd_loss_weight: float = 1.0,
        mrsd_loss_weight: float = 1.0,
        mpd_adv_loss_weight: float = 1.0,
        msd_adv_loss_weight: float = 1.0,
        mrsd_adv_loss_weight: float = 1.0,
        mpd_fm_loss_weight: float = 1.0,
        msd_fm_loss_weight: float = 1.0,
        mrsd_fm_loss_weight: float = 1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.gan_adv_loss_weight = gan_adv_loss_weight
        self.gan_feature_matching_loss_weight = gan_feature_matching_loss_weight
        self.discriminator_update_frequency = discriminator_update_frequency
        self.gan_start_step = gan_start_step
        self.mpd_loss_weight = mpd_loss_weight
        self.msd_loss_weight = msd_loss_weight
        self.mrsd_loss_weight = mrsd_loss_weight
        self.mpd_adv_loss_weight = mpd_adv_loss_weight
        self.msd_adv_loss_weight = msd_adv_loss_weight
        self.mrsd_adv_loss_weight = mrsd_adv_loss_weight
        self.mpd_fm_loss_weight = mpd_fm_loss_weight
        self.msd_fm_loss_weight = msd_fm_loss_weight
        self.mrsd_fm_loss_weight = mrsd_fm_loss_weight
        self.writer = None

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
        self._ensure_tensorboard_writer()

        # Forward pass through generator (vocoder)
        waveform_labels = inputs["waveform_labels"]
        outputs = model(
            mel_spec=inputs["mel_spec"],
            waveform_labels=waveform_labels,
            target_complex_stfts = inputs["target_complex_stfts"],
        )

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

        # GAN losses (only after gan_start_step)
        gan_enabled = (
            self.discriminator is not None and
            self.state.global_step >= self.gan_start_step
        )

        g_loss_gan = torch.tensor(0.0, device=pred_waveform.device)
        g_loss_fm = torch.tensor(0.0, device=pred_waveform.device)
        d_loss = torch.tensor(0.0, device=pred_waveform.device)

        if gan_enabled:
            # Discriminator Update
            if self.state.global_step % self.discriminator_update_frequency == 0:
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

                if self.state.global_step % self.args.logging_steps == 0 and self.writer is not None:
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

                # Update discriminator
                if self.discriminator_optimizer is not None and self.discriminator.training:
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

            if self.state.global_step % self.args.logging_steps == 0 and self.writer is not None:
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
        if self.state.global_step % self.args.logging_steps == 0 and self.writer is not None:
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
            self._log_scalar(f"{prefix}recon_loss", recon_loss)
            self._log_scalar(f"{prefix}total_loss", total_loss)

        return (total_loss, outputs) if return_outputs else total_loss

    def _log_scalar(self, tag, value):
        if self.writer is not None:
            if isinstance(value, torch.Tensor):
                value = value.item()
            if value != 0.0:
                self.writer.add_scalar(tag, value, self.state.global_step)

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

        gan_enabled = (
            self.discriminator is not None and
            self.state.global_step >= self.gan_start_step
        )

        # Save discriminator
        if gan_enabled:
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

    # Select model configuration
    if args.config not in vocoders.model_config_lookup:
        raise ValueError(f"Unknown vocoder config: {args.config}. Available: {list(vocoders.model_config_lookup.keys())}")

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
    mel_window = unk_dict.get("mel_window", "hann_window")

    # GAN training settings
    use_gan = unk_dict.get("use_gan", "false").lower() == "true"
    gan_start_step = int(unk_dict.get("gan_start_step", 0))
    discriminator_lr = float(unk_dict.get("discriminator_lr", 2e-4))

    gan_adv_loss_weight = float(unk_dict.get("gan_adv_loss_weight", 1.0))
    gan_feature_matching_loss_weight = float(unk_dict.get("gan_feature_matching_loss_weight", 2.0))
    
    mpd_loss_weight = float(unk_dict.get("mpd_loss_weight", 1.0))
    msd_loss_weight = float(unk_dict.get("msd_loss_weight", 1.0))
    mrsd_loss_weight = float(unk_dict.get("mrsd_loss_weight", 1.0))
    mpd_adv_loss_weight = float(unk_dict.get("mpd_adv_loss_weight", 1.0))
    msd_adv_loss_weight = float(unk_dict.get("msd_adv_loss_weight", 1.0))
    mrsd_adv_loss_weight = float(unk_dict.get("mrsd_adv_loss_weight", 1.0))
    mpd_fm_loss_weight = float(unk_dict.get("mpd_fm_loss_weight", 1.0))
    msd_fm_loss_weight = float(unk_dict.get("msd_fm_loss_weight", 1.0))
    mrsd_fm_loss_weight = float(unk_dict.get("mrsd_fm_loss_weight", 1.0))

    discriminator_config = unk_dict.get("discriminator_config", "combined_disc")
    
    shared_window_buffer = SharedWindowBuffer()

    model = vocoders.model_config_lookup[args.config](
        shared_window_buffer,
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
    )
    model, model_loaded = megatransformer_utils.load_model(False, model, run_dir)

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
            print(f"  GAN start step: {gan_start_step}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"  Vocoder parameters: {sum(p.numel() for p in model.vocoder.parameters()):,}")
        print(f"    Vocoder initial_conv parameters: {sum(p.numel() for p in model.vocoder.initial_conv.parameters()):,}")
        print(f"    Vocoder upsample_blocks parameters: {sum(p.numel() for p in model.vocoder.upsample_blocks.parameters()):,}")
        print(f"    Vocoder residual_blocks parameters: {sum(p.numel() for p in model.vocoder.residual_blocks.parameters()):,}")
        print(f"    Vocoder final_layers parameters: {sum(p.numel() for p in model.vocoder.final_layers.parameters()):,}")
        if use_gan and discriminator is not None:
            print(f"  Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
            print(f"    MPD parameters: {sum(p.numel() for p in discriminator.mpd.parameters()):,}")
            print(f"    MSD parameters: {sum(p.numel() for p in discriminator.msd.parameters()):,}")
            print(f"    MRSD parameters: {sum(p.numel() for p in discriminator.mrsd.parameters()):,}")

    model = megatransformer_utils.setup_int8_training(args, model)

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
        lr_scheduler_type="cosine",
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs if args.num_train_epochs > 0 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        weight_decay=args.weight_decay,
        report_to="tensorboard",
        logging_dir=run_dir,
        logging_steps=args.logging_steps,
        # eval_steps=args.eval_steps,
        # eval_strategy="epoch" if args.eval_steps == 0 else "steps",
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
        remove_unused_columns=False
    )

    train_dataset = CachedVocoderDataset(
        cache_dir="./cached_datasets/librispeech_train_cached",
        audio_max_frames=model.config.audio_max_frames,
    )

    eval_dataset = CachedVocoderDataset(
        cache_dir="./cached_datasets/librispeech_val_cached",
        audio_max_frames=model.config.audio_max_frames,
    )

    # Create data collator
    data_collator = VocoderDataCollator(
        audio_max_frames=model.config.audio_max_frames,
        audio_max_waveform_length=model.config.audio_max_waveform_length,
        n_mels=model.config.audio_n_mels,
    )

    # Create trainer (with or without GAN)
    trainer = VocoderGANTrainer(
        model=model,
        n_fft=model.config.audio_n_fft,
        hop_length=model.config.audio_hop_length,
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
        gan_start_step=gan_start_step,
        mpd_adv_loss_weight=mpd_adv_loss_weight,
        msd_adv_loss_weight=msd_adv_loss_weight,
        mrsd_adv_loss_weight=mrsd_adv_loss_weight,
        mpd_fm_loss_weight=mpd_fm_loss_weight,
        msd_fm_loss_weight=msd_fm_loss_weight,
        mrsd_fm_loss_weight=mrsd_fm_loss_weight,
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
