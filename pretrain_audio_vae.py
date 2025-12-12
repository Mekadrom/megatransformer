import os

import torchaudio

from dataset_loading import audio_loading
from dataset_loading.audio_diffusion_dataset import CachedAudioDiffusionDataset
from model.audio.shared_window_buffer import SharedWindowBuffer
from model.audio.vae import model_config_lookup
from model.audio.vocoders.vocoders import model_config_lookup as vocoder_config_lookup
from model.audio.discriminators import (
    mel_discriminator_config_lookup,
    compute_mel_discriminator_loss,
    compute_mel_generator_gan_loss,
)

os.environ["DEEPSPEED_UNIT_TEST"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ['NCCL_TIMEOUT'] = '1200000'

from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers.integrations import TensorBoardCallback
from typing import Any, Dict, List, Mapping, Optional, Union

import librosa
import librosa.display
import matplotlib.pyplot as plt
import megatransformer_utils
import numpy as np
import torch
import torch.nn.functional as F


def get_writer(trainer: Trainer) -> Optional[SummaryWriter]:
    """Get TensorBoard writer from trainer callbacks."""
    if hasattr(trainer, 'callback_handler'):
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback):
                if callback.tb_writer is not None:
                    return callback.tb_writer
    return None


class AudioVAEDataCollator:
    """Data collator for audio VAE training. Reuses diffusion dataset, discards text."""
    def __init__(
        self,
        audio_max_frames: int = 1875,
        n_mels: int = 80,
    ):
        self.audio_max_frames = audio_max_frames
        self.n_mels = n_mels

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        mel_specs = []
        mel_spec_masks = []
        for ex in examples:
            if ex is None:
                continue

            mel = ex["mel_spec"]
            mel_length = ex.get("mel_spec_length", mel.shape[-1])

            # Ensure correct shape [1, n_mels, T] for single-channel input
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)  # [n_mels, T] -> [1, n_mels, T]
            elif mel.dim() == 3 and mel.shape[0] != 1:
                # If shape is [n_mels, T, 1] or similar, fix it
                if mel.shape[-1] == 1:
                    mel = mel.squeeze(-1).unsqueeze(0)

            # Create mel spec padding mask (1 = valid, 0 = padding)
            mel_mask = torch.ones(self.audio_max_frames, dtype=torch.float32)
            if mel_length < self.audio_max_frames:
                mel_mask[mel_length:] = 0.0

            # Pad or truncate time dimension
            if mel.shape[-1] < self.audio_max_frames:
                mel = F.pad(mel, (0, self.audio_max_frames - mel.shape[-1]), value=0)
            elif mel.shape[-1] > self.audio_max_frames:
                mel = mel[..., :self.audio_max_frames]
                mel_mask[:] = 1.0  # All valid if truncated

            mel_specs.append(mel)
            mel_spec_masks.append(mel_mask)

        batch = {
            "mel_spec": torch.stack(mel_specs),
            "mel_spec_mask": torch.stack(mel_spec_masks),  # [B, T] mask for mel spectrogram
        }

        return batch


class AudioVAEReconstructionCallback(TrainerCallback):
    """
    Callback for logging VAE mel spectrogram reconstruction during training.
    Periodically reconstructs test mel specs and logs to TensorBoard.
    Optionally converts mel spectrograms to audio using a vocoder.
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
        audio_max_frames: int = 1875,
        vocoder_checkpoint_path: Optional[str] = None,
        vocoder_config: str = "experimental",
    ):
        self.shared_window_buffer = shared_window_buffer

        self.trainer: Optional[Trainer] = None
        self.step_offset = step_offset if step_offset is not None else 0
        self.generation_steps = generation_steps
        self.audio_sample_rate = audio_sample_rate
        self.audio_n_mels = audio_n_mels
        self.audio_n_fft = audio_n_fft
        self.audio_hop_length = audio_hop_length
        self.audio_max_frames = audio_max_frames

        # Vocoder settings
        self.vocoder_checkpoint_path = vocoder_checkpoint_path
        self.vocoder_config = vocoder_config
        self.vocoder = None
        self._vocoder_load_attempted = False

        # Load example audio files and compute mel specs
        self.example_paths = [
            "inference/examples/test_alm_1.mp3",
            "inference/examples/test_alm_2.mp3",
        ]
        self.example_mels = []
        self.example_original_lengths = []  # Store original mel lengths before padding

        for path in self.example_paths:
            if os.path.exists(path):
                try:
                    waveform, orig_sr = torchaudio.load(path)
                    # Resample if needed
                    if orig_sr != audio_sample_rate:
                        waveform = torchaudio.transforms.Resample(
                            orig_freq=orig_sr, new_freq=audio_sample_rate
                        )(waveform)
                    # Convert to mono if stereo
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)

                    # Compute mel spectrogram
                    mel = audio_loading.extract_mels(
                        self.shared_window_buffer,
                        waveform,
                        self.audio_sample_rate,
                        self.audio_n_mels,
                        self.audio_n_fft,
                        self.audio_hop_length
                    )

                    # Store original length before padding
                    original_length = mel.shape[-1]

                    # Pad or truncate to max frames
                    if mel.shape[-1] < audio_max_frames:
                        mel = F.pad(mel, (0, audio_max_frames - mel.shape[-1]), value=0)
                    elif mel.shape[-1] > audio_max_frames:
                        mel = mel[..., :audio_max_frames]
                        original_length = audio_max_frames  # Truncated, so original is the max

                    # Add channel dimension: [n_mels, T] -> [1, n_mels, T]
                    if mel.dim() == 2:
                        mel = mel.unsqueeze(0)

                    self.example_mels.append(mel)
                    self.example_original_lengths.append(original_length)
                    print(f"Loaded example mel from {path}: shape {mel.shape}, original length {original_length}")
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
            else:
                print(f"Example audio not found: {path}")

    def _load_vocoder(self):
        """Lazily load vocoder on first use."""
        if self._vocoder_load_attempted:
            return
        self._vocoder_load_attempted = True

        if self.vocoder_checkpoint_path is None:
            return

        if not os.path.exists(self.vocoder_checkpoint_path):
            print(f"Vocoder checkpoint not found at {self.vocoder_checkpoint_path}")
            return
        else:
            print(f"Loading vocoder from {self.vocoder_checkpoint_path}...")

        try:
            vocoder = vocoder_config_lookup[self.vocoder_config](
                shared_window_buffer=self.shared_window_buffer,
            )

            print(f"Loading vocoder model from {self.vocoder_checkpoint_path}...")

            megatransformer_utils.load_model(False, vocoder, self.vocoder_checkpoint_path)

            vocoder.eval()
            self.vocoder = vocoder
            print(f"Loaded vocoder from {self.vocoder_checkpoint_path}")
            print(f"Vocoder parameters: {sum(p.numel() for p in vocoder.parameters()):,}")
        except Exception as e:
            print(f"Failed to load vocoder: {e}")
            self.vocoder = None

    def _log_vocoder_audio(self, writer: SummaryWriter, mel_spec: torch.Tensor, global_step: int, tag: str):
        """Convert mel spectrogram to audio using vocoder and log to TensorBoard."""
        try:
            # Ensure mel is [B, n_mels, T] for vocoder
            if mel_spec.dim() == 2:
                mel_spec = mel_spec.unsqueeze(0)  # [1, n_mels, T]

            # Run vocoder on CPU (float32 for stability)
            mel_spec = mel_spec.float()

            with torch.no_grad():
                outputs = self.vocoder(mel_spec)
                if isinstance(outputs, dict):
                    waveform = outputs["pred_waveform"]
                else:
                    waveform = outputs

            # Ensure 1D waveform
            if waveform.dim() > 1:
                waveform = waveform.squeeze()

            # Normalize to [-1, 1] range
            waveform = waveform / (waveform.abs().max() + 1e-8)

            # Log audio to TensorBoard
            writer.add_audio(
                tag,
                waveform.numpy(),
                global_step,
                sample_rate=self.audio_sample_rate
            )
        except Exception as e:
            print(f"Failed to generate audio with vocoder: {e}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        global_step = state.global_step + self.step_offset

        if ((global_step == 1) or (global_step % self.generation_steps == 0)) and state.is_world_process_zero:
            writer = get_writer(self.trainer)
            if writer is None:
                print("No TensorBoard writer found, skipping visualization...")
                return

            if len(self.example_mels) == 0:
                print("No example mels loaded, skipping visualization...")
                return

            print(f"Generating mel reconstructions at step {global_step}...")

            # Lazily load vocoder
            self._load_vocoder()

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
                    for i, mel in enumerate(self.example_mels):
                        recon, mu, logvar, losses = model(mel.unsqueeze(0).to(device))

                        # Get original length for this example
                        original_length = self.example_original_lengths[i] if i < len(self.example_original_lengths) else mel.shape[-1]

                        # Get tensors for visualization
                        mel_cpu = mel.squeeze(0).cpu().numpy()  # [n_mels, T]
                        recon_cpu = recon[0].squeeze(0).float().cpu().numpy()  # [n_mels, T]

                        # Trimmed versions (without padding)
                        mel_trimmed = mel_cpu[..., :original_length]
                        recon_trimmed = recon_cpu[..., :original_length]

                        # Log trimmed (content only) mel spectrograms
                        writer.add_image(
                            f"audio_vae/original_trimmed/{i}",
                            self._visualize_mel_spec(mel_trimmed),
                            global_step
                        )
                        writer.add_image(
                            f"audio_vae/recon_trimmed/{i}",
                            self._visualize_mel_spec(recon_trimmed),
                            global_step
                        )

                        # Log trimmed comparison
                        self._log_mel_comparison(
                            writer, recon_trimmed, mel_trimmed, global_step,
                            tag=f"audio_vae/comparison_trimmed/{i}"
                        )

                        # Log padded (full) mel spectrograms for diagnosing silence reconstruction
                        writer.add_image(
                            f"audio_vae/original_padded/{i}",
                            self._visualize_mel_spec(mel_cpu),
                            global_step
                        )
                        writer.add_image(
                            f"audio_vae/recon_padded/{i}",
                            self._visualize_mel_spec(recon_cpu),
                            global_step
                        )

                        # Log padded comparison
                        self._log_mel_comparison(
                            writer, recon_cpu, mel_cpu, global_step,
                            tag=f"audio_vae/comparison_padded/{i}"
                        )

                        # Log per-example losses
                        for loss_name, loss_val in losses.items():
                            if isinstance(loss_val, torch.Tensor):
                                loss_val = loss_val.item()
                            writer.add_scalar(f"audio_vae/example_{i}/{loss_name}", loss_val, global_step)

                        # Log latent channel visualizations
                        mu_sample = mu[0].float().cpu()  # [latent_channels, H, W]
                        mu_min, mu_max = mu_sample.min(), mu_sample.max()
                        mu_norm = (mu_sample - mu_min) / (mu_max - mu_min + 1e-5)
                        for c in range(mu_norm.shape[0]):
                            writer.add_image(
                                f"audio_vae/example_{i}/mu_channel_{c}",
                                mu_norm[c:c+1, :, :],
                                global_step
                            )

                        # Convert reconstructed mel to audio using vocoder (trimmed version)
                        if self.vocoder is not None:
                            recon_mel_tensor = recon[0].squeeze(0).float().cpu()  # [n_mels, T]
                            # Log trimmed audio
                            self._log_vocoder_audio(
                                writer, recon_mel_tensor[..., :original_length], global_step,
                                tag=f"audio_vae/recon_audio_trimmed/{i}"
                            )
                            # Log full padded audio for comparison
                            self._log_vocoder_audio(
                                writer, recon_mel_tensor, global_step,
                                tag=f"audio_vae/recon_audio_padded/{i}"
                            )

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


class AudioVAEGANTrainer(Trainer):
    """
    Custom trainer for VAE with optional GAN training.
    Handles alternating generator/discriminator updates.
    """

    def __init__(
        self,
        *args,
        cmdline,
        git_commit_hash,
        step_offset: int = 0,
        discriminator: Optional[torch.nn.Module] = None,
        discriminator_optimizer: Optional[torch.optim.Optimizer] = None,
        gan_loss_weight: float = 0.5,
        feature_matching_weight: float = 0.0,
        discriminator_update_frequency: int = 1,
        gan_start_step: int = 0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.writer = None

        self.step_offset = step_offset if step_offset is not None else 0
        self.cmdline = cmdline
        self.git_commit_hash = git_commit_hash

        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.gan_loss_weight = gan_loss_weight
        self.feature_matching_weight = feature_matching_weight
        self.discriminator_update_frequency = discriminator_update_frequency
        self.gan_start_step = gan_start_step

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model.
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

        self._ensure_tensorboard_writer()

        if global_step == 0 and self.writer is not None:
            self.writer.add_text("training/command_line", self.cmdline, global_step)
            self.writer.add_text("training/git_commit_hash", self.git_commit_hash, global_step)

        mel_spec = inputs["mel_spec"]
        mel_spec_mask = inputs.get("mel_spec_mask", None)

        # Forward pass through VAE model (with optional mask for reconstruction loss)
        recon, mu, logvar, losses = model(mel_spec, mask=mel_spec_mask)

        # Get VAE reconstruction loss
        vae_loss = losses["total_loss"]

        # GAN losses (only after gan_start_step)
        gan_enabled = (
            self.discriminator is not None and
            global_step >= self.gan_start_step
        )

        g_gan_loss = torch.tensor(0.0, device=mel_spec.device)
        d_loss = torch.tensor(0.0, device=mel_spec.device)

        if gan_enabled:
            # Discriminator Update
            if global_step % self.discriminator_update_frequency == 0:
                self.discriminator.train()

                # Get discriminator loss
                device_type = mel_spec.device.type
                dtype = torch.bfloat16 if self.args.bf16 else torch.float16 if self.args.fp16 else torch.float32
                with autocast(device_type, dtype=dtype, enabled=self.args.fp16 or self.args.bf16):
                    d_loss, d_loss_dict = compute_mel_discriminator_loss(
                        self.discriminator,
                        real_mels=mel_spec,
                        fake_mels=recon.detach(),
                    )

                if global_step % self.args.logging_steps == 0 and self.writer is not None:
                    for key, val in d_loss_dict.items():
                        self._log_scalar(f"train/{key}", val, global_step)

                # Update discriminator
                if self.discriminator_optimizer is not None and self.discriminator.training:
                    self.discriminator_optimizer.zero_grad()
                    d_loss.backward()
                    self.discriminator_optimizer.step()

            # Generator GAN Loss
            device_type = mel_spec.device.type
            dtype = torch.bfloat16 if self.args.bf16 else torch.float16 if self.args.fp16 else torch.float32
            with autocast(device_type, dtype=dtype, enabled=self.args.fp16 or self.args.bf16):
                g_gan_loss, g_loss_dict = compute_mel_generator_gan_loss(
                    self.discriminator,
                    real_mels=mel_spec,
                    fake_mels=recon,
                    feature_matching_weight=self.feature_matching_weight,
                )

            if global_step % self.args.logging_steps == 0 and self.writer is not None:
                for key, val in g_loss_dict.items():
                    self._log_scalar(f"train/{key}", val, global_step)

        # Total generator loss
        total_loss = vae_loss + self.gan_loss_weight * g_gan_loss

        # Log losses
        if global_step % self.args.logging_steps == 0 and self.writer is not None:
            prefix = "train/" if model.training else "eval/"
            for loss_name, loss in losses.items():
                self._log_scalar(f"{prefix}vae_{loss_name}", loss.mean(), global_step)
            # Log mu and logvar stats
            self._log_scalar(f"{prefix}vae_mu_mean", mu.mean(), global_step)
            self._log_scalar(f"{prefix}vae_mu_std", mu.std(), global_step)
            self._log_scalar(f"{prefix}vae_logvar_mean", logvar.mean(), global_step)
            self._log_scalar(f"{prefix}g_gan_loss", g_gan_loss, global_step)
            self._log_scalar(f"{prefix}total_loss", total_loss.mean(), global_step)

        outputs = {
            "loss": total_loss,
            "rec": recon,
        }

        return (total_loss, outputs) if return_outputs else total_loss

    def _log_scalar(self, tag, value, global_step):
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
        """Save both VAE and discriminator."""
        super().save_model(output_dir, _internal_call)

        global_step = self.state.global_step + self.step_offset

        if output_dir is None:
            output_dir = self.args.output_dir

        gan_enabled = (
            self.discriminator is not None and
            global_step >= self.gan_start_step
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
    discriminator: torch.nn.Module,
    discriminator_optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.nn.Module, Optional[torch.optim.Optimizer], bool]:
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
    if args.config not in model_config_lookup:
        raise ValueError(f"Unknown audio VAE config: {args.config}. Available: {list(model_config_lookup.keys())}")

    # Parse extra arguments
    unk_dict = {}
    for i in range(0, len(unk), 2):
        unk_dict[unk[i].lstrip('-')] = unk[i+1]

    # Dataset settings (reuse diffusion dataset paths)
    train_cache_dir = unk_dict.get("train_cache_dir", "./cached_datasets/librispeech_train_diffusion_full_speakers_cached")
    val_cache_dir = unk_dict.get("val_cache_dir", "./cached_datasets/librispeech_val_diffusion_full_speakers_cached")

    # Audio settings
    audio_max_frames = int(unk_dict.get("audio_max_frames", 1875))
    n_mels = int(unk_dict.get("n_mels", 80))
    audio_sample_rate = int(unk_dict.get("audio_sample_rate", 16000))
    audio_n_fft = int(unk_dict.get("audio_n_fft", 1024))
    audio_hop_length = int(unk_dict.get("audio_hop_length", 256))

    # VAE settings
    latent_channels = int(unk_dict.get("latent_channels", 4))

    # VAE loss weights
    recon_loss_weight = float(unk_dict.get("recon_loss_weight", 1.0))
    mse_loss_weight = float(unk_dict.get("mse_loss_weight", 1.0))
    l1_loss_weight = float(unk_dict.get("l1_loss_weight", 0.0))
    kl_divergence_loss_weight = float(unk_dict.get("kl_divergence_loss_weight", 0.001))

    # Perceptual loss type: "vgg", "lpips", or "none" EXPERIMENTAL
    perceptual_loss_weight = float(unk_dict.get("perceptual_loss_weight", 0.0))
    perceptual_loss_type = unk_dict.get("perceptual_loss_type", "none")
    lpips_net = unk_dict.get("lpips_net", "vgg")  # "alex", "vgg", or "squeeze"

    # Vocoder settings (optional - for audio generation during visualization)
    vocoder_checkpoint_path = unk_dict.get("vocoder_checkpoint_path", None)
    vocoder_config = unk_dict.get("vocoder_config", "experimental")

    # GAN training settings
    use_gan = unk_dict.get("use_gan", "false").lower() == "true"
    gan_start_step = int(unk_dict.get("gan_start_step", 0))
    discriminator_lr = float(unk_dict.get("discriminator_lr", 2e-4))
    gan_loss_weight = float(unk_dict.get("gan_loss_weight", 0.5))
    feature_matching_weight = float(unk_dict.get("feature_matching_weight", 0.0))
    discriminator_config = unk_dict.get("discriminator_config", "mini_multi_scale")

    # Create shared window buffer for audio processing
    shared_window_buffer = SharedWindowBuffer()

    model = model_config_lookup[args.config](
        latent_channels=latent_channels,
        perceptual_loss_type=perceptual_loss_type,
        lpips_net=lpips_net,
        recon_loss_weight=recon_loss_weight,
        mse_loss_weight=mse_loss_weight,
        l1_loss_weight=l1_loss_weight,
        kl_divergence_loss_weight=kl_divergence_loss_weight,
        perceptual_loss_weight=perceptual_loss_weight,
    )

    # Try to load existing checkpoint
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
        if discriminator_config not in mel_discriminator_config_lookup:
            raise ValueError(f"Unknown discriminator config: {discriminator_config}. Available: {list(mel_discriminator_config_lookup.keys())}")

        discriminator = mel_discriminator_config_lookup[discriminator_config]().to(device)

        discriminator_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=discriminator_lr,
            betas=(0.0, 0.99),
            weight_decay=0.0,
        )

        # Try to load existing discriminator checkpoint
        discriminator, discriminator_optimizer, disc_loaded = load_discriminator(
            args.resume_from_checkpoint, discriminator, discriminator_optimizer, device
        )
        if disc_loaded:
            print("Loaded discriminator from checkpoint")

    if args.local_rank == 0 or not args.use_deepspeed:
        print(f"Model structure: {model}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  VAE Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
        print(f"  VAE Decoder parameters: {sum(p.numel() for p in model.decoder.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"Audio settings:")
        print(f"  Sample rate: {audio_sample_rate}")
        print(f"  N mels: {n_mels}")
        print(f"  N FFT: {audio_n_fft}")
        print(f"  Hop length: {audio_hop_length}")
        print(f"  Max frames: {audio_max_frames}")
        print(f"  Latent channels: {latent_channels}")
        if use_gan and discriminator is not None:
            print(f"GAN training: enabled")
            print(f"  Discriminator config: {discriminator_config}")
            print(f"  Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
            print(f"  GAN loss weight: {gan_loss_weight}")
            print(f"  Feature matching weight: {feature_matching_weight}")
            print(f"  Discriminator LR: {discriminator_lr}")
            print(f"  GAN start step: {gan_start_step}")

    model = megatransformer_utils.setup_int8_training(args, model)

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

    # Load datasets
    train_dataset = CachedAudioDiffusionDataset(
        cache_dir=train_cache_dir,
        audio_max_frames=audio_max_frames,
    )

    eval_dataset = CachedAudioDiffusionDataset(
        cache_dir=val_cache_dir,
        audio_max_frames=audio_max_frames,
    )

    # Create data collator
    data_collator = AudioVAEDataCollator(
        audio_max_frames=audio_max_frames,
        n_mels=n_mels,
    )

    # Create trainer
    trainer = AudioVAEGANTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        cmdline=args.cmdline,
        git_commit_hash=args.commit_hash,
        step_offset=args.start_step,
        discriminator=discriminator if use_gan else None,
        discriminator_optimizer=discriminator_optimizer if use_gan else None,
        gan_loss_weight=gan_loss_weight,
        feature_matching_weight=feature_matching_weight,
        gan_start_step=gan_start_step,
    )

    # Add visualization callback
    visualization_callback = AudioVAEReconstructionCallback(
        shared_window_buffer=shared_window_buffer,
        step_offset=args.start_step,
        generation_steps=args.generation_steps,
        audio_sample_rate=audio_sample_rate,
        audio_n_mels=n_mels,
        audio_n_fft=audio_n_fft,
        audio_hop_length=audio_hop_length,
        audio_max_frames=audio_max_frames,
        vocoder_checkpoint_path=vocoder_checkpoint_path,
        vocoder_config=vocoder_config,
    )
    trainer.add_callback(visualization_callback)

    if args.stop_step > 0:
        early_stopping_callback = megatransformer_utils.EarlyStoppingCallback(stop_step=args.stop_step)
        trainer.add_callback(early_stopping_callback)

    visualization_callback.trainer = trainer

    # Log scheduler info
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

    print(f"Starting audio VAE training with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
