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
import torch.nn.functional as F
import torchaudio

from typing import Any, Dict, List, Mapping, Optional, Union

from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers.integrations import TensorBoardCallback

from dataset_loading import audio_loading
from dataset_loading.audio_diffusion_dataset import CachedAudioDiffusionDataset
from shard_utils import AudioVAEShardedDataset, ShardAwareSampler
from model.audio.discriminators import (
    MelMultiPeriodDiscriminator,
    MelMultiScaleDiscriminator,
    mel_discriminator_config_lookup,
    compute_mel_discriminator_loss,
    compute_mel_generator_gan_loss,
    add_mel_instance_noise,
    r1_mel_gradient_penalty,
    MelInstanceNoiseScheduler,
)
from model.audio.criteria import AudioPerceptualLoss
from model.audio.vae import model_config_lookup
from model.audio.vocoders.vocoders import model_config_lookup as vocoder_config_lookup
from utils import megatransformer_utils
from utils.audio_utils import SharedWindowBuffer
from utils.model_loading_utils import load_model
from utils.training_utils import EarlyStoppingCallback, setup_int8_training


def get_writer(trainer: Trainer) -> Optional[SummaryWriter]:
    """Get TensorBoard writer from trainer callbacks."""
    if hasattr(trainer, 'callback_handler'):
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback):
                if callback.tb_writer is not None:
                    return callback.tb_writer
    return None


class AudioVAEDataCollator:
    """
    Data collator for audio VAE training.

    Pads to batch max length instead of global max for efficiency.
    Creates masks for both loss computation and attention masking.
    """
    def __init__(
        self,
        audio_max_frames: int = 1875,
        n_mels: int = 80,
        speaker_embedding_dim: int = 192,
    ):
        self.audio_max_frames = audio_max_frames
        self.n_mels = n_mels
        self.speaker_embedding_dim = speaker_embedding_dim

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        # Filter None examples and collect lengths first
        valid_examples = [ex for ex in examples if ex is not None]
        if not valid_examples:
            raise ValueError("All examples in batch are None")

        # Collect mel specs and lengths
        raw_mel_specs = []
        mel_lengths = []
        speaker_embeddings = []

        for ex in valid_examples:
            mel = ex["mel_spec"]
            mel_length = ex.get("mel_spec_length", mel.shape[-1])

            # Ensure correct shape [1, n_mels, T] for single-channel input
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)  # [n_mels, T] -> [1, n_mels, T]
            elif mel.dim() == 3 and mel.shape[0] != 1:
                # If shape is [n_mels, T, 1] or similar, fix it
                if mel.shape[-1] == 1:
                    mel = mel.squeeze(-1).unsqueeze(0)

            # Clamp length to actual mel length and global max
            mel_length = min(mel_length, mel.shape[-1], self.audio_max_frames)

            raw_mel_specs.append(mel)
            mel_lengths.append(mel_length)

            # Get speaker embedding if available
            speaker_emb = ex.get("speaker_embedding", None)
            if speaker_emb is not None:
                # Ensure shape is [1, speaker_embedding_dim]
                if speaker_emb.dim() == 1:
                    speaker_emb = speaker_emb.unsqueeze(0)
                speaker_embeddings.append(speaker_emb)
            else:
                # Use zeros if no speaker embedding
                speaker_embeddings.append(torch.zeros(1, self.speaker_embedding_dim))

        # Compute batch max length (dynamic padding)
        batch_max_length = max(mel_lengths)

        # Pad/truncate to batch max length and create masks
        mel_specs = []
        mel_spec_masks = []

        for mel, mel_length in zip(raw_mel_specs, mel_lengths):
            # Create mel spec padding mask (1 = valid, 0 = padding)
            mel_mask = torch.zeros(batch_max_length, dtype=torch.float32)
            mel_mask[:mel_length] = 1.0

            # Truncate to batch max length (since data is pre-padded to global max)
            mel = mel[..., :batch_max_length]

            # Pad if needed (shouldn't be needed if data is pre-padded, but just in case)
            if mel.shape[-1] < batch_max_length:
                mel = F.pad(mel, (0, batch_max_length - mel.shape[-1]), value=0)

            mel_specs.append(mel)
            mel_spec_masks.append(mel_mask)

        # Convert lengths to tensor for attention masking
        mel_lengths_tensor = torch.tensor(mel_lengths, dtype=torch.long)

        batch = {
            "mel_spec": torch.stack(mel_specs),
            "mel_spec_mask": torch.stack(mel_spec_masks),  # [B, T] mask for loss (1=valid, 0=padding)
            "mel_spec_lengths": mel_lengths_tensor,  # [B] original lengths for attention masking
            "speaker_embedding": torch.stack(speaker_embeddings),  # [B, 1, speaker_embedding_dim]
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
        num_eval_samples: int = 8,
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
        self.num_eval_samples = num_eval_samples

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

        try:
            vocoder = vocoder_config_lookup[self.vocoder_config](
                shared_window_buffer=self.shared_window_buffer,
            )

            load_model(False, vocoder, self.vocoder_checkpoint_path)

            # Remove weight normalization for inference optimization
            if hasattr(vocoder.vocoder, 'remove_weight_norm'):
                vocoder.vocoder.remove_weight_norm()

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

    def _get_device(self):
        """Determine the device to use for inference."""
        if torch.distributed.is_initialized():
            return torch.device(f"cuda:{torch.distributed.get_rank()}")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Generate and log reconstructions during evaluation from eval dataset."""
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

        print(f"Generating eval mel reconstructions at step {global_step}...")

        # Lazily load vocoder
        self._load_vocoder()

        device = self._get_device()
        model.eval()

        # Sample random indices from eval dataset
        num_samples = min(self.num_eval_samples, len(eval_dataset))
        indices = torch.randperm(len(eval_dataset))[:num_samples].tolist()

        # Collect aggregate statistics
        all_losses = {}
        all_mu_means = []
        all_mu_stds = []
        all_logvar_means = []

        with torch.no_grad():
            dtype = torch.bfloat16 if bool(args.bf16) else torch.float16 if args.fp16 else torch.float32

            with autocast(device.type, dtype=dtype, enabled=args.bf16 or args.fp16):
                for i, idx in enumerate(indices):
                    sample = eval_dataset[idx]
                    mel = sample["mel_spec"]
                    speaker_embedding = sample.get("speaker_embedding", None)

                    # Ensure correct shape [1, n_mels, T]
                    if mel.dim() == 2:
                        mel = mel.unsqueeze(0)

                    mel_length = sample.get("mel_spec_length", mel.shape[-1])

                    # Prepare speaker embedding if available
                    spk_emb = None
                    if speaker_embedding is not None:
                        spk_emb = speaker_embedding.unsqueeze(0).to(device)
                        if spk_emb.dim() == 2:
                            spk_emb = spk_emb.unsqueeze(1)  # [B, 1, D]

                    # Use reconstruct_with_attention to get attention weights
                    mel_input = mel.unsqueeze(0).to(device)
                    recon, mu, logvar, enc_attn, dec_attn = model.reconstruct_with_attention(
                        mel_input,
                        speaker_embedding=spk_emb,
                    )

                    # Generate mu-only reconstruction (no sampling, z = mu)
                    # This is what diffusion will see during inference
                    recon_mu_only = model.decode(mu)

                    # Compute losses manually for logging
                    losses = {}
                    losses["mse_loss"] = F.mse_loss(recon, mel_input)
                    losses["kl_divergence"] = -0.5 * torch.mean(
                        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])
                    )

                    # Collect statistics
                    all_mu_means.append(mu.mean().item())
                    all_mu_stds.append(mu.std().item())
                    all_logvar_means.append(logvar.mean().item())

                    for loss_name, loss_val in losses.items():
                        if loss_name not in all_losses:
                            all_losses[loss_name] = []
                        if isinstance(loss_val, torch.Tensor):
                            all_losses[loss_name].append(loss_val.item())
                        else:
                            all_losses[loss_name].append(loss_val)

                    # Get tensors for visualization
                    mel_cpu = mel.squeeze(0).cpu().numpy()
                    recon_cpu = recon[0].squeeze(0).float().cpu().numpy()
                    recon_mu_only_cpu = recon_mu_only[0].squeeze(0).float().cpu().numpy()

                    # Trimmed versions (without padding)
                    mel_trimmed = mel_cpu[..., :mel_length]
                    recon_trimmed = recon_cpu[..., :mel_length]
                    recon_mu_only_trimmed = recon_mu_only_cpu[..., :mel_length]

                    # Log individual mel spectrograms
                    writer.add_image(
                        f"eval_vae/original/{i}",
                        self._visualize_mel_spec(mel_trimmed),
                        global_step
                    )
                    writer.add_image(
                        f"eval_vae/reconstruction/{i}",
                        self._visualize_mel_spec(recon_trimmed),
                        global_step
                    )
                    writer.add_image(
                        f"eval_vae/reconstruction_mu_only/{i}",
                        self._visualize_mel_spec(recon_mu_only_trimmed),
                        global_step
                    )

                    # Log trimmed comparison
                    self._log_mel_comparison(
                        writer, recon_trimmed, mel_trimmed, global_step,
                        tag=f"eval_vae/comparison/{i}"
                    )
                    self._log_mel_comparison(
                        writer, recon_mu_only_trimmed, mel_trimmed, global_step,
                        tag=f"eval_vae/comparison_mu_only/{i}"
                    )

                    # Log per-example losses
                    for loss_name, loss_val in losses.items():
                        if isinstance(loss_val, torch.Tensor):
                            loss_val = loss_val.item()
                        writer.add_scalar(f"eval_vae/example_{i}/{loss_name}", loss_val, global_step)

                    # Log latent channel visualizations for first few samples
                    if i < 4:
                        mu_sample = mu[0].float().cpu()  # [latent_channels, H, W]
                        mu_min, mu_max = mu_sample.min(), mu_sample.max()
                        mu_norm = (mu_sample - mu_min) / (mu_max - mu_min + 1e-5)
                        for c in range(min(mu_norm.shape[0], 8)):  # Limit to first 8 channels
                            writer.add_image(
                                f"eval_vae/example_{i}/mu_channel_{c}",
                                mu_norm[c:c+1, :, :],
                                global_step
                            )

                    # Log attention weights for first few samples
                    if i < 4:
                        # Compute downsampled T for trimming padding from attention visualizations
                        # Time strides depend on config, but for "small" it's 5*5*1 = 25x
                        # We can infer this from the attention shape
                        enc_T_actual = None
                        if hasattr(model, 'encoder') and hasattr(model.encoder, 'time_strides'):
                            # Compute downsampled length
                            T_down = mel_length
                            for stride in model.encoder.time_strides:
                                T_down = (T_down + stride - 1) // stride
                            enc_T_actual = T_down

                        self._log_attention_weights(
                            writer, enc_attn, global_step,
                            tag_prefix=f"eval_vae/example_{i}/encoder_attention",
                            T_actual=enc_T_actual,
                        )
                        self._log_attention_weights(
                            writer, dec_attn, global_step,
                            tag_prefix=f"eval_vae/example_{i}/decoder_attention",
                            T_actual=enc_T_actual,  # Decoder bottleneck has same resolution
                        )

                    # Convert mel spectrograms to audio using vocoder
                    if self.vocoder is not None:
                        mel_tensor = mel.squeeze(0).float().cpu()
                        recon_mel_tensor = recon[0].squeeze(0).float().cpu()
                        recon_mu_only_mel_tensor = recon_mu_only[0].squeeze(0).float().cpu()

                        # Log ground truth audio
                        self._log_vocoder_audio(
                            writer, mel_tensor[..., :mel_length], global_step,
                            tag=f"eval_vae/original_audio/{i}"
                        )
                        # Log reconstruction audio
                        self._log_vocoder_audio(
                            writer, recon_mel_tensor[..., :mel_length], global_step,
                            tag=f"eval_vae/recon_audio/{i}"
                        )
                        # Log mu-only reconstruction audio (what diffusion will produce)
                        self._log_vocoder_audio(
                            writer, recon_mu_only_mel_tensor[..., :mel_length], global_step,
                            tag=f"eval_vae/recon_mu_only_audio/{i}"
                        )

        # Log aggregate statistics
        for loss_name, loss_vals in all_losses.items():
            writer.add_scalar(f"eval_vae/mean_{loss_name}", np.mean(loss_vals), global_step)
            writer.add_scalar(f"eval_vae/std_{loss_name}", np.std(loss_vals), global_step)

        writer.add_scalar("eval_vae/mean_mu_mean", np.mean(all_mu_means), global_step)
        writer.add_scalar("eval_vae/mean_mu_std", np.mean(all_mu_stds), global_step)
        writer.add_scalar("eval_vae/mean_logvar_mean", np.mean(all_logvar_means), global_step)

        print(f"Eval visualization complete: {num_samples} samples logged")
        writer.flush()

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

                        # Generate mu-only reconstruction (no sampling, z = mu)
                        # This is what diffusion will see during inference
                        recon_mu_only = model.decode(mu)

                        # Get original length for this example
                        original_length = self.example_original_lengths[i] if i < len(self.example_original_lengths) else mel.shape[-1]

                        # Get tensors for visualization
                        mel_cpu = mel.squeeze(0).cpu().numpy()  # [n_mels, T]
                        recon_cpu = recon[0].squeeze(0).float().cpu().numpy()  # [n_mels, T]
                        recon_mu_only_cpu = recon_mu_only[0].squeeze(0).float().cpu().numpy()  # [n_mels, T]

                        # Trimmed versions (without padding)
                        mel_trimmed = mel_cpu[..., :original_length]
                        recon_trimmed = recon_cpu[..., :original_length]
                        recon_mu_only_trimmed = recon_mu_only_cpu[..., :original_length]

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
                        writer.add_image(
                            f"audio_vae/recon_mu_only_trimmed/{i}",
                            self._visualize_mel_spec(recon_mu_only_trimmed),
                            global_step
                        )

                        # Log trimmed comparison
                        self._log_mel_comparison(
                            writer, recon_trimmed, mel_trimmed, global_step,
                            tag=f"audio_vae/comparison_trimmed/{i}"
                        )
                        self._log_mel_comparison(
                            writer, recon_mu_only_trimmed, mel_trimmed, global_step,
                            tag=f"audio_vae/comparison_mu_only_trimmed/{i}"
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
                            recon_mu_only_mel_tensor = recon_mu_only[0].squeeze(0).float().cpu()  # [n_mels, T]
                            # Log trimmed audio
                            self._log_vocoder_audio(
                                writer, recon_mel_tensor[..., :original_length], global_step,
                                tag=f"audio_vae/recon_audio_trimmed/{i}"
                            )
                            # Log mu-only trimmed audio (what diffusion will produce)
                            self._log_vocoder_audio(
                                writer, recon_mu_only_mel_tensor[..., :original_length], global_step,
                                tag=f"audio_vae/recon_mu_only_audio_trimmed/{i}"
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

    def _log_attention_weights(
        self,
        writer: SummaryWriter,
        attn_dict: Dict[str, Optional[torch.Tensor]],
        global_step: int,
        tag_prefix: str,
        M: int = 10,  # Expected mel bins in bottleneck (default for small config)
        T_actual: Optional[int] = None,  # Actual T (without padding) for trimming
    ):
        """
        Log 2D attention weight visualizations to TensorBoard.

        Args:
            writer: TensorBoard writer
            attn_dict: Dictionary with "weights" key containing attention tensor
                      Shape: [B, n_heads, M*T, M*T] for full 2D attention with RoPE
            global_step: Current training step
            tag_prefix: Tag prefix for TensorBoard (e.g., "eval_vae/example_0/encoder_attention")
            M: Number of mel bins in bottleneck (used to infer T from sequence length)
            T_actual: Actual valid timesteps (before padding). If provided, visualizations
                      are trimmed to show only valid positions.
        """
        weights = attn_dict.get("weights", None)
        if weights is None:
            return

        # Move to CPU and convert to numpy
        # Shape: [n_heads, seq_len, seq_len] where seq_len = M * T
        weights = weights[0].float().cpu().numpy()
        n_heads, seq_len, _ = weights.shape

        # Infer T from seq_len = M * T
        T_full = seq_len // M

        # Use T_actual if provided, otherwise use full T
        T = T_actual if T_actual is not None else T_full

        # Build mask for valid positions if we have padding
        # Positions are ordered as (m=0, t=0), (m=0, t=1), ..., (m=M-1, t=T-1)
        # Valid positions are those where t < T_actual
        if T_actual is not None and T_actual < T_full:
            # Create indices for valid positions
            valid_indices = []
            for m in range(M):
                for t in range(T_actual):
                    valid_indices.append(m * T_full + t)
            valid_indices = np.array(valid_indices)

            # Slice out valid positions from attention weights
            # weights shape: [n_heads, M*T_full, M*T_full] -> [n_heads, M*T_actual, M*T_actual]
            weights = weights[:, valid_indices, :][:, :, valid_indices]

        # 1. Global average attention map (avg across heads)
        global_avg_weights = weights.mean(axis=0)  # [M*T, M*T]

        # Log full 2D attention map
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(global_avg_weights, aspect='auto', origin='lower', cmap='viridis')
        title_suffix = " (trimmed)" if T_actual is not None and T_actual < T_full else ""
        ax.set_title(f'2D Attention (avg {n_heads} heads, M={M}, T={T}){title_suffix}')
        ax.set_xlabel('Key position (flattened M×T)')
        ax.set_ylabel('Query position (flattened M×T)')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        writer.add_figure(f"{tag_prefix}/global_2d", fig, global_step)
        plt.close(fig)

        # 2. Cross-frequency attention analysis
        # For each time position, look at how different mel bins attend to each other
        # Sum attention from (m, t) to (m', t) for same t to see freq-freq patterns
        freq_freq_attn = np.zeros((M, M))
        for t in range(T):
            for m_q in range(M):
                for m_k in range(M):
                    q_idx = m_q * T + t
                    k_idx = m_k * T + t
                    freq_freq_attn[m_q, m_k] += global_avg_weights[q_idx, k_idx]
        freq_freq_attn /= T  # Average over time positions

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(freq_freq_attn, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f'Cross-Frequency Attention (same timestep){title_suffix}')
        ax.set_xlabel('Key mel bin')
        ax.set_ylabel('Query mel bin')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        writer.add_figure(f"{tag_prefix}/cross_freq", fig, global_step)
        plt.close(fig)

        # 3. Cross-time attention analysis
        # For each mel bin, look at how different time positions attend to each other
        # Sum attention from (m, t) to (m, t') for same m to see time-time patterns
        time_time_attn = np.zeros((T, T))
        for m in range(M):
            for t_q in range(T):
                for t_k in range(T):
                    q_idx = m * T + t_q
                    k_idx = m * T + t_k
                    time_time_attn[t_q, t_k] += global_avg_weights[q_idx, k_idx]
        time_time_attn /= M  # Average over mel bins

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(time_time_attn, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f'Cross-Time Attention (same mel bin){title_suffix}')
        ax.set_xlabel('Key timestep')
        ax.set_ylabel('Query timestep')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        writer.add_figure(f"{tag_prefix}/cross_time", fig, global_step)
        plt.close(fig)

        # 4. Per-head attention maps (first 4 heads)
        num_heads_to_log = min(n_heads, 4)
        fig, axes = plt.subplots(1, num_heads_to_log, figsize=(4 * num_heads_to_log, 4))
        if num_heads_to_log == 1:
            axes = [axes]
        for h in range(num_heads_to_log):
            im = axes[h].imshow(weights[h], aspect='auto', origin='lower', cmap='viridis')
            axes[h].set_title(f'Head {h}')
            axes[h].set_xlabel('Key')
            axes[h].set_ylabel('Query')
        plt.tight_layout()
        writer.add_figure(f"{tag_prefix}/per_head", fig, global_step)
        plt.close(fig)

        # 5. Log attention statistics (on valid positions only)
        writer.add_scalar(f"{tag_prefix}/mean", float(global_avg_weights.mean()), global_step)
        writer.add_scalar(f"{tag_prefix}/max", float(global_avg_weights.max()), global_step)
        writer.add_scalar(f"{tag_prefix}/entropy", float(self._attention_entropy(global_avg_weights)), global_step)

        # Log diagonal strength (how much attention stays on same position)
        diag_strength = np.diag(global_avg_weights).mean()
        writer.add_scalar(f"{tag_prefix}/diag_strength", float(diag_strength), global_step)

        # Log cross-freq vs cross-time attention balance
        cross_freq_strength = freq_freq_attn.sum() / (M * M)
        cross_time_strength = time_time_attn.sum() / (T * T)
        writer.add_scalar(f"{tag_prefix}/cross_freq_strength", float(cross_freq_strength), global_step)
        writer.add_scalar(f"{tag_prefix}/cross_time_strength", float(cross_time_strength), global_step)

    def _attention_entropy(self, attn_weights: np.ndarray) -> float:
        """
        Compute average entropy of attention distributions.
        Higher entropy = more uniform attention, lower = more peaked.
        """
        # Clip to avoid log(0)
        attn_clipped = np.clip(attn_weights, 1e-10, 1.0)
        # Normalize rows to sum to 1 (they should already, but just in case)
        row_sums = attn_clipped.sum(axis=-1, keepdims=True)
        attn_norm = attn_clipped / (row_sums + 1e-10)
        # Compute entropy per row, then average
        entropy_per_row = -np.sum(attn_norm * np.log(attn_norm + 1e-10), axis=-1)
        return float(entropy_per_row.mean())

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

    Supports discriminator regularization:
    - Instance noise: adds Gaussian noise to both real and fake mel spectrograms
    - R1 gradient penalty: penalizes gradient norm on real mel spectrograms

    For sharded datasets, uses ShardAwareSampler to minimize shard loading overhead.

    Supports audio perceptual losses:
    - Multi-scale mel loss (works on mel spectrograms directly)
    - Wav2Vec2 perceptual loss (requires vocoder to convert to waveform)
    - PANNs perceptual loss (requires vocoder to convert to waveform)
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
        gan_start_condition_key: Optional[str] = None,
        gan_start_condition_value: Optional[Any] = None,
        # Discriminator regularization
        instance_noise_std: float = 0.0,  # Initial std for instance noise (0 = disabled)
        instance_noise_decay_steps: int = 50000,  # Steps to decay noise to 0
        r1_penalty_weight: float = 0.0,  # Weight for R1 gradient penalty (0 = disabled)
        r1_penalty_interval: int = 16,  # Apply R1 penalty every N steps (expensive)
        # GAN warmup (ramps GAN loss contribution from 0 to full weight)
        gan_warmup_steps: int = 0,  # Steps to ramp up GAN loss (0 = no warmup)
        # KL annealing (ramps KL weight from 0 to full over training)
        kl_annealing_steps: int = 0,  # Steps to ramp KL weight from 0 to 1 (0 = disabled)
        # Audio perceptual loss
        audio_perceptual_loss: Optional[torch.nn.Module] = None,
        audio_perceptual_loss_weight: float = 0.0,
        audio_perceptual_loss_start_step: int = 0,  # Step to start applying perceptual loss
        vocoder: Optional[torch.nn.Module] = None,  # For waveform-based losses
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Store shard-aware sampler if available
        self._shard_sampler = None
        if hasattr(self.train_dataset, 'get_sampler'):
            self._shard_sampler = self.train_dataset.get_sampler(shuffle=True, seed=42)
            print("Using ShardAwareSampler for efficient shard loading")
        self.writer = None

        self.step_offset = step_offset if step_offset is not None else 0
        self.cmdline = cmdline
        self.git_commit_hash = git_commit_hash

        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.gan_loss_weight = gan_loss_weight
        self.feature_matching_weight = feature_matching_weight
        self.discriminator_update_frequency = discriminator_update_frequency
        self.gan_start_condition_key = gan_start_condition_key
        self.gan_start_condition_value = gan_start_condition_value
        self.gan_already_started = False
        self.gan_start_step = None  # Track when GAN training started (for warmup)
        self.gan_warmup_steps = gan_warmup_steps

        # KL annealing settings
        self.kl_annealing_steps = kl_annealing_steps

        # Discriminator regularization settings
        self.r1_penalty_weight = r1_penalty_weight
        self.r1_penalty_interval = r1_penalty_interval

        # Audio perceptual loss settings
        self.audio_perceptual_loss = audio_perceptual_loss
        self.audio_perceptual_loss_weight = audio_perceptual_loss_weight
        self.audio_perceptual_loss_start_step = audio_perceptual_loss_start_step
        self.vocoder = vocoder

        # Instance noise scheduler (decays over training)
        self.noise_scheduler = None
        if instance_noise_std > 0:
            self.noise_scheduler = MelInstanceNoiseScheduler(
                initial_std=instance_noise_std,
                final_std=0.0,
                decay_steps=instance_noise_decay_steps,
                decay_type="linear",
            )

        # GradScaler for discriminator when using mixed precision
        # The Trainer has its own scaler for the main model, but discriminator needs separate one
        self.discriminator_scaler = None
        if discriminator is not None:
            self.discriminator_scaler = torch.amp.GradScaler(enabled=False)  # Will be enabled in compute_loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        Override to use shard-aware sampler for sharded datasets.

        This ensures samples are grouped by shard, minimizing disk I/O
        by loading each shard only once per epoch.
        """
        if self._shard_sampler is not None:
            # Update epoch for proper shuffling reproducibility
            epoch = 0
            if self.state is not None and self.state.epoch is not None:
                epoch = int(self.state.epoch)
            self._shard_sampler.set_epoch(epoch)
            return self._shard_sampler

        # Fall back to default sampler for non-sharded datasets
        return super()._get_train_sampler()

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
        mel_spec_lengths = inputs.get("mel_spec_lengths", None)
        speaker_embedding = inputs.get("speaker_embedding", None)

        # Compute KL weight multiplier for KL annealing (ramps from 0 to 1)
        kl_weight_multiplier = 1.0
        if self.kl_annealing_steps > 0:
            kl_weight_multiplier = min(1.0, global_step / self.kl_annealing_steps)

        # Forward pass through VAE model (with optional mask for reconstruction loss and lengths for attention)
        recon, mu, logvar, losses = model(
            mel_spec,
            mask=mel_spec_mask,
            speaker_embedding=speaker_embedding,
            lengths=mel_spec_lengths,
            kl_weight_multiplier=kl_weight_multiplier,
        )

        # Get VAE reconstruction loss
        vae_loss = losses["total_loss"]

        # GAN losses (only after condition is met)
        g_gan_loss = torch.tensor(0.0, device=mel_spec.device)
        d_loss = torch.tensor(0.0, device=mel_spec.device)

        if self.is_gan_enabled(global_step, vae_loss):
            if not self.gan_already_started:
                # First step of GAN training - record start step for warmup
                self.gan_start_step = global_step
                print(f"GAN training starting at step {global_step}")
            self.gan_already_started = True

            # Compute GAN warmup factor (ramps from 0 to 1 over gan_warmup_steps)
            gan_warmup_factor = 1.0
            if self.gan_warmup_steps > 0 and self.gan_start_step is not None:
                steps_since_gan_start = global_step - self.gan_start_step
                gan_warmup_factor = min(1.0, steps_since_gan_start / self.gan_warmup_steps)

            # Get current instance noise std (decays over training)
            noise_std = 0.0
            if self.noise_scheduler is not None:
                noise_std = self.noise_scheduler.get_std(global_step)

            # Discriminator Update
            if global_step % self.discriminator_update_frequency == 0:
                self.discriminator.train()

                # Ensure discriminator is on the same device as inputs
                if next(self.discriminator.parameters()).device != mel_spec.device:
                    self.discriminator.to(mel_spec.device)

                # Apply instance noise to both real and fake mel spectrograms
                real_for_disc = add_mel_instance_noise(mel_spec, noise_std) if noise_std > 0 else mel_spec
                fake_for_disc = add_mel_instance_noise(recon.detach(), noise_std) if noise_std > 0 else recon.detach()

                # Compute discriminator loss in fp32 to avoid gradient underflow
                # Mixed precision can cause discriminator gradients to vanish
                with autocast(mel_spec.device.type, dtype=torch.float32, enabled=False):
                    # Cast inputs to fp32 for discriminator
                    real_fp32 = real_for_disc.float()
                    fake_fp32 = fake_for_disc.float()

                    d_loss, d_loss_dict = compute_mel_discriminator_loss(
                        self.discriminator,
                        real_mels=real_fp32,
                        fake_mels=fake_fp32,
                    )

                # R1 gradient penalty (on clean real mels, not noisy)
                r1_loss = torch.tensor(0.0, device=mel_spec.device)
                if self.r1_penalty_weight > 0 and global_step % self.r1_penalty_interval == 0:
                    r1_loss = r1_mel_gradient_penalty(mel_spec.float(), self.discriminator)
                    d_loss = d_loss + self.r1_penalty_weight * r1_loss

                # Log discriminator diagnostics
                if global_step % self.args.logging_steps == 0 and self.writer is not None:
                    for key, val in d_loss_dict.items():
                        self._log_scalar(f"train/{key}", val, global_step)
                    if self.r1_penalty_weight > 0:
                        self._log_scalar("train/r1_penalty", r1_loss, global_step)
                    if noise_std > 0:
                        self._log_scalar("train/instance_noise_std", noise_std, global_step)

                    # Log how different real and fake mel spectrograms are
                    with torch.no_grad():
                        real_fake_mse = torch.nn.functional.mse_loss(mel_spec, recon).item()
                        real_fake_l1 = torch.nn.functional.l1_loss(mel_spec, recon).item()
                        self._log_scalar("train/real_fake_mse", real_fake_mse, global_step)
                        self._log_scalar("train/real_fake_l1", real_fake_l1, global_step)

                # Update discriminator (only during training when gradients are enabled)
                if self.discriminator_optimizer is not None and self.discriminator.training and torch.is_grad_enabled():
                    self.discriminator_optimizer.zero_grad()
                    d_loss.backward()

                    # Log gradient statistics to diagnose training issues
                    if global_step % self.args.logging_steps == 0 and self.writer is not None:
                        total_grad_norm = 0.0
                        for p in self.discriminator.parameters():
                            if p.grad is not None:
                                total_grad_norm += p.grad.norm().item() ** 2
                        total_grad_norm = total_grad_norm ** 0.5
                        self._log_scalar("train/d_grad_norm", total_grad_norm, global_step)

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
                # Log warmup factor
                self._log_scalar("train/gan_warmup_factor", gan_warmup_factor, global_step)

            # Apply warmup factor to GAN loss
            g_gan_loss = gan_warmup_factor * g_gan_loss

        # Total generator loss
        total_loss = vae_loss + self.gan_loss_weight * g_gan_loss

        # Audio perceptual loss (requires vocoder for waveform-based losses)
        # Only apply after audio_perceptual_loss_start_step to let L1/MSE settle first
        audio_perceptual_loss_value = torch.tensor(0.0, device=mel_spec.device)
        audio_perceptual_losses = {}
        perceptual_loss_enabled = (
            self.audio_perceptual_loss is not None
            and self.audio_perceptual_loss_weight > 0
            and global_step >= self.audio_perceptual_loss_start_step
        )
        if perceptual_loss_enabled:
            # Get waveforms if vocoder is available (needed for Wav2Vec2 and PANNs)
            pred_waveform = None
            target_waveform = None
            if self.vocoder is not None:
                with torch.no_grad():
                    # Vocoder expects [B, n_mels, T] - recon is already in that format
                    vocoder_outputs = self.vocoder(recon.float())
                    if isinstance(vocoder_outputs, dict):
                        pred_waveform = vocoder_outputs["pred_waveform"]
                    else:
                        pred_waveform = vocoder_outputs

                    target_vocoder_outputs = self.vocoder(mel_spec.float())
                    if isinstance(target_vocoder_outputs, dict):
                        target_waveform = target_vocoder_outputs["pred_waveform"]
                    else:
                        target_waveform = target_vocoder_outputs

            # Compute audio perceptual losses
            # Mel spec is [B, 1, n_mels, T], squeeze channel dim for multi-scale mel loss
            audio_perceptual_losses = self.audio_perceptual_loss(
                pred_mel=recon.squeeze(1),  # [B, n_mels, T]
                target_mel=mel_spec.squeeze(1),  # [B, n_mels, T]
                target_speaker_embedding=speaker_embedding,
                pred_waveform=pred_waveform,
                target_waveform=target_waveform,
            )
            audio_perceptual_loss_value = audio_perceptual_losses.get("total_perceptual_loss", torch.tensor(0.0, device=mel_spec.device))
            total_loss = total_loss + self.audio_perceptual_loss_weight * audio_perceptual_loss_value

        # Log losses
        if global_step % self.args.logging_steps == 0 and self.writer is not None:
            prefix = "train/" if model.training else "eval/"
            for loss_name, loss in losses.items():
                self._log_scalar(f"{prefix}vae_{loss_name}", loss.mean(), global_step)
            # Log mu and logvar stats
            self._log_scalar(f"{prefix}vae_mu_mean", mu.mean(), global_step)
            self._log_scalar(f"{prefix}vae_mu_std", mu.std(), global_step)
            self._log_scalar(f"{prefix}vae_logvar_mean", logvar.mean(), global_step)
            # Mean variance (what diffusion will see) - useful for setting latent_std
            self._log_scalar(f"{prefix}vae_mean_variance", logvar.exp().mean(), global_step)
            self._log_scalar(f"{prefix}vae_mean_std", logvar.exp().mean().sqrt(), global_step)
            self._log_scalar(f"{prefix}g_gan_loss", g_gan_loss, global_step)
            self._log_scalar(f"{prefix}total_loss", total_loss.mean(), global_step)
            # Log KL weight multiplier when annealing is enabled
            if self.kl_annealing_steps > 0:
                self._log_scalar(f"{prefix}kl_weight_multiplier", kl_weight_multiplier, global_step)

            # Per-channel latent statistics (for detecting channel collapse)
            # mu shape: [B, C, M, T] - compute stats per channel (average over batch, mel, time)
            per_channel_mu_mean = mu.mean(dim=(0, 2, 3))  # [C]
            per_channel_mu_std = mu.std(dim=(0, 2, 3))  # [C]
            per_channel_var = logvar.exp().mean(dim=(0, 2, 3))  # [C]
            # Per-channel KL: 0.5 * (mu^2 + var - log(var) - 1), averaged over batch and spatial
            per_channel_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean(dim=(0, 2, 3))  # [C]

            for c in range(mu.shape[1]):
                self._log_scalar(f"{prefix}channel_{c}/mu_mean", per_channel_mu_mean[c], global_step)
                self._log_scalar(f"{prefix}channel_{c}/mu_std", per_channel_mu_std[c], global_step)
                self._log_scalar(f"{prefix}channel_{c}/variance", per_channel_var[c], global_step)
                self._log_scalar(f"{prefix}channel_{c}/kl", per_channel_kl[c], global_step)

            # Log audio perceptual losses
            if audio_perceptual_losses:
                for loss_name, loss_val in audio_perceptual_losses.items():
                    self._log_scalar(f"{prefix}audio_perceptual/{loss_name}", loss_val, global_step)
                self._log_scalar(f"{prefix}audio_perceptual_weighted", self.audio_perceptual_loss_weight * audio_perceptual_loss_value, global_step)

            # Log speaker embedding statistics (before any normalization in model)
            if speaker_embedding is not None:
                # Flatten to [B, D] if needed
                spk_emb = speaker_embedding.squeeze(1) if speaker_embedding.dim() == 3 else speaker_embedding
                self._log_scalar(f"{prefix}speaker_emb/mean", spk_emb.mean(), global_step)
                self._log_scalar(f"{prefix}speaker_emb/std", spk_emb.std(), global_step)
                # L2 norm per sample, then average
                l2_norms = torch.norm(spk_emb, p=2, dim=-1)
                self._log_scalar(f"{prefix}speaker_emb/l2_norm_mean", l2_norms.mean(), global_step)
                self._log_scalar(f"{prefix}speaker_emb/l2_norm_min", l2_norms.min(), global_step)
                self._log_scalar(f"{prefix}speaker_emb/l2_norm_max", l2_norms.max(), global_step)

        outputs = {
            "loss": total_loss,
            "rec": recon,
        }

        return (total_loss, outputs) if return_outputs else total_loss

    def is_gan_enabled(self, global_step: int, vae_loss: torch.Tensor) -> bool:
        """
        Check if GAN training should be enabled based on the configured conditions.

        Supports two modes:
        - "step": Start GAN training after a specific step
        - "reconstruction_criteria_met": Start when VAE loss drops below threshold

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
            # Start GAN when VAE loss drops below threshold
            threshold = float(self.gan_start_condition_value)
            return vae_loss.item() < threshold

        return False

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

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """
        Override prediction_step to handle VAE inputs correctly during evaluation.
        The default Trainer calls model(**inputs) which doesn't work with VAE.forward().
        """
        model.eval()

        with torch.no_grad():
            # Unpack inputs the same way as compute_loss
            mel_spec = inputs["mel_spec"]
            mel_spec_mask = inputs.get("mel_spec_mask", None)
            mel_spec_lengths = inputs.get("mel_spec_lengths", None)
            speaker_embedding = inputs.get("speaker_embedding", None)

            # Move to device
            mel_spec = mel_spec.to(self.args.device)
            if mel_spec_mask is not None:
                mel_spec_mask = mel_spec_mask.to(self.args.device)
            if mel_spec_lengths is not None:
                mel_spec_lengths = mel_spec_lengths.to(self.args.device)
            if speaker_embedding is not None:
                speaker_embedding = speaker_embedding.to(self.args.device)

            # Use autocast for mixed precision (same as training)
            dtype = torch.bfloat16 if self.args.bf16 else torch.float16 if self.args.fp16 else torch.float32
            with autocast(self.args.device.type, dtype=dtype, enabled=self.args.bf16 or self.args.fp16):
                # Forward pass
                _, _, _, losses = model(
                    mel_spec,
                    mask=mel_spec_mask,
                    speaker_embedding=speaker_embedding,
                    lengths=mel_spec_lengths,
                )

                loss = losses["total_loss"]

        # Return (loss, logits, labels) - for VAE we don't have traditional logits/labels
        return (loss, None, None)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save both VAE and discriminator."""
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
    discriminator: torch.nn.Module,
    discriminator_optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.nn.Module, Optional[torch.optim.Optimizer], bool]:
    """
    Load discriminator from checkpoint if it exists.

    Handles errors gracefully - if loading fails, returns the fresh discriminator
    and continues training from scratch.
    """
    if resume_from_checkpoint is None:
        print("No checkpoint path provided, training discriminator from scratch")
        return discriminator, discriminator_optimizer, False

    discriminator_path = os.path.join(resume_from_checkpoint, "discriminator.pt")
    if os.path.exists(discriminator_path):
        print(f"Loading discriminator from {discriminator_path}")
        try:
            checkpoint = torch.load(discriminator_path, map_location=device, weights_only=True)
            discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

            if discriminator_optimizer is not None and checkpoint.get("discriminator_optimizer_state_dict"):
                try:
                    discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer_state_dict"])
                except Exception as e:
                    print(f"Warning: Failed to load discriminator optimizer state: {e}")
                    print("Continuing with fresh optimizer state...")

            return discriminator, discriminator_optimizer, True
        except Exception as e:
            print(f"Warning: Failed to load discriminator checkpoint: {e}")
            print("Continuing with fresh discriminator...")
            return discriminator, discriminator_optimizer, False

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

    # Dataset settings
    use_sharded_dataset = unk_dict.get("use_sharded_dataset", "true").lower() == "true"
    train_cache_dir = unk_dict.get("train_cache_dir", "./cached_datasets/audio_vae_train")
    val_cache_dir = unk_dict.get("val_cache_dir", "./cached_datasets/audio_vae_val")

    # Audio settings (CLI args override unk_dict defaults)
    audio_max_frames = int(unk_dict.get("audio_max_frames", 1875))
    n_mels = int(unk_dict.get("n_mels", 80))
    audio_sample_rate = args.audio_sample_rate if args.audio_sample_rate is not None else int(unk_dict.get("audio_sample_rate", 16000))
    audio_n_fft = args.audio_n_fft if args.audio_n_fft is not None else int(unk_dict.get("audio_n_fft", 1024))
    audio_hop_length = args.audio_hop_length if args.audio_hop_length is not None else int(unk_dict.get("audio_hop_length", 256))

    # VAE settings
    latent_channels = int(unk_dict.get("latent_channels", 4))
    speaker_embedding_dim = int(unk_dict.get("speaker_embedding_dim", 192))  # ECAPA-TDNN dim
    normalize_speaker_embedding = unk_dict.get("normalize_speaker_embedding", "true").lower() == "true"
    # FiLM bounding - prevents extreme scale/shift values that can cause artifacts
    # Scale bound of 0.5 means (1 + scale) ranges from 0.5 to 1.5 (never zeroes out)
    # Set to 0 to disable bounding (unbounded FiLM)
    film_scale_bound = float(unk_dict.get("film_scale_bound", 0.5))
    film_shift_bound = float(unk_dict.get("film_shift_bound", 0.5))

    # VAE loss weights
    recon_loss_weight = float(unk_dict.get("recon_loss_weight", 1.0))
    mse_loss_weight = float(unk_dict.get("mse_loss_weight", 1.0))
    l1_loss_weight = float(unk_dict.get("l1_loss_weight", 0.0))
    kl_divergence_loss_weight = float(unk_dict.get("kl_divergence_loss_weight", 1e-4))

    # Audio perceptual loss settings (speech-focused)
    # Total weight for all audio perceptual losses (0 = disabled)
    audio_perceptual_loss_weight = float(unk_dict.get("audio_perceptual_loss_weight", 0.0))
    # Step to start applying perceptual loss (0 = from start, >0 = delay to let L1/MSE settle)
    audio_perceptual_loss_start_step = int(unk_dict.get("audio_perceptual_loss_start_step", 0))
    # Individual component weights (relative to total audio perceptual loss weight)
    multi_scale_mel_weight = float(unk_dict.get("multi_scale_mel_weight", 1.0))
    wav2vec2_weight = float(unk_dict.get("wav2vec2_weight", 0.0))  # Requires vocoder for waveform
    panns_weight = float(unk_dict.get("panns_weight", 0.0))  # Requires panns-inference, vocoder
    speaker_embedding_weight = float(unk_dict.get("speaker_embedding_weight", 0.0))  # Requires speaker embedding model, vocoder
    # Wav2Vec2 model selection: 'facebook/wav2vec2-base' (~95M) or 'facebook/wav2vec2-large' (~317M)
    wav2vec2_model = unk_dict.get("wav2vec2_model", "facebook/wav2vec2-base")

    # Vocoder settings (optional - for audio generation during visualization AND perceptual loss)
    vocoder_checkpoint_path = unk_dict.get("vocoder_checkpoint_path", None)
    vocoder_config = unk_dict.get("vocoder_config", "tiny_attention_freq_domain_vocoder")

    # GAN training settings
    use_gan = unk_dict.get("use_gan", "false").lower() == "true"
    gan_start_condition_key = unk_dict.get("gan_start_condition_key", "step")  # "step" or "reconstruction_criteria_met"
    gan_start_condition_value = unk_dict.get("gan_start_condition_value", "0")  # step number or loss threshold
    discriminator_lr = float(unk_dict.get("discriminator_lr", 2e-4))
    gan_loss_weight = float(unk_dict.get("gan_loss_weight", 0.5))
    feature_matching_weight = float(unk_dict.get("feature_matching_weight", 0.0))
    discriminator_config = unk_dict.get("discriminator_config", "mini_multi_scale")

    # Discriminator regularization settings
    instance_noise_std = float(unk_dict.get("instance_noise_std", 0.0))  # Initial std (0 = disabled)
    instance_noise_decay_steps = int(unk_dict.get("instance_noise_decay_steps", 50000))
    r1_penalty_weight = float(unk_dict.get("r1_penalty_weight", 0.0))  # Weight (0 = disabled)
    r1_penalty_interval = int(unk_dict.get("r1_penalty_interval", 16))  # Apply every N steps
    gan_warmup_steps = int(unk_dict.get("gan_warmup_steps", 0))  # Steps to ramp GAN loss from 0 to full (0 = no warmup)

    # KL annealing: ramps KL weight from 0 to full over N steps (0 = disabled, no annealing)
    kl_annealing_steps = int(unk_dict.get("kl_annealing_steps", 0))

    # Free bits: minimum KL per channel to prevent posterior collapse (0 = disabled)
    free_bits = float(unk_dict.get("free_bits", 0.0))

    # Create shared window buffer for audio processing
    shared_window_buffer = SharedWindowBuffer()

    model = model_config_lookup[args.config](
        latent_channels=latent_channels,
        speaker_embedding_dim=speaker_embedding_dim,
        normalize_speaker_embedding=normalize_speaker_embedding,
        film_scale_bound=film_scale_bound,
        film_shift_bound=film_shift_bound,
        recon_loss_weight=recon_loss_weight,
        mse_loss_weight=mse_loss_weight,
        l1_loss_weight=l1_loss_weight,
        kl_divergence_loss_weight=kl_divergence_loss_weight,
        free_bits=free_bits,
    )

    # Try to load existing checkpoint
    model, model_loaded = load_model(False, model, run_dir)

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

    # Create audio perceptual loss if enabled
    audio_perceptual_loss = None
    perceptual_loss_vocoder = None
    if audio_perceptual_loss_weight > 0:
        # Check if waveform-based losses need a vocoder
        needs_vocoder = (wav2vec2_weight > 0 or panns_weight > 0)
        if needs_vocoder:
            if vocoder_checkpoint_path is None:
                print("Warning: Wav2Vec2/PANNs losses require a vocoder but no vocoder_checkpoint_path provided.")
                print("  Only multi-scale mel loss will be used. Set wav2vec2_weight=0 and panns_weight=0 to suppress this warning.")
            elif os.path.exists(vocoder_checkpoint_path):
                # Load vocoder for perceptual loss (separate from visualization vocoder)
                print(f"Loading vocoder for perceptual loss from {vocoder_checkpoint_path}")
                perceptual_loss_vocoder = vocoder_config_lookup[vocoder_config](
                    shared_window_buffer=shared_window_buffer,
                )
                perceptual_loss_vocoder, _ = load_model(False, perceptual_loss_vocoder, vocoder_checkpoint_path)
                # Remove weight normalization for inference optimization
                if hasattr(perceptual_loss_vocoder.vocoder, 'remove_weight_norm'):
                    perceptual_loss_vocoder.vocoder.remove_weight_norm()
                perceptual_loss_vocoder.eval()
                perceptual_loss_vocoder.to(device)
                # Freeze vocoder weights
                for param in perceptual_loss_vocoder.parameters():
                    param.requires_grad = False
                print(f"Loaded vocoder for perceptual loss: {sum(p.numel() for p in perceptual_loss_vocoder.parameters()):,} parameters")
            else:
                print(f"Warning: Vocoder checkpoint not found at {vocoder_checkpoint_path}")
                print("  Wav2Vec2/PANNs losses will be disabled.")

        # Create audio perceptual loss
        audio_perceptual_loss = AudioPerceptualLoss(
            sample_rate=audio_sample_rate,
            multi_scale_mel_weight=multi_scale_mel_weight,
            wav2vec2_weight=wav2vec2_weight if perceptual_loss_vocoder is not None else 0.0,
            panns_weight=panns_weight if perceptual_loss_vocoder is not None else 0.0,
            wav2vec2_model=wav2vec2_model,
            speaker_embedding_weight=speaker_embedding_weight,
        )
        audio_perceptual_loss.to(device)
        # Freeze all perceptual loss weights
        for param in audio_perceptual_loss.parameters():
            param.requires_grad = False

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
        print(f"  Speaker embedding dim: {speaker_embedding_dim}")
        print(f"  Normalize speaker embedding: {normalize_speaker_embedding}")
        print(f"  FiLM scale bound: {film_scale_bound} (0=unbounded)")
        print(f"  FiLM shift bound: {film_shift_bound} (0=unbounded)")
        if use_gan and discriminator is not None:
            print(f"GAN training: enabled")
            print(f"  Discriminator config: {discriminator_config}")
            print(f"  Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
            multi_scale_discs = [d for d in discriminator.discriminators if isinstance(d, MelMultiScaleDiscriminator)]
            multi_period_discs = [d for d in discriminator.discriminators if isinstance(d, MelMultiPeriodDiscriminator)]
            if multi_scale_discs:
                print(f"    Multi-scale discriminators parameters: {sum(p.numel() for d in multi_scale_discs for p in d.parameters()):,}")
            if multi_period_discs:
                print(f"    Multi-period discriminators parameters: {sum(p.numel() for d in multi_period_discs for p in d.parameters()):,}")
            print(f"  GAN loss weight: {gan_loss_weight}")
            print(f"  Feature matching weight: {feature_matching_weight}")
            print(f"  Discriminator LR: {discriminator_lr}")
            print(f"  GAN start condition: {gan_start_condition_key}={gan_start_condition_value}")
            if instance_noise_std > 0:
                print(f"  Instance noise: initial_std={instance_noise_std}, decay_steps={instance_noise_decay_steps}")
            if r1_penalty_weight > 0:
                print(f"  R1 penalty: weight={r1_penalty_weight}, interval={r1_penalty_interval}")
            if gan_warmup_steps > 0:
                print(f"  GAN warmup: {gan_warmup_steps} steps (ramps loss from 0 to full)")
        if audio_perceptual_loss is not None:
            print(f"Audio perceptual loss: enabled (total_weight={audio_perceptual_loss_weight})")
            if audio_perceptual_loss_start_step > 0:
                print(f"  Start step: {audio_perceptual_loss_start_step} (delayed to let L1/MSE settle)")
            print(f"  Multi-scale mel weight: {multi_scale_mel_weight}")
            print(f"  Wav2Vec2 weight: {wav2vec2_weight if perceptual_loss_vocoder is not None else 0.0} (model: {wav2vec2_model})")
            print(f"  PANNs weight: {panns_weight if perceptual_loss_vocoder is not None else 0.0}")
            if perceptual_loss_vocoder is not None:
                print(f"  Using vocoder for waveform conversion: {vocoder_config}")
            else:
                print(f"  No vocoder loaded - only multi-scale mel loss active")
        if kl_annealing_steps > 0:
            print(f"KL annealing: {kl_annealing_steps} steps (ramps KL weight from 0 to 1)")
        if free_bits > 0:
            print(f"Free bits: {free_bits} nats per channel (prevents posterior collapse)")

    model = setup_int8_training(args, model)

    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        tpu_num_cores=8 if args.use_xla else None,
        output_dir=run_dir,
        overwrite_output_dir=True,
        lr_scheduler_type=args.lr_scheduler_type,
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
        eval_strategy="steps" if args.eval_steps > 0 else "no",
        eval_steps=args.eval_steps,
    )

    # Load datasets
    if use_sharded_dataset:
        print(f"Using sharded dataset format")
        print(f"  Train: {train_cache_dir}")
        print(f"  Val: {val_cache_dir}")
        train_dataset = AudioVAEShardedDataset(
            shard_dir=train_cache_dir,
            cache_size=32,
            audio_max_frames=audio_max_frames,
        )
        eval_dataset = AudioVAEShardedDataset(
            shard_dir=val_cache_dir,
            cache_size=32,
            audio_max_frames=audio_max_frames,
        )
    else:
        print(f"Using legacy diffusion dataset format")
        print(f"  Train: {train_cache_dir}")
        print(f"  Val: {val_cache_dir}")
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
        speaker_embedding_dim=speaker_embedding_dim,
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
        gan_start_condition_key=gan_start_condition_key if use_gan else None,
        gan_start_condition_value=gan_start_condition_value if use_gan else None,
        instance_noise_std=instance_noise_std,
        instance_noise_decay_steps=instance_noise_decay_steps,
        r1_penalty_weight=r1_penalty_weight,
        r1_penalty_interval=r1_penalty_interval,
        gan_warmup_steps=gan_warmup_steps,
        audio_perceptual_loss=audio_perceptual_loss,
        audio_perceptual_loss_weight=audio_perceptual_loss_weight,
        audio_perceptual_loss_start_step=audio_perceptual_loss_start_step,
        vocoder=perceptual_loss_vocoder,
        kl_annealing_steps=kl_annealing_steps,
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
        early_stopping_callback = EarlyStoppingCallback(stop_step=args.stop_step)
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
