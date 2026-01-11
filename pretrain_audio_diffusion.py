import os

os.environ["DEEPSPEED_UNIT_TEST"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ['NCCL_TIMEOUT'] = '1200000'

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio

from contextlib import nullcontext
from typing import Any, Mapping, Optional, Union

from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments, Trainer, TrainerCallback, T5EncoderModel, T5Tokenizer
from transformers.integrations import TensorBoardCallback

from dataset_loading.audio_diffusion_dataset import CachedAudioDiffusionDataset, AudioDiffusionDataCollator
from shard_utils import AudioDiffusionShardedDataset
from model.audio.diffusion import AudioConditionalGaussianDiffusion, model_config_lookup
from model.audio.vocoders.vocoders import model_config_lookup as vocoder_config_lookup
from model.audio.vae import model_config_lookup as audio_vae_config_lookup
from model.ema import EMAModel
from utils import megatransformer_utils
from utils.audio_utils import SharedWindowBuffer
from utils.model_loading_utils import load_model
from utils.training_utils import EarlyStoppingCallback


def get_writer(trainer: Trainer) -> Optional[SummaryWriter]:
    """Get TensorBoard writer from trainer callbacks."""
    if hasattr(trainer, 'callback_handler'):
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback):
                if callback.tb_writer is not None:
                    return callback.tb_writer
    return None


def load_audio_vae(checkpoint_path: str, vae_config: str, latent_channels: int, speaker_embedding_dim: int, device: str = "cuda"):
    """
    Load an audio VAE from a checkpoint for latent diffusion.

    Args:
        checkpoint_path: Path to checkpoint directory (containing model.safetensors or pytorch_model.bin)
        vae_config: Config name from model_config_lookup (e.g., "mini", "tiny", "mini_deep")
        latent_channels: Number of latent channels the VAE was trained with
        speaker_embedding_dim: Dimension of speaker embeddings for decoder conditioning
        device: Device to load the model on

    Returns:
        VAE model in eval mode
    """
    if vae_config not in audio_vae_config_lookup:
        raise ValueError(f"Unknown VAE config: {vae_config}. Available: {list(audio_vae_config_lookup.keys())}")

    # Create model with same config
    model = audio_vae_config_lookup[vae_config](
        latent_channels=latent_channels,
        speaker_embedding_dim=speaker_embedding_dim,
        perceptual_loss_type="none",  # Don't need loss for inference
    )

    # Try to load from safetensors first, then pytorch_model.bin
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            missing = [k for k in missing if "lpips" not in k.lower()]
            if missing:
                print(f"Warning: Missing keys: {missing}")
        print(f"Loaded VAE from {safetensors_path}")
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            missing = [k for k in missing if "lpips" not in k.lower()]
            if missing:
                print(f"Warning: Missing keys: {missing}")
        print(f"Loaded VAE from {pytorch_path}")
    else:
        raise FileNotFoundError(
            f"No model checkpoint found at {checkpoint_path}. "
            f"Expected model.safetensors or pytorch_model.bin"
        )

    model = model.to(device)
    model.eval()

    return model


class AudioDiffusionModelWithT5ConditioningAdapter(nn.Module):
    """Wrapper for AudioConditionalGaussianDiffusion to add T5 text conditioning adapter."""
    def __init__(self, model: AudioConditionalGaussianDiffusion, context_dim: int):
        super().__init__()
        self.model = model
        self.config = model.config
        self.context_dim = context_dim

        self.condition_adapter = nn.Linear(context_dim, context_dim)
        self._init_adapter()

    def _init_adapter(self):
        """Initialize adapter to preserve input distribution.

        Default PyTorch Linear init (kaiming uniform) with 512->512 has weight std ~0.025,
        which shrinks std=0.2 input to std=0.12 output (40% reduction).

        We use xavier uniform with gain=1.0 to approximately preserve variance.
        For a 512->512 linear: output_std ≈ input_std * weight_std * sqrt(fan_in)
        With xavier gain=1.0: weight_std ≈ sqrt(2 / (fan_in + fan_out)) ≈ 0.044
        Output_std ≈ 0.2 * 0.044 * 22.6 ≈ 0.2 (preserved)
        """
        nn.init.xavier_uniform_(self.condition_adapter.weight, gain=1.0)
        nn.init.zeros_(self.condition_adapter.bias)

    def forward(self, x_0: torch.Tensor, condition: Optional[torch.Tensor] = None, return_diagnostics: bool = False):
        # condition is expected to be T5 text embeddings of shape [B, T_text, context_dim]
        # this can be switched out after pretraining to take other condition embedding spaces with retraining
        if condition is not None:
            condition = self.condition_adapter(condition)
        return self.model(x_0=x_0, condition=condition, return_diagnostics=return_diagnostics)

    def sample(self, *args, **kwargs):
        # Apply adapter to condition if provided (fixes train/inference mismatch)
        if 'condition' in kwargs and kwargs['condition'] is not None:
            kwargs['condition'] = self.condition_adapter(kwargs['condition'])
        return self.model.sample(*args, **kwargs)


class AudioDiffusionVisualizationCallback(TrainerCallback):
    """
    Callback for visualizing audio diffusion training progress.
    Periodically generates mel spectrograms and logs comparisons to TensorBoard.
    Optionally converts mel spectrograms to audio using a vocoder.
    Supports latent diffusion with VAE decoding.
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
        ddim_sampling_steps: int = 20,  # 20 steps is sufficient for DPM-Solver++
        vocoder_checkpoint_path: Optional[str] = None,
        ema: Optional[EMAModel] = None,
        use_latent_diffusion: bool = False,
        vae: Optional[nn.Module] = None,
        latent_mean: float = 0.0,
        latent_std: float = 1.0,
    ):
        self.trainer: Optional[Trainer] = None
        self.step_offset = step_offset if step_offset is not None else 0
        self.generation_steps = generation_steps
        self.audio_sample_rate = audio_sample_rate
        self.audio_n_mels = audio_n_mels
        self.audio_n_fft = audio_n_fft
        self.audio_hop_length = audio_hop_length
        self.audio_max_frames = audio_max_frames
        self.ddim_sampling_steps = ddim_sampling_steps

        self.shared_window_buffer = shared_window_buffer
        self.vocoder_checkpoint_path = vocoder_checkpoint_path
        self.vocoder = None
        self._vocoder_load_attempted = False
        self.ema = ema
        self.use_latent_diffusion = use_latent_diffusion
        self.vae = vae
        self.latent_mean = latent_mean
        self.latent_std = latent_std

        t5_model = T5EncoderModel.from_pretrained("t5-small")
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        t5_model.eval()
        self.t5_model = t5_model

        self.text = "It is from Westport, above the villages of Murrisk and Lecanvey."
        self.text_inputs = t5_tokenizer(
            [self.text],
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.text_embeddings = self.t5_model(**self.text_inputs).last_hidden_state

        # Load pre-extracted speaker embedding (no runtime extraction)
        speaker_embedding_path = "inference/examples/test_alm_speaker_embedding_1.pt"
        if os.path.exists(speaker_embedding_path):
            self.speaker_embedding = torch.load(speaker_embedding_path, weights_only=True)
            print(f"Loaded speaker embedding from {speaker_embedding_path}: shape {self.speaker_embedding.shape}")
        else:
            raise FileNotFoundError(
                f"Pre-extracted speaker embedding not found: {speaker_embedding_path}\n"
                f"Please extract it using: python scripts/extract_speaker_embedding.py "
                f"--audio_path inference/examples/test_alm_1.mp3 --output_path {speaker_embedding_path}"
            )

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
            # Try to load the vocoder - assume light-headed freq domain vocoder
            # which is ~1.1M params
            vocoder = vocoder_config_lookup["experimental"](
                shared_window_buffer=self.shared_window_buffer,
            )

            load_model(False, vocoder, self.vocoder_checkpoint_path)

            # Remove weight norm for inference optimization
            if hasattr(vocoder, 'vocoder') and hasattr(vocoder.vocoder, 'remove_weight_norm'):
                vocoder.vocoder.remove_weight_norm()

            vocoder.eval()
            self.vocoder = vocoder
            print(f"Loaded vocoder from {self.vocoder_checkpoint_path}")
            print(f"Vocoder parameters: {sum(p.numel() for p in vocoder.parameters()):,}")
        except Exception as e:
            print(f"Failed to load vocoder: {e}")
            self.vocoder = None

    def on_step_end(self, args, state, control, model: AudioConditionalGaussianDiffusion = None, **kwargs):
        global_step = state.global_step + self.step_offset

        if ((global_step == 1) or (global_step % self.generation_steps == 0)) and state.is_world_process_zero:
            writer = get_writer(self.trainer)
            if writer is None:
                print("No TensorBoard writer found, skipping visualization...")
                return

            print(f"Generating mel spectrograms at step {global_step}...")

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

                # Use EMA weights for sampling if available
                ema_context = self.ema.apply_ema() if self.ema is not None else nullcontext()

                with ema_context:
                    with autocast(device.type, dtype=dtype):
                        # Detect if model is Flow Matching or Gaussian Diffusion
                        # Flow Matching models have 'backbone' attribute or use AudioConditionalFlowMatching
                        is_flow_matching = hasattr(model, 'backbone') or (hasattr(model, 'model') and hasattr(model.model, 'backbone'))

                        # Build sampling kwargs based on model type
                        if is_flow_matching:
                            sample_kwargs = {
                                "device": device,
                                "batch_size": 1,
                                "condition": None,
                                "return_intermediate": True,
                                "override_sampling_steps": self.ddim_sampling_steps,
                                "guidance_scale": 3.0,
                                "solver": "euler",
                            }
                        else:
                            sample_kwargs = {
                                "device": device,
                                "batch_size": 1,
                                "condition": None,
                                "return_intermediate": True,
                                "override_sampling_steps": self.ddim_sampling_steps,
                                "guidance_scale": 3.0,
                                "sampler": "dpm_solver_pp",
                                "dpm_solver_order": 2,
                            }

                        # Generate unconditional samples with intermediate steps
                        result = model.sample(**sample_kwargs)
                        generated_mels, noise_preds, x_start_preds = result

                        # Log generated mel spectrograms and audio (decode from latent if needed)
                        # Speaker embedding is only used for VAE decoding in latent diffusion
                        self._log_mel_and_audio(
                            generated_mels, writer, global_step,
                            speaker_embedding=self.speaker_embedding if self.use_latent_diffusion else None
                        )

                        # Log intermediate denoising steps
                        self._log_intermediate_steps(x_start_preds, writer, global_step, tag_prefix="audio_diffusion/uncond_intermediate")

                        # Generate from t5 text conditions
                        sample_kwargs["condition"] = self.text_embeddings.to(device)
                        result = model.sample(**sample_kwargs)
                        generated_mels, noise_preds, x_start_preds = result

                        self._log_mel_and_audio(
                            generated_mels, writer, global_step, step_offset=1,
                            speaker_embedding=self.speaker_embedding if self.use_latent_diffusion else None
                        )

                        # Log intermediate denoising steps for conditioned generation
                        self._log_intermediate_steps(x_start_preds, writer, global_step, tag_prefix="audio_diffusion/cond_intermediate")

    def _log_intermediate_steps(self, x_start_preds, writer, global_step, tag_prefix="audio_diffusion/intermediate"):
        """Log intermediate denoising steps to TensorBoard.

        Args:
            x_start_preds: List of x_start predictions at each sampling step
            writer: TensorBoard writer
            global_step: Current training step
            tag_prefix: Prefix for TensorBoard tags
        """
        if x_start_preds is None or len(x_start_preds) == 0:
            return

        # Log a subset of intermediate steps to avoid overwhelming TensorBoard
        num_steps = len(x_start_preds)
        # Log ~10 evenly spaced steps including first and last
        num_to_log = min(10, num_steps)
        if num_to_log > 1:
            step_indices = [int(i * (num_steps - 1) / (num_to_log - 1)) for i in range(num_to_log)]
        else:
            step_indices = [0]
        step_indices = sorted(set(step_indices))  # Remove duplicates and sort

        for idx in step_indices:
            if idx >= len(x_start_preds):
                continue

            x_start = x_start_preds[idx]

            # Handle batch dimension - take first sample
            if x_start.dim() == 4:
                mel = x_start[0].squeeze(0)  # [n_mels, T]
            elif x_start.dim() == 3:
                mel = x_start[0]  # [n_mels, T]
            else:
                mel = x_start

            mel_cpu = mel.cpu()

            # Log mel spectrogram visualization
            # idx=0 is early denoising (noisy x_0 estimate), idx=num_steps-1 is near-final
            self.log_mel_spec_visualization(
                writer, mel_cpu, global_step,
                tag=f"{tag_prefix}/step_{idx:03d}_of_{num_steps}"
            )

    def _log_mel_and_audio(self, generated_outputs, writer, global_step, step_offset=0, speaker_embedding=None):
        for i, output in enumerate(generated_outputs):
            # If using latent diffusion, decode latents to mel specs
            if self.use_latent_diffusion and self.vae is not None:
                latent = output
                if latent.dim() == 3:
                    latent = latent.unsqueeze(0)  # [C, H, T] -> [1, C, H, T]

                # Denormalize latent back to original VAE distribution before decoding
                latent = latent * self.latent_std + self.latent_mean

                # Decode latent to mel spec using VAE with speaker embedding
                with torch.no_grad():
                    device = latent.device
                    # Prepare speaker embedding for VAE decoder
                    spk_emb = speaker_embedding.to(device) if speaker_embedding is not None else None
                    mel = self.vae.decoder(latent, speaker_embedding=spk_emb)
                    mel = mel.squeeze(0).squeeze(0)  # [1, 1, n_mels, T] -> [n_mels, T]
            else:
                mel = output
                if mel.dim() == 4:
                    mel = mel.squeeze(0).squeeze(0)
                elif mel.dim() == 3:
                    mel = mel.squeeze(0)

            mel_cpu = mel.cpu()

            # Log mel spectrogram visualization
            self.log_mel_spec_visualization(
                writer, mel_cpu, global_step,
                tag=f"audio_diffusion/generated_mel/{i+step_offset}"
            )

            # Convert to audio using vocoder (on CPU to save GPU memory)
            if self.vocoder is not None:
                self._log_vocoder_audio(
                    writer, mel_cpu, global_step,
                    tag=f"audio_diffusion/generated_audio/{i+step_offset}"
                )

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

    def log_mel_spec_visualization(self, writer: SummaryWriter, mel_spec: torch.Tensor, global_step: int, tag: str):
        writer.add_image(tag, self._visualize_mel_spec(mel_spec.numpy(), self.audio_sample_rate), global_step)

    def log_mel_comparison(self, writer: SummaryWriter, pred_mel: torch.Tensor, target_mel: torch.Tensor, global_step: int, tag: str):
        """Log side-by-side comparison of predicted and target mel spectrograms."""
        pred_np = pred_mel.cpu().numpy() if isinstance(pred_mel, torch.Tensor) else pred_mel
        target_np = target_mel.cpu().numpy() if isinstance(target_mel, torch.Tensor) else target_mel

        # Ensure 2D
        if pred_np.ndim == 3:
            pred_np = pred_np.squeeze(0)
        if target_np.ndim == 3:
            target_np = target_np.squeeze(0)

        # Align lengths
        min_len = min(pred_np.shape[-1], target_np.shape[-1])
        pred_np = pred_np[..., :min_len]
        target_np = target_np[..., :min_len]

        # Create comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Normalize for visualization
        vmin = min(pred_np.min(), target_np.min())
        vmax = max(pred_np.max(), target_np.max())

        im0 = axes[0].imshow(target_np, aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
        axes[0].set_title('Target Mel')
        axes[0].set_ylabel('Mel bin')
        axes[0].set_xlabel('Time frame')
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(pred_np, aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
        axes[1].set_title('Predicted Mel')
        axes[1].set_ylabel('Mel bin')
        axes[1].set_xlabel('Time frame')
        plt.colorbar(im1, ax=axes[1])

        # Error map
        error = np.abs(pred_np - target_np)
        im2 = axes[2].imshow(error, aspect='auto', origin='lower', cmap='viridis')
        axes[2].set_title(f'Absolute Error (mean={error.mean():.4f})')
        axes[2].set_ylabel('Mel bin')
        axes[2].set_xlabel('Time frame')
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        writer.add_figure(tag, fig, global_step)
        plt.close(fig)

    def _visualize_mel_spec(self, mel_spec: np.ndarray, sample_rate: int) -> np.ndarray:
        """Generate mel spectrogram visualization for TensorBoard."""
        # Handle tensor input
        if hasattr(mel_spec, 'numpy'):
            mel_spec = mel_spec.numpy()

        # Ensure 2D
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
            sr=sample_rate,
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


class EMAUpdateCallback(TrainerCallback):
    """Callback to update EMA weights after each training step."""

    def __init__(self, ema: Optional[EMAModel] = None):
        self.ema = ema

    def on_step_end(self, args, state, control, **kwargs):
        if self.ema is not None:
            self.ema.update()


class AudioDiffusionTrainer(Trainer):
    """
    Custom trainer for audio diffusion model with EMA support.
    Supports both latent diffusion (with VAE) and mel-space diffusion (direct on spectrograms).
    Supports shard-aware sampling for efficient training with sharded datasets.
    """

    def __init__(
        self,
        *args,
        cmdline,
        git_commit_hash,
        step_offset: int = 0,
        ema: Optional[EMAModel] = None,
        use_latent_diffusion: bool = False,
        vae: Optional[nn.Module] = None,
        latent_mean: float = 0.0,
        latent_std: float = 1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.writer = None

        self.step_offset = step_offset if step_offset is not None else 0
        self.cmdline = cmdline
        self.git_commit_hash = git_commit_hash
        self.ema = ema
        self.use_latent_diffusion = use_latent_diffusion
        self.vae = vae
        self.latent_mean = latent_mean
        self.latent_std = latent_std

        # Store shard-aware sampler if available
        self._shard_sampler = None
        if hasattr(self.train_dataset, 'get_sampler'):
            self._shard_sampler = self.train_dataset.get_sampler(shuffle=True, seed=42)

        self.has_logged_cli = False

    def _get_train_sampler(self):
        """Override to use shard-aware sampler for sharded datasets."""
        if self._shard_sampler is not None:
            # Update epoch for proper shuffling across epochs
            epoch = 0
            if self.state is not None and self.state.epoch is not None:
                epoch = int(self.state.epoch)
            self._shard_sampler.set_epoch(epoch)
            return self._shard_sampler
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

        if not self.has_logged_cli:
            self.writer.add_text("training/command_line", self.cmdline, global_step)
            self.writer.add_text("training/git_commit_hash", self.git_commit_hash, global_step)
            self.has_logged_cli = True

        text_embeddings = inputs["text_embeddings"]

        # Get input based on diffusion mode
        if self.use_latent_diffusion:
            # Latent diffusion: use VAE-encoded latents
            # Speaker conditioning is handled by VAE decoder, not diffusion
            if "latent_mu" not in inputs:
                raise ValueError(
                    "latent_mu not found in inputs. For latent diffusion, ensure your dataset "
                    "was preprocessed with VAE encoding (--vae_checkpoint)."
                )
            x_0 = inputs["latent_mu"]
            # Add channel dimension if needed: [B, C, H, T] for latent
            if x_0.dim() == 3:
                x_0 = x_0.unsqueeze(1)
            # Normalize latents to zero-mean unit-variance for diffusion
            x_0 = (x_0 - self.latent_mean) / self.latent_std
            speaker_embedding = None  # Not used for latent diffusion
        else:
            # Mel-space diffusion: use mel spectrograms directly
            # Speaker conditioning is done in the diffusion model
            if "mel_spec" not in inputs:
                raise ValueError(
                    "mel_spec not found in inputs. For mel diffusion, ensure your dataset "
                    "includes mel spectrograms (preprocess with --include_mel_specs)."
                )
            x_0 = inputs["mel_spec"]  # [B, n_mels, T]
            x_0 = x_0.unsqueeze(1)    # [B, 1, n_mels, T]
            # Get speaker embedding for mel-space diffusion
            speaker_embedding = inputs.get("speaker_embedding", None)

        # Request diagnostics every N steps for debugging
        should_log_diagnostics = (global_step % (self.args.logging_steps * 10) == 0) and self.writer is not None

        # Forward pass through diffusion model
        if should_log_diagnostics:
            predicted_noise, loss, diagnostics = model(
                x_0=x_0,
                condition=text_embeddings,
                speaker_embedding=speaker_embedding,
                return_diagnostics=True,
            )
        else:
            predicted_noise, loss = model(
                x_0=x_0,
                condition=text_embeddings,
                speaker_embedding=speaker_embedding,
            )
            diagnostics = None

        # Log losses
        if global_step % self.args.logging_steps == 0 and self.writer is not None:
            prefix = "train/" if model.training else "eval/"
            self._log_scalar(f"{prefix}diffusion_loss", loss, global_step)

        # Log detailed diagnostics periodically
        if should_log_diagnostics and diagnostics is not None:
            self._log_diagnostics(diagnostics, global_step)

        outputs = {
            "loss": loss,
            "predicted_noise": predicted_noise,
        }

        return (loss, outputs) if return_outputs else loss

    def _log_diagnostics(self, diagnostics: dict, global_step: int):
        """Log detailed training diagnostics to TensorBoard."""
        if self.writer is None:
            return

        # Log mel spectrogram statistics
        mel_stats = diagnostics.get("mel_stats", {})
        if mel_stats:
            self._log_scalar("diagnostics/mel_min", mel_stats.get("mel_min", 0), global_step)
            self._log_scalar("diagnostics/mel_max", mel_stats.get("mel_max", 0), global_step)
            self._log_scalar("diagnostics/mel_mean", mel_stats.get("mel_mean", 0), global_step)
            self._log_scalar("diagnostics/mel_std", mel_stats.get("mel_std", 0), global_step)

            if "mel_normalized_min" in mel_stats:
                self._log_scalar("diagnostics/mel_normalized_min", mel_stats["mel_normalized_min"], global_step)
                self._log_scalar("diagnostics/mel_normalized_max", mel_stats["mel_normalized_max"], global_step)
                self._log_scalar("diagnostics/mel_normalized_mean", mel_stats["mel_normalized_mean"], global_step)
                self._log_scalar("diagnostics/mel_normalized_std", mel_stats["mel_normalized_std"], global_step)

        # Log per-timestep loss statistics
        timesteps = diagnostics.get("timesteps")
        loss_per_sample = diagnostics.get("loss_per_sample")
        loss_weighted = diagnostics.get("loss_weighted_per_sample")

        if timesteps is not None and loss_per_sample is not None:
            # Bucket timesteps into ranges and compute mean loss per bucket
            num_timesteps = 1000  # Assuming 1000 timesteps
            num_buckets = 4
            bucket_size = num_timesteps // num_buckets

            for bucket_idx in range(num_buckets):
                bucket_start = bucket_idx * bucket_size
                bucket_end = (bucket_idx + 1) * bucket_size

                # Find samples in this bucket
                mask = (timesteps >= bucket_start) & (timesteps < bucket_end)
                if mask.sum() > 0:
                    bucket_loss = loss_per_sample[mask].mean().item()
                    bucket_loss_weighted = loss_weighted[mask].mean().item() if loss_weighted is not None else 0

                    self._log_scalar(f"diagnostics/loss_t_{bucket_start}_{bucket_end}", bucket_loss, global_step)
                    self._log_scalar(f"diagnostics/loss_weighted_t_{bucket_start}_{bucket_end}", bucket_loss_weighted, global_step)

            # Also log overall unweighted loss for comparison
            self._log_scalar("diagnostics/loss_unweighted_mean", loss_per_sample.mean().item(), global_step)

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


def main():
    args, unk = megatransformer_utils.parse_args()
    run_dir = os.path.join(args.logging_base_dir, args.run_name)

    # Select model configuration
    if args.config not in model_config_lookup:
        raise ValueError(f"Unknown audio diffusion config: {args.config}. Available: {list(model_config_lookup.keys())}")

    # Parse extra arguments
    unk_dict = {}
    for i in range(0, len(unk), 2):
        unk_dict[unk[i].lstrip('-')] = unk[i+1]

    # Diffusion-specific settings
    num_timesteps = int(unk_dict.get("num_timesteps", 1000))
    sampling_timesteps = int(unk_dict.get("sampling_timesteps", 50))
    betas_schedule = unk_dict.get("betas_schedule", "cosine")
    context_dim = int(unk_dict.get("context_dim", 512))  # T5-small
    normalize = unk_dict.get("normalize", "true").lower() == "true"
    min_snr_loss_weight = unk_dict.get("min_snr_loss_weight", "true").lower() == "true"
    min_snr_gamma = float(unk_dict.get("min_snr_gamma", 5.0))

    # Dataset settings
    train_cache_dir = unk_dict.get("train_cache_dir", "./cached_datasets/librispeech_train_diffusion_full_speakers_cached")
    val_cache_dir = unk_dict.get("val_cache_dir", "./cached_datasets/librispeech_val_diffusion_full_speakers_cached")
    audio_max_frames = int(unk_dict.get("audio_max_frames", 1875))
    max_conditions = int(unk_dict.get("max_conditions", 1024))
    n_mels = int(unk_dict.get("n_mels", 80))
    use_sharded_dataset = unk_dict.get("use_sharded_dataset", "false").lower() == "true"

    # Vocoder settings (optional - for audio generation during visualization)
    vocoder_checkpoint_path = unk_dict.get("vocoder_checkpoint_path", None)

    # EMA settings
    use_ema = unk_dict.get("use_ema", "true").lower() == "true"
    ema_decay = float(unk_dict.get("ema_decay", 0.9999))
    ema_update_after_step = int(unk_dict.get("ema_update_after_step", 100))

    # Latent diffusion settings (optional - if no VAE provided, uses mel-space diffusion)
    vae_checkpoint = unk_dict.get("vae_checkpoint", None)
    vae_config = unk_dict.get("vae_config", "mini")
    latent_channels = int(unk_dict.get("latent_channels", 16))
    speaker_embedding_dim = int(unk_dict.get("speaker_embedding_dim", 192))
    latent_max_frames = int(unk_dict.get("latent_max_frames", 25))  # audio_max_frames / time_compression

    # Latent normalization (for normalizing VAE latents to zero-mean unit-variance)
    latent_mean = float(unk_dict.get("latent_mean", 0.0))
    latent_std = float(unk_dict.get("latent_std", 1.0))

    # Determine diffusion mode based on VAE availability
    use_latent_diffusion = vae_checkpoint is not None

    shared_window_buffer = SharedWindowBuffer()

    # Load VAE for latent diffusion (optional)
    vae = None
    if use_latent_diffusion:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vae = load_audio_vae(vae_checkpoint, vae_config, latent_channels, speaker_embedding_dim, device)
        print(f"Loaded VAE for latent diffusion: {vae_config}, latent_channels={latent_channels}")
    else:
        print("No VAE checkpoint provided - using mel-space diffusion")

    # Determine if we should use speaker conditioning in diffusion
    # (only when not using latent diffusion, since latent diffusion handles speaker in VAE decoder)
    diffusion_speaker_embedding_dim = 0 if use_latent_diffusion else speaker_embedding_dim

    model = model_config_lookup[args.config](
        num_timesteps=num_timesteps,
        sampling_timesteps=sampling_timesteps,
        betas_schedule=betas_schedule,
        context_dim=context_dim,
        speaker_embedding_dim=diffusion_speaker_embedding_dim,
        normalize=normalize,
        min_snr_loss_weight=min_snr_loss_weight,
        min_snr_gamma=min_snr_gamma,
    )

    # Try to load existing checkpoint (only if not using resume_from_checkpoint,
    # which handles loading itself with proper RNG state restoration)
    if args.resume_from_checkpoint is None:
        model, model_loaded = load_model(False, model, run_dir)
    else:
        model_loaded = False  # Let trainer.train(resume_from_checkpoint=...) handle it

    if args.local_rank == 0 or not args.use_deepspeed:
        print(f"Model structure: {model}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # Check if it's a DiT-style model or UNet-style model
        backbone = model.unet if hasattr(model, 'unet') else model.backbone if hasattr(model, 'backbone') else None
        if backbone is not None:
            print(f"Backbone parameters: {sum(p.numel() for p in backbone.parameters()):,}")
            if hasattr(backbone, 'down_blocks'):
                # UNet-style model
                print(f"  Downblocks: {sum(p.numel() for p in backbone.down_blocks.parameters()):,}")
                print(f"  Middle: {sum(p.numel() for p in backbone.middle_attn_block.parameters()) + sum(p.numel() for p in backbone.middle_res_block.parameters()) + sum(p.numel() for p in backbone.middle_res_block2.parameters()):,}")
                print(f"  Upblocks: {sum(p.numel() for p in backbone.up_blocks.parameters()):,}")
            elif hasattr(backbone, 'blocks'):
                # DiT-style model
                print(f"  Transformer blocks: {sum(p.numel() for p in backbone.blocks.parameters()):,}")
                print(f"  Input projection: {sum(p.numel() for p in backbone.input_proj.parameters()):,}")
                print(f"  Time embedding: {sum(p.numel() for p in backbone.time_embed.parameters()):,}")
                print(f"  Final layer: {sum(p.numel() for p in backbone.final_layer.parameters()):,}")
        print(f"Diffusion settings:")
        print(f"  Num timesteps: {num_timesteps}")
        print(f"  Sampling timesteps: {sampling_timesteps}")
        print(f"  Betas schedule: {betas_schedule}")
        print(f"  Context dim: {context_dim}")
        print(f"  Normalize: {normalize}")
        print(f"  Min SNR loss weight: {min_snr_loss_weight}")
        if vocoder_checkpoint_path:
            print(f"  Vocoder checkpoint: {vocoder_checkpoint_path}")
        print(f"Latent diffusion settings:")
        print(f"  VAE checkpoint: {vae_checkpoint}")
        print(f"  VAE config: {vae_config}")
        print(f"  Latent channels: {latent_channels}")
        print(f"  Speaker embedding dim: {speaker_embedding_dim}")
        print(f"  Latent max frames: {latent_max_frames}")
        if latent_mean != 0.0 or latent_std != 1.0:
            print(f"  Latent normalization: mean={latent_mean}, std={latent_std}")

    model = megatransformer_utils.setup_int8_training(args, model)

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
    if use_sharded_dataset:
        print(f"Using sharded dataset format")
        train_dataset = AudioDiffusionShardedDataset(
            shard_dir=train_cache_dir,
            audio_max_frames=audio_max_frames,
            latent_max_frames=latent_max_frames,
        )
        eval_dataset = AudioDiffusionShardedDataset(
            shard_dir=val_cache_dir,
            audio_max_frames=audio_max_frames,
            latent_max_frames=latent_max_frames,
        )
    else:
        print(f"Using legacy individual-file dataset format")
        train_dataset = CachedAudioDiffusionDataset(
            cache_dir=train_cache_dir,
            audio_max_frames=audio_max_frames,
        )
        eval_dataset = CachedAudioDiffusionDataset(
            cache_dir=val_cache_dir,
            audio_max_frames=audio_max_frames,
        )

    # Create data collator
    data_collator = AudioDiffusionDataCollator(
        audio_max_frames=audio_max_frames,
        max_conditions=max_conditions,
        n_mels=n_mels,
        use_latent_diffusion=use_latent_diffusion,
        latent_max_frames=latent_max_frames,
    )

    # Create EMA if enabled
    ema = None
    if use_ema:
        ema = EMAModel(
            model=model,
            decay=ema_decay,
            update_after_step=ema_update_after_step,
            device=torch.distributed.get_rank() if torch.distributed.is_initialized() else "cuda" if torch.cuda.is_available() else "cpu",
        )
        if args.local_rank == 0 or not args.use_deepspeed:
            print(f"EMA enabled: decay={ema_decay}, update_after_step={ema_update_after_step}")

    model = AudioDiffusionModelWithT5ConditioningAdapter(model, context_dim=context_dim)

    # Create trainer
    trainer = AudioDiffusionTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        cmdline=args.cmdline,
        git_commit_hash=args.commit_hash,
        step_offset=args.start_step,
        ema=ema,
        use_latent_diffusion=use_latent_diffusion,
        vae=vae,
        latent_mean=latent_mean,
        latent_std=latent_std,
    )

    # Add EMA update callback
    if ema is not None:
        ema_callback = EMAUpdateCallback(ema=ema)
        trainer.add_callback(ema_callback)

    # Add visualization callback
    visualization_callback = AudioDiffusionVisualizationCallback(
        shared_window_buffer,
        step_offset=args.start_step,
        generation_steps=args.generation_steps,
        audio_sample_rate=model.config.audio_sample_rate,
        audio_n_mels=model.config.audio_n_mels,
        audio_n_fft=model.config.audio_n_fft,
        audio_hop_length=model.config.audio_hop_length,
        audio_max_frames=audio_max_frames,
        ddim_sampling_steps=sampling_timesteps,
        vocoder_checkpoint_path=vocoder_checkpoint_path,
        ema=ema,
        use_latent_diffusion=use_latent_diffusion,
        vae=vae,
        latent_mean=latent_mean,
        latent_std=latent_std,
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

    print(f"Starting audio diffusion training with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()