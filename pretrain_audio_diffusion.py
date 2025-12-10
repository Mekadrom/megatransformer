import os

from dataset_loading.audio_diffusion_dataset import CachedAudioDiffusionDataset, AudioDiffusionDataCollator
from model.audio.shared_window_buffer import SharedWindowBuffer
from model.audio.diffusion import AudioConditionalGaussianDiffusion, model_config_lookup
from model.audio.vocoders.vocoders import model_config_lookup as vocoder_config_lookup
from model.ema import EMAModel

os.environ["DEEPSPEED_UNIT_TEST"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ['NCCL_TIMEOUT'] = '1200000'

from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments, Trainer, TrainerCallback, T5EncoderModel, T5Tokenizer
from transformers.integrations import TensorBoardCallback
from contextlib import nullcontext
from typing import Any, Mapping, Optional, Union


import librosa
import librosa.display
import matplotlib.pyplot as plt
import megatransformer_utils
import numpy as np
import torch


def get_writer(trainer: Trainer) -> Optional[SummaryWriter]:
    """Get TensorBoard writer from trainer callbacks."""
    if hasattr(trainer, 'callback_handler'):
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback):
                if callback.tb_writer is not None:
                    return callback.tb_writer
    return None


class AudioDiffusionVisualizationCallback(TrainerCallback):
    """
    Callback for visualizing audio diffusion training progress.
    Periodically generates mel spectrograms and logs comparisons to TensorBoard.
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
        ddim_sampling_steps: int = 50,
        vocoder_checkpoint_path: Optional[str] = None,
        ema: Optional[EMAModel] = None,
    ):
        self.trainer: Optional[Trainer] = None
        self.step_offset = self.step_offset = step_offset if step_offset is not None else 0
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

        t5_model = T5EncoderModel.from_pretrained("t5-small")
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        t5_model.eval()
        self.t5_model = t5_model

        self.text = "The quick brown fox jumps over the lazy dog."
        self.text_inputs = t5_tokenizer(
            [self.text],
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.text_embeddings = self.t5_model(**self.text_inputs).last_hidden_state

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

            # Load checkpoint
            # checkpoint = torch.load(self.vocoder_checkpoint_path, map_location="cpu")
            # if "model_state_dict" in checkpoint:
            #     vocoder.load_state_dict(checkpoint["model_state_dict"])
            # elif "state_dict" in checkpoint:
            #     vocoder.load_state_dict(checkpoint["state_dict"])
            # else:
            #     vocoder.load_state_dict(checkpoint)

            megatransformer_utils.load_model(False, vocoder, self.vocoder_checkpoint_path)

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
                        # Generate unconditional samples with intermediate steps
                        result = model.sample(
                            device=device,
                            batch_size=1,  # Single sample to reduce memory for intermediates
                            condition=None,
                            return_intermediate=True,
                            override_ddim_sampling_steps=self.ddim_sampling_steps,
                        )
                        generated_mels, noise_preds, x_start_preds = result

                        # Log generated mel spectrograms and audio
                        self._log_mel_and_audio(generated_mels, writer, global_step)

                        # Log intermediate denoising steps
                        self._log_intermediate_steps(x_start_preds, writer, global_step, tag_prefix="audio_diffusion/uncond_intermediate")

                        # generate from t5 text conditions
                        result = model.sample(
                            device=device,
                            batch_size=1,
                            condition=self.text_embeddings.to(device),
                            return_intermediate=True,
                            override_ddim_sampling_steps=self.ddim_sampling_steps,
                        )
                        generated_mels, noise_preds, x_start_preds = result

                        self._log_mel_and_audio(generated_mels, writer, global_step, step_offset=1)

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

    def _log_mel_and_audio(self, generated_mels, writer, global_step, step_offset=0):
        for i, mel in enumerate(generated_mels):
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
    """

    def __init__(
        self,
        *args,
        cmdline,
        git_commit_hash,
        step_offset: int = 0,
        ema: Optional[EMAModel] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.writer = None

        self.step_offset = self.step_offset = step_offset if step_offset is not None else 0
        self.cmdline = cmdline
        self.git_commit_hash = git_commit_hash
        self.ema = ema

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
        text_embeddings = inputs.get("text_embeddings", None)

        # megatransformer_utils.print_debug_tensor("mel_spec", mel_spec)

        # Forward pass through diffusion model
        predicted_noise, loss = model(
            x_0=mel_spec,
            # condition=text_embeddings,
            condition=None,  # unconditional training TODO remove
        )

        # Log losses
        if global_step % self.args.logging_steps == 0 and self.writer is not None:
            prefix = "train/" if model.training else "eval/"
            self._log_scalar(f"{prefix}diffusion_loss", loss, global_step)

        outputs = {
            "loss": loss,
            "predicted_noise": predicted_noise,
        }

        return (loss, outputs) if return_outputs else loss

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
    train_cache_dir = unk_dict.get("train_cache_dir", "./cached_datasets/librispeech_train_diffusion_cached")
    val_cache_dir = unk_dict.get("val_cache_dir", "./cached_datasets/librispeech_val_diffusion_cached")
    audio_max_frames = int(unk_dict.get("audio_max_frames", 1875))
    max_conditions = int(unk_dict.get("max_conditions", 1024))
    n_mels = int(unk_dict.get("n_mels", 80))

    # Vocoder settings (optional - for audio generation during visualization)
    vocoder_checkpoint_path = unk_dict.get("vocoder_checkpoint_path", None)

    # EMA settings
    use_ema = unk_dict.get("use_ema", "true").lower() == "true"
    ema_decay = float(unk_dict.get("ema_decay", 0.9999))
    ema_update_after_step = int(unk_dict.get("ema_update_after_step", 100))

    shared_window_buffer = SharedWindowBuffer()

    model = model_config_lookup[args.config](
        num_timesteps=num_timesteps,
        sampling_timesteps=sampling_timesteps,
        betas_schedule=betas_schedule,
        context_dim=context_dim,
        normalize=normalize,
        min_snr_loss_weight=min_snr_loss_weight,
        min_snr_gamma=min_snr_gamma,
    )

    # Try to load existing checkpoint
    model, model_loaded = megatransformer_utils.load_model(False, model, run_dir)

    if args.local_rank == 0 or not args.use_deepspeed:
        print(f"Model structure: {model}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"UNet parameters: {sum(p.numel() for p in model.unet.parameters()):,}")
        print(f"  Downblocks: {sum(p.numel() for p in model.unet.down_blocks.parameters()):,}")
        print(f"  Middle: {sum(p.numel() for p in model.unet.middle_attn_block.parameters()) + sum(p.numel() for p in model.unet.middle_res_block.parameters()) + sum(p.numel() for p in model.unet.middle_res_block2.parameters()):,}")
        print(f"  Upblocks: {sum(p.numel() for p in model.unet.up_blocks.parameters()):,}")
        print(f"Diffusion settings:")
        print(f"  Num timesteps: {num_timesteps}")
        print(f"  Sampling timesteps: {sampling_timesteps}")
        print(f"  Betas schedule: {betas_schedule}")
        print(f"  Context dim: {context_dim}")
        print(f"  Normalize: {normalize}")
        print(f"  Min SNR loss weight: {min_snr_loss_weight}")
        if vocoder_checkpoint_path:
            print(f"  Vocoder checkpoint: {vocoder_checkpoint_path}")

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
    data_collator = AudioDiffusionDataCollator(
        audio_max_frames=audio_max_frames,
        max_conditions=max_conditions,
        n_mels=n_mels,
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

    print(f"Starting audio diffusion training with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()