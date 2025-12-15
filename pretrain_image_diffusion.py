import os

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

from dataset_loading.image_diffusion_dataset import CachedImageDiffusionDataset, ImageDiffusionDataCollator
from model.image.diffusion import ImageConditionalGaussianDiffusion, model_config_lookup
from model.image.vae import model_config_lookup as image_vae_config_lookup
from model.ema import EMAModel

import matplotlib.pyplot as plt
import megatransformer_utils
import numpy as np
import torch
import torch.nn as nn


def get_writer(trainer: Trainer) -> Optional[SummaryWriter]:
    """Get TensorBoard writer from trainer callbacks."""
    if hasattr(trainer, 'callback_handler'):
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback):
                if callback.tb_writer is not None:
                    return callback.tb_writer
    return None


def load_image_vae(checkpoint_path: str, vae_config: str, latent_channels: int, device: str = "cuda"):
    """
    Load an image VAE from a checkpoint for latent diffusion.

    Args:
        checkpoint_path: Path to checkpoint directory (containing model.safetensors or pytorch_model.bin)
        vae_config: Config name from model_config_lookup (e.g., "mini", "tiny")
        latent_channels: Number of latent channels the VAE was trained with
        device: Device to load the model on

    Returns:
        VAE model in eval mode
    """
    if vae_config not in image_vae_config_lookup:
        raise ValueError(f"Unknown VAE config: {vae_config}. Available: {list(image_vae_config_lookup.keys())}")

    # Create model with same config
    model = image_vae_config_lookup[vae_config](
        latent_channels=latent_channels,
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


class ImageDiffusionModelWithT5ConditioningAdapter(nn.Module):
    """Wrapper for ImageConditionalGaussianDiffusion to add T5 text conditioning adapter."""
    def __init__(self, model: ImageConditionalGaussianDiffusion, context_dim: int):
        super().__init__()
        self.model = model
        self.config = model.config
        self.context_dim = context_dim

        self.condition_adapter = nn.Linear(context_dim, context_dim)

    def forward(self, x_0: torch.Tensor, condition: Optional[torch.Tensor] = None, return_diagnostics: bool = False):
        if condition is not None:
            condition = self.condition_adapter(condition)
        return self.model(x_0=x_0, condition=condition, return_diagnostics=return_diagnostics)

    def sample(self, *args, **kwargs):
        return self.model.sample(*args, **kwargs)


class ImageDiffusionVisualizationCallback(TrainerCallback):
    """
    Callback for visualizing image diffusion training progress.
    Periodically generates images and logs to TensorBoard.
    """

    def __init__(
        self,
        step_offset: int = 0,
        generation_steps: int = 1000,
        image_size: int = 32,
        latent_channels: int = 4,
        ddim_sampling_steps: int = 50,
        ema: Optional[EMAModel] = None,
        vae: Optional[nn.Module] = None,
    ):
        self.trainer: Optional[Trainer] = None
        self.step_offset = step_offset if step_offset is not None else 0
        self.generation_steps = generation_steps
        self.image_size = image_size
        self.latent_channels = latent_channels
        self.ddim_sampling_steps = ddim_sampling_steps

        self.ema = ema
        self.vae = vae

        # Load T5 for text conditioning
        t5_model = T5EncoderModel.from_pretrained("t5-small")
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        t5_model.eval()
        self.t5_model = t5_model

        self.text = "A photo of a cat sitting on a couch"
        self.text_inputs = t5_tokenizer(
            [self.text],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.text_embeddings = self.t5_model(**self.text_inputs).last_hidden_state

    def on_step_end(self, args, state, control, model: ImageConditionalGaussianDiffusion = None, **kwargs):
        global_step = state.global_step + self.step_offset

        if ((global_step == 1) or (global_step % self.generation_steps == 0)) and state.is_world_process_zero:
            writer = get_writer(self.trainer)
            if writer is None:
                print("No TensorBoard writer found, skipping visualization...")
                return

            print(f"Generating images at step {global_step}...")

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
                        # Generate unconditional samples
                        result = model.sample(
                            device=device,
                            batch_size=1,
                            condition=None,
                            return_intermediate=True,
                            override_ddim_sampling_steps=self.ddim_sampling_steps,
                            image_size=self.image_size,
                        )
                        generated_latents, noise_preds, x_start_preds = result

                        # Decode and log
                        self._log_generated_images(
                            generated_latents, writer, global_step,
                            tag_prefix="image_diffusion/uncond"
                        )

                        # Log intermediate denoising steps
                        self._log_intermediate_steps(
                            x_start_preds, writer, global_step,
                            tag_prefix="image_diffusion/uncond_intermediate"
                        )

                        # Generate from text conditions
                        result = model.sample(
                            device=device,
                            batch_size=1,
                            condition=self.text_embeddings.to(device),
                            return_intermediate=True,
                            override_ddim_sampling_steps=self.ddim_sampling_steps,
                            image_size=self.image_size,
                        )
                        generated_latents, noise_preds, x_start_preds = result

                        self._log_generated_images(
                            generated_latents, writer, global_step,
                            tag_prefix="image_diffusion/cond"
                        )

                        self._log_intermediate_steps(
                            x_start_preds, writer, global_step,
                            tag_prefix="image_diffusion/cond_intermediate"
                        )

    def _log_intermediate_steps(self, x_start_preds, writer, global_step, tag_prefix="image_diffusion/intermediate"):
        """Log intermediate denoising steps to TensorBoard."""
        if x_start_preds is None or len(x_start_preds) == 0:
            return

        # Log a subset of intermediate steps
        num_steps = len(x_start_preds)
        num_to_log = min(10, num_steps)
        if num_to_log > 1:
            step_indices = [int(i * (num_steps - 1) / (num_to_log - 1)) for i in range(num_to_log)]
        else:
            step_indices = [0]
        step_indices = sorted(set(step_indices))

        for idx in step_indices:
            if idx >= len(x_start_preds):
                continue

            x_start = x_start_preds[idx]

            # Decode latent to image if VAE is available
            if self.vae is not None:
                with torch.no_grad():
                    if x_start.dim() == 3:
                        x_start = x_start.unsqueeze(0)
                    img = self.vae.decoder(x_start)
                    img = img.squeeze(0)  # [C, H, W]
            else:
                img = x_start[0] if x_start.dim() == 4 else x_start

            img_cpu = img.cpu()

            # Log image
            self._log_image_visualization(
                writer, img_cpu, global_step,
                tag=f"{tag_prefix}/step_{idx:03d}_of_{num_steps}"
            )

    def _log_generated_images(self, generated_outputs, writer, global_step, tag_prefix="image_diffusion/generated"):
        """Log generated images to TensorBoard."""
        for i, latent in enumerate(generated_outputs):
            # Log latent space statistics (mu_mean, mu_std)
            self._log_latent_statistics(latent, writer, global_step, tag_prefix=f"{tag_prefix}/latent_stats")

            # Log raw latent channels before VAE decoding
            self._log_latent_channels(latent, writer, global_step, tag_prefix=f"{tag_prefix}/latent")

            # Decode latent to image using VAE
            if self.vae is not None:
                with torch.no_grad():
                    if latent.dim() == 3:
                        latent = latent.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
                    device = latent.device
                    img = self.vae.decoder(latent)
                    img = img.squeeze(0)  # [1, C, H, W] -> [C, H, W]
            else:
                # If no VAE, just visualize the latent
                img = latent
                if img.dim() == 4:
                    img = img.squeeze(0)

            img_cpu = img.cpu()

            # Log image visualization
            self._log_image_visualization(
                writer, img_cpu, global_step,
                tag=f"{tag_prefix}/{i}"
            )

    def _log_latent_channels(self, latent: torch.Tensor, writer: SummaryWriter, global_step: int, tag_prefix: str = "latent"):
        """Log all channels of the latent tensor to TensorBoard."""
        latent_cpu = latent.cpu().float()

        # Handle batch dimension
        if latent_cpu.dim() == 4:
            latent_cpu = latent_cpu.squeeze(0)  # [1, C, H, W] -> [C, H, W]

        num_channels = latent_cpu.shape[0]

        # Log each channel separately
        for c in range(num_channels):
            channel = latent_cpu[c]  # [H, W]
            # Normalize to [0, 1] for visualization
            channel_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
            writer.add_image(f"{tag_prefix}/channel_{c}", channel_norm.unsqueeze(0), global_step)

        # Also log all channels as a grid (2x2 for 4 channels)
        if num_channels == 4:
            # Create 2x2 grid of channels
            h, w = latent_cpu.shape[1], latent_cpu.shape[2]
            grid = torch.zeros(1, h * 2, w * 2)
            for c in range(4):
                channel = latent_cpu[c]
                channel_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
                row, col = c // 2, c % 2
                grid[0, row * h:(row + 1) * h, col * w:(col + 1) * w] = channel_norm
            writer.add_image(f"{tag_prefix}/all_channels_grid", grid, global_step)

    def _log_latent_statistics(self, latent: torch.Tensor, writer: SummaryWriter, global_step: int, tag_prefix: str = "latent_stats"):
        """Log latent space statistics (mu_mean, mu_std) to TensorBoard."""
        latent_float = latent.float()

        # Compute overall statistics
        mu_mean = latent_float.mean().item()
        mu_std = latent_float.std().item()
        mu_min = latent_float.min().item()
        mu_max = latent_float.max().item()

        writer.add_scalar(f"{tag_prefix}/mu_mean", mu_mean, global_step)
        writer.add_scalar(f"{tag_prefix}/mu_std", mu_std, global_step)
        writer.add_scalar(f"{tag_prefix}/mu_min", mu_min, global_step)
        writer.add_scalar(f"{tag_prefix}/mu_max", mu_max, global_step)

        # Also log per-channel statistics
        latent_cpu = latent_float.cpu()
        if latent_cpu.dim() == 4:
            latent_cpu = latent_cpu.squeeze(0)  # [1, C, H, W] -> [C, H, W]

        num_channels = latent_cpu.shape[0]
        for c in range(num_channels):
            channel = latent_cpu[c]
            writer.add_scalar(f"{tag_prefix}/channel_{c}_mean", channel.mean().item(), global_step)
            writer.add_scalar(f"{tag_prefix}/channel_{c}_std", channel.std().item(), global_step)

    def _log_image_visualization(self, writer: SummaryWriter, img: torch.Tensor, global_step: int, tag: str):
        """Log image tensor to TensorBoard."""
        # img is [C, H, W]
        if img.dim() == 2:
            img = img.unsqueeze(0)

        # Normalize to [0, 1] for visualization
        img = img.float()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # If 3 channels, it's RGB
        if img.shape[0] == 3:
            writer.add_image(tag, img, global_step)
        else:
            # For latent visualization (multiple channels), show first channel
            writer.add_image(tag, img[0:1], global_step)


class EMAUpdateCallback(TrainerCallback):
    """Callback to update EMA weights after each training step."""

    def __init__(self, ema: Optional[EMAModel] = None):
        self.ema = ema

    def on_step_end(self, args, state, control, **kwargs):
        if self.ema is not None:
            self.ema.update()


class ImageDiffusionTrainer(Trainer):
    """
    Custom trainer for image latent diffusion model with EMA support.
    Operates on VAE-encoded latents (pixel-space diffusion is not supported).
    """

    def __init__(
        self,
        *args,
        cmdline,
        git_commit_hash,
        step_offset: int = 0,
        ema: Optional[EMAModel] = None,
        vae: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.writer = None

        self.step_offset = step_offset if step_offset is not None else 0
        self.cmdline = cmdline
        self.git_commit_hash = git_commit_hash
        self.ema = ema
        self.vae = vae

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """Prepares one `data` before feeding it to the model."""
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

        text_embeddings = inputs["text_embeddings"]

        # Use latent_mu (latent-only diffusion - pixel-space not supported)
        if "latent_mu" not in inputs:
            raise ValueError(
                "latent_mu not found in inputs. This model only supports latent diffusion. "
                "Ensure your dataset was preprocessed with VAE encoding (--vae_checkpoint)."
            )
        x_0 = inputs["latent_mu"]

        # Request diagnostics every N steps for debugging
        should_log_diagnostics = (global_step % (self.args.logging_steps * 10) == 0) and self.writer is not None

        # Forward pass through diffusion model
        if should_log_diagnostics:
            predicted_noise, loss, diagnostics = model(
                x_0=x_0,
                condition=text_embeddings,
                return_diagnostics=True,
            )
        else:
            predicted_noise, loss = model(
                x_0=x_0,
                condition=text_embeddings,
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

        # Log latent statistics
        latent_stats = diagnostics.get("latent_stats", {})
        if latent_stats:
            self._log_scalar("diagnostics/latent_min", latent_stats.get("latent_min", 0), global_step)
            self._log_scalar("diagnostics/latent_max", latent_stats.get("latent_max", 0), global_step)
            self._log_scalar("diagnostics/latent_mean", latent_stats.get("latent_mean", 0), global_step)
            self._log_scalar("diagnostics/latent_std", latent_stats.get("latent_std", 0), global_step)

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
        raise ValueError(f"Unknown image diffusion config: {args.config}. Available: {list(model_config_lookup.keys())}")

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
    train_cache_dir = unk_dict.get("train_cache_dir", "./cached_datasets/relaion_val_diffusion_latents_gan_4_2_checkpoint-238500")
    val_cache_dir = unk_dict.get("val_cache_dir", "./cached_datasets/relaion_val_diffusion_latents_gan_4_2_checkpoint-238500")
    max_conditions = int(unk_dict.get("max_conditions", 512))

    # EMA settings
    use_ema = unk_dict.get("use_ema", "true").lower() == "true"
    ema_decay = float(unk_dict.get("ema_decay", 0.9999))
    ema_update_after_step = int(unk_dict.get("ema_update_after_step", 100))

    # Latent diffusion settings (required - pixel-space diffusion not supported)
    vae_checkpoint = unk_dict.get("vae_checkpoint", None)
    vae_config = unk_dict.get("vae_config", "mini")
    latent_channels = int(unk_dict.get("latent_channels", 4))
    image_size = int(unk_dict.get("image_size", 32))  # Latent image size

    # Load VAE for latent diffusion (required)
    if vae_checkpoint is None:
        raise ValueError("vae_checkpoint is required - this model only supports latent diffusion")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = load_image_vae(vae_checkpoint, vae_config, latent_channels, device)
    print(f"Loaded VAE for latent diffusion: {vae_config}, latent_channels={latent_channels}")

    model = model_config_lookup[args.config](
        latent_channels=latent_channels,
        num_timesteps=num_timesteps,
        sampling_timesteps=sampling_timesteps,
        betas_schedule=betas_schedule,
        context_dim=context_dim,
        normalize=normalize,
        min_snr_loss_weight=min_snr_loss_weight,
        min_snr_gamma=min_snr_gamma,
    )

    # Try to load existing checkpoint
    if args.resume_from_checkpoint is None:
        model, model_loaded = megatransformer_utils.load_model(False, model, run_dir)
    else:
        model_loaded = False

    if args.local_rank == 0 or not args.use_deepspeed:
        print(f"Model structure: {model}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"UNet parameters: {sum(p.numel() for p in model.unet.parameters()):,}")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  Trainable parameter: {name} - {sum(p.numel() for p in param):,}")
        print(f"Diffusion settings:")
        print(f"  Num timesteps: {num_timesteps}")
        print(f"  Sampling timesteps: {sampling_timesteps}")
        print(f"  Betas schedule: {betas_schedule}")
        print(f"  Context dim: {context_dim}")
        print(f"  Normalize: {normalize}")
        print(f"  Min SNR loss weight: {min_snr_loss_weight}")
        print(f"Latent diffusion settings:")
        print(f"  VAE checkpoint: {vae_checkpoint}")
        print(f"  VAE config: {vae_config}")
        print(f"  Latent channels: {latent_channels}")
        print(f"  Latent image size: {image_size}")

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
    train_dataset = CachedImageDiffusionDataset(
        cache_dir=train_cache_dir,
    )

    eval_dataset = CachedImageDiffusionDataset(
        cache_dir=val_cache_dir,
    )

    # Create data collator
    data_collator = ImageDiffusionDataCollator(
        max_conditions=max_conditions,
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

    model = ImageDiffusionModelWithT5ConditioningAdapter(model, context_dim=context_dim)

    # Create trainer
    trainer = ImageDiffusionTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        cmdline=args.cmdline,
        git_commit_hash=args.commit_hash,
        step_offset=args.start_step,
        ema=ema,
        vae=vae,
    )

    # Add EMA update callback
    if ema is not None:
        ema_callback = EMAUpdateCallback(ema=ema)
        trainer.add_callback(ema_callback)

    # Add visualization callback
    visualization_callback = ImageDiffusionVisualizationCallback(
        step_offset=args.start_step,
        generation_steps=args.generation_steps,
        image_size=image_size,
        latent_channels=latent_channels,
        ddim_sampling_steps=sampling_timesteps,
        ema=ema,
        vae=vae,
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

    print(f"Starting image diffusion training with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
