import os

os.environ["DEEPSPEED_UNIT_TEST"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ['NCCL_TIMEOUT'] = '1200000'

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from contextlib import nullcontext
from typing import Any, Mapping, Optional, Union

from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments, Trainer, TrainerCallback, T5EncoderModel, T5Tokenizer
from transformers.integrations import TensorBoardCallback

from dataset_loading.image_diffusion_dataset import CachedImageDiffusionDataset, ImageDiffusionDataCollator
from model.ema import EMAModel
from model.image.diffusion import ImageConditionalGaussianDiffusion, model_config_lookup
from model.image.vae import model_config_lookup as image_vae_config_lookup
from utils import megatransformer_utils
from utils.model_loading_utils import load_model
from utils.training_utils import CLIPScoreEvaluationCallback, EarlyStoppingCallback, ReduceLROnPlateauCallback, gradfilter_ema, setup_int8_training


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
        if condition is not None:
            condition = self.condition_adapter(condition)
        return self.model(x_0=x_0, condition=condition, return_diagnostics=return_diagnostics)

    def sample(self, *args, **kwargs):
        # Apply adapter to condition if provided (fixes train/inference mismatch)
        if 'condition' in kwargs and kwargs['condition'] is not None:
            kwargs['condition'] = self.condition_adapter(kwargs['condition'])
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
        sampling_timesteps: int = 50,
        ema: Optional[EMAModel] = None,
        vae: Optional[nn.Module] = None,
        train_dataset=None,  # Reference to training dataset for in-distribution sampling
        latent_mean: float = 0.0,
        latent_std: float = 1.0,
    ):
        self.trainer: Optional[Trainer] = None
        self.step_offset = step_offset if step_offset is not None else 0
        self.generation_steps = generation_steps
        self.image_size = image_size
        self.latent_channels = latent_channels
        self.sampling_timesteps = sampling_timesteps

        self.ema = ema
        self.vae = vae
        self.train_dataset = train_dataset
        self.latent_mean = latent_mean
        self.latent_std = latent_std

        # Load T5 for text conditioning
        t5_model = T5EncoderModel.from_pretrained("t5-small")
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        t5_model.eval()
        self.t5_model = t5_model

        self.text = [
            "A photo of a cat sitting on a couch",
            "A man riding a horse in a field",
            "A beautiful landscape with mountains and a lake",
        ]
        self.text_inputs = t5_tokenizer(
            self.text,
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

            if global_step == 1:
                print("Logging text prompts")
                for i, text in enumerate(self.text):
                    writer.add_text(f"image_diffusion/prompt_{i}", text, global_step)

            # Determine device
            if torch.distributed.is_initialized():
                device = torch.device(f"cuda:{torch.distributed.get_rank()}")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            with torch.no_grad():
                dtype = torch.bfloat16 if bool(args.bf16) else torch.float16 if args.fp16 else torch.float32

                # Set model to eval mode for sampling
                was_training = model.training
                model.eval()

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
                            override_sampling_steps=self.sampling_timesteps,
                            image_size=self.image_size,
                            generator=torch.Generator(device).manual_seed(42),
                            guidance_scale=7.5,
                            sampler="dpm_solver_pp",
                            dpm_solver_order=2,
                        )
                        generated_latents, noise_preds, x_start_preds = result

                        # Decode and log
                        self._log_generated_images(
                            generated_latents, writer, global_step,
                            tag_prefix="example_uncond"
                        )

                        # Log intermediate denoising steps
                        self._log_intermediate_steps(
                            x_start_preds, writer, global_step,
                            tag_prefix="example_uncond_intermediate"
                        )

                        for i, text in enumerate(self.text_embeddings):
                            # Generate from text conditions
                            result = model.sample(
                                device=device,
                                batch_size=1,
                                condition=text.unsqueeze(0).to(device),
                                return_intermediate=True,
                                override_sampling_steps=self.sampling_timesteps,
                                image_size=self.image_size,
                                generator=torch.Generator(device).manual_seed(42),
                                guidance_scale=7.5,
                                sampler="dpm_solver_pp",
                                dpm_solver_order=2,
                            )
                            generated_latents, noise_preds, x_start_preds = result

                            self._log_generated_images(
                                generated_latents, writer, global_step,
                                tag_prefix=f"example_{i}/cond"
                            )

                            self._log_intermediate_steps(
                                x_start_preds, writer, global_step,
                                tag_prefix=f"example_{i}/cond_intermediate"
                            )

                        # Generate from TRAINING SET samples (in-distribution)
                        # This helps verify if the model is learning the training data
                        if self.train_dataset is not None and len(self.train_dataset) > 0:
                            # Sample a few fixed indices from training set for consistent visualization
                            num_train_samples = min(3, len(self.train_dataset))
                            train_indices = [0, len(self.train_dataset) // 2, len(self.train_dataset) - 1][:num_train_samples]

                            for idx in train_indices:
                                try:
                                    sample = self.train_dataset[idx]
                                    train_latent = sample["latent_mu"].unsqueeze(0).to(device)
                                    train_condition = sample["text_embeddings"].unsqueeze(0).to(device)

                                    # Log ground truth (decoded from latent)
                                    if self.vae is not None:
                                        gt_decoded = self.vae.decoder(train_latent)
                                        self._log_image_visualization(
                                            writer, gt_decoded.squeeze(0).cpu(), global_step,
                                            tag=f"train_sample_{idx}/ground_truth"
                                        )

                                    # Generate using training sample's conditioning
                                    result = model.sample(
                                        device=device,
                                        batch_size=1,
                                        condition=train_condition,
                                        return_intermediate=True,
                                        override_sampling_steps=self.sampling_timesteps,
                                        image_size=self.image_size,
                                        generator=torch.Generator(device).manual_seed(42),
                                        guidance_scale=7.5,
                                        sampler="dpm_solver_pp",
                                        dpm_solver_order=2,
                                    )
                                    generated_latents, noise_preds, x_start_preds = result

                                    self._log_generated_images(
                                        generated_latents, writer, global_step,
                                        tag_prefix=f"train_sample_{idx}/generated"
                                    )

                                    # Also try with guidance
                                    result = model.sample(
                                        device=device,
                                        batch_size=1,
                                        condition=train_condition,
                                        return_intermediate=True,
                                        override_sampling_steps=self.sampling_timesteps,
                                        image_size=self.image_size,
                                        generator=torch.Generator(device).manual_seed(42),
                                        guidance_scale=7.5,
                                        sampler="dpm_solver_pp",
                                        dpm_solver_order=2,
                                    )
                                    generated_latents, noise_preds, x_start_preds = result

                                    self._log_generated_images(
                                        generated_latents, writer, global_step,
                                        tag_prefix=f"train_sample_{idx}/generated_cfg7.5"
                                    )

                                except Exception as e:
                                    print(f"Error sampling from training set index {idx}: {e}")

                        for guidance in [1.0, 3.0, 5.0, 7.5, 10.0]:
                            # Generate with different guidance scales
                            result = model.sample(
                                device=device,
                                batch_size=1,
                                condition=self.text_embeddings[0:1].to(device),
                                return_intermediate=True,
                                override_sampling_steps=self.sampling_timesteps,
                                image_size=self.image_size,
                                generator=torch.Generator(device).manual_seed(42),
                                guidance_scale=guidance,
                                sampler="dpm_solver_pp",
                                dpm_solver_order=2,
                            )
                            generated_latents, noise_preds, x_start_preds = result

                            self._log_generated_images(
                                generated_latents, writer, global_step,
                                tag_prefix=f"example_guidance_{guidance:.1f}/cond"
                            )

                            self._log_intermediate_steps(
                                x_start_preds, writer, global_step,
                                tag_prefix=f"example_guidance_{guidance:.1f}/cond_intermediate"
                            )

                # Restore training mode
                if was_training:
                    model.train()

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
                    # Denormalize latent back to original VAE distribution before decoding
                    x_start_denorm = x_start * self.latent_std + self.latent_mean
                    img = self.vae.decoder(x_start_denorm)
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
                    # Denormalize latent back to original VAE distribution before decoding
                    latent_denorm = latent * self.latent_std + self.latent_mean
                    img = self.vae.decoder(latent_denorm)
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
    """Callback to update EMA weights after each training step and save/load EMA state."""

    def __init__(self, ema: Optional[EMAModel] = None):
        self.ema = ema

    def on_step_end(self, args, state, control, **kwargs):
        if self.ema is not None:
            self.ema.update()

    def on_save(self, args, state, control, **kwargs):
        """Save EMA weights alongside model checkpoint."""
        if self.ema is None:
            return

        # Determine checkpoint directory
        checkpoint_folder = f"checkpoint-{state.global_step}"
        output_dir = os.path.join(args.output_dir, checkpoint_folder)

        # Save EMA state dict
        ema_path = os.path.join(output_dir, "ema_state.pt")
        torch.save(self.ema.state_dict(), ema_path)
        print(f"Saved EMA state to {ema_path}")


def load_ema_state(ema: EMAModel, checkpoint_path: str) -> bool:
    """Load EMA state from checkpoint if available.

    Args:
        ema: The EMA model to load state into
        checkpoint_path: Path to checkpoint directory

    Returns:
        True if EMA state was loaded, False otherwise
    """
    ema_path = os.path.join(checkpoint_path, "ema_state.pt")
    if os.path.exists(ema_path):
        state_dict = torch.load(ema_path, map_location="cpu")
        ema.load_state_dict(state_dict)
        print(f"Loaded EMA state from {ema_path} (step {ema.step})")
        return True
    return False


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
        use_grokfast_ema: bool = False,
        grokfast_ema_alpha = 0.98,
        grokfast_ema_lambda = 2.0,
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
        self.vae = vae
        self.use_grokfast_ema = use_grokfast_ema
        self.grokfast_ema_alpha = grokfast_ema_alpha
        self.grokfast_ema_lambda = grokfast_ema_lambda
        self.latent_mean = latent_mean
        self.latent_std = latent_std

        self.grads = None
        self.last_logged_loss = None

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

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs)
        # Apply gradfilter_ema after gradients are computed but before optimizer step
        if self.use_grokfast_ema:
            self.grads = gradfilter_ema(model, self.grads, self.grokfast_ema_alpha, self.grokfast_ema_lambda)
        return loss

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

        # Normalize latents to zero-mean unit-variance for diffusion
        x_0 = (x_0 - self.latent_mean) / self.latent_std

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
            predicted_noise, loss, _ = model(
                x_0=x_0,
                condition=text_embeddings,
            )
            diagnostics = None

        # Log losses
        if global_step % self.args.logging_steps == 0 and self.writer is not None:
            prefix = "train/" if model.training else "eval/"
            self._log_scalar(f"{prefix}diffusion_loss", loss, global_step)
            self.last_logged_loss = loss.item()

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
    betas_schedule = unk_dict.get("betas_schedule", "karras")
    context_dim = int(unk_dict.get("context_dim", 512))  # T5-small
    # For latent diffusion, normalize should be False since VAE latents are already ~N(0,1)
    # The normalize function does x*2-1 which is meant for pixel images in [0,1] range
    normalize = unk_dict.get("normalize", "false").lower() == "true"
    min_snr_loss_weight = unk_dict.get("min_snr_loss_weight", "true").lower() == "true"
    min_snr_gamma = float(unk_dict.get("min_snr_gamma", 5.0))

    # SOTA diffusion improvements (can be disabled for debugging)
    cfg_dropout_prob = float(unk_dict.get("cfg_dropout_prob", 0.1))
    zero_terminal_snr = unk_dict.get("zero_terminal_snr", "true").lower() == "true"
    offset_noise_strength = float(unk_dict.get("offset_noise_strength", 0.1))
    timestep_sampling = unk_dict.get("timestep_sampling", "logit_normal")  # "uniform" or "logit_normal"

    # Debug mode - enable verbose tensor statistics logging
    debug_diffusion = unk_dict.get("debug_diffusion", "false").lower() == "true"
    debug_start_at_step = int(unk_dict.get("debug_start_at_step", 0))
    if debug_diffusion:
        from model.diffusion import set_debug_mode
        set_debug_mode(True, start_at_step=debug_start_at_step, initial_steps=5)

    # Dataset settings
    train_cache_dir = unk_dict.get("train_cache_dir", "./cached_datasets/relaion_train_diffusion_latents_best_0_checkpoint-314000/")
    val_cache_dir = unk_dict.get("val_cache_dir", "./cached_datasets/cc3m_val_diffusion_latents_best_0_checkpoint-314000")
    max_conditions = int(unk_dict.get("max_conditions", 512))
    max_train_samples = int(unk_dict.get("max_train_samples", -1))  # -1 means use all

    # EMA settings
    use_ema = unk_dict.get("use_ema", "true").lower() == "true"
    ema_decay = float(unk_dict.get("ema_decay", 0.9999))
    ema_update_after_step = int(unk_dict.get("ema_update_after_step", 100))

    # Latent diffusion settings (required - pixel-space diffusion not supported)
    vae_checkpoint = unk_dict.get("vae_checkpoint", None)
    vae_config = unk_dict.get("vae_config", "mini")
    latent_channels = int(unk_dict.get("latent_channels", 4))
    image_size = int(unk_dict.get("image_size", 32))  # Latent image size

    # Latent normalization (for normalizing VAE latents to zero-mean unit-variance)
    latent_mean = float(unk_dict.get("latent_mean", 0.0))
    latent_std = float(unk_dict.get("latent_std", 1.0))

    use_step_lr = unk_dict.get("use_step_lr", "false").lower() == "true"
    step_lr_factor = float(unk_dict.get("step_lr_factor", 0.5))
    step_lr_patience = int(unk_dict.get("step_lr_patience", 2))
    step_lr_check_every_n_steps = int(unk_dict.get("step_lr_check_every_n_steps", 100))
    step_lr_min = float(unk_dict.get("min_step_lr", 1e-6))

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
        cfg_dropout_prob=cfg_dropout_prob,
        zero_terminal_snr=zero_terminal_snr,
        offset_noise_strength=offset_noise_strength,
        timestep_sampling=timestep_sampling,
    )

    # Try to load existing checkpoint
    if args.resume_from_checkpoint is None:
        model, model_loaded = load_model(False, model, run_dir)
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
        print(f"SOTA improvements:")
        print(f"  CFG dropout prob: {cfg_dropout_prob}")
        print(f"  Zero Terminal SNR: {zero_terminal_snr}")
        print(f"  Offset noise strength: {offset_noise_strength}")
        print(f"  Timestep sampling: {timestep_sampling}")
        print(f"Debug settings:")
        print(f"  Debug diffusion: {debug_diffusion}")
        print(f"Latent diffusion settings:")
        print(f"  VAE checkpoint: {vae_checkpoint}")
        print(f"  VAE config: {vae_config}")
        print(f"  Latent channels: {latent_channels}")
        print(f"  Latent image size: {image_size}")

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
    )

    # Load datasets
    train_dataset = CachedImageDiffusionDataset(
        cache_dir=train_cache_dir,
    )

    # Subset training dataset if max_train_samples is specified
    if max_train_samples > 0 and max_train_samples < len(train_dataset):
        from torch.utils.data import Subset
        original_size = len(train_dataset)
        indices = list(range(max_train_samples))
        train_dataset = Subset(train_dataset, indices)
        print(f"Using subset of {max_train_samples} training samples (out of {original_size} available)")

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
        cmdline=args.cmdline,
        git_commit_hash=args.commit_hash,
        step_offset=args.start_step,
        ema=ema,
        vae=vae,
        use_grokfast_ema=args.trainer.lower() == "grokfast_ema",
        grokfast_ema_alpha=args.grokfast_ema_alpha,
        grokfast_ema_lambda=args.grokfast_ema_lambda,
        latent_mean=latent_mean,
        latent_std=latent_std,
    )

    # Add EMA update callback
    if ema is not None:
        ema_callback = EMAUpdateCallback(ema=ema)
        trainer.add_callback(ema_callback)

    t5_model = T5EncoderModel.from_pretrained("t5-small")
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Add visualization callback
    visualization_callback = ImageDiffusionVisualizationCallback(
        step_offset=args.start_step,
        generation_steps=args.generation_steps,
        image_size=image_size,
        latent_channels=latent_channels,
        sampling_timesteps=sampling_timesteps,
        ema=ema,
        vae=vae,
        train_dataset=train_dataset,  # For in-distribution visualization
        latent_mean=latent_mean,
        latent_std=latent_std,
    )
    trainer.add_callback(visualization_callback)

    eval_callback = CLIPScoreEvaluationCallback(
        eval_steps=args.eval_steps,
        vae=vae,
        text_encoder=t5_model,
        text_tokenizer=t5_tokenizer,
        train_prompts=[
            "high waist sleeveless mini soft jeans dress frilled women ruffles casual summer sundress short denim beach dress cotton",
            "CultureShock! South Africa. A Survival Guide to Customs and Etiquette, Dee Rissik"
        ],  # pulled from laion/relaion400m training set
        guidance_scale=7.5,
    )
    trainer.add_callback(eval_callback)

    if use_step_lr:
        lr_callback = ReduceLROnPlateauCallback(
            monitor="train/diffusion_loss",
            mode="min",
            factor=step_lr_factor,
            patience=step_lr_patience,
            check_every_n_steps=step_lr_check_every_n_steps,  # Check at step intervals, not just eval
            min_lr=step_lr_min
        )
        trainer.add_callback(lr_callback)
        lr_callback.trainer = trainer

    if args.stop_step > 0:
        early_stopping_callback = EarlyStoppingCallback(stop_step=args.stop_step)
        trainer.add_callback(early_stopping_callback)

    visualization_callback.trainer = trainer
    eval_callback.trainer = trainer

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

        # Load EMA state if available
        if ema is not None:
            load_ema_state(ema, checkpoint_path)

    print(f"Starting image diffusion training with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
