import os

os.environ["DEEPSPEED_UNIT_TEST"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ['NCCL_TIMEOUT'] = '1200000'

import torch
import numpy as np
import matplotlib.pyplot as plt

from contextlib import nullcontext
from typing import Any, Dict, Mapping, Optional, Union

from PIL import Image
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers.integrations import TensorBoardCallback

from dataset_loading.image_vae_dataset import CachedImageVAEDataset, ImageVAEDataCollator
from model.image import discriminators
from model.image.discriminators import (
    compute_discriminator_loss,
    compute_generator_gan_loss,
    add_instance_noise,
    r1_gradient_penalty,
    InstanceNoiseScheduler,
)
from model.ema import EMAModel
from model.image.vae import VAE, model_config_lookup
from utils import megatransformer_utils
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


class ImageVAEReconstructionCallback(TrainerCallback):
    """
    Callback for logging VAE image reconstruction during training and evaluation.
    Periodically reconstructs test images and logs to TensorBoard.

    VAE uses [-1, 1] normalization (not ImageNet), with tanh output activation.
    """

    def __init__(
        self,
        image_size: int = 256,
        step_offset: int = 0,
        generation_steps: int = 1000,
        num_eval_samples: int = 8,
    ):
        self.trainer: Optional[Trainer] = None
        self.step_offset = step_offset if step_offset is not None else 0
        self.generation_steps = generation_steps
        self.image_size = image_size
        self.num_eval_samples = num_eval_samples

        self.example_paths = [
            "inference/examples/test_vlm1_x256.png",
            "inference/examples/test_vlm2_x256.png",
            "inference/examples/test_vlm3_x256.png",
        ]

        # VAE uses [-1, 1] normalization (for tanh output)
        transform = transforms.Compose([
            # transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1]
        ])

        self.example_images = []
        self.example_images_unnorm = []  # For visualization (in [0, 1] range)
        for path in self.example_paths:
            if os.path.exists(path):
                image = Image.open(path).convert('RGB')
                image_tensor = transform(image)
                self.example_images.append(image_tensor)
                # Store unnormalized version for visualization
                unnorm_transform = transforms.Compose([
                    # transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),  # [0, 1] for direct visualization
                ])
                self.example_images_unnorm.append(unnorm_transform(image))

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Unnormalize image tensor from [-1, 1] (tanh output) back to [0, 1] for visualization."""
        return (x + 1.0) / 2.0

    def _get_device(self):
        """Determine the device to use for inference."""
        if torch.distributed.is_initialized():
            return torch.device(f"cuda:{torch.distributed.get_rank()}")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

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

    def _log_reconstruction(self, writer, model, image, global_step, prefix, idx, args, log_attention: bool = True):
        """Log a single image reconstruction to TensorBoard."""
        device = self._get_device()
        dtype = torch.bfloat16 if bool(args.bf16) else torch.float16 if args.fp16 else torch.float32

        with torch.no_grad():
            with autocast(device.type, dtype=dtype):
                # Use reconstruct_with_attention to get attention weights if available
                recon, mu, logvar, enc_attn, dec_attn = model.reconstruct_with_attention(
                    image.unsqueeze(0).to(device)
                )

                # Also compute losses via forward pass
                _, _, _, losses = model(image.unsqueeze(0).to(device))

                # Unnormalize both original and reconstruction for visualization
                orig_unnorm = self.unnormalize(image.float().cpu())
                orig_unnorm = torch.clamp(orig_unnorm, 0, 1)

                recon_unnorm = self.unnormalize(recon[0].float().cpu())
                recon_unnorm = torch.clamp(recon_unnorm, 0, 1)

                # Log original and reconstructed images
                writer.add_image(f"{prefix}/original/{idx}", orig_unnorm, global_step)
                writer.add_image(f"{prefix}/recon/{idx}", recon_unnorm, global_step)

                # Generate mu-only reconstruction (no sampling, z = mu)
                # This is what diffusion will see during inference
                recon_mu_only = model.decode(mu)
                recon_mu_only_unnorm = self.unnormalize(recon_mu_only[0].float().cpu())
                recon_mu_only_unnorm = torch.clamp(recon_mu_only_unnorm, 0, 1)
                writer.add_image(f"{prefix}/recon_mu_only/{idx}", recon_mu_only_unnorm, global_step)

                # Log per-example losses
                for loss_name, loss_val in losses.items():
                    if isinstance(loss_val, torch.Tensor):
                        loss_val = loss_val.item()
                    writer.add_scalar(f"{prefix}/example_{idx}/{loss_name}", loss_val, global_step)

                # Normalize and log mu as latent_dim number of grayscale images
                mu_unnorm = (mu[0].float().cpu() - mu[0].float().cpu().min()) / (mu[0].float().cpu().max() - mu[0].float().cpu().min() + 1e-5)
                for c in range(mu_unnorm.shape[0]):
                    writer.add_image(f"{prefix}/example_{idx}/mu_channel_{c}", mu_unnorm[c:c+1, :, :], global_step)

                # Log attention weights if available
                if log_attention:
                    # Get spatial dimensions from mu (latent space)
                    _, _, H, W = mu.shape

                    # Log encoder attention
                    enc_weights = enc_attn.get("weights") if enc_attn else None
                    if enc_weights is not None:
                        self._log_attention_weights(
                            writer, enc_weights, global_step,
                            tag_prefix=f"{prefix}/example_{idx}/encoder_attention",
                            H=H, W=W,
                        )

                    # Log decoder attention
                    dec_weights = dec_attn.get("weights") if dec_attn else None
                    if dec_weights is not None:
                        self._log_attention_weights(
                            writer, dec_weights, global_step,
                            tag_prefix=f"{prefix}/example_{idx}/decoder_attention",
                            H=H, W=W,
                        )

    def on_evaluate(self, args, state, control, model: VAE = None, **kwargs):
        """Generate and log reconstructions during evaluation."""
        global_step = state.global_step + self.step_offset

        if not state.is_world_process_zero:
            return

        writer = get_writer(self.trainer)
        if writer is None:
            print("No TensorBoard writer found, skipping eval visualization...")
            return

        print(f"Generating eval image reconstructions at step {global_step}...")

        # Get eval dataset from trainer
        eval_dataset = self.trainer.eval_dataset
        if eval_dataset is None or len(eval_dataset) == 0:
            print("No eval dataset available, skipping eval visualization...")
            return

        model.eval()

        # Sample random indices from eval dataset
        num_samples = min(self.num_eval_samples, len(eval_dataset))
        indices = torch.randperm(len(eval_dataset))[:num_samples].tolist()

        for i, idx in enumerate(indices):
            sample = eval_dataset[idx]
            image = sample["image"]
            self._log_reconstruction(writer, model, image, global_step, "eval_vae", i, args)

        writer.flush()

    def on_step_end(self, args, state, control, model: VAE = None, **kwargs):
        global_step = state.global_step + self.step_offset

        if ((global_step == 1) or (global_step % self.generation_steps == 0)) and state.is_world_process_zero:
            writer = get_writer(self.trainer)
            if writer is None:
                print("No TensorBoard writer found, skipping visualization...")
                return

            print(f"Generating image reconstructions at step {global_step}...")

            for i, image in enumerate(self.example_images):
                self._log_reconstruction(writer, model, image, global_step, "image_vae", i, args)


class ImageVAEGANTrainer(Trainer):
    """
    Custom trainer for VAE with optional GAN training.
    Handles alternating generator/discriminator updates.

    Supports discriminator regularization:
    - Instance noise: adds Gaussian noise to both real and fake images
    - R1 gradient penalty: penalizes gradient norm on real images
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
        # Perceptual loss delayed start
        perceptual_loss_start_step: int = 0,  # Step to start applying perceptual loss (0 = from start)
        # KL annealing (ramps KL weight from 0 to full over training)
        kl_annealing_steps: int = 0,  # Steps to ramp KL weight from 0 to 1 (0 = disabled)
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
        self.gan_start_condition_key = gan_start_condition_key
        self.gan_start_condition_value = gan_start_condition_value
        self.gan_already_started = False
        self.gan_start_step = None  # Track when GAN training started (for warmup)
        self.gan_warmup_steps = gan_warmup_steps
        self.perceptual_loss_start_step = perceptual_loss_start_step

        # KL annealing settings
        self.kl_annealing_steps = kl_annealing_steps

        # Discriminator regularization settings
        self.r1_penalty_weight = r1_penalty_weight
        self.r1_penalty_interval = r1_penalty_interval

        # Instance noise scheduler (decays over training)
        self.noise_scheduler = None
        if instance_noise_std > 0:
            self.noise_scheduler = InstanceNoiseScheduler(
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

        image = inputs["image"]

        # Compute KL weight multiplier for KL annealing (ramps from 0 to 1)
        kl_weight_multiplier = 1.0
        if self.kl_annealing_steps > 0:
            kl_weight_multiplier = min(1.0, global_step / self.kl_annealing_steps)

        # Forward pass through VAE model
        recon, mu, logvar, losses = model(image, kl_weight_multiplier=kl_weight_multiplier)

        # Get VAE reconstruction loss
        vae_loss = losses["total_loss"]

        # GAN losses (only after condition is met)
        g_gan_loss = torch.tensor(0.0, device=image.device)
        d_loss = torch.tensor(0.0, device=image.device)

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
                if next(self.discriminator.parameters()).device != image.device:
                    self.discriminator.to(image.device)

                # Apply instance noise to both real and fake images
                real_for_disc = add_instance_noise(image, noise_std) if noise_std > 0 else image
                fake_for_disc = add_instance_noise(recon.detach(), noise_std) if noise_std > 0 else recon.detach()

                # Compute discriminator loss in fp32 to avoid gradient underflow
                # Mixed precision can cause discriminator gradients to vanish
                with autocast(image.device.type, dtype=torch.float32, enabled=False):
                    # Cast inputs to fp32 for discriminator
                    real_fp32 = real_for_disc.float()
                    fake_fp32 = fake_for_disc.float()

                    d_loss, d_loss_dict = compute_discriminator_loss(
                        self.discriminator,
                        real_images=real_fp32,
                        fake_images=fake_fp32,
                    )

                # R1 gradient penalty (on clean real images, not noisy)
                r1_loss = torch.tensor(0.0, device=image.device)
                if self.r1_penalty_weight > 0 and global_step % self.r1_penalty_interval == 0:
                    r1_loss = r1_gradient_penalty(image.float(), self.discriminator)
                    d_loss = d_loss + self.r1_penalty_weight * r1_loss

                # Log discriminator diagnostics
                if global_step % self.args.logging_steps == 0 and self.writer is not None:
                    for key, val in d_loss_dict.items():
                        self._log_scalar(f"train/{key}", val, global_step)
                    if self.r1_penalty_weight > 0:
                        self._log_scalar("train/r1_penalty", r1_loss, global_step)
                    if noise_std > 0:
                        self._log_scalar("train/instance_noise_std", noise_std, global_step)

                    # Log how different real and fake images are
                    with torch.no_grad():
                        real_fake_mse = torch.nn.functional.mse_loss(image, recon).item()
                        real_fake_l1 = torch.nn.functional.l1_loss(image, recon).item()
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
            device_type = image.device.type
            dtype = torch.bfloat16 if self.args.bf16 else torch.float16 if self.args.fp16 else torch.float32
            with autocast(device_type, dtype=dtype, enabled=self.args.fp16 or self.args.bf16):
                g_gan_loss, g_loss_dict = compute_generator_gan_loss(
                    self.discriminator,
                    real_images=image,
                    fake_images=recon,
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

            # Per-channel latent statistics (for detecting channel collapse)
            # mu shape: [B, C, H, W] - compute stats per channel
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

            # Log KL weight multiplier if annealing is enabled
            if self.kl_annealing_steps > 0:
                self._log_scalar(f"{prefix}kl_weight_multiplier", kl_weight_multiplier, global_step)

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

        if output_dir is None:
            output_dir = self.args.output_dir

        # Save discriminator if GAN training has started
        if self.gan_already_started and self.discriminator is not None:
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

    def is_gan_enabled(self, global_step: int, vae_loss: torch.Tensor) -> bool:
        """
        Check if GAN training should be enabled at the current step.

        Args:
            global_step: Current training step
            vae_loss: Current VAE loss tensor

        Returns:
            True if GAN training should be active
        """
        if self.discriminator is None:
            return False
        if self.gan_start_condition_key is None or self.gan_start_condition_value is None:
            return False

        # Once started, always enabled
        if self.gan_already_started:
            return True

        # Check start conditions
        if self.gan_start_condition_key == "step":
            return global_step >= int(self.gan_start_condition_value)
        elif self.gan_start_condition_key == "reconstruction_criteria_met":
            return vae_loss.mean().item() <= float(self.gan_start_condition_value)

        return False

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

    try:
        discriminator_path = os.path.join(resume_from_checkpoint, "discriminator.pt")
        if os.path.exists(discriminator_path):
            print(f"Loading discriminator from {discriminator_path}")
            checkpoint = torch.load(discriminator_path, map_location=device, weights_only=True)
            discriminator.load_state_dict(checkpoint["discriminator_state_dict"], strict=False)

            if discriminator_optimizer is not None and checkpoint.get("discriminator_optimizer_state_dict"):
                discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer_state_dict"])

            return discriminator, discriminator_optimizer, True
    except Exception as e:
        print(f"Error loading discriminator checkpoint: {e}")
        print(f"!!!! IMPORTANT !!!! NOT CONTINUING WITH DISCRIMNATOR FROM CHECKPOINT !!!!, RESTARTING DISCRIMINATOR TRAINING FROM SCRATCH")

    print("No existing discriminator checkpoint found, training from scratch")
    return discriminator, discriminator_optimizer, False


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


def main():
    args, unk = megatransformer_utils.parse_args()
    run_dir = os.path.join(args.logging_base_dir, args.run_name)

    # Select model configuration
    if args.config not in model_config_lookup:
        raise ValueError(f"Unknown image VAE config: {args.config}. Available: {list(model_config_lookup.keys())}")

    # Parse extra arguments
    unk_dict = {}
    for i in range(0, len(unk), 2):
        unk_dict[unk[i].lstrip('-')] = unk[i+1]

    # Dataset settings
    train_cache_dir = unk_dict.get("train_cache_dir", "./cached_datasets/cc3m_train_vae_cached")
    val_cache_dir = unk_dict.get("val_cache_dir", "./cached_datasets/coco_val_vae_cached")
    image_size = int(unk_dict.get("image_size", 256))
    latent_channels = int(unk_dict.get("latent_channels", 4))

    # VAE loss weights
    recon_loss_weight = float(unk_dict.get("recon_loss_weight", 1.0))
    mse_loss_weight = float(unk_dict.get("mse_loss_weight", 1.0))
    l1_loss_weight = float(unk_dict.get("l1_loss_weight", 0.0))
    kl_divergence_loss_weight = float(unk_dict.get("kl_divergence_loss_weight", 1e-3))
    perceptual_loss_weight = float(unk_dict.get("perceptual_loss_weight", 0.1))

    # Perceptual loss type: "vgg", "lpips", or "none"
    perceptual_loss_type = unk_dict.get("perceptual_loss_type", "vgg")
    lpips_net = unk_dict.get("lpips_net", "alex")  # "alex", "vgg", or "squeeze"

    # GAN training settings
    use_gan = unk_dict.get("use_gan", "false").lower() == "true"
    gan_start_condition_key = unk_dict.get("gan_start_condition_key", None)  # "step" or "reconstruction_criteria_met"
    gan_start_condition_value = unk_dict.get("gan_start_condition_value", None)
    discriminator_lr = float(unk_dict.get("discriminator_lr", 2e-4))
    gan_loss_weight = float(unk_dict.get("gan_loss_weight", 0.5))
    feature_matching_weight = float(unk_dict.get("feature_matching_weight", 0.0))
    discriminator_config = unk_dict.get("discriminator_config", "multi_scale")

    # Discriminator regularization settings
    # Instance noise: adds Gaussian noise to prevent shortcut learning (e.g., detecting blur)
    instance_noise_std = float(unk_dict.get("instance_noise_std", 0.0))  # Initial std (0 = disabled)
    instance_noise_decay_steps = int(unk_dict.get("instance_noise_decay_steps", 50000))  # Decay to 0 over N steps
    # R1 gradient penalty: penalizes gradient norm on real images to learn real distribution
    r1_penalty_weight = float(unk_dict.get("r1_penalty_weight", 0.0))  # Weight (0 = disabled, 10.0 typical)
    r1_penalty_interval = int(unk_dict.get("r1_penalty_interval", 16))  # Apply every N steps (expensive)
    # GAN warmup: ramps GAN loss from 0 to full over N steps (0 = no warmup)
    gan_warmup_steps = int(unk_dict.get("gan_warmup_steps", 0))
    # Perceptual loss delayed start (0 = from start, >0 = delay to let L1/MSE settle)
    perceptual_loss_start_step = int(unk_dict.get("perceptual_loss_start_step", 0))

    # KL annealing: ramps KL weight from 0 to full over N steps (0 = disabled, no annealing)
    kl_annealing_steps = int(unk_dict.get("kl_annealing_steps", 0))

    # Free bits: minimum KL per channel to prevent posterior collapse (0 = disabled)
    free_bits = float(unk_dict.get("free_bits", 0.0))

    # EMA settings
    use_ema = unk_dict.get("use_ema", "false").lower() == "true"
    ema_decay = float(unk_dict.get("ema_decay", 0.9999))
    ema_update_after_step = int(unk_dict.get("ema_update_after_step", 0))

    model = model_config_lookup[args.config](
        latent_channels=latent_channels,
        perceptual_loss_type=perceptual_loss_type,
        lpips_net=lpips_net,
        recon_loss_weight=recon_loss_weight,
        mse_loss_weight=mse_loss_weight,
        l1_loss_weight=l1_loss_weight,
        kl_divergence_loss_weight=kl_divergence_loss_weight,
        free_bits=free_bits,
        perceptual_loss_weight=perceptual_loss_weight,
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
        if discriminator_config not in discriminators.model_config_lookup:
            raise ValueError(f"Unknown discriminator config: {discriminator_config}. Available: {list(discriminators.model_config_lookup.keys())}")

        # Handle stylegan discriminator which needs image_size
        if discriminator_config == "stylegan":
            discriminator = discriminators.model_config_lookup[discriminator_config](image_size=image_size).to(device)
        else:
            discriminator = discriminators.model_config_lookup[discriminator_config]().to(device)

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
        if use_gan and discriminator is not None:
            print(f"GAN training: enabled")
            print(f"  Discriminator config: {discriminator_config}")
            print(f"  Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
            print(f"  GAN loss weight: {gan_loss_weight}")
            print(f"  Feature matching weight: {feature_matching_weight}")
            print(f"  Discriminator LR: {discriminator_lr}")
            print(f"  GAN start condition key: {gan_start_condition_key}")
            print(f"  GAN start condition value: {gan_start_condition_value}")
            if instance_noise_std > 0:
                print(f"  Instance noise: std={instance_noise_std}, decay_steps={instance_noise_decay_steps}")
            if r1_penalty_weight > 0:
                print(f"  R1 penalty: weight={r1_penalty_weight}, interval={r1_penalty_interval}")
            if gan_warmup_steps > 0:
                print(f"  GAN warmup: {gan_warmup_steps} steps (ramps loss from 0 to full)")
        if perceptual_loss_start_step > 0:
            print(f"Perceptual loss: delayed start at step {perceptual_loss_start_step}")
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
        warmup_ratio=args.warmup_ratio if args.warmup_steps == 0 else 0.0,
        warmup_steps=args.warmup_steps if args.warmup_steps > 0 else 0,
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
        eval_steps=args.eval_steps if args.eval_steps > 0 else None,
    )

    # Load datasets
    train_dataset = CachedImageVAEDataset(
        cache_dir=train_cache_dir,
    )

    eval_dataset = CachedImageVAEDataset(
        cache_dir=val_cache_dir,
    )

    # Create data collator
    data_collator = ImageVAEDataCollator()

    # Create trainer
    trainer = ImageVAEGANTrainer(
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
        gan_start_condition_key=gan_start_condition_key,
        gan_start_condition_value=gan_start_condition_value,
        # Discriminator regularization
        instance_noise_std=instance_noise_std,
        instance_noise_decay_steps=instance_noise_decay_steps,
        r1_penalty_weight=r1_penalty_weight,
        r1_penalty_interval=r1_penalty_interval,
        gan_warmup_steps=gan_warmup_steps,
        perceptual_loss_start_step=perceptual_loss_start_step,
        kl_annealing_steps=kl_annealing_steps,
    )

    # Add visualization callback
    visualization_callback = ImageVAEReconstructionCallback(
        image_size=image_size,
        step_offset=args.start_step,
        generation_steps=args.generation_steps,
    )
    trainer.add_callback(visualization_callback)

    if args.stop_step > 0:
        early_stopping_callback = EarlyStoppingCallback(stop_step=args.stop_step)
        trainer.add_callback(early_stopping_callback)

    visualization_callback.trainer = trainer

    # Create EMA if enabled
    ema = None
    if use_ema:
        ema = EMAModel(
            model,
            decay=ema_decay,
            update_after_step=ema_update_after_step,
        )
        ema_callback = EMAUpdateCallback(ema=ema)
        trainer.add_callback(ema_callback)

        # Load EMA state if resuming from checkpoint
        if args.resume_from_checkpoint is not None:
            load_ema_state(ema, args.resume_from_checkpoint)

        if args.local_rank == 0 or not args.use_deepspeed:
            print(f"EMA enabled: decay={ema_decay}, update_after_step={ema_update_after_step}")

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

    print(f"Starting image VAE training with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
