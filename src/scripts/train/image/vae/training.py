import os
import torch


from model.discriminator import compute_adaptive_weight
from model.image.vae.discriminator import InstanceNoiseScheduler, MultiScalePatchDiscriminator, add_instance_noise, compute_discriminator_loss, compute_generator_gan_loss, r1_gradient_penalty
from model.image.vae.vae import ImageVAE
from scripts.train.trainer import CommonTrainer
from torch.amp import autocast
from transformers.integrations import TensorBoardCallback
from typing import Any, Mapping, Optional, Union

from utils import model_loading_utils


class ImageVAEGANTrainer(CommonTrainer):
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
        # Adaptive discriminator weighting (VQGAN-style)
        use_adaptive_weight: bool = False,  # Automatically balance GAN vs reconstruction gradients
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.writer = None

        self.step_offset = step_offset if step_offset is not None else 0
        self.cmdline = cmdline
        self.git_commit_hash = git_commit_hash

        self.discriminator = discr
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

        # Adaptive discriminator weighting (VQGAN-style)
        self.use_adaptive_weight = use_adaptive_weight

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

        self.has_logged_cli = False

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

        # Total generator loss with optional adaptive weighting
        adaptive_weight = torch.tensor(self.gan_loss_weight, device=image.device)
        if self.use_adaptive_weight and self.gan_already_started and g_gan_loss.requires_grad:
            # VQGAN-style adaptive weighting: balance GAN and reconstruction gradients
            # Uses the last decoder layer as proxy for decoder output gradients
            try:
                last_layer = model.decoder.final_conv.weight
                adaptive_weight = compute_adaptive_weight(
                    vae_loss, g_gan_loss, last_layer, self.gan_loss_weight
                )
            except (AttributeError, RuntimeError) as e:
                # Fallback to fixed weight if adaptive weight fails
                # (e.g., model doesn't have expected structure, or grad computation fails)
                if global_step % self.args.logging_steps == 0:
                    print(f"Warning: adaptive weight computation failed ({e}), using fixed weight")
                adaptive_weight = torch.tensor(self.gan_loss_weight, device=image.device)

        total_loss = vae_loss + adaptive_weight * g_gan_loss

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
            # Log adaptive weight when using adaptive weighting
            if self.use_adaptive_weight and self.gan_already_started:
                self._log_scalar(f"{prefix}adaptive_gan_weight", adaptive_weight, global_step)

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


def load_model(args):
    return model_loading_utils.load_model(ImageVAE, args.config,  checkpoint_path=args.resume_from_checkpoint, overrides={
        "latent_channels": args.latent_channels,
    })


def create_trainer(
    args,
    model,
    training_args,
    data_collator,
    train_dataset,
    eval_dataset,
    device,
):
    # Create discriminator if GAN training is enabled
    discriminator = None
    discriminator_optimizer = None
    if args.use_gan:
        discriminator = MultiScalePatchDiscriminator.from_config(args.discriminator_config, overrides={"use_spectral_norm": args.discriminator_spectral_norm}).to(device)

        discriminator_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=args.discriminator_lr,
            betas=(0.0, 0.99),
            weight_decay=0.0,
        )

        # Try to load existing discriminator checkpoint
        discriminator, discriminator_optimizer, disc_loaded = model_loading_utils.load_discriminator(
            args.resume_from_checkpoint, discriminator, discriminator_optimizer, device
        )
        if disc_loaded:
            print("Loaded discriminator from checkpoint")

    return ImageVAEGANTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        cmdline=args.cmdline,
        git_commit_hash=args.commit_hash,
        step_offset=args.start_step,
        discriminator=discriminator if args.use_gan else None,
        discriminator_optimizer=discriminator_optimizer if args.use_gan else None,
        gan_loss_weight=args.gan_loss_weight,
        feature_matching_weight=args.feature_matching_weight,
        discriminator_update_frequency=args.discriminator_update_frequency,
        gan_start_condition_key=args.gan_start_condition_key,
        gan_start_condition_value=args.gan_start_condition_value,
        # Discriminator regularization
        instance_noise_std=args.instance_noise_std,
        instance_noise_decay_steps=args.instance_noise_decay_steps,
        r1_penalty_weight=args.r1_penalty_weight,
        r1_penalty_interval=args.r1_penalty_interval,
        gan_warmup_steps=args.gan_warmup_steps,
        perceptual_loss_start_step=args.perceptual_loss_start_step,
        kl_annealing_steps=args.kl_annealing_steps,
        use_adaptive_weight=args.use_adaptive_weight,
    )


def add_cli_args(subparsers):
    sub_parser = subparsers.add_parser("image-vae", help="Train image VAE with optional GAN")

    sub_parser.add_argument("--image_size", type=int, default=256,
                           help="Input image size (assumed square)")
    sub_parser.add_argument("--latent_channels", type=int, default=4,
                            help="Number of latent channels in VAE")

    # VAE loss weights
    sub_parser.add_argument("--recon_loss_weight", type=float, default=1.0,
                           help="Weight for reconstruction loss")
    sub_parser.add_argument("--mse_loss_weight", type=float, default=1.0,
                           help="Weight for MSE loss")
    sub_parser.add_argument("--l1_loss_weight", type=float, default=1.0,
                           help="Weight for L1 loss")
    sub_parser.add_argument("--kl_divergence_loss_weight", type=float, default=1e-6,
                           help="Weight for KL divergence loss")
    sub_parser.add_argument("--perceptual_loss_weight", type=float, default=0.1,
                           help="Weight for perceptual loss (VGG or LPIPS)")

    # Perceptual loss type: "vgg", "lpips", or "none"
    sub_parser.add_argument("--perceptual_loss_type", type=str, default="lpips",
                           help="Type of perceptual loss to use (vgg, lpips, none)")
    sub_parser.add_argument("--lpips_net", type=str, default="vgg",
                           help="LPIPS network to use (alex, vgg, squeeze)")

    # DINO perceptual loss (semantic features, complementary to VGG/LPIPS)
    # Helps preserve content correctness while being less sensitive to texture artifacts
    sub_parser.add_argument("--dino_loss_weight", type=float, default=0.0,
                            help="Weight for DINO perceptual loss")
    sub_parser.add_argument("--dino_model", type=str, default="dinov2_vits14",
                            help="DINO model to use (dinov2_vits14, dinov2_vitb14, dinov2_vitl14)")

    # GAN training settings
    sub_parser.add_argument("--use_gan", action="store_true",
                           help="Enable GAN training with discriminator")
    sub_parser.add_argument("--gan_start_condition_key", type=str, default=None,
                           help="Condition key to start GAN training (step or reconstruction_criteria_met)")
    sub_parser.add_argument("--gan_start_condition_value", type=str, default=None,
                           help="Condition value to start GAN training")
    sub_parser.add_argument("--discriminator_lr", type=float, default=2e-4,
                           help="Learning rate for discriminator optimizer")
    sub_parser.add_argument("--gan_loss_weight", type=float, default=0.5,
                           help="Weight for GAN loss contribution to generator")
    sub_parser.add_argument("--feature_matching_weight", type=float, default=0.0,
                            help="Weight for feature matching loss from discriminator")
    sub_parser.add_argument("--discriminator_config", type=str, default="multi_scale",
                           help="Discriminator configuration (multi_scale, etc.)")

    # Discriminator regularization settings
    # Instance noise: adds Gaussian noise to prevent shortcut learning (e.g., detecting blur)
    sub_parser.add_argument("--instance_noise_std", type=float, default=0.0,
                           help="Initial stddev for instance noise added to real and fake images (0 = disabled)")
    sub_parser.add_argument("--instance_noise_decay_steps", type=int, default=50000,
                           help="Number of steps to decay instance noise to 0")
    # R1 gradient penalty: penalizes gradient norm on real images to learn real distribution
    sub_parser.add_argument("--r1_penalty_weight", type=float, default=0.0,
                           help="Weight for R1 gradient penalty (0 = disabled)")
    sub_parser.add_argument("--r1_penalty_interval", type=int, default=16,
                           help="Interval (in steps) to apply R1 penalty (expensive)")
    # GAN warmup: ramps GAN loss from 0 to full over N steps (0 = no warmup)
    sub_parser.add_argument("--gan_warmup_steps", type=int, default=0,
                           help="Number of steps to ramp up GAN loss (0 = no warmup)")
    # Discriminator update frequency: update D every N generator steps (1 = every step, 2 = every other step)
    sub_parser.add_argument("--discriminator_update_frequency", type=int, default=1,
                           help="Discriminator update frequency: update D every N generator steps (1 = every step")
    # Adaptive discriminator weighting (VQGAN-style): automatically balances GAN vs reconstruction gradients
    # This prevents the discriminator from dominating and causing artifacts
    sub_parser.add_argument("--use_adaptive_weight", action="store_true",
                           help="Use adaptive weighting for GAN loss contribution")
    # Perceptual loss delayed start (0 = from start, >0 = delay to let L1/MSE settle)
    sub_parser.add_argument("--perceptual_loss_start_step", type=int, default=0,
                           help="Step to start applying perceptual loss (0 = from start)")

    # CONFLICTS WITH R1 PENALTY; ONLY ENABLE ONE OF THESE REGULARIZATION TYPES AT A TIME
    sub_parser.add_argument("--discriminator_spectral_norm", type=str, default="true",
                           help="Whether to use spectral normalization in the discriminator (true/false)")

    # KL annealing: ramps KL weight from 0 to full over N steps (0 = disabled, no annealing)
    sub_parser.add_argument("--kl_annealing_steps", type=int, default=0,
                           help="Number of steps to ramp up KL weight (0 = disabled)")

    # Free bits: minimum KL per channel to prevent posterior collapse (0 = disabled)
    sub_parser.add_argument("--free_bits", type=float, default=0.0,
                           help="Free bits for KL divergence loss per channel (0 = disabled)")

    # Dataset caching directory
    sub_parser.add_argument("--cache_dir", type=str, default=None,
                           help="Directory to cache datasets")
    
    return sub_parser
