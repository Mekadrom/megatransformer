"""
Training script for RecurrentVAE with optional GAN training.

This is a separate experimental script for the recurrent VAE architecture.
Kept separate from pretrain_image_vae.py to avoid entanglement.

Example usage (without GAN):
    python pretrain_recurrent_vae.py \
        --run_name recurrent_vae_test \
        --config recurrent_vae_small \
        --train_cache_dir cached_datasets/cc3m_train_vae_cached \
        --val_cache_dir cached_datasets/coco_val_vae_cached \
        --max_steps 50000 \
        --batch_size 8 \
        --learning_rate 1e-4

Example usage (with GAN, triggered by reconstruction loss threshold):
    python pretrain_recurrent_vae.py \
        --run_name recurrent_vae_gan_test \
        --config recurrent_vae_small \
        --train_cache_dir cached_datasets/cc3m_train_vae_cached \
        --val_cache_dir cached_datasets/coco_val_vae_cached \
        --max_steps 100000 \
        --use_gan true \
        --gan_start_condition_key reconstruction_criteria_met \
        --gan_start_condition_value 0.01 \
        --discriminator_config multi_scale \
        --gan_loss_weight 0.5
"""

import os

os.environ["DEEPSPEED_UNIT_TEST"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ['NCCL_TIMEOUT'] = '1200000'

import torch

from typing import Any, Mapping, Optional, Union

from PIL import Image
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers.integrations import TensorBoardCallback

from dataset_loading.image_vae_dataset import CachedImageVAEDataset, ImageVAEDataCollator
from model.image.recurrent_vae import RecurrentVAE, model_config_lookup
from model.image import discriminators
from model.image.discriminators import compute_discriminator_loss, compute_generator_gan_loss
from utils import megatransformer_utils


def get_writer(trainer: Trainer) -> Optional[SummaryWriter]:
    """Get TensorBoard writer from trainer callbacks."""
    if hasattr(trainer, 'callback_handler'):
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback):
                if callback.tb_writer is not None:
                    return callback.tb_writer
    return None


class DebugStartCallback(TrainerCallback):
    """
    Callback that enables debug mode on the model after a specified step.
    Useful for debugging issues that appear later in training without
    the verbosity of debug logging from the start.
    """

    def __init__(self, debug_start_step: int, step_offset: Optional[int] = 0):
        self.debug_start_step = debug_start_step
        self.step_offset = step_offset if step_offset is not None else 0
        self.debug_enabled = False

    def on_step_begin(self, args, state, control, model: RecurrentVAE = None, **kwargs):
        if self.debug_enabled:
            return

        global_step = state.global_step + self.step_offset
        if global_step >= self.debug_start_step:
            print(f"[DebugStartCallback] Enabling debug mode at step {global_step}")
            model.set_debug(True)
            self.debug_enabled = True


class RecurrentVAEReconstructionCallback(TrainerCallback):
    """
    Callback for logging RecurrentVAE image reconstruction during training.
    Logs reconstruction quality, iteration counts, and convergence metrics.
    """

    def __init__(
        self,
        image_size: int = 256,
        step_offset: int = 0,
        generation_steps: int = 1000,
    ):
        self.trainer: Optional[Trainer] = None
        self.step_offset = step_offset if step_offset is not None else 0
        self.generation_steps = generation_steps
        self.image_size = image_size

        self.example_paths = [
            "inference/examples/test_vlm1_x256.png",
            "inference/examples/test_vlm2_x256.png",
            "inference/examples/test_vlm3_x256.png",
        ]

        # VAE uses [-1, 1] normalization (for tanh output)
        transform = transforms.Compose([
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
                    transforms.ToTensor(),  # [0, 1] for direct visualization
                ])
                self.example_images_unnorm.append(unnorm_transform(image))

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Unnormalize image tensor from [-1, 1] (tanh output) back to [0, 1] for visualization."""
        return (x + 1.0) / 2.0

    def on_step_end(self, args, state, control, model: RecurrentVAE = None, **kwargs):
        global_step = state.global_step + self.step_offset

        if ((global_step == 1) or (global_step % self.generation_steps == 0)) and state.is_world_process_zero:
            writer = get_writer(self.trainer)
            if writer is None:
                print("No TensorBoard writer found, skipping visualization...")
                return

            print(f"Generating image reconstructions at step {global_step}...")

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
                    for i, (image, image_unnorm) in enumerate(zip(self.example_images, self.example_images_unnorm)):
                        recon, info = model(image.unsqueeze(0).to(device))

                        # Unnormalize reconstruction for visualization
                        recon_unnorm = self.unnormalize(recon[0].float().cpu())
                        recon_unnorm = torch.clamp(recon_unnorm, 0, 1)

                        # Log original and reconstructed images (image_vae/ prefix to match pretrain_image_vae.py)
                        writer.add_image(f"image_vae/original/{i}", image_unnorm, global_step)
                        writer.add_image(f"image_vae/recon/{i}", recon_unnorm, global_step)

                        # Log per-example metrics (image_vae/ prefix to match pretrain_image_vae.py)
                        writer.add_scalar(f"image_vae/example_{i}/recon_loss", info["recon_loss"].item(), global_step)
                        writer.add_scalar(f"image_vae/example_{i}/kl_loss", info["kl_loss"].item(), global_step)

                        # Recurrence-specific per-example metrics
                        writer.add_scalar(f"image_vae/example_{i}/encoder_iters", info["encoder_iterations"], global_step)
                        writer.add_scalar(f"image_vae/example_{i}/decoder_iters", info["decoder_iterations"], global_step)

                        # Log mu channels as grayscale images
                        mu = info["mu"]
                        mu_unnorm = (mu[0].float().cpu() - mu[0].float().cpu().min()) / (mu[0].float().cpu().max() - mu[0].float().cpu().min() + 1e-5)
                        for c in range(mu_unnorm.shape[0]):
                            writer.add_image(f"image_vae/example_{i}/mu_channel_{c}", mu_unnorm[c:c+1, :, :], global_step)


class RecurrentVAEGANTrainer(Trainer):
    """
    Custom trainer for RecurrentVAE with optional GAN training.
    Handles alternating generator/discriminator updates and logs recurrence-specific metrics.
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
        discriminator_update_frequency: int = 1,
        gan_start_condition_key: Optional[str] = None,
        gan_start_condition_value: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.writer = None
        self.step_offset = step_offset if step_offset is not None else 0
        self.cmdline = cmdline
        self.git_commit_hash = git_commit_hash

        # GAN components
        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.gan_loss_weight = gan_loss_weight
        self.discriminator_update_frequency = discriminator_update_frequency
        self.gan_start_condition_key = gan_start_condition_key
        self.gan_start_condition_value = gan_start_condition_value
        self.gan_already_started = False

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

    def is_gan_enabled(self, global_step, vae_loss) -> bool:
        """Check if GAN training should be enabled based on conditions."""
        return (
            self.discriminator is not None and
            self.gan_start_condition_key is not None and
            self.gan_start_condition_value is not None and
            (
                self.gan_already_started or
                (self.gan_start_condition_key == "step" and global_step >= self.gan_start_condition_value) or
                (self.gan_start_condition_key == 'reconstruction_criteria_met' and vae_loss.mean().item() <= float(self.gan_start_condition_value))
            )
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        global_step = self.state.global_step + self.step_offset

        self._ensure_tensorboard_writer()

        if global_step == 0 and self.writer is not None:
            self.writer.add_text("training/command_line", self.cmdline, global_step)
            self.writer.add_text("training/git_commit_hash", self.git_commit_hash, global_step)

        image = inputs["image"]

        # Forward pass through RecurrentVAE
        recon, info = model(image)

        # Get VAE loss from info dict (without GAN)
        vae_loss = info["loss"]

        # GAN losses (only after condition is met)
        g_gan_loss = torch.tensor(0.0, device=image.device)
        d_loss = torch.tensor(0.0, device=image.device)
        d_loss_dict = {}

        if self.is_gan_enabled(global_step, vae_loss):
            if not self.gan_already_started:
                print(f"GAN training starting at step {global_step}")
            self.gan_already_started = True

            # Discriminator update
            if global_step % self.discriminator_update_frequency == 0:
                self.discriminator.train()

                # Get discriminator loss
                with torch.enable_grad():
                    recon_detached = recon.detach()
                    d_loss, d_loss_dict = compute_discriminator_loss(
                        self.discriminator,
                        image,
                        recon_detached,
                        loss_type="hinge"
                    )

                # Update discriminator
                if self.discriminator_optimizer is not None and self.discriminator.training:
                    self.discriminator_optimizer.zero_grad()
                    d_loss.backward()
                    self.discriminator_optimizer.step()

            # Generator GAN Loss
            self.discriminator.eval()
            with torch.no_grad():
                self.discriminator.requires_grad_(False)
            g_gan_loss, g_loss_dict = compute_generator_gan_loss(
                self.discriminator,
                recon,
                loss_type="hinge"
            )
            with torch.no_grad():
                self.discriminator.requires_grad_(True)

        # Total loss = VAE loss + weighted GAN loss
        total_loss = vae_loss + self.gan_loss_weight * g_gan_loss

        # Log losses and metrics
        if global_step % self.args.logging_steps == 0 and self.writer is not None:
            prefix = "train/" if model.training else "eval/"

            # Core losses (vae_ prefix to match pretrain_image_vae.py)
            self._log_scalar(f"{prefix}total_loss", total_loss.mean(), global_step)
            self._log_scalar(f"{prefix}vae_loss", vae_loss.mean(), global_step)
            self._log_scalar(f"{prefix}vae_recon_loss", info["recon_loss"].mean(), global_step)
            self._log_scalar(f"{prefix}vae_kl_divergence", info["kl_loss"].mean(), global_step)
            self._log_scalar(f"{prefix}vae_perceptual_loss", info["perceptual_loss"].mean(), global_step)

            # Latent stats (vae_ prefix to match pretrain_image_vae.py)
            mu = info["mu"]
            logvar = info["logvar"]
            self._log_scalar(f"{prefix}vae_mu_mean", mu.mean(), global_step)
            self._log_scalar(f"{prefix}vae_mu_std", mu.std(), global_step)
            self._log_scalar(f"{prefix}vae_logvar_mean", logvar.mean(), global_step)

            # GAN losses
            self._log_scalar(f"{prefix}g_gan_loss", g_gan_loss, global_step)
            self._log_scalar(f"{prefix}d_loss", d_loss, global_step)
            for key, value in d_loss_dict.items():
                self._log_scalar(f"{prefix}d_{key}", value, global_step)

            # Recurrence metrics (recurrence/ prefix for recurrent-specific metrics)
            self._log_scalar(f"{prefix}recurrence/encoder_n_steps", info["encoder_n_steps"], global_step)
            self._log_scalar(f"{prefix}recurrence/encoder_k_steps", info["encoder_k_steps"], global_step)
            self._log_scalar(f"{prefix}recurrence/encoder_iterations", info["encoder_iterations"], global_step)
            self._log_scalar(f"{prefix}recurrence/decoder_n_steps", info["decoder_n_steps"], global_step)
            self._log_scalar(f"{prefix}recurrence/decoder_k_steps", info["decoder_k_steps"], global_step)
            self._log_scalar(f"{prefix}recurrence/decoder_iterations", info["decoder_iterations"], global_step)
            self._log_scalar(f"{prefix}recurrence/total_iterations", info["encoder_iterations"] + info["decoder_iterations"], global_step)

            # Log KL convergence (how much KL changed over encoder iterations)
            kl_history = info.get("kl_history", [])
            if len(kl_history) > 1:
                kl_delta = abs(kl_history[-1] - kl_history[-2])
                self._log_scalar(f"{prefix}recurrence/kl_delta_final", kl_delta, global_step)

            # Log output convergence (how much output changed over decoder iterations)
            delta_history = info.get("delta_history", [])
            if len(delta_history) > 0:
                self._log_scalar(f"{prefix}recurrence/output_delta_final", delta_history[-1], global_step)

        outputs = {
            "loss": total_loss,
            "rec": recon,
        }

        return (total_loss, outputs) if return_outputs else total_loss

    def _save(self, output_dir, state_dict=None):
        """Save both VAE and discriminator."""
        # Let parent class handle VAE saving
        super()._save(output_dir, state_dict)

        global_step = self.state.global_step + self.step_offset

        # Save discriminator if GAN is enabled
        if self.is_gan_enabled(global_step, torch.tensor(float('inf'))):
            if self.discriminator is not None:
                discriminator_path = os.path.join(output_dir, "discriminator.pt")
                torch.save({
                    "discriminator_state_dict": self.discriminator.state_dict(),
                    "discriminator_optimizer_state_dict": (
                        self.discriminator_optimizer.state_dict()
                        if self.discriminator_optimizer is not None else None
                    ),
                }, discriminator_path)
                print(f"Discriminator saved to {discriminator_path}")

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


def load_discriminator(
    resume_from_checkpoint: Optional[str],
    discriminator: torch.nn.Module,
    discriminator_optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
):
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
        raise ValueError(f"Unknown RecurrentVAE config: {args.config}. Available: {list(model_config_lookup.keys())}")

    # Parse extra arguments
    unk_dict = {}
    for i in range(0, len(unk), 2):
        unk_dict[unk[i].lstrip('-')] = unk[i+1]

    # Dataset settings
    train_cache_dir = unk_dict.get("train_cache_dir", "./cached_datasets/cc3m_train_vae_cached")
    val_cache_dir = unk_dict.get("val_cache_dir", "./cached_datasets/coco_val_vae_cached")
    image_size = int(unk_dict.get("image_size", 256))

    # RecurrentVAE specific settings
    latent_channels = int(unk_dict.get("latent_channels", 4))
    mean_iterations = int(unk_dict.get("mean_iterations", 16))
    backprop_depth = int(unk_dict.get("backprop_depth", 8))
    encoder_exit_threshold = float(unk_dict.get("encoder_exit_threshold", 0.01))
    decoder_exit_threshold = float(unk_dict.get("decoder_exit_threshold", 0.001))
    gradient_injection_scale = float(unk_dict.get("gradient_injection_scale", 0.1))

    # Lockstep settings (sync iterations across GPUs)
    lockstep_n = unk_dict.get("lockstep_n", "true").lower() == "true"
    lockstep_k = unk_dict.get("lockstep_k", "true").lower() == "true"

    # Gradient injection type: "additive" or "concat"
    h_injection_type = unk_dict.get("h_injection_type", "additive")  # "additive" or "concat"

    # Loss weights (named to match pretrain_image_vae.py for consistency)
    kl_divergence_loss_weight = float(unk_dict.get("kl_divergence_loss_weight", 1e-6))
    perceptual_loss_weight = float(unk_dict.get("perceptual_loss_weight", 0.1))
    iteration_cost_weight = float(unk_dict.get("iteration_cost_weight", 0.0))

    # Perceptual loss settings
    perceptual_loss_type = unk_dict.get("perceptual_loss_type", "vgg")  # "vgg", "lpips", or "none"
    lpips_net = unk_dict.get("lpips_net", "vgg")  # "alex", "vgg", or "squeeze"

    # Debug mode for numerical stability diagnostics
    debug = unk_dict.get("debug", "false").lower() == "true"
    debug_start_step = int(unk_dict.get("debug_start_step", 0))  # Delay debug logging until this step

    # GAN training settings
    use_gan = unk_dict.get("use_gan", "false").lower() == "true"
    gan_start_condition_key = unk_dict.get("gan_start_condition_key", None)  # "step" or "reconstruction_criteria_met"
    gan_start_condition_value = unk_dict.get("gan_start_condition_value", None)
    discriminator_lr = float(unk_dict.get("discriminator_lr", 2e-4))
    gan_loss_weight = float(unk_dict.get("gan_loss_weight", 0.5))
    discriminator_update_frequency = int(unk_dict.get("discriminator_update_frequency", 1))
    discriminator_config = unk_dict.get("discriminator_config", "multi_scale")

    # Configure debug logging if enabled (either immediately or delayed)
    if debug or debug_start_step > 0:
        import logging
        logging.getLogger('model.image.recurrent_vae').setLevel(logging.DEBUG)
        # Also configure a handler if none exists
        logger = logging.getLogger('model.image.recurrent_vae')
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    # If debug_start_step > 0, start with debug disabled (callback will enable it later)
    initial_debug = debug and (debug_start_step == 0)

    # Create model
    model = model_config_lookup[args.config](
        latent_channels=latent_channels,
        mean_iterations=mean_iterations,
        backprop_depth=backprop_depth,
        encoder_exit_threshold=encoder_exit_threshold,
        decoder_exit_threshold=decoder_exit_threshold,
        gradient_injection_scale=gradient_injection_scale,
        lockstep_n=lockstep_n,
        lockstep_k=lockstep_k,
        h_injection_type=h_injection_type,
        activation_fn='tanh',
        kl_weight=kl_divergence_loss_weight,
        perceptual_loss_weight=perceptual_loss_weight,
        iteration_cost_weight=iteration_cost_weight,
        perceptual_loss_type=perceptual_loss_type,
        lpips_net=lpips_net,
        debug=initial_debug,
    )

    # Try to load existing checkpoint
    model, model_loaded = megatransformer_utils.load_model(False, model, run_dir)

    # Determine device for discriminator
    if torch.distributed.is_initialized():
        device = torch.device(f"cuda:{torch.distributed.get_rank()}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
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
        print(f"  Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
        print(f"  Decoder parameters: {sum(p.numel() for p in model.decoder.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"RecurrentVAE settings:")
        print(f"  mean_iterations: {mean_iterations}")
        print(f"  backprop_depth: {backprop_depth}")
        print(f"  encoder_exit_threshold: {encoder_exit_threshold}")
        print(f"  decoder_exit_threshold: {decoder_exit_threshold}")
        print(f"  gradient_injection_scale: {gradient_injection_scale}")
        print(f"  lockstep_n: {lockstep_n}")
        print(f"  lockstep_k: {lockstep_k}")
        print(f"  kl_divergence_loss_weight: {kl_divergence_loss_weight}")
        print(f"  perceptual_loss_weight: {perceptual_loss_weight}")
        print(f"  perceptual_loss_type: {perceptual_loss_type}")
        if perceptual_loss_type == "lpips":
            print(f"  lpips_net: {lpips_net}")
        print(f"  iteration_cost_weight: {iteration_cost_weight}")
        print(f"  debug: {debug}")
        if debug_start_step > 0:
            print(f"  debug_start_step: {debug_start_step}")

        if use_gan and discriminator is not None:
            print(f"GAN training: enabled")
            print(f"  Discriminator config: {discriminator_config}")
            print(f"  Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
            print(f"  GAN loss weight: {gan_loss_weight}")
            print(f"  Discriminator update frequency: {discriminator_update_frequency}")
            print(f"  Discriminator LR: {discriminator_lr}")
            print(f"  GAN start condition key: {gan_start_condition_key}")
            print(f"  GAN start condition value: {gan_start_condition_value}")

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
        remove_unused_columns=False
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
    trainer = RecurrentVAEGANTrainer(
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
        discriminator_update_frequency=discriminator_update_frequency,
        gan_start_condition_key=gan_start_condition_key,
        gan_start_condition_value=gan_start_condition_value,
    )

    # Add visualization callback
    visualization_callback = RecurrentVAEReconstructionCallback(
        image_size=image_size,
        step_offset=args.start_step,
        generation_steps=args.generation_steps,
    )
    trainer.add_callback(visualization_callback)

    if args.stop_step > 0:
        early_stopping_callback = megatransformer_utils.EarlyStoppingCallback(stop_step=args.stop_step)
        trainer.add_callback(early_stopping_callback)

    # Add debug start callback if delayed debug is enabled
    if debug_start_step > 0:
        debug_callback = DebugStartCallback(
            debug_start_step=debug_start_step,
            step_offset=args.start_step,
        )
        trainer.add_callback(debug_callback)

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

    print(f"Starting RecurrentVAE training with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()