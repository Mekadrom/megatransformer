import os

from dataset_loading.image_vae_dataset import CachedImageVAEDataset, ImageVAEDataCollator
from model.image.vae import VAE, model_config_lookup

os.environ["DEEPSPEED_UNIT_TEST"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ['NCCL_TIMEOUT'] = '1200000'

from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers.integrations import TensorBoardCallback
from typing import Any, Mapping, Optional, Union

import megatransformer_utils
import torch
from torchvision import transforms
from PIL import Image


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
    Callback for logging VAE image reconstruction during training.
    Periodically reconstructs test images and logs to TensorBoard.
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

        # ImageNet normalization stats
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        self.example_paths = ["inference/examples/test_vlm1_x256.png", "inference/examples/test_vlm2_x256.png"]

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.example_images = []
        self.example_images_unnorm = []  # For visualization
        for path in self.example_paths:
            if os.path.exists(path):
                image = Image.open(path).convert('RGB')
                image_tensor = transform(image)
                self.example_images.append(image_tensor)
                # Store unnormalized version for visualization
                unnorm_transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                ])
                self.example_images_unnorm.append(unnorm_transform(image))

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Unnormalize image tensor from ImageNet stats back to [0, 1]."""
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def on_step_end(self, args, state, control, model: VAE = None, **kwargs):
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
                        recon, mu, logvar, losses = model(image.unsqueeze(0).to(device))

                        # Unnormalize reconstruction for visualization
                        recon_unnorm = self.unnormalize(recon[0].float().cpu())
                        recon_unnorm = torch.clamp(recon_unnorm, 0, 1)

                        # Log original and reconstructed images
                        writer.add_image(f"image_vae/original/{i}", image_unnorm, global_step)
                        writer.add_image(f"image_vae/recon/{i}", recon_unnorm, global_step)

                        # Log per-example losses
                        for loss_name, loss_val in losses.items():
                            if isinstance(loss_val, torch.Tensor):
                                loss_val = loss_val.item()
                            writer.add_scalar(f"image_vae/example_{i}/{loss_name}", loss_val, global_step)

class ImageVAETrainer(Trainer):
    def __init__(
        self,
        *args,
        cmdline,
        git_commit_hash,
        step_offset: int = 0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.writer = None

        self.step_offset = step_offset if step_offset is not None else 0
        self.cmdline = cmdline
        self.git_commit_hash = git_commit_hash

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

        images = inputs["images"]

        # Forward pass through vae model
        recon, mu, logvar, losses = model(images)

        # Log losses
        if global_step % self.args.logging_steps == 0 and self.writer is not None:
            for loss_name, loss in losses.items():
                prefix = "train/" if self.training else "eval/"
                self._log_scalar(f"{prefix}vae_{loss_name}", loss, global_step)
            # log mu and logvar stats
            self._log_scalar(f"{prefix}vae_mu_mean", mu.mean(), global_step)
            self._log_scalar(f"{prefix}vae_logvar_mean", logvar.mean(), global_step)

        outputs = {
            "loss": losses["total_loss"],
            "rec": recon,
        }

        return (losses["total_loss"], outputs) if return_outputs else losses["total_loss"]

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
        raise ValueError(f"Unknown image VAE config: {args.config}. Available: {list(model_config_lookup.keys())}")

    # Parse extra arguments
    unk_dict = {}
    for i in range(0, len(unk), 2):
        unk_dict[unk[i].lstrip('-')] = unk[i+1]

    # Dataset settings
    train_cache_dir = unk_dict.get("train_cache_dir", "./cached_datasets/relaion400m_train_vae_cached")
    val_cache_dir = unk_dict.get("val_cache_dir", "./cached_datasets/relaion400m_val_vae_cached")
    image_size = int(unk_dict.get("image_size", 256))
    latent_channels = int(unk_dict.get("latent_channels", 4))

    # loss weights
    recon_loss_weight = float(unk_dict.get("recon_loss_weight", 1.0))
    kl_divergence_loss_weight = float(unk_dict.get("kl_divergence_loss_weight", 0.001))
    perceptual_loss_weight = float(unk_dict.get("perceptual_loss_weight", 0.1))

    model = model_config_lookup[args.config](
        latent_channels=latent_channels,
        recon_loss_weight=recon_loss_weight,
        kl_divergence_loss_weight=kl_divergence_loss_weight,
        perceptual_loss_weight=perceptual_loss_weight,
    )

    # Try to load existing checkpoint
    model, model_loaded = megatransformer_utils.load_model(False, model, run_dir)

    if args.local_rank == 0 or not args.use_deepspeed:
        print(f"Model structure: {model}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  VAE Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
        print(f"  VAE Decoder parameters: {sum(p.numel() for p in model.decoder.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

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
    train_dataset = CachedImageVAEDataset(
        cache_dir=train_cache_dir,
    )

    eval_dataset = CachedImageVAEDataset(
        cache_dir=val_cache_dir,
    )

    # Create data collator
    data_collator = ImageVAEDataCollator()

    # Create trainer
    trainer = ImageVAETrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        cmdline=args.cmdline,
        git_commit_hash=args.commit_hash,
        step_offset=args.start_step,
    )

    # Add visualization callback
    visualization_callback = ImageVAEReconstructionCallback(
        image_size=image_size,
        step_offset=args.start_step,
        generation_steps=args.generation_steps,
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

    print(f"Starting image VAE training with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
