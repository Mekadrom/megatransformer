import os

import matplotlib.pyplot as plt
import torch


from PIL import Image
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from transformers import Trainer
from typing import Optional

from model.image.vae.vae import ImageVAE
from scripts.train.visualization_callback import VisualizationCallback
from utils.train_utils import get_writer


class ImageVAEVisualizationCallback(VisualizationCallback):
    """
    Callback for logging VAE image reconstruction during training and evaluation.
    Periodically reconstructs test images and logs to TensorBoard.

    VAE uses [-1, 1] normalization (not ImageNet), with tanh output activation.
    """

    def __init__(
        self,
        image_size: int = 256,
        step_offset: int = 0,
        num_eval_samples: int = 8,
    ):
        self.trainer: Optional[Trainer] = None
        self.step_offset = step_offset if step_offset is not None else 0
        self.image_size = image_size
        self.num_eval_samples = num_eval_samples

        # VAE uses [-1, 1] normalization (for tanh output)
        transform = transforms.Compose([
            # transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1]
        ])

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

    def on_evaluate(self, args, state, control, model: ImageVAE = None, **kwargs):
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
