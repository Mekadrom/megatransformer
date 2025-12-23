#!/usr/bin/env python3
"""
Analyze what a diffusion model pays attention to in its conditioning.

This script helps understand which input signals a diffusion model uses
to determine its output, useful for debugging memorization and conditioning.

Methods:
1. Cross-attention weight visualization - shows which conditioning tokens
   are attended to at each spatial location
2. Gradient-based saliency - shows which conditioning dimensions most
   affect the noise prediction
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from model.image.diffusion import model_config_lookup as diffusion_config_lookup
from model.image.vae import model_config_lookup as vae_config_lookup
from dataset_loading.image_diffusion_dataset import CachedImageDiffusionDataset
from pretrain_image_diffusion import ImageDiffusionModelWithT5ConditioningAdapter, load_ema_state
from model.ema import EMAModel


class AttentionHook:
    """Hook to capture attention weights from cross-attention layers."""

    def __init__(self):
        self.attention_weights: List[torch.Tensor] = []
        self.handles = []

    def hook_fn(self, module, input, output):
        """Capture attention weights computed inside the forward pass."""
        # This will be called after forward, so we need to recompute attention
        # For analysis purposes, we'll store the output and compute separately
        pass

    def clear(self):
        self.attention_weights = []

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def compute_cross_attention_weights(
    q: torch.Tensor,  # [B, n_heads, H*W, d_queries]
    k: torch.Tensor,  # [B, n_heads, T, d_queries]
    d_queries: int,
) -> torch.Tensor:
    """Compute attention weights for visualization."""
    scale = 1.0 / (d_queries ** 0.5)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, n_heads, H*W, T]
    attn_weights = F.softmax(attn_scores, dim=-1)
    return attn_weights


def load_diffusion_model(
    checkpoint_path: str,
    config_name: str,
    context_dim: int = 512,
    device: str = "cuda",
    load_ema: bool = True,
) -> Tuple[nn.Module, Optional[EMAModel]]:
    """Load diffusion model from checkpoint with condition adapter wrapper.

    This loads the model the same way pretrain_image_diffusion.py does,
    including the ImageDiffusionModelWithT5ConditioningAdapter wrapper.

    Args:
        checkpoint_path: Path to checkpoint directory
        config_name: Name of model config
        context_dim: Context dimension for conditioning
        device: Device to load model to
        load_ema: Whether to load EMA weights if available

    Returns:
        Tuple of (model, ema) where ema is None if not available or not requested
    """
    if config_name not in diffusion_config_lookup:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(diffusion_config_lookup.keys())}")

    # Create base model
    # IMPORTANT: normalize=False for latent diffusion (VAE latents are already ~N(0,1))
    # This must match the training configuration in pretrain_image_diffusion.py
    base_model = diffusion_config_lookup[config_name](context_dim=context_dim, normalize=False)

    # Wrap with condition adapter (same as training script)
    model = ImageDiffusionModelWithT5ConditioningAdapter(base_model, context_dim=context_dim)

    # Try to load checkpoint
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        # Load full state dict including condition_adapter
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: Missing keys: {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys: {unexpected}")
        print(f"Loaded diffusion model from {safetensors_path}")
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: Missing keys: {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys: {unexpected}")
        print(f"Loaded diffusion model from {pytorch_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    model = model.to(device)
    model.eval()

    # Try to load EMA weights
    ema = None
    if load_ema:
        ema_path = os.path.join(checkpoint_path, "ema_state.pt")
        if os.path.exists(ema_path):
            ema = EMAModel(model, decay=0.9999, device=device)
            load_ema_state(ema, checkpoint_path)
            print(f"Loaded EMA weights (step {ema.step})")
        else:
            print(f"No EMA weights found at {ema_path}")

    return model, ema


def load_vae(
    checkpoint_path: str,
    config_name: str,
    latent_channels: int = 4,
    device: str = "cuda",
) -> nn.Module:
    """Load VAE from checkpoint."""
    if config_name not in vae_config_lookup:
        raise ValueError(f"Unknown VAE config: {config_name}")

    model = vae_config_lookup[config_name](
        latent_channels=latent_channels,
        perceptual_loss_type="none",
    )

    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded VAE from {safetensors_path}")
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded VAE from {pytorch_path}")
    else:
        raise FileNotFoundError(f"No VAE checkpoint found at {checkpoint_path}")

    model = model.to(device)
    model.eval()
    return model


def compute_conditioning_saliency(
    model: nn.Module,
    latent: torch.Tensor,
    condition: torch.Tensor,
    timestep: int = 500,
) -> torch.Tensor:
    """
    Compute gradient-based saliency of conditioning on noise prediction.

    Returns gradient magnitude for each conditioning dimension, showing
    which parts of the conditioning most affect the model output.
    """
    condition = condition.clone().requires_grad_(True)

    # Get the base diffusion model (handles wrapper)
    base_model = model.model if hasattr(model, 'model') else model

    # Apply condition adapter if present (same as during training/sampling)
    adapted_condition = condition
    if hasattr(model, 'condition_adapter'):
        adapted_condition = model.condition_adapter(condition)
        adapted_condition.retain_grad()  # Keep grad for adapted condition too

    # Get noise prediction at specified timestep
    t = torch.tensor([timestep], device=latent.device)

    # Add noise to latent
    noise = torch.randn_like(latent)
    alpha_bar = base_model.alphas_cumprod[t].view(-1, 1, 1, 1)
    noisy_latent = torch.sqrt(alpha_bar) * latent + torch.sqrt(1 - alpha_bar) * noise

    # Forward pass
    noise_pred = base_model.unet(noisy_latent, t, adapted_condition)

    # Compute gradient of output w.r.t. conditioning
    # We use the L2 norm of the noise prediction as the scalar output
    output_norm = noise_pred.pow(2).sum()
    output_norm.backward()

    # Saliency is the gradient magnitude
    saliency = condition.grad.abs()  # [1, T, D]

    return saliency.detach()


def compute_attention_maps(
    model: nn.Module,
    latent: torch.Tensor,
    condition: torch.Tensor,
    timestep: int = 500,
    layer_idx: int = -1,
) -> Dict[str, torch.Tensor]:
    """
    Extract cross-attention maps from the model.

    Returns attention weights showing which conditioning tokens each
    spatial location attends to.
    """
    attention_maps = {}

    # Get the base diffusion model (handles wrapper)
    base_model = model.model if hasattr(model, 'model') else model

    # Apply condition adapter if present (same as during training/sampling)
    adapted_condition = condition
    if hasattr(model, 'condition_adapter'):
        with torch.no_grad():
            adapted_condition = model.condition_adapter(condition)

    # Hook to capture attention computations - use pre-hook to capture inputs
    def make_hook(name):
        def hook_fn(module, input, output):
            # Only process if we have the right number of inputs
            if len(input) < 2:
                return

            x, context = input[0], input[1]

            # Skip if context is None
            if context is None:
                return

            try:
                B, C, H, W = x.size()
                _, T, _ = context.size()

                # Get projections (need to run them again since we're in the hook)
                with torch.no_grad():
                    q = module.q_proj(x)  # [B, n_heads*d_queries, H, W]
                    k = module.k_proj(context)  # [B, T, n_heads*d_queries]

                    # Reshape for attention
                    n_heads = module.n_heads
                    d_queries = module.d_queries

                    q = q.view(B, n_heads, d_queries, -1).transpose(-2, -1)  # [B, n_heads, H*W, d_q]
                    k = k.view(B, T, n_heads, d_queries).permute(0, 2, 1, 3)  # [B, n_heads, T, d_q]

                    # Compute attention weights
                    scale = 1.0 / (d_queries ** 0.5)
                    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
                    attn_weights = F.softmax(attn_weights, dim=-1)  # [B, n_heads, H*W, T]

                    # Average over heads and reshape to spatial
                    attn_avg = attn_weights.mean(dim=1)  # [B, H*W, T]
                    attn_spatial = attn_avg.view(B, H, W, T)  # [B, H, W, T]

                    attention_maps[name] = attn_spatial.detach().cpu()
            except Exception as e:
                pass  # Skip modules that don't match expected structure

        return hook_fn

    # Find cross-attention layers and hook them (only ImageCrossAttentionBlockSimple)
    hooks = []
    cross_attn_idx = 0
    for name, module in base_model.unet.named_modules():
        # Only hook the actual cross-attention blocks, not their children
        if type(module).__name__ == "ImageCrossAttentionBlockSimple":
            hook = module.register_forward_hook(make_hook(f"cross_attn_{cross_attn_idx}_{name}"))
            hooks.append(hook)
            cross_attn_idx += 1

    print(f"Hooked {cross_attn_idx} cross-attention layers")

    # Forward pass
    t = torch.tensor([timestep], device=latent.device)
    noise = torch.randn_like(latent)
    alpha_bar = base_model.alphas_cumprod[t].view(-1, 1, 1, 1)
    noisy_latent = torch.sqrt(alpha_bar) * latent + torch.sqrt(1 - alpha_bar) * noise

    with torch.no_grad():
        _ = base_model.unet(noisy_latent, t, adapted_condition)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return attention_maps


def visualize_attention_to_conditioning(
    attention_maps: Dict[str, torch.Tensor],
    condition_tokens: int,
    output_dir: str,
    sample_idx: int,
):
    """Visualize which conditioning tokens each spatial location attends to."""
    os.makedirs(output_dir, exist_ok=True)

    for layer_name, attn in attention_maps.items():
        # attn shape: [B, H, W, T]
        attn = attn[0]  # Take first batch element
        H, W, T = attn.shape

        # Create figure with attention to each token
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Mean attention per spatial location (which tokens are attended most)
        ax = axes[0, 0]
        mean_attn = attn.mean(dim=-1)  # [H, W] - average attention weight
        im = ax.imshow(mean_attn.numpy(), cmap='viridis')
        ax.set_title(f'Mean attention weight per location')
        plt.colorbar(im, ax=ax)

        # 2. Which token index receives most attention at each location
        ax = axes[0, 1]
        max_token_idx = attn.argmax(dim=-1)  # [H, W]
        im = ax.imshow(max_token_idx.numpy(), cmap='tab20', vmin=0, vmax=min(20, T))
        ax.set_title(f'Most attended token index (0-{T-1})')
        plt.colorbar(im, ax=ax)

        # 3. Entropy of attention (low = focused, high = diffuse)
        ax = axes[0, 2]
        # Add small epsilon to avoid log(0)
        entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1)
        im = ax.imshow(entropy.numpy(), cmap='plasma')
        ax.set_title('Attention entropy (low=focused)')
        plt.colorbar(im, ax=ax)

        # 4-6. Attention to specific token ranges
        token_ranges = [
            (0, T // 3, "First third of tokens"),
            (T // 3, 2 * T // 3, "Middle third of tokens"),
            (2 * T // 3, T, "Last third of tokens"),
        ]

        for i, (start, end, label) in enumerate(token_ranges):
            ax = axes[1, i]
            range_attn = attn[:, :, start:end].sum(dim=-1)
            im = ax.imshow(range_attn.numpy(), cmap='viridis')
            ax.set_title(f'Attention to {label}')
            plt.colorbar(im, ax=ax)

        plt.suptitle(f'{layer_name} - Sample {sample_idx}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'attention_{layer_name}_sample_{sample_idx}.png'), dpi=150)
        plt.close()

    print(f"Saved attention visualizations to {output_dir}")


def visualize_conditioning_saliency(
    saliency: torch.Tensor,  # [1, T, D]
    output_dir: str,
    sample_idx: int,
):
    """Visualize which conditioning dimensions are most important."""
    os.makedirs(output_dir, exist_ok=True)

    saliency = saliency[0]  # [T, D]
    T, D = saliency.shape

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Saliency heatmap (tokens x dimensions)
    ax = axes[0, 0]
    # Subsample dimensions for visibility
    step = max(1, D // 128)
    saliency_sub = saliency[:, ::step].numpy()
    im = ax.imshow(saliency_sub, aspect='auto', cmap='hot')
    ax.set_xlabel(f'Dimension (subsampled 1/{step})')
    ax.set_ylabel('Token index')
    ax.set_title('Conditioning saliency (gradient magnitude)')
    plt.colorbar(im, ax=ax)

    # 2. Total saliency per token
    ax = axes[0, 1]
    token_saliency = saliency.sum(dim=-1).numpy()
    ax.bar(range(T), token_saliency)
    ax.set_xlabel('Token index')
    ax.set_ylabel('Total gradient magnitude')
    ax.set_title('Importance per conditioning token')

    # 3. Total saliency per dimension
    ax = axes[1, 0]
    dim_saliency = saliency.sum(dim=0).numpy()
    ax.plot(dim_saliency)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Total gradient magnitude')
    ax.set_title('Importance per embedding dimension')

    # 4. Top tokens
    ax = axes[1, 1]
    top_k = min(20, T)
    top_tokens = token_saliency.argsort()[-top_k:][::-1]
    ax.barh(range(top_k), token_saliency[top_tokens])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f'Token {i}' for i in top_tokens])
    ax.set_xlabel('Saliency')
    ax.set_title(f'Top {top_k} most important tokens')

    plt.suptitle(f'Conditioning Saliency Analysis - Sample {sample_idx}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'saliency_sample_{sample_idx}.png'), dpi=150)
    plt.close()


def visualize_sample_comparison(
    vae: nn.Module,
    latent: torch.Tensor,
    generated_latent: torch.Tensor,
    output_dir: str,
    sample_idx: int,
):
    """Compare original and generated images."""
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        # Decode latents to images
        original_img = vae.decoder(latent)
        generated_img = vae.decoder(generated_latent)

        # Convert to numpy for plotting
        original_np = original_img[0].float().cpu().permute(1, 2, 0).numpy()
        generated_np = generated_img[0].float().cpu().permute(1, 2, 0).numpy()

        # Use dynamic min-max normalization (same as training visualization)
        # This handles cases where VAE output isn't exactly in [-1, 1]
        original_np = (original_np - original_np.min()) / (original_np.max() - original_np.min() + 1e-8)
        generated_np = (generated_np - generated_np.min()) / (generated_np.max() - generated_np.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_np)
    axes[0].set_title('Original (from training data)')
    axes[0].axis('off')

    axes[1].imshow(generated_np)
    axes[1].set_title('Generated (from model)')
    axes[1].axis('off')

    # Difference
    diff = np.abs(original_np - generated_np)
    axes[2].imshow(diff)
    axes[2].set_title(f'Absolute difference (mean={diff.mean():.4f})')
    axes[2].axis('off')

    plt.suptitle(f'Sample {sample_idx} - Original vs Generated')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_sample_{sample_idx}.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze diffusion model attention patterns")
    parser.add_argument("--diffusion_checkpoint", type=str, required=True,
                       help="Path to diffusion model checkpoint")
    parser.add_argument("--diffusion_config", type=str, required=True,
                       help="Diffusion model config name")
    parser.add_argument("--vae_checkpoint", type=str, required=True,
                       help="Path to VAE checkpoint")
    parser.add_argument("--vae_config", type=str, required=True,
                       help="VAE config name")
    parser.add_argument("--train_data_dir", type=str, default="./cached_datasets/image_train_diffusion_latents",
                       help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="./logs/attention_analysis",
                       help="Output directory for visualizations")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of samples to analyze")
    parser.add_argument("--timesteps", type=str, default="100,500,900",
                       help="Comma-separated timesteps to analyze")
    parser.add_argument("--context_dim", type=int, default=512,
                       help="Context dimension for conditioning")
    parser.add_argument("--latent_channels", type=int, default=4,
                       help="VAE latent channels")
    parser.add_argument("--sampling_steps", type=int, default=50,
                       help="Steps for generating comparison samples (50+ recommended for DPM-Solver++)")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Classifier-free guidance scale")
    parser.add_argument("--image_size", type=int, default=32,
                       help="Latent image size")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    timesteps = [int(t) for t in args.timesteps.split(",")]

    print(f"Loading diffusion model from {args.diffusion_checkpoint}...")
    diffusion_model, ema = load_diffusion_model(
        args.diffusion_checkpoint,
        args.diffusion_config,
        context_dim=args.context_dim,
        device=device,
        load_ema=True,
    )

    print(f"Loading VAE from {args.vae_checkpoint}...")
    vae = load_vae(
        args.vae_checkpoint,
        args.vae_config,
        latent_channels=args.latent_channels,
        device=device,
    )

    print(f"Loading training data from {args.train_data_dir}...")
    dataset = CachedImageDiffusionDataset(args.train_data_dir)

    print(f"\nAnalyzing {args.num_samples} samples at timesteps {timesteps}...")

    for sample_idx in range(min(args.num_samples, len(dataset))):
        print(f"\n{'='*50}")
        print(f"Analyzing sample {sample_idx}")
        print(f"{'='*50}")

        # Load sample
        sample = dataset[sample_idx]
        latent = sample["latent_mu"].unsqueeze(0).to(device)
        condition = sample["text_embeddings"].unsqueeze(0).to(device)

        print(f"  Latent shape: {latent.shape}")
        print(f"  Condition shape: {condition.shape}")

        sample_output_dir = os.path.join(args.output_dir, f"sample_{sample_idx}")

        for timestep in timesteps:
            print(f"\n  Timestep {timestep}:")
            timestep_dir = os.path.join(sample_output_dir, f"timestep_{timestep}")

            # 1. Compute attention maps
            print("    Computing attention maps...")
            attention_maps = compute_attention_maps(
                diffusion_model, latent, condition, timestep=timestep
            )
            if attention_maps:
                visualize_attention_to_conditioning(
                    attention_maps, condition.shape[1], timestep_dir, sample_idx
                )
            else:
                print("    No cross-attention layers found!")

            # 2. Compute conditioning saliency
            print("    Computing conditioning saliency...")
            saliency = compute_conditioning_saliency(
                diffusion_model, latent, condition, timestep=timestep
            )
            visualize_conditioning_saliency(saliency.cpu(), timestep_dir, sample_idx)

        # 3. Generate sample and compare (using same sampling as pretrain_image_diffusion.py)
        print("\n  Generating sample for comparison...")

        # Use autocast to match training (bf16)
        from torch.amp import autocast
        from contextlib import nullcontext
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_type = "cuda" if "cuda" in device else "cpu"

        # Use EMA weights for sampling if available (same as training visualization)
        ema_context = ema.apply_ema() if ema is not None else nullcontext()

        with torch.no_grad():
            with ema_context:
                with autocast(device_type, dtype=dtype):
                    result = diffusion_model.sample(
                        device=device,
                        batch_size=1,
                        condition=condition,
                        return_intermediate=True,
                        override_sampling_steps=args.sampling_steps,
                        image_size=args.image_size,
                        generator=torch.Generator(device).manual_seed(42),
                        guidance_scale=args.guidance_scale,
                        sampler="dpm_solver_pp",
                        dpm_solver_order=2,
                    )
                # Result is (generated_latents, noise_preds, x_start_preds)
                generated_latents, noise_preds, x_start_preds = result
                generated = generated_latents[0] if isinstance(generated_latents, list) else generated_latents

            # Print latent statistics for debugging
            print(f"    Original latent: mean={latent.mean():.4f}, std={latent.std():.4f}, min={latent.min():.4f}, max={latent.max():.4f}")
            print(f"    Generated latent: mean={generated.mean():.4f}, std={generated.std():.4f}, min={generated.min():.4f}, max={generated.max():.4f}")

        visualize_sample_comparison(
            vae, latent, generated, sample_output_dir, sample_idx
        )

    print(f"\nAnalysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
