#!/usr/bin/env python3
"""
Analyze a trained DiT (Diffusion Transformer) checkpoint.

Features:
1. Parameter distribution analysis
2. Attention pattern visualization
3. Sample generation with attention maps
4. Layer-wise activation analysis

Usage:
    python scripts/analyze_dit_checkpoint.py \
        --diffusion_checkpoint runs/image_diffusion/test_flow_dit_2_2/checkpoint-6000/ \
        --vae_checkpoint runs/image_vae/best_0/checkpoint-314000/ \
        --output_dir logs/analyze_attention_dit_01/
"""

import argparse
import os
import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from model.image.diffusion import model_config_lookup
from model.image.vae import VAE, model_config_lookup as vae_config_lookup
from model.diffusion import DiTBackbone, DiTBlock


def count_parameters(module: nn.Module) -> int:
    """Count total parameters in a module."""
    return sum(p.numel() for p in module.parameters())


def analyze_dit_architecture(model, output_dir: Path):
    """Analyze DiT architecture and parameter distribution."""
    print("\n" + "=" * 80)
    print("DiT ARCHITECTURE ANALYSIS")
    print("=" * 80)

    # Find the DiT backbone
    dit = None
    if hasattr(model, 'unet') and isinstance(model.unet, DiTBackbone):
        dit = model.unet
    elif isinstance(model, DiTBackbone):
        dit = model
    else:
        print("Could not find DiTBackbone in model")
        return

    total_params = count_parameters(dit)
    print(f"\nTotal DiT parameters: {total_params:,}")

    # Component breakdown
    components = {
        "patch_embed": count_parameters(dit.patch_embed),
        "time_embed": count_parameters(dit.time_embed),
        "context_proj": count_parameters(dit.context_proj),
        "transformer_blocks": sum(count_parameters(b) for b in dit.blocks),
        "final_layer": count_parameters(dit.final_layer),
        "smooth (if exists)": count_parameters(dit.smooth) if hasattr(dit, 'smooth') else 0,
    }

    print(f"\n{'Component':<30} {'Params':>15} {'Percentage':>12}")
    print("-" * 60)
    for name, params in components.items():
        if params > 0:
            pct = 100 * params / total_params
            bar = "█" * int(pct / 2)
            print(f"{name:<30} {params:>15,} {pct:>10.1f}%  {bar}")

    # Per-block analysis
    print(f"\n{'Block':<20} {'Params':>15} {'Percentage':>12}")
    print("-" * 50)
    for i, block in enumerate(dit.blocks):
        params = count_parameters(block)
        pct = 100 * params / total_params
        print(f"Block {i:<15} {params:>15,} {pct:>10.1f}%")

    # Analyze a single block in detail
    if len(dit.blocks) > 0:
        block = dit.blocks[0]
        print(f"\n--- Block 0 Detailed Breakdown ---")
        for name, child in block.named_children():
            params = count_parameters(child)
            print(f"  {name}: {params:,}")

    # Save analysis to file
    analysis = {
        "total_params": total_params,
        "components": {k: v for k, v in components.items() if v > 0},
        "n_layers": len(dit.blocks),
        "hidden_size": dit.hidden_size,
        "patch_size": dit.patch_size,
        "grid_size": dit.grid_size,
        "num_patches": dit.num_patches,
    }

    with open(output_dir / "architecture_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\nArchitecture analysis saved to {output_dir / 'architecture_analysis.json'}")

    return dit


def register_attention_hooks(dit, attention_maps: dict):
    """Register forward hooks to capture attention patterns."""
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            # For flash attention, we can't easily get attention weights
            # But we can capture the input/output shapes and norms
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            attention_maps[name] = {
                "output_shape": list(out.shape),
                "output_norm": out.norm().item(),
                "output_mean": out.mean().item(),
                "output_std": out.std().item(),
            }
        return hook

    for i, block in enumerate(dit.blocks):
        # Hook self-attention
        if hasattr(block, 'self_attn'):
            h = block.self_attn.register_forward_hook(make_hook(f"block_{i}_self_attn"))
            hooks.append(h)
        # Hook cross-attention
        if hasattr(block, 'cross_attn') and block.cross_attn is not None:
            h = block.cross_attn.register_forward_hook(make_hook(f"block_{i}_cross_attn"))
            hooks.append(h)
        # Hook MLP
        if hasattr(block, 'mlp'):
            h = block.mlp.register_forward_hook(make_hook(f"block_{i}_mlp"))
            hooks.append(h)

    return hooks


def remove_hooks(hooks):
    """Remove registered hooks."""
    for h in hooks:
        h.remove()


def generate_samples_with_analysis(model, vae, output_dir: Path, device, n_samples=4, n_steps=50):
    """Generate samples and analyze intermediate activations."""
    print("\n" + "=" * 80)
    print("GENERATING SAMPLES WITH ACTIVATION ANALYSIS")
    print("=" * 80)

    model.eval()
    vae.eval()

    # Find DiT backbone
    dit = model.unet if hasattr(model, 'unet') else model

    # Register hooks to capture attention
    attention_maps = {}
    hooks = register_attention_hooks(dit, attention_maps)

    try:
        with torch.no_grad():
            # Generate unconditional samples
            print(f"Generating {n_samples} samples...")

            # Get latent shape from model
            grid_size = dit.grid_size
            in_channels = dit.in_channels
            latent_shape = (n_samples, in_channels, grid_size, grid_size)

            # Sample using the model's sample method
            if hasattr(model, 'sample'):
                # For FlowMatching or GaussianDiffusion
                samples = model.sample(
                    batch_size=n_samples,
                    device=device,
                    condition=None,
                    num_inference_steps=n_steps,
                )
            else:
                print("Model doesn't have sample() method")
                return

            # Decode through VAE
            print("Decoding latents through VAE...")
            images = vae.decode(samples)

            # Convert to displayable format
            images = (images + 1) / 2  # [-1, 1] -> [0, 1]
            images = torch.clamp(images, 0, 1)
            images = images.cpu()

            # Save samples
            fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 4))
            if n_samples == 1:
                axes = [axes]

            for i, ax in enumerate(axes):
                img = images[i].permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f'Sample {i}')

            plt.tight_layout()
            plt.savefig(output_dir / "generated_samples.png", dpi=150)
            plt.close()
            print(f"Saved samples to {output_dir / 'generated_samples.png'}")

            # Save attention analysis
            print("\nAttention/MLP activation statistics:")
            for name, stats in sorted(attention_maps.items()):
                print(f"  {name}: norm={stats['output_norm']:.4f}, mean={stats['output_mean']:.4f}, std={stats['output_std']:.4f}")

            with open(output_dir / "attention_stats.json", "w") as f:
                json.dump(attention_maps, f, indent=2)

    finally:
        remove_hooks(hooks)


def analyze_timestep_embeddings(dit, output_dir: Path, device):
    """Analyze how timestep embeddings vary across timesteps."""
    print("\n" + "=" * 80)
    print("TIMESTEP EMBEDDING ANALYSIS")
    print("=" * 80)

    timesteps = torch.linspace(0, 1, 100).to(device)

    with torch.no_grad():
        embeddings = dit.time_embed(timesteps)

    embeddings = embeddings.cpu().numpy()

    # Plot embedding norms across timesteps
    norms = np.linalg.norm(embeddings, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(timesteps.cpu().numpy(), norms)
    axes[0].set_xlabel('Timestep (t)')
    axes[0].set_ylabel('Embedding L2 Norm')
    axes[0].set_title('Timestep Embedding Magnitude')
    axes[0].grid(True)

    # Plot first few dimensions
    for i in range(min(5, embeddings.shape[1])):
        axes[1].plot(timesteps.cpu().numpy(), embeddings[:, i], label=f'dim {i}', alpha=0.7)
    axes[1].set_xlabel('Timestep (t)')
    axes[1].set_ylabel('Embedding Value')
    axes[1].set_title('First 5 Embedding Dimensions')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "timestep_embeddings.png", dpi=150)
    plt.close()
    print(f"Saved timestep analysis to {output_dir / 'timestep_embeddings.png'}")


def analyze_positional_embeddings(dit, output_dir: Path):
    """Visualize 2D positional embeddings."""
    print("\n" + "=" * 80)
    print("POSITIONAL EMBEDDING ANALYSIS")
    print("=" * 80)

    pos_embed = dit.pos_embed.squeeze(0).cpu().numpy()  # [N, D]
    grid_size = dit.grid_size

    print(f"Positional embedding shape: {pos_embed.shape}")
    print(f"Grid size: {grid_size}x{grid_size}")

    # Reshape to grid for visualization
    pos_embed_grid = pos_embed.reshape(grid_size, grid_size, -1)

    # Plot first 4 dimensions as heatmaps
    n_dims = min(8, pos_embed.shape[1])
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(n_dims):
        im = axes[i].imshow(pos_embed_grid[:, :, i], cmap='viridis')
        axes[i].set_title(f'Dim {i}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046)

    plt.suptitle('Positional Embedding Dimensions (reshaped to grid)')
    plt.tight_layout()
    plt.savefig(output_dir / "positional_embeddings.png", dpi=150)
    plt.close()
    print(f"Saved positional embedding visualization to {output_dir / 'positional_embeddings.png'}")


def load_dit_checkpoint(checkpoint_path: str, config_name: str = "small_dit_flow", context_dim: int = 512):
    """Load DiT model from checkpoint."""
    print(f"Loading DiT from {checkpoint_path}...")

    # Create model with same config
    model = model_config_lookup[config_name](
        context_dim=context_dim,
        cfg_dropout_prob=0.0,
        timestep_sampling="uniform",
    )

    # Load weights
    state_dict_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.exists(state_dict_path):
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {state_dict_path}")
    else:
        print(f"Warning: No pytorch_model.bin found at {state_dict_path}")

    return model


def load_vae_checkpoint(checkpoint_path: str, config_name: str = "small", latent_channels: int = 4):
    """Load VAE model from checkpoint."""
    print(f"Loading VAE from {checkpoint_path}...")

    # Create model
    model = vae_config_lookup[config_name](latent_channels=latent_channels)

    # Load weights
    state_dict_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.exists(state_dict_path):
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {state_dict_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Analyze DiT checkpoint")
    parser.add_argument("--diffusion_checkpoint", type=str, required=True,
                        help="Path to diffusion model checkpoint")
    parser.add_argument("--vae_checkpoint", type=str, required=True,
                        help="Path to VAE checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save analysis outputs")
    parser.add_argument("--dit_config", type=str, default="small_dit_flow",
                        help="DiT config name (tiny_dit_flow, small_dit_flow)")
    parser.add_argument("--vae_config", type=str, default="small",
                        help="VAE config name")
    parser.add_argument("--context_dim", type=int, default=512,
                        help="Context dimension for conditioning")
    parser.add_argument("--n_samples", type=int, default=4,
                        help="Number of samples to generate")
    parser.add_argument("--n_steps", type=int, default=50,
                        help="Number of sampling steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--latent_channels", type=int, default=4,
                        help="Number of latent channels")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip sample generation (faster)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load models
    dit_model = load_dit_checkpoint(
        args.diffusion_checkpoint,
        config_name=args.dit_config,
        context_dim=args.context_dim,
    )
    dit_model = dit_model.to(device)
    dit_model.eval()

    vae_model = load_vae_checkpoint(args.vae_checkpoint, config_name=args.vae_config, latent_channels=args.latent_channels)
    vae_model = vae_model.to(device)
    vae_model.eval()

    # Run analyses
    dit = analyze_dit_architecture(dit_model, output_dir)

    if dit is not None:
        analyze_timestep_embeddings(dit, output_dir, device)
        analyze_positional_embeddings(dit, output_dir)

    if not args.skip_generation:
        generate_samples_with_analysis(
            dit_model, vae_model, output_dir, device,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
        )

    print("\n" + "=" * 80)
    print(f"Analysis complete! Results saved to {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
