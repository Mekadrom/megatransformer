#!/usr/bin/env python3
"""
Generate interpolated images between two source images using a trained image VAE.

This script encodes two images to latent space, interpolates between them,
and decodes the interpolated latents back to images.

Usage:
    python interpolate_image_latents.py \
        --image1 path/to/image1.jpg \
        --image2 path/to/image2.jpg \
        --vae_checkpoint runs/image_vae/checkpoint-10000 \
        --output_dir outputs/interpolation \
        --num_steps 10
"""

import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image, make_grid


def load_image_vae(checkpoint_path: str, vae_config: str, latent_channels: int, device: str = "cuda"):
    """Load an image VAE from a checkpoint."""
    from model.image.vae import model_config_lookup

    if vae_config not in model_config_lookup:
        raise ValueError(f"Unknown VAE config: {vae_config}. Available: {list(model_config_lookup.keys())}")

    model = model_config_lookup[vae_config](
        latent_channels=latent_channels,
        perceptual_loss_type="none",
    )

    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        # strict=False ignores LPIPS weights if checkpoint was trained with perceptual loss
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            # Filter out expected missing keys (lpips)
            missing = [k for k in missing if "lpips" not in k.lower()]
            if missing:
                print(f"Warning: Missing keys: {missing}")
        print(f"Loaded VAE from {safetensors_path}")
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location=device)
        # strict=False ignores LPIPS weights if checkpoint was trained with perceptual loss
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            # Filter out expected missing keys (lpips)
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


def load_and_preprocess_image(image_path: str, image_size: int = 256) -> torch.Tensor:
    """Load and preprocess an image for VAE encoding."""
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
    ])

    image = Image.open(image_path).convert("RGB")
    return transform(image)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize from [-1, 1] to [0, 1] for saving."""
    return (tensor + 1) / 2


@torch.no_grad()
def encode_image(vae, image: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """Encode an image to latent space, returning the mean (mu)."""
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    mu, _ = vae.encoder(image)
    return mu


@torch.no_grad()
def decode_latent(vae, latent: torch.Tensor) -> torch.Tensor:
    """Decode a latent back to image space."""
    return vae.decoder(latent)


def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """
    Spherical linear interpolation between two tensors.

    Better than linear interpolation for high-dimensional spaces
    as it maintains constant speed along the geodesic.
    """
    # Flatten for computation
    v0_flat = v0.flatten()
    v1_flat = v1.flatten()

    # Normalize
    v0_norm = v0_flat / (torch.norm(v0_flat) + 1e-8)
    v1_norm = v1_flat / (torch.norm(v1_flat) + 1e-8)

    # Compute angle
    dot = torch.clamp(torch.dot(v0_norm, v1_norm), -1.0, 1.0)
    theta = torch.acos(dot)

    # If vectors are nearly parallel, use linear interpolation
    if theta.abs() < 1e-4:
        return (1 - t) * v0 + t * v1

    sin_theta = torch.sin(theta)
    s0 = torch.sin((1 - t) * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta

    result_flat = s0 * v0_flat + s1 * v1_flat
    return result_flat.view(v0.shape)


def lerp(t: float, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """Linear interpolation between two tensors."""
    return (1 - t) * v0 + t * v1


def main():
    parser = argparse.ArgumentParser(
        description="Interpolate between two images in VAE latent space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--image1", type=str, required=True,
                        help="Path to first image")
    parser.add_argument("--image2", type=str, required=True,
                        help="Path to second image")
    parser.add_argument("--vae_checkpoint", type=str, required=True,
                        help="Path to VAE checkpoint directory")
    parser.add_argument("--vae_config", type=str, default="mini",
                        help="VAE config name (tiny, mini)")
    parser.add_argument("--latent_channels", type=int, default=4,
                        help="Number of latent channels")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size for processing")
    parser.add_argument("--output_dir", type=str, default="outputs/interpolation",
                        help="Directory to save interpolated images")
    parser.add_argument("--num_steps", type=int, default=10,
                        help="Number of interpolation steps (including endpoints)")
    parser.add_argument("--interpolation", type=str, default="slerp",
                        choices=["slerp", "lerp"],
                        help="Interpolation method: slerp (spherical) or lerp (linear)")
    parser.add_argument("--save_grid", action="store_true",
                        help="Also save a grid of all interpolated images")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load VAE
    print(f"Loading VAE from {args.vae_checkpoint}...")
    vae = load_image_vae(args.vae_checkpoint, args.vae_config, args.latent_channels, args.device)

    # Load and preprocess images
    print(f"Loading images...")
    print(f"  Image 1: {args.image1}")
    print(f"  Image 2: {args.image2}")

    image1 = load_and_preprocess_image(args.image1, args.image_size)
    image2 = load_and_preprocess_image(args.image2, args.image_size)

    # Encode to latent space
    print("Encoding images to latent space...")
    latent1 = encode_image(vae, image1, args.device)
    latent2 = encode_image(vae, image2, args.device)

    print(f"  Latent shape: {latent1.shape}")

    # Choose interpolation function
    interp_fn = slerp if args.interpolation == "slerp" else lerp

    # Generate interpolations
    print(f"Generating {args.num_steps} interpolation steps using {args.interpolation}...")

    interpolated_images = []

    for i, t in enumerate(torch.linspace(0, 1, args.num_steps)):
        t_val = t.item()

        # Interpolate in latent space
        latent_interp = interp_fn(t_val, latent1, latent2)

        # Decode back to image
        decoded = decode_latent(vae, latent_interp)
        decoded = denormalize(decoded.squeeze(0).cpu())
        decoded = decoded.clamp(0, 1)

        interpolated_images.append(decoded)

        # Save individual image
        save_path = os.path.join(args.output_dir, f"interp_{i:03d}_t{t_val:.3f}.png")
        save_image(decoded, save_path)
        print(f"  Saved: {save_path}")

    # Save grid if requested
    if args.save_grid:
        grid = make_grid(interpolated_images, nrow=args.num_steps, padding=2)
        grid_path = os.path.join(args.output_dir, "interpolation_grid.png")
        save_image(grid, grid_path)
        print(f"\nSaved grid: {grid_path}")

    # Also save reconstructions of original images for comparison
    print("\nSaving reconstructions of original images...")

    recon1 = decode_latent(vae, latent1)
    recon1 = denormalize(recon1.squeeze(0).cpu()).clamp(0, 1)
    save_image(recon1, os.path.join(args.output_dir, "recon_image1.png"))

    recon2 = decode_latent(vae, latent2)
    recon2 = denormalize(recon2.squeeze(0).cpu()).clamp(0, 1)
    save_image(recon2, os.path.join(args.output_dir, "recon_image2.png"))

    # Save originals for comparison (denormalized)
    orig1 = denormalize(image1).clamp(0, 1)
    orig2 = denormalize(image2).clamp(0, 1)
    save_image(orig1, os.path.join(args.output_dir, "orig_image1.png"))
    save_image(orig2, os.path.join(args.output_dir, "orig_image2.png"))

    print(f"\nDone! All images saved to {args.output_dir}")
    print(f"  - orig_image1.png, orig_image2.png: Original images")
    print(f"  - recon_image1.png, recon_image2.png: VAE reconstructions")
    print(f"  - interp_*.png: Interpolated images")
    if args.save_grid:
        print(f"  - interpolation_grid.png: All steps in a grid")


if __name__ == "__main__":
    main()