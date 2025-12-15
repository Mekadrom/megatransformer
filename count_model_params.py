#!/usr/bin/env python3
"""
Utility script to instantiate models and print their parameter counts.

Usage:
    python count_model_params.py --model_type image_vae --config mini --latent_channels 4
    python count_model_params.py --model_type audio_vae --config small --latent_channels 8
    python count_model_params.py --model_type audio_vocoder --config tiny_lightheaded_freq_domain_vocoder
    python count_model_params.py --model_type audio_diffusion --config small_audio_diffusion
    python count_model_params.py --model_type image_diffusion --config small
    python count_model_params.py --list  # List all available configs
"""

import argparse
import torch


def count_parameters(model, verbose=False):
    """Count total, trainable, and non-trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    if verbose:
        print("\nParameter breakdown by module:")
        for name, module in model.named_children():
            params = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {params:,}")

    return total, trainable, non_trainable


def format_params(n):
    """Format parameter count with appropriate suffix."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


def get_image_vae(config: str, latent_channels: int = 4):
    """Load image VAE model."""
    from model.image.vae import model_config_lookup

    if config not in model_config_lookup:
        raise ValueError(f"Unknown config: {config}. Available: {list(model_config_lookup.keys())}")

    return model_config_lookup[config](
        latent_channels=latent_channels,
        perceptual_loss_type="none",
    )


def get_audio_vae(config: str, latent_channels: int = 8):
    """Load audio VAE model."""
    from model.audio.vae import model_config_lookup

    if config not in model_config_lookup:
        raise ValueError(f"Unknown config: {config}. Available: {list(model_config_lookup.keys())}")

    return model_config_lookup[config](
        latent_channels=latent_channels,
        perceptual_loss_type="none",
    )


def get_audio_vocoder(config: str):
    """Load audio vocoder model."""
    from model.audio.vocoders.vocoders import model_config_lookup
    from model.audio.shared_window_buffer import SharedWindowBuffer

    if config not in model_config_lookup:
        raise ValueError(f"Unknown config: {config}. Available: {list(model_config_lookup.keys())}")

    shared_window_buffer = SharedWindowBuffer()
    return model_config_lookup[config](shared_window_buffer=shared_window_buffer)


def get_audio_diffusion(config: str):
    """Load audio diffusion model."""
    from model.audio.diffusion import model_config_lookup

    if config not in model_config_lookup:
        raise ValueError(f"Unknown config: {config}. Available: {list(model_config_lookup.keys())}")

    return model_config_lookup[config]()


def get_image_diffusion(config: str, in_channels: int = 4, image_size: int = 32):
    """
    Load image diffusion UNet model.

    Note: Image diffusion doesn't have a config lookup, so we provide preset configs.
    in_channels should match VAE latent channels (or 3 for pixel-space diffusion).
    image_size is the latent size (image_size / VAE_downscale_factor).

    Returns the UNet directly since that contains the trainable parameters.
    """
    from model.megatransformer_diffusion import UNet
    from model.megatransformer_image_decoder import ImageSelfAttentionBlock, ImageCrossAttentionBlock
    from model import norms

    # Preset configurations
    configs = {
        "tiny": {
            "model_channels": 64,
            "channel_multipliers": [1, 2, 4],
            "attention_levels": [False, False, True],
            "num_res_blocks": 1,
        },
        "small": {
            "model_channels": 128,
            "channel_multipliers": [1, 2, 4],
            "attention_levels": [False, True, True],
            "num_res_blocks": 2,
        },
        "medium": {
            "model_channels": 192,
            "channel_multipliers": [1, 2, 3, 4],
            "attention_levels": [False, False, True, True],
            "num_res_blocks": 2,
        },
        "large": {
            "model_channels": 256,
            "channel_multipliers": [1, 2, 3, 4],
            "attention_levels": [False, True, True, True],
            "num_res_blocks": 3,
        },
    }

    if config not in configs:
        raise ValueError(f"Unknown config: {config}. Available: {list(configs.keys())}")

    cfg = configs[config]

    unet = UNet(
        activation="silu",
        self_attn_class=ImageSelfAttentionBlock,
        cross_attn_class=ImageCrossAttentionBlock,
        norm_class=norms.RMSNorm,
        in_channels=in_channels,
        out_channels=in_channels,
        model_channels=cfg["model_channels"],
        channel_multipliers=cfg["channel_multipliers"],
        attention_levels=cfg["attention_levels"],
        num_res_blocks=cfg["num_res_blocks"],
        time_embedding_dim=cfg["model_channels"] * 4,
    )

    return unet


def list_all_configs():
    """List all available configurations for each model type."""
    print("\n" + "=" * 60)
    print("AVAILABLE MODEL CONFIGURATIONS")
    print("=" * 60)

    # Image VAE
    print("\n[image_vae]")
    from model.image.vae import model_config_lookup as image_vae_configs
    for config in image_vae_configs.keys():
        print(f"  - {config}")
    print("  Options: --latent_channels (default: 4)")

    # Audio VAE
    print("\n[audio_vae]")
    from model.audio.vae import model_config_lookup as audio_vae_configs
    for config in audio_vae_configs.keys():
        print(f"  - {config}")
    print("  Options: --latent_channels (default: 8)")

    # Audio Vocoder
    print("\n[audio_vocoder]")
    from model.audio.vocoders.vocoders import model_config_lookup as vocoder_configs
    for config in vocoder_configs.keys():
        print(f"  - {config}")

    # Audio Diffusion
    print("\n[audio_diffusion]")
    from model.audio.diffusion import model_config_lookup as audio_diff_configs
    for config in audio_diff_configs.keys():
        print(f"  - {config}")

    # Image Diffusion
    print("\n[image_diffusion]")
    print("  - tiny")
    print("  - small")
    print("  - medium")
    print("  - large")
    print("  Options: --in_channels (default: 4), --image_size (default: 32)")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Count parameters for various model types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--model_type", type=str,
                        choices=["image_vae", "audio_vae", "audio_vocoder",
                                 "audio_diffusion", "image_diffusion"],
                        help="Type of model to instantiate")
    parser.add_argument("--config", type=str,
                        help="Model configuration name")
    parser.add_argument("--latent_channels", type=int, default=None,
                        help="Number of latent channels (for VAE models)")
    parser.add_argument("--in_channels", type=int, default=4,
                        help="Input channels for diffusion (default: 4 for latent diffusion)")
    parser.add_argument("--image_size", type=int, default=32,
                        help="Image/latent size for diffusion (default: 32)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show parameter breakdown by module")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List all available configurations")

    args = parser.parse_args()

    if args.list:
        list_all_configs()
        return

    if not args.model_type or not args.config:
        parser.print_help()
        print("\nError: --model_type and --config are required (or use --list)")
        return

    print(f"\nInstantiating {args.model_type} with config '{args.config}'...")

    try:
        if args.model_type == "image_vae":
            latent_channels = args.latent_channels or 4
            model = get_image_vae(args.config, latent_channels)
            print(f"  Latent channels: {latent_channels}")

        elif args.model_type == "audio_vae":
            latent_channels = args.latent_channels or 8
            model = get_audio_vae(args.config, latent_channels)
            print(f"  Latent channels: {latent_channels}")

        elif args.model_type == "audio_vocoder":
            model = get_audio_vocoder(args.config)

        elif args.model_type == "audio_diffusion":
            model = get_audio_diffusion(args.config)

        elif args.model_type == "image_diffusion":
            model = get_image_diffusion(args.config, args.in_channels, args.image_size)
            print(f"  In channels: {args.in_channels}")
            print(f"  Image size: {args.image_size}")

        total, trainable, non_trainable = count_parameters(model, verbose=args.verbose)

        print(f"\n{'=' * 40}")
        print(f"MODEL: {args.model_type} / {args.config}")
        print(f"{'=' * 40}")
        print(f"Total parameters:         {total:>12,}  ({format_params(total)})")
        print(f"Trainable parameters:     {trainable:>12,}  ({format_params(trainable)})")
        print(f"Non-trainable parameters: {non_trainable:>12,}  ({format_params(non_trainable)})")
        print(f"{'=' * 40}\n")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()