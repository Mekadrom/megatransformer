#!/usr/bin/env python3
"""
Analyze and configure image diffusion model architecture.

This script helps design balanced diffusion models by:
1. Creating models with specified configurations
2. Analyzing parameter distribution across all components
3. Identifying imbalances and suggesting fixes
4. Comparing different configurations

Usage:
    python analyze_diffusion_architecture.py --model_channels 128 --channel_multipliers 1,2,4 --num_res_blocks 2
    python analyze_diffusion_architecture.py --preset small
    python analyze_diffusion_architecture.py --target_params 30000000  # Auto-configure for ~30M params
"""

import argparse
import sys
import os
from collections import defaultdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn


def count_parameters(module: nn.Module) -> int:
    """Count total parameters in a module."""
    return sum(p.numel() for p in module.parameters())


def count_trainable_parameters(module: nn.Module) -> int:
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def analyze_parameter_distribution(model: nn.Module, prefix: str = "") -> dict:
    """
    Recursively analyze parameter distribution in a model.
    Returns a nested dict with parameter counts at each level.
    """
    result = {
        "total_params": count_parameters(model),
        "trainable_params": count_trainable_parameters(model),
        "children": {}
    }

    for name, child in model.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        result["children"][name] = analyze_parameter_distribution(child, child_prefix)

    return result


def print_parameter_tree(analysis: dict, name: str = "model", indent: int = 0, min_params: int = 100):
    """Print parameter distribution as a tree."""
    total = analysis["total_params"]
    if total < min_params:
        return

    indent_str = "  " * indent
    print(f"{indent_str}{name}: {total:,} params")

    # Sort children by parameter count (descending)
    children = sorted(
        analysis["children"].items(),
        key=lambda x: x[1]["total_params"],
        reverse=True
    )

    for child_name, child_analysis in children:
        print_parameter_tree(child_analysis, child_name, indent + 1, min_params)


def print_flat_parameter_list(model: nn.Module, min_params: int = 100):
    """Print flat list of all named parameters with counts."""
    print("\n" + "=" * 80)
    print("FLAT PARAMETER LIST (sorted by size)")
    print("=" * 80)

    param_list = []
    for name, param in model.named_parameters():
        param_list.append((name, param.numel(), param.shape))

    # Sort by size descending
    param_list.sort(key=lambda x: x[1], reverse=True)

    for name, count, shape in param_list:
        if count >= min_params:
            print(f"{count:>12,}  {str(shape):>30}  {name}")


def print_module_summary(model: nn.Module):
    """Print summary of top-level modules."""
    print("\n" + "=" * 80)
    print("TOP-LEVEL MODULE SUMMARY")
    print("=" * 80)

    total = count_parameters(model)

    modules = []
    for name, child in model.named_children():
        params = count_parameters(child)
        pct = 100 * params / total if total > 0 else 0
        modules.append((name, params, pct))

    # Sort by params descending
    modules.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Module':<40} {'Params':>15} {'Percentage':>12}")
    print("-" * 70)
    for name, params, pct in modules:
        bar = "█" * int(pct / 2)
        print(f"{name:<40} {params:>15,} {pct:>10.1f}%  {bar}")
    print("-" * 70)
    print(f"{'TOTAL':<40} {total:>15,} {100.0:>10.1f}%")


def print_block_analysis(model: nn.Module):
    """Analyze parameter distribution within UNet blocks."""
    print("\n" + "=" * 80)
    print("UNET BLOCK ANALYSIS")
    print("=" * 80)

    if not hasattr(model, 'unet'):
        print("Model does not have a 'unet' attribute")
        return

    unet = model.unet
    total_unet = count_parameters(unet)

    sections = {
        "time_embedding": count_parameters(unet.time_embedding) + count_parameters(unet.time_transform),
        "init_conv": count_parameters(unet.init_conv),
        "down_blocks": sum(count_parameters(b) for b in unet.down_blocks),
        "middle": (count_parameters(unet.middle_res_block) +
                   count_parameters(unet.middle_attn_norm) +
                   count_parameters(unet.middle_attn_block) +
                   count_parameters(unet.middle_res_block2)),
        "up_blocks": sum(count_parameters(b) for b in unet.up_blocks),
        "final": count_parameters(unet.final_res_block) + count_parameters(unet.final_conv),
    }

    print(f"\n{'Section':<20} {'Params':>15} {'Percentage':>12}")
    print("-" * 50)
    for name, params in sections.items():
        pct = 100 * params / total_unet if total_unet > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"{name:<20} {params:>15,} {pct:>10.1f}%  {bar}")

    # Per-block breakdown
    print("\n" + "-" * 50)
    print("Individual blocks:")
    print("-" * 50)

    for i, block in enumerate(unet.down_blocks):
        params = count_parameters(block)
        pct = 100 * params / total_unet
        print(f"  down_block[{i}]: {params:>12,} ({pct:>5.1f}%)")

    for i, block in enumerate(unet.up_blocks):
        params = count_parameters(block)
        pct = 100 * params / total_unet
        print(f"  up_block[{i}]:   {params:>12,} ({pct:>5.1f}%)")


def analyze_attention_vs_conv(model: nn.Module):
    """Analyze ratio of attention parameters to convolution parameters."""
    print("\n" + "=" * 80)
    print("ATTENTION vs CONVOLUTION ANALYSIS")
    print("=" * 80)

    attn_params = 0
    conv_params = 0
    linear_params = 0
    norm_params = 0
    other_params = 0

    for name, module in model.named_modules():
        if isinstance(module, (nn.MultiheadAttention,)):
            attn_params += count_parameters(module)
        elif 'attn' in name.lower() or 'attention' in name.lower():
            # Custom attention modules
            for p in module.parameters(recurse=False):
                attn_params += p.numel()
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_params += count_parameters(module)
        elif isinstance(module, nn.Linear):
            linear_params += count_parameters(module)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            norm_params += count_parameters(module)

    # Recalculate to avoid double counting
    total = count_parameters(model)

    print(f"{'Type':<20} {'Params':>15}")
    print("-" * 40)
    print(f"{'Convolutions':<20} {conv_params:>15,}")
    print(f"{'Linear layers':<20} {linear_params:>15,}")
    print(f"{'Normalization':<20} {norm_params:>15,}")
    print(f"{'Total':<20} {total:>15,}")


def find_imbalances(model: nn.Module, threshold_ratio: float = 100.0):
    """Find parameter imbalances (modules with very different param counts at same level)."""
    print("\n" + "=" * 80)
    print(f"IMBALANCE DETECTION (ratio > {threshold_ratio}x)")
    print("=" * 80)

    imbalances = []

    def check_siblings(parent_name: str, parent: nn.Module):
        children = list(parent.named_children())
        if len(children) < 2:
            return

        param_counts = [(name, count_parameters(child)) for name, child in children]
        param_counts = [(n, c) for n, c in param_counts if c > 0]  # Filter zero-param modules

        if len(param_counts) < 2:
            return

        max_params = max(c for _, c in param_counts)
        min_params = min(c for _, c in param_counts)

        if min_params > 0 and max_params / min_params > threshold_ratio:
            imbalances.append({
                "parent": parent_name,
                "max": max(param_counts, key=lambda x: x[1]),
                "min": min(param_counts, key=lambda x: x[1]),
                "ratio": max_params / min_params
            })

        for name, child in children:
            child_name = f"{parent_name}.{name}" if parent_name else name
            check_siblings(child_name, child)

    check_siblings("", model)

    if imbalances:
        for imb in sorted(imbalances, key=lambda x: x["ratio"], reverse=True):
            print(f"\nIn {imb['parent'] or 'root'}:")
            print(f"  Largest:  {imb['max'][0]} = {imb['max'][1]:,} params")
            print(f"  Smallest: {imb['min'][0]} = {imb['min'][1]:,} params")
            print(f"  Ratio: {imb['ratio']:.1f}x")
    else:
        print("No significant imbalances detected.")


def recommend_balanced_config(target_params: int, latent_channels: int = 4) -> dict:
    """
    Recommend a balanced configuration for a target parameter count.

    The key insight is that for a UNet:
    - Down/up blocks dominate parameters (should be ~70-80% of UNet)
    - Middle block should be ~10-15%
    - Init/final convs should be small (~5%)
    - Time embedding should be modest (~5%)

    Within blocks:
    - ResidualBlocks are the main parameter sinks
    - Attention (if used) adds significant params
    - Deeper networks (more blocks) often better than wider (more channels)
    """
    print("\n" + "=" * 80)
    print(f"RECOMMENDED CONFIGURATION FOR ~{target_params:,} PARAMETERS")
    print("=" * 80)

    # Rough parameter estimation formulas for our UNet:
    # ResidualBlock(in_c, out_c) ≈ 2 * (9 * in_c * out_c + 9 * out_c * out_c) + norms + biases
    #                           ≈ 18 * in_c * out_c + 18 * out_c^2
    # DownBlock with 2 res_blocks: ~2 * ResidualBlock + downsample conv
    # Time embedding: model_channels + 2 * time_emb_dim^2

    configs = []

    # Try different base channel counts
    for model_channels in [64, 96, 128, 192, 256]:
        for multipliers in [[1, 2], [1, 2, 4], [2, 4], [1, 2, 2], [2, 2, 4]]:
            for num_res_blocks in [1, 2, 3]:
                for time_emb_dim in [128, 256, 512]:
                    # Estimate parameters
                    channels = [model_channels] + [model_channels * m for m in multipliers]

                    # Init conv: 3x3 conv, latent_channels -> model_channels
                    init_params = 9 * latent_channels * model_channels + model_channels

                    # Time embedding
                    time_params = model_channels + 2 * time_emb_dim * time_emb_dim + 2 * time_emb_dim

                    # Down blocks
                    down_params = 0
                    for i in range(len(multipliers)):
                        in_c, out_c = channels[i], channels[i + 1]
                        # 2 blocks per res_block, each has 2 convs
                        res_params = num_res_blocks * (18 * in_c * out_c + 18 * out_c * out_c)
                        # Time MLP in each res block
                        res_params += num_res_blocks * (time_emb_dim * out_c + out_c)
                        # Downsample conv
                        down_conv = 9 * out_c * out_c + out_c
                        down_params += res_params + down_conv

                    # Middle blocks (2 res blocks + attention)
                    mid_c = channels[-1]
                    mid_params = 2 * (18 * mid_c * mid_c + 18 * mid_c * mid_c)
                    mid_params += 2 * (time_emb_dim * mid_c + mid_c)
                    # Self-attention (rough estimate)
                    mid_params += 4 * mid_c * mid_c  # Q, K, V, O projections

                    # Up blocks (similar to down but with skip connections)
                    up_params = 0
                    for i in range(len(multipliers)):
                        idx = len(multipliers) - 1 - i
                        in_c, out_c = channels[idx + 1], channels[idx]
                        # First res_block takes concatenated input (in_c * 2)
                        res_params = 18 * (in_c * 2) * out_c + 18 * out_c * out_c
                        res_params += (num_res_blocks - 1) * (18 * out_c * out_c + 18 * out_c * out_c)
                        res_params += num_res_blocks * (time_emb_dim * out_c + out_c)
                        # Upsample conv
                        up_conv = 9 * in_c * in_c + in_c
                        up_params += res_params + up_conv

                    # Final blocks
                    final_params = 18 * (model_channels * 2) * model_channels + 18 * model_channels * model_channels
                    final_params += 9 * model_channels * latent_channels + latent_channels

                    total = init_params + time_params + down_params + mid_params + up_params + final_params

                    configs.append({
                        "model_channels": model_channels,
                        "channel_multipliers": multipliers,
                        "num_res_blocks": num_res_blocks,
                        "time_embedding_dim": time_emb_dim,
                        "estimated_params": total,
                        "error": abs(total - target_params) / target_params
                    })

    # Sort by closest to target
    configs.sort(key=lambda x: x["error"])

    print("\nTop 5 closest configurations:")
    print("-" * 80)
    for i, cfg in enumerate(configs[:5]):
        print(f"\n{i+1}. Estimated: {cfg['estimated_params']:,} params ({cfg['error']*100:.1f}% off target)")
        print(f"   model_channels: {cfg['model_channels']}")
        print(f"   channel_multipliers: {cfg['channel_multipliers']}")
        print(f"   num_res_blocks: {cfg['num_res_blocks']}")
        print(f"   time_embedding_dim: {cfg['time_embedding_dim']}")

    return configs[0]


def create_model(
    model_channels: int = 128,
    channel_multipliers: list = [1, 2, 4],
    num_res_blocks: int = 2,
    time_embedding_dim: int = 256,
    latent_channels: int = 4,
    context_dim: int = 512,
    attention_levels: list = None,
    num_timesteps: int = 1000,
    dropout_p: float = 0.1,
    self_attn_n_heads: int = 4,
    self_attn_d_queries: int = 64,
    self_attn_d_values: int = 64,
    cross_attn_n_heads: int = 4,
    cross_attn_d_queries: int = 64,
    cross_attn_d_values: int = 64,
):
    """Create a diffusion model with specified configuration."""
    from model.image.diffusion import create_image_diffusion_model
    from megatransformer_utils import MegaTransformerConfig

    # Default attention at deeper levels
    if attention_levels is None:
        attention_levels = [False] * (len(channel_multipliers) - 1) + [True]

    # Ensure attention_levels matches channel_multipliers length
    while len(attention_levels) < len(channel_multipliers):
        attention_levels.append(True)
    attention_levels = attention_levels[:len(channel_multipliers)]

    config = MegaTransformerConfig(
        image_size=32,
        image_decoder_model_channels=model_channels,
        image_decoder_time_embedding_dim=time_embedding_dim,
        image_decoder_num_res_blocks=num_res_blocks,
        image_decoder_channel_multipliers=channel_multipliers,
        image_decoder_unet_dropout_p=dropout_p,
        image_decoder_down_block_self_attn_n_heads=self_attn_n_heads,
        image_decoder_down_block_self_attn_d_queries=self_attn_d_queries,
        image_decoder_down_block_self_attn_d_values=self_attn_d_values,
        image_decoder_down_block_self_attn_use_flash_attention=True,
        image_decoder_up_block_self_attn_n_heads=self_attn_n_heads,
        image_decoder_up_block_self_attn_d_queries=self_attn_d_queries,
        image_decoder_up_block_self_attn_d_values=self_attn_d_values,
        image_decoder_up_block_self_attn_use_flash_attention=True,
        image_decoder_cross_attn_n_heads=cross_attn_n_heads,
        image_decoder_cross_attn_d_queries=cross_attn_d_queries,
        image_decoder_cross_attn_d_values=cross_attn_d_values,
        image_decoder_cross_attn_use_flash_attention=True,
    )

    model = create_image_diffusion_model(
        config=config,
        latent_channels=latent_channels,
        num_timesteps=num_timesteps,
        sampling_timesteps=20,
        betas_schedule="cosine",
        context_dim=context_dim,
        normalize=False,  # For latent diffusion
        min_snr_loss_weight=False,
        prediction_type="epsilon",
        cfg_dropout_prob=0.0,
        zero_terminal_snr=False,
        offset_noise_strength=0.0,
        timestep_sampling="uniform",
    )

    return model


def main():
    parser = argparse.ArgumentParser(description="Analyze diffusion model architecture")

    # Model configuration
    parser.add_argument("--model_channels", type=int, default=128,
                        help="Base channel count for UNet")
    parser.add_argument("--channel_multipliers", type=str, default="1,2,4",
                        help="Channel multipliers for each level (comma-separated)")
    parser.add_argument("--num_res_blocks", type=int, default=2,
                        help="Number of residual blocks per level")
    parser.add_argument("--time_embedding_dim", type=int, default=256,
                        help="Dimension of time embedding")
    parser.add_argument("--latent_channels", type=int, default=4,
                        help="Number of latent channels (from VAE)")
    parser.add_argument("--context_dim", type=int, default=512,
                        help="Context dimension for conditioning")
    parser.add_argument("--attention_levels", type=str, default=None,
                        help="Which levels have attention (comma-separated bools, e.g., '0,0,1')")

    # Attention configuration
    parser.add_argument("--self_attn_n_heads", type=int, default=4,
                        help="Number of self-attention heads")
    parser.add_argument("--self_attn_d_queries", type=int, default=64,
                        help="Self-attention query dimension")
    parser.add_argument("--self_attn_d_values", type=int, default=64,
                        help="Self-attention value dimension")
    parser.add_argument("--cross_attn_n_heads", type=int, default=4,
                        help="Number of cross-attention heads")
    parser.add_argument("--cross_attn_d_queries", type=int, default=64,
                        help="Cross-attention query dimension")
    parser.add_argument("--cross_attn_d_values", type=int, default=64,
                        help="Cross-attention value dimension")

    # Presets
    parser.add_argument("--preset", type=str, default=None,
                        choices=["tiny", "small", "medium", "large"],
                        help="Use a preset configuration")

    # Auto-configuration
    parser.add_argument("--target_params", type=int, default=None,
                        help="Target parameter count for auto-configuration")

    # Analysis options
    parser.add_argument("--min_params", type=int, default=100,
                        help="Minimum params to show in tree view")
    parser.add_argument("--show_flat", action="store_true",
                        help="Show flat parameter list")
    parser.add_argument("--show_tree", action="store_true",
                        help="Show parameter tree")
    parser.add_argument("--imbalance_threshold", type=float, default=50.0,
                        help="Ratio threshold for imbalance detection")

    args = parser.parse_args()

    # Handle presets
    if args.preset:
        presets = {
            "tiny": {"model_channels": 64, "channel_multipliers": [2, 4],
                     "num_res_blocks": 2, "time_embedding_dim": 64,
                     "self_attn_n_heads": 2, "self_attn_d": 16,
                     "cross_attn_n_heads": 2, "cross_attn_d": 16},
            "small": {"model_channels": 128, "channel_multipliers": [1, 2, 4],
                      "num_res_blocks": 2, "time_embedding_dim": 256,
                      "self_attn_n_heads": 4, "self_attn_d": 64,
                      "cross_attn_n_heads": 4, "cross_attn_d": 64},
            "medium": {"model_channels": 192, "channel_multipliers": [1, 2, 4, 4],
                       "num_res_blocks": 2, "time_embedding_dim": 512,
                       "self_attn_n_heads": 8, "self_attn_d": 64,
                       "cross_attn_n_heads": 8, "cross_attn_d": 64},
            "large": {"model_channels": 256, "channel_multipliers": [1, 2, 4, 8],
                      "num_res_blocks": 3, "time_embedding_dim": 512,
                      "self_attn_n_heads": 8, "self_attn_d": 64,
                      "cross_attn_n_heads": 8, "cross_attn_d": 64},
        }
        preset = presets[args.preset]
        args.model_channels = preset["model_channels"]
        args.channel_multipliers = ",".join(map(str, preset["channel_multipliers"]))
        args.num_res_blocks = preset["num_res_blocks"]
        args.time_embedding_dim = preset["time_embedding_dim"]
        args.self_attn_n_heads = preset["self_attn_n_heads"]
        args.self_attn_d_queries = preset["self_attn_d"]
        args.self_attn_d_values = preset["self_attn_d"]
        args.cross_attn_n_heads = preset["cross_attn_n_heads"]
        args.cross_attn_d_queries = preset["cross_attn_d"]
        args.cross_attn_d_values = preset["cross_attn_d"]

    # Auto-configure if target_params specified
    if args.target_params:
        recommended = recommend_balanced_config(args.target_params, args.latent_channels)

        print("\nWould you like to use the recommended configuration? (Creating model with it...)")
        args.model_channels = recommended["model_channels"]
        args.channel_multipliers = ",".join(map(str, recommended["channel_multipliers"]))
        args.num_res_blocks = recommended["num_res_blocks"]
        args.time_embedding_dim = recommended["time_embedding_dim"]

    # Parse channel multipliers
    channel_multipliers = [int(x) for x in args.channel_multipliers.split(",")]

    # Parse attention levels
    attention_levels = None
    if args.attention_levels:
        attention_levels = [x.lower() in ("1", "true", "yes") for x in args.attention_levels.split(",")]

    print("\n" + "=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"model_channels: {args.model_channels}")
    print(f"channel_multipliers: {channel_multipliers}")
    print(f"num_res_blocks: {args.num_res_blocks}")
    print(f"time_embedding_dim: {args.time_embedding_dim}")
    print(f"latent_channels: {args.latent_channels}")
    print(f"context_dim: {args.context_dim}")
    print(f"attention_levels: {attention_levels or 'auto'}")

    # Create model
    model = create_model(
        model_channels=args.model_channels,
        channel_multipliers=channel_multipliers,
        num_res_blocks=args.num_res_blocks,
        time_embedding_dim=args.time_embedding_dim,
        latent_channels=args.latent_channels,
        context_dim=args.context_dim,
        attention_levels=attention_levels,
        self_attn_n_heads=args.self_attn_n_heads,
        self_attn_d_queries=args.self_attn_d_queries,
        self_attn_d_values=args.self_attn_d_values,
        cross_attn_n_heads=args.cross_attn_n_heads,
        cross_attn_d_queries=args.cross_attn_d_queries,
        cross_attn_d_values=args.cross_attn_d_values,
    )

    total_params = count_parameters(model)
    print(f"\nTotal parameters: {total_params:,}")

    # Print model structure
    print("\n" + "=" * 80)
    print("MODEL STRUCTURE")
    print("=" * 80)
    print(model)

    # Analysis
    print_module_summary(model)
    print_block_analysis(model)
    analyze_attention_vs_conv(model)
    find_imbalances(model, args.imbalance_threshold)

    if args.show_tree:
        print("\n" + "=" * 80)
        print("PARAMETER TREE")
        print("=" * 80)
        analysis = analyze_parameter_distribution(model)
        print_parameter_tree(analysis, "model", min_params=args.min_params)

    if args.show_flat:
        print_flat_parameter_list(model, args.min_params)

    # Final recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    unet_params = count_parameters(model.unet) if hasattr(model, 'unet') else 0

    if unet_params > 0:
        # Check for common issues
        down_params = sum(count_parameters(b) for b in model.unet.down_blocks)
        up_params = sum(count_parameters(b) for b in model.unet.up_blocks)

        down_up_ratio = down_params / up_params if up_params > 0 else float('inf')

        if abs(down_up_ratio - 1.0) > 0.3:
            print(f"⚠ Down/Up block ratio is {down_up_ratio:.2f}x - consider balancing")
        else:
            print(f"✓ Down/Up block ratio is balanced ({down_up_ratio:.2f}x)")

        time_params = count_parameters(model.unet.time_embedding) + count_parameters(model.unet.time_transform)
        time_pct = 100 * time_params / unet_params

        if time_pct > 15:
            print(f"⚠ Time embedding is {time_pct:.1f}% of UNet - consider reducing time_embedding_dim")
        else:
            print(f"✓ Time embedding is reasonable ({time_pct:.1f}% of UNet)")

        init_final_params = count_parameters(model.unet.init_conv) + count_parameters(model.unet.final_conv)
        init_final_pct = 100 * init_final_params / unet_params

        if init_final_pct > 10:
            print(f"⚠ Init/Final convs are {init_final_pct:.1f}% of UNet - this is high")
        else:
            print(f"✓ Init/Final convs are reasonable ({init_final_pct:.1f}% of UNet)")

    print(f"\nFor ~{total_params:,} parameters, this configuration looks {'reasonable' if total_params > 5_000_000 else 'small'}.")

    if total_params < 10_000_000:
        print("\nNote: For latent diffusion on 32x32 latents, 20-50M parameters is typically recommended.")
        print("Consider increasing model_channels or adding more levels.")


if __name__ == "__main__":
    main()
