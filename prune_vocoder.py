"""
Structured pruning utility for frequency-domain vocoders.

Prunes hidden_dim channels from a trained checkpoint and creates a new smaller model.
Supports gradient-based importance scoring via a calibration dataset.

Usage:
    # Basic pruning (magnitude-based importance)
    python prune_vocoder.py \
        --checkpoint runs/my_vocoder/checkpoint_100000.pt \
        --output_dir runs/my_vocoder_pruned \
        --target_hidden_dim 96

    # Pruning with gradient-based importance (better quality)
    python prune_vocoder.py \
        --checkpoint runs/my_vocoder/checkpoint_100000.pt \
        --output_dir runs/my_vocoder_pruned \
        --target_hidden_dim 96 \
        --calibration_data cached_datasets/librispeech_val_vocoder_cached \
        --calibration_samples 100

    # Layer-wise pruning (different ratios per layer)
    python prune_vocoder.py \
        --checkpoint runs/my_vocoder/checkpoint_100000.pt \
        --output_dir runs/my_vocoder_pruned \
        --prune_ratio 0.25 \
        --importance_method taylor
"""

import argparse
import json
import os
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.audio.shared_window_buffer import SharedWindowBuffer
from model.audio.vocoders.freq_domain_vocoder import (
    ConvNeXtBlock,
    LightHeadedFrequencyDomainVocoder,
    SplitBandFrequencyDomainVocoder,
)
from model.audio.vocoders.vocoders import VocoderWithLoss, model_config_lookup
from pretrain_vocoder import CachedVocoderDataset, collate_fn


def compute_magnitude_importance(model: nn.Module) -> dict[str, torch.Tensor]:
    """
    Compute channel importance based on weight magnitude.
    Simple but effective baseline.
    """
    importance = {}

    # Input projection: importance = sum of abs output weights
    # Shape: [hidden_dim, n_mels, kernel]
    importance['input_proj'] = model.input_proj.weight.abs().sum(dim=(1, 2))

    # Backbone ConvNeXt blocks
    for i, block in enumerate(model.backbone):
        # Depthwise conv: each channel is independent
        # Shape: [dim, 1, kernel] for groups=dim
        dw_importance = block.dwconv.weight.abs().sum(dim=(1, 2))

        # Pointwise convs (Linear layers)
        # pwconv1: [dim * expansion, dim] - sum over output dim
        pw1_importance = block.pwconv1.weight.abs().sum(dim=0)

        # pwconv2: [dim, dim * expansion] - sum over input dim
        pw2_importance = block.pwconv2.weight.abs().sum(dim=1)

        # Combine: geometric mean of all pathways
        if block.ovr_out_dim is None:
            # Regular block - same input/output dim
            combined = (dw_importance * pw1_importance * pw2_importance) ** (1/3)
        else:
            # Dimension-changing block - handle separately
            combined = dw_importance * pw1_importance
            importance[f'backbone.{i}.out'] = block.pwconv2.weight.abs().sum(dim=1)

        importance[f'backbone.{i}'] = combined

    return importance


def compute_gradient_importance(
    model: VocoderWithLoss,
    dataloader: DataLoader,
    num_samples: int = 100,
    device: torch.device = torch.device('cuda'),
) -> dict[str, torch.Tensor]:
    """
    Compute channel importance based on gradient * weight (Taylor approximation).
    More accurate than magnitude-based but requires calibration data.
    """
    model = model.to(device)
    model.train()

    # Accumulate importance scores
    importance_accum = {}

    vocoder = model.vocoder

    # Initialize accumulators
    importance_accum['input_proj'] = torch.zeros(
        vocoder.input_proj.weight.shape[0], device=device
    )

    for i, block in enumerate(vocoder.backbone):
        dim = block.dwconv.weight.shape[0]
        importance_accum[f'backbone.{i}'] = torch.zeros(dim, device=device)
        if block.ovr_out_dim is not None:
            importance_accum[f'backbone.{i}.out'] = torch.zeros(
                block.ovr_out_dim, device=device
            )

    samples_processed = 0

    for batch in tqdm(dataloader, desc="Computing gradient importance", total=num_samples):
        if samples_processed >= num_samples:
            break

        mel_spec = batch['mel_spec'].to(device)
        waveform_labels = batch['waveform_labels'].to(device)
        target_stfts = batch.get('target_complex_stfts')
        if target_stfts is not None:
            target_stfts = target_stfts.to(device)

        model.zero_grad()

        outputs = model(mel_spec, waveform_labels, target_stfts)
        loss = outputs['loss']
        loss.backward()

        # Accumulate Taylor importance: |weight * grad|
        with torch.no_grad():
            # Input projection
            importance_accum['input_proj'] += (
                vocoder.input_proj.weight * vocoder.input_proj.weight.grad
            ).abs().sum(dim=(1, 2))

            # Backbone blocks
            for i, block in enumerate(vocoder.backbone):
                # Depthwise conv
                dw_imp = (
                    block.dwconv.weight * block.dwconv.weight.grad
                ).abs().sum(dim=(1, 2))

                # Pointwise convs
                pw1_imp = (
                    block.pwconv1.weight * block.pwconv1.weight.grad
                ).abs().sum(dim=0)

                pw2_imp = (
                    block.pwconv2.weight * block.pwconv2.weight.grad
                ).abs().sum(dim=1)

                if block.ovr_out_dim is None:
                    importance_accum[f'backbone.{i}'] += (dw_imp * pw1_imp * pw2_imp) ** (1/3)
                else:
                    importance_accum[f'backbone.{i}'] += dw_imp * pw1_imp
                    importance_accum[f'backbone.{i}.out'] += pw2_imp

        samples_processed += mel_spec.shape[0]

    # Normalize by number of samples
    for key in importance_accum:
        importance_accum[key] /= samples_processed

    return importance_accum


def get_channels_to_keep(
    importance: torch.Tensor,
    target_dim: int,
) -> torch.Tensor:
    """Get indices of top-k most important channels."""
    _, indices = importance.sort(descending=True)
    keep_indices = indices[:target_dim].sort().values
    return keep_indices


def prune_conv1d(
    old_conv: nn.Conv1d,
    in_indices: torch.Tensor | None,
    out_indices: torch.Tensor | None,
) -> nn.Conv1d:
    """Create a pruned Conv1d layer."""
    old_weight = old_conv.weight.data  # [out, in, kernel]
    old_bias = old_conv.bias.data if old_conv.bias is not None else None

    # Apply index selection
    new_weight = old_weight
    if out_indices is not None:
        new_weight = new_weight[out_indices]
    if in_indices is not None:
        # For depthwise conv (groups=in_channels), don't slice input dim
        if old_conv.groups == 1:
            new_weight = new_weight[:, in_indices]

    new_out = new_weight.shape[0]
    new_in = new_weight.shape[1]

    # Handle depthwise conv
    if old_conv.groups > 1:
        new_groups = new_out  # groups = out_channels for depthwise
        new_in = 1
    else:
        new_groups = 1

    new_conv = nn.Conv1d(
        in_channels=new_in * new_groups,
        out_channels=new_out,
        kernel_size=old_conv.kernel_size[0],
        padding=old_conv.padding[0],
        groups=new_groups,
        bias=old_bias is not None,
    )

    new_conv.weight.data = new_weight
    if old_bias is not None:
        new_bias = old_bias[out_indices] if out_indices is not None else old_bias
        new_conv.bias.data = new_bias

    return new_conv


def prune_linear(
    old_linear: nn.Linear,
    in_indices: torch.Tensor | None,
    out_indices: torch.Tensor | None,
) -> nn.Linear:
    """Create a pruned Linear layer."""
    old_weight = old_linear.weight.data  # [out, in]
    old_bias = old_linear.bias.data if old_linear.bias is not None else None

    new_weight = old_weight
    if out_indices is not None:
        new_weight = new_weight[out_indices]
    if in_indices is not None:
        new_weight = new_weight[:, in_indices]

    new_linear = nn.Linear(
        in_features=new_weight.shape[1],
        out_features=new_weight.shape[0],
        bias=old_bias is not None,
    )

    new_linear.weight.data = new_weight
    if old_bias is not None:
        new_bias = old_bias[out_indices] if out_indices is not None else old_bias
        new_linear.bias.data = new_bias

    return new_linear


def prune_layernorm(
    old_norm: nn.LayerNorm,
    indices: torch.Tensor,
) -> nn.LayerNorm:
    """Create a pruned LayerNorm."""
    new_dim = len(indices)
    new_norm = nn.LayerNorm(new_dim, elementwise_affine=old_norm.elementwise_affine)

    if old_norm.elementwise_affine:
        new_norm.weight.data = old_norm.weight.data[indices]
        new_norm.bias.data = old_norm.bias.data[indices]

    return new_norm


def prune_convnext_block(
    old_block: ConvNeXtBlock,
    in_indices: torch.Tensor,
    out_indices: torch.Tensor | None = None,
    expansion: int = 4,
) -> ConvNeXtBlock:
    """
    Prune a ConvNeXtBlock.

    Args:
        old_block: Original block
        in_indices: Indices of input channels to keep
        out_indices: Indices of output channels (None = same as input for regular blocks)
        expansion: FFN expansion factor
    """
    in_dim = len(in_indices)
    out_dim = len(out_indices) if out_indices is not None else in_dim

    # Create new block
    new_block = ConvNeXtBlock(
        dim=in_dim,
        ovr_out_dim=out_dim if out_indices is not None else None,
        kernel_size=old_block.dwconv.kernel_size[0],
        expansion=expansion,
    )

    # Prune depthwise conv (groups=dim, so in/out indices are the same)
    new_block.dwconv = prune_conv1d(old_block.dwconv, in_indices, in_indices)

    # Prune LayerNorm
    new_block.norm = prune_layernorm(old_block.norm, in_indices)

    # Prune pointwise convs (Linear layers)
    # pwconv1: [dim, dim * expansion] -> [in_dim, in_dim * expansion]
    # We need to figure out which expanded channels to keep
    # For simplicity, keep channels corresponding to kept input channels
    expanded_in_indices = torch.arange(in_dim * expansion, device=in_indices.device)

    # Actually we need to slice the input dim only, output is new
    new_block.pwconv1 = prune_linear(old_block.pwconv1, in_indices, None)
    # Resize to correct expansion
    new_pwconv1 = nn.Linear(in_dim, in_dim * expansion)
    new_pwconv1.weight.data = old_block.pwconv1.weight.data[:, in_indices]
    if old_block.pwconv1.bias is not None:
        new_pwconv1.bias.data = old_block.pwconv1.bias.data
    new_block.pwconv1 = new_pwconv1

    # pwconv2: [dim * expansion, dim] or [dim * expansion, ovr_out_dim]
    if out_indices is not None:
        new_pwconv2 = nn.Linear(in_dim * expansion, out_dim)
        new_pwconv2.weight.data = old_block.pwconv2.weight.data[out_indices]
        if old_block.pwconv2.bias is not None:
            new_pwconv2.bias.data = old_block.pwconv2.bias.data[out_indices]
    else:
        new_pwconv2 = nn.Linear(in_dim * expansion, in_dim)
        new_pwconv2.weight.data = old_block.pwconv2.weight.data[in_indices]
        if old_block.pwconv2.bias is not None:
            new_pwconv2.bias.data = old_block.pwconv2.bias.data[in_indices]
    new_block.pwconv2 = new_pwconv2

    return new_block


def prune_light_headed_vocoder(
    old_model: LightHeadedFrequencyDomainVocoder,
    importance: dict[str, torch.Tensor],
    target_hidden_dim: int,
    convnext_mult: int = 8,
) -> LightHeadedFrequencyDomainVocoder:
    """
    Prune a LightHeadedFrequencyDomainVocoder to target_hidden_dim.
    """
    # Get indices to keep for backbone (hidden_dim)
    # Average importance across all backbone layers for global pruning
    backbone_importance = torch.stack([
        importance[f'backbone.{i}'] for i in range(len(old_model.backbone) - 1)
    ]).mean(dim=0)

    backbone_indices = get_channels_to_keep(backbone_importance, target_hidden_dim)

    # For the last block, we also need head indices (hidden_dim // 2)
    target_head_dim = target_hidden_dim // 2
    last_block_key = f'backbone.{len(old_model.backbone) - 1}.out'
    if last_block_key in importance:
        head_indices = get_channels_to_keep(importance[last_block_key], target_head_dim)
    else:
        # Fallback: just take first target_head_dim indices
        head_indices = torch.arange(target_head_dim)

    # Create new model with target dimensions
    shared_buffer = SharedWindowBuffer()
    new_model = LightHeadedFrequencyDomainVocoder(
        shared_window_buffer=shared_buffer,
        n_mels=old_model.input_proj.in_channels,
        n_fft=old_model.n_fft,
        hop_length=old_model.hop_length,
        hidden_dim=target_hidden_dim,
        num_layers=len(old_model.backbone),
        convnext_mult=convnext_mult,
    )

    # Prune input projection
    new_model.input_proj = prune_conv1d(old_model.input_proj, None, backbone_indices)

    # Prune backbone blocks
    new_backbone = nn.ModuleList()
    for i, old_block in enumerate(old_model.backbone):
        if i < len(old_model.backbone) - 1:
            # Regular block
            new_block = prune_convnext_block(
                old_block, backbone_indices, None, expansion=convnext_mult
            )
        else:
            # Last block with dimension change
            new_block = prune_convnext_block(
                old_block, backbone_indices, head_indices, expansion=convnext_mult
            )
        new_backbone.append(new_block)
    new_model.backbone = new_backbone

    # Prune mag/phase heads (input from head_dim, output stays at freq_bins)
    new_model.mag_head = nn.Sequential(
        prune_conv1d(old_model.mag_head[0], head_indices, None)
    )
    new_model.phase_head = nn.Sequential(
        prune_conv1d(old_model.phase_head[0], head_indices, None)
    )

    return new_model


def prune_split_band_vocoder(
    old_model: SplitBandFrequencyDomainVocoder,
    importance: dict[str, torch.Tensor],
    target_hidden_dim: int,
    convnext_mult: int = 8,
) -> SplitBandFrequencyDomainVocoder:
    """
    Prune a SplitBandFrequencyDomainVocoder to target_hidden_dim.
    """
    # Same backbone pruning logic as LightHeaded
    backbone_importance = torch.stack([
        importance[f'backbone.{i}'] for i in range(len(old_model.backbone) - 1)
    ]).mean(dim=0)

    backbone_indices = get_channels_to_keep(backbone_importance, target_hidden_dim)

    target_head_dim = target_hidden_dim // 2
    last_block_key = f'backbone.{len(old_model.backbone) - 1}.out'
    if last_block_key in importance:
        head_indices = get_channels_to_keep(importance[last_block_key], target_head_dim)
    else:
        head_indices = torch.arange(target_head_dim)

    # Create new model
    shared_buffer = SharedWindowBuffer()
    new_model = SplitBandFrequencyDomainVocoder(
        shared_window_buffer=shared_buffer,
        n_mels=old_model.input_proj.in_channels,
        n_fft=old_model.n_fft,
        hop_length=old_model.hop_length,
        hidden_dim=target_hidden_dim,
        num_layers=len(old_model.backbone),
        convnext_mult=convnext_mult,
        cutoff_bin=old_model.cutoff_bin,
    )

    # Prune input projection
    new_model.input_proj = prune_conv1d(old_model.input_proj, None, backbone_indices)

    # Prune backbone blocks
    new_backbone = nn.ModuleList()
    for i, old_block in enumerate(old_model.backbone):
        if i < len(old_model.backbone) - 1:
            new_block = prune_convnext_block(
                old_block, backbone_indices, None, expansion=convnext_mult
            )
        else:
            new_block = prune_convnext_block(
                old_block, backbone_indices, head_indices, expansion=convnext_mult
            )
        new_backbone.append(new_block)
    new_model.backbone = new_backbone

    # Prune magnitude heads
    new_model.mag_head_low = prune_conv1d(old_model.mag_head_low, head_indices, None)
    new_model.mag_head_high = prune_conv1d(old_model.mag_head_high, head_indices, None)

    # Prune phase heads
    # Low-freq phase head is Sequential with Conv1d + SiLU
    new_phase_low_conv = prune_conv1d(old_model.phase_head_low[0], head_indices, None)
    new_model.phase_head_low = nn.Sequential(new_phase_low_conv, nn.SiLU())

    # High-freq phase: Snake + Conv
    # Snake has per-channel alpha, need to prune it
    from model.activations import Snake
    old_snake = old_model.phase_head_high_snake
    new_snake = Snake(target_head_dim)
    new_snake.alpha.data = old_snake.alpha.data[head_indices]
    new_model.phase_head_high_snake = new_snake

    new_model.phase_head_high = prune_conv1d(
        old_model.phase_head_high, head_indices, None
    )

    return new_model


def main():
    parser = argparse.ArgumentParser(description="Prune a frequency-domain vocoder")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained vocoder checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for pruned model")
    parser.add_argument("--target_hidden_dim", type=int, default=None,
                        help="Target hidden dimension (must be even)")
    parser.add_argument("--prune_ratio", type=float, default=None,
                        help="Fraction of channels to remove (alternative to target_hidden_dim)")
    parser.add_argument("--importance_method", type=str, default="magnitude",
                        choices=["magnitude", "taylor"],
                        help="Method for computing channel importance")
    parser.add_argument("--calibration_data", type=str, default=None,
                        help="Path to calibration dataset (required for taylor method)")
    parser.add_argument("--calibration_samples", type=int, default=100,
                        help="Number of samples for gradient-based importance")
    parser.add_argument("--vocoder_type", type=str, default="tiny_lightheaded_freq_domain_vocoder",
                        help="Vocoder type from model_config_lookup")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for importance computation")

    args = parser.parse_args()

    # Validate arguments
    if args.target_hidden_dim is None and args.prune_ratio is None:
        parser.error("Must specify either --target_hidden_dim or --prune_ratio")

    if args.importance_method == "taylor" and args.calibration_data is None:
        parser.error("--calibration_data required for taylor importance method")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Create model from config
    shared_buffer = SharedWindowBuffer()
    model_with_loss = model_config_lookup[args.vocoder_type](shared_buffer)
    model_with_loss.load_state_dict(checkpoint['model_state_dict'])

    vocoder = model_with_loss.vocoder
    old_hidden_dim = vocoder.input_proj.out_channels

    # Determine target dimension
    if args.target_hidden_dim is not None:
        target_hidden_dim = args.target_hidden_dim
    else:
        target_hidden_dim = int(old_hidden_dim * (1 - args.prune_ratio))
        # Round to even number
        target_hidden_dim = (target_hidden_dim // 2) * 2

    print(f"Pruning from hidden_dim={old_hidden_dim} to {target_hidden_dim}")
    print(f"Reduction: {100 * (1 - target_hidden_dim / old_hidden_dim):.1f}%")

    # Compute importance scores
    if args.importance_method == "magnitude":
        print("Computing magnitude-based importance...")
        importance = compute_magnitude_importance(vocoder)
    else:
        print("Computing gradient-based importance...")
        # Load calibration data
        dataset = CachedVocoderDataset(args.calibration_data)
        dataloader = DataLoader(
            dataset, batch_size=8, shuffle=True,
            collate_fn=collate_fn, num_workers=2
        )
        importance = compute_gradient_importance(
            model_with_loss, dataloader,
            num_samples=args.calibration_samples, device=device
        )

    # Move importance to CPU for indexing
    importance = {k: v.cpu() for k, v in importance.items()}

    # Determine vocoder type and prune
    if isinstance(vocoder, SplitBandFrequencyDomainVocoder):
        print("Pruning SplitBandFrequencyDomainVocoder...")
        # Get convnext_mult from first backbone block
        convnext_mult = vocoder.backbone[0].pwconv1.out_features // vocoder.backbone[0].pwconv1.in_features
        new_vocoder = prune_split_band_vocoder(
            vocoder, importance, target_hidden_dim, convnext_mult
        )
    elif isinstance(vocoder, LightHeadedFrequencyDomainVocoder):
        print("Pruning LightHeadedFrequencyDomainVocoder...")
        convnext_mult = vocoder.backbone[0].pwconv1.out_features // vocoder.backbone[0].pwconv1.in_features
        new_vocoder = prune_light_headed_vocoder(
            vocoder, importance, target_hidden_dim, convnext_mult
        )
    else:
        raise ValueError(f"Unsupported vocoder type: {type(vocoder)}")

    # Count parameters
    old_params = sum(p.numel() for p in vocoder.parameters())
    new_params = sum(p.numel() for p in new_vocoder.parameters())
    print(f"Parameters: {old_params:,} -> {new_params:,} ({100 * new_params / old_params:.1f}%)")

    # Create new VocoderWithLoss wrapper
    new_model_with_loss = VocoderWithLoss(
        vocoder=new_vocoder,
        shared_window_buffer=shared_buffer,
        config=model_with_loss.config,
        sc_loss_weight=model_with_loss.sc_loss_weight,
        mag_loss_weight=model_with_loss.mag_loss_weight,
        waveform_l1_loss_weight=model_with_loss.waveform_l1_loss_weight,
        mel_recon_loss_weight=model_with_loss.mel_recon_loss_weight,
        mel_recon_loss_weight_linspace_max=model_with_loss.mel_recon_loss_weight_linspace_max,
        complex_stft_loss_weight=model_with_loss.complex_stft_loss_weight,
        phase_loss_weight=model_with_loss.phase_loss_weight,
        phase_ip_loss_weight=model_with_loss.phase_ip_loss_weight,
        phase_iaf_loss_weight=model_with_loss.phase_iaf_loss_weight,
        phase_gd_loss_weight=model_with_loss.phase_gd_loss_weight,
        high_freq_stft_loss_weight=model_with_loss.high_freq_stft_loss_weight,
        direct_mag_loss_weight=model_with_loss.direct_mag_loss_weight,
    )

    # Save pruned checkpoint
    output_path = os.path.join(args.output_dir, "pruned_checkpoint.pt")
    torch.save({
        'model_state_dict': new_model_with_loss.state_dict(),
        'pruning_config': {
            'original_checkpoint': args.checkpoint,
            'original_hidden_dim': old_hidden_dim,
            'target_hidden_dim': target_hidden_dim,
            'importance_method': args.importance_method,
            'original_params': old_params,
            'pruned_params': new_params,
        }
    }, output_path)
    print(f"Saved pruned checkpoint to {output_path}")

    # Save config for reference
    config_path = os.path.join(args.output_dir, "pruning_config.json")
    with open(config_path, 'w') as f:
        json.dump({
            'original_checkpoint': args.checkpoint,
            'vocoder_type': args.vocoder_type,
            'original_hidden_dim': old_hidden_dim,
            'target_hidden_dim': target_hidden_dim,
            'importance_method': args.importance_method,
            'original_params': old_params,
            'pruned_params': new_params,
            'param_reduction': f"{100 * (1 - new_params / old_params):.1f}%",
        }, f, indent=2)
    print(f"Saved config to {config_path}")

    print("\nPruning complete! Next steps:")
    print("1. Fine-tune the pruned model to recover quality:")
    print(f"   python pretrain_vocoder.py --checkpoint {output_path} --run_name vocoder_pruned_finetune ...")
    print("2. Compare quality before/after pruning")


if __name__ == "__main__":
    main()