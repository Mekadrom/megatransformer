#!/usr/bin/env python3
"""
GuBERT Layer Probe Analysis

Analyzes what information is encoded at each layer of a pretrained GuBERT model
by training linear probes for:
  - Phoneme classification (CTC labels)
  - Speaker classification (speaker ID)
  - Mel reconstruction (original mel spectrogram frames)

Usage:
    python scripts/analyze_gubert_layers.py \
        --checkpoint runs/gubert/my_run/checkpoint-STEP \
        --config tiny_deep \
        --data_dir cached_datasets/gubert_ctc_train \
        --probe_layers 4 8 12 \
        --num_epochs 5 \
        --batch_size 32

The script will:
1. Load the pretrained GuBERT with frozen weights
2. Train separate linear probes for each layer
3. Evaluate and report losses for each probe type at each layer
4. Save results to a JSON file and optionally plot

Lower phoneme loss = better phonetic encoding at that layer
Higher speaker loss = better speaker disentanglement (GRL working)
Higher mel loss = more abstraction away from raw acoustics
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.audio.gubert import GuBERTEncoder, GUBERT_CONFIGS, GuBERTConfig, CTCVocab
from pretrain_gubert import GuBERTShardedDataset, GuBERTDataCollator


class LayerProbe(nn.Module):
    """Simple linear probe for a single target type."""

    def __init__(self, d_model: int, output_dim: int, pooling: str = "none"):
        """
        Args:
            d_model: Input feature dimension
            output_dim: Output dimension (vocab_size, num_speakers, or n_mels)
            pooling: "none" for frame-level, "mean" for utterance-level
        """
        super().__init__()
        self.linear = nn.Linear(d_model, output_dim)
        self.pooling = pooling

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] features
            mask: [B, T] True for valid positions
        Returns:
            [B, T, output_dim] if pooling="none"
            [B, output_dim] if pooling="mean"
        """
        if self.pooling == "mean":
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
        return self.linear(x)


class ProbeSet(nn.Module):
    """Collection of probes for all layers and target types."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        num_speakers: int,
        n_mels: int,
        probe_layers: List[int],
    ):
        super().__init__()
        self.probe_layers = probe_layers

        # Create probes for each layer
        self.phoneme_probes = nn.ModuleDict({
            str(layer): LayerProbe(d_model, vocab_size, pooling="none")
            for layer in probe_layers
        })
        self.speaker_probes = nn.ModuleDict({
            str(layer): LayerProbe(d_model, num_speakers, pooling="mean")
            for layer in probe_layers
        })
        self.mel_probes = nn.ModuleDict({
            str(layer): LayerProbe(d_model, n_mels, pooling="none")
            for layer in probe_layers
        })

    def forward(
        self,
        layer_outputs: Dict[int, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run all probes on layer outputs.

        Args:
            layer_outputs: {layer_idx: [B, T, D]} hidden states
            mask: [B, T] True for valid positions

        Returns:
            Dict with keys like "phoneme_4", "speaker_8", "mel_12"
        """
        results = {}

        for layer in self.probe_layers:
            if layer not in layer_outputs:
                continue

            features = layer_outputs[layer]
            layer_key = str(layer)

            results[f"phoneme_{layer}"] = self.phoneme_probes[layer_key](features, mask)
            results[f"speaker_{layer}"] = self.speaker_probes[layer_key](features, mask)
            results[f"mel_{layer}"] = self.mel_probes[layer_key](features, mask)

        return results


def load_gubert_checkpoint(
    checkpoint_path: str,
    config_name: str,
    device: torch.device,
    **config_overrides,
) -> GuBERTEncoder:
    """Load GuBERT from checkpoint."""

    # Create model
    model = GuBERTEncoder.from_config(config_name, **config_overrides)

    # Load checkpoint
    checkpoint_dir = Path(checkpoint_path)
    if checkpoint_dir.is_dir():
        # HuggingFace-style checkpoint directory
        model_file = checkpoint_dir / "pytorch_model.bin"
        if not model_file.exists():
            model_file = checkpoint_dir / "model.safetensors"
        if not model_file.exists():
            # Try to find any .bin or .pt file
            model_files = list(checkpoint_dir.glob("*.bin")) + list(checkpoint_dir.glob("*.pt"))
            if model_files:
                model_file = model_files[0]
            else:
                raise FileNotFoundError(f"No model file found in {checkpoint_dir}")
    else:
        model_file = checkpoint_dir

    print(f"Loading weights from {model_file}")

    if str(model_file).endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(str(model_file))
    else:
        state_dict = torch.load(model_file, map_location=device, weights_only=True)

    # Handle potential key prefix issues
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Freeze encoder weights
    for param in model.parameters():
        param.requires_grad = False

    print(f"Loaded GuBERT with {model.get_num_params():,} parameters (frozen)")

    return model


def extract_layer_outputs(
    model: GuBERTEncoder,
    mel_specs: torch.Tensor,
    lengths: torch.Tensor,
    probe_layers: List[int],
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Extract hidden states from specified layers.

    Returns:
        layer_outputs: {layer_idx: [B, T, D]}
        feature_lengths: [B]
        padding_mask: [B, T] True for valid positions
    """
    with torch.no_grad():
        result = model(mel_specs, lengths=lengths, return_all_hiddens=True)

    all_hiddens = result["all_hiddens"]  # List of [B, T, D], length = num_layers + 1
    feature_lengths = result["feature_lengths"]

    # Create padding mask
    max_len = all_hiddens[-1].size(1)
    padding_mask = torch.arange(max_len, device=mel_specs.device).unsqueeze(0) < feature_lengths.unsqueeze(1)

    # Extract specified layers (1-indexed, so layer 1 is all_hiddens[1])
    layer_outputs = {}
    for layer in probe_layers:
        if layer <= len(all_hiddens) - 1:
            layer_outputs[layer] = all_hiddens[layer]

    return layer_outputs, feature_lengths, padding_mask


def compute_probe_losses(
    probe_outputs: Dict[str, torch.Tensor],
    targets: dict,
    feature_lengths: torch.Tensor,
    padding_mask: torch.Tensor,
    probe_layers: List[int],
    ctc_blank_idx: int = 0,
) -> Dict[str, float]:
    """Compute losses for all probes."""
    losses = {}

    for layer in probe_layers:
        # Phoneme loss (CTC)
        phoneme_logits = probe_outputs[f"phoneme_{layer}"]  # [B, T, vocab_size]
        log_probs = F.log_softmax(phoneme_logits, dim=-1).transpose(0, 1)  # [T, B, V]

        ctc_loss = F.ctc_loss(
            log_probs,
            targets["text_tokens"],
            feature_lengths,
            targets["text_lengths"],
            blank=ctc_blank_idx,
            reduction="mean",
            zero_infinity=True,
        )
        losses[f"phoneme_{layer}"] = ctc_loss.item()

        # Speaker loss (cross-entropy)
        speaker_logits = probe_outputs[f"speaker_{layer}"]  # [B, num_speakers]
        speaker_loss = F.cross_entropy(speaker_logits, targets["speaker_ids"])
        losses[f"speaker_{layer}"] = speaker_loss.item()

        # Mel loss (MSE per frame, masked)
        mel_pred = probe_outputs[f"mel_{layer}"]  # [B, T_feat, n_mels]

        # Need to downsample mel targets to match feature length
        mel_targets = targets["mel_specs"]  # [B, n_mels, T_mel]
        mel_targets = mel_targets.transpose(1, 2)  # [B, T_mel, n_mels]

        # Downsample to feature resolution (4x by default)
        if mel_targets.size(1) != mel_pred.size(1):
            mel_targets = F.interpolate(
                mel_targets.transpose(1, 2),
                size=mel_pred.size(1),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        # Masked MSE
        mask_expanded = padding_mask.unsqueeze(-1).float()
        mel_diff = (mel_pred - mel_targets) ** 2
        mel_loss = (mel_diff * mask_expanded).sum() / mask_expanded.sum() / mel_pred.size(-1)
        losses[f"mel_{layer}"] = mel_loss.item()

    return losses


def train_probes(
    model: GuBERTEncoder,
    probes: ProbeSet,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    probe_layers: List[int],
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
) -> Dict[str, List[float]]:
    """Train probes and track losses."""

    optimizer = torch.optim.AdamW(probes.parameters(), lr=learning_rate)

    history = {f"{probe}_{layer}": [] for probe in ["phoneme", "speaker", "mel"] for layer in probe_layers}

    for epoch in range(num_epochs):
        probes.train()
        epoch_losses = {k: 0.0 for k in history.keys()}
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            # Move to device
            mel_specs = batch["mel_specs"].to(device)
            mel_lengths = batch["mel_lengths"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            text_lengths = batch["text_lengths"].to(device)
            speaker_ids = batch["speaker_ids"].to(device)

            # Extract layer outputs (frozen encoder)
            layer_outputs, feature_lengths, padding_mask = extract_layer_outputs(
                model, mel_specs, mel_lengths, probe_layers
            )

            # Forward through probes
            probe_outputs = probes(layer_outputs, mask=padding_mask)

            # Compute losses
            targets = {
                "text_tokens": text_tokens,
                "text_lengths": text_lengths,
                "speaker_ids": speaker_ids,
                "mel_specs": mel_specs,
            }

            losses = compute_probe_losses(
                probe_outputs, targets, feature_lengths, padding_mask, probe_layers
            )

            # Total loss (sum all probe losses)
            total_loss = sum(
                F.ctc_loss(
                    F.log_softmax(probe_outputs[f"phoneme_{layer}"], dim=-1).transpose(0, 1),
                    text_tokens, feature_lengths, text_lengths,
                    blank=0, reduction="mean", zero_infinity=True
                ) +
                F.cross_entropy(probe_outputs[f"speaker_{layer}"], speaker_ids) +
                _masked_mse(probe_outputs[f"mel_{layer}"], mel_specs, padding_mask)
                for layer in probe_layers
            )

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Track losses
            for k, v in losses.items():
                epoch_losses[k] += v
            num_batches += 1

            # Update progress bar
            avg_phoneme = sum(losses[f"phoneme_{l}"] for l in probe_layers) / len(probe_layers)
            avg_speaker = sum(losses[f"speaker_{l}"] for l in probe_layers) / len(probe_layers)
            pbar.set_postfix({"phoneme": f"{avg_phoneme:.3f}", "speaker": f"{avg_speaker:.3f}"})

        # Average epoch losses
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
            history[k].append(epoch_losses[k])

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        for layer in probe_layers:
            print(f"  Layer {layer}: phoneme={epoch_losses[f'phoneme_{layer}']:.4f}, "
                  f"speaker={epoch_losses[f'speaker_{layer}']:.4f}, "
                  f"mel={epoch_losses[f'mel_{layer}']:.4f}")

    return history


def _masked_mse(pred: torch.Tensor, mel_specs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute masked MSE between prediction and mel spec."""
    mel_targets = mel_specs.transpose(1, 2)  # [B, T, n_mels]

    if mel_targets.size(1) != pred.size(1):
        mel_targets = F.interpolate(
            mel_targets.transpose(1, 2),
            size=pred.size(1),
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    mask_expanded = mask.unsqueeze(-1).float()
    mel_diff = (pred - mel_targets) ** 2
    return (mel_diff * mask_expanded).sum() / mask_expanded.sum() / pred.size(-1)


def evaluate_probes(
    model: GuBERTEncoder,
    probes: ProbeSet,
    data_loader: DataLoader,
    probe_layers: List[int],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate probes on a dataset."""
    probes.eval()

    total_losses = {f"{probe}_{layer}": 0.0 for probe in ["phoneme", "speaker", "mel"] for layer in probe_layers}
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            mel_specs = batch["mel_specs"].to(device)
            mel_lengths = batch["mel_lengths"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            text_lengths = batch["text_lengths"].to(device)
            speaker_ids = batch["speaker_ids"].to(device)

            layer_outputs, feature_lengths, padding_mask = extract_layer_outputs(
                model, mel_specs, mel_lengths, probe_layers
            )

            probe_outputs = probes(layer_outputs, mask=padding_mask)

            targets = {
                "text_tokens": text_tokens,
                "text_lengths": text_lengths,
                "speaker_ids": speaker_ids,
                "mel_specs": mel_specs,
            }

            losses = compute_probe_losses(
                probe_outputs, targets, feature_lengths, padding_mask, probe_layers
            )

            for k, v in losses.items():
                total_losses[k] += v
            num_batches += 1

    # Average
    for k in total_losses:
        total_losses[k] /= num_batches

    return total_losses


def print_analysis_report(
    final_losses: Dict[str, float],
    probe_layers: List[int],
):
    """Print a formatted analysis report."""
    print("\n" + "=" * 60)
    print("GuBERT Layer Probe Analysis Report")
    print("=" * 60)

    print("\nLoss by Layer and Probe Type:")
    print("-" * 60)
    print(f"{'Layer':<10} {'Phoneme (CTC)':<15} {'Speaker (CE)':<15} {'Mel (MSE)':<15}")
    print("-" * 60)

    for layer in probe_layers:
        phoneme = final_losses[f"phoneme_{layer}"]
        speaker = final_losses[f"speaker_{layer}"]
        mel = final_losses[f"mel_{layer}"]
        print(f"{layer:<10} {phoneme:<15.4f} {speaker:<15.4f} {mel:<15.4f}")

    print("-" * 60)

    # Analysis
    print("\nInterpretation:")

    # Find best layer for each probe type
    best_phoneme = min(probe_layers, key=lambda l: final_losses[f"phoneme_{l}"])
    worst_phoneme = max(probe_layers, key=lambda l: final_losses[f"phoneme_{l}"])

    best_speaker = max(probe_layers, key=lambda l: final_losses[f"speaker_{l}"])  # Higher = better disentanglement
    worst_speaker = min(probe_layers, key=lambda l: final_losses[f"speaker_{l}"])

    print(f"  - Best phonetic encoding: Layer {best_phoneme} (lowest CTC loss)")
    print(f"  - Best speaker disentanglement: Layer {best_speaker} (highest speaker loss)")
    print(f"  - Layer {worst_phoneme} has worst phonetic encoding")
    print(f"  - Layer {worst_speaker} retains most speaker info")

    # Trend analysis
    phoneme_trend = final_losses[f"phoneme_{probe_layers[-1]}"] - final_losses[f"phoneme_{probe_layers[0]}"]
    speaker_trend = final_losses[f"speaker_{probe_layers[-1]}"] - final_losses[f"speaker_{probe_layers[0]}"]
    mel_trend = final_losses[f"mel_{probe_layers[-1]}"] - final_losses[f"mel_{probe_layers[0]}"]

    print(f"\n  Trends (first -> last layer):")
    print(f"  - Phoneme loss: {'decreasing' if phoneme_trend < 0 else 'increasing'} ({phoneme_trend:+.4f})")
    print(f"  - Speaker loss: {'increasing' if speaker_trend > 0 else 'decreasing'} ({speaker_trend:+.4f})")
    print(f"  - Mel loss: {'increasing' if mel_trend > 0 else 'decreasing'} ({mel_trend:+.4f})")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze GuBERT layer representations with linear probes")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to GuBERT checkpoint")
    parser.add_argument("--config", type=str, default="tiny_deep", help="Model config name")

    # Data
    parser.add_argument("--data_dir", type=str, required=True, help="Path to preprocessed training data")
    parser.add_argument("--val_data_dir", type=str, default=None, help="Path to validation data (optional)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max training samples to use")

    # Probing
    parser.add_argument("--probe_layers", type=int, nargs="+", default=None,
                       help="Layers to probe (1-indexed). Default: evenly spaced")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of probe training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Probe learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")

    # Config overrides
    parser.add_argument("--num_speakers", type=int, default=None, help="Number of speakers")
    parser.add_argument("--use_rotary_embedding", type=str, default=None)
    parser.add_argument("--use_conformer_conv", type=str, default=None)
    parser.add_argument("--use_macaron", type=str, default=None)
    parser.add_argument("--activation", type=str, default=None)

    # Output
    parser.add_argument("--output_file", type=str, default=None, help="Save results to JSON")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Build config overrides
    config_overrides = {}
    if args.num_speakers is not None:
        config_overrides["num_speakers"] = args.num_speakers
    if args.use_rotary_embedding is not None:
        config_overrides["use_rotary_embedding"] = args.use_rotary_embedding.lower() in ("true", "1", "yes")
    if args.use_conformer_conv is not None:
        config_overrides["use_conformer_conv"] = args.use_conformer_conv.lower() in ("true", "1", "yes")
    if args.use_macaron is not None:
        config_overrides["use_macaron"] = args.use_macaron.lower() in ("true", "1", "yes")
    if args.activation is not None:
        config_overrides["activation"] = args.activation

    # Load model
    model = load_gubert_checkpoint(args.checkpoint, args.config, device, **config_overrides)

    # Determine probe layers
    num_layers = model.config.num_layers
    if args.probe_layers is None:
        # Default: evenly spaced
        if num_layers >= 9:
            probe_layers = [num_layers // 3, 2 * num_layers // 3, num_layers]
        elif num_layers >= 6:
            probe_layers = [num_layers // 2, num_layers]
        else:
            probe_layers = list(range(1, num_layers + 1))
    else:
        probe_layers = args.probe_layers

    print(f"Probing layers: {probe_layers}")

    # Create probes
    probes = ProbeSet(
        d_model=model.config.encoder_dim,
        vocab_size=model.config.vocab_size,
        num_speakers=model.config.num_speakers,
        n_mels=model.config.n_mels,
        probe_layers=probe_layers,
    ).to(device)

    probe_params = sum(p.numel() for p in probes.parameters())
    print(f"Probe parameters: {probe_params:,}")

    # Load data using pretrain_gubert's dataset and collator
    train_dataset = GuBERTShardedDataset(args.data_dir, max_samples=args.max_samples, mode="ctc")
    collator = GuBERTDataCollator(n_mels=model.config.n_mels, mode="ctc")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = None
    if args.val_data_dir:
        val_dataset = GuBERTShardedDataset(args.val_data_dir, mode="ctc")
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # Train probes
    print("\nTraining probes...")
    history = train_probes(
        model=model,
        probes=probes,
        train_loader=train_loader,
        val_loader=val_loader,
        probe_layers=probe_layers,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
    )

    # Final evaluation
    print("\nFinal evaluation...")
    final_losses = evaluate_probes(model, probes, train_loader, probe_layers, device)

    # Print report
    print_analysis_report(final_losses, probe_layers)

    # Save results
    if args.output_file:
        results = {
            "checkpoint": args.checkpoint,
            "config": args.config,
            "probe_layers": probe_layers,
            "final_losses": final_losses,
            "training_history": history,
        }
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
