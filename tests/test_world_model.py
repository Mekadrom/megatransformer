"""
Test script for the Megatransformer World Model.

This script tests model instantiation, prints architecture information,
and runs forward pass tests with various input configurations.

Usage:
    python tests/test_world_model.py
    python tests/test_world_model.py --config tiny
    python tests/test_world_model.py --config small --device cuda
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.world.world_model import (
    MegatransformerWorldModel,
    MegatransformerWorldModelConfig,
    get_wm_config,
    tiny_world_model_config,
    small_world_model_config,
    medium_world_model_config,
)
from utils.configuration import AudioConfig, ImageConfig


# =============================================================================
# Utility Functions
# =============================================================================

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_param_count(count: int) -> str:
    """Format parameter count with appropriate suffix (K, M, B)."""
    if count >= 1e9:
        return f"{count / 1e9:.2f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.2f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.2f}K"
    return str(count)


def print_model_summary(model: nn.Module, model_name: str = "Model"):
    """Print a summary of the model architecture and parameters."""
    print(f"\n{'=' * 60}")
    print(f"{model_name} Summary")
    print(f"{'=' * 60}")

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    print(f"Total parameters:     {format_param_count(total_params)} ({total_params:,})")
    print(f"Trainable parameters: {format_param_count(trainable_params)} ({trainable_params:,})")
    print(f"Non-trainable:        {format_param_count(total_params - trainable_params)}")

    print(f"\n{'─' * 60}")
    print("Parameter breakdown by component:")
    print(f"{'─' * 60}")

    for name, module in model.named_children():
        params = count_parameters(module, trainable_only=False)
        pct = 100 * params / total_params if total_params > 0 else 0
        print(f"  {name:35s} {format_param_count(params):>10s} ({pct:5.1f}%)")

    print(f"{'─' * 60}\n")


def print_architecture(model: nn.Module, max_depth: int = 2):
    """Print model architecture up to specified depth."""
    print(f"\n{'=' * 60}")
    print("Model Architecture")
    print(f"{'=' * 60}")

    def _print_module(module, prefix="", depth=0):
        if depth > max_depth:
            return

        for name, child in module.named_children():
            child_params = count_parameters(child, trainable_only=False)
            print(f"{prefix}{name}: {child.__class__.__name__} ({format_param_count(child_params)})")
            _print_module(child, prefix + "  ", depth + 1)

    _print_module(model)
    print()


# =============================================================================
# Test Functions
# =============================================================================

def test_model_instantiation(config: MegatransformerWorldModelConfig, device: str = "cpu"):
    """Test that the model can be instantiated without errors."""
    print("\n[TEST] Model Instantiation")
    print("-" * 40)

    try:
        model = MegatransformerWorldModel(config)
        model = model.to(device)
        print(f"✓ Model instantiated successfully on {device}")
        return model
    except Exception as e:
        print(f"✗ Failed to instantiate model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_text_only_forward(model: MegatransformerWorldModel, device: str = "cpu"):
    """Test forward pass with text-only input."""
    print("\n[TEST] Text-Only Forward Pass")
    print("-" * 40)

    batch_size = 2
    seq_len = 32
    vocab_size = model.config.text_feature_config.vocab_size

    try:
        # Create random text input (no placeholder tokens)
        text_input_ids = torch.randint(0, vocab_size - 10, (batch_size, seq_len), device=device)

        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(text_input_ids=text_input_ids)

        print(f"✓ Text-only forward pass successful")
        print(f"  Input shape:  {text_input_ids.shape}")
        print(f"  Output keys:  {list(outputs.keys())}")

        if "logits" in outputs:
            print(f"  Logits shape: {outputs['logits'].shape}")

        return True
    except Exception as e:
        print(f"✗ Text-only forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_with_audio_forward(model: MegatransformerWorldModel, device: str = "cpu"):
    """Test forward pass with text + audio input."""
    print("\n[TEST] Text + Audio Forward Pass")
    print("-" * 40)

    batch_size = 2
    text_seq_len = 32
    n_audio = 1
    vocab_size = model.config.text_feature_config.vocab_size

    # Get placeholder token ID
    audio_placeholder = model.config.token_interleaver_config.audio_placeholder_token_id

    # Get audio latent dimensions from config
    audio_config: AudioConfig = model.config.audio_prelude_config.audio_config
    latent_channels = audio_config.latent_channels
    latent_mel_bins = audio_config.n_mels // audio_config.latent_compression_factor[0]
    latent_timesteps = 16  # Arbitrary for testing

    try:
        # Create text input with one audio placeholder per batch item
        text_input_ids = torch.randint(0, vocab_size - 10, (batch_size, text_seq_len), device=device)
        # Insert audio placeholder at position 10
        text_input_ids[:, 10] = audio_placeholder

        # Create audio latents: (batch, n_audio, latent_channels, latent_mel_bins, timesteps)
        audio_inputs = torch.randn(
            batch_size, n_audio, latent_channels, latent_mel_bins, latent_timesteps,
            device=device
        )

        # Audio lengths (in latent timesteps)
        audio_lengths = torch.full((batch_size, n_audio), latent_timesteps, device=device)

        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(
                text_input_ids=text_input_ids,
                audio_inputs=audio_inputs,
                audio_lengths=audio_lengths,
                precomputed_latents=True,
            )

        print(f"✓ Text + Audio forward pass successful")
        print(f"  Text shape:   {text_input_ids.shape}")
        print(f"  Audio shape:  {audio_inputs.shape}")
        print(f"  Output keys:  {list(outputs.keys())}")

        if "audio_latent_preds" in outputs:
            print(f"  Audio latent preds shape: {outputs['audio_latent_preds'].shape}")

        return True
    except Exception as e:
        print(f"✗ Text + Audio forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_with_image_forward(model: MegatransformerWorldModel, device: str = "cpu"):
    """Test forward pass with text + image input."""
    print("\n[TEST] Text + Image Forward Pass")
    print("-" * 40)

    batch_size = 2
    text_seq_len = 32
    n_images = 1
    vocab_size = model.config.text_feature_config.vocab_size

    # Get placeholder token ID
    image_placeholder = model.config.token_interleaver_config.image_placeholder_token_id

    # Get image latent dimensions from config
    image_config: ImageConfig = model.config.image_prelude_config.image_config
    latent_channels = image_config.latent_channels
    latent_size = image_config.image_size // image_config.latent_compression_factor

    try:
        # Create text input with one image placeholder per batch item
        text_input_ids = torch.randint(0, vocab_size - 10, (batch_size, text_seq_len), device=device)
        # Insert image placeholder at position 15
        text_input_ids[:, 15] = image_placeholder

        # Create image latents: (batch, n_images, latent_channels, latent_h, latent_w)
        image_inputs = torch.randn(
            batch_size, n_images, latent_channels, latent_size, latent_size,
            device=device
        )

        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(
                text_input_ids=text_input_ids,
                image_inputs=image_inputs,
                precomputed_latents=True,
            )

        print(f"✓ Text + Image forward pass successful")
        print(f"  Text shape:   {text_input_ids.shape}")
        print(f"  Image shape:  {image_inputs.shape}")
        print(f"  Output keys:  {list(outputs.keys())}")

        if "image_latent_preds" in outputs:
            print(f"  Image latent preds shape: {outputs['image_latent_preds'].shape}")

        return True
    except Exception as e:
        print(f"✗ Text + Image forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multimodal_forward(model: MegatransformerWorldModel, device: str = "cpu"):
    """Test forward pass with all modalities."""
    print("\n[TEST] Full Multimodal Forward Pass")
    print("-" * 40)

    batch_size = 2
    text_seq_len = 64
    n_audio = 1
    n_voice = 1
    n_images = 1
    vocab_size = model.config.text_feature_config.vocab_size

    # Get placeholder token IDs
    audio_placeholder = model.config.token_interleaver_config.audio_placeholder_token_id
    voice_placeholder = model.config.token_interleaver_config.voice_placeholder_token_id
    image_placeholder = model.config.token_interleaver_config.image_placeholder_token_id

    # Get latent dimensions
    audio_config: AudioConfig = model.config.audio_prelude_config.audio_config
    voice_config: AudioConfig = model.config.voice_prelude_config.audio_config
    image_config: ImageConfig = model.config.image_prelude_config.image_config

    audio_latent_channels = audio_config.latent_channels
    audio_latent_mel_bins = audio_config.n_mels // audio_config.latent_compression_factor[0]
    audio_latent_timesteps = 16

    voice_latent_channels = voice_config.latent_channels
    voice_latent_mel_bins = voice_config.n_mels // voice_config.latent_compression_factor[0]
    voice_latent_timesteps = 20

    image_latent_channels = image_config.latent_channels
    image_latent_size = image_config.image_size // image_config.latent_compression_factor

    try:
        # Create text input with placeholders
        text_input_ids = torch.randint(0, vocab_size - 10, (batch_size, text_seq_len), device=device)
        text_input_ids[:, 10] = audio_placeholder
        text_input_ids[:, 25] = voice_placeholder
        text_input_ids[:, 40] = image_placeholder

        # Create audio latents
        audio_inputs = torch.randn(
            batch_size, n_audio, audio_latent_channels, audio_latent_mel_bins, audio_latent_timesteps,
            device=device
        )
        audio_lengths = torch.full((batch_size, n_audio), audio_latent_timesteps, device=device)

        # Create voice latents
        voice_inputs = torch.randn(
            batch_size, n_voice, voice_latent_channels, voice_latent_mel_bins, voice_latent_timesteps,
            device=device
        )
        voice_lengths = torch.full((batch_size, n_voice), voice_latent_timesteps, device=device)

        # Create image latents
        image_inputs = torch.randn(
            batch_size, n_images, image_latent_channels, image_latent_size, image_latent_size,
            device=device
        )

        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(
                text_input_ids=text_input_ids,
                audio_inputs=audio_inputs,
                audio_lengths=audio_lengths,
                voice_inputs=voice_inputs,
                voice_lengths=voice_lengths,
                image_inputs=image_inputs,
                precomputed_latents=True,
            )

        print(f"✓ Full multimodal forward pass successful")
        print(f"  Text shape:   {text_input_ids.shape}")
        print(f"  Audio shape:  {audio_inputs.shape}")
        print(f"  Voice shape:  {voice_inputs.shape}")
        print(f"  Image shape:  {image_inputs.shape}")
        print(f"  Output keys:  {list(outputs.keys())}")

        return True
    except Exception as e:
        print(f"✗ Full multimodal forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_forward_with_loss(model: MegatransformerWorldModel, device: str = "cpu"):
    """Test forward pass with labels for loss computation."""
    print("\n[TEST] Training Forward Pass with Loss Computation")
    print("-" * 40)

    batch_size = 2
    text_seq_len = 32
    vocab_size = model.config.text_feature_config.vocab_size

    try:
        # Create text input and targets (shifted by 1 for next-token prediction)
        text_input_ids = torch.randint(0, vocab_size - 10, (batch_size, text_seq_len), device=device)
        text_targets = torch.randint(0, vocab_size - 10, (batch_size, text_seq_len), device=device)

        # Forward pass with targets
        model.train()
        outputs = model(
            text_input_ids=text_input_ids,
            text_targets=text_targets,
        )

        print(f"✓ Training forward pass with loss successful")
        print(f"  Output keys: {list(outputs.keys())}")

        if "text_classification_loss" in outputs:
            loss = outputs["text_classification_loss"]
            print(f"  Text loss: {loss.item():.4f}")

            # Test backward pass
            loss.backward()
            print(f"✓ Backward pass successful")

        return True
    except Exception as e:
        print(f"✗ Training forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow(model: MegatransformerWorldModel, device: str = "cpu"):
    """Test that gradients flow through all components."""
    print("\n[TEST] Gradient Flow Check")
    print("-" * 40)

    batch_size = 2
    text_seq_len = 32
    vocab_size = model.config.text_feature_config.vocab_size

    try:
        # Zero all gradients
        model.zero_grad()

        # Forward pass
        text_input_ids = torch.randint(0, vocab_size - 10, (batch_size, text_seq_len), device=device)
        text_targets = torch.randint(0, vocab_size - 10, (batch_size, text_seq_len), device=device)

        model.train()
        outputs = model(text_input_ids=text_input_ids, text_targets=text_targets)

        if "text_classification_loss" in outputs:
            outputs["text_classification_loss"].backward()

        # Check gradients for each major component
        components_with_grads = []
        components_without_grads = []

        for name, module in model.named_children():
            has_grad = False
            for param in module.parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_grad = True
                    break

            if has_grad:
                components_with_grads.append(name)
            else:
                components_without_grads.append(name)

        print(f"  Components with gradients: {components_with_grads}")
        if components_without_grads:
            print(f"  Components without gradients: {components_without_grads}")

        if len(components_with_grads) > 0:
            print(f"✓ Gradient flow check passed")
            return True
        else:
            print(f"✗ No gradients found!")
            return False

    except Exception as e:
        print(f"✗ Gradient flow check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Main
# =============================================================================

def get_config_by_name(name: str) -> MegatransformerWorldModelConfig:
    """Get a pre-defined config by name."""
    configs = {
        "tiny": tiny_world_model_config,
        "small": small_world_model_config,
        "medium": medium_world_model_config,
    }

    if name not in configs:
        raise ValueError(f"Unknown config '{name}'. Available: {list(configs.keys())}")

    return configs[name]


def main():
    parser = argparse.ArgumentParser(description="Test the Megatransformer World Model")
    parser.add_argument(
        "--config",
        type=str,
        default="tiny",
        choices=["tiny", "small", "medium"],
        help="Model configuration to use"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run tests on (cpu, cuda, cuda:0, etc.)"
    )
    parser.add_argument(
        "--print-model",
        action="store_true",
        help="Print full model architecture (can be verbose)"
    )
    parser.add_argument(
        "--skip-multimodal",
        action="store_true",
        help="Skip multimodal tests (faster for debugging)"
    )

    args = parser.parse_args()

    print(f"\n{'#' * 60}")
    print(f"# Megatransformer World Model Tests")
    print(f"# Config: {args.config}")
    print(f"# Device: {args.device}")
    print(f"{'#' * 60}")

    # Get config
    try:
        config = get_config_by_name(args.config)
        print(f"\n✓ Loaded {args.config} configuration")
    except Exception as e:
        print(f"\n✗ Failed to load config: {e}")
        return 1

    # Test instantiation
    model = test_model_instantiation(config, args.device)
    if model is None:
        print("\n⚠ Model instantiation failed, cannot continue with tests")
        return 1

    # Print model info
    print_model_summary(model, f"World Model ({args.config})")
    print_architecture(model, max_depth=2)

    if args.print_model:
        print("\n[Full Model Structure]")
        print(model)

    # Run tests
    results = {}

    results["text_only"] = test_text_only_forward(model, args.device)

    if not args.skip_multimodal:
        results["text_audio"] = test_text_with_audio_forward(model, args.device)
        results["text_image"] = test_text_with_image_forward(model, args.device)
        results["multimodal"] = test_multimodal_forward(model, args.device)

    results["training_loss"] = test_training_forward_with_loss(model, args.device)
    results["gradient_flow"] = test_gradient_flow(model, args.device)

    # Summary
    print(f"\n{'=' * 60}")
    print("Test Summary")
    print(f"{'=' * 60}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name:25s} {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
