"""
Test script for the GuBERT Feature VAE.

This script tests model instantiation, forward pass, gradient flow,
and various configurations of the GuBERT Feature VAE.

Usage:
    python tests/test_gubert_feature_vae.py
    python tests/test_gubert_feature_vae.py --config small
    python tests/test_gubert_feature_vae.py --device cuda
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.audio.vae import (
    Snake1d,
    ResidualBlock1d,
    GuBERTFeatureVAEEncoder,
    GuBERTFeatureVAEDecoder,
    GuBERTFeatureVAE,
    gubert_feature_vae_config_lookup,
    create_gubert_feature_vae,
)

import pytest


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def device() -> str:
    """Provide device string for tests."""
    return "cpu"


@pytest.fixture
def vae(device: str) -> GuBERTFeatureVAE:
    """Provide a tiny GuBERTFeatureVAE model for tests."""
    vae = create_gubert_feature_vae(config="tiny")
    return vae.to(device)


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

    print(f"\n{'─' * 60}")
    print("Parameter breakdown by component:")
    print(f"{'─' * 60}")

    for name, module in model.named_children():
        params = count_parameters(module, trainable_only=False)
        pct = 100 * params / total_params if total_params > 0 else 0
        print(f"  {name:35s} {format_param_count(params):>10s} ({pct:5.1f}%)")

    print(f"{'─' * 60}\n")


# =============================================================================
# Test Functions: Components
# =============================================================================

def test_snake1d_activation(device: str = "cpu"):
    """Test Snake1d activation module."""
    print("\n[TEST] Snake1d Activation")
    print("-" * 40)

    try:
        channels = 64
        snake = Snake1d(channels).to(device)

        x = torch.randn(2, channels, 50, device=device)
        y = snake(x)

        assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"

        # Snake should not be identity
        assert not torch.allclose(x, y), "Snake should modify input"

        # Check learnable alpha parameter
        assert hasattr(snake, 'alpha')
        assert snake.alpha.shape == (1, channels, 1)

        print(f"✓ Snake1d works")
        print(f"  Input/Output shape: {x.shape}")
        print(f"  Alpha shape: {snake.alpha.shape}")
        return True
    except Exception as e:
        print(f"✗ Snake1d test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_residual_block_1d(device: str = "cpu"):
    """Test ResidualBlock1d module."""
    print("\n[TEST] ResidualBlock1d")
    print("-" * 40)

    try:
        # Test same channels
        block = ResidualBlock1d(
            in_channels=128,
            out_channels=128,
            kernel_size=5,
            activation_fn="silu",
        ).to(device)

        x = torch.randn(2, 128, 50, device=device)
        y = block(x)

        assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
        print(f"  Same channels: {x.shape} -> {y.shape} ✓")

        # Test channel change
        block_proj = ResidualBlock1d(
            in_channels=128,
            out_channels=256,
            kernel_size=5,
        ).to(device)

        y_proj = block_proj(x)
        assert y_proj.shape == (2, 256, 50), f"Wrong shape: {y_proj.shape}"
        assert block_proj.skip_proj is not None, "Should have skip projection"
        print(f"  Channel change: {x.shape} -> {y_proj.shape} ✓")

        # Test with snake activation
        block_snake = ResidualBlock1d(
            in_channels=64,
            kernel_size=3,
            activation_fn="snake",
        ).to(device)

        x_small = torch.randn(2, 64, 30, device=device)
        y_snake = block_snake(x_small)
        assert y_snake.shape == x_small.shape
        print(f"  Snake activation: {x_small.shape} -> {y_snake.shape} ✓")

        print("✓ ResidualBlock1d works")
        return True
    except Exception as e:
        print(f"✗ ResidualBlock1d test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Encoder
# =============================================================================

def test_encoder_instantiation(device: str = "cpu"):
    """Test GuBERTFeatureVAEEncoder instantiation."""
    print("\n[TEST] Encoder Instantiation")
    print("-" * 40)

    try:
        encoder = GuBERTFeatureVAEEncoder(
            input_dim=256,
            latent_dim=32,
            intermediate_channels=[256, 192, 128],
            kernel_sizes=[5, 5, 3],
            strides=[2, 2, 2],
            n_residual_blocks=1,
        ).to(device)

        params = count_parameters(encoder)
        print(f"✓ Encoder instantiated")
        print(f"  Parameters: {format_param_count(params)}")
        return encoder
    except Exception as e:
        print(f"✗ Encoder instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_encoder_forward(device: str = "cpu"):
    """Test encoder forward pass."""
    print("\n[TEST] Encoder Forward Pass")
    print("-" * 40)

    try:
        encoder = GuBERTFeatureVAEEncoder(
            input_dim=256,
            latent_dim=32,
            intermediate_channels=[256, 128],
            kernel_sizes=[5, 3],
            strides=[2, 2],
            n_residual_blocks=1,
        ).to(device)

        batch_size = 2
        seq_len = 50
        input_dim = 256

        x = torch.randn(batch_size, seq_len, input_dim, device=device)
        lengths = torch.tensor([50, 40], device=device)

        mu, logvar, output_lengths = encoder(x, lengths)

        # Check shapes
        expected_latent_len = encoder.get_output_length(seq_len)
        assert mu.shape == (batch_size, 32, expected_latent_len), f"Mu shape wrong: {mu.shape}"
        assert logvar.shape == mu.shape, f"Logvar shape wrong: {logvar.shape}"
        assert output_lengths is not None

        print(f"✓ Encoder forward pass successful")
        print(f"  Input shape:   {x.shape}")
        print(f"  Mu shape:      {mu.shape}")
        print(f"  Logvar shape:  {logvar.shape}")
        print(f"  Input lengths:  {lengths.tolist()}")
        print(f"  Output lengths: {output_lengths.tolist()}")
        return True
    except Exception as e:
        print(f"✗ Encoder forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_encoder_output_length_calculation(device: str = "cpu"):
    """Test encoder output length calculation."""
    print("\n[TEST] Encoder Output Length Calculation")
    print("-" * 40)

    try:
        encoder = GuBERTFeatureVAEEncoder(
            input_dim=128,
            latent_dim=16,
            intermediate_channels=[128, 64],
            kernel_sizes=[5, 3],
            strides=[2, 2],  # 4x total downsampling
            n_residual_blocks=0,
        ).to(device)

        test_lengths = [16, 25, 50, 100, 200]

        print(f"  Strides: {encoder.strides}")
        print(f"  Total downsampling: {2 * 2}x")

        for input_len in test_lengths:
            # Calculate expected
            calculated = encoder.get_output_length(input_len)

            # Verify with actual forward pass
            x = torch.randn(1, input_len, 128, device=device)
            mu, _, _ = encoder(x)
            actual = mu.shape[2]

            assert calculated == actual, f"Length mismatch for {input_len}: {calculated} vs {actual}"
            print(f"  {input_len:4d} -> {actual:4d}")

        print("✓ Output length calculation correct")
        return True
    except Exception as e:
        print(f"✗ Output length calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Decoder
# =============================================================================

def test_decoder_instantiation(device: str = "cpu"):
    """Test GuBERTFeatureVAEDecoder instantiation."""
    print("\n[TEST] Decoder Instantiation")
    print("-" * 40)

    try:
        decoder = GuBERTFeatureVAEDecoder(
            latent_dim=32,
            output_dim=256,
            intermediate_channels=[128, 192, 256],
            kernel_sizes=[3, 5, 5],
            scale_factors=[2, 2, 2],
            n_residual_blocks=1,
        ).to(device)

        params = count_parameters(decoder)
        print(f"✓ Decoder instantiated")
        print(f"  Parameters: {format_param_count(params)}")
        return decoder
    except Exception as e:
        print(f"✗ Decoder instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_decoder_forward(device: str = "cpu"):
    """Test decoder forward pass."""
    print("\n[TEST] Decoder Forward Pass")
    print("-" * 40)

    try:
        decoder = GuBERTFeatureVAEDecoder(
            latent_dim=32,
            output_dim=256,
            intermediate_channels=[128, 256],
            kernel_sizes=[3, 5],
            scale_factors=[2, 2],
            n_residual_blocks=1,
        ).to(device)

        batch_size = 2
        latent_len = 13
        latent_dim = 32
        target_length = 50

        z = torch.randn(batch_size, latent_dim, latent_len, device=device)

        # Without target length
        recon = decoder(z)
        expected_len = decoder.get_output_length(latent_len)
        assert recon.shape == (batch_size, expected_len, 256), f"Wrong shape: {recon.shape}"
        print(f"  Without target: {z.shape} -> {recon.shape}")

        # With target length (trimming)
        recon_trimmed = decoder(z, target_length=target_length)
        assert recon_trimmed.shape == (batch_size, target_length, 256), f"Wrong shape: {recon_trimmed.shape}"
        print(f"  With target {target_length}: {z.shape} -> {recon_trimmed.shape}")

        print("✓ Decoder forward pass successful")
        return True
    except Exception as e:
        print(f"✗ Decoder forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Full VAE
# =============================================================================

def test_predefined_configs():
    """Test that all predefined configs can be instantiated."""
    print("\n[TEST] Predefined Configs")
    print("-" * 40)

    try:
        expected_configs = ["tiny", "small", "medium", "large", "xlarge"]

        for name in expected_configs:
            assert name in gubert_feature_vae_config_lookup, f"Missing config: {name}"
            vae = create_gubert_feature_vae(name)
            params = vae.get_num_params()
            print(f"  {name:8s}: {format_param_count(params):>10s}")

        print("✓ All predefined configs valid")
        return True
    except Exception as e:
        print(f"✗ Predefined configs test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vae_instantiation(config_name: str = "tiny", device: str = "cpu"):
    """Test VAE instantiation."""
    print("\n[TEST] VAE Instantiation")
    print("-" * 40)

    try:
        vae = create_gubert_feature_vae(config_name).to(device)
        params = vae.get_num_params()
        print(f"✓ VAE instantiated ({config_name})")
        print(f"  Parameters: {format_param_count(params)}")
        return vae
    except Exception as e:
        print(f"✗ VAE instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_vae_forward_basic(vae: GuBERTFeatureVAE, device: str = "cpu"):
    """Test basic VAE forward pass."""
    print("\n[TEST] VAE Basic Forward Pass")
    print("-" * 40)

    try:
        batch_size = 2
        seq_len = 50
        input_dim = vae.encoder.input_dim

        x = torch.randn(batch_size, seq_len, input_dim, device=device)

        vae.eval()
        with torch.no_grad():
            result = vae(x)

        # Check output keys
        assert "recon" in result
        assert "mu" in result
        assert "logvar" in result

        # Check reconstruction shape matches input
        assert result["recon"].shape == x.shape, f"Recon shape mismatch: {result['recon'].shape} vs {x.shape}"

        print(f"✓ Basic forward pass successful")
        print(f"  Input shape:  {x.shape}")
        print(f"  Recon shape:  {result['recon'].shape}")
        print(f"  Mu shape:     {result['mu'].shape}")
        print(f"  Logvar shape: {result['logvar'].shape}")
        return True
    except Exception as e:
        print(f"✗ Basic forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vae_forward_with_lengths(vae: GuBERTFeatureVAE, device: str = "cpu"):
    """Test VAE forward pass with lengths."""
    print("\n[TEST] VAE Forward Pass with Lengths")
    print("-" * 40)

    try:
        batch_size = 3
        seq_len = 64
        input_dim = vae.encoder.input_dim

        x = torch.randn(batch_size, seq_len, input_dim, device=device)
        lengths = torch.tensor([64, 50, 32], device=device)

        vae.eval()
        with torch.no_grad():
            result = vae(x, lengths=lengths)

        assert result["output_lengths"] is not None
        print(f"✓ Forward pass with lengths successful")
        print(f"  Input lengths:  {lengths.tolist()}")
        print(f"  Output lengths: {result['output_lengths'].tolist()}")
        return True
    except Exception as e:
        print(f"✗ Forward pass with lengths failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vae_forward_return_latent(vae: GuBERTFeatureVAE, device: str = "cpu"):
    """Test VAE forward pass with return_latent=True."""
    print("\n[TEST] VAE Forward with Return Latent")
    print("-" * 40)

    try:
        batch_size = 2
        seq_len = 40
        input_dim = vae.encoder.input_dim

        x = torch.randn(batch_size, seq_len, input_dim, device=device)

        vae.train()  # In train mode, reparameterization samples
        result = vae(x, return_latent=True)

        assert "z" in result, "Missing 'z' in result"
        assert result["z"].shape == result["mu"].shape

        print(f"✓ Return latent successful")
        print(f"  Z shape: {result['z'].shape}")
        return True
    except Exception as e:
        print(f"✗ Return latent failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vae_reparameterization(vae: GuBERTFeatureVAE, device: str = "cpu"):
    """Test reparameterization behavior in train vs eval mode."""
    print("\n[TEST] VAE Reparameterization")
    print("-" * 40)

    try:
        batch_size = 2
        seq_len = 30
        input_dim = vae.encoder.input_dim

        x = torch.randn(batch_size, seq_len, input_dim, device=device)

        # In eval mode, reparameterize should return mu directly
        vae.eval()
        with torch.no_grad():
            result_eval = vae(x, return_latent=True)
            # Run again - should be deterministic
            result_eval2 = vae(x, return_latent=True)

        assert torch.allclose(result_eval["z"], result_eval["mu"]), "Eval mode should return mu"
        assert torch.allclose(result_eval["z"], result_eval2["z"]), "Eval mode should be deterministic"
        print(f"  Eval mode: z == mu ✓")

        # In train mode, reparameterize should sample
        vae.train()
        torch.manual_seed(42)
        result_train1 = vae(x, return_latent=True)
        torch.manual_seed(43)
        result_train2 = vae(x, return_latent=True)

        # Different seeds should give different z (with high probability)
        assert not torch.allclose(result_train1["z"], result_train2["z"]), "Train mode should sample"
        print(f"  Train mode: z sampled ✓")

        print("✓ Reparameterization behavior correct")
        return True
    except Exception as e:
        print(f"✗ Reparameterization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Loss
# =============================================================================

def test_vae_loss_computation(vae: GuBERTFeatureVAE, device: str = "cpu"):
    """Test VAE loss computation."""
    print("\n[TEST] VAE Loss Computation")
    print("-" * 40)

    try:
        batch_size = 4
        seq_len = 50
        input_dim = vae.encoder.input_dim

        x = torch.randn(batch_size, seq_len, input_dim, device=device)

        vae.train()
        result = vae(x)

        losses = vae.compute_loss(x, result["recon"], result["mu"], result["logvar"])

        assert "reconstruction_loss" in losses
        assert "kl_loss" in losses
        assert "total_loss" in losses

        recon_loss = losses["reconstruction_loss"]
        kl_loss = losses["kl_loss"]
        total_loss = losses["total_loss"]

        # Check loss values are reasonable
        assert recon_loss >= 0, "Recon loss should be non-negative"
        assert kl_loss >= 0, "KL loss should be non-negative"
        assert torch.isfinite(total_loss), "Total loss should be finite"

        # Check total = recon + kl_weight * kl
        expected_total = recon_loss + vae.kl_weight * kl_loss
        assert torch.allclose(total_loss, expected_total), "Total loss computation wrong"

        print(f"✓ Loss computation correct")
        print(f"  Reconstruction loss: {recon_loss.item():.4f}")
        print(f"  KL loss:            {kl_loss.item():.4f}")
        print(f"  Total loss:         {total_loss.item():.4f}")
        return True
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vae_masked_loss(vae: GuBERTFeatureVAE, device: str = "cpu"):
    """Test VAE loss computation with length masking."""
    print("\n[TEST] VAE Masked Loss")
    print("-" * 40)

    try:
        batch_size = 3
        seq_len = 50
        input_dim = vae.encoder.input_dim

        x = torch.randn(batch_size, seq_len, input_dim, device=device)
        lengths = torch.tensor([50, 30, 20], device=device)

        vae.train()
        result = vae(x, lengths=lengths)

        # Compute masked loss
        losses_masked = vae.compute_loss(
            x, result["recon"], result["mu"], result["logvar"], lengths=lengths
        )

        # Compute unmasked loss
        losses_unmasked = vae.compute_loss(
            x, result["recon"], result["mu"], result["logvar"], lengths=None
        )

        # Masked and unmasked should be different (different normalization)
        assert not torch.allclose(
            losses_masked["reconstruction_loss"],
            losses_unmasked["reconstruction_loss"]
        ), "Masked and unmasked loss should differ"

        print(f"✓ Masked loss computation works")
        print(f"  Unmasked recon loss: {losses_unmasked['reconstruction_loss'].item():.4f}")
        print(f"  Masked recon loss:   {losses_masked['reconstruction_loss'].item():.4f}")
        return True
    except Exception as e:
        print(f"✗ Masked loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Gradient Flow
# =============================================================================

def test_gradient_flow(vae: GuBERTFeatureVAE, device: str = "cpu"):
    """Test that gradients flow through all components."""
    print("\n[TEST] Gradient Flow")
    print("-" * 40)

    try:
        batch_size = 2
        seq_len = 40
        input_dim = vae.encoder.input_dim

        vae.zero_grad()
        vae.train()

        x = torch.randn(batch_size, seq_len, input_dim, device=device)

        result = vae(x)
        losses = vae.compute_loss(x, result["recon"], result["mu"], result["logvar"])
        losses["total_loss"].backward()

        # Check gradients for each component
        components_with_grads = []
        components_without_grads = []

        for name, module in vae.named_children():
            has_grad = any(
                param.grad is not None and param.grad.abs().sum() > 0
                for param in module.parameters()
            )
            if has_grad:
                components_with_grads.append(name)
            else:
                components_without_grads.append(name)

        print(f"  Components with gradients: {components_with_grads}")
        if components_without_grads:
            print(f"  Components without gradients: {components_without_grads}")

        # Both encoder and decoder should have gradients
        assert "encoder" in components_with_grads, "Encoder should have gradients"
        assert "decoder" in components_with_grads, "Decoder should have gradients"

        print("✓ Gradient flow check passed")
        return True
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Various Input Sizes
# =============================================================================

def test_different_sequence_lengths(vae: GuBERTFeatureVAE, device: str = "cpu"):
    """Test VAE with various sequence lengths."""
    print("\n[TEST] Different Sequence Lengths")
    print("-" * 40)

    try:
        input_dim = vae.encoder.input_dim
        test_lengths = [16, 25, 50, 100, 200]

        vae.eval()

        for seq_len in test_lengths:
            x = torch.randn(1, seq_len, input_dim, device=device)

            with torch.no_grad():
                result = vae(x)

            assert result["recon"].shape == x.shape, f"Shape mismatch at length {seq_len}"
            latent_len = result["mu"].shape[2]
            print(f"  {seq_len:4d} -> latent {latent_len:4d} -> recon {result['recon'].shape[1]:4d}")

        print("✓ All sequence lengths handled correctly")
        return True
    except Exception as e:
        print(f"✗ Sequence length test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_batch_sizes(vae: GuBERTFeatureVAE, device: str = "cpu"):
    """Test VAE with various batch sizes."""
    print("\n[TEST] Different Batch Sizes")
    print("-" * 40)

    try:
        input_dim = vae.encoder.input_dim
        seq_len = 50
        batch_sizes = [1, 2, 4, 8, 16]

        vae.eval()

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, seq_len, input_dim, device=device)

            with torch.no_grad():
                result = vae(x)

            assert result["recon"].shape[0] == batch_size
            assert result["mu"].shape[0] == batch_size

        print(f"✓ All batch sizes handled correctly")
        print(f"  Tested: {batch_sizes}")
        return True
    except Exception as e:
        print(f"✗ Batch size test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Create Function
# =============================================================================

def test_create_function_with_overrides():
    """Test create_gubert_feature_vae with parameter overrides."""
    print("\n[TEST] Create Function with Overrides")
    print("-" * 40)

    try:
        # Override input_dim
        vae1 = create_gubert_feature_vae("small", input_dim=512)
        assert vae1.encoder.input_dim == 512
        assert vae1.decoder.output_dim == 512
        print(f"  input_dim override: 512 ✓")

        # Override latent_dim
        vae2 = create_gubert_feature_vae("small", latent_dim=64)
        assert vae2.encoder.latent_dim == 64
        assert vae2.decoder.latent_dim == 64
        print(f"  latent_dim override: 64 ✓")

        # Override kl_weight
        vae3 = create_gubert_feature_vae("tiny", kl_weight=0.001)
        assert vae3.kl_weight == 0.001
        print(f"  kl_weight override: 0.001 ✓")

        # Multiple overrides
        vae4 = create_gubert_feature_vae("tiny", input_dim=64, latent_dim=8, kl_weight=0.5)
        assert vae4.encoder.input_dim == 64
        assert vae4.encoder.latent_dim == 8
        assert vae4.kl_weight == 0.5
        print(f"  Multiple overrides ✓")

        print("✓ Create function overrides work correctly")
        return True
    except Exception as e:
        print(f"✗ Create function override test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_invalid_config():
    """Test that invalid config raises error."""
    print("\n[TEST] Invalid Config Error")
    print("-" * 40)

    try:
        try:
            create_gubert_feature_vae("nonexistent_config")
            print("✗ Should have raised ValueError")
            return False
        except ValueError as e:
            print(f"  Caught expected error: {e}")
            print("✓ Invalid config raises ValueError")
            return True
    except Exception as e:
        print(f"✗ Invalid config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Encode/Decode Methods
# =============================================================================

def test_encode_decode_methods(vae: GuBERTFeatureVAE, device: str = "cpu"):
    """Test separate encode and decode methods."""
    print("\n[TEST] Encode/Decode Methods")
    print("-" * 40)

    try:
        batch_size = 2
        seq_len = 50
        input_dim = vae.encoder.input_dim

        x = torch.randn(batch_size, seq_len, input_dim, device=device)

        vae.eval()
        with torch.no_grad():
            # Test encode
            mu, logvar, _ = vae.encode(x)
            assert mu.shape[0] == batch_size
            assert mu.shape[1] == vae.encoder.latent_dim
            print(f"  Encode: {x.shape} -> mu {mu.shape}")

            # Test decode
            recon = vae.decode(mu, target_length=seq_len)
            assert recon.shape == x.shape
            print(f"  Decode: {mu.shape} -> {recon.shape}")

        print("✓ Encode/Decode methods work correctly")
        return True
    except Exception as e:
        print(f"✗ Encode/Decode methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test GuBERT Feature VAE")
    parser.add_argument(
        "--config",
        type=str,
        default="tiny",
        choices=list(gubert_feature_vae_config_lookup.keys()),
        help="Model configuration to use for full VAE tests"
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
        help="Print full model architecture"
    )

    args = parser.parse_args()

    print(f"\n{'#' * 60}")
    print(f"# GuBERT Feature VAE Tests")
    print(f"# Config: {args.config}")
    print(f"# Device: {args.device}")
    print(f"{'#' * 60}")

    results = {}

    # Component tests
    results["snake1d"] = test_snake1d_activation(args.device)
    results["residual_block_1d"] = test_residual_block_1d(args.device)

    # Encoder tests
    encoder = test_encoder_instantiation(args.device)
    results["encoder_instantiation"] = encoder is not None
    results["encoder_forward"] = test_encoder_forward(args.device)
    results["encoder_length_calc"] = test_encoder_output_length_calculation(args.device)

    # Decoder tests
    decoder = test_decoder_instantiation(args.device)
    results["decoder_instantiation"] = decoder is not None
    results["decoder_forward"] = test_decoder_forward(args.device)

    # Full VAE tests
    results["predefined_configs"] = test_predefined_configs()

    vae = test_vae_instantiation(args.config, args.device)
    results["vae_instantiation"] = vae is not None

    if vae is None:
        print("\nVAE instantiation failed, skipping further tests")
        return 1

    print_model_summary(vae, f"GuBERT Feature VAE ({args.config})")

    if args.print_model:
        print("\n[Full Model Structure]")
        print(vae)

    results["vae_forward_basic"] = test_vae_forward_basic(vae, args.device)
    results["vae_forward_lengths"] = test_vae_forward_with_lengths(vae, args.device)
    results["vae_forward_latent"] = test_vae_forward_return_latent(vae, args.device)
    results["vae_reparameterization"] = test_vae_reparameterization(vae, args.device)

    # Loss tests
    results["vae_loss"] = test_vae_loss_computation(vae, args.device)
    results["vae_masked_loss"] = test_vae_masked_loss(vae, args.device)

    # Gradient tests
    results["gradient_flow"] = test_gradient_flow(vae, args.device)

    # Input size tests
    results["sequence_lengths"] = test_different_sequence_lengths(vae, args.device)
    results["batch_sizes"] = test_different_batch_sizes(vae, args.device)

    # Create function tests
    results["create_overrides"] = test_create_function_with_overrides()
    results["invalid_config"] = test_invalid_config()

    # Encode/Decode tests
    results["encode_decode"] = test_encode_decode_methods(vae, args.device)

    # Summary
    print(f"\n{'=' * 60}")
    print("Test Summary")
    print(f"{'=' * 60}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name:30s} {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
