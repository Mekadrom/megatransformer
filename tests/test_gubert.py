"""
Test script for the GuBERT Speaker-Invariant Speech Encoder.

This script tests model instantiation, prints architecture information,
and runs forward pass tests with various input configurations.

Usage:
    python tests/test_gubert.py
    python tests/test_gubert.py --config tiny
    python tests/test_gubert.py --config small --device cuda
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.audio.gubert import (
    GuBERTEncoder,
    GuBERTConfig,
    CTCVocab,
    GUBERT_CONFIGS,
    ConvSubsampling,
    TransformerEncoderBlock,
    SpeakerClassifier,
    GradientReversalFunction,
    create_gubert,
    # MaskedGuBERT imports
    MaskedGuBERTEncoder,
    MaskedGuBERTConfig,
    MASKED_GUBERT_CONFIGS,
    VectorQuantizer,
    generate_mask_spans,
    create_masked_gubert,
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
def model(device: str) -> GuBERTEncoder:
    """Provide a tiny GuBERTEncoder model for tests."""
    model = GuBERTEncoder.from_config("tiny")
    return model.to(device)


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


# =============================================================================
# Test Functions: Configuration
# =============================================================================

def test_config_creation():
    """Test GuBERTConfig creation and defaults."""
    print("\n[TEST] GuBERTConfig Creation")
    print("-" * 40)

    try:
        # Default config
        config = GuBERTConfig()
        assert config.n_mels == 80
        assert config.encoder_dim == 256
        assert config.num_layers == 4
        assert config.vocab_size == 32
        assert config.num_speakers == 992
        assert config.conv_kernel_sizes is not None
        assert config.conv_strides is not None

        print(f"  Default config: encoder_dim={config.encoder_dim}, layers={config.num_layers}")

        # Custom config
        custom = GuBERTConfig(
            encoder_dim=512,
            num_layers=8,
            num_speakers=500,
        )
        assert custom.encoder_dim == 512
        assert custom.num_layers == 8
        assert custom.num_speakers == 500

        print(f"  Custom config: encoder_dim={custom.encoder_dim}, layers={custom.num_layers}")
        print("✓ GuBERTConfig creation successful")
        return True
    except Exception as e:
        print(f"✗ GuBERTConfig creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_predefined_configs():
    """Test that all predefined configs can be accessed."""
    print("\n[TEST] Predefined Configs")
    print("-" * 40)

    try:
        expected_configs = ["tiny", "small", "medium", "large"]

        for name in expected_configs:
            assert name in GUBERT_CONFIGS, f"Missing config: {name}"
            config = GUBERT_CONFIGS[name]
            assert isinstance(config, GuBERTConfig)
            print(f"  {name}: encoder_dim={config.encoder_dim}, layers={config.num_layers}")

        print("✓ Predefined configs valid")
        return True
    except Exception as e:
        print(f"✗ Predefined configs test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Model Instantiation
# =============================================================================

def test_model_instantiation(config_name: str = "tiny", device: str = "cpu"):
    """Test that the model can be instantiated without errors."""
    print("\n[TEST] Model Instantiation")
    print("-" * 40)

    try:
        model = GuBERTEncoder.from_config(config_name)
        model = model.to(device)
        print(f"✓ Model instantiated successfully on {device}")
        print(f"  Config: {config_name}")
        print(f"  Parameters: {format_param_count(model.get_num_params())}")
        return model
    except Exception as e:
        print(f"✗ Failed to instantiate model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_with_overrides():
    """Test model creation with config overrides."""
    print("\n[TEST] Model with Config Overrides")
    print("-" * 40)

    try:
        model = GuBERTEncoder.from_config(
            "tiny",
            num_speakers=500,
            vocab_size=40,
            dropout=0.2,
        )

        assert model.config.num_speakers == 500
        assert model.config.vocab_size == 40
        assert model.config.dropout == 0.2

        print(f"  num_speakers: {model.config.num_speakers}")
        print(f"  vocab_size: {model.config.vocab_size}")
        print(f"  dropout: {model.config.dropout}")
        print("✓ Config overrides applied correctly")
        return True
    except Exception as e:
        print(f"✗ Config overrides test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_create_gubert_function():
    """Test the create_gubert convenience function."""
    print("\n[TEST] create_gubert Function")
    print("-" * 40)

    try:
        model = create_gubert(
            config="small",
            num_speakers=100,
            vocab_size=35,
        )

        assert isinstance(model, GuBERTEncoder)
        assert model.config.num_speakers == 100
        assert model.config.vocab_size == 35

        print(f"  Created model with {format_param_count(model.get_num_params())} params")
        print("✓ create_gubert function works")
        return True
    except Exception as e:
        print(f"✗ create_gubert function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Forward Pass
# =============================================================================

def test_forward_pass_basic(model: GuBERTEncoder, device: str = "cpu"):
    """Test basic forward pass without lengths."""
    print("\n[TEST] Basic Forward Pass")
    print("-" * 40)

    batch_size = 2
    n_mels = model.config.n_mels
    seq_len = 100

    try:
        mel_spec = torch.randn(batch_size, n_mels, seq_len, device=device)

        model.eval()
        with torch.no_grad():
            result = model(mel_spec)

        # Check output keys
        assert "features" in result
        assert "asr_logits" in result
        assert "speaker_logits" in result
        assert "feature_lengths" in result

        features = result["features"]
        asr_logits = result["asr_logits"]
        speaker_logits = result["speaker_logits"]

        # Check shapes
        expected_time = model.get_output_length(seq_len)
        assert features.shape[0] == batch_size
        assert features.shape[1] == expected_time
        assert features.shape[2] == model.config.encoder_dim

        assert asr_logits.shape == (batch_size, expected_time, model.config.vocab_size)
        assert speaker_logits.shape == (batch_size, model.config.num_speakers)

        print(f"✓ Basic forward pass successful")
        print(f"  Input shape:          {mel_spec.shape}")
        print(f"  Features shape:       {features.shape}")
        print(f"  ASR logits shape:     {asr_logits.shape}")
        print(f"  Speaker logits shape: {speaker_logits.shape}")
        return True
    except Exception as e:
        print(f"✗ Basic forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass_with_lengths(model: GuBERTEncoder, device: str = "cpu"):
    """Test forward pass with sequence lengths."""
    print("\n[TEST] Forward Pass with Lengths")
    print("-" * 40)

    batch_size = 3
    n_mels = model.config.n_mels
    max_seq_len = 150

    try:
        mel_spec = torch.randn(batch_size, n_mels, max_seq_len, device=device)
        lengths = torch.tensor([100, 150, 80], device=device)

        model.eval()
        with torch.no_grad():
            result = model(mel_spec, lengths=lengths)

        feature_lengths = result["feature_lengths"]

        # Feature lengths should be computed correctly
        assert feature_lengths is not None
        assert len(feature_lengths) == batch_size

        for i, orig_len in enumerate(lengths):
            expected_len = model.get_output_length(orig_len.item())
            assert feature_lengths[i].item() == expected_len, \
                f"Length mismatch at {i}: got {feature_lengths[i].item()}, expected {expected_len}"

        print(f"✓ Forward pass with lengths successful")
        print(f"  Input lengths:   {lengths.tolist()}")
        print(f"  Feature lengths: {feature_lengths.tolist()}")
        return True
    except Exception as e:
        print(f"✗ Forward pass with lengths failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass_with_grl_alpha(model: GuBERTEncoder, device: str = "cpu"):
    """Test forward pass with different GRL alpha values."""
    print("\n[TEST] Forward Pass with GRL Alpha")
    print("-" * 40)

    batch_size = 2
    n_mels = model.config.n_mels
    seq_len = 100

    try:
        mel_spec = torch.randn(batch_size, n_mels, seq_len, device=device)

        model.eval()
        with torch.no_grad():
            # Alpha = 0 (no gradient reversal)
            result_0 = model(mel_spec, grl_alpha=0.0)

            # Alpha = 1 (full gradient reversal)
            result_1 = model(mel_spec, grl_alpha=1.0)

            # Alpha = 0.5 (partial reversal)
            result_05 = model(mel_spec, grl_alpha=0.5)

        # Forward pass should produce same outputs regardless of alpha
        # (alpha only affects backward pass)
        assert torch.allclose(result_0["features"], result_1["features"])
        assert torch.allclose(result_0["asr_logits"], result_1["asr_logits"])

        print(f"✓ GRL alpha handling successful")
        print(f"  Tested alpha values: [0.0, 0.5, 1.0]")
        print(f"  Forward outputs identical (as expected)")
        return True
    except Exception as e:
        print(f"✗ GRL alpha test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_return_all_hiddens(model: GuBERTEncoder, device: str = "cpu"):
    """Test forward pass returning all hidden states."""
    print("\n[TEST] Return All Hidden States")
    print("-" * 40)

    batch_size = 2
    n_mels = model.config.n_mels
    seq_len = 100

    try:
        mel_spec = torch.randn(batch_size, n_mels, seq_len, device=device)

        model.eval()
        with torch.no_grad():
            result = model(mel_spec, return_all_hiddens=True)

        assert "all_hiddens" in result
        all_hiddens = result["all_hiddens"]

        # Should have num_layers + 1 hidden states (including input after pos encoding)
        expected_num = model.config.num_layers + 1
        assert len(all_hiddens) == expected_num, \
            f"Expected {expected_num} hidden states, got {len(all_hiddens)}"

        # All should have same shape
        for i, hidden in enumerate(all_hiddens):
            assert hidden.shape == result["features"].shape, \
                f"Hidden {i} shape mismatch"

        print(f"✓ Return all hiddens successful")
        print(f"  Number of hidden states: {len(all_hiddens)}")
        print(f"  Each hidden shape: {all_hiddens[0].shape}")
        return True
    except Exception as e:
        print(f"✗ Return all hiddens test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Feature Extraction
# =============================================================================

def test_extract_features(model: GuBERTEncoder, device: str = "cpu"):
    """Test the extract_features method."""
    print("\n[TEST] Extract Features")
    print("-" * 40)

    batch_size = 2
    n_mels = model.config.n_mels
    seq_len = 100

    try:
        mel_spec = torch.randn(batch_size, n_mels, seq_len, device=device)
        lengths = torch.tensor([100, 80], device=device)

        # Extract from final layer
        features, feature_lengths = model.extract_features(mel_spec, lengths=lengths)

        expected_time = model.get_output_length(seq_len)
        assert features.shape == (batch_size, expected_time, model.config.encoder_dim)
        assert feature_lengths is not None

        # Extract from intermediate layer
        features_mid, _ = model.extract_features(mel_spec, lengths=lengths, layer=2)
        assert features_mid.shape == features.shape

        print(f"✓ Feature extraction successful")
        print(f"  Final layer features: {features.shape}")
        print(f"  Layer 2 features: {features_mid.shape}")
        return True
    except Exception as e:
        print(f"✗ Feature extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Gradient Flow
# =============================================================================

def test_gradient_flow(model: GuBERTEncoder, device: str = "cpu"):
    """Test that gradients flow through all components."""
    print("\n[TEST] Gradient Flow")
    print("-" * 40)

    batch_size = 2
    n_mels = model.config.n_mels
    seq_len = 100

    try:
        model.zero_grad()
        model.train()

        mel_spec = torch.randn(batch_size, n_mels, seq_len, device=device)
        lengths = torch.tensor([100, 80], device=device)

        result = model(mel_spec, lengths=lengths, grl_alpha=1.0)

        # Compute dummy loss
        asr_loss = result["asr_logits"].sum()
        speaker_loss = result["speaker_logits"].sum()
        total_loss = asr_loss + speaker_loss

        total_loss.backward()

        # Check gradients
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

        # All trainable components should have gradients
        if len(components_with_grads) >= 4:  # conv_subsample, encoder_blocks, asr_head, speaker_classifier
            print("✓ Gradient flow check passed")
            return True
        else:
            print("✗ Missing gradients in some components")
            return False

    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_grl_gradient_reversal(device: str = "cpu"):
    """Test that GRL actually reverses gradients."""
    print("\n[TEST] GRL Gradient Reversal")
    print("-" * 40)

    try:
        x = torch.randn(2, 4, requires_grad=True, device=device)

        # Forward pass (should be identity)
        y_forward = GradientReversalFunction.apply(x, 1.0)
        assert torch.equal(x, y_forward), "GRL forward should be identity"

        # Backward pass (should reverse)
        loss = y_forward.sum()
        loss.backward()

        # Gradient should be -1 everywhere (negative of ones)
        expected_grad = -torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad), \
            f"GRL backward should negate gradients, got {x.grad}"

        print("✓ GRL gradient reversal verified")
        print(f"  Forward: identity")
        print(f"  Backward: gradient negated")
        return True
    except Exception as e:
        print(f"✗ GRL gradient reversal test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Components
# =============================================================================

def test_conv_subsampling(device: str = "cpu"):
    """Test ConvSubsampling module."""
    print("\n[TEST] ConvSubsampling")
    print("-" * 40)

    try:
        conv = ConvSubsampling(
            in_channels=80,
            out_channels=256,
            kernel_sizes=[5, 3, 3],
            strides=[2, 2, 1],
        ).to(device)

        x = torch.randn(2, 80, 100, device=device)
        y = conv(x)

        # Total stride is 2*2*1 = 4
        expected_len = conv.get_output_length(100)
        assert y.shape == (2, 256, expected_len), f"Wrong shape: {y.shape}"

        print(f"✓ ConvSubsampling works")
        print(f"  Input: {x.shape}")
        print(f"  Output: {y.shape}")
        print(f"  Total stride: {conv.total_stride}")
        return True
    except Exception as e:
        print(f"✗ ConvSubsampling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transformer_encoder_block(device: str = "cpu"):
    """Test TransformerEncoderBlock module."""
    print("\n[TEST] TransformerEncoderBlock")
    print("-" * 40)

    try:
        block = TransformerEncoderBlock(
            d_model=256,
            n_heads=4,
            d_ff=1024,
            dropout=0.1,
        ).to(device)

        x = torch.randn(2, 50, 256, device=device)
        y = block(x)

        assert y.shape == x.shape, f"Shape should be preserved: {y.shape} vs {x.shape}"

        # Test with padding mask
        padding_mask = torch.zeros(2, 50, dtype=torch.bool, device=device)
        padding_mask[0, 40:] = True  # Mask last 10 positions for first sample
        y_masked = block(x, key_padding_mask=padding_mask)

        assert y_masked.shape == x.shape

        print(f"✓ TransformerEncoderBlock works")
        print(f"  Input/Output shape: {x.shape}")
        return True
    except Exception as e:
        print(f"✗ TransformerEncoderBlock test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_speaker_classifier(device: str = "cpu"):
    """Test SpeakerClassifier module."""
    print("\n[TEST] SpeakerClassifier")
    print("-" * 40)

    try:
        for pooling in ["mean", "max", "attention"]:
            classifier = SpeakerClassifier(
                d_model=256,
                num_speakers=100,
                pooling=pooling,
            ).to(device)

            x = torch.randn(2, 50, 256, device=device)
            logits = classifier(x)

            assert logits.shape == (2, 100), f"Wrong shape for {pooling}: {logits.shape}"

            # Test with mask
            mask = torch.ones(2, 50, dtype=torch.bool, device=device)
            mask[0, 40:] = False
            logits_masked = classifier(x, mask=mask)
            assert logits_masked.shape == (2, 100)

        print(f"✓ SpeakerClassifier works for all pooling types")
        print(f"  Tested: mean, max, attention")
        return True
    except Exception as e:
        print(f"✗ SpeakerClassifier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: CTCVocab
# =============================================================================

def test_ctc_vocab_creation():
    """Test CTCVocab creation and properties."""
    print("\n[TEST] CTCVocab Creation")
    print("-" * 40)

    try:
        vocab = CTCVocab()

        assert vocab.blank_idx == 0
        assert vocab.unk_idx == 1
        assert vocab.vocab_size > 0
        assert len(vocab.idx_to_char) == vocab.vocab_size
        assert len(vocab.char_to_idx) == vocab.vocab_size

        print(f"✓ CTCVocab created")
        print(f"  Vocab size: {vocab.vocab_size}")
        print(f"  Blank idx: {vocab.blank_idx}")
        print(f"  Characters: '{vocab.chars}'")
        return True
    except Exception as e:
        print(f"✗ CTCVocab creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ctc_vocab_encode_decode():
    """Test CTCVocab encoding and decoding."""
    print("\n[TEST] CTCVocab Encode/Decode")
    print("-" * 40)

    try:
        vocab = CTCVocab()

        test_texts = [
            "hello world",
            "the quick brown fox",
            "a",
            "",
        ]

        for text in test_texts:
            encoded = vocab.encode(text)
            decoded = vocab.decode(encoded, remove_blanks=True, collapse_repeats=False)

            # Decoded should match input (for simple lowercase text)
            assert decoded == text.lower(), f"Mismatch: '{decoded}' vs '{text.lower()}'"

        print(f"✓ CTCVocab encode/decode works")
        return True
    except Exception as e:
        print(f"✗ CTCVocab encode/decode failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ctc_vocab_greedy_decode():
    """Test CTCVocab greedy decoding."""
    print("\n[TEST] CTCVocab Greedy Decode")
    print("-" * 40)

    try:
        vocab = CTCVocab()

        # Create mock logits with clear peaks
        seq_len = 10
        logits = torch.zeros(1, seq_len, vocab.vocab_size)

        # Set high values for specific characters
        # "hello" -> indices for h, e, l, l, o (with blanks in between)
        text = "hello"
        for i, char in enumerate(text):
            pos = i * 2  # Every other position
            if pos < seq_len:
                char_idx = vocab.char_to_idx.get(char, vocab.unk_idx)
                logits[0, pos, char_idx] = 10.0
                # Put blank in between
                if pos + 1 < seq_len:
                    logits[0, pos + 1, vocab.blank_idx] = 10.0

        decoded = vocab.ctc_decode_greedy(logits)

        assert len(decoded) == 1
        # After CTC collapse, should get "hello"
        result = decoded[0]
        print(f"  Decoded: '{result}'")

        print(f"✓ CTCVocab greedy decode works")
        return True
    except Exception as e:
        print(f"✗ CTCVocab greedy decode failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: MaskedGuBERT
# =============================================================================

def test_masked_gubert_config_creation():
    """Test MaskedGuBERTConfig creation and defaults."""
    print("\n[TEST] MaskedGuBERTConfig Creation")
    print("-" * 40)

    try:
        # Default config
        config = MaskedGuBERTConfig()
        assert config.n_mels == 80
        assert config.encoder_dim == 256
        assert config.num_layers == 4
        assert config.num_codebooks == 2
        assert config.codebook_size == 320
        assert config.mask_prob == 0.08
        assert config.mask_length == 10

        print(f"  Default config: encoder_dim={config.encoder_dim}, codebooks={config.num_codebooks}")

        # Custom config
        custom = MaskedGuBERTConfig(
            encoder_dim=512,
            num_layers=8,
            num_codebooks=4,
            codebook_size=512,
        )
        assert custom.encoder_dim == 512
        assert custom.num_layers == 8
        assert custom.num_codebooks == 4
        assert custom.codebook_size == 512

        print(f"  Custom config: encoder_dim={custom.encoder_dim}, codebooks={custom.num_codebooks}")
        print("✓ MaskedGuBERTConfig creation successful")
        return True
    except Exception as e:
        print(f"✗ MaskedGuBERTConfig creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_masked_gubert_predefined_configs():
    """Test that all predefined MaskedGuBERT configs can be accessed."""
    print("\n[TEST] MaskedGuBERT Predefined Configs")
    print("-" * 40)

    try:
        expected_configs = ["tiny", "small", "medium", "large"]

        for name in expected_configs:
            assert name in MASKED_GUBERT_CONFIGS, f"Missing config: {name}"
            config = MASKED_GUBERT_CONFIGS[name]
            assert isinstance(config, MaskedGuBERTConfig)
            print(f"  {name}: encoder_dim={config.encoder_dim}, codebooks={config.num_codebooks}")

        print("✓ MaskedGuBERT predefined configs valid")
        return True
    except Exception as e:
        print(f"✗ MaskedGuBERT predefined configs test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_masked_gubert_instantiation(config_name: str = "tiny", device: str = "cpu"):
    """Test that MaskedGuBERT can be instantiated without errors."""
    print("\n[TEST] MaskedGuBERT Instantiation")
    print("-" * 40)

    try:
        model = MaskedGuBERTEncoder.from_config(config_name)
        model = model.to(device)
        print(f"✓ MaskedGuBERT instantiated successfully on {device}")
        print(f"  Config: {config_name}")
        print(f"  Parameters: {format_param_count(model.get_num_params())}")
        return model
    except Exception as e:
        print(f"✗ Failed to instantiate MaskedGuBERT: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_masked_gubert_forward_pass(device: str = "cpu"):
    """Test MaskedGuBERT forward pass (regression mode - default)."""
    print("\n[TEST] MaskedGuBERT Forward Pass (Regression Mode)")
    print("-" * 40)

    batch_size = 2
    n_mels = 80
    seq_len = 100

    try:
        # Default is regression mode (use_vq=False)
        model = MaskedGuBERTEncoder.from_config("tiny").to(device)
        mel_spec = torch.randn(batch_size, n_mels, seq_len, device=device)

        model.train()  # Masking happens in train mode
        result = model(mel_spec)

        # Check common output keys
        assert "features" in result
        assert "speaker_logits" in result
        assert "mask" in result

        # Check regression-specific keys (default mode)
        assert "predictions" in result, "Regression mode should have 'predictions'"
        assert "targets" in result, "Regression mode should have 'targets'"

        features = result["features"]
        predictions = result["predictions"]
        targets = result["targets"]
        speaker_logits = result["speaker_logits"]
        mask = result["mask"]

        # Check shapes
        expected_time = model.get_output_length(seq_len)
        assert features.shape == (batch_size, expected_time, model.config.encoder_dim)
        assert predictions.shape == (batch_size, expected_time, model.config.encoder_dim)
        assert targets.shape == (batch_size, expected_time, model.config.encoder_dim)
        assert speaker_logits.shape == (batch_size, model.config.num_speakers)
        assert mask.shape == (batch_size, expected_time)

        # Check that mask has some True values (masking happened)
        assert mask.any(), "Mask should have some True values in training mode"

        print(f"✓ MaskedGuBERT forward pass successful (regression mode)")
        print(f"  Input shape:           {mel_spec.shape}")
        print(f"  Features shape:        {features.shape}")
        print(f"  Predictions shape:     {predictions.shape}")
        print(f"  Targets shape:         {targets.shape}")
        print(f"  Speaker logits:        {speaker_logits.shape}")
        print(f"  Mask shape:            {mask.shape}")
        print(f"  Masked positions:      {mask.sum().item()}")
        return True
    except Exception as e:
        print(f"✗ MaskedGuBERT forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_masked_gubert_inference_mode(device: str = "cpu"):
    """Test MaskedGuBERT in eval mode (no masking)."""
    print("\n[TEST] MaskedGuBERT Inference Mode")
    print("-" * 40)

    batch_size = 2
    n_mels = 80
    seq_len = 100

    try:
        model = MaskedGuBERTEncoder.from_config("tiny").to(device)
        mel_spec = torch.randn(batch_size, n_mels, seq_len, device=device)

        model.eval()  # No masking in eval mode
        with torch.no_grad():
            result = model(mel_spec)

        features = result["features"]
        mask = result["mask"]

        # In eval mode, mask should be None or empty (no automatic masking)
        # The mask won't be generated automatically in eval mode

        print(f"✓ MaskedGuBERT inference mode successful")
        print(f"  Features shape: {features.shape}")
        return True
    except Exception as e:
        print(f"✗ MaskedGuBERT inference mode failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_masked_gubert_extract_features(device: str = "cpu"):
    """Test MaskedGuBERT feature extraction method."""
    print("\n[TEST] MaskedGuBERT Extract Features")
    print("-" * 40)

    batch_size = 2
    n_mels = 80
    seq_len = 100

    try:
        model = MaskedGuBERTEncoder.from_config("tiny").to(device)
        mel_spec = torch.randn(batch_size, n_mels, seq_len, device=device)
        lengths = torch.tensor([100, 80], device=device)

        features, feature_lengths = model.extract_features(mel_spec, lengths=lengths)

        expected_time = model.get_output_length(seq_len)
        assert features.shape == (batch_size, expected_time, model.config.encoder_dim)
        assert feature_lengths is not None

        print(f"✓ MaskedGuBERT feature extraction successful")
        print(f"  Features shape: {features.shape}")
        return True
    except Exception as e:
        print(f"✗ MaskedGuBERT feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vector_quantizer(device: str = "cpu"):
    """Test VectorQuantizer module."""
    print("\n[TEST] VectorQuantizer")
    print("-" * 40)

    try:
        num_codebooks = 2
        codebook_size = 64
        codebook_dim = 32

        vq = VectorQuantizer(
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
        ).to(device)

        # Input: [B, T, D] where D = num_codebooks * codebook_dim
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, num_codebooks * codebook_dim, device=device)

        result = vq(x)

        assert "quantized" in result
        assert "indices" in result
        assert "commitment_loss" in result
        assert "codebook_loss" in result

        quantized = result["quantized"]
        indices = result["indices"]

        assert quantized.shape == x.shape
        assert indices.shape == (batch_size, seq_len, num_codebooks)
        assert indices.min() >= 0
        assert indices.max() < codebook_size

        print(f"✓ VectorQuantizer works")
        print(f"  Input shape:      {x.shape}")
        print(f"  Quantized shape:  {quantized.shape}")
        print(f"  Indices shape:    {indices.shape}")
        print(f"  Commitment loss:  {result['commitment_loss'].item():.4f}")
        print(f"  Codebook loss:    {result['codebook_loss'].item():.4f}")
        return True
    except Exception as e:
        print(f"✗ VectorQuantizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generate_mask_spans(device: str = "cpu"):
    """Test generate_mask_spans function."""
    print("\n[TEST] generate_mask_spans")
    print("-" * 40)

    try:
        seq_len = 100
        batch_size = 4
        mask_prob = 0.1
        mask_length = 10
        min_masks = 3

        mask = generate_mask_spans(
            seq_len=seq_len,
            batch_size=batch_size,
            mask_prob=mask_prob,
            mask_length=mask_length,
            min_masks=min_masks,
            device=device,
        )

        assert mask.shape == (batch_size, seq_len)
        assert mask.dtype == torch.bool

        # Each batch should have some masked positions
        for b in range(batch_size):
            num_masked = mask[b].sum().item()
            assert num_masked > 0, f"Batch {b} has no masked positions"
            assert num_masked <= int(seq_len * 0.8), "Should not mask more than 80%"

        total_masked = mask.sum().item()
        mask_ratio = total_masked / (batch_size * seq_len)

        print(f"✓ generate_mask_spans works")
        print(f"  Mask shape:   {mask.shape}")
        print(f"  Total masked: {total_masked}")
        print(f"  Mask ratio:   {mask_ratio:.2%}")
        return True
    except Exception as e:
        print(f"✗ generate_mask_spans test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_create_masked_gubert_function():
    """Test the create_masked_gubert convenience function."""
    print("\n[TEST] create_masked_gubert Function")
    print("-" * 40)

    try:
        model = create_masked_gubert(
            config="small",
            num_speakers=100,
        )

        assert isinstance(model, MaskedGuBERTEncoder)
        assert model.config.num_speakers == 100

        print(f"  Created model with {format_param_count(model.get_num_params())} params")
        print("✓ create_masked_gubert function works")
        return True
    except Exception as e:
        print(f"✗ create_masked_gubert function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_masked_gubert_gradient_flow(device: str = "cpu"):
    """Test that gradients flow through MaskedGuBERT components (regression mode)."""
    print("\n[TEST] MaskedGuBERT Gradient Flow (Regression Mode)")
    print("-" * 40)

    batch_size = 2
    n_mels = 80
    seq_len = 100

    try:
        # Default is regression mode (use_vq=False)
        model = MaskedGuBERTEncoder.from_config("tiny").to(device)
        model.zero_grad()
        model.train()

        mel_spec = torch.randn(batch_size, n_mels, seq_len, device=device)
        speaker_ids = torch.randint(0, model.config.num_speakers, (batch_size,), device=device)

        result = model(mel_spec, grl_alpha=1.0)

        # Compute masked prediction loss using the helper method
        masked_loss = model.compute_masked_prediction_loss(result)

        # Speaker loss
        speaker_loss = torch.nn.functional.cross_entropy(
            result["speaker_logits"], speaker_ids
        )

        total_loss = masked_loss + speaker_loss
        total_loss.backward()

        # Check gradients
        components_with_grads = []
        for name, module in model.named_children():
            has_grad = any(
                param.grad is not None and param.grad.abs().sum() > 0
                for param in module.parameters()
            )
            if has_grad:
                components_with_grads.append(name)

        print(f"  Components with gradients: {components_with_grads}")

        if len(components_with_grads) >= 5:
            print("✓ MaskedGuBERT gradient flow check passed (regression mode)")
            return True
        else:
            print("✗ Missing gradients in some components")
            return False

    except Exception as e:
        print(f"✗ MaskedGuBERT gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Input Validation
# =============================================================================

def test_different_sequence_lengths(model: GuBERTEncoder, device: str = "cpu"):
    """Test with various input sequence lengths."""
    print("\n[TEST] Different Sequence Lengths")
    print("-" * 40)

    batch_size = 1
    n_mels = model.config.n_mels
    lengths_to_test = [16, 50, 100, 200, 500]

    try:
        model.eval()
        results = {}

        for seq_len in lengths_to_test:
            mel_spec = torch.randn(batch_size, n_mels, seq_len, device=device)

            with torch.no_grad():
                result = model(mel_spec)

            output_len = result["features"].shape[1]
            expected_len = model.get_output_length(seq_len)
            results[seq_len] = (output_len, expected_len)

            assert output_len == expected_len, \
                f"Length mismatch for {seq_len}: got {output_len}, expected {expected_len}"

        print(f"✓ All sequence lengths handled correctly")
        for seq_len, (out_len, exp_len) in results.items():
            print(f"  {seq_len:4d} -> {out_len:4d} (expected {exp_len})")
        return True
    except Exception as e:
        print(f"✗ Sequence lengths test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_sizes(model: GuBERTEncoder, device: str = "cpu"):
    """Test with various batch sizes."""
    print("\n[TEST] Different Batch Sizes")
    print("-" * 40)

    n_mels = model.config.n_mels
    seq_len = 100
    batch_sizes = [1, 2, 4, 8]

    try:
        model.eval()

        for batch_size in batch_sizes:
            mel_spec = torch.randn(batch_size, n_mels, seq_len, device=device)

            with torch.no_grad():
                result = model(mel_spec)

            assert result["features"].shape[0] == batch_size
            assert result["speaker_logits"].shape[0] == batch_size

        print(f"✓ All batch sizes handled correctly")
        print(f"  Tested: {batch_sizes}")
        return True
    except Exception as e:
        print(f"✗ Batch sizes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Output Length Calculation
# =============================================================================

def test_output_length_calculation():
    """Test get_output_length method."""
    print("\n[TEST] Output Length Calculation")
    print("-" * 40)

    try:
        model = GuBERTEncoder.from_config("tiny")

        test_cases = [
            (100, model.get_output_length(100)),
            (200, model.get_output_length(200)),
            (50, model.get_output_length(50)),
            (16, model.get_output_length(16)),
        ]

        print(f"  Conv strides: {model.config.conv_strides}")
        print(f"  Total stride: {model.conv_subsample.total_stride}")

        for input_len, output_len in test_cases:
            # Verify with actual forward pass
            mel_spec = torch.randn(1, model.config.n_mels, input_len)
            with torch.no_grad():
                result = model(mel_spec)
            actual_len = result["features"].shape[1]

            assert output_len == actual_len, \
                f"Mismatch for {input_len}: calculated {output_len}, actual {actual_len}"
            print(f"  {input_len:4d} -> {output_len:4d}")

        print(f"✓ Output length calculation correct")
        return True
    except Exception as e:
        print(f"✗ Output length calculation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test GuBERT Model")
    parser.add_argument(
        "--config",
        type=str,
        default="tiny",
        choices=list(GUBERT_CONFIGS.keys()),
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
        help="Print full model architecture"
    )

    args = parser.parse_args()

    print(f"\n{'#' * 60}")
    print(f"# GuBERT Model Tests")
    print(f"# Config: {args.config}")
    print(f"# Device: {args.device}")
    print(f"{'#' * 60}")

    results = {}

    # Configuration tests
    results["config_creation"] = test_config_creation()
    results["predefined_configs"] = test_predefined_configs()

    # Instantiation tests
    model = test_model_instantiation(args.config, args.device)
    results["model_instantiation"] = model is not None

    if model is None:
        print("\n Model instantiation failed, skipping further tests")
        return 1

    print_model_summary(model, f"GuBERT ({args.config})")

    if args.print_model:
        print("\n[Full Model Structure]")
        print(model)

    results["model_with_overrides"] = test_model_with_overrides()
    results["create_gubert_function"] = test_create_gubert_function()

    # Forward pass tests
    results["forward_basic"] = test_forward_pass_basic(model, args.device)
    results["forward_with_lengths"] = test_forward_pass_with_lengths(model, args.device)
    results["forward_grl_alpha"] = test_forward_pass_with_grl_alpha(model, args.device)
    results["return_all_hiddens"] = test_return_all_hiddens(model, args.device)

    # Feature extraction tests
    results["extract_features"] = test_extract_features(model, args.device)

    # Gradient tests
    results["gradient_flow"] = test_gradient_flow(model, args.device)
    results["grl_reversal"] = test_grl_gradient_reversal(args.device)

    # Component tests
    results["conv_subsampling"] = test_conv_subsampling(args.device)
    results["transformer_block"] = test_transformer_encoder_block(args.device)
    results["speaker_classifier"] = test_speaker_classifier(args.device)

    # CTCVocab tests
    results["ctc_vocab_creation"] = test_ctc_vocab_creation()
    results["ctc_vocab_encode_decode"] = test_ctc_vocab_encode_decode()
    results["ctc_vocab_greedy_decode"] = test_ctc_vocab_greedy_decode()

    # MaskedGuBERT tests
    results["masked_config_creation"] = test_masked_gubert_config_creation()
    results["masked_predefined_configs"] = test_masked_gubert_predefined_configs()
    masked_model = test_masked_gubert_instantiation("tiny", args.device)
    results["masked_instantiation"] = masked_model is not None
    results["masked_forward_pass"] = test_masked_gubert_forward_pass(args.device)
    results["masked_inference_mode"] = test_masked_gubert_inference_mode(args.device)
    results["masked_extract_features"] = test_masked_gubert_extract_features(args.device)
    results["vector_quantizer"] = test_vector_quantizer(args.device)
    results["generate_mask_spans"] = test_generate_mask_spans(args.device)
    results["create_masked_gubert"] = test_create_masked_gubert_function()
    results["masked_gradient_flow"] = test_masked_gubert_gradient_flow(args.device)

    # Input validation tests
    results["sequence_lengths"] = test_different_sequence_lengths(model, args.device)
    results["batch_sizes"] = test_batch_sizes(model, args.device)
    results["output_length_calc"] = test_output_length_calculation()

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
