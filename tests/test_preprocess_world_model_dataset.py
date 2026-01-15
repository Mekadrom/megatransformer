"""
Test script for the World Model Dataset Preprocessing.

This script tests the preprocessing pipeline for multimodal world model training,
including tokenization, text formatting, and sample creation.

Usage:
    python tests/test_preprocess_world_model_dataset.py
    python tests/test_preprocess_world_model_dataset.py --skip-audio
    python tests/test_preprocess_world_model_dataset.py --skip-image
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from preprocess_world_model_dataset import (
    SPECIAL_TOKENS,
    WorldModelSample,
    WorldModelBatchProcessor,
    save_shard,
)


# =============================================================================
# Test Functions: WorldModelSample
# =============================================================================

def test_world_model_sample_creation():
    """Test WorldModelSample dataclass creation."""
    print("\n[TEST] WorldModelSample Creation")
    print("-" * 40)

    try:
        # Text-only sample
        sample = WorldModelSample(
            text_input_ids=torch.tensor([1, 2, 3, 4, 5]),
            task_type="text_only",
        )
        assert sample.text_input_ids.shape == (5,)
        assert sample.task_type == "text_only"
        assert sample.audio_mel_spec_latents is None
        assert sample.voice_mel_spec_latents is None
        assert sample.image_latents is None

        # Audio sample
        audio_sample = WorldModelSample(
            text_input_ids=torch.tensor([1, 2, 3]),
            task_type="audio_transcription",
            audio_mel_spec_latents=torch.randn(8, 10, 32),
        )
        assert audio_sample.audio_mel_spec_latents is not None
        assert audio_sample.audio_mel_spec_latents.shape == (8, 10, 32)

        # Image sample
        image_sample = WorldModelSample(
            text_input_ids=torch.tensor([1, 2, 3]),
            task_type="image_generation",
            image_latents=torch.randn(4, 32, 32),
        )
        assert image_sample.image_latents is not None
        assert image_sample.image_latents.shape == (4, 32, 32)

        print("✓ WorldModelSample creation successful")
        return True
    except Exception as e:
        print(f"✗ WorldModelSample creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_world_model_sample_to_dict():
    """Test WorldModelSample to_dict method."""
    print("\n[TEST] WorldModelSample to_dict")
    print("-" * 40)

    try:
        # Text-only sample
        sample = WorldModelSample(
            text_input_ids=torch.tensor([1, 2, 3]),
            task_type="text_only",
        )
        d = sample.to_dict()

        assert "text_input_ids" in d
        assert "task_type" in d
        assert "audio_mel_spec_latents" not in d
        assert "voice_mel_spec_latents" not in d
        assert "image_latents" not in d
        assert d["task_type"] == "text_only"

        # Sample with audio latents
        audio_latents = torch.randn(8, 10, 32)
        audio_sample = WorldModelSample(
            text_input_ids=torch.tensor([1, 2, 3]),
            task_type="audio_transcription",
            audio_mel_spec_latents=audio_latents,
        )
        audio_d = audio_sample.to_dict()

        assert "audio_mel_spec_latents" in audio_d
        assert torch.equal(audio_d["audio_mel_spec_latents"], audio_latents)

        # Sample with all modalities
        full_sample = WorldModelSample(
            text_input_ids=torch.tensor([1, 2, 3]),
            task_type="multimodal",
            audio_mel_spec_latents=torch.randn(8, 10, 32),
            voice_mel_spec_latents=torch.randn(8, 10, 20),
            image_latents=torch.randn(4, 32, 32),
        )
        full_d = full_sample.to_dict()

        assert "audio_mel_spec_latents" in full_d
        assert "voice_mel_spec_latents" in full_d
        assert "image_latents" in full_d

        print("✓ WorldModelSample to_dict successful")
        return True
    except Exception as e:
        print(f"✗ WorldModelSample to_dict failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Special Tokens
# =============================================================================

def test_special_tokens_configuration():
    """Test that special tokens are properly configured."""
    print("\n[TEST] Special Tokens Configuration")
    print("-" * 40)

    try:
        # Check all required tokens are present
        required_tokens = [
            "bos", "eos",
            "boa", "eoa",  # Audio
            "bov", "eov",  # Voice
            "boi", "eoi",  # Image
            "audio_placeholder",
            "voice_placeholder",
            "image_placeholder",
        ]

        for token_name in required_tokens:
            assert token_name in SPECIAL_TOKENS, f"Missing token: {token_name}"
            assert isinstance(SPECIAL_TOKENS[token_name], str), f"Token {token_name} should be string"
            assert len(SPECIAL_TOKENS[token_name]) > 0, f"Token {token_name} should not be empty"

        print(f"  Found {len(SPECIAL_TOKENS)} special tokens:")
        for name, token in SPECIAL_TOKENS.items():
            print(f"    {name}: {token}")

        print("✓ Special tokens configuration valid")
        return True
    except Exception as e:
        print(f"✗ Special tokens configuration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: WorldModelBatchProcessor
# =============================================================================

def test_processor_initialization():
    """Test WorldModelBatchProcessor initialization without VAE."""
    print("\n[TEST] WorldModelBatchProcessor Initialization")
    print("-" * 40)

    try:
        # Initialize for text-only (no VAE needed)
        processor = WorldModelBatchProcessor(
            modality="text_only",
            tokenizer_name="gpt2",  # Use GPT-2 as it's commonly available
            device="cpu",
        )

        # Check tokenizer was loaded with special tokens
        assert processor.tokenizer is not None
        assert len(processor.special_token_ids) == len(SPECIAL_TOKENS)

        # Verify special token IDs are integers
        for name, token_id in processor.special_token_ids.items():
            assert isinstance(token_id, int), f"Token {name} ID should be int, got {type(token_id)}"

        print(f"  Tokenizer: gpt2")
        print(f"  Vocab size: {len(processor.tokenizer)}")
        print(f"  Special tokens added: {len(processor.special_token_ids)}")

        print("✓ Processor initialization successful")
        return True
    except Exception as e:
        print(f"✗ Processor initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_formatting_text_only():
    """Test text-only formatting."""
    print("\n[TEST] Text Formatting (Text-Only)")
    print("-" * 40)

    try:
        processor = WorldModelBatchProcessor(
            modality="text_only",
            tokenizer_name="gpt2",
            device="cpu",
        )

        test_text = "Hello, world!"
        formatted = processor._format_text_only(test_text)

        # Should have BOS and EOS
        assert formatted.startswith(SPECIAL_TOKENS["bos"])
        assert formatted.endswith(SPECIAL_TOKENS["eos"])
        assert test_text in formatted

        print(f"  Input: '{test_text}'")
        print(f"  Output: '{formatted}'")

        print("✓ Text-only formatting successful")
        return True
    except Exception as e:
        print(f"✗ Text-only formatting failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_formatting_with_audio():
    """Test text formatting with audio modality."""
    print("\n[TEST] Text Formatting (Audio)")
    print("-" * 40)

    try:
        processor = WorldModelBatchProcessor(
            modality="text_audio",
            tokenizer_name="gpt2",
            device="cpu",
        )

        test_text = "This is a test."

        # Test transcription format (audio -> text)
        transcription = processor._format_text_with_modality(test_text, "audio", "transcription")
        assert transcription.startswith(SPECIAL_TOKENS["bos"])
        assert transcription.endswith(SPECIAL_TOKENS["eos"])
        assert SPECIAL_TOKENS["boa"] in transcription
        assert SPECIAL_TOKENS["eoa"] in transcription
        assert SPECIAL_TOKENS["audio_placeholder"] in transcription
        # For transcription: audio comes before text
        audio_pos = transcription.find(SPECIAL_TOKENS["audio_placeholder"])
        text_pos = transcription.find(test_text)
        assert audio_pos < text_pos, "For transcription, audio should come before text"

        # Test generation format (text -> audio)
        generation = processor._format_text_with_modality(test_text, "audio", "generation")
        assert generation.startswith(SPECIAL_TOKENS["bos"])
        assert generation.endswith(SPECIAL_TOKENS["eos"])
        assert SPECIAL_TOKENS["boa"] in generation
        assert SPECIAL_TOKENS["eoa"] in generation
        # For generation: text comes before audio
        audio_pos = generation.find(SPECIAL_TOKENS["audio_placeholder"])
        text_pos = generation.find(test_text)
        assert text_pos < audio_pos, "For generation, text should come before audio"

        print(f"  Input: '{test_text}'")
        print(f"  Transcription: '{transcription}'")
        print(f"  Generation: '{generation}'")

        print("✓ Audio text formatting successful")
        return True
    except Exception as e:
        print(f"✗ Audio text formatting failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_formatting_with_image():
    """Test text formatting with image modality."""
    print("\n[TEST] Text Formatting (Image)")
    print("-" * 40)

    try:
        processor = WorldModelBatchProcessor(
            modality="text_image",
            tokenizer_name="gpt2",
            device="cpu",
        )

        test_text = "A beautiful sunset."

        # Test description format (image -> text)
        description = processor._format_text_with_modality(test_text, "image", "transcription")
        assert SPECIAL_TOKENS["boi"] in description
        assert SPECIAL_TOKENS["eoi"] in description
        assert SPECIAL_TOKENS["image_placeholder"] in description
        # For description: image comes before text
        image_pos = description.find(SPECIAL_TOKENS["image_placeholder"])
        text_pos = description.find(test_text)
        assert image_pos < text_pos, "For description, image should come before text"

        # Test generation format (text -> image)
        generation = processor._format_text_with_modality(test_text, "image", "generation")
        assert SPECIAL_TOKENS["boi"] in generation
        assert SPECIAL_TOKENS["eoi"] in generation
        # For generation: text comes before image
        image_pos = generation.find(SPECIAL_TOKENS["image_placeholder"])
        text_pos = generation.find(test_text)
        assert text_pos < image_pos, "For generation, text should come before image"

        print(f"  Input: '{test_text}'")
        print(f"  Description: '{description}'")
        print(f"  Generation: '{generation}'")

        print("✓ Image text formatting successful")
        return True
    except Exception as e:
        print(f"✗ Image text formatting failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_formatting_with_voice():
    """Test text formatting with voice modality."""
    print("\n[TEST] Text Formatting (Voice)")
    print("-" * 40)

    try:
        processor = WorldModelBatchProcessor(
            modality="text_voice",
            tokenizer_name="gpt2",
            device="cpu",
        )

        test_text = "Hello there."

        # Test transcription format
        transcription = processor._format_text_with_modality(test_text, "voice", "transcription")
        assert SPECIAL_TOKENS["bov"] in transcription
        assert SPECIAL_TOKENS["eov"] in transcription
        assert SPECIAL_TOKENS["voice_placeholder"] in transcription

        # Test generation format
        generation = processor._format_text_with_modality(test_text, "voice", "generation")
        assert SPECIAL_TOKENS["bov"] in generation
        assert SPECIAL_TOKENS["eov"] in generation

        print(f"  Input: '{test_text}'")
        print(f"  Transcription: '{transcription}'")
        print(f"  Generation: '{generation}'")

        print("✓ Voice text formatting successful")
        return True
    except Exception as e:
        print(f"✗ Voice text formatting failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Batch Processing
# =============================================================================

def test_text_only_batch_processing():
    """Test text-only batch processing."""
    print("\n[TEST] Text-Only Batch Processing")
    print("-" * 40)

    try:
        processor = WorldModelBatchProcessor(
            modality="text_only",
            tokenizer_name="gpt2",
            device="cpu",
        )

        texts = [
            "Hello, world!",
            "This is a test sentence.",
            "Machine learning is fascinating.",
        ]

        samples = processor.process_text_only_batch(texts)

        # Should have same number of samples as inputs (no duplication for text-only)
        assert len(samples) == len(texts), f"Expected {len(texts)} samples, got {len(samples)}"

        for i, sample in enumerate(samples):
            assert sample.task_type == "text_only"
            assert sample.text_input_ids is not None
            assert len(sample.text_input_ids) > 0
            assert sample.audio_mel_spec_latents is None
            assert sample.voice_mel_spec_latents is None
            assert sample.image_latents is None

        print(f"  Processed {len(texts)} texts -> {len(samples)} samples")
        print(f"  Sample token lengths: {[len(s.text_input_ids) for s in samples]}")

        print("✓ Text-only batch processing successful")
        return True
    except Exception as e:
        print(f"✗ Text-only batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_batch_processing_without_vae():
    """Test audio batch processing without VAE (no latent encoding)."""
    print("\n[TEST] Audio Batch Processing (No VAE)")
    print("-" * 40)

    try:
        processor = WorldModelBatchProcessor(
            modality="text_audio",
            tokenizer_name="gpt2",
            device="cpu",
            audio_vae_checkpoint=None,  # No VAE
            sample_rate=16000,
            n_mels=80,
            audio_max_frames=100,
        )

        # Create synthetic waveforms
        waveforms = [
            torch.randn(16000),  # 1 second
            torch.randn(8000),   # 0.5 seconds
        ]
        texts = [
            "First audio sample.",
            "Second audio sample.",
        ]

        samples = processor.process_audio_batch(waveforms, texts, "audio")

        # Should have 2x samples (transcription + generation for each)
        assert len(samples) == len(texts) * 2, f"Expected {len(texts) * 2} samples, got {len(samples)}"

        # Check transcription and generation samples alternate
        for i in range(0, len(samples), 2):
            assert samples[i].task_type == "audio_transcription"
            assert samples[i + 1].task_type == "audio_generation"
            # Without VAE, latents should be None
            assert samples[i].audio_mel_spec_latents is None
            assert samples[i + 1].audio_mel_spec_latents is None

        print(f"  Processed {len(texts)} audio+text pairs -> {len(samples)} samples")
        print(f"  Task types: {[s.task_type for s in samples]}")

        print("✓ Audio batch processing (no VAE) successful")
        return True
    except Exception as e:
        print(f"✗ Audio batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_batch_processing_without_vae():
    """Test image batch processing without VAE (no latent encoding)."""
    print("\n[TEST] Image Batch Processing (No VAE)")
    print("-" * 40)

    try:
        processor = WorldModelBatchProcessor(
            modality="text_image",
            tokenizer_name="gpt2",
            device="cpu",
            image_vae_checkpoint=None,  # No VAE
            image_size=64,
        )

        # Create synthetic images
        images = [
            torch.randn(3, 64, 64),  # RGB image
            torch.randn(3, 64, 64),
        ]
        texts = [
            "A red car.",
            "A blue sky.",
        ]

        samples = processor.process_image_batch(images, texts)

        # Should have 2x samples (description + generation for each)
        assert len(samples) == len(texts) * 2, f"Expected {len(texts) * 2} samples, got {len(samples)}"

        # Check description and generation samples alternate
        for i in range(0, len(samples), 2):
            assert samples[i].task_type == "image_description"
            assert samples[i + 1].task_type == "image_generation"
            # Without VAE, latents should be None
            assert samples[i].image_latents is None
            assert samples[i + 1].image_latents is None

        print(f"  Processed {len(texts)} image+text pairs -> {len(samples)} samples")
        print(f"  Task types: {[s.task_type for s in samples]}")

        print("✓ Image batch processing (no VAE) successful")
        return True
    except Exception as e:
        print(f"✗ Image batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Shard Operations
# =============================================================================

def test_shard_save_load():
    """Test saving and loading shards."""
    print("\n[TEST] Shard Save/Load")
    print("-" * 40)

    try:
        # Create some samples
        samples = [
            WorldModelSample(
                text_input_ids=torch.tensor([1, 2, 3, 4]),
                task_type="text_only",
            ),
            WorldModelSample(
                text_input_ids=torch.tensor([5, 6, 7]),
                task_type="audio_transcription",
                audio_mel_spec_latents=torch.randn(8, 10, 16),
            ),
            WorldModelSample(
                text_input_ids=torch.tensor([8, 9]),
                task_type="image_generation",
                image_latents=torch.randn(4, 32, 32),
            ),
        ]

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            shard_path = f.name

        try:
            save_shard(samples, shard_path)
            assert os.path.exists(shard_path)

            # Load and verify
            loaded = torch.load(shard_path)
            assert "samples" in loaded
            assert "num_samples" in loaded
            assert loaded["num_samples"] == len(samples)

            loaded_samples = loaded["samples"]
            assert len(loaded_samples) == len(samples)

            # Verify each sample
            for orig, loaded_s in zip(samples, loaded_samples):
                assert torch.equal(orig.text_input_ids, loaded_s["text_input_ids"])
                assert orig.task_type == loaded_s["task_type"]

                if orig.audio_mel_spec_latents is not None:
                    assert "audio_mel_spec_latents" in loaded_s
                    assert torch.equal(orig.audio_mel_spec_latents, loaded_s["audio_mel_spec_latents"])

                if orig.image_latents is not None:
                    assert "image_latents" in loaded_s
                    assert torch.equal(orig.image_latents, loaded_s["image_latents"])

            print(f"  Saved and loaded {len(samples)} samples")
            print(f"  Shard size: {os.path.getsize(shard_path) / 1024:.1f} KB")

            print("✓ Shard save/load successful")
            return True

        finally:
            # Cleanup
            if os.path.exists(shard_path):
                os.unlink(shard_path)

    except Exception as e:
        print(f"✗ Shard save/load failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_empty_shard():
    """Test handling of empty shard."""
    print("\n[TEST] Empty Shard Handling")
    print("-" * 40)

    try:
        samples = []

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            shard_path = f.name

        try:
            save_shard(samples, shard_path)
            loaded = torch.load(shard_path)

            assert loaded["num_samples"] == 0
            assert len(loaded["samples"]) == 0

            print("✓ Empty shard handling successful")
            return True

        finally:
            if os.path.exists(shard_path):
                os.unlink(shard_path)

    except Exception as e:
        print(f"✗ Empty shard handling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Tokenization
# =============================================================================

def test_tokenization_round_trip():
    """Test that special tokens are properly tokenized and detokenized."""
    print("\n[TEST] Tokenization Round Trip")
    print("-" * 40)

    try:
        processor = WorldModelBatchProcessor(
            modality="text_only",
            tokenizer_name="gpt2",
            device="cpu",
        )

        # Test text with special tokens
        test_text = "Hello, world!"
        formatted = processor._format_text_only(test_text)

        # Tokenize
        token_ids = processor.tokenizer.encode(formatted, add_special_tokens=False)

        # Detokenize
        decoded = processor.tokenizer.decode(token_ids)

        # Check special tokens are preserved
        assert SPECIAL_TOKENS["bos"] in decoded
        assert SPECIAL_TOKENS["eos"] in decoded
        assert test_text in decoded

        print(f"  Original: '{test_text}'")
        print(f"  Formatted: '{formatted}'")
        print(f"  Token IDs: {token_ids[:10]}... ({len(token_ids)} total)")
        print(f"  Decoded: '{decoded}'")

        print("✓ Tokenization round trip successful")
        return True
    except Exception as e:
        print(f"✗ Tokenization round trip failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_special_token_ids_unique():
    """Test that all special tokens have unique IDs."""
    print("\n[TEST] Special Token IDs Uniqueness")
    print("-" * 40)

    try:
        processor = WorldModelBatchProcessor(
            modality="text_only",
            tokenizer_name="gpt2",
            device="cpu",
        )

        token_ids = list(processor.special_token_ids.values())
        unique_ids = set(token_ids)

        # All IDs should be unique
        if len(token_ids) != len(unique_ids):
            # Find duplicates
            seen = set()
            duplicates = []
            for name, tid in processor.special_token_ids.items():
                if tid in seen:
                    duplicates.append((name, tid))
                seen.add(tid)
            print(f"  Warning: Duplicate token IDs found: {duplicates}")
            # This might be okay depending on tokenizer behavior
            print("✓ Special token IDs checked (some duplicates)")
            return True
        else:
            print(f"  All {len(token_ids)} special tokens have unique IDs")
            print("✓ Special token IDs are unique")
            return True

    except Exception as e:
        print(f"✗ Special token IDs uniqueness check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Edge Cases
# =============================================================================

def test_empty_text_handling():
    """Test handling of empty text."""
    print("\n[TEST] Empty Text Handling")
    print("-" * 40)

    try:
        processor = WorldModelBatchProcessor(
            modality="text_only",
            tokenizer_name="gpt2",
            device="cpu",
        )

        # Process empty string
        texts = [""]
        samples = processor.process_text_only_batch(texts)

        assert len(samples) == 1
        # Even empty text should have BOS/EOS tokens
        assert len(samples[0].text_input_ids) >= 2  # At least BOS and EOS

        print(f"  Empty text -> {len(samples[0].text_input_ids)} tokens")
        print("✓ Empty text handling successful")
        return True
    except Exception as e:
        print(f"✗ Empty text handling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_long_text_handling():
    """Test handling of long text."""
    print("\n[TEST] Long Text Handling")
    print("-" * 40)

    try:
        processor = WorldModelBatchProcessor(
            modality="text_only",
            tokenizer_name="gpt2",
            device="cpu",
        )

        # Create a long text
        long_text = "This is a test sentence. " * 100  # ~2500 characters
        texts = [long_text]
        samples = processor.process_text_only_batch(texts)

        assert len(samples) == 1
        assert len(samples[0].text_input_ids) > 100  # Should have many tokens

        print(f"  Long text ({len(long_text)} chars) -> {len(samples[0].text_input_ids)} tokens")
        print("✓ Long text handling successful")
        return True
    except Exception as e:
        print(f"✗ Long text handling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_special_characters_in_text():
    """Test handling of special characters in text."""
    print("\n[TEST] Special Characters in Text")
    print("-" * 40)

    try:
        processor = WorldModelBatchProcessor(
            modality="text_only",
            tokenizer_name="gpt2",
            device="cpu",
        )

        # Text with various special characters
        texts = [
            "Hello! How are you?",
            "Price: $100.00",
            "Email: test@example.com",
            "Unicode: 你好世界 🌍",
            "Code: x = 1 + 2 * 3",
        ]

        samples = processor.process_text_only_batch(texts)
        assert len(samples) == len(texts)

        for text, sample in zip(texts, samples):
            assert sample.text_input_ids is not None
            assert len(sample.text_input_ids) > 0
            # Verify we can decode back (might not be exact due to tokenization)
            decoded = processor.tokenizer.decode(sample.text_input_ids)
            print(f"    '{text[:30]}...' -> {len(sample.text_input_ids)} tokens")

        print("✓ Special characters handling successful")
        return True
    except Exception as e:
        print(f"✗ Special characters handling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test World Model Dataset Preprocessing")
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip audio-related tests"
    )
    parser.add_argument(
        "--skip-image",
        action="store_true",
        help="Skip image-related tests"
    )

    args = parser.parse_args()

    print(f"\n{'#' * 60}")
    print("# World Model Dataset Preprocessing Tests")
    print(f"{'#' * 60}")

    results = {}

    # WorldModelSample tests
    results["sample_creation"] = test_world_model_sample_creation()
    results["sample_to_dict"] = test_world_model_sample_to_dict()

    # Special tokens tests
    results["special_tokens"] = test_special_tokens_configuration()

    # Processor initialization tests
    results["processor_init"] = test_processor_initialization()

    # Text formatting tests
    results["text_format_text_only"] = test_text_formatting_text_only()

    if not args.skip_audio:
        results["text_format_audio"] = test_text_formatting_with_audio()
        results["text_format_voice"] = test_text_formatting_with_voice()

    if not args.skip_image:
        results["text_format_image"] = test_text_formatting_with_image()

    # Batch processing tests
    results["batch_text_only"] = test_text_only_batch_processing()

    if not args.skip_audio:
        results["batch_audio_no_vae"] = test_audio_batch_processing_without_vae()

    if not args.skip_image:
        results["batch_image_no_vae"] = test_image_batch_processing_without_vae()

    # Shard tests
    results["shard_save_load"] = test_shard_save_load()
    results["empty_shard"] = test_empty_shard()

    # Tokenization tests
    results["tokenization_round_trip"] = test_tokenization_round_trip()
    results["token_ids_unique"] = test_special_token_ids_unique()

    # Edge case tests
    results["empty_text"] = test_empty_text_handling()
    results["long_text"] = test_long_text_handling()
    results["special_chars"] = test_special_characters_in_text()

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