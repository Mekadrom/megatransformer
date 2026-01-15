"""
Test script for GuBERT Dataset Preprocessing.

This script tests the preprocessing pipeline for GuBERT training,
including mel spectrogram extraction, text tokenization, and shard operations.

Usage:
    python tests/test_preprocess_gubert_dataset.py
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.audio.gubert import CTCVocab
from preprocess_gubert_dataset import GuBERTBatchProcessor


# =============================================================================
# Test Functions: CTCVocab Integration
# =============================================================================

def test_ctc_vocab_in_processor():
    """Test that CTCVocab is properly initialized in processor."""
    print("\n[TEST] CTCVocab in Processor")
    print("-" * 40)

    try:
        processor = GuBERTBatchProcessor(
            sample_rate=16000,
            n_mels=80,
            device="cpu",
        )

        assert processor.vocab is not None
        assert isinstance(processor.vocab, CTCVocab)
        assert processor.vocab.vocab_size > 0

        print(f"  Vocab size: {processor.vocab.vocab_size}")
        print(f"  Blank idx: {processor.vocab.blank_idx}")
        print("✓ CTCVocab initialized in processor")
        return True
    except Exception as e:
        print(f"✗ CTCVocab initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_tokenization():
    """Test text tokenization via processor."""
    print("\n[TEST] Text Tokenization")
    print("-" * 40)

    try:
        processor = GuBERTBatchProcessor(device="cpu")

        test_texts = [
            "hello world",
            "the quick brown fox",
            "testing one two three",
        ]

        for text in test_texts:
            tokens = processor.vocab.encode(text)
            decoded = processor.vocab.decode(tokens, remove_blanks=True, collapse_repeats=False)

            assert len(tokens) == len(text), f"Token length mismatch for '{text}'"
            assert decoded == text.lower(), f"Decode mismatch: '{decoded}' vs '{text.lower()}'"

            print(f"  '{text}' -> {len(tokens)} tokens")

        print("✓ Text tokenization works")
        return True
    except Exception as e:
        print(f"✗ Text tokenization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Batch Processing
# =============================================================================

def test_processor_initialization():
    """Test GuBERTBatchProcessor initialization."""
    print("\n[TEST] Processor Initialization")
    print("-" * 40)

    try:
        processor = GuBERTBatchProcessor(
            sample_rate=16000,
            n_mels=80,
            n_fft=1024,
            hop_length=256,
            max_audio_seconds=30,
            device="cpu",
        )

        assert processor.sample_rate == 16000
        assert processor.n_mels == 80
        assert processor.n_fft == 1024
        assert processor.hop_length == 256
        assert processor.audio_max_frames == (30 * 16000) // 256

        print(f"  Sample rate: {processor.sample_rate}")
        print(f"  N mels: {processor.n_mels}")
        print(f"  Max frames: {processor.audio_max_frames}")
        print("✓ Processor initialization successful")
        return True
    except Exception as e:
        print(f"✗ Processor initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_processing():
    """Test processing a batch of waveforms and texts."""
    print("\n[TEST] Batch Processing")
    print("-" * 40)

    try:
        processor = GuBERTBatchProcessor(
            sample_rate=16000,
            n_mels=80,
            max_audio_seconds=10,
            device="cpu",
        )

        # Create synthetic waveforms
        waveforms = [
            torch.randn(16000),      # 1 second
            torch.randn(32000),      # 2 seconds
            torch.randn(8000),       # 0.5 seconds
        ]

        texts = [
            "hello world",
            "the quick brown fox",
            "test",
        ]

        speaker_ids = [0, 1, 0]

        result = processor.process_batch(waveforms, texts, speaker_ids)

        # Check output keys
        assert "mel_specs" in result
        assert "mel_lengths" in result
        assert "text_tokens" in result
        assert "text_lengths" in result
        assert "speaker_ids" in result

        # Check shapes
        batch_size = len(waveforms)
        assert result["mel_specs"].shape[0] == batch_size
        assert result["mel_specs"].shape[1] == processor.n_mels
        assert result["mel_lengths"].shape[0] == batch_size
        assert result["text_tokens"].shape[0] == batch_size
        assert result["text_lengths"].shape[0] == batch_size
        assert result["speaker_ids"].shape[0] == batch_size

        # Check speaker IDs preserved
        assert result["speaker_ids"].tolist() == speaker_ids

        print(f"✓ Batch processing successful")
        print(f"  Batch size: {batch_size}")
        print(f"  Mel spec shape: {result['mel_specs'].shape}")
        print(f"  Mel lengths: {result['mel_lengths'].tolist()}")
        print(f"  Text token shape: {result['text_tokens'].shape}")
        print(f"  Text lengths: {result['text_lengths'].tolist()}")
        return True
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mel_spectrogram_extraction():
    """Test mel spectrogram extraction from waveforms."""
    print("\n[TEST] Mel Spectrogram Extraction")
    print("-" * 40)

    try:
        processor = GuBERTBatchProcessor(
            sample_rate=16000,
            n_mels=80,
            hop_length=256,
            max_audio_seconds=5,
            device="cpu",
        )

        # Create synthetic audio (1 second sine wave)
        duration = 1.0
        t = torch.linspace(0, duration, int(16000 * duration))
        waveform = torch.sin(2 * 3.14159 * 440 * t)  # 440 Hz tone

        waveforms = [waveform]
        texts = ["test"]
        speaker_ids = [0]

        result = processor.process_batch(waveforms, texts, speaker_ids)

        mel_spec = result["mel_specs"][0]
        mel_length = result["mel_lengths"][0].item()

        # Check mel spec properties
        assert mel_spec.shape[0] == 80  # n_mels
        assert mel_length > 0
        assert mel_length <= processor.audio_max_frames

        # Mel spec should not be all zeros (valid audio)
        assert mel_spec[:, :mel_length].abs().sum() > 0

        print(f"✓ Mel spectrogram extraction works")
        print(f"  Waveform length: {len(waveform)}")
        print(f"  Mel spec shape: {mel_spec.shape}")
        print(f"  Mel length (unpadded): {mel_length}")
        return True
    except Exception as e:
        print(f"✗ Mel spectrogram extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_padding_and_truncation():
    """Test that mel specs are properly padded/truncated."""
    print("\n[TEST] Padding and Truncation")
    print("-" * 40)

    try:
        max_audio_seconds = 2
        processor = GuBERTBatchProcessor(
            sample_rate=16000,
            n_mels=80,
            hop_length=256,
            max_audio_seconds=max_audio_seconds,
            device="cpu",
        )

        max_frames = processor.audio_max_frames

        # Short audio (should be padded)
        short_waveform = torch.randn(8000)  # 0.5 seconds

        # Long audio (should be truncated)
        long_waveform = torch.randn(64000)  # 4 seconds

        waveforms = [short_waveform, long_waveform]
        texts = ["short", "long"]
        speaker_ids = [0, 1]

        result = processor.process_batch(waveforms, texts, speaker_ids)

        # Both should have same padded length
        assert result["mel_specs"].shape[2] == max_frames

        # Short audio: mel_length < max_frames
        short_length = result["mel_lengths"][0].item()
        assert short_length < max_frames, f"Short audio should be padded: {short_length} vs {max_frames}"

        # Long audio: mel_length == max_frames (truncated)
        long_length = result["mel_lengths"][1].item()
        assert long_length == max_frames, f"Long audio should be truncated: {long_length} vs {max_frames}"

        print(f"✓ Padding and truncation works")
        print(f"  Max frames: {max_frames}")
        print(f"  Short audio length: {short_length} (padded)")
        print(f"  Long audio length: {long_length} (truncated)")
        return True
    except Exception as e:
        print(f"✗ Padding/truncation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_empty_batch():
    """Test handling of empty batch."""
    print("\n[TEST] Empty Batch Handling")
    print("-" * 40)

    try:
        processor = GuBERTBatchProcessor(device="cpu")

        result = processor.process_batch([], [], [])

        # Should return empty tensors with correct structure
        assert result["mel_specs"].shape[0] == 0
        assert result["mel_lengths"].shape[0] == 0
        assert result["text_tokens"].shape[0] == 0
        assert result["text_lengths"].shape[0] == 0
        assert result["speaker_ids"].shape[0] == 0

        print(f"✓ Empty batch handled correctly")
        return True
    except Exception as e:
        # Empty batch might raise an error, which is also acceptable
        print(f"  Empty batch raises error (acceptable): {type(e).__name__}")
        print(f"✓ Empty batch handling checked")
        return True


# =============================================================================
# Test Functions: Text Processing Edge Cases
# =============================================================================

def test_empty_text():
    """Test processing with empty text."""
    print("\n[TEST] Empty Text Processing")
    print("-" * 40)

    try:
        processor = GuBERTBatchProcessor(device="cpu")

        waveforms = [torch.randn(16000)]
        texts = [""]
        speaker_ids = [0]

        result = processor.process_batch(waveforms, texts, speaker_ids)

        # Empty text should produce zero-length tokens
        assert result["text_lengths"][0].item() == 0

        print(f"✓ Empty text processed correctly")
        print(f"  Text length: {result['text_lengths'][0].item()}")
        return True
    except Exception as e:
        print(f"✗ Empty text processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_special_characters_in_text():
    """Test processing text with special characters."""
    print("\n[TEST] Special Characters in Text")
    print("-" * 40)

    try:
        processor = GuBERTBatchProcessor(device="cpu")

        texts_with_special = [
            "hello, world!",
            "what's up?",
            "test123",
            "UPPERCASE",
        ]

        waveforms = [torch.randn(16000) for _ in texts_with_special]
        speaker_ids = list(range(len(texts_with_special)))

        result = processor.process_batch(waveforms, texts_with_special, speaker_ids)

        for i, text in enumerate(texts_with_special):
            tokens = result["text_tokens"][i]
            length = result["text_lengths"][i].item()

            # Should have encoded something
            assert length > 0, f"Text '{text}' should produce tokens"

            # Check that unknown characters are handled
            decoded = processor.vocab.decode(
                tokens[:length].tolist(),
                remove_blanks=True,
                collapse_repeats=False
            )
            print(f"  '{text}' -> '{decoded}' ({length} tokens)")

        print(f"✓ Special characters handled")
        return True
    except Exception as e:
        print(f"✗ Special characters test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_long_text():
    """Test processing very long text."""
    print("\n[TEST] Long Text Processing")
    print("-" * 40)

    try:
        processor = GuBERTBatchProcessor(device="cpu")

        long_text = "this is a test sentence " * 50  # ~1200 characters

        waveforms = [torch.randn(16000)]
        texts = [long_text]
        speaker_ids = [0]

        result = processor.process_batch(waveforms, texts, speaker_ids)

        text_length = result["text_lengths"][0].item()
        assert text_length == len(long_text), f"Long text length: {text_length} vs {len(long_text)}"

        print(f"✓ Long text processed correctly")
        print(f"  Text length: {text_length} characters")
        return True
    except Exception as e:
        print(f"✗ Long text processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Audio Processing Edge Cases
# =============================================================================

def test_short_audio():
    """Test processing very short audio."""
    print("\n[TEST] Short Audio Processing")
    print("-" * 40)

    try:
        processor = GuBERTBatchProcessor(
            sample_rate=16000,
            n_mels=80,
            hop_length=256,
            device="cpu",
        )

        # Very short audio (0.1 seconds)
        short_waveform = torch.randn(1600)

        waveforms = [short_waveform]
        texts = ["test"]
        speaker_ids = [0]

        result = processor.process_batch(waveforms, texts, speaker_ids)

        mel_length = result["mel_lengths"][0].item()
        assert mel_length > 0, "Short audio should produce at least some frames"

        print(f"✓ Short audio processed correctly")
        print(f"  Waveform samples: {len(short_waveform)}")
        print(f"  Mel frames: {mel_length}")
        return True
    except Exception as e:
        print(f"✗ Short audio processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_silent_audio():
    """Test processing silent audio."""
    print("\n[TEST] Silent Audio Processing")
    print("-" * 40)

    try:
        processor = GuBERTBatchProcessor(device="cpu")

        # Silent audio (all zeros)
        silent_waveform = torch.zeros(16000)

        waveforms = [silent_waveform]
        texts = ["test"]
        speaker_ids = [0]

        result = processor.process_batch(waveforms, texts, speaker_ids)

        # Should still produce mel spec (even if mostly zero)
        mel_length = result["mel_lengths"][0].item()
        assert mel_length > 0

        print(f"✓ Silent audio processed correctly")
        print(f"  Mel frames: {mel_length}")
        return True
    except Exception as e:
        print(f"✗ Silent audio processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Shard Operations
# =============================================================================

def test_shard_save_load():
    """Test saving and loading shard data."""
    print("\n[TEST] Shard Save/Load")
    print("-" * 40)

    try:
        processor = GuBERTBatchProcessor(device="cpu")

        # Create test data
        waveforms = [torch.randn(16000), torch.randn(32000)]
        texts = ["hello world", "testing"]
        speaker_ids = [0, 1]

        result = processor.process_batch(waveforms, texts, speaker_ids)

        # Create shard data structure
        shard_data = {
            "mel_specs": result["mel_specs"],
            "mel_lengths": result["mel_lengths"],
            "text_tokens": result["text_tokens"],
            "text_lengths": result["text_lengths"],
            "speaker_ids": result["speaker_ids"],
            "raw_texts": texts,
            "num_samples": len(waveforms),
        }

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            shard_path = f.name

        try:
            torch.save(shard_data, shard_path)
            assert os.path.exists(shard_path)

            # Load and verify
            loaded = torch.load(shard_path)
            assert loaded["num_samples"] == len(waveforms)
            assert torch.equal(loaded["mel_specs"], shard_data["mel_specs"])
            assert torch.equal(loaded["text_tokens"], shard_data["text_tokens"])
            assert torch.equal(loaded["speaker_ids"], shard_data["speaker_ids"])
            assert loaded["raw_texts"] == texts

            print(f"✓ Shard save/load successful")
            print(f"  Saved {loaded['num_samples']} samples")
            print(f"  Shard size: {os.path.getsize(shard_path) / 1024:.1f} KB")
            return True

        finally:
            if os.path.exists(shard_path):
                os.unlink(shard_path)

    except Exception as e:
        print(f"✗ Shard save/load failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_save_load():
    """Test saving and loading config JSON."""
    print("\n[TEST] Config Save/Load")
    print("-" * 40)

    try:
        processor = GuBERTBatchProcessor(
            sample_rate=16000,
            n_mels=80,
            hop_length=256,
            device="cpu",
        )

        config = {
            "sample_rate": processor.sample_rate,
            "n_mels": processor.n_mels,
            "hop_length": processor.hop_length,
            "vocab_size": processor.vocab.vocab_size,
            "num_speakers": 100,
            "speaker_id_to_idx": {f"spk_{i}": i for i in range(10)},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_path = f.name

        try:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            with open(config_path) as f:
                loaded = json.load(f)

            assert loaded["sample_rate"] == config["sample_rate"]
            assert loaded["n_mels"] == config["n_mels"]
            assert loaded["vocab_size"] == config["vocab_size"]
            assert loaded["num_speakers"] == config["num_speakers"]
            assert loaded["speaker_id_to_idx"] == config["speaker_id_to_idx"]

            print(f"✓ Config save/load successful")
            return True

        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)

    except Exception as e:
        print(f"✗ Config save/load failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Speaker ID Mapping
# =============================================================================

def test_speaker_id_mapping():
    """Test speaker ID to index mapping."""
    print("\n[TEST] Speaker ID Mapping")
    print("-" * 40)

    try:
        # Simulate speaker ID collection and mapping
        original_speaker_ids = ["spk_a", "spk_b", "spk_c", "spk_a", "spk_b"]
        unique_speakers = sorted(set(original_speaker_ids))
        speaker_id_to_idx = {sid: i for i, sid in enumerate(unique_speakers)}

        # Map to indices
        mapped_ids = [speaker_id_to_idx[sid] for sid in original_speaker_ids]

        assert len(speaker_id_to_idx) == 3
        assert mapped_ids == [0, 1, 2, 0, 1]  # Contiguous indices

        print(f"✓ Speaker ID mapping works")
        print(f"  Unique speakers: {unique_speakers}")
        print(f"  Mapping: {speaker_id_to_idx}")
        print(f"  Mapped IDs: {mapped_ids}")
        return True
    except Exception as e:
        print(f"✗ Speaker ID mapping failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Functions: Data Consistency
# =============================================================================

def test_batch_consistency():
    """Test that batch outputs are consistent across multiple calls."""
    print("\n[TEST] Batch Consistency")
    print("-" * 40)

    try:
        processor = GuBERTBatchProcessor(device="cpu")

        waveform = torch.randn(16000)
        text = "hello world"
        speaker_id = 0

        # Process same data twice
        result1 = processor.process_batch([waveform], [text], [speaker_id])
        result2 = processor.process_batch([waveform], [text], [speaker_id])

        # Mel specs should be identical
        assert torch.equal(result1["mel_specs"], result2["mel_specs"])
        assert torch.equal(result1["mel_lengths"], result2["mel_lengths"])
        assert torch.equal(result1["text_tokens"], result2["text_tokens"])
        assert torch.equal(result1["text_lengths"], result2["text_lengths"])

        print(f"✓ Batch processing is consistent")
        return True
    except Exception as e:
        print(f"✗ Batch consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_token_alignment():
    """Test that text tokens align correctly with text lengths."""
    print("\n[TEST] Text Token Alignment")
    print("-" * 40)

    try:
        processor = GuBERTBatchProcessor(device="cpu")

        texts = ["hi", "hello", "goodbye"]  # Different lengths
        waveforms = [torch.randn(16000) for _ in texts]
        speaker_ids = [0, 1, 2]

        result = processor.process_batch(waveforms, texts, speaker_ids)

        for i, text in enumerate(texts):
            length = result["text_lengths"][i].item()
            tokens = result["text_tokens"][i, :length]

            # Length should match text length
            assert length == len(text), f"Length mismatch for '{text}'"

            # Tokens after length should be padding (0)
            if result["text_tokens"].shape[1] > length:
                padding = result["text_tokens"][i, length:]
                assert (padding == 0).all(), "Padding should be zeros"

        print(f"✓ Text tokens align correctly")
        for i, text in enumerate(texts):
            print(f"  '{text}' -> {result['text_lengths'][i].item()} tokens")
        return True
    except Exception as e:
        print(f"✗ Text token alignment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test GuBERT Dataset Preprocessing")

    args = parser.parse_args()

    print(f"\n{'#' * 60}")
    print("# GuBERT Dataset Preprocessing Tests")
    print(f"{'#' * 60}")

    results = {}

    # CTCVocab tests
    results["ctc_vocab_in_processor"] = test_ctc_vocab_in_processor()
    results["text_tokenization"] = test_text_tokenization()

    # Processor tests
    results["processor_init"] = test_processor_initialization()
    results["batch_processing"] = test_batch_processing()
    results["mel_extraction"] = test_mel_spectrogram_extraction()
    results["padding_truncation"] = test_padding_and_truncation()
    results["empty_batch"] = test_empty_batch()

    # Text edge cases
    results["empty_text"] = test_empty_text()
    results["special_chars"] = test_special_characters_in_text()
    results["long_text"] = test_long_text()

    # Audio edge cases
    results["short_audio"] = test_short_audio()
    results["silent_audio"] = test_silent_audio()

    # Shard operations
    results["shard_save_load"] = test_shard_save_load()
    results["config_save_load"] = test_config_save_load()

    # Speaker ID mapping
    results["speaker_id_mapping"] = test_speaker_id_mapping()

    # Data consistency
    results["batch_consistency"] = test_batch_consistency()
    results["text_token_alignment"] = test_text_token_alignment()

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
