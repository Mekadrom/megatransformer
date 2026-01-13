import pytest
import torch
from model.world.token_alignment import TokenInterleaver, TokenInterleaverConfig


# Define placeholder token IDs for testing
AUDIO_PLACEHOLDER_ID = 32000
VOICE_PLACEHOLDER_ID = 32001
IMAGE_PLACEHOLDER_ID = 32002


class TestTokenInterleaver:
    """Tests for TokenInterleaver module.

    Expected behavior based on architecture design:
    - Text hidden states contain placeholder tokens for media
    - Placeholder positions are detected from token IDs
    - Media hidden states replace these placeholders
    - Output interleaves text and media, skipping placeholder tokens
    """

    @pytest.fixture
    def config(self):
        return TokenInterleaverConfig(
            audio_placeholder_token_id=AUDIO_PLACEHOLDER_ID,
            voice_placeholder_token_id=VOICE_PLACEHOLDER_ID,
            image_placeholder_token_id=IMAGE_PLACEHOLDER_ID,
        )

    @pytest.fixture
    def interleaver(self, config):
        return TokenInterleaver(config)

    @pytest.fixture
    def d_model(self):
        return 64

    def test_text_only_passthrough(self, interleaver, d_model):
        """When no media is provided, should return text with correct shape."""
        batch_size = 2
        text_seq_len = 10
        # Text shape: (batch, seq_len, d_model) - no n_example dim since text is the driver
        text = torch.randn(batch_size, text_seq_len, d_model)
        # Token IDs with no placeholders
        token_ids = torch.randint(0, 1000, (batch_size, text_seq_len))

        tokens, attention_mask, modality_map = interleaver(
            text_hidden_states=text,
            text_token_ids=token_ids,
        )

        # Should return same shape for tokens
        assert tokens.shape == (batch_size, text_seq_len, d_model), \
            f"Expected shape {(batch_size, text_seq_len, d_model)}, got {tokens.shape}"
        # All positions should be attended
        assert attention_mask.all()
        # All positions should be text
        assert (modality_map == 0).all()  # MODALITY_TEXT = 0

    def test_single_audio_interleaving(self, interleaver, d_model):
        """Test interleaving text with a single audio example.

        Scenario:
        - Text: [BOS, BOA, <audio_placeholder>, EOA, text, text, EOS] (7 tokens)
        - Audio placeholder at position 2
        - Audio has 5 timesteps
        - Expected output: [BOS, BOA, audio*5, EOA, text, text, EOS] (11 tokens)
        """
        batch_size = 1
        text_seq_len = 7
        audio_seq_len = 5

        # Create identifiable tensors - text has no n_example dim
        text = torch.zeros(batch_size, text_seq_len, d_model)
        for i in range(text_seq_len):
            text[0, i, 0] = i  # Mark each position with its index

        # Token IDs with audio placeholder at position 2
        token_ids = torch.randint(0, 1000, (batch_size, text_seq_len))
        token_ids[0, 2] = AUDIO_PLACEHOLDER_ID

        # Audio has n_example dim: (batch, n_examples, seq_len, d_model)
        audio = torch.ones(batch_size, 1, audio_seq_len, d_model) * 100  # Distinct value
        audio_lengths = torch.tensor([[audio_seq_len]])

        tokens, attention_mask, modality_map = interleaver(
            text_hidden_states=text,
            text_token_ids=token_ids,
            audio_hidden_states=audio,
            audio_lengths=audio_lengths,
        )

        # Expected length: 7 - 1 (placeholder removed) + 5 (audio) = 11
        expected_len = text_seq_len - 1 + audio_seq_len
        assert tokens.shape == (batch_size, expected_len, d_model), \
            f"Expected shape {(batch_size, expected_len, d_model)}, got {tokens.shape}"

        # Check token ordering:
        # Positions 0-1: text tokens 0, 1 (BOS, BOA)
        assert tokens[0, 0, 0].item() == 0, "Position 0 should be text token 0"
        assert tokens[0, 1, 0].item() == 1, "Position 1 should be text token 1"

        # Positions 2-6: audio tokens (value 100)
        for i in range(2, 2 + audio_seq_len):
            assert tokens[0, i, 0].item() == 100, f"Position {i} should be audio"

        # Positions 7-10: text tokens 3, 4, 5, 6 (EOA, text, text, EOS)
        # Note: text token 2 (placeholder) is skipped
        assert tokens[0, 7, 0].item() == 3, "Position 7 should be text token 3 (EOA)"
        assert tokens[0, 8, 0].item() == 4, "Position 8 should be text token 4"
        assert tokens[0, 9, 0].item() == 5, "Position 9 should be text token 5"
        assert tokens[0, 10, 0].item() == 6, "Position 10 should be text token 6 (EOS)"

        # Check modality map
        assert (modality_map[0, :2] == 0).all()  # text
        assert (modality_map[0, 2:7] == 1).all()  # audio
        assert (modality_map[0, 7:] == 0).all()  # text

    def test_multiple_audio_interleaving(self, interleaver, d_model):
        """Test interleaving with two audio examples.

        Scenario from docstring:
        - Text has placeholders at positions 2 and 32 (in text-only indexing)
        - First audio: 150 timesteps
        - Second audio: 20 timesteps
        - 29 text tokens between placeholders (positions 3-31 in original)

        Expected:
        - text[0:2] + audio_0[150] + text[3:32] + audio_1[20] + text[33:]
        - Second audio starts at position 2 + 150 + 29 = 181 in interleaved
        """
        batch_size = 1
        text_seq_len = 35  # BOS, BOA, placeholder, 29 text, BOA, placeholder, EOA, text

        # Text: (batch, seq, d_model) - no n_example dim
        text = torch.zeros(batch_size, text_seq_len, d_model)
        for i in range(text_seq_len):
            text[0, i, 0] = i

        # Token IDs with audio placeholders at positions 2 and 32
        token_ids = torch.randint(0, 1000, (batch_size, text_seq_len))
        token_ids[0, 2] = AUDIO_PLACEHOLDER_ID
        token_ids[0, 32] = AUDIO_PLACEHOLDER_ID

        # Audio: (batch, n_examples, seq, d_model)
        audio = torch.zeros(batch_size, 2, 150, d_model)  # 2 examples, max 150 timesteps
        audio[0, 0, :, 0] = 1000  # First audio marker
        audio[0, 1, :20, 0] = 2000  # Second audio marker (only 20 timesteps used)

        audio_lengths = torch.tensor([[150, 20]])

        tokens, attention_mask, modality_map = interleaver(
            text_hidden_states=text,
            text_token_ids=token_ids,
            audio_hidden_states=audio,
            audio_lengths=audio_lengths,
        )

        # Expected length: 35 - 2 (placeholders) + 150 + 20 = 203
        expected_len = text_seq_len - 2 + 150 + 20
        assert tokens.shape == (batch_size, expected_len, d_model), \
            f"Expected shape {(batch_size, expected_len, d_model)}, got {tokens.shape}"

        # Check structure:
        # [0:2] = text 0, 1
        assert tokens[0, 0, 0].item() == 0
        assert tokens[0, 1, 0].item() == 1

        # [2:152] = first audio (value 1000)
        assert tokens[0, 2, 0].item() == 1000
        assert tokens[0, 151, 0].item() == 1000

        # [152:181] = text 3-31 (29 tokens, skipping placeholder at 2)
        assert tokens[0, 152, 0].item() == 3, "After first audio should be text token 3"
        assert tokens[0, 180, 0].item() == 31, "Should have text up to token 31"

        # [181:201] = second audio (value 2000)
        assert tokens[0, 181, 0].item() == 2000, "Second audio should start at 181"

        # [201:203] = text 33, 34 (skipping placeholder at 32)
        assert tokens[0, 201, 0].item() == 33, "After second audio should be text token 33"

    def test_image_fixed_size(self, interleaver, d_model):
        """Test that images don't need length parameter (fixed size)."""
        batch_size = 1
        text_seq_len = 10
        image_patch_count = 64  # e.g., 8x8 patches

        # Text: (batch, seq, d_model)
        text = torch.zeros(batch_size, text_seq_len, d_model)
        for i in range(text_seq_len):
            text[0, i, 0] = i

        # Token IDs with image placeholder at position 3
        token_ids = torch.randint(0, 1000, (batch_size, text_seq_len))
        token_ids[0, 3] = IMAGE_PLACEHOLDER_ID

        # Image: (batch, n_examples, patches, d_model)
        image = torch.ones(batch_size, 1, image_patch_count, d_model) * 500

        tokens, attention_mask, modality_map = interleaver(
            text_hidden_states=text,
            text_token_ids=token_ids,
            image_hidden_states=image,
        )

        # Expected: 10 - 1 + 64 = 73
        expected_len = text_seq_len - 1 + image_patch_count
        assert tokens.shape == (batch_size, expected_len, d_model)

        # Verify image is inserted at correct position
        assert tokens[0, 3, 0].item() == 500, "Image should start at position 3"
        assert tokens[0, 66, 0].item() == 500, "Image should end at position 66"
        assert tokens[0, 67, 0].item() == 4, "Text token 4 should follow image"

    def test_mixed_modalities(self, interleaver, d_model):
        """Test interleaving with audio, voice, and image together.

        Scenario:
        - Text: 20 tokens with placeholders at 2 (audio), 5 (voice), 10 (image)
        - Audio: 10 timesteps
        - Voice: 15 timesteps
        - Image: 16 patches
        """
        batch_size = 1
        text_seq_len = 20

        # Text: (batch, seq, d_model)
        text = torch.zeros(batch_size, text_seq_len, d_model)
        for i in range(text_seq_len):
            text[0, i, 0] = i

        # Token IDs with placeholders
        token_ids = torch.randint(0, 1000, (batch_size, text_seq_len))
        token_ids[0, 2] = AUDIO_PLACEHOLDER_ID
        token_ids[0, 5] = VOICE_PLACEHOLDER_ID
        token_ids[0, 10] = IMAGE_PLACEHOLDER_ID

        # Media: (batch, n_examples, seq, d_model)
        audio = torch.ones(batch_size, 1, 10, d_model) * 100
        voice = torch.ones(batch_size, 1, 15, d_model) * 200
        image = torch.ones(batch_size, 1, 16, d_model) * 300

        tokens, attention_mask, modality_map = interleaver(
            text_hidden_states=text,
            text_token_ids=token_ids,
            audio_hidden_states=audio,
            audio_lengths=torch.tensor([[10]]),
            voice_hidden_states=voice,
            voice_lengths=torch.tensor([[15]]),
            image_hidden_states=image,
        )

        # Expected: 20 - 3 (placeholders) + 10 + 15 + 16 = 58
        expected_len = text_seq_len - 3 + 10 + 15 + 16
        assert tokens.shape == (batch_size, expected_len, d_model)

        # Verify ordering: text[0:2], audio, text[3:5], voice, text[6:10], image, text[11:20]
        assert tokens[0, 0, 0].item() == 0  # text 0
        assert tokens[0, 1, 0].item() == 1  # text 1
        assert tokens[0, 2, 0].item() == 100  # audio starts
        assert tokens[0, 11, 0].item() == 100  # audio ends
        assert tokens[0, 12, 0].item() == 3  # text 3 (skipped placeholder at 2)
        assert tokens[0, 13, 0].item() == 4  # text 4

    def test_batch_padding(self, interleaver, d_model):
        """Test that batches with different lengths get padded correctly."""
        batch_size = 2
        text_seq_len = 10

        # Text: (batch, seq, d_model)
        text = torch.zeros(batch_size, text_seq_len, d_model)
        for b in range(batch_size):
            for i in range(text_seq_len):
                text[b, i, 0] = b * 100 + i

        # Token IDs - both have placeholder at position 2
        token_ids = torch.randint(0, 1000, (batch_size, text_seq_len))
        token_ids[0, 2] = AUDIO_PLACEHOLDER_ID
        token_ids[1, 2] = AUDIO_PLACEHOLDER_ID

        # Audio: (batch, n_examples, seq, d_model)
        # Batch 0: 5 audio tokens, Batch 1: 20 audio tokens
        audio = torch.zeros(batch_size, 1, 20, d_model)
        audio[0, 0, :, 0] = 1000
        audio[1, 0, :, 0] = 2000

        audio_lengths = torch.tensor([[5], [20]])

        tokens, attention_mask, modality_map = interleaver(
            text_hidden_states=text,
            text_token_ids=token_ids,
            audio_hidden_states=audio,
            audio_lengths=audio_lengths,
        )

        # Batch 0 length: 10 - 1 + 5 = 14
        # Batch 1 length: 10 - 1 + 20 = 29
        # Padded to max: 29
        assert tokens.shape == (batch_size, 29, d_model)

        # Batch 0 should have padding (zeros) at the end
        assert tokens[0, 14, 0].item() == 0, "Batch 0 should be padded after position 13"

        # Batch 1 should have content to the end
        assert tokens[1, 28, 0].item() != 0 or tokens[1, 28, 0].item() == 9, \
            "Batch 1 should have content at position 28"

        # Check attention mask
        assert attention_mask[0, :14].all(), "Batch 0 should attend first 14"
        assert not attention_mask[0, 14:].any(), "Batch 0 should not attend padding"
        assert attention_mask[1, :29].all(), "Batch 1 should attend all 29"

    def test_should_return_attention_mask(self, interleaver, d_model):
        """Test that attention mask is returned for padded sequences."""
        batch_size = 2
        # Text: (batch, seq, d_model)
        text = torch.randn(batch_size, 10, d_model)

        # Token IDs with audio placeholders
        token_ids = torch.randint(0, 1000, (batch_size, 10))
        token_ids[0, 2] = AUDIO_PLACEHOLDER_ID
        token_ids[1, 2] = AUDIO_PLACEHOLDER_ID

        # Audio: (batch, n_examples, seq, d_model)
        audio = torch.randn(batch_size, 1, 20, d_model)
        audio_lengths = torch.tensor([[5], [15]])  # Different lengths

        result = interleaver(
            text_hidden_states=text,
            text_token_ids=token_ids,
            audio_hidden_states=audio,
            audio_lengths=audio_lengths,
        )

        # Should return (tokens, attention_mask, modality_map) tuple
        assert isinstance(result, tuple), "Should return tuple with attention mask"
        tokens, attention_mask, modality_map = result

        # Batch 0: 10 - 1 + 5 = 14, Batch 1: 10 - 1 + 15 = 24
        # Padded to 24
        assert tokens.shape == (batch_size, 24, d_model)
        assert attention_mask.shape == (batch_size, 24)

        # Batch 0 should have 1s for first 14 positions, 0s after
        assert attention_mask[0, :14].all(), "Batch 0 should have attention for first 14"
        assert not attention_mask[0, 14:].any(), "Batch 0 should be masked after 14"

        # Batch 1 should have 1s for all 24 positions
        assert attention_mask[1, :24].all(), "Batch 1 should have attention for all 24"

    def test_no_media_at_position_zero(self, interleaver, d_model):
        """Media at position 0 means no text before the audio."""
        batch_size = 1
        # Text: (batch, seq, d_model)
        text = torch.randn(batch_size, 10, d_model)

        # Token IDs with audio placeholder at position 0
        token_ids = torch.randint(0, 1000, (batch_size, 10))
        token_ids[0, 0] = AUDIO_PLACEHOLDER_ID

        # Audio: (batch, n_examples, seq, d_model)
        audio = torch.randn(batch_size, 1, 5, d_model)
        audio_lengths = torch.tensor([[5]])

        # Should work - just means audio comes first
        tokens, attention_mask, modality_map = interleaver(
            text_hidden_states=text,
            text_token_ids=token_ids,
            audio_hidden_states=audio,
            audio_lengths=audio_lengths,
        )

        # Length: 10 - 1 + 5 = 14
        assert tokens.shape == (batch_size, 14, d_model)

    def test_adjacent_media_examples(self, interleaver, d_model):
        """Test when two media examples are adjacent (no text between them)."""
        batch_size = 1
        text_seq_len = 10

        # Text: (batch, seq, d_model)
        text = torch.zeros(batch_size, text_seq_len, d_model)
        for i in range(text_seq_len):
            text[0, i, 0] = i

        # Token IDs with adjacent placeholders at positions 2 and 3
        token_ids = torch.randint(0, 1000, (batch_size, text_seq_len))
        token_ids[0, 2] = AUDIO_PLACEHOLDER_ID
        token_ids[0, 3] = VOICE_PLACEHOLDER_ID

        # Audio at position 2, voice at position 3 (adjacent placeholders)
        audio = torch.ones(batch_size, 1, 5, d_model) * 100
        voice = torch.ones(batch_size, 1, 8, d_model) * 200

        tokens, attention_mask, modality_map = interleaver(
            text_hidden_states=text,
            text_token_ids=token_ids,
            audio_hidden_states=audio,
            audio_lengths=torch.tensor([[5]]),
            voice_hidden_states=voice,
            voice_lengths=torch.tensor([[8]]),
        )

        # Expected: text[0:2] + audio[5] + voice[8] + text[4:10]
        # Length: 2 + 5 + 8 + 6 = 21
        expected_len = text_seq_len - 2 + 5 + 8
        assert tokens.shape == (batch_size, expected_len, d_model)

        # Verify no text between audio and voice
        assert tokens[0, 6, 0].item() == 100, "Last audio token"
        assert tokens[0, 7, 0].item() == 200, "First voice token (immediately after audio)"

    def test_variable_audio_lengths_within_batch(self, interleaver, d_model):
        """Test that audio_lengths properly truncates each example."""
        batch_size = 1
        # Text: (batch, seq, d_model)
        text = torch.randn(batch_size, 10, d_model)

        # Token IDs with two audio placeholders
        token_ids = torch.randint(0, 1000, (batch_size, 10))
        token_ids[0, 2] = AUDIO_PLACEHOLDER_ID
        token_ids[0, 5] = AUDIO_PLACEHOLDER_ID

        # Two audio examples with different lengths
        # Audio: (batch, n_examples, seq, d_model)
        audio = torch.zeros(batch_size, 2, 100, d_model)  # Max capacity 100
        audio[0, 0, :, 0] = 1000  # First example
        audio[0, 1, :, 0] = 2000  # Second example

        # But we only want 10 from first, 25 from second
        audio_lengths = torch.tensor([[10, 25]])

        tokens, attention_mask, modality_map = interleaver(
            text_hidden_states=text,
            text_token_ids=token_ids,
            audio_hidden_states=audio,
            audio_lengths=audio_lengths,
        )

        # Length: 10 - 2 + 10 + 25 = 43
        expected_len = 10 - 2 + 10 + 25
        assert tokens.shape == (batch_size, expected_len, d_model)

    def test_modality_map_returned(self, interleaver, d_model):
        """Test that modality position map is returned correctly."""
        batch_size = 1
        text_seq_len = 10

        # Text: (batch, seq, d_model)
        text = torch.zeros(batch_size, text_seq_len, d_model)

        # Token IDs with audio at 2, image at 6
        token_ids = torch.randint(0, 1000, (batch_size, text_seq_len))
        token_ids[0, 2] = AUDIO_PLACEHOLDER_ID
        token_ids[0, 6] = IMAGE_PLACEHOLDER_ID

        # Audio: (batch, n_examples, seq, d_model)
        audio = torch.ones(batch_size, 1, 5, d_model)
        # Image: (batch, n_examples, patches, d_model)
        image = torch.ones(batch_size, 1, 8, d_model)

        tokens, attention_mask, modality_map = interleaver(
            text_hidden_states=text,
            text_token_ids=token_ids,
            audio_hidden_states=audio,
            audio_lengths=torch.tensor([[5]]),
            image_hidden_states=image,
        )

        # Expected: text[0:2] + audio[5] + text[3:6] + image[8] + text[7:10]
        # Length: 2 + 5 + 3 + 8 + 3 = 21
        expected_len = text_seq_len - 2 + 5 + 8
        assert modality_map.shape == (batch_size, expected_len)

        # Check modality values
        # Positions 0-1: text (0)
        assert (modality_map[0, :2] == 0).all()
        # Positions 2-6: audio (1)
        assert (modality_map[0, 2:7] == 1).all()
        # Positions 7-9: text (0)
        assert (modality_map[0, 7:10] == 0).all()
        # Positions 10-17: image (3)
        assert (modality_map[0, 10:18] == 3).all()
        # Positions 18-20: text (0)
        assert (modality_map[0, 18:] == 0).all()

    def test_assertions_missing_lengths(self, interleaver, d_model):
        """Test that assertions catch missing audio lengths."""
        # Text: (batch, seq, d_model)
        text = torch.randn(1, 10, d_model)
        token_ids = torch.randint(0, 1000, (1, 10))
        token_ids[0, 2] = AUDIO_PLACEHOLDER_ID

        # Audio: (batch, n_examples, seq, d_model)
        audio = torch.randn(1, 1, 5, d_model)

        with pytest.raises(AssertionError):
            interleaver(
                text_hidden_states=text,
                text_token_ids=token_ids,
                audio_hidden_states=audio,
                # Missing audio_lengths
            )

    def test_assertions_missing_text(self, config, d_model):
        """Test that text is required."""
        interleaver = TokenInterleaver(config)
        audio = torch.randn(1, 1, 5, d_model)

        with pytest.raises(ValueError, match="Text hidden states must be provided"):
            interleaver(
                text_hidden_states=None,
                text_token_ids=torch.randint(0, 1000, (1, 10)),
                audio_hidden_states=audio,
                audio_lengths=torch.tensor([[5]]),
            )

    def test_assertions_missing_token_ids(self, config, d_model):
        """Test that token IDs are required."""
        interleaver = TokenInterleaver(config)
        text = torch.randn(1, 10, d_model)

        with pytest.raises(ValueError, match="Text token IDs must be provided"):
            interleaver(
                text_hidden_states=text,
                text_token_ids=None,
            )

    def test_placeholder_count_mismatch(self, interleaver, d_model):
        """Test that assertion fails when placeholder count doesn't match examples."""
        text = torch.randn(1, 10, d_model)
        # Only one placeholder but two audio examples
        token_ids = torch.randint(0, 1000, (1, 10))
        token_ids[0, 2] = AUDIO_PLACEHOLDER_ID

        audio = torch.randn(1, 2, 5, d_model)  # 2 examples
        audio_lengths = torch.tensor([[5, 5]])

        with pytest.raises(AssertionError, match="found 1 audio placeholders but have 2"):
            interleaver(
                text_hidden_states=text,
                text_token_ids=token_ids,
                audio_hidden_states=audio,
                audio_lengths=audio_lengths,
            )


class TestTokenInterleaverEdgeCases:
    """Edge case tests for TokenInterleaver."""

    @pytest.fixture
    def config(self):
        return TokenInterleaverConfig(
            audio_placeholder_token_id=AUDIO_PLACEHOLDER_ID,
            voice_placeholder_token_id=VOICE_PLACEHOLDER_ID,
            image_placeholder_token_id=IMAGE_PLACEHOLDER_ID,
        )

    @pytest.fixture
    def interleaver(self, config):
        return TokenInterleaver(config)

    def test_empty_batch(self, interleaver):
        """Test behavior with batch_size=0."""
        d_model = 64
        # Text: (batch, seq, d_model)
        text = torch.randn(0, 10, d_model)
        token_ids = torch.randint(0, 1000, (0, 10))

        tokens, attention_mask, modality_map = interleaver(
            text_hidden_states=text,
            text_token_ids=token_ids,
        )
        assert tokens.shape[0] == 0

    def test_media_at_end_of_sequence(self, interleaver):
        """Test when media placeholder is at the last position."""
        d_model = 64
        # Text: (batch, seq, d_model)
        text = torch.zeros(1, 5, d_model)
        for i in range(5):
            text[0, i, 0] = i

        # Token IDs with audio at last position
        token_ids = torch.randint(0, 1000, (1, 5))
        token_ids[0, 4] = AUDIO_PLACEHOLDER_ID

        # Audio: (batch, n_examples, seq, d_model)
        audio = torch.ones(1, 1, 10, d_model) * 100
        audio_lengths = torch.tensor([[10]])

        tokens, attention_mask, modality_map = interleaver(
            text_hidden_states=text,
            text_token_ids=token_ids,
            audio_hidden_states=audio,
            audio_lengths=audio_lengths,
        )

        # Length: 5 - 1 + 10 = 14
        assert tokens.shape == (1, 14, d_model)
        # No text after audio
        assert tokens[0, 13, 0].item() == 100, "Should end with audio"

    def test_single_token_text(self, interleaver):
        """Test with minimal text sequence."""
        d_model = 64
        # Text: (batch, seq, d_model)
        text = torch.zeros(1, 1, d_model)
        text[0, 0, 0] = 42

        token_ids = torch.randint(0, 1000, (1, 1))

        # No media, just the single token
        tokens, attention_mask, modality_map = interleaver(
            text_hidden_states=text,
            text_token_ids=token_ids,
        )
        assert tokens.shape == (1, 1, d_model)
        assert tokens[0, 0, 0].item() == 42

    def test_unconfigured_modality_raises(self):
        """Test that using a modality without configuring its token ID raises an error."""
        config = TokenInterleaverConfig(
            audio_placeholder_token_id=AUDIO_PLACEHOLDER_ID,
            # voice and image not configured
        )
        interleaver = TokenInterleaver(config)

        d_model = 64
        text = torch.randn(1, 10, d_model)
        token_ids = torch.randint(0, 1000, (1, 10))

        # Should work with audio
        audio = torch.randn(1, 1, 5, d_model)
        token_ids[0, 2] = AUDIO_PLACEHOLDER_ID
        interleaver(
            text_hidden_states=text,
            text_token_ids=token_ids,
            audio_hidden_states=audio,
            audio_lengths=torch.tensor([[5]]),
        )

        # Should fail with image (not configured)
        image = torch.randn(1, 1, 16, d_model)
        with pytest.raises(ValueError, match="Image placeholder token ID must be configured"):
            interleaver(
                text_hidden_states=text,
                text_token_ids=token_ids,
                image_hidden_states=image,
            )
