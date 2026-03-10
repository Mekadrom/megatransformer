"""Test that BO*/PH/EO* tokens are placed after content, before padding."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import random
import torch
import pytest

from utils.constants import (
    BOA_TOKEN_ID, EOA_TOKEN_ID,
    BOV_TOKEN_ID, EOV_TOKEN_ID,
    BOI_TOKEN_ID, EOI_TOKEN_ID,
    AUDIO_PLACEHOLDER_TOKEN_ID,
    VOICE_PLACEHOLDER_TOKEN_ID,
    IMAGE_PLACEHOLDER_TOKEN_ID,
)

BOUNDARY_AND_PH = {
    BOA_TOKEN_ID, EOA_TOKEN_ID,
    BOV_TOKEN_ID, EOV_TOKEN_ID,
    BOI_TOKEN_ID, EOI_TOKEN_ID,
    AUDIO_PLACEHOLDER_TOKEN_ID,
    VOICE_PLACEHOLDER_TOKEN_ID,
    IMAGE_PLACEHOLDER_TOKEN_ID,
}

PLACEHOLDER_IDS = {AUDIO_PLACEHOLDER_TOKEN_ID, VOICE_PLACEHOLDER_TOKEN_ID, IMAGE_PLACEHOLDER_TOKEN_ID}


def _build_token_sequence(text_token_ids, text_length, has_voice=False, has_image=False, has_audio=False, direction="synthesis"):
    """Reproduce collator's _build_token_sequence with fixed direction for testing."""
    text_tokens = text_token_ids[:text_length]

    media_blocks = []
    if has_audio:
        media_blocks.append(torch.tensor([BOA_TOKEN_ID, AUDIO_PLACEHOLDER_TOKEN_ID, EOA_TOKEN_ID], dtype=text_tokens.dtype))
    if has_voice:
        media_blocks.append(torch.tensor([BOV_TOKEN_ID, VOICE_PLACEHOLDER_TOKEN_ID, EOV_TOKEN_ID], dtype=text_tokens.dtype))
    if has_image:
        media_blocks.append(torch.tensor([BOI_TOKEN_ID, IMAGE_PLACEHOLDER_TOKEN_ID, EOI_TOKEN_ID], dtype=text_tokens.dtype))

    if not media_blocks:
        return text_tokens

    media_sequence = torch.cat(media_blocks)

    if direction == "synthesis":
        return torch.cat([text_tokens, media_sequence])
    else:
        return torch.cat([media_sequence, text_tokens])


def _pad_batch(sequences):
    """Pad sequences to max length with 0s, like the collator does."""
    max_len = max(s.shape[0] for s in sequences)
    padded = []
    for s in sequences:
        if s.shape[0] < max_len:
            padded.append(torch.cat([s, s.new_zeros(max_len - s.shape[0])]))
        else:
            padded.append(s)
    return torch.stack(padded)


def test_voice_synthesis_tokens_after_content():
    """Synthesis: [content] [BOV] [VOICE_PH] [EOV]."""
    text_ids = torch.tensor([10, 20, 30, 40, 50, 0, 0, 0])  # 5 real + 3 pad
    seq = _build_token_sequence(text_ids, text_length=5, has_voice=True, direction="synthesis")
    expected = [10, 20, 30, 40, 50, BOV_TOKEN_ID, VOICE_PLACEHOLDER_TOKEN_ID, EOV_TOKEN_ID]
    assert seq.tolist() == expected


def test_voice_transcription_tokens_before_content():
    """Transcription: [BOV] [VOICE_PH] [EOV] [content]."""
    text_ids = torch.tensor([10, 20, 30, 0, 0])
    seq = _build_token_sequence(text_ids, text_length=3, has_voice=True, direction="transcription")
    expected = [BOV_TOKEN_ID, VOICE_PLACEHOLDER_TOKEN_ID, EOV_TOKEN_ID, 10, 20, 30]
    assert seq.tolist() == expected


def test_padded_batch_tokens_before_padding():
    """In a padded batch, special tokens must come before any 0-padding."""
    # Batch with different lengths
    seq1 = _build_token_sequence(torch.tensor([10, 20, 30, 40, 50]), text_length=5,
                                  has_voice=True, direction="synthesis")
    seq2 = _build_token_sequence(torch.tensor([10, 20, 30]), text_length=3,
                                  has_voice=True, direction="synthesis")
    batch = _pad_batch([seq1, seq2])

    for b in range(batch.shape[0]):
        row = batch[b].tolist()
        # Find the last non-zero position
        last_nonzero = -1
        for i in range(len(row) - 1, -1, -1):
            if row[i] != 0:
                last_nonzero = i
                break
        # Find the last special token position
        last_special = -1
        for i in range(len(row) - 1, -1, -1):
            if row[i] in BOUNDARY_AND_PH:
                last_special = i
                break
        # Special tokens must be ≤ last non-zero (no specials in padding zone)
        if last_special >= 0:
            assert last_special <= last_nonzero, (
                f"Batch item {b}: last special token at {last_special} but last non-zero at {last_nonzero}"
            )


def test_no_placeholder_at_sequence_end():
    """After padding, no sequence should end with a placeholder token.

    This is critical: if the last token in the padded sequence is a placeholder,
    the :-1 causal shift drops it, creating a target alignment off-by-1.
    """
    # Synthesis: last token is always EO*
    seq = _build_token_sequence(torch.tensor([10, 20, 30]), text_length=3,
                                 has_voice=True, has_image=True, direction="synthesis")
    assert seq[-1].item() not in PLACEHOLDER_IDS

    # Transcription: last token is always text
    seq = _build_token_sequence(torch.tensor([10, 20, 30]), text_length=3,
                                 has_voice=True, has_image=True, direction="transcription")
    assert seq[-1].item() not in PLACEHOLDER_IDS


def test_padded_batch_last_token_not_placeholder():
    """After batch padding, no item should have a placeholder as the last token of the full padded row."""
    seqs = [
        _build_token_sequence(torch.tensor([10, 20, 30, 40, 50]), text_length=5,
                               has_voice=True, direction="synthesis"),
        _build_token_sequence(torch.tensor([10, 20]), text_length=2,
                               has_voice=True, direction="transcription"),
    ]
    batch = _pad_batch(seqs)

    for b in range(batch.shape[0]):
        last_token = batch[b, -1].item()
        assert last_token not in PLACEHOLDER_IDS, (
            f"Batch item {b}: last token is placeholder {last_token}"
        )


def test_target_alignment_with_padded_batch():
    """Verify that target count matches logit count for padded batches.

    Logits = non-PH positions in model_input (full[:, :-1]).
    Targets = remove PH from full, shift by 1.
    They should be equal for all items.
    """
    seqs = [
        _build_token_sequence(torch.tensor([10, 20, 30, 40, 50]), text_length=5,
                               has_voice=True, direction="synthesis"),
        _build_token_sequence(torch.tensor([10, 20, 30]), text_length=3,
                               has_voice=True, direction="transcription"),
        _build_token_sequence(torch.tensor([10, 20, 30, 40]), text_length=4,
                               has_voice=True, has_image=True, direction="synthesis"),
    ]
    batch = _pad_batch(seqs)

    # Model input (standard causal shift)
    model_input = batch[:, :-1]

    # Count logits per item: non-PH in model_input
    non_ph_input = torch.ones_like(model_input, dtype=torch.bool)
    for pid in PLACEHOLDER_IDS:
        non_ph_input &= (model_input != pid)
    n_logits = [non_ph_input[b].sum().item() for b in range(batch.shape[0])]

    # Build targets: remove PH from full, shift by 1
    non_ph_full = torch.ones_like(batch, dtype=torch.bool)
    for pid in PLACEHOLDER_IDS:
        non_ph_full &= (batch != pid)

    n_targets = []
    for b in range(batch.shape[0]):
        clean = batch[b][non_ph_full[b]]
        targets = clean[1:]  # causal shift
        n_targets.append(targets.shape[0])

    # They should be equal per-item
    for b in range(batch.shape[0]):
        assert n_logits[b] == n_targets[b], (
            f"Item {b}: {n_logits[b]} logits vs {n_targets[b]} targets. "
            f"Sequence: {batch[b].tolist()}"
        )


def test_target_alignment_max_lengths_equal():
    """The max logit count and max target count across batch should be equal.

    This is what determines tensor shapes in training (after uninterleaver padding).
    """
    random.seed(42)
    for _ in range(50):  # random trials
        n_items = random.randint(2, 6)
        seqs = []
        for _ in range(n_items):
            text_len = random.randint(3, 20)
            text = torch.arange(10, 10 + text_len, dtype=torch.long)
            has_v = random.random() < 0.5
            has_i = random.random() < 0.5
            direction = "synthesis" if random.random() < 0.5 else "transcription"
            seqs.append(_build_token_sequence(text, text_length=text_len,
                                               has_voice=has_v, has_image=has_i, direction=direction))

        batch = _pad_batch(seqs)
        model_input = batch[:, :-1]

        # Max logits
        non_ph_input = torch.ones_like(model_input, dtype=torch.bool)
        for pid in PLACEHOLDER_IDS:
            non_ph_input &= (model_input != pid)
        max_logits = max(non_ph_input[b].sum().item() for b in range(batch.shape[0]))

        # Max targets
        non_ph_full = torch.ones_like(batch, dtype=torch.bool)
        for pid in PLACEHOLDER_IDS:
            non_ph_full &= (batch != pid)
        max_targets = max(batch[b][non_ph_full[b]][1:].shape[0] for b in range(batch.shape[0]))

        assert max_logits == max_targets, (
            f"max_logits={max_logits} vs max_targets={max_targets}. "
            f"Batch shapes: {[s.shape[0] for s in seqs]}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
