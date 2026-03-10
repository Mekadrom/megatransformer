"""Test that text targets align with text coda logits in the world model trainer.

The collator injects placeholder tokens (VOICE_PH, IMAGE_PH, AUDIO_PH) into
text sequences. The interleaver replaces these with media embeddings, so the
text coda only sees non-placeholder positions. Targets must be built to match.

Key invariant: BOV's target should be EOV (the next text token), NOT VOICE_PH.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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

PLACEHOLDER_IDS = {AUDIO_PLACEHOLDER_TOKEN_ID, VOICE_PLACEHOLDER_TOKEN_ID, IMAGE_PLACEHOLDER_TOKEN_ID}


def build_targets(text_input_ids: torch.Tensor):
    """Reproduce the target-building logic from WorldModelTrainer.compute_loss.

    Returns (targets, n_logits_per_item, model_input_ids).
    """
    full_ids = text_input_ids

    # Remove placeholders from full sequence, then shift
    non_ph_mask = torch.ones_like(full_ids, dtype=torch.bool)
    for pid in PLACEHOLDER_IDS:
        non_ph_mask &= (full_ids != pid)

    # Model input (:-1, keeps placeholders)
    model_input = full_ids[:, :-1]

    # Count logits: non-placeholder positions in model input
    input_non_ph = torch.ones_like(model_input, dtype=torch.bool)
    for pid in PLACEHOLDER_IDS:
        input_non_ph &= (model_input != pid)
    n_logits = [input_non_ph[b].sum().item() for b in range(model_input.shape[0])]

    # Build targets: for each item, take exactly n_logits[b] targets
    target_list = []
    for b in range(full_ids.shape[0]):
        clean = full_ids[b][non_ph_mask[b]]
        shifted = clean[1:]  # causal shift
        K = n_logits[b]
        if shifted.shape[0] >= K:
            target_list.append(shifted[:K])
        else:
            target_list.append(torch.cat([shifted, shifted.new_zeros(K - shifted.shape[0])]))

    # Pad and stack targets
    max_len = max(t.shape[0] for t in target_list)
    padded = []
    for t in target_list:
        if t.shape[0] < max_len:
            padded.append(torch.cat([t, t.new_zeros(max_len - t.shape[0])]))
        else:
            padded.append(t)

    return torch.stack(padded), n_logits, model_input


def test_text_only():
    """Text-only sequence: no placeholders, standard causal shift."""
    ids = torch.tensor([[10, 20, 30, 40, 50]])
    targets, n_logits, _ = build_targets(ids)

    assert n_logits[0] == 4
    assert targets.shape == (1, 4)
    assert targets[0].tolist() == [20, 30, 40, 50]


def test_synthesis_voice():
    """Synthesis: [text] [BOV] [VOICE_PH] [EOV]."""
    ids = torch.tensor([[10, 20, 30, BOV_TOKEN_ID, VOICE_PLACEHOLDER_TOKEN_ID, EOV_TOKEN_ID]])
    targets, n_logits, _ = build_targets(ids)

    # Full no-PH: [10, 20, 30, BOV, EOV] → targets = [20, 30, BOV, EOV]
    # Model input (:-1): [10, 20, 30, BOV, VOICE_PH] → 4 non-PH → 4 logits
    assert n_logits[0] == 4
    assert targets.shape[1] == 4
    assert targets[0].tolist() == [20, 30, BOV_TOKEN_ID, EOV_TOKEN_ID]


def test_synthesis_voice_bov_target_is_eov():
    """Critical: at the BOV position, the target must be EOV, not VOICE_PH."""
    ids = torch.tensor([[10, BOV_TOKEN_ID, VOICE_PLACEHOLDER_TOKEN_ID, EOV_TOKEN_ID]])
    targets, n_logits, _ = build_targets(ids)

    # Full no-PH: [10, BOV, EOV] → targets = [BOV, EOV]
    # Model input (:-1): [10, BOV, VOICE_PH] → 2 non-PH → 2 logits
    assert n_logits[0] == 2
    assert targets[0, 0].item() == BOV_TOKEN_ID  # at pos 10, predict BOV
    assert targets[0, 1].item() == EOV_TOKEN_ID  # at pos BOV, predict EOV (NOT VOICE_PH)


def test_transcription_voice():
    """Transcription: [BOV] [VOICE_PH] [EOV] [text]."""
    ids = torch.tensor([[BOV_TOKEN_ID, VOICE_PLACEHOLDER_TOKEN_ID, EOV_TOKEN_ID, 10, 20, 30]])
    targets, n_logits, _ = build_targets(ids)

    # Full no-PH: [BOV, EOV, 10, 20, 30] → targets = [EOV, 10, 20, 30]
    # Model input (:-1): [BOV, VOICE_PH, EOV, 10, 20] → 4 non-PH → 4 logits
    assert n_logits[0] == 4
    assert targets.shape[1] == 4
    assert targets[0].tolist() == [EOV_TOKEN_ID, 10, 20, 30]


def test_voice_and_image():
    """Mixed: [text] [BOV] [VOICE_PH] [EOV] [BOI] [IMAGE_PH] [EOI]."""
    ids = torch.tensor([[10, 20, BOV_TOKEN_ID, VOICE_PLACEHOLDER_TOKEN_ID, EOV_TOKEN_ID,
                          BOI_TOKEN_ID, IMAGE_PLACEHOLDER_TOKEN_ID, EOI_TOKEN_ID]])
    targets, n_logits, _ = build_targets(ids)

    # Full no-PH: [10, 20, BOV, EOV, BOI, EOI] → targets = [20, BOV, EOV, BOI, EOI]
    # Model input (:-1): [10, 20, BOV, VOICE_PH, EOV, BOI, IMAGE_PH] → 5 non-PH → 5 logits
    assert n_logits[0] == 5
    assert targets.shape[1] == 5
    assert targets[0].tolist() == [20, BOV_TOKEN_ID, EOV_TOKEN_ID, BOI_TOKEN_ID, EOI_TOKEN_ID]


def test_batch_alignment():
    """Different sequence lengths in a batch still align."""
    ids = torch.tensor([
        [10, 20, 30, 0, 0, 0],
        [10, BOV_TOKEN_ID, VOICE_PLACEHOLDER_TOKEN_ID, EOV_TOKEN_ID, 20, 30],
    ])
    targets, n_logits, _ = build_targets(ids)

    # Item 0: no placeholders, input [10,20,30,0,0] → 5 logits
    assert n_logits[0] == 5
    # Item 1: input [10,BOV,VOICE_PH,EOV,20] → 4 non-PH → 4 logits
    assert n_logits[1] == 4

    # Targets: item 0 has 5, item 1 has 4 → padded to 5
    assert targets.shape == (2, 5)
    assert targets[0].tolist() == [20, 30, 0, 0, 0]
    # Item 1 no-PH: [10, BOV, EOV, 20, 30] → targets = [BOV, EOV, 20, 30]
    assert targets[1].tolist() == [BOV_TOKEN_ID, EOV_TOKEN_ID, 20, 30, 0]


def test_n_logits_equals_n_targets():
    """For each batch item, n_logits must equal the number of real targets."""
    ids = torch.tensor([
        [10, 20, BOV_TOKEN_ID, VOICE_PLACEHOLDER_TOKEN_ID, EOV_TOKEN_ID,
         BOI_TOKEN_ID, IMAGE_PLACEHOLDER_TOKEN_ID, EOI_TOKEN_ID, 30, 40],
    ])
    targets, n_logits, _ = build_targets(ids)

    # Full no-PH: [10,20,BOV,EOV,BOI,EOI,30,40] → 8 tokens → 7 targets
    # Input (:-1): [10,20,BOV,VOICE_PH,EOV,BOI,IMAGE_PH,EOI,30] → 7 non-PH → 7 logits
    assert n_logits[0] == 7
    assert targets.shape[1] == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
