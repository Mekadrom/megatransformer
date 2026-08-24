"""Per-utterance rendering helpers on the world visualization callback.

These exist because the bug they fix was invisible: `voice_unit_id_trace` is FLAT across every
voice block a generate() call produced, so a model that ended one utterance and started
another rendered as a single long clip (measured: 437 units -> 17.48 s against a 250-frame,
10 s budget). The `n` dimension means DISJOINT utterances and they must come back separately.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from megatransformer.scripts.train.world.visualization_callback import (
    WorldModelVisualizationCallback as V,
)


def out(segs=None, trace=None, lengths=None, counts=None):
    o = {}
    if segs is not None:
        o["voice_unit_id_segments"] = segs
    if trace is not None:
        o["voice_unit_id_trace"] = trace
    if lengths is not None:
        o["voice_lengths"] = torch.tensor(lengths)
    if counts is not None:
        o["voice_counts"] = torch.tensor(counts)
    return o


def test_prefers_segments_over_flat_trace():
    """Two utterances: utt 0 must be ITS OWN ids, not the concatenation."""
    o = out(segs=[[[1, 2, 3], [7, 8]]], trace=[[1, 2, 3, 7, 8]])
    assert V._generated_unit_ids(o, utt=0).tolist() == [1, 2, 3]
    assert V._generated_unit_ids(o, utt=1).tolist() == [7, 8]
    assert V._generated_utterance_count(o) == 2


def test_no_flat_fallback_for_later_utterances():
    """Without segments there is no way to know where utt 1 starts; returning a slice of the
    flat trace would be the concatenation bug in a new disguise."""
    o = out(trace=[[1, 2, 3, 7, 8]])
    assert V._generated_unit_ids(o, utt=0).tolist() == [1, 2, 3, 7, 8]
    assert V._generated_unit_ids(o, utt=1) is None


def test_teacher_forced_outputs_still_work():
    o = {"voice_unit_logits": torch.zeros(1, 4, 10)}
    o["voice_unit_logits"][0, :, 3] = 5.0
    assert V._generated_unit_ids(o).tolist() == [3, 3, 3, 3]
    assert V._generated_utterance_count(o) == 1


def test_empty_segments_do_not_count():
    o = out(segs=[[[1, 2], []]])
    assert V._generated_utterance_count(o) == 1


def test_trim_uses_real_length_not_padded_width():
    """voice_latent_preds is padded across utterances; the decoder renders a zero tail as
    babble, so utt 0 must be cut to its own length."""
    pred = torch.arange(2 * 10, dtype=torch.float32).reshape(2, 10)   # (C=2, max_T=10)
    o = out(lengths=[[4, 10]])
    assert V._trim_generated_voice(o, pred, utt=0).shape[-1] == 4
    assert V._trim_generated_voice(o, pred, utt=1).shape[-1] == 10


def test_trim_is_a_noop_without_lengths():
    pred = torch.zeros(2, 10)
    assert V._trim_generated_voice({}, pred).shape[-1] == 10


def test_trim_ignores_out_of_range_lengths():
    """A length longer than the tensor (or zero) must not silently produce an empty render."""
    pred = torch.zeros(2, 10)
    assert V._trim_generated_voice(out(lengths=[[99]]), pred).shape[-1] == 10
    assert V._trim_generated_voice(out(lengths=[[0]]), pred).shape[-1] == 10


def test_count_falls_back_to_voice_counts():
    assert V._generated_utterance_count(out(counts=[3])) == 3
