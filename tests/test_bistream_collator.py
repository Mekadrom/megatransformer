"""Bistream chunk-interleaving in the world data collator.

Two things are being protected here:

1. **Byte-identity when bistream is off.** The discrete voice path was rewritten to build its
   output from a segment description instead of pad_and_mask, so every tensor it emits with
   `bistream_text_chunk=0` must equal what the old code produced, bit for bit. A regression
   here is invisible in training loss and would quietly corrupt every existing run's data.

2. **The chunked layout is what the plan specifies.** Alternating text/voice blocks, chunk
   lengths summing to the utterance, fill_token at interior chunk ends only, EOV once at the
   very end, and the chunk map slicing the emitted stream exactly.
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from megatransformer.scripts.data.world.data_collator import MultimodalDataCollator
from megatransformer.utils import constants

K = 6561                      # CosyVoice 2 codebook size; EOV = K, FILL = K + 1
FILL = K + 1
CAP = 250
SP = constants.special_token_ids(constants.SPECIAL_TOKEN_BASE)


def make_sample(text_len, n_frames, D=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "_modality": "voice", "_task": "voice_synthesis", "_direction": "synthesis",
        "text_token_ids": torch.randint(0, 30000, (text_len,), generator=g),
        "text_text_length": text_len,
        "voice_unit_ids": torch.randint(0, K, (n_frames,), generator=g),
        "voice_features": torch.randn(D, n_frames, generator=g),
        "voice_feature_length": n_frames,
    }


def collator(**kw):
    return MultimodalDataCollator(
        max_seq_len=1024, max_waveforms=16000, max_mel_spec_frames=100,
        max_sive_feature_frames=CAP, voice_eov_id=K, **kw)


SAMPLES = lambda: [make_sample(20, 130, seed=1), make_sample(11, 73, seed=2),
                   make_sample(31, 193, seed=3)]


def test_bistream_off_is_unchanged():
    """The rewritten discrete path must reproduce the original tensors exactly."""
    c = collator()
    c.force_direction = "synthesis"
    b = c(SAMPLES())

    for i, (tl, n) in enumerate([(20, 130), (11, 73), (31, 193)]):
        # content frames + exactly one EOV slot
        assert int(b["voice_feature_lengths"][i]) == n + 1
        u = b["voice_unit_ids"][i]
        assert int(u[n]) == K, "EOV sits immediately after the last content frame"
        assert (u[:n] >= 0).all() and (u[:n] < K).all()
        assert (u[n + 1:] == -100).all(), "padding is the CE ignore_index"
        # the EOV frame's feature column is zero
        assert torch.equal(b["voice_features"][i][:, n], torch.zeros_like(b["voice_features"][i][:, n]))
        assert float(b["voice_feature_masks"][i][n]) == 1.0
        assert float(b["voice_feature_masks"][i][n + 1:].sum()) == 0.0
        # unistream token layout: [text][BOV][PH][EOV][eos]
        t = b["text_token_ids"][i][:int(b["text_lengths"][i])]
        assert t[tl].item() == SP.BOV
        assert t[tl + 1].item() == SP.VOICE_PLACEHOLDER
        assert t[tl + 2].item() == SP.EOV
        assert (t == SP.VOICE_PLACEHOLDER).sum().item() == 1

    assert "voice_chunk_starts" not in b, "no chunk map when bistream is off"


def test_features_match_source_frames():
    c = collator()
    c.force_direction = "synthesis"
    s = SAMPLES()
    b = c(s)
    for i, ex in enumerate(s):
        n = ex["voice_feature_length"]
        assert torch.equal(b["voice_features"][i][:, :n], ex["voice_features"][:, :n])
        assert torch.equal(b["voice_unit_ids"][i][:n], ex["voice_unit_ids"][:n].long())


def test_bistream_layout():
    random.seed(0)
    c = collator(bistream_text_chunk=5, bistream_voice_chunk=30,
                 bistream_prob=1.0, voice_fill_id=FILL)
    c.force_direction = "synthesis"
    s = SAMPLES()
    b = c(s)

    assert "voice_chunk_starts" in b
    for i, ex in enumerate(s):
        tl = ex["text_text_length"]
        n = ex["voice_feature_length"]
        m = int(b["voice_chunk_counts"][i])
        assert m == (tl + 4) // 5, "one voice chunk per text chunk"

        starts = b["voice_chunk_starts"][i][:m]
        lens = b["voice_chunk_lengths"][i][:m]
        # chunks tile the emitted stream with no gap and no overlap
        assert int(starts[0]) == 0
        for j in range(1, m):
            assert int(starts[j]) == int(starts[j - 1]) + int(lens[j - 1])
        total = int(starts[-1]) + int(lens[-1])
        assert total == int(b["voice_feature_lengths"][i])
        # content frames + one terminal per chunk
        assert total == n + m

        u = b["voice_unit_ids"][i]
        for j in range(m):
            end = int(starts[j]) + int(lens[j]) - 1
            assert int(u[end]) == (FILL if j < m - 1 else K), \
                "fill_token ends interior chunks, EOV ends the utterance"
        assert (u[:total] != -100).all()
        assert (u[total:] == -100).all()
        assert int((u[:total] == K).sum()) == 1, "exactly one EOV"
        assert int((u[:total] == FILL).sum()) == m - 1

        # content, terminals stripped, is the original unit sequence in order
        content = torch.cat([u[int(starts[j]):int(starts[j]) + int(lens[j]) - 1] for j in range(m)])
        assert torch.equal(content, ex["voice_unit_ids"][:n].long())

        # every terminal slot has a zeroed feature column
        for j in range(m):
            end = int(starts[j]) + int(lens[j]) - 1
            assert float(b["voice_features"][i][:, end].abs().sum()) == 0.0

        # token stream alternates: m x ([<=5 text][BOV][PH][EOV]) then eos
        t = b["text_token_ids"][i][:int(b["text_lengths"][i])]
        assert int((t == SP.VOICE_PLACEHOLDER).sum()) == m
        pos = 0
        for j in range(m):
            want = min(5, tl - j * 5)
            assert int((t[pos:pos + want] < constants.SPECIAL_TOKEN_BASE).sum()) == want
            pos += want
            assert t[pos].item() == SP.BOV
            assert t[pos + 1].item() == SP.VOICE_PLACEHOLDER
            assert t[pos + 2].item() == SP.EOV
            pos += 3
        assert pos == len(t) - 1, "only the eos remains"


def test_one_chunk_plan_equals_unistream():
    """A plan that yields a single chunk must reproduce the unistream layout exactly."""
    short = [make_sample(4, 40, seed=9)]          # 4 tokens -> 1 text chunk -> rejected
    a = collator(); a.force_direction = "synthesis"
    b1 = a(short)
    c = collator(bistream_text_chunk=5, bistream_voice_chunk=30,
                 bistream_prob=1.0, voice_fill_id=FILL)
    c.force_direction = "synthesis"
    b2 = c(short)
    for k in ("text_token_ids", "voice_unit_ids", "voice_feature_lengths", "voice_features"):
        assert torch.equal(b1[k], b2[k]), k


def test_infeasible_samples_stay_unistream():
    """Not enough speech for the chunks before the last one => no chunking, no crash."""
    dense = [make_sample(40, 60, seed=4)]         # 8 text chunks would need > 210 frames
    c = collator(bistream_text_chunk=5, bistream_voice_chunk=30,
                 bistream_prob=1.0, voice_fill_id=FILL)
    c.force_direction = "synthesis"
    b = c(dense)
    assert int((b["text_token_ids"][0] == SP.VOICE_PLACEHOLDER).sum()) == 1
    assert int(b["voice_feature_lengths"][0]) == 61


def test_transcription_is_never_chunked():
    c = collator(bistream_text_chunk=5, bistream_voice_chunk=30,
                 bistream_prob=1.0, voice_fill_id=FILL)
    c.force_direction = "transcription"
    b = c(SAMPLES())
    for i in range(3):
        assert int((b["text_token_ids"][i] == SP.VOICE_PLACEHOLDER).sum()) == 1


def test_fill_id_required():
    try:
        collator(bistream_text_chunk=5, bistream_voice_chunk=30)
    except ValueError as e:
        assert "voice_fill_id" in str(e)
    else:
        raise AssertionError("expected a ValueError when fill id is missing")
