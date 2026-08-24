"""Interleaver round-trip and M-RoPE geometry for bistream chunk interleaving.

The chunk map exists so ONE utterance can occupy several placeholders without overloading
`n_voice_examples`, which means DISJOINT UTTERANCES. These tests pin both halves of that:
the voice frames survive interleave -> uninterleave in order, and the two position axes still
separate utterance 0 chunk j from utterance 1 chunk j.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from megatransformer.model.world.token_alignment import (
    TokenInterleaver, TokenUninterleaver, build_mrope_position_ids,
    MODALITY_TEXT, MODALITY_VOICE,
)
from megatransformer.utils import constants

SP = constants.special_token_ids(constants.SPECIAL_TOKEN_BASE)
D = 4


class Cfg:
    audio_placeholder_token_id = SP.AUDIO_PLACEHOLDER
    voice_placeholder_token_id = SP.VOICE_PLACEHOLDER
    image_placeholder_token_id = SP.IMAGE_PLACEHOLDER


def build(text_ids):
    """text_hidden_states carrying a unique, recoverable signature per position."""
    B, S = text_ids.shape
    h = torch.arange(B * S, dtype=torch.float32).reshape(B, S, 1).repeat(1, 1, D)
    return h


def test_chunked_roundtrip_recovers_frames_in_order():
    """One utterance spread over 3 placeholders comes back as one contiguous stream."""
    # [t t][BOV][PH][EOV][t t][BOV][PH][EOV][t t][BOV][PH][EOV][eos]
    row = ([1, 2, SP.BOV, SP.VOICE_PLACEHOLDER, SP.EOV] * 1
           + [3, 4, SP.BOV, SP.VOICE_PLACEHOLDER, SP.EOV]
           + [5, 6, SP.BOV, SP.VOICE_PLACEHOLDER, SP.EOV] + [2])
    text_ids = torch.tensor([row])
    text_h = build(text_ids)

    # one utterance, 12 frames, sliced 4 / 4 / 4
    voice = torch.arange(1000, 1000 + 12 * D, dtype=torch.float32).reshape(1, 1, 12, D)
    voice_lengths = torch.tensor([[12]])
    chunk_map = torch.tensor([[[0, 0, 4], [0, 4, 4], [0, 8, 4]]])

    il = TokenInterleaver(Cfg())
    tokens, mask, modmap = il(text_h, text_ids,
                              voice_hidden_states=voice, voice_lengths=voice_lengths,
                              voice_chunk_map=chunk_map)

    assert int((modmap == MODALITY_VOICE).sum()) == 12
    un = TokenUninterleaver()(tokens, modmap)
    assert int(un["voice_lengths"][0]) == 12
    # exact frames, in the original order
    assert torch.equal(un["voice"][0, :12], voice[0, 0])


def test_default_map_is_historical_behaviour():
    """No map: placeholder i -> whole utterance i, unchanged."""
    row = [1, 2, SP.BOV, SP.VOICE_PLACEHOLDER, SP.EOV, 3, SP.BOV, SP.VOICE_PLACEHOLDER, SP.EOV]
    text_ids = torch.tensor([row])
    text_h = build(text_ids)
    voice = torch.randn(1, 2, 6, D)
    voice_lengths = torch.tensor([[6, 4]])

    il = TokenInterleaver(Cfg())
    t1, _, m1 = il(text_h, text_ids, voice_hidden_states=voice, voice_lengths=voice_lengths)
    # the equivalent explicit map
    cmap = torch.tensor([[[0, 0, 6], [1, 0, 4]]])
    t2, _, m2 = il(text_h, text_ids, voice_hidden_states=voice, voice_lengths=voice_lengths,
                   voice_chunk_map=cmap)
    assert torch.equal(t1, t2) and torch.equal(m1, m2)


def test_two_utterances_chunked_stay_separable():
    """Two utterances, each in 2 chunks. Frames must not cross utterances, and M-RoPE must
    still distinguish utterance 0 chunk 1 from utterance 1 chunk 1."""
    blk = [SP.BOV, SP.VOICE_PLACEHOLDER, SP.EOV]
    row = [1, 2] + blk + [3, 4] + blk + [5, 6] + blk + [7, 8] + blk + [2]
    text_ids = torch.tensor([row])
    text_h = build(text_ids)

    v0 = torch.arange(0, 6 * D, dtype=torch.float32).reshape(6, D)
    v1 = torch.arange(500, 500 + 6 * D, dtype=torch.float32).reshape(6, D)
    voice = torch.stack([v0, v1]).unsqueeze(0)                # (1, 2, 6, D)
    voice_lengths = torch.tensor([[6, 6]])
    chunk_map = torch.tensor([[[0, 0, 3], [0, 3, 3], [1, 0, 3], [1, 3, 3]]])

    il = TokenInterleaver(Cfg())
    tokens, _, modmap = il(text_h, text_ids, voice_hidden_states=voice,
                           voice_lengths=voice_lengths, voice_chunk_map=chunk_map)
    un = TokenUninterleaver()(tokens, modmap)
    assert int(un["voice_lengths"][0]) == 12
    assert torch.equal(un["voice"][0, :6], v0)
    assert torch.equal(un["voice"][0, 6:12], v1)

    pos = build_mrope_position_ids(modmap, voice_rate=6.0, scale_side="text")
    vpos = pos[0][modmap[0] == MODALITY_VOICE]                # (12, 2) global/local
    glob, loc = vpos[:, 0], vpos[:, 1]
    # LOCAL re-anchors per contiguous voice run -- i.e. per chunk. That is desirable: the
    # frame's coordinate is relative to the text chunk it was just given.
    assert loc[:3].tolist() == loc[3:6].tolist() == loc[6:9].tolist() == loc[9:].tolist()
    # ...which means GLOBAL is the ONLY thing separating utterance 0 chunk 1 from
    # utterance 1 chunk 1. It must be strictly increasing across the whole sequence.
    assert torch.all(glob[1:] > glob[:-1]), "global axis must separate identical local runs"
