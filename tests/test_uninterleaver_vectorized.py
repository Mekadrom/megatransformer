"""Bit-identity test for the vectorized TokenUninterleaver.

The uninterleaver was a per-batch boolean-index loop (~4*B GPU->CPU syncs/step). The
vectorized masked-scatter version must produce BYTE-IDENTICAL output — same tokens, same
order, same zero padding, same None-for-absent-modality, same length tensors — so it's a
pure throughput change safe on a live-run resume. This checks it against a reference
reimplementation of the old logic across varied modality layouts.
"""
import torch

from megatransformer.model.world.token_alignment import (
    MODALITY_AUDIO,
    MODALITY_IMAGE,
    MODALITY_PAD,
    MODALITY_TEXT,
    MODALITY_VOICE,
    TokenUninterleaver,
)


def _ref_uninterleave(tokens, modality_map):
    """The original per-batch boolean-index + pad logic (reference)."""
    B = tokens.shape[0]
    d = tokens.shape[-1]
    device, dtype = tokens.device, tokens.dtype
    out = {}
    for name, mod in (("text", MODALITY_TEXT), ("audio", MODALITY_AUDIO),
                      ("voice", MODALITY_VOICE), ("image", MODALITY_IMAGE)):
        lists = [tokens[b][modality_map[b] == mod] for b in range(B)]
        lengths = [t.size(0) for t in lists]
        max_len = max(lengths) if lengths else 0
        if max_len == 0:
            out[name] = None
            out[name + "_lengths"] = None
            continue
        padded = []
        for t, l in zip(lists, lengths):
            if l == 0:
                padded.append(torch.zeros(max_len, d, device=device, dtype=dtype))
            elif l < max_len:
                padded.append(torch.cat([t, t.new_zeros(max_len - l, d)], dim=0))
            else:
                padded.append(t)
        out[name] = torch.stack(padded, dim=0)
        out[name + "_lengths"] = torch.tensor(lengths, dtype=torch.long, device=device)
    return out


def _assert_equal(got, ref):
    for key in ("text", "audio", "voice", "image",
                "text_lengths", "audio_lengths", "voice_lengths", "image_lengths"):
        g, r = got[key], ref[key]
        if r is None:
            assert g is None, f"{key}: expected None, got tensor"
        else:
            assert g is not None, f"{key}: expected tensor, got None"
            assert g.shape == r.shape, f"{key}: shape {g.shape} != {r.shape}"
            assert torch.equal(g, r), f"{key}: values differ"


def _random_modality_map(B, S, gen, include=(MODALITY_TEXT, MODALITY_AUDIO,
                                             MODALITY_VOICE, MODALITY_IMAGE, MODALITY_PAD)):
    choices = torch.tensor(include)
    idx = torch.randint(0, len(include), (B, S), generator=gen)
    return choices[idx]


def test_uninterleaver_bit_identical_full():
    gen = torch.Generator().manual_seed(0)
    for _ in range(20):
        B = int(torch.randint(1, 5, (1,), generator=gen))
        S = int(torch.randint(1, 40, (1,), generator=gen))
        d = 8
        tokens = torch.randn(B, S, d, generator=gen)
        mm = _random_modality_map(B, S, gen)
        _assert_equal(TokenUninterleaver()(tokens, mm), _ref_uninterleave(tokens, mm))


def test_uninterleaver_absent_modality_is_none():
    # Only text + voice present -> audio and image must come back None.
    gen = torch.Generator().manual_seed(1)
    tokens = torch.randn(3, 20, 8, generator=gen)
    mm = _random_modality_map(3, 20, gen, include=(MODALITY_TEXT, MODALITY_VOICE, MODALITY_PAD))
    got = TokenUninterleaver()(tokens, mm)
    assert got["audio"] is None and got["image"] is None
    assert got["text"] is not None and got["voice"] is not None
    _assert_equal(got, _ref_uninterleave(tokens, mm))


def test_uninterleaver_row_with_zero_tokens():
    # Row 1 is entirely padding -> voice length 0 for that row, zeros, others intact.
    tokens = torch.randn(2, 6, 4)
    mm = torch.tensor([
        [MODALITY_VOICE, MODALITY_VOICE, MODALITY_TEXT, MODALITY_PAD, MODALITY_PAD, MODALITY_PAD],
        [MODALITY_TEXT,  MODALITY_TEXT,  MODALITY_TEXT, MODALITY_PAD, MODALITY_PAD, MODALITY_PAD],
    ])
    got = TokenUninterleaver()(tokens, mm)
    ref = _ref_uninterleave(tokens, mm)
    _assert_equal(got, ref)
    assert got["voice_lengths"].tolist() == [2, 0]
