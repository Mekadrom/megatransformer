"""EOV terminal-token placement in the world collator.

The discrete voice path replaces the old binary stop head with an end-of-voice (EOV)
token in the SIVE-VQ vocabulary: a K+1-way unit head where class K == EOV. The collator
must append that terminal token at index `length` (immediately after the last content
frame, indices 0..length-1) for EVERY utterance, and it must NEVER land inside padding.
This test guards that invariant, including the longest-utterance edge case (length == the
batch max, which is why features/units are padded to max+1).
"""
import torch
import pytest

from megatransformer.scripts.data.world.data_collator import MultimodalDataCollator

K = 250          # codebook size; EOV id == K, unit vocab == K+1
EOV = K
C = 8            # feature channels (tiny)


def _make_ex(length):
    """A voice example with exactly `length` content frames (no stored past-length
    frames -- the realistic case). Unit ids 1..length are nonzero so content is spottable."""
    feats = torch.ones((C, length))
    units = torch.arange(1, length + 1, dtype=torch.long)
    return {
        "voice_features": feats,
        "voice_feature_length": torch.tensor(length, dtype=torch.long),
        "voice_unit_ids": units,
    }


@pytest.mark.parametrize("lengths", [
    [3, 7, 1, 7],   # 7 == batch max: longest utterance still gets an EOV slot
    [5, 5, 5],      # all equal
    [1],            # single, minimal
    [2, 4],         # simple mixed
])
def test_eov_lands_at_length_never_in_padding(lengths):
    exs = [_make_ex(n) for n in lengths]
    batch = MultimodalDataCollator(max_seq_len=64, voice_eov_id=EOV)(exs)
    uids = batch["voice_unit_ids"]          # (B, T)
    flens = batch["voice_feature_lengths"]  # (B,)
    feats = batch["voice_features"]         # (B, C, T)
    fmask = batch["voice_feature_masks"]    # (B, T)

    T = uids.shape[1]
    assert T == max(lengths) + 1            # padded to max+1 so the longest fits an EOV

    for i, n in enumerate(lengths):
        row = uids[i]
        assert int(flens[i]) == n + 1                              # length bumped by 1
        assert torch.equal(row[:n], torch.arange(1, n + 1))        # content preserved
        assert int(row[n]) == EOV                                  # EOV exactly at index length
        assert int((row == EOV).sum()) == 1                        # exactly one EOV
        assert torch.all(row[n + 1:] == -100)                      # nothing but -100 after it
        assert torch.all(feats[i, :, n] == 0.0)                    # terminal feature frame zeroed
        assert float(fmask[i, n]) == 1.0                           # terminal position is valid
        assert torch.all(fmask[i, n + 1:] == 0.0)                  # padding is invalid


def test_maxed_out_utterance_gets_no_eov():
    """An utterance at the frame cap (== the inference generation budget) was truncated,
    not ended: it must get NO EOV (mirroring how the text collator withholds EOS from
    truncated text, and matching inference which budget-stops at the cap without EOV).
    A shorter utterance in the same batch still gets its EOV."""
    cap = 7
    col = MultimodalDataCollator(max_seq_len=64, max_sive_feature_frames=cap, voice_eov_id=EOV)
    batch = col([_make_ex(cap), _make_ex(3)])   # one maxed, one normal
    uids = batch["voice_unit_ids"]
    flens = batch["voice_feature_lengths"]

    assert uids.shape[1] == cap                  # width stays at the cap, never cap+1
    # maxed row: full content, NO EOV, NO padding
    assert torch.equal(uids[0], torch.arange(1, cap + 1))
    assert int((uids[0] == EOV).sum()) == 0
    assert int(flens[0]) == cap                  # length NOT bumped
    # normal row: EOV at index 3, padding after
    assert int(uids[1, 3]) == EOV and torch.all(uids[1, 4:] == -100)
    assert int(flens[1]) == 4                     # length bumped by 1


def test_all_maxed_batch_has_no_eov_and_no_crash():
    cap = 5
    col = MultimodalDataCollator(max_seq_len=64, max_sive_feature_frames=cap, voice_eov_id=EOV)
    batch = col([_make_ex(cap), _make_ex(cap)])
    uids = batch["voice_unit_ids"]
    assert uids.shape[1] == cap
    assert int((uids == EOV).sum()) == 0
    assert torch.all(batch["voice_feature_lengths"] == cap)


def test_no_eov_id_is_backward_compatible():
    """Without voice_eov_id (continuous path / debug scripts), the collator must be
    unchanged: original width, no terminal token, no length bumping."""
    exs = [_make_ex(3), _make_ex(5)]
    batch = MultimodalDataCollator(max_seq_len=64)(exs)   # voice_eov_id defaults to None
    uids = batch["voice_unit_ids"]
    assert uids.shape[1] == 5                              # max content length, not +1
    assert int((uids == EOV).sum()) == 0                  # no EOV injected
    assert int(batch["voice_feature_lengths"][0]) == 3    # length not bumped


def test_eov_gated_on_unit_ids_present():
    """A features-only voice example (no unit ids) must not get an EOV even if an
    eov_id is configured -- EOV is a unit-vocabulary token."""
    ex = {"voice_features": torch.ones((C, 4)),
          "voice_feature_length": torch.tensor(4, dtype=torch.long)}
    batch = MultimodalDataCollator(max_seq_len=64, voice_eov_id=EOV)([ex])
    assert "voice_unit_ids" not in batch
    assert int(batch["voice_feature_lengths"][0]) == 4    # no bump without units
