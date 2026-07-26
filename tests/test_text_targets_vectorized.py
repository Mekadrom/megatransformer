"""Bit-identity test for the vectorized _build_text_targets (world compute_loss).

Target building strips placeholders, causal-shifts, ignores pad, and truncates/pads each
row to its logit count. The vectorized version must match the former per-item loop EXACTLY
(it feeds the text CE loss on a live run), so this checks it against a reference
reimplementation across randomized placeholder layouts and validity masks.
"""
import torch

from megatransformer.scripts.train.world.training import (
    TEXT_LOSS_IGNORE_INDEX as IGNORE,
    _build_text_targets,
)

PH = {32007, 32008}  # voice + image placeholder ids


def _ref(full_ids, non_ph_mask_full, valid_full, non_ph_input):
    target_list = []
    for b in range(full_ids.shape[0]):
        clean = full_ids[b][non_ph_mask_full[b]]
        clean_valid = valid_full[b][non_ph_mask_full[b]]
        shifted = clean[1:].masked_fill(~clean_valid[1:], IGNORE)
        K = int(non_ph_input[b].sum().item())
        if shifted.shape[0] >= K:
            target_list.append(shifted[:K])
        else:
            target_list.append(torch.cat(
                [shifted, shifted.new_full((K - shifted.shape[0],), IGNORE)]))
    max_len = max(t.shape[0] for t in target_list)
    padded = [t if t.shape[0] == max_len else
              torch.cat([t, t.new_full((max_len - t.shape[0],), IGNORE)])
              for t in target_list]
    return torch.stack(padded)


def _make_case(B, T, gen, ph_prob=0.15, invalid_prob=0.1):
    ids = torch.randint(1, 100, (B, T), generator=gen)
    ph_list = list(PH)
    ph_hit = torch.rand(B, T, generator=gen) < ph_prob
    ph_choice = torch.tensor(ph_list)[torch.randint(0, len(ph_list), (B, T), generator=gen)]
    ids = torch.where(ph_hit, ph_choice, ids)

    non_ph_mask_full = torch.ones_like(ids, dtype=torch.bool)
    for pid in PH:
        non_ph_mask_full &= (ids != pid)

    valid_full = torch.rand(B, T, generator=gen) >= invalid_prob

    text_input_ids = ids[:, :-1]
    non_ph_input = torch.ones_like(text_input_ids, dtype=torch.bool)
    for pid in PH:
        non_ph_input &= (text_input_ids != pid)
    return ids, non_ph_mask_full, valid_full, non_ph_input


def test_build_text_targets_bit_identical_random():
    gen = torch.Generator().manual_seed(0)
    for _ in range(50):
        B = int(torch.randint(1, 6, (1,), generator=gen))
        T = int(torch.randint(2, 40, (1,), generator=gen))
        ids, nphf, vf, nphi = _make_case(B, T, gen)
        got = _build_text_targets(ids, nphf, vf, nphi, IGNORE)
        ref = _ref(ids, nphf, vf, nphi)
        assert got.shape == ref.shape, f"shape {got.shape} != {ref.shape}"
        assert torch.equal(got, ref)


def test_all_valid_no_placeholders():
    gen = torch.Generator().manual_seed(3)
    ids, nphf, vf, nphi = _make_case(4, 20, gen, ph_prob=0.0, invalid_prob=0.0)
    assert torch.equal(_build_text_targets(ids, nphf, vf, nphi, IGNORE), _ref(ids, nphf, vf, nphi))


def test_heavy_placeholders_and_padding():
    gen = torch.Generator().manual_seed(7)
    ids, nphf, vf, nphi = _make_case(5, 30, gen, ph_prob=0.5, invalid_prob=0.4)
    assert torch.equal(_build_text_targets(ids, nphf, vf, nphi, IGNORE), _ref(ids, nphf, vf, nphi))


def test_row_all_placeholders():
    # One row entirely placeholders (no non-PH tokens) alongside normal rows.
    ids = torch.tensor([
        [5, 6, 7, 8, 9],
        [32007, 32007, 32007, 32007, 32007],
    ])
    nphf = torch.ones_like(ids, dtype=torch.bool)
    for pid in PH:
        nphf &= (ids != pid)
    vf = torch.ones_like(ids, dtype=torch.bool)
    nphi = torch.ones_like(ids[:, :-1], dtype=torch.bool)
    for pid in PH:
        nphi &= (ids[:, :-1] != pid)
    assert torch.equal(_build_text_targets(ids, nphf, vf, nphi, IGNORE), _ref(ids, nphf, vf, nphi))
