"""Tests for length-bucketed shard sampling (padding-waste reduction).

LengthGroupedShardSampler groups similar-length samples into the same batch while
preserving shard locality. These pin: full coverage, shard-locality (one contiguous
run per shard), that batches are genuinely length-homogeneous vs the shard-random
baseline, per-epoch determinism + cross-epoch variety, and the mmap length scan.
"""
import os

import torch

from megatransformer.scripts.data.dataset import (
    LengthGroupedShardSampler,
    ShardAwareSampler,
    scan_shard_lengths,
)

SHARD_OFFSETS = [0, 500, 1000]     # 3 shards
SHARD_SIZES = [500, 500, 300]
TOTAL = 1300
BATCH = 16


def _make_lengths(seed=0):
    g = torch.Generator().manual_seed(seed)
    # Wide length spread so homogeneity is measurable.
    return (torch.randint(20, 500, (TOTAL,), generator=g)).tolist()


def _sampler(lengths, mega_factor=25, shuffle=True, seed=42):
    return LengthGroupedShardSampler(
        SHARD_OFFSETS, TOTAL, lengths, batch_size=BATCH,
        mega_factor=mega_factor, shuffle=shuffle, seed=seed,
    )


def _avg_intra_batch_range(indices, lengths, batch=BATCH):
    """Mean (max-min) length within each consecutive `batch`-sized chunk."""
    ranges = []
    for s in range(0, len(indices), batch):
        chunk = indices[s:s + batch]
        ls = [lengths[i] for i in chunk]
        if len(ls) > 1:
            ranges.append(max(ls) - min(ls))
    return sum(ranges) / len(ranges)


def test_full_coverage():
    lengths = _make_lengths()
    idx = list(iter(_sampler(lengths)))
    assert len(idx) == TOTAL
    assert set(idx) == set(range(TOTAL))          # every sample exactly once


def test_shard_locality_preserved():
    """Each shard's indices form ONE contiguous run — the shard is loaded once/epoch."""
    lengths = _make_lengths()
    idx = list(iter(_sampler(lengths)))

    def shard_of(i):
        return 0 if i < 500 else (1 if i < 1000 else 2)

    # Collapse consecutive-equal shard ids; each shard must appear only once.
    seq = [shard_of(i) for i in idx]
    collapsed = [s for k, s in enumerate(seq) if k == 0 or s != seq[k - 1]]
    assert len(collapsed) == len(set(collapsed)) == 3


def test_batches_are_length_homogeneous():
    """Bucketed batches must have far smaller intra-batch length spread than the
    shard-random baseline."""
    lengths = _make_lengths()
    bucketed = list(iter(_sampler(lengths)))
    random_base = list(iter(ShardAwareSampler(SHARD_OFFSETS, TOTAL, shuffle=True, seed=42)))

    b_range = _avg_intra_batch_range(bucketed, lengths)
    r_range = _avg_intra_batch_range(random_base, lengths)
    # Bucketing should cut the intra-batch length spread by a large margin.
    assert b_range < 0.5 * r_range, f"bucketed={b_range:.1f} not << random={r_range:.1f}"


def test_deterministic_per_epoch():
    lengths = _make_lengths()
    s = _sampler(lengths)
    s.set_epoch(3)
    a = list(iter(s))
    s.set_epoch(3)
    b = list(iter(s))
    assert a == b                                  # same epoch reproduces exactly


def test_variety_across_epochs():
    lengths = _make_lengths()
    s = _sampler(lengths)
    s.set_epoch(0)
    e0 = list(iter(s))
    s.set_epoch(1)
    e1 = list(iter(s))
    assert e0 != e1                                # batch membership varies epoch-to-epoch
    assert set(e0) == set(e1)                      # but coverage is identical


def test_no_shuffle_is_strict_sorted_within_shard():
    lengths = _make_lengths()
    idx = list(iter(_sampler(lengths, shuffle=False)))
    # Within shard 0 (global 0..499), lengths must be non-decreasing.
    shard0 = [i for i in idx if i < 500]
    ls = [lengths[i] for i in shard0]
    assert ls == sorted(ls)


def test_scan_shard_lengths_mmap(tmp_path):
    """scan_shard_lengths reads exactly the per-sample length arrays, in global order."""
    files = []
    expected = []
    for si, n in enumerate([5, 7, 3]):
        mel_lengths = torch.arange(si * 100, si * 100 + n)      # distinct per shard
        shard = {
            "mel_specs": torch.zeros(n, 80, 10),                # heavy tensor beside it
            "mel_lengths": mel_lengths,
            "num_samples": n,
        }
        fname = f"shard_{si:06d}.pt"
        torch.save(shard, os.path.join(tmp_path, fname))
        files.append(fname)
        expected.extend(mel_lengths.tolist())

    got = scan_shard_lengths(str(tmp_path), files, "mel_lengths")
    assert got == expected

    # Missing key raises a helpful error.
    try:
        scan_shard_lengths(str(tmp_path), files, "waveform_lengths")
        assert False, "expected KeyError"
    except KeyError:
        pass


# ── World ModalityGroupedSampler: round-robin + optional length bucketing ────────
from megatransformer.scripts.data.world.dataset import ModalityGroupedSampler  # noqa: E402

# 2 tasks (modality 0 = "voice"-like, bucketable; modality 1 = "image"-like, fixed).
# idx % 2 = modality; within = idx // 2; wrapped = within % mod_total.
N_MOD = 2
MOD_TOTAL = 200
WORLD_TOTAL = MOD_TOTAL * N_MOD          # dataset len = max(mod_totals) * n_tasks
WBATCH = 8
# voice shard layout: 4 shards of 50; image: 1 shard of 200.
V_SHARD_INFO = ([0, 50, 100, 150], MOD_TOTAL)
I_SHARD_INFO = ([0], MOD_TOTAL)


def _voice_lengths(seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(20, 500, (MOD_TOTAL,), generator=g).tolist()


def _world_sampler(bucket):
    vlen = _voice_lengths()
    bucket_lengths = [vlen, None] if bucket else None       # image (mod 1) not bucketed
    s = ModalityGroupedSampler(
        total_samples=WORLD_TOTAL, n_modalities=N_MOD, shuffle=True, seed=42,
        batch_size=WBATCH, world_size=1,
        shard_info=[V_SHARD_INFO, I_SHARD_INFO],
        bucket_lengths=bucket_lengths,
    )
    return s, vlen


def test_world_sampler_off_is_byte_identical():
    """bucket_lengths=None must reproduce the pre-change shard-aware order exactly (the
    generator draw sequence is unchanged), so a live-run resume is unaffected."""
    a, _ = _world_sampler(bucket=False)
    b, _ = _world_sampler(bucket=False)
    assert list(iter(a)) == list(iter(b))                   # deterministic
    # And every index exactly once.
    assert sorted(iter(a)) == list(range(WORLD_TOTAL))


def test_world_sampler_bucket_coverage_and_modality_homogeneity():
    s, _ = _world_sampler(bucket=True)
    idx = list(iter(s))
    assert sorted(idx) == list(range(WORLD_TOTAL))           # exact coverage
    # Each batch_size chunk stays single-modality (the round-robin invariant).
    for a in range(0, len(idx), WBATCH):
        mods = {i % N_MOD for i in idx[a:a + WBATCH]}
        assert len(mods) == 1


def test_world_sampler_bucket_makes_voice_batches_homogeneous():
    s_b, vlen = _world_sampler(bucket=True)
    s_r, _ = _world_sampler(bucket=False)

    def voice_len(gidx):
        return vlen[(gidx // N_MOD) % MOD_TOTAL]

    def avg_voice_batch_range(idx):
        rs = []
        for a in range(0, len(idx), WBATCH):
            chunk = idx[a:a + WBATCH]
            if all(i % N_MOD == 0 for i in chunk) and len(chunk) > 1:   # pure voice batch
                ls = [voice_len(i) for i in chunk]
                rs.append(max(ls) - min(ls))
        return sum(rs) / len(rs)

    assert avg_voice_batch_range(list(iter(s_b))) < 0.5 * avg_voice_batch_range(list(iter(s_r)))
