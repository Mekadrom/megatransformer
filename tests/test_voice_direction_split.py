"""Tests for per-direction voice corpus routing in MultimodalShardedDataset.

The world dataset emits two voice tasks from one modality: voice_synthesis
(text->voice) and voice_transcription (voice->text). By default both draw from
the same voice corpus. `voice_synthesis_shard_dir` overrides the synthesis
direction with a separate corpus (intended use: a clean subset for synthesis vs.
the clean+noisy superset for transcription).

These tests build tiny shards on disk and trace provenance via `speaker_ids`,
which we use as a unique per-example id so we can prove which corpus each sample
came from.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from megatransformer.scripts.data.world.dataset import MultimodalShardedDataset


# ---------------------------------------------------------------------------
# Shard fixtures
# ---------------------------------------------------------------------------

def _write_text_shard(shard_dir, n, id_base=0):
    os.makedirs(shard_dir, exist_ok=True)
    ids = torch.arange(id_base, id_base + n)
    shard = {
        "num_samples": n,
        "token_ids": ids.unsqueeze(1).repeat(1, 3),
        "text_lengths": torch.full((n,), 3, dtype=torch.long),
        "text": [f"text{int(i)}" for i in ids],
    }
    torch.save(shard, os.path.join(shard_dir, "shard_000000.pt"))


def _write_voice_shard(shard_dir, n, id_base, feat_dim=4, t=5):
    """Write one voice shard. `speaker_ids` carries a globally-unique id per
    example so tests can trace which corpus a sample was drawn from."""
    os.makedirs(shard_dir, exist_ok=True)
    ids = torch.arange(id_base, id_base + n)
    shard = {
        "num_samples": n,
        "features": torch.randn(n, feat_dim, t),
        "feature_lengths": torch.full((n,), t, dtype=torch.long),
        "token_ids": ids.unsqueeze(1).repeat(1, 3),
        "text_lengths": torch.full((n,), 3, dtype=torch.long),
        "speaker_ids": ids.clone(),  # unique provenance tracer
        "text": [f"voice{int(i)}" for i in ids],
    }
    torch.save(shard, os.path.join(shard_dir, "shard_000000.pt"))


# id ranges chosen disjoint so provenance is unambiguous
SUPERSET_BASE = 1000   # transcription corpus (clean + noisy)
SUBSET_BASE = 0        # synthesis corpus (clean subset)


def _task_index(dataset, task_name):
    for i, (name, _mod, _dir) in enumerate(dataset.task_types):
        if name == task_name:
            return i
    raise KeyError(task_name)


def _global_idx(dataset, task_name, within_task_idx):
    n_tasks = len(dataset.task_types)
    return _task_index(dataset, task_name) + within_task_idx * n_tasks


def _collect_ids_by_task(dataset):
    """Sweep the whole dataset, returning {task_name: set(speaker_id)}."""
    out = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if "voice_speaker_id" in sample:
            out.setdefault(sample["_task"], set()).add(int(sample["voice_speaker_id"]))
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_override_shares_corpus(tmp_path):
    """Legacy behavior: with no override, synthesis and transcription draw from
    the SAME corpus, and the same within_task_idx maps to the same example in
    both directions."""
    text_dir = str(tmp_path / "text")
    voice_dir = str(tmp_path / "voice")
    _write_text_shard(text_dir, 6)
    _write_voice_shard(voice_dir, 4, id_base=SUPERSET_BASE)

    ds = MultimodalShardedDataset(text_shard_dir=text_dir, voice_shard_dir=voice_dir)

    # Same underlying voice example for both directions at within_task_idx w.
    for w in range(4):
        syn = ds[_global_idx(ds, "voice_synthesis", w)]
        tra = ds[_global_idx(ds, "voice_transcription", w)]
        assert int(syn["voice_speaker_id"]) == int(tra["voice_speaker_id"])

    # All ids come from the single superset corpus.
    ids = _collect_ids_by_task(ds)
    assert ids["voice_synthesis"] <= set(range(SUPERSET_BASE, SUPERSET_BASE + 4))
    assert ids["voice_transcription"] <= set(range(SUPERSET_BASE, SUPERSET_BASE + 4))


def test_synthesis_override_routes_disjoint_corpora(tmp_path):
    """With an override, synthesis draws ONLY from the subset corpus and
    transcription ONLY from the superset corpus."""
    text_dir = str(tmp_path / "text")
    voice_dir = str(tmp_path / "voice_all")        # superset / transcription
    voice_syn_dir = str(tmp_path / "voice_clean")  # subset / synthesis
    _write_text_shard(text_dir, 6)
    _write_voice_shard(voice_dir, 4, id_base=SUPERSET_BASE)
    _write_voice_shard(voice_syn_dir, 2, id_base=SUBSET_BASE)

    ds = MultimodalShardedDataset(
        text_shard_dir=text_dir,
        voice_shard_dir=voice_dir,
        voice_synthesis_shard_dir=voice_syn_dir,
    )

    ids = _collect_ids_by_task(ds)
    subset_ids = set(range(SUBSET_BASE, SUBSET_BASE + 2))
    superset_ids = set(range(SUPERSET_BASE, SUPERSET_BASE + 4))

    assert ids["voice_synthesis"]  # non-empty
    assert ids["voice_transcription"]
    assert ids["voice_synthesis"] <= subset_ids, "synthesis leaked outside the clean subset"
    assert ids["voice_transcription"] <= superset_ids, "transcription leaked outside the superset"
    assert ids["voice_synthesis"].isdisjoint(ids["voice_transcription"])


def test_length_anchor_and_full_coverage(tmp_path):
    """Dataset length anchors to the largest corpus * n_tasks, and a full sweep
    reaches every example in BOTH the subset and the superset (smaller corpus
    wraps via modulo)."""
    text_dir = str(tmp_path / "text")
    voice_dir = str(tmp_path / "voice_all")
    voice_syn_dir = str(tmp_path / "voice_clean")
    _write_text_shard(text_dir, 6)        # largest -> anchor
    _write_voice_shard(voice_dir, 4, id_base=SUPERSET_BASE)
    _write_voice_shard(voice_syn_dir, 2, id_base=SUBSET_BASE)

    ds = MultimodalShardedDataset(
        text_shard_dir=text_dir,
        voice_shard_dir=voice_dir,
        voice_synthesis_shard_dir=voice_syn_dir,
    )

    n_tasks = len(ds.task_types)  # text + voice_synthesis + voice_transcription = 3
    assert len(ds) == 6 * n_tasks

    ids = _collect_ids_by_task(ds)
    # Every subset example reachable by synthesis; every superset by transcription.
    assert ids["voice_synthesis"] == set(range(SUBSET_BASE, SUBSET_BASE + 2))
    assert ids["voice_transcription"] == set(range(SUPERSET_BASE, SUPERSET_BASE + 4))


def test_override_requires_base_voice(tmp_path):
    """Providing a synthesis override without the base voice corpus is an error
    (the base anchors length and is the transcription source)."""
    text_dir = str(tmp_path / "text")
    voice_syn_dir = str(tmp_path / "voice_clean")
    _write_text_shard(text_dir, 6)
    _write_voice_shard(voice_syn_dir, 2, id_base=SUBSET_BASE)

    try:
        MultimodalShardedDataset(
            text_shard_dir=text_dir,
            voice_synthesis_shard_dir=voice_syn_dir,
        )
    except ValueError as e:
        assert "voice_shard_dir" in str(e)
    else:
        raise AssertionError("expected ValueError when override given without base voice corpus")


def test_sampler_indices_in_range_with_override(tmp_path):
    """Shard-aware sampler must produce only valid indices when a per-direction
    override changes one task's source offsets."""
    text_dir = str(tmp_path / "text")
    voice_dir = str(tmp_path / "voice_all")
    voice_syn_dir = str(tmp_path / "voice_clean")
    _write_text_shard(text_dir, 6)
    _write_voice_shard(voice_dir, 4, id_base=SUPERSET_BASE)
    _write_voice_shard(voice_syn_dir, 2, id_base=SUBSET_BASE)

    ds = MultimodalShardedDataset(
        text_shard_dir=text_dir,
        voice_shard_dir=voice_dir,
        voice_synthesis_shard_dir=voice_syn_dir,
    )

    sampler = ds.get_sampler(shuffle=True, batch_size=2, world_size=1, shard_aware=True)
    indices = list(sampler)
    assert indices, "sampler yielded nothing"
    assert all(0 <= i < len(ds) for i in indices)
    # Every index must still resolve to a valid sample (no out-of-range source idx).
    for i in indices:
        _ = ds[i]
