"""Tests for text preprocessing: document packing, truncation fallback, token counting.

Each test builds a TextDatasetPreprocessor with a fake tokenizer (words ->
content-hashed token IDs, with reserved slots for EOS=0 and PAD=1), drives it
through synthetic examples, and verifies the invariants the pretraining path
actually depends on.
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch


class FakeTokenizer:
    """Minimal tokenizer: one token per whitespace-split word, deterministic IDs.

    Reserves ID 0 for EOS and ID 1 for PAD so tests can assert their positions.
    All content tokens land in [2, vocab_size).
    """

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        self.pad_token_id = 1

    def __len__(self):
        return self.vocab_size

    def _encode(self, text: str):
        # Deterministic per-word id, never 0 or 1.
        ids = []
        for w in text.split():
            if not w:
                continue
            h = abs(hash(w)) % (self.vocab_size - 2)
            ids.append(2 + h)
        return ids

    def __call__(
        self,
        texts,
        truncation: bool = False,
        padding: bool = False,
        max_length=None,
        return_attention_mask: bool = False,
    ):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            ids = self._encode(t)
            if truncation and max_length is not None:
                ids = ids[:max_length]
            out.append(ids)
        return {"input_ids": out}


def _make_args(**overrides):
    defaults = dict(
        tokenizer_name="fake",
        text_column="text",
        max_seq_len=10,
        min_text_len=1,
        packing="pack",
        max_tokens=None,
        dataset_name="fake",
        dataset_config=None,
        split="train",
        shard_size=100,
        gpu_batch_size=4,
        total_gpus=1,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_preprocessor(monkeypatch, tmp_path, **arg_overrides):
    import transformers
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda name: FakeTokenizer()),
    )
    from scripts.data.text.preprocess import TextDatasetPreprocessor

    args = _make_args(**arg_overrides)
    shard_fields = {"shard_idx": 0}
    batch_accumulators = {}
    stats = {
        "processed": 0,
        "saved": 0,
        "tokens_saved": 0,
        "skipped": defaultdict(int),
    }
    pp = TextDatasetPreprocessor(
        args, dataset=None, output_dir=str(tmp_path),
        shard_fields=shard_fields,
        batch_accumulators=batch_accumulators,
        stats_accumulator=stats,
        device="cpu",
    )
    return pp, stats, shard_fields, str(tmp_path)


def _drain(pp):
    """Finalize any pending batch so emitted blocks are counted."""
    pp.process_and_accumulate()


# ---------------------------------------------------------------------------
# Pack mode
# ---------------------------------------------------------------------------


def test_pack_mode_emits_fixed_length_blocks(monkeypatch, tmp_path):
    """Every block emitted in pack mode is exactly max_seq_len tokens."""
    pp, stats, sf, _ = _make_preprocessor(monkeypatch, tmp_path, max_seq_len=10)
    for _ in range(20):
        pp.preprocess_example({"text": "alpha beta gamma delta"})  # 4 tokens
    _drain(pp)
    assert len(sf["shard_token_ids"]) >= 5
    for t in sf["shard_token_ids"]:
        assert t.shape[-1] == 10, f"non-uniform block length: {t.shape[-1]}"


def test_pack_mode_inserts_eos_between_documents(monkeypatch, tmp_path):
    """EOS (id=0) should appear at document boundaries before any new content."""
    pp, stats, sf, _ = _make_preprocessor(monkeypatch, tmp_path, max_seq_len=100)
    for text in ["alpha", "beta", "gamma"]:
        pp.preprocess_example({"text": text})
    _drain(pp)
    # Buffer holds (content + eos) × 3 = 6 tokens, no block emitted yet.
    assert len(sf["shard_token_ids"]) == 0
    buf = pp._token_buffer
    assert len(buf) == 6
    assert buf[1] == 0, "expected EOS after first doc's content token"
    assert buf[3] == 0, "expected EOS after second doc's content token"
    assert buf[5] == 0, "expected EOS after third doc's content token"
    # And the content tokens themselves are non-EOS
    assert buf[0] != 0 and buf[2] != 0 and buf[4] != 0


def test_pack_mode_buffer_persists_across_examples(monkeypatch, tmp_path):
    """Tokens from separate documents must accumulate into the same block."""
    pp, stats, sf, _ = _make_preprocessor(monkeypatch, tmp_path, max_seq_len=10)
    # Each doc = 3 content tokens + 1 EOS = 4 tokens per doc. Three docs = 12
    # tokens in the buffer; should emit one block of 10 and leave 2.
    for _ in range(3):
        pp.preprocess_example({"text": "a b c"})
    _drain(pp)
    assert len(sf["shard_token_ids"]) == 1
    assert sf["shard_token_ids"][0].shape[-1] == 10
    assert len(pp._token_buffer) == 2


def test_pack_mode_long_document_spans_multiple_blocks(monkeypatch, tmp_path):
    """A single document longer than max_seq_len should produce multiple blocks,
    never silent truncation."""
    pp, stats, sf, _ = _make_preprocessor(monkeypatch, tmp_path, max_seq_len=10)
    words = " ".join(f"w{i}" for i in range(30))  # 30 content tokens + 1 EOS
    pp.preprocess_example({"text": words})
    _drain(pp)
    # 31 tokens / 10 = 3 full blocks + 1 leftover
    assert len(sf["shard_token_ids"]) == 3
    for t in sf["shard_token_ids"]:
        assert t.shape[-1] == 10
    assert len(pp._token_buffer) == 1


def test_pack_mode_tokens_saved_matches_blocks_times_seq_len(monkeypatch, tmp_path):
    """stats['tokens_saved'] must equal num_blocks × max_seq_len in pack mode."""
    pp, stats, sf, _ = _make_preprocessor(monkeypatch, tmp_path, max_seq_len=10, shard_size=100)
    # 50 docs × (2 content + 1 EOS) = 150 tokens → 15 blocks of 10.
    for _ in range(50):
        pp.preprocess_example({"text": "aa bb"})
    _drain(pp)
    num_blocks = len(sf["shard_token_ids"])
    assert num_blocks == 15
    assert stats["tokens_saved"] == num_blocks * 10
    assert stats["saved"] == num_blocks  # one "sample" = one block


def test_pack_mode_shard_has_no_raw_text_field(monkeypatch, tmp_path):
    """Blocks can span multiple documents, so pack-mode shards drop the
    raw-text field to avoid ambiguity."""
    pp, stats, sf, out_dir = _make_preprocessor(monkeypatch, tmp_path, max_seq_len=5, shard_size=3)
    # Produce at least 3 blocks so flush triggers.
    for _ in range(20):
        pp.preprocess_example({"text": "a b c"})
    _drain(pp)
    pp.flush_shard()

    shard = torch.load(os.path.join(out_dir, "shard_000000.pt"), map_location="cpu", weights_only=True)
    assert "text" not in shard, "pack mode should not persist raw text strings"
    assert "token_ids" in shard
    assert "text_lengths" in shard
    # All block lengths equal max_seq_len
    assert int(shard["text_lengths"].max()) == 5
    assert int(shard["text_lengths"].min()) == 5


def test_pack_mode_skips_eos_when_tokenizer_has_no_eos(monkeypatch, tmp_path):
    """If tokenizer.eos_token_id is None, pack mode must not insert separators."""
    pp, stats, sf, _ = _make_preprocessor(monkeypatch, tmp_path, max_seq_len=100)
    pp.tokenizer.eos_token_id = None
    for _ in range(3):
        pp.preprocess_example({"text": "a b c"})
    _drain(pp)
    # Exactly 9 tokens, no separators.
    assert len(pp._token_buffer) == 9
    assert all(t != 0 for t in pp._token_buffer)  # no EOS tokens inserted


# ---------------------------------------------------------------------------
# Truncate mode (legacy backward-compatibility)
# ---------------------------------------------------------------------------


def test_truncate_mode_one_sample_per_document(monkeypatch, tmp_path):
    """Truncate mode preserves 1:1 document-to-sample mapping."""
    pp, stats, sf, _ = _make_preprocessor(monkeypatch, tmp_path, packing="truncate", max_seq_len=10)
    texts = [
        "single",                        # 1 token
        "a b c d e f g h i j k l",        # 12 tokens -> truncated to 10
        "two words",                      # 2 tokens
    ]
    for t in texts:
        pp.preprocess_example({"text": t})
    _drain(pp)

    assert stats["saved"] == 3
    assert len(sf["shard_token_ids"]) == 3
    assert sf["shard_token_ids"][0].shape[-1] == 1
    assert sf["shard_token_ids"][1].shape[-1] == 10  # truncated
    assert sf["shard_token_ids"][2].shape[-1] == 2
    # Token count tracks real (post-truncation) tokens emitted
    assert stats["tokens_saved"] == 1 + 10 + 2


def test_truncate_mode_preserves_raw_text(monkeypatch, tmp_path):
    """Truncate-mode shards must retain the raw text column (1:1 w/ samples)."""
    pp, stats, sf, out_dir = _make_preprocessor(
        monkeypatch, tmp_path, packing="truncate", max_seq_len=10
    )
    pp.preprocess_example({"text": "hello world"})
    pp.preprocess_example({"text": "second document"})
    _drain(pp)
    pp.flush_shard()

    shard = torch.load(os.path.join(out_dir, "shard_000000.pt"), map_location="cpu", weights_only=True)
    assert "text" in shard
    assert shard["text"] == ["hello world", "second document"]
    assert shard["token_ids"].shape[0] == 2


# ---------------------------------------------------------------------------
# Shared: skip behavior and batch boundary correctness
# ---------------------------------------------------------------------------


def test_empty_text_is_skipped_in_both_modes(monkeypatch, tmp_path):
    for mode in ("pack", "truncate"):
        pp, stats, sf, _ = _make_preprocessor(monkeypatch, tmp_path, packing=mode, min_text_len=1)
        pp.preprocess_example({"text": ""})
        pp.preprocess_example({"text": "   "})
        pp.preprocess_example({"text": None})
        _drain(pp)
        assert stats["skipped"]["too_short"] >= 1
        assert stats["saved"] == 0


def test_pack_mode_respects_batch_boundary(monkeypatch, tmp_path):
    """Auto-flush at batch boundary should not drop tokens or duplicate them."""
    # gpu_batch_size=4 means every 4th preprocess_example triggers a flush of
    # the pending batch. Pack mode should accumulate seamlessly across these.
    pp, stats, sf, _ = _make_preprocessor(
        monkeypatch, tmp_path, max_seq_len=10, gpu_batch_size=4, shard_size=1000
    )
    # 20 docs × (3 content + 1 EOS) = 80 tokens → 8 full blocks.
    for _ in range(20):
        pp.preprocess_example({"text": "x y z"})
    _drain(pp)
    assert len(sf["shard_token_ids"]) == 8
    assert stats["tokens_saved"] == 80
