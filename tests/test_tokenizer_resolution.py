"""Tokenizer provenance: a checkpoint must say which vocabulary its ids belong to.

The bug being closed (docs/findings/world-text.md 2026-09-18): checkpoints carried text ids
with no record of their tokenizer, so eight consumers each re-derived it from their own flags
and each could be wrong independently -- silently, with WER computed on garbage decode.
"""
import json
import os
from argparse import Namespace

import pytest

from megatransformer.utils.tokenizer_resolution import (
    MISTRAL_DEFAULT, describe_resolution, read_checkpoint_meta, resolve_tokenizer_name,
)

SMOL = "HuggingFaceTB/SmolLM2-135M"


def _ckpt(tmp_path, name=SMOL):
    d = tmp_path / "checkpoint-1000"
    d.mkdir()
    (d / "pytorch_model.bin").write_bytes(b"")
    if name is not None:
        (d / "megatransformer_meta.json").write_text(
            json.dumps({"text_tokenizer_name": name, "special_token_base": 49152}))
    return str(d)


def test_default_is_unchanged_when_nothing_is_known():
    """Pre-2026-09-18 checkpoints with no flag must behave exactly as before."""
    assert resolve_tokenizer_name() == MISTRAL_DEFAULT
    assert resolve_tokenizer_name(config=None, args=Namespace()) == MISTRAL_DEFAULT


def test_flags_still_work_for_checkpoints_without_meta():
    assert resolve_tokenizer_name(args=Namespace(text_encoder_model=SMOL)) == SMOL
    assert resolve_tokenizer_name(args=Namespace(text_tokenizer=SMOL)) == SMOL
    # text_encoder_model wins over text_tokenizer
    assert resolve_tokenizer_name(
        args=Namespace(text_encoder_model=SMOL, text_tokenizer="other")) == SMOL


def test_checkpoint_meta_beats_a_contradicting_flag(tmp_path):
    """The config is a FACT about the data; a flag is an assertion. The fact wins."""
    ck = _ckpt(tmp_path)
    got = resolve_tokenizer_name(
        args=Namespace(checkpoint_path=ck, text_encoder_model="mistralai/Mistral-7B-v0.1"))
    assert got == SMOL


def test_force_overrides_everything(tmp_path):
    """Deliberate mismatch must stay possible, for measuring what a mismatch costs."""
    ck = _ckpt(tmp_path)
    got = resolve_tokenizer_name(args=Namespace(checkpoint_path=ck), force="deliberate/other")
    assert got == "deliberate/other"


def test_config_is_used_when_there_is_no_checkpoint_on_disk():
    cfg = Namespace(text_tokenizer_name=SMOL)
    assert resolve_tokenizer_name(config=cfg) == SMOL


def test_missing_or_corrupt_meta_never_raises(tmp_path):
    assert read_checkpoint_meta(None) == {}
    assert read_checkpoint_meta(str(tmp_path / "nope")) == {}
    d = tmp_path / "checkpoint-2000"
    d.mkdir()
    (d / "megatransformer_meta.json").write_text("{ not json")
    assert read_checkpoint_meta(str(d)) == {}
    assert resolve_tokenizer_name(args=Namespace(checkpoint_path=str(d))) == MISTRAL_DEFAULT


def test_a_weights_file_path_resolves_to_its_directory(tmp_path):
    """Callers pass either the dir or pytorch_model.bin; both must work."""
    ck = _ckpt(tmp_path)
    assert read_checkpoint_meta(os.path.join(ck, "pytorch_model.bin")).get(
        "text_tokenizer_name") == SMOL


def test_describe_names_the_source(tmp_path):
    """The failure mode is silent, so the log line is the tripwire."""
    ck = _ckpt(tmp_path)
    assert "megatransformer_meta.json" in describe_resolution(args=Namespace(checkpoint_path=ck))
    assert "--text_encoder_model" in describe_resolution(args=Namespace(text_encoder_model=SMOL))
    assert "DEFAULT" in describe_resolution()


def test_config_dataclass_defaults_to_none():
    """Old presets must not suddenly claim a tokenizer."""
    from megatransformer.config.world.world_model import WORLD_MODEL_CONFIGS
    assert WORLD_MODEL_CONFIGS["small_sum"].text_tokenizer_name is None
