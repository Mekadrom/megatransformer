"""Dotted-path column access for WebDataset corpora (Emilia).

Emilia's features are ['json', 'mp3', '__key__', '__url__'] -- text, speaker and duration all
live inside the 'json' struct, so a flat lookup finds nothing and returns empty labels
SILENTLY. These tests pin both halves: nested paths resolve, and plain names are untouched.
"""
import pytest

from megatransformer.scripts.data.voice.preprocess import column_value, column_present


FLAT = {"audio": "AUDIO", "text_original": "hello there", "speaker_id": "spk7",
        "audio_duration": 4.25}

EMILIA = {
    "__key__": "EN-B000000/EN_B00000_S00000_W000000",
    "mp3": {"bytes": b"...", "path": "x.mp3"},
    "json": {"id": "EN_B00000_S00000_W000000", "text": "a spontaneous utterance",
             "duration": 7.5, "speaker": "EN_B00000_S00000", "language": "en",
             "dnsmos": 3.41},
}


def test_plain_names_behave_exactly_as_before():
    for k, v in FLAT.items():
        assert column_value(FLAT, k) == v
    assert column_value(FLAT, "nope") is None
    assert column_value(FLAT, "nope", "dflt") == "dflt"


def test_dotted_path_walks_into_the_struct():
    assert column_value(EMILIA, "json.text") == "a spontaneous utterance"
    assert column_value(EMILIA, "json.speaker") == "EN_B00000_S00000"
    assert column_value(EMILIA, "json.duration") == 7.5
    assert column_value(EMILIA, "mp3.path") == "x.mp3"


def test_missing_nested_key_returns_default_rather_than_raising():
    """Every call site used .get() semantics; a dotted path must not start raising."""
    assert column_value(EMILIA, "json.nonexistent") is None
    assert column_value(EMILIA, "json.nonexistent", -1) == -1
    assert column_value(EMILIA, "absent.text") is None
    # walking THROUGH a non-dict must not raise either
    assert column_value(EMILIA, "json.text.deeper") is None


def test_none_or_empty_path_is_the_default():
    assert column_value(EMILIA, None) is None
    assert column_value(EMILIA, None, "d") == "d"
    assert column_value(EMILIA, "", "d") == "d"


def test_column_present_distinguishes_missing_from_none_valued():
    assert column_present(EMILIA, "json.text") is True
    assert column_present(EMILIA, "json.nope") is False
    assert column_present(EMILIA, None) is False
    # a key that EXISTS but holds None is present -- the duration branch relies on this
    assert column_present({"d": None}, "d") is True
    assert column_value({"d": None}, "d", "fallback") is None


def test_emilia_flag_set_resolves_end_to_end():
    """The exact flags an Emilia preprocessing run would pass."""
    got = {
        "audio": column_value(EMILIA, "mp3"),
        "text": column_value(EMILIA, "json.text"),
        "speaker": column_value(EMILIA, "json.speaker"),
        "duration": float(column_value(EMILIA, "json.duration")),
    }
    assert got["audio"]["path"] == "x.mp3"
    assert got["text"] == "a spontaneous utterance"
    assert got["speaker"] == "EN_B00000_S00000"
    assert got["duration"] == pytest.approx(7.5)
