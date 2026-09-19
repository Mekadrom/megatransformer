"""Spell-out normalization for synthesis input.

Measured justification (docs/findings/world-voice.md 2026-09-19): only 3.67% of LibriHeavy
transcripts contain a digit, and much of that is non-speech (chapter headers, OCR noise), so
the digit form is the weak regime and the spelled-out form is the strong one.
"""
import pytest

from megatransformer.utils.text_normalization import (
    cardinal, ordinal, spell_out_numerics, year,
)


@pytest.mark.parametrize("n,want", [
    (0, "zero"), (7, "seven"), (13, "thirteen"), (20, "twenty"), (42, "forty two"),
    (100, "one hundred"), (101, "one hundred one"), (999, "nine hundred ninety nine"),
    (1000, "one thousand"), (5000, "five thousand"),
    (1042, "one thousand forty two"),
    (1_000_000, "one million"), (2_500_000, "two million five hundred thousand"),
])
def test_cardinal(n, want):
    assert cardinal(n) == want


@pytest.mark.parametrize("n,want", [
    (1, "first"), (2, "second"), (3, "third"), (4, "fourth"), (5, "fifth"),
    (8, "eighth"), (9, "ninth"), (12, "twelfth"), (16, "sixteenth"),
    (20, "twentieth"), (22, "twenty second"), (30, "thirtieth"), (100, "one hundredth"),
])
def test_ordinal(n, want):
    assert ordinal(n) == want


@pytest.mark.parametrize("n,want", [
    (1942, "nineteen forty two"), (1905, "nineteen oh five"),
    (2000, "two thousand"), (1900, "nineteen hundred"), (2026, "twenty twenty six"),
])
def test_year(n, want):
    assert year(n) == want


@pytest.mark.parametrize("raw,want", [
    ("$5,000", "five thousand dollars"),
    ("$1", "one dollar"),
    ("£3", "three pounds"),
    ("50%", "fifty percent"),
    ("the 16th of May", "the sixteenth of May"),
    ("3.5 miles", "three point five miles"),
    ("in 1942", "in nineteen forty two"),
    ("2 tablespoonfuls", "two tablespoonfuls"),
])
def test_real_corpus_shapes(raw, want):
    """Every input here is a form actually observed in the LibriHeavy sample."""
    assert spell_out_numerics(raw) == want


def test_symbols():
    assert spell_out_numerics("salt & pepper") == "salt and pepper"
    assert spell_out_numerics("a + b = c") == "a plus b equals c"


def test_is_a_noop_on_plain_text():
    """Safe to apply unconditionally: text with no digits or symbols is untouched."""
    for s in ["The quick brown fox jumped.", "Already spelled out one hundred times.",
              '"Yes," she said.', "hyphen-joined words, commas; colons: fine."]:
        assert spell_out_numerics(s) == s


def test_is_idempotent():
    for s in ["$5,000", "the 16th", "50%", "in 1942 & 1943"]:
        once = spell_out_numerics(s)
        assert spell_out_numerics(once) == once


def test_empty_and_none_pass_through():
    assert spell_out_numerics(None) is None
    assert spell_out_numerics("") == ""
    assert spell_out_numerics("   ") == "   "


def test_years_can_be_disabled():
    assert spell_out_numerics("in 1942", years=False) == "in one thousand nine hundred forty two"


def test_comma_separated_four_digits_is_a_quantity_not_a_year():
    """'1,942' was written as a quantity; '1942' was probably a year."""
    assert spell_out_numerics("1,942 sheep") == "one thousand nine hundred forty two sheep"
    assert spell_out_numerics("1942 sheep") == "nineteen forty two sheep"


def test_currency_is_consumed_before_the_bare_number_rule():
    """Ordering bug guard: a bare-number pass first would leave an orphan '$'."""
    out = spell_out_numerics("another $5,000 in similar fashion")
    assert "$" not in out and out == "another five thousand dollars in similar fashion"
