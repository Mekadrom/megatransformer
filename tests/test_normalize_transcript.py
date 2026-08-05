"""Unit tests for voice-transcript normalization (normalize_transcript).

The inputs are real / real-derived LibriTTS-R transcripts (LibriTTS-R is mixed-case WITH
punctuation, and heavy on quoted dialogue). Every case asserts the CORRECT normalized output.

The three "regression" groups below guard bugs that were found from the mimi cache (~14% of
transcripts carried the group-A artifact, e.g. '"Yes.".', 'disappeared!".', 'Gryce?".') and
fixed in normalize_transcript. They previously required xfail; they now pass.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from megatransformer.scripts.data.voice.preprocess import normalize_transcript


# ---------------------------------------------------------------------------
# Cases the normalizer already handles correctly.
# ---------------------------------------------------------------------------
CORRECT = [
    # LibriSpeech-style ALL CAPS -> sentence case + trailing period (the documented purpose).
    ("THE QUICK BROWN FOX JUMPED", "The quick brown fox jumped."),
    ("HELLO, WORLD!", "Hello, world!"),
    # Already-normalized text is left alone.
    ("Already normal text.", "Already normal text."),
    ("The weapon must still have been there.", "The weapon must still have been there."),
    ("Perhaps you can.", "Perhaps you can."),
    ("Purgatorio: Canto twenty nine.", "Purgatorio: Canto twenty nine."),
    # Lowercase -> capitalized first letter + trailing period.
    ("the cat sat on the mat", "The cat sat on the mat."),
    ("it's a lovely day", "It's a lovely day."),
    ("chapter twenty nine", "Chapter twenty nine."),
    ("Chapter 29", "Chapter 29."),
    # Multi-sentence: capitalize after each sentence terminator.
    ("hello world. this is fine", "Hello world. This is fine."),
    ("one thing. another thing? a third!", "One thing. Another thing? A third!"),
    # Already ends in terminal punctuation -> no extra period added.
    ("What do you make of it?", "What do you make of it?"),
    ("Stop right there!", "Stop right there!"),
    # A quoted fragment already ending in terminal punctuation (no trailing quote).
    ('"But the blood?', '"But the blood?'),
    # Whitespace is stripped.
    ("  leading and trailing  ", "Leading and trailing."),
    ("\tPerhaps you can.\n", "Perhaps you can."),
]


@pytest.mark.parametrize("raw,expected", CORRECT)
def test_normalize_correct(raw, expected):
    assert normalize_transcript(raw) == expected


# ---------------------------------------------------------------------------
# Regression A: terminal punctuation followed by a CLOSING QUOTE. The old code checked only
# text[-1], so the closing quote (") read as "no punctuation" and a spurious period was
# appended: '."' -> '.".', '!"' -> '!".', '?"' -> '?".'. The '"Yes.".' artifact (~14% of cache).
# ---------------------------------------------------------------------------
CLOSING_QUOTE_TERMINAL = [
    ('"Yes."', '"Yes."'),
    ('How quickly he disappeared!"', 'How quickly he disappeared!"'),
    ('"So they tell me."', '"So they tell me."'),
    ('What do you make of it, Gryce?"', 'What do you make of it, Gryce?"'),
    ('"Look at me well; in sooth I\'m Beatrice!"',
     '"Look at me well; in sooth I\'m Beatrice!"'),
]


@pytest.mark.parametrize("raw,expected", CLOSING_QUOTE_TERMINAL)
def test_closing_quote_terminal(raw, expected):
    assert normalize_transcript(raw) == expected


# ---------------------------------------------------------------------------
# Regression B: the first word preceded by an OPENING QUOTE must still be capitalized
# ('"but ...' -> '"But ...'). The old ^([a-z]) regex and first-char check both skipped it.
# ---------------------------------------------------------------------------
OPENING_QUOTE_CAP = [
    ('"but it works!', '"But it works!'),
    ("'tis a fine morning", "'Tis a fine morning."),
    ('"come in," said the host.', '"Come in," said the host.'),
]


@pytest.mark.parametrize("raw,expected", OPENING_QUOTE_CAP)
def test_opening_quote_capitalization(raw, expected):
    assert normalize_transcript(raw) == expected


# ---------------------------------------------------------------------------
# Regression C: a transcript ending in NON-terminal punctuation (comma/semicolon/colon) -- e.g.
# a LibriTTS-R mid-clause sentence split -- must be left alone, not get a period: ',' !-> ',.'.
# ---------------------------------------------------------------------------
TRAILING_NONTERMINAL_PUNCT = [
    ("Uplifting light the reinvested flesh,", "Uplifting light the reinvested flesh,"),
    ("less dear and less delightful;", "Less dear and less delightful;"),
    ("as follows:", "As follows:"),
]


@pytest.mark.parametrize("raw,expected", TRAILING_NONTERMINAL_PUNCT)
def test_trailing_nonterminal_punct(raw, expected):
    assert normalize_transcript(raw) == expected


# ---------------------------------------------------------------------------
# Edge cases: falsy / whitespace-only input is returned unchanged (guard clause).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("", ""),
    ("   ", "   "),
    (None, None),
])
def test_normalize_edge_cases(raw, expected):
    assert normalize_transcript(raw) == expected
