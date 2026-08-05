"""Unit tests for voice-transcript normalization (normalize_transcript).

The inputs are real / real-derived LibriTTS-R transcripts (LibriTTS-R is mixed-case WITH
punctuation, and heavy on quoted dialogue). Every case asserts the CORRECT normalized output.

The bug groups are marked xfail(strict=True): they encode the intended behavior and currently
FAIL, so the suite stays green while documenting the defects. When normalize_transcript is
fixed, the strict xfail turns into an XPASS failure -- a reminder to drop the marker. The bugs
were found from the mimi cache, where ~14% of transcripts carry the group-A artifact
(e.g. '"Yes.".', 'disappeared!".', 'Gryce?".').
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
# BUG A: terminal punctuation followed by a CLOSING QUOTE. normalize_transcript
# checks only text[-1], so the closing quote (") is treated as "no punctuation" and a
# spurious period is appended: '."' -> '.".', '!"' -> '!".', '?"' -> '?".'.
# This is the '"Yes.".' artifact and the most common one (~14% of the mimi cache).
# ---------------------------------------------------------------------------
BUG_QUOTE_TERMINAL = [
    ('"Yes."', '"Yes."'),
    ('How quickly he disappeared!"', 'How quickly he disappeared!"'),
    ('"So they tell me."', '"So they tell me."'),
    ('What do you make of it, Gryce?"', 'What do you make of it, Gryce?"'),
    ('"Look at me well; in sooth I\'m Beatrice!"',
     '"Look at me well; in sooth I\'m Beatrice!"'),
]


@pytest.mark.xfail(strict=True, reason="BUG A: appends '.' after a closing quote that already "
                                       "follows terminal punctuation (.\"/!\"/?\") -> '.\".'")
@pytest.mark.parametrize("raw,expected", BUG_QUOTE_TERMINAL)
def test_bug_quote_terminal(raw, expected):
    assert normalize_transcript(raw) == expected


# ---------------------------------------------------------------------------
# BUG B: the first word is preceded by an OPENING QUOTE, so neither the sentence-start
# regex (^ must be immediately followed by [a-z]) nor the first-char check ('"'.islower()
# is False) capitalizes it. '"but ...' stays lowercase.
# ---------------------------------------------------------------------------
BUG_OPENING_QUOTE_CAP = [
    ('"but it works!', '"But it works!'),
    ("'tis a fine morning", "'Tis a fine morning."),
    ('"come in," said the host.', '"Come in," said the host.'),
]


@pytest.mark.xfail(strict=True, reason="BUG B: a first word preceded by an opening quote "
                                       "is not capitalized ('\"but' stays lowercase)")
@pytest.mark.parametrize("raw,expected", BUG_OPENING_QUOTE_CAP)
def test_bug_opening_quote_capitalization(raw, expected):
    assert normalize_transcript(raw) == expected


# ---------------------------------------------------------------------------
# BUG C: a transcript ending in NON-terminal punctuation (comma/semicolon/colon) -- e.g. a
# LibriTTS-R sentence split mid-clause -- gets a period appended anyway: ',' -> ',.'.
# ---------------------------------------------------------------------------
BUG_TRAILING_PUNCT = [
    ("Uplifting light the reinvested flesh,", "Uplifting light the reinvested flesh,"),
    ("less dear and less delightful;", "Less dear and less delightful;"),
    ("as follows:", "As follows:"),
]


@pytest.mark.xfail(strict=True, reason="BUG C: appends '.' after non-terminal trailing "
                                       "punctuation (',' ';' ':') -> ',.' ';.' ':.'")
@pytest.mark.parametrize("raw,expected", BUG_TRAILING_PUNCT)
def test_bug_trailing_nonterminal_punct(raw, expected):
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
