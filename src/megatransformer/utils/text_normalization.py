"""Spell out digits and symbols so synthesis input matches the training distribution.

WHY. Measured on 12,000 LibriHeavy transcripts (2026-09-19): only 3.67% contain a digit, and
much of that fraction is not speech-bearing at all -- chapter headers ("8 In the Attic"), OCR
noise ("pah 1 chock"). Meanwhile the spelled-out forms carry the bulk of the signal ("one"
838, "first" 237, "hundred" 62). So the digit -> speech mapping is learned from a small, noisy
slice while the word form is learned from everything else. A user typing "16" or "50%" lands
in the weak regime; the same content spelled out lands in the strong one.

This is the job CosyVoice 2 does in its text frontend, which we bypass entirely because we
feed unit ids directly rather than going through its tokenizer.

NO NEW DEPENDENCY. `num2words` is not installed and dependency state is not ours to change, so
the English cardinal/ordinal expansion is implemented here. It covers 0..999,999,999,999.

⚠️ WHAT THIS DOES NOT FIX. It maps an unusual SURFACE FORM onto a word the model knows better.
It cannot help a word that is simply rare in the corpus -- "sixteenth" occurs twice in 12,000
transcripts, and "16th" -> "sixteenth" just routes you to a word with ~160 corpus occurrences.
That is the rare-word ceiling (recall 0.623 very-rare vs 0.984 very-common) and only data
scale moves it.

⚠️ YEAR AMBIGUITY is genuine and unresolvable from the string alone. "1942" is "nineteen forty
two" as a year and "one thousand nine hundred forty two" as a quantity. Four-digit integers in
[1100, 2099] are read year-style, which is right for prose and wrong for "1942 dollars". Pass
`years=False` to disable.
"""
import re
from typing import Optional

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand")]

# Irregular ordinals; everything else takes the regular -th / -ieth transformation.
_ORDINAL_IRREGULAR = {
    "one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
    "nine": "ninth", "twelve": "twelfth",
}

_SYMBOLS = {"&": " and ", "+": " plus ", "=": " equals ", "@": " at ", "#": " number ",
            "/": " slash ", "~": " approximately ", "<": " less than ", ">": " greater than "}

_CURRENCY = {"$": ("dollar", "dollars"), "£": ("pound", "pounds"),
             "€": ("euro", "euros"), "¥": ("yen", "yen")}


def _under_thousand(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, r = divmod(n, 10)
        return _TENS[t] + (f" {_ONES[r]}" if r else "")
    h, r = divmod(n, 100)
    return f"{_ONES[h]} hundred" + (f" {_under_thousand(r)}" if r else "")


def cardinal(n: int) -> str:
    """0 -> 'zero', 1042 -> 'one thousand forty two'."""
    if n < 0:
        return "minus " + cardinal(-n)
    if n < 1000:
        return _under_thousand(n)
    for value, name in _SCALES:
        if n >= value:
            q, r = divmod(n, value)
            return f"{cardinal(q)} {name}" + (f" {cardinal(r)}" if r else "")
    return _under_thousand(n)


def ordinal(n: int) -> str:
    """1 -> 'first', 16 -> 'sixteenth', 42 -> 'forty second'."""
    words = cardinal(n).split()
    last = words[-1]
    if last in _ORDINAL_IRREGULAR:
        words[-1] = _ORDINAL_IRREGULAR[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def year(n: int) -> str:
    """1942 -> 'nineteen forty two'. 2000 -> 'two thousand'. 1905 -> 'nineteen oh five'."""
    if not (1100 <= n <= 2099):
        return cardinal(n)
    hi, lo = divmod(n, 100)
    if lo == 0:
        return f"{cardinal(hi)} hundred" if hi % 10 else cardinal(n)
    if lo < 10:
        return f"{cardinal(hi)} oh {cardinal(lo)}"
    return f"{cardinal(hi)} {cardinal(lo)}"


def _decimal(whole: str, frac: str) -> str:
    w = cardinal(int(whole)) if whole else "zero"
    return w + " point " + " ".join(_ONES[int(d)] for d in frac)


def spell_out_numerics(text: Optional[str], years: bool = True,
                       symbols: bool = True) -> Optional[str]:
    """Expand digits, currency, percentages and symbols into spoken words.

    Idempotent on already-spelled-out text, and a no-op on text containing neither digits nor
    handled symbols -- so it is safe to apply unconditionally.
    """
    if not text or not text.strip():
        return text
    out = text

    # Currency first: the symbol precedes its number, so it must be consumed before the bare
    # integer rule eats the digits and leaves an orphan '$'.
    def _cur(m):
        sym, num = m.group(1), m.group(2).replace(",", "")
        sing, plur = _CURRENCY[sym]
        if "." in num:
            whole, frac = num.split(".", 1)
            return _decimal(whole, frac) + " " + plur
        n = int(num)
        return f"{cardinal(n)} {sing if n == 1 else plur}"
    out = re.sub(r"([$£€¥])\s?(\d[\d,]*(?:\.\d+)?)", _cur, out)

    # Percent: '50%' -> 'fifty percent'.
    out = re.sub(r"(\d[\d,]*(?:\.\d+)?)\s?%",
                 lambda m: _num_words(m.group(1), years=years) + " percent", out)

    # Ordinals: '16th', '1st', '22nd'.
    out = re.sub(r"\b(\d[\d,]*)(st|nd|rd|th)\b",
                 lambda m: ordinal(int(m.group(1).replace(",", ""))), out, flags=re.I)

    # Bare numbers, decimals included.
    out = re.sub(r"\b\d[\d,]*(?:\.\d+)?\b",
                 lambda m: _num_words(m.group(0), years=years), out)

    if symbols:
        for sym, word in _SYMBOLS.items():
            if sym in out:
                out = out.replace(sym, word)

    return re.sub(r"\s+", " ", out).strip()


def _num_words(raw: str, years: bool = True) -> str:
    # Test the ORIGINAL spelling before stripping separators: a comma is the writer telling
    # you it is a quantity. Strip first and "1,942" becomes a 4-digit string and gets read as
    # a year.
    had_separator = "," in raw
    raw = raw.replace(",", "")
    if "." in raw:
        whole, frac = raw.split(".", 1)
        return _decimal(whole, frac)
    n = int(raw)
    # A 4-digit integer is read year-style ONLY when written without separators --
    # "1,942" is a quantity, "1942" is probably a year. See the module docstring caveat.
    if years and not had_separator and 1100 <= n <= 2099 and len(raw) == 4:
        return year(n)
    return cardinal(n)
