"""Caption quality gate: normalize + detect/triage malformed prompts, then
re-tokenize — writing to a DUPLICATE directory (never overwriting the source).

Two jobs:
  1. FIX  — strip leading VLM preambles ("This image shows/depicts/is/appears to
     be…") that CogVLM (ProGamerGov) adds and Qwen2-VL (jackyhate)/COCO don't.
  2. GATE — drop genuinely broken captions and report the rest:
       - DROP  empty / whitespace-only (a null CLIP target is useless), and
               degenerate repetitions ("euro banknotes, euro banknotes, …" —
               near-total information loss).
       - FLAG  (kept, informational) very-short (<=2 words) and very-long
               (>=150 words, hits CLIP/SmolLM2 truncation) captions.

Dropping reduces a shard's sample count; the source is untouched (writes go to
--output_base). --dry_run reports the full breakdown + examples with no writes.

Usage:
    # preview across all image sets, no writes
    python scripts_local/normalize_captions_retokenize.py --src DIR1 DIR2 ... --dry_run
    # real: mirror each src to output_base/<basename>, re-tokenized
    python scripts_local/normalize_captions_retokenize.py --src DIR1 DIR2 ... \
        --output_base OUT --tokenizer_name HuggingFaceTB/SmolLM2-135M --max_seq_len 256
"""

import argparse
import glob
import os
import re

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Conservative leading-preamble matcher (verbs + is/is-of + appears-to-be + "in this image,").
_PREAMBLE = re.compile(
    r'^\s*(?:'
    r'(?:the|this)\s+(?:image|picture|photo(?:graph)?|artwork|illustration|painting|drawing|'
        r'render(?:ing)?|scene|graphic|shot|figure)\s+'
        r'(?:shows?|showcases?|depicts?|features?|captures?|displays?|presents?|portrays?|'
        r'illustrates?|contains?|highlights?|reveals?|is(?:\s+of)?|'
        r'appears?\s+to\s+(?:be|show|depict|feature|contain|portray|capture|present|display|showcase)s?)'
    r'|in\s+this\s+(?:image|picture|photo(?:graph)?|scene)\s*,?'
    r'|this\s+is\s+(?:an?\s+)?(?:image|picture|photo(?:graph)?)\s+(?:of|showing|depicting|'
        r'that\s+(?:shows|depicts))'
    r'|here(?:\s+(?:is|we\s+see))\s+(?:an?\s+)?(?:image|picture|photo)'
    r')\s*[:,]?\s+',
    re.IGNORECASE,
)


def normalize_caption(c: str):
    """(normalized, changed). Never empties a non-empty caption via the strip."""
    if not isinstance(c, str):
        return c, False
    s = _PREAMBLE.sub('', c, count=1).strip()
    if not s or s == c.strip():
        return c, False
    if s[0].islower() and s[0].isalpha():
        s = s[0].upper() + s[1:]
    return s, True


def _max_consec_repeat(seq):
    """Largest number of consecutive repeats of any length-1/2/3 phrase in seq."""
    n = len(seq); best = 1
    for L in (1, 2, 3):
        i = 0
        while i + 2 * L <= n:
            reps = 1
            while i + (reps + 1) * L <= n and seq[i + reps * L:i + (reps + 1) * L] == seq[i:i + L]:
                reps += 1
            if reps > best:
                best = reps
            i += reps * L if reps > 1 else 1
    return best


def analyze_caption(c):
    """Preamble-strip + malformation triage.
    Returns (kept_text | None, flags:set). kept_text is None => DROP the sample.
    flags subset of: preamble, empty, repetition, very_short, very_long.
    """
    if not isinstance(c, str) or not c.strip():
        return None, {"empty"}
    flags = set()
    s, stripped = normalize_caption(c)
    if stripped:
        flags.add("preamble")
    s = s.strip()
    if not s:
        return None, {"empty"}                      # emptied by the strip
    words = s.split()
    nw = len(words)
    # degenerate repetition: a short phrase repeated CONSECUTIVELY >=4x (BLIP-2's
    # "euro banknotes, euro banknotes, …"). A global unique-word ratio was too
    # aggressive on long legit captions (function-word repetition), so detect the
    # actual signature — consecutive n-gram repeats — punctuation-normalized.
    norm = [w for w in (re.sub(r'[^a-z0-9]+', '', x.lower()) for x in words) if w]
    if len(norm) >= 6 and _max_consec_repeat(norm) >= 4:
        flags.add("repetition")
        return None, flags                          # DROP: genuinely broken
    if nw <= 2:
        flags.add("very_short")                     # kept, informational
    if nw >= 150:
        flags.add("very_long")                      # kept, informational
    return s, flags


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", nargs="+", required=True, help="Source shard dirs.")
    p.add_argument("--output_base", type=str, default=None,
                   help="Parent for duplicates; each src dir mirrors to output_base/<basename>. "
                        "Required unless --dry_run.")
    p.add_argument("--tokenizer_name", type=str, default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--dry_run", action="store_true", help="Metrics + examples only, no writes.")
    p.add_argument("--examples", type=int, default=4, help="Examples to print per category.")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.dry_run and not args.output_base:
        raise SystemExit("--output_base is required for a real run (or pass --dry_run)")

    tok = pad_id = None
    if not args.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer_name)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    counts = dict(total=0, preamble=0, dropped_empty=0, dropped_repetition=0,
                  very_short=0, very_long=0, kept=0)
    ex = {k: [] for k in ("preamble", "empty", "repetition", "very_short", "very_long")}

    def stash(cat, item):
        if len(ex[cat]) < args.examples:
            ex[cat].append(item)

    for src in args.src:
        shards = sorted(glob.glob(os.path.join(src, "shard_*.pt")))
        if not shards:
            print(f"WARNING: no shards in {src}"); continue
        dst_dir = None
        if not args.dry_run:
            # parent+basename so e.g. jackyhate/train_gp0 and progamergov/train_gp0
            # don't collide on a shared "train_gp0" basename.
            p = src.rstrip("/")
            uniq = f"{os.path.basename(os.path.dirname(p))}__{os.path.basename(p)}"
            dst_dir = os.path.join(args.output_base, uniq)
            os.makedirs(dst_dir, exist_ok=True)

        for f in tqdm(shards, desc=os.path.basename(src.rstrip("/")), leave=False):
            sh = torch.load(f, map_location="cpu", weights_only=False)
            texts = sh.get("text")
            if texts is None:
                continue
            kept_texts = []
            for c in texts:
                counts["total"] += 1
                kept, flags = analyze_caption(c)
                if "preamble" in flags:
                    counts["preamble"] += 1; stash("preamble", (str(c)[:80], str(kept)[:80]))
                if kept is None:
                    if "empty" in flags:
                        counts["dropped_empty"] += 1; stash("empty", str(c)[:80])
                    else:
                        counts["dropped_repetition"] += 1; stash("repetition", str(c)[:80])
                    continue
                if "very_short" in flags:
                    counts["very_short"] += 1; stash("very_short", kept[:80])
                if "very_long" in flags:
                    counts["very_long"] += 1; stash("very_long", kept[:80])
                counts["kept"] += 1
                kept_texts.append(kept)

            if args.dry_run or not kept_texts:
                continue
            enc = tok([str(t) for t in kept_texts], truncation=True, max_length=args.max_seq_len,
                      padding=False, return_attention_mask=False)["input_ids"]
            lens = [len(x) for x in enc]
            mx = max(lens) if lens else 1
            ids = [F.pad(torch.tensor(x, dtype=torch.long), (0, mx - len(x)), value=pad_id)
                   if len(x) < mx else torch.tensor(x, dtype=torch.long) for x in enc]
            sh["text"] = kept_texts
            sh["token_ids"] = torch.stack(ids, dim=0)
            sh["text_lengths"] = torch.tensor(lens, dtype=torch.long)
            sh["num_samples"] = len(kept_texts)
            dst = os.path.join(dst_dir, os.path.basename(f))
            tmp = dst + ".tmp"; torch.save(sh, tmp); os.replace(tmp, dst)

    t = counts["total"] or 1
    print(f"\n=== caption quality gate {'(DRY RUN)' if args.dry_run else ''} ===")
    print(f"captions scanned      : {counts['total']:,}")
    print(f"FIXED preamble strip  : {counts['preamble']:,}  ({100*counts['preamble']/t:.1f}%)")
    print(f"DROPPED empty         : {counts['dropped_empty']:,}  ({100*counts['dropped_empty']/t:.2f}%)")
    print(f"DROPPED repetition    : {counts['dropped_repetition']:,}  ({100*counts['dropped_repetition']/t:.2f}%)")
    print(f"flag very_short (kept): {counts['very_short']:,}  ({100*counts['very_short']/t:.2f}%)")
    print(f"flag very_long  (kept): {counts['very_long']:,}  ({100*counts['very_long']/t:.2f}%)")
    print(f"KEPT total            : {counts['kept']:,}  ({100*counts['kept']/t:.1f}%)")
    for cat, label in (("preamble", "preamble  (before -> after)"), ("empty", "DROPPED empty"),
                       ("repetition", "DROPPED repetition"), ("very_short", "very_short (kept)"),
                       ("very_long", "very_long (kept, head)")):
        if ex[cat]:
            print(f"\n  {label}:")
            for e in ex[cat]:
                print(f"    - {e[0]!r}\n      -> {e[1]!r}" if isinstance(e, tuple) else f"    - {e!r}")
    if not args.dry_run:
        print(f"\nwritten to: {args.output_base}/<group>/  (re-run stat-shards after merge)")


if __name__ == "__main__":
    main()
