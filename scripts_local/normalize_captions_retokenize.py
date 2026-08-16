"""Strip VLM caption preambles ("This image shows…", "The image depicts…") and
re-tokenize, writing to a DUPLICATE directory (never overwriting the source).

CogVLM (ProGamerGov) captions often open with a meta-descriptive preamble that
Qwen2-VL (jackyhate) captions don't. Stripping it removes filler from the CLIP
target, harmonizes the two caption styles, and closes the train/inference gap
(inference prompts have no such preamble). Stripping changes the text, so this
also re-tokenizes so token_ids match; the CLIP target (computed from `text` at
train time) also benefits.

Non-preamble captions pass through unchanged, so it's safe to run on jackyhate too
(it just won't match). Prints match metrics + before/after examples.

Usage (preview only, no writes):
    python scripts_local/normalize_captions_retokenize.py \
        --src ./cached_datasets/.../progamergov_sdxl_clip/train_gp{0,1,2,3} --dry_run

Usage (real, mirrors each src dir under --output_base):
    python scripts_local/normalize_captions_retokenize.py \
        --src ./cached_datasets/.../progamergov_sdxl_clip/train_gp{0,1,2,3} \
        --output_base ./cached_datasets/.../progamergov_sdxl_clip_norm \
        --tokenizer_name HuggingFaceTB/SmolLM2-135M --max_seq_len 256
"""

import argparse
import glob
import os
import re

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Conservative leading-preamble matcher. Matches a clause + trailing separator/space;
# deliberately omits the bare "The image is X" (too easy to over-strip content).
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
    """Returns (normalized, changed). Never returns empty (keeps original if the
    strip would empty the caption)."""
    if not isinstance(c, str):
        return c, False
    s = _PREAMBLE.sub('', c, count=1).strip()
    if not s or s == c.strip():
        return c, False
    if s[0].islower() and s[0].isalpha():
        s = s[0].upper() + s[1:]
    return s, True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", nargs="+", required=True, help="Source shard dirs.")
    p.add_argument("--output_base", type=str, default=None,
                   help="Parent for duplicates; each src dir mirrors to output_base/<basename>. "
                        "Required unless --dry_run.")
    p.add_argument("--tokenizer_name", type=str, default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--dry_run", action="store_true", help="Metrics + examples only, no writes.")
    p.add_argument("--examples", type=int, default=8, help="Before/after examples to print.")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.dry_run and not args.output_base:
        raise SystemExit("--output_base is required for a real run (or pass --dry_run)")

    tok = None
    if not args.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer_name)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    total = changed = 0
    examples = []
    for src in args.src:
        shards = sorted(glob.glob(os.path.join(src, "shard_*.pt")))
        if not shards:
            print(f"WARNING: no shards in {src}")
            continue
        dst_dir = None
        if not args.dry_run:
            dst_dir = os.path.join(args.output_base, os.path.basename(src.rstrip("/")))
            os.makedirs(dst_dir, exist_ok=True)

        for f in tqdm(shards, desc=os.path.basename(src.rstrip("/")), leave=False):
            sh = torch.load(f, map_location="cpu", weights_only=False)
            texts = sh.get("text")
            if texts is None:
                continue
            new_texts, new_ids, new_lens, shard_changed = [], [], [], 0
            for c in texts:
                s, ch = normalize_caption(c)
                new_texts.append(s)
                total += 1
                if ch:
                    changed += 1; shard_changed += 1
                    if len(examples) < args.examples:
                        examples.append((c[:90], s[:90]))
            if args.dry_run:
                continue
            # re-tokenize the (possibly stripped) captions, per-shard pad
            enc = tok([str(t) for t in new_texts], truncation=True, max_length=args.max_seq_len,
                      padding=False, return_attention_mask=False)["input_ids"]
            lens = [len(x) for x in enc]
            mx = max(lens) if lens else 1
            for ids in enc:
                t = torch.tensor(ids, dtype=torch.long)
                if t.numel() < mx:
                    t = F.pad(t, (0, mx - t.numel()), value=pad_id)
                new_ids.append(t)
            sh["text"] = new_texts
            sh["token_ids"] = torch.stack(new_ids, dim=0) if new_ids else torch.zeros((0, mx), dtype=torch.long)
            sh["text_lengths"] = torch.tensor(lens, dtype=torch.long)
            dst = os.path.join(dst_dir, os.path.basename(f))
            tmp = dst + ".tmp"
            torch.save(sh, tmp); os.replace(tmp, dst)

    pct = (100.0 * changed / total) if total else 0.0
    print(f"\n=== caption preamble normalization {'(DRY RUN)' if args.dry_run else ''} ===")
    print(f"captions scanned : {total:,}")
    print(f"preamble stripped: {changed:,}  ({pct:.1f}%)")
    print(f"unchanged        : {total - changed:,}")
    if examples:
        print("\nbefore -> after:")
        for a, b in examples:
            print(f"  - {a!r}\n    -> {b!r}")
    if not args.dry_run:
        print(f"\nwritten to: {args.output_base}/<group>/  (re-run stat-shards on the merged output)")


if __name__ == "__main__":
    main()
