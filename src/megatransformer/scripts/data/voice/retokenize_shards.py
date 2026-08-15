"""Re-tokenize an existing preprocessed voice cache with a new tokenizer, de-corrupting
the stored transcripts on the way, and write the result to a NEW output directory.

Two jobs, both text-only (every non-text tensor -- mel_specs, unit_ids, f0, vuv, speaker
embeddings, etc. -- is copied through byte-for-byte):

  1. DE-CORRUPTION. The mimi cache was built by an older `normalize_transcript` whose
     terminal-period rule (`text[-1] not in '.!?'`) appended a spurious '.' whenever the
     transcript ended in a NON-terminal char -- a closing quote, comma, semicolon, colon,
     bracket -- producing artifacts like '"Yes.".' and 'flesh,.' (commit 62b1e09 fixed the
     rule for future runs only; ~18% of the mimi val shard carries it). We can reverse it
     deterministically: the appended '.' is exactly the trailing '.' whose preceding char is
     neither alphanumeric nor already terminal punctuation. Strip that, then re-run the
     FIXED `normalize_transcript` for full parity (idempotent; also repairs the quote-prefixed
     capitalization bug B where an all-caps utterance was lowercased).

  2. RE-TOKENIZATION. Re-encode the de-corrupted transcript with a new tokenizer (default
     SmolLM2-135M, for the pretrained-LLM text encoder) using the SAME convention the
     preprocessor used: `tokenizer(text, truncation=True, max_length=max_seq_len,
     padding=False)` with default add_special_tokens, then right-pad the shard's token_ids to
     the shard max. The 9 multimodal control tokens are NOT stored in shards -- the collator
     injects them at train time at ids >= the tokenizer's native vocab -- so nothing here
     touches them. eos is likewise appended by the collator, not stored.

The input cache is left untouched. Usage:

  uv run python -m megatransformer.scripts.data.voice.retokenize_shards \
      --input_dir  cached_datasets/Mekadrom/libritts_r_clean_mimi_wavlm \
      --output_dir cached_datasets/Mekadrom/libritts_r_clean_mimi_wavlm_smollm2
"""

import argparse
import json
import os
import shutil

import torch
import torch.nn.functional as F

from megatransformer.scripts.data.voice.preprocess import normalize_transcript

# Non-shard files copied verbatim into each split dir (besides config.json, which we rewrite).
_PASSTHROUGH_FILES = ("shard_index.json", "mimi_semantic_codebook.pt")


def decorrupt(text: str) -> str:
    """Reverse the old normalize_transcript's spurious terminal period, then re-normalize.

    The old bug appended '.' iff the final char was not in '.!?'. So a trailing '.' is
    spurious exactly when the char before it is neither alphanumeric (a legit sentence end
    that the FIXED normalizer also produces) nor already terminal punctuation. Everything
    else -- comma, semicolon, colon, closing quote/bracket before the '.' -- is the artifact.
    """
    if not isinstance(text, str) or not text:
        return text
    t = text.strip()
    if len(t) >= 2 and t[-1] == "." and not (t[-2].isalnum() or t[-2] in ".!?"):
        t = t[:-1]
    return normalize_transcript(t)


def _list_shard_files(split_dir: str) -> list:
    return sorted(f for f in os.listdir(split_dir)
                  if f.startswith("shard_") and f.endswith(".pt"))


def _rewrite_config(src_cfg_path: str, dst_cfg_path: str, tokenizer_name: str,
                    vocab_size: int, special_token_base: int) -> None:
    with open(src_cfg_path) as f:
        cfg = json.load(f)
    cfg["tokenizer_name"] = tokenizer_name
    cfg["vocab_size"] = vocab_size
    # Record the control-token base so downstream tooling can cross-check the model config.
    cfg["special_token_base"] = special_token_base
    cfg["retokenized_from"] = os.path.basename(os.path.dirname(src_cfg_path.rstrip("/")))
    cfg["text_decorrupted"] = True
    with open(dst_cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)


def process_split(split_dir: str, out_dir: str, tokenizer, max_seq_len: int,
                  vocab_size: int, special_token_base: int) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Rewrite config.json; copy the other non-shard files verbatim.
    _rewrite_config(os.path.join(split_dir, "config.json"),
                    os.path.join(out_dir, "config.json"),
                    tokenizer.name_or_path, vocab_size, special_token_base)
    for fname in _PASSTHROUGH_FILES:
        src = os.path.join(split_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, fname))

    pad_id = tokenizer.pad_token_id or 0
    shard_files = _list_shard_files(split_dir)
    n_fixed_total = 0
    n_rows_total = 0
    printed_samples = False

    for si, fname in enumerate(shard_files):
        shard = torch.load(os.path.join(split_dir, fname),
                           map_location="cpu", weights_only=False)
        texts = shard.get("text")
        if texts is None:
            # No text in this shard -- copy through untouched.
            torch.save(shard, os.path.join(out_dir, fname))
            print(f"  [{si + 1}/{len(shard_files)}] {fname}: no text, copied")
            continue

        clean_texts = []
        n_fixed = 0
        for t in texts:
            ct = decorrupt(t)
            if ct != t:
                n_fixed += 1
            clean_texts.append(ct)

        # Show a few real before/after fixes from the first shard as a sanity check.
        if not printed_samples:
            shown = 0
            for orig, ct in zip(texts, clean_texts):
                if orig != ct:
                    print(f"      fix: {orig!r} -> {ct!r}")
                    shown += 1
                    if shown >= 6:
                        break
            printed_samples = True

        encoded = tokenizer(
            clean_texts,
            truncation=True,
            max_length=max_seq_len,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]

        rows = [torch.tensor(ids, dtype=torch.long) for ids in encoded]
        lengths = torch.tensor([r.shape[-1] for r in rows], dtype=torch.long)
        max_len = int(lengths.max()) if len(rows) else 0
        padded = torch.stack([
            F.pad(r, (0, max_len - r.shape[-1]), value=pad_id) if r.shape[-1] < max_len else r
            for r in rows
        ], dim=0) if rows else torch.zeros((0, 0), dtype=torch.long)

        shard["text"] = clean_texts
        shard["token_ids"] = padded
        shard["text_lengths"] = lengths

        torch.save(shard, os.path.join(out_dir, fname))
        n_fixed_total += n_fixed
        n_rows_total += len(texts)
        print(f"  [{si + 1}/{len(shard_files)}] {fname}: {len(texts)} rows, "
              f"{n_fixed} decorrupted, max_tok_len={max_len}")

    pct = (100.0 * n_fixed_total / n_rows_total) if n_rows_total else 0.0
    print(f"  split done: {n_rows_total} rows, {n_fixed_total} decorrupted ({pct:.1f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True,
                    help="Existing cache root (contains split subdirs, e.g. train/ val/).")
    ap.add_argument("--output_dir", required=True,
                    help="New cache root to write (must differ from --input_dir).")
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M",
                    help="HF tokenizer id for re-tokenization.")
    ap.add_argument("--splits", default="train,val",
                    help="Comma-separated split subdir names to process.")
    ap.add_argument("--max_seq_len", type=int, default=None,
                    help="Truncation length; default = the split config.json's max_seq_len.")
    args = ap.parse_args()

    in_root = os.path.abspath(args.input_dir)
    out_root = os.path.abspath(args.output_dir)
    if in_root == out_root:
        raise SystemExit("--output_dir must differ from --input_dir (in-place is unsafe)")

    from transformers import AutoTokenizer, AutoConfig
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = int(AutoConfig.from_pretrained(args.tokenizer).vocab_size)
    special_token_base = vocab_size  # control tokens live at >= native vocab
    print(f"  vocab_size={vocab_size}  pad_id={tokenizer.pad_token_id}  "
          f"eos_id={tokenizer.eos_token_id}")

    for split in args.splits.split(","):
        split = split.strip()
        split_dir = os.path.join(in_root, split)
        if not os.path.isdir(split_dir):
            print(f"[skip] no such split dir: {split_dir}")
            continue
        cfg_path = os.path.join(split_dir, "config.json")
        max_seq_len = args.max_seq_len
        if max_seq_len is None:
            with open(cfg_path) as f:
                max_seq_len = int(json.load(f).get("max_seq_len", 256))
        out_dir = os.path.join(out_root, split)
        print(f"\n=== split '{split}'  (max_seq_len={max_seq_len}) -> {out_dir} ===")
        process_split(split_dir, out_dir, tokenizer, max_seq_len,
                      vocab_size, special_token_base)

    print(f"\nDone. New cache at: {out_root}")


if __name__ == "__main__":
    main()
