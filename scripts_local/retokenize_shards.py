"""Re-tokenize preprocessed shards from their stored raw `text`, WITHOUT
reprocessing the expensive modality data (image latents / audio features).

Use when the tokenizer changes (e.g. Mistral -> SmolLM2) but the images/audio
and captions are unchanged. Each shard keeps every field except `token_ids` and
`text_lengths`, which are rebuilt from `text` with the new tokenizer, matching the
preprocess exactly (per-caption truncation to --max_seq_len, add_special_tokens,
then per-shard right-padding to the shard's max length with pad_id or 0).

In-place by default (atomic temp+rename per shard, so an interrupt can't corrupt a
shard) — the big latent/feature tensors are rewritten unchanged, so no extra disk.
Pass --output_dir to write copies elsewhere instead (preserves the old tokenizer
version, at full disk cost).

Usage:
    python scripts_local/retokenize_shards.py --shard_dir ./cached_datasets/.../train \
        --tokenizer_name HuggingFaceTB/SmolLM2-135M --max_seq_len 256
"""

import argparse
import glob
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shard_dir", type=str, required=True)
    p.add_argument("--tokenizer_name", type=str, default="HuggingFaceTB/SmolLM2-135M",
                   help="New HF tokenizer to re-tokenize `text` with.")
    p.add_argument("--max_seq_len", type=int, default=256,
                   help="Per-caption truncation length (match the original preprocess).")
    p.add_argument("--text_key", type=str, default="text")
    p.add_argument("--output_dir", type=str, default=None,
                   help="If set, write copies here instead of in-place (preserves the old version).")
    p.add_argument("--limit", type=int, default=0, help="Only process this many shards (0=all; for a dry run).")
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    print(f"tokenizer={args.tokenizer_name} vocab={len(tok)} pad_id={pad_id} max_seq_len={args.max_seq_len}")

    shards = sorted(glob.glob(os.path.join(args.shard_dir, "shard_*.pt")))
    if args.limit:
        shards = shards[:args.limit]
    if not shards:
        raise SystemExit(f"no shard_*.pt in {args.shard_dir}")
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    changed = 0
    for f in tqdm(shards, desc="re-tokenizing"):
        sh = torch.load(f, map_location="cpu", weights_only=False)
        if args.text_key not in sh:
            raise SystemExit(f"{f}: no '{args.text_key}' key (keys: {list(sh.keys())}) — nothing to re-tokenize from")
        texts = sh[args.text_key]
        if not isinstance(texts, list):
            raise SystemExit(f"{f}: '{args.text_key}' is {type(texts).__name__}, expected list of strings")

        enc = tok([str(t) for t in texts], truncation=True, max_length=args.max_seq_len,
                  padding=False, return_attention_mask=False)["input_ids"]
        lengths = [len(ids) for ids in enc]
        max_len = max(lengths) if lengths else 1
        rows = []
        for ids in enc:
            t = torch.tensor(ids, dtype=torch.long)
            if t.numel() < max_len:
                t = F.pad(t, (0, max_len - t.numel()), value=pad_id)
            rows.append(t)
        sh["token_ids"] = torch.stack(rows, dim=0) if rows else torch.zeros((0, max_len), dtype=torch.long)
        sh["text_lengths"] = torch.tensor(lengths, dtype=torch.long)

        # sanity: alignment with the modality data
        n = sh.get("num_samples", len(texts))
        assert sh["token_ids"].shape[0] == n == sh["text_lengths"].shape[0], \
            f"{f}: misalignment tokens={sh['token_ids'].shape[0]} num_samples={n} lengths={sh['text_lengths'].shape[0]}"

        dst = os.path.join(args.output_dir, os.path.basename(f)) if args.output_dir else f
        tmp = dst + ".tmp"
        torch.save(sh, tmp)
        os.replace(tmp, dst)   # atomic
        changed += 1

    print(f"done: re-tokenized {changed} shards -> {args.output_dir or args.shard_dir}")
    print("Remember to re-run stat-shards on the dir (token_ids width changed).")


if __name__ == "__main__":
    main()
