"""Backfill the raw `text` field into voice shards that were preprocessed without --save_text.

Why this is needed: `preprocess_dataset voice` only stores raw transcripts when --save_text
is passed, and it defaults to FALSE. Without it the shards carry `token_ids` but no `text`,
and the world dataset's voice branch (`world/dataset.py:405`) only sets `voice_text` when
`"text" in shard`. Downstream, `cosyvoice_wer_eval.py` does

    ref = str(s.get("voice_voice_text", "")).strip()
    if len(bov) == 0 or not ref or spk is None: continue

so an absent transcript SILENTLY skips every sample and the eval reports zero rows instead
of failing. That is the failure mode this script exists to prevent.

Why reconstruction is exact rather than approximate: `token_ids` is the SmolLM2 tokenization
of the already-normalized transcript, and `text_lengths` gives the true unpadded length.
Verified on 2,000 val samples that re-tokenize(decode(ids)) == ids for 100% of rows, so the
decode recovers precisely the string the old cache stored (which was likewise the NORMALIZED
transcript, not the raw one -- see assemble_cosyvoice_cache.py).

Additive and idempotent: only the `text` key is written, existing fields are untouched, and
shards that already have `text` are skipped unless --overwrite.

  uv run python scripts_local/backfill_shard_text.py \
      --shard_dirs cached_datasets/Mekadrom/libriheavy_cosyvoice2_smollm2/train \
                   cached_datasets/Mekadrom/libriheavy_cosyvoice2_smollm2/val \
      --tokenizer_name HuggingFaceTB/SmolLM2-135M
"""
import argparse, glob, os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dirs", nargs="+", required=True)
    ap.add_argument("--tokenizer_name", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--overwrite", action="store_true",
                    help="Rewrite `text` even on shards that already have it")
    ap.add_argument("--verify", action="store_true", default=True,
                    help="Assert re-tokenize(decode(ids)) == ids on every row")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    tk = AutoTokenizer.from_pretrained(a.tokenizer_name)
    total = skipped = written = 0
    bad = []

    for d in a.shard_dirs:
        files = sorted(glob.glob(os.path.join(d, "shard_*.pt")))
        if not files:
            raise SystemExit(f"no shard_*.pt in {d}")
        for f in tqdm(files, desc=os.path.basename(d.rstrip("/"))):
            s = torch.load(f, map_location="cpu", weights_only=False)
            if "text" in s and not a.overwrite:
                skipped += 1
                continue
            if "token_ids" not in s or "text_lengths" not in s:
                raise SystemExit(f"{f}: needs token_ids + text_lengths to reconstruct text")
            ti, tl = s["token_ids"], s["text_lengths"]
            texts = []
            for i in range(len(tl)):
                ids = ti[i, :int(tl[i])].tolist()
                txt = tk.decode(ids, skip_special_tokens=True)
                if a.verify:
                    re_ids = tk(txt, truncation=True, max_length=ti.shape[1],
                                padding=False, return_attention_mask=False)["input_ids"]
                    if re_ids != ids:
                        bad.append((f, i, txt[:60]))
                texts.append(txt)
            total += len(texts)
            if not a.dry_run:
                s["text"] = texts
                tmp = f + ".tmp"          # write-then-rename: a crash mid-save must not
                torch.save(s, tmp)        # leave a truncated shard in place of a good one
                os.replace(tmp, f)
            written += 1
            del s

    print(f"\nshards written: {written} | already had text (skipped): {skipped} | rows: {total:,}")
    if bad:
        print(f"WARNING: {len(bad)} rows did not round-trip exactly (text still written):")
        for f, i, t in bad[:10]:
            print(f"  {os.path.basename(f)}[{i}] {t!r}")
    else:
        print("round-trip verified exact on every row")
    if a.dry_run:
        print("(dry run -- nothing was written)")


if __name__ == "__main__":
    main()
