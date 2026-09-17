"""Re-tokenize a preprocessed TEXT cache with a different tokenizer, writing a NEW shard dir.

Why this exists: a text cache is expensive to rebuild. `cached_datasets/text_train_merged` is
5.25B tokens assembled from the ~30-source mixture in `logs/huginn_dataset_mixture.md`, and
re-running that means re-streaming terabytes from HF. But the tokenizer is baked into the
stored ids, and a distillation teacher fixes the tokenizer (logit-KL needs a shared vocabulary),
so changing teachers otherwise means re-downloading everything.

HOW THIS DIFFERS FROM THE VOICE VERSION (`scripts/data/voice/retokenize_shards.py`). That one
reads the transcript STRING stored alongside each utterance and re-encodes it. Text shards in
pack mode store no text at all -- just `token_ids`, `text_lengths`, `num_samples` -- because a
packed block can span several documents and "the text of this block" is ill-defined. So the
source text has to be recovered by DECODING, which makes the round-trip the thing to be careful
about.

Measured on the real cache (Mistral -> Qwen3, 194 documents):
  - decoded text round-trips EXACTLY, 194/194, through both tokenizers.
  - source ids do NOT round-trip exactly, 0/194 -- but the only difference is the FIRST token
    of a document ('hip' vs '▁hip'). SentencePiece writes a word-boundary marker when encoding
    fresh text, so a document that was cut mid-word loses the fact that it was mid-word.
    Everything from index 1 on is identical and the decoded text is unchanged. This is inherent
    to going through text and it costs at most one token per document.

ALGORITHM. Pack mode concatenated documents with the tokenizer's EOS as a separator and sliced
the result into fixed blocks, so block boundaries are meaningless and document boundaries are
the EOS ids. Therefore:

  1. Stream shards in order and flatten -- the blocks are one continuous token stream.
  2. Split on the SOURCE eos id to recover documents. A document straddling a shard boundary is
     carried in `pending` and completed by the next shard; NEVER split at a shard boundary, or
     a document gets cut mid-word for no reason.
  3. Decode each document with the source tokenizer, re-encode with the target, append the
     TARGET eos.
  4. Re-pack into fixed `max_seq_len` blocks and flush shards of `shard_size` samples.

Nothing is read from the input dir except `shard_*.pt`, and nothing is ever written to it.

Usage:

  uv run python -m megatransformer.scripts.data.text.retokenize_shards --input_dir cached_datasets/text_train_merged --output_dir cached_datasets/text_train_merged_qwen3 --target_tokenizer Qwen/Qwen3-4B
"""

import argparse
import glob
import json
import os
import time

import torch


def _iter_shard_paths(input_dir: str):
    paths = sorted(glob.glob(os.path.join(input_dir, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"no shard_*.pt in {input_dir}")
    return paths


class _ShardWriter:
    """Accumulates blocks and flushes fixed-size shards, matching the preprocessor's layout."""

    def __init__(self, output_dir: str, shard_size: int, max_seq_len: int):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.max_seq_len = max_seq_len
        self.blocks: list[list[int]] = []
        self.shard_idx = 0
        self.total_samples = 0
        self.shard_files: list[str] = []
        self.shard_offsets: list[int] = []
        os.makedirs(output_dir, exist_ok=True)

    def add_block(self, block: list[int]):
        self.blocks.append(block)
        if len(self.blocks) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.blocks:
            return
        ids = torch.tensor(self.blocks, dtype=torch.int64)
        n = ids.shape[0]
        data = {
            "token_ids": ids,
            # Pack mode emits full blocks, so every length is max_seq_len. The final block of
            # the whole corpus is the one exception and is padded to width by the caller.
            "text_lengths": torch.full((n,), ids.shape[1], dtype=torch.int64),
            "num_samples": n,
        }
        name = f"shard_{self.shard_idx:06d}.pt"
        torch.save(data, os.path.join(self.output_dir, name))
        self.shard_files.append(name)
        self.shard_offsets.append(self.total_samples)
        self.total_samples += n
        self.shard_idx += 1
        self.blocks = []

    def write_index(self, extra_config: dict):
        self.flush()
        # Written here rather than left to first use: without it the dataset scans every shard
        # to build one, which on a 40GB corpus looks exactly like a hang before step 1.
        with open(os.path.join(self.output_dir, "shard_index.json"), "w") as f:
            json.dump({
                "shard_files": self.shard_files,
                "shard_offsets": self.shard_offsets,
                "total_samples": self.total_samples,
                "dataset_type": "stat-shards",
            }, f)
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(extra_config, f, indent=2)


def _split_docs(stream, eos):
    out, cur = [], []
    for t in stream:
        if t == eos:
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        out.append(cur)
    return out


def _verify(in_dir, out_dir, src, tgt, src_eos, tgt_eos, n_docs):
    """Compare decoded text of the first n_docs documents, source vs output."""
    def _head(paths, eos, want):
        docs, stream = [], []
        for p in paths:
            stream += torch.load(p, map_location="cpu", weights_only=False)[
                "token_ids"].reshape(-1).tolist()
            docs = _split_docs(stream, eos)
            if len(docs) > want:
                break
        return docs[:want]

    s_docs = _head(_iter_shard_paths(in_dir), src_eos, n_docs)
    t_docs = _head(_iter_shard_paths(out_dir), tgt_eos, n_docs)
    n = min(len(s_docs), len(t_docs))
    same = sum(
        src.decode(s_docs[i], skip_special_tokens=True)
        == tgt.decode(t_docs[i], skip_special_tokens=True)
        for i in range(n)
    )
    return same, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True, help="Source shard dir (read-only).")
    ap.add_argument("--output_dir", required=True, help="Destination dir. Must not be the input.")
    ap.add_argument("--source_tokenizer", default="mistralai/Mistral-7B-v0.1",
                    help="Tokenizer the input shards were built with.")
    ap.add_argument("--target_tokenizer", required=True,
                    help="Tokenizer to re-encode with, e.g. Qwen/Qwen3-4B. All Qwen3 sizes "
                         "share vocab 151936, so any of them yields the same corpus.")
    ap.add_argument("--max_seq_len", type=int, default=1024, help="Block width to re-pack to.")
    ap.add_argument("--shard_size", type=int, default=100_000, help="Blocks per output shard.")
    ap.add_argument("--batch_docs", type=int, default=1000,
                    help="Documents per tokenizer call. Batching is where the speed is.")
    ap.add_argument("--limit_shards", type=int, default=None,
                    help="Process only the first N input shards (for a smoke test).")
    ap.add_argument("--verify", type=int, default=300,
                    help="After writing, compare this many documents' decoded text against "
                         "the source. 0 disables. Cheap, and the failure it catches is silent.")
    args = ap.parse_args()

    in_dir = os.path.abspath(args.input_dir)
    out_dir = os.path.abspath(args.output_dir)
    if in_dir == out_dir:
        raise SystemExit("refusing to write into the input dir; pass a different --output_dir")
    if os.path.isdir(out_dir) and glob.glob(os.path.join(out_dir, "shard_*.pt")):
        raise SystemExit(f"{out_dir} already contains shards; refusing to overwrite")

    from transformers import AutoTokenizer
    src = AutoTokenizer.from_pretrained(args.source_tokenizer)
    tgt = AutoTokenizer.from_pretrained(args.target_tokenizer)
    src_eos, tgt_eos = src.eos_token_id, tgt.eos_token_id
    if src_eos is None or tgt_eos is None:
        raise SystemExit("both tokenizers need an eos_token_id (it is the document separator)")
    print(f"source {args.source_tokenizer} (vocab {src.vocab_size}, eos {src_eos})")
    print(f"target {args.target_tokenizer} (vocab {tgt.vocab_size}, eos {tgt_eos})")

    paths = _iter_shard_paths(in_dir)
    if args.limit_shards:
        paths = paths[:args.limit_shards]
    writer = _ShardWriter(out_dir, args.shard_size, args.max_seq_len)

    pending: list[int] = []      # tail of a document straddling a shard boundary
    out_buf: list[int] = []      # target-token stream awaiting packing
    doc_batch: list[list[int]] = []
    n_src_tokens = n_tgt_tokens = n_docs = 0
    t0 = time.time()

    def drain_docs():
        """Decode + re-encode a batch of documents and append to the output stream."""
        nonlocal n_tgt_tokens, n_docs
        if not doc_batch:
            return
        texts = src.batch_decode(doc_batch, skip_special_tokens=True)
        encoded = tgt(texts, add_special_tokens=False)["input_ids"]
        for ids in encoded:
            out_buf.extend(ids)
            out_buf.append(tgt_eos)       # re-establish the document separator
            n_tgt_tokens += len(ids) + 1
        n_docs += len(doc_batch)
        doc_batch.clear()

    def pack():
        """Emit as many full blocks as the buffer allows."""
        L = args.max_seq_len
        i = 0
        while len(out_buf) - i >= L:
            writer.add_block(out_buf[i:i + L])
            i += L
        if i:
            del out_buf[:i]

    for si, p in enumerate(paths):
        shard = torch.load(p, map_location="cpu", weights_only=False)
        ids = shard["token_ids"]
        del shard

        # Row-at-a-time rather than one .tolist() over the whole shard: a 100k x 1024 shard
        # is 102M Python ints, ~3.7GB of boxed objects, and paying that an hour into the run
        # is a bad way to discover it. Rows are 1024 ints and the stream is unaffected --
        # pack mode made block boundaries meaningless, only the EOS ids matter.
        cur = pending
        for row in ids:
            chunk = row.tolist()
            n_src_tokens += len(chunk)
            for t in chunk:
                if t == src_eos:
                    if cur:
                        doc_batch.append(cur)
                        if len(doc_batch) >= args.batch_docs:
                            drain_docs()
                            pack()
                    cur = []
                else:
                    cur.append(t)
        pending = cur                      # incomplete: carried to the next shard
        del ids

        el = time.time() - t0
        rate = n_src_tokens / max(el, 1e-6)
        print(f"  [{si + 1}/{len(paths)}] {os.path.basename(p)} | "
              f"src {n_src_tokens / 1e9:.3f}B -> tgt {n_tgt_tokens / 1e9:.3f}B | "
              f"{n_docs:,} docs | {rate / 1e3:.0f}k tok/s | "
              f"eta {(len(paths) - si - 1) * el / (si + 1) / 60:.1f} min", flush=True)

    # Final document (the stream may not end on an EOS) and whatever is left in the buffers.
    if pending:
        doc_batch.append(pending)
    drain_docs()
    pack()
    if out_buf:
        # Pad the one trailing partial block rather than discarding real tokens.
        block = out_buf + [tgt_eos] * (args.max_seq_len - len(out_buf))
        writer.add_block(block)
        out_buf.clear()

    # vocab_size must be the MODEL's, not the tokenizer's. They differ: Qwen3's tokenizer
    # reports 151643 while its config reports 151936, and eos (151645) sits ABOVE the
    # tokenizer count -- so recording tokenizer.vocab_size would describe a corpus whose own
    # separator is out of range, and anything sizing an embedding from it would be too small.
    try:
        from transformers import AutoConfig
        model_vocab = int(AutoConfig.from_pretrained(args.target_tokenizer).vocab_size)
    except Exception:
        model_vocab = max(int(tgt.vocab_size), int(tgt_eos) + 1)
    writer.write_index({
        "source_tokenizer": args.source_tokenizer,
        "tokenizer_name": args.target_tokenizer,
        "vocab_size": model_vocab,
        "tokenizer_vocab_size": int(tgt.vocab_size),
        "eos_token_id": int(tgt_eos),
        "max_seq_len": args.max_seq_len,
        "packing": "pack",
        "retokenized_from": in_dir,
        "source_tokens": n_src_tokens,
        "target_tokens": n_tgt_tokens,
        "documents": n_docs,
    })
    # Self-check: re-read the written shards and confirm the text survived. A corpus that is
    # silently wrong here trains without complaint (ids stay numerically in range), so the
    # verification belongs in the converter rather than in a script nobody remembers to run.
    if args.verify:
        print(f"\nverifying {args.verify} documents against the source...")
        ok = _verify(in_dir, out_dir, src, tgt, src_eos, tgt_eos, args.verify)
        print(f"  text identical: {ok[0]}/{ok[1]}" + ("" if ok[0] == ok[1] else "   <-- MISMATCH"))
        if ok[0] != ok[1]:
            raise SystemExit("verification FAILED; output is not faithful to the source")

    el = time.time() - t0
    print(f"\ndone in {el / 60:.1f} min")
    print(f"  {n_src_tokens:,} src tokens -> {n_tgt_tokens:,} tgt tokens "
          f"({n_tgt_tokens / max(n_src_tokens, 1):.3f}x) across {n_docs:,} documents")
    print(f"  {writer.total_samples:,} blocks in {len(writer.shard_files)} shards -> {out_dir}")


if __name__ == "__main__":
    main()
