import os
import traceback

import torch
import torch.nn.functional as F

from megatransformer.scripts.data.preprocessor import Preprocessor, validate_shard_alignment


class TextDatasetPreprocessor(Preprocessor):
    """Preprocess a text dataset into tokenized shards.

    Two packing modes:
      - "pack" (default, industry-standard): tokenize each document without
        truncation, append the tokenizer EOS as a document separator, extend a
        rolling token buffer across examples, and emit fixed max_seq_len-sized
        blocks from the buffer. No wasted tokens, no padding, no artificial
        per-document truncation. A single training sample can span multiple
        documents (separated by EOS) or contain only part of one long document.
      - "truncate" (legacy): one sample per document, hard-truncated at
        max_seq_len. Keeps documents 1:1 with samples but discards the tail
        of any document longer than max_seq_len.
    """

    def __init__(self, args, dataset, output_dir, shard_fields, batch_accumulators, stats_accumulator, device):
        self.args = args
        self.dataset = dataset
        self.output_dir = output_dir
        self.shard_fields = shard_fields
        self.batch_accumulators = batch_accumulators
        self.stats_accumulator = stats_accumulator
        self.device = device

        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {args.tokenizer_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        self.vocab_size = len(self.tokenizer)
        print(f"  Vocab size: {self.vocab_size}")

        try:
            print(f"  Total samples in dataset: {len(self.dataset):,}")
        except TypeError:
            print(f"  Dataset is streaming (size unknown)")
        print(f"  Max sequence length: {args.max_seq_len}")
        print(f"  Packing mode: {args.packing}")
        if getattr(args, "max_tokens", None):
            print(f"  Max tokens budget: {args.max_tokens:,}")

        # Ensure the token counter exists so the main loop can check against
        # --max_tokens regardless of which mode is active.
        stats_accumulator.setdefault("tokens_saved", 0)

        # Persistent token buffer used only in pack mode. Bridges tokens
        # across examples so long docs span blocks and short docs share.
        self._token_buffer: list[int] = []

        shard_fields.update({
            "shard_token_ids": [],
            "shard_text_lengths": [],
            "shard_text": [],
        })

    @classmethod
    def add_cli_args(cls, subparsers):
        sub_parser = subparsers.add_parser("text", help="Preprocess text dataset into tokenized shards")

        sub_parser.add_argument("--tokenizer_name", type=str, default="mistralai/Mistral-7B-v0.1",
                                help="HuggingFace tokenizer name or path")
        sub_parser.add_argument("--text_column", type=str, default="text",
                                help="Name of the text column in the dataset")
        sub_parser.add_argument("--max_seq_len", type=int, default=2048,
                                help="Per-sample sequence length. In 'pack' mode: the size of each "
                                     "emitted fixed-length block. In 'truncate' mode: hard cap above "
                                     "which documents are truncated.")
        sub_parser.add_argument("--min_text_len", type=int, default=1,
                                help="Minimum text length in characters (skip shorter texts)")
        sub_parser.add_argument("--packing", type=str, default="pack", choices=["pack", "truncate"],
                                help="'pack' (default): concatenate documents with EOS separators "
                                     "into a continuous token stream, emit fixed max_seq_len blocks "
                                     "— no wasted tokens. 'truncate': legacy one-doc-per-sample with "
                                     "hard cap at max_seq_len.")
        sub_parser.add_argument("--max_tokens", type=int, default=None,
                                help="Cumulative token budget across all emitted samples (DIFFERENT "
                                     "from --max_seq_len which is per-sample). Use for pretraining "
                                     "corpus sizing (e.g. --max_tokens 1_000_000_000 = 1B-token cap). "
                                     "Preprocessing stops when the cumulative count reaches this limit.")

        return sub_parser

    def flush_shard(self):
        if not self.shard_fields["shard_token_ids"]:
            return

        # In pack mode every block is exactly max_seq_len; padding is a no-op.
        # In truncate mode shorter docs still exist and need intra-shard padding.
        max_len = max(t.shape[-1] for t in self.shard_fields["shard_token_ids"])
        padded_token_ids = []
        for tokens in self.shard_fields["shard_token_ids"]:
            if tokens.shape[-1] < max_len:
                tokens = F.pad(tokens, (0, max_len - tokens.shape[-1]), value=self.tokenizer.pad_token_id or 0)
            padded_token_ids.append(tokens)

        num_samples = len(padded_token_ids)

        shard_data = {
            "token_ids": torch.stack(padded_token_ids, dim=0),
            "text_lengths": torch.stack(self.shard_fields["shard_text_lengths"], dim=0),
            "num_samples": num_samples,
        }
        # Raw text is ambiguous in pack mode (blocks can span multiple docs),
        # so it's skipped there. Truncate mode keeps 1:1 doc->sample so we
        # preserve the original strings for debugging and downstream uses.
        if self.args.packing == "truncate":
            shard_data["text"] = self.shard_fields["shard_text"]

        # Catch accumulator-lifecycle bugs before they go to disk.
        validate_shard_alignment(shard_data, num_samples)

        shard_path = os.path.join(self.output_dir, f"shard_{self.shard_fields['shard_idx']:06d}.pt")
        torch.save(shard_data, shard_path)

        print(f"  Saved shard {self.shard_fields['shard_idx']} ({num_samples} samples)")

        self.shard_fields["shard_token_ids"] = []
        self.shard_fields["shard_text_lengths"] = []
        self.shard_fields["shard_text"] = []
        self.shard_fields["shard_idx"] += self.args.total_gpus

    def _emit_block(self, block: list[int]) -> None:
        """Commit one fixed-length block to the current shard (pack mode only)."""
        self.shard_fields["shard_token_ids"].append(torch.tensor(block, dtype=torch.long))
        self.shard_fields["shard_text_lengths"].append(torch.tensor(len(block), dtype=torch.long))
        self.stats_accumulator["saved"] += 1
        self.stats_accumulator["tokens_saved"] += len(block)

    def process_and_accumulate(self):
        if not self.batch_accumulators.get("batch_texts"):
            return

        try:
            texts = self.batch_accumulators["batch_texts"]

            if self.args.packing == "pack":
                encoded = self.tokenizer(
                    texts,
                    truncation=False,
                    padding=False,
                    return_attention_mask=False,
                )
                eos_id = self.tokenizer.eos_token_id
                for input_ids in encoded["input_ids"]:
                    self._token_buffer.extend(input_ids)
                    if eos_id is not None:
                        self._token_buffer.append(eos_id)

                seq_len = self.args.max_seq_len
                while len(self._token_buffer) >= seq_len:
                    block = self._token_buffer[:seq_len]
                    del self._token_buffer[:seq_len]
                    self._emit_block(block)
                    if len(self.shard_fields["shard_token_ids"]) >= self.args.shard_size:
                        self.flush_shard()
            else:
                encoded = self.tokenizer(
                    texts,
                    truncation=True,
                    max_length=self.args.max_seq_len,
                    padding=False,
                    return_attention_mask=False,
                )
                for i, input_ids in enumerate(encoded["input_ids"]):
                    token_ids = torch.tensor(input_ids, dtype=torch.long)
                    text_length = torch.tensor(len(input_ids), dtype=torch.long)

                    self.shard_fields["shard_token_ids"].append(token_ids)
                    self.shard_fields["shard_text_lengths"].append(text_length)
                    self.shard_fields["shard_text"].append(texts[i])
                    self.stats_accumulator["saved"] += 1
                    self.stats_accumulator["tokens_saved"] += len(input_ids)

                if len(self.shard_fields["shard_token_ids"]) >= self.args.shard_size:
                    self.flush_shard()

        except Exception as e:
            print(f"Batch processing error: {e}")
            traceback.print_exc()
            self.stats_accumulator["skipped"]["error"] += len(self.batch_accumulators["batch_texts"])

        self.batch_accumulators["batch_texts"] = []

    def preprocess_example(self, example) -> bool:
        text = example.get(self.args.text_column, None)
        if text is None or not isinstance(text, str):
            self.stats_accumulator["skipped"]["missing_text"] += 1
            return False

        if len(text.strip()) < self.args.min_text_len:
            self.stats_accumulator["skipped"]["too_short"] += 1
            return False

        if "batch_texts" not in self.batch_accumulators:
            self.batch_accumulators["batch_texts"] = []

        self.batch_accumulators["batch_texts"].append(text)

        if len(self.batch_accumulators["batch_texts"]) >= self.args.gpu_batch_size:
            self.process_and_accumulate()

        return True

    def parse_config(self) -> dict:
        return {
            "tokenizer_name": self.args.tokenizer_name,
            "vocab_size": self.vocab_size,
            "text_column": self.args.text_column,
            "max_seq_len": self.args.max_seq_len,
            "min_text_len": self.args.min_text_len,
            "packing": self.args.packing,
            "max_tokens": getattr(self.args, "max_tokens", None),
            "dataset_name": self.args.dataset_name,
            "dataset_config": self.args.dataset_config,
            "split": self.args.split,
            "shard_size": self.args.shard_size,
            "stats": self.stats_accumulator,
        }
