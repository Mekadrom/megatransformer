import os
import traceback

import torch
import torch.nn.functional as F

from scripts.data.preprocessor import Preprocessor


class TextDatasetPreprocessor(Preprocessor):
    """Preprocess a text dataset into tokenized shards."""

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
                                help="Maximum token sequence length (longer texts are truncated)")
        sub_parser.add_argument("--min_text_len", type=int, default=1,
                                help="Minimum text length in characters (skip shorter texts)")

        return sub_parser

    def flush_shard(self):
        if not self.shard_fields["shard_token_ids"]:
            return

        # Pad token_ids to same length within shard
        max_len = max(t.shape[-1] for t in self.shard_fields["shard_token_ids"])
        padded_token_ids = []
        for tokens in self.shard_fields["shard_token_ids"]:
            if tokens.shape[-1] < max_len:
                tokens = F.pad(tokens, (0, max_len - tokens.shape[-1]), value=self.tokenizer.pad_token_id or 0)
            padded_token_ids.append(tokens)

        num_samples = len(padded_token_ids)

        shard_data = {
            "token_ids": torch.stack(padded_token_ids, dim=0),        # [N, T]
            "text_lengths": torch.stack(self.shard_fields["shard_text_lengths"], dim=0),  # [N]
            "text": self.shard_fields["shard_text"],                  # list[str]
            "num_samples": num_samples,
        }

        shard_path = os.path.join(self.output_dir, f"shard_{self.shard_fields['shard_idx']:06d}.pt")
        torch.save(shard_data, shard_path)

        print(f"  Saved shard {self.shard_fields['shard_idx']} ({num_samples} samples)")

        self.shard_fields["shard_token_ids"] = []
        self.shard_fields["shard_text_lengths"] = []
        self.shard_fields["shard_text"] = []
        self.shard_fields["shard_idx"] += self.args.total_gpus

    def process_and_accumulate(self):
        if not self.batch_accumulators.get("batch_texts"):
            return

        try:
            texts = self.batch_accumulators["batch_texts"]

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

            self.stats_accumulator["saved"] += len(texts)

            # Flush shard if full
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
            "dataset_name": self.args.dataset_name,
            "dataset_config": self.args.dataset_config,
            "split": self.args.split,
            "shard_size": self.args.shard_size,
            "stats": self.stats_accumulator,
        }
