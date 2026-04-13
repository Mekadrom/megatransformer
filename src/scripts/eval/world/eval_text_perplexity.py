"""Evaluate text-only perplexity on held-out data.

Runs the world model in teacher-forcing mode on text samples and computes
per-token cross-entropy and perplexity.

Usage:
    python -m src.scripts.eval.world.eval_text_perplexity --checkpoint_path runs/my_run/checkpoint-3000 --config small_sum_dit --text_cache_dir ../cached_datasets/text_pile --max_samples 500 --bf16
"""

import argparse
import math
import os

import torch
import torch.nn.functional as F
from torch.amp import autocast

from model.world.world_model import MegaTransformerWorldModel
from utils import model_loading_utils


def parse_args():
    p = argparse.ArgumentParser(description="Text perplexity eval")
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--config", type=str, default="small_sum_dit")
    p.add_argument("--text_cache_dir", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--use_memorization_dataset", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--tie_word_embeddings", action="store_true")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--split", type=str, default="val", help="Dataset split (train/val)")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_world_model(args, device):
    overrides = {"include_modes": ["text"]}
    if args.tie_word_embeddings:
        overrides["tie_word_embeddings"] = True
    return model_loading_utils.load_model(
        MegaTransformerWorldModel, args.config,
        checkpoint_path=args.checkpoint_path,
        overrides=overrides, device=device,
    )


def load_dataset(args, split="val"):
    def resolve(specific, base, s):
        d = specific or base
        if d is None:
            return None
        for candidate in [d + "_" + s, d]:
            if os.path.isdir(candidate):
                return candidate
        return None

    text_dir = resolve(args.text_cache_dir, args.cache_dir, split)

    if args.use_memorization_dataset:
        from scripts.data.world.memorization_dataset import MultimodalMemorizationDataset
        return MultimodalMemorizationDataset(
            text_shard_dir=text_dir, max_samples=args.max_samples,
        )
    else:
        from scripts.data.world.dataset import MultimodalShardedDataset
        return MultimodalShardedDataset(
            text_shard_dir=text_dir, cache_size=32, max_samples=args.max_samples,
        )


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    print(f"Loading world model from {args.checkpoint_path}...")
    model = load_world_model(args, device)
    model.to(device)
    model.eval()

    print("Loading dataset...")
    dataset = load_dataset(args, split=args.split)
    print(f"Dataset: {len(dataset)} samples")

    total_loss = 0.0
    total_tokens = 0

    for i in range(len(dataset)):
        sample = dataset[i]
        token_ids = sample.get("text_token_ids")
        if token_ids is None:
            continue

        text_length = sample.get("text_text_length", len(token_ids))
        if isinstance(text_length, torch.Tensor):
            text_length = text_length.item()
        text_length = min(text_length, args.max_seq_len)
        if text_length < 2:
            continue

        tokens = token_ids[:text_length].unsqueeze(0).to(device)  # (1, T)
        input_ids = tokens[:, :-1]
        target_ids = tokens[:, 1:]

        with torch.no_grad():
            with autocast(device, dtype=dtype, enabled=args.bf16):
                outputs = model(text_input_ids=input_ids)

        logits = outputs.get("logits")
        if logits is None:
            continue

        T_min = min(logits.shape[1], target_ids.shape[1])
        logits = logits[:, :T_min, :].contiguous()
        targets = target_ids[:, :T_min].contiguous()

        # Per-token cross-entropy (no reduction)
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            targets.view(-1),
            reduction="sum",
        )

        n_tokens = T_min
        total_loss += loss.item()
        total_tokens += n_tokens

        if (i + 1) % 50 == 0:
            running_ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
            print(f"  [{i+1}/{len(dataset)}] running perplexity: {running_ppl:.2f}")

    if total_tokens > 0:
        avg_ce = total_loss / total_tokens
        perplexity = math.exp(avg_ce)
        print(f"\n{'='*60}")
        print(f"Text Perplexity Results ({total_tokens:,} tokens from {len(dataset)} samples)")
        print(f"  Cross-entropy:  {avg_ce:.4f} nats")
        print(f"  Perplexity:     {perplexity:.2f}")
        print(f"  Bits per token: {avg_ce / math.log(2):.4f}")
        print(f"{'='*60}")
    else:
        print("No text samples found.")


if __name__ == "__main__":
    main()
