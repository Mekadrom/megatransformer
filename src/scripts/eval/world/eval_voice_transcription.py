"""Evaluate voice transcription (voice → text) using Word Error Rate.

For each voice clip in the evaluation dataset, generates a text transcript
from the world model and computes WER against the ground-truth text.

Requires: pip install jiwer

Usage:
    python -m src.scripts.eval.world.eval_voice_transcription --checkpoint_path runs/my_run/checkpoint-3000 --config small_sum_dit --cache_dir ../cached_datasets/sive --include_modes text,voice --max_samples 100 --bf16
"""

import argparse
import os
import sys

import torch
from torch.amp import autocast

from model.world.world_model import MegaTransformerWorldModel
from utils import model_loading_utils
from utils.constants import (
    BOV_TOKEN_ID, EOV_TOKEN_ID, VOICE_PLACEHOLDER_TOKEN_ID,
)


def parse_args():
    p = argparse.ArgumentParser(description="Voice transcription eval (WER)")
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--config", type=str, default="small_sum_dit")
    p.add_argument("--include_modes", type=str, default="text,voice")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--text_cache_dir", type=str, default=None)
    p.add_argument("--voice_cache_dir", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--use_memorization_dataset", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--tie_word_embeddings", action="store_true")
    p.add_argument("--split", type=str, default="val", help="Dataset split (train/val)")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_world_model(args, device):
    include_modes = [m.strip() for m in args.include_modes.split(",")]
    overrides = {"include_modes": include_modes}
    if args.tie_word_embeddings:
        overrides["tie_word_embeddings"] = True
    return model_loading_utils.load_model(
        MegaTransformerWorldModel, args.config,
        checkpoint_path=args.checkpoint_path,
        overrides=overrides, device=device,
    )


def load_dataset(args, split="val"):
    include_modes = [m.strip() for m in args.include_modes.split(",")]

    def resolve(specific, base, s):
        d = specific or base
        if d is None:
            return None
        for candidate in [d + "_" + s, d]:
            if os.path.isdir(candidate):
                return candidate
        return None

    text_dir = resolve(args.text_cache_dir, args.cache_dir, split) if "text" in include_modes else None
    voice_dir = resolve(args.voice_cache_dir, args.cache_dir, split) if "voice" in include_modes else None

    if args.use_memorization_dataset:
        from scripts.data.world.memorization_dataset import MultimodalMemorizationDataset
        return MultimodalMemorizationDataset(
            text_shard_dir=text_dir, voice_shard_dir=voice_dir,
            max_samples=args.max_samples,
        )
    else:
        from scripts.data.world.dataset import MultimodalShardedDataset
        return MultimodalShardedDataset(
            text_shard_dir=text_dir, voice_shard_dir=voice_dir,
            cache_size=32, max_samples=args.max_samples,
        )


def decode_tokens(token_ids, tokenizer):
    text_ids = [t for t in token_ids if t < 32000 and t != 0]
    return tokenizer.decode(text_ids, skip_special_tokens=True)


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    try:
        from jiwer import wer as compute_wer
    except ImportError:
        print("ERROR: jiwer not installed. Run: pip install jiwer")
        sys.exit(1)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    # Load world model
    print(f"Loading world model from {args.checkpoint_path}...")
    model = load_world_model(args, device)
    model.to(device)
    model.eval()

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args, split=args.split)
    print(f"Dataset: {len(dataset)} samples")

    # Eval loop
    wer_scores = []
    references = []
    hypotheses = []

    for i in range(len(dataset)):
        sample = dataset[i]
        voice_features = sample.get("voice_features")
        if voice_features is None:
            continue

        text_token_ids = sample.get("text_token_ids")
        if text_token_ids is None:
            continue

        # Get ground-truth text
        target_text = sample.get("text_text", sample.get("voice_voice_text", ""))
        if not target_text:
            text_length = sample.get("text_text_length", len(text_token_ids))
            if isinstance(text_length, torch.Tensor):
                text_length = text_length.item()
            target_text = decode_tokens(text_token_ids[:text_length].tolist(), tokenizer)
        if isinstance(target_text, list):
            target_text = target_text[0] if target_text else ""
        target_text = str(target_text).strip()
        if not target_text:
            continue

        # Build transcription prompt: [BOV] [VOICE_PH] [EOV]
        prompt = torch.tensor(
            [[BOV_TOKEN_ID, VOICE_PLACEHOLDER_TOKEN_ID, EOV_TOKEN_ID]],
            dtype=torch.long, device=device,
        )
        voice_input = voice_features.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, C, T)
        feat_length = sample.get("voice_feature_length", voice_features.shape[-1])
        if isinstance(feat_length, torch.Tensor):
            feat_length = feat_length.item()
        voice_len = torch.tensor([[feat_length]], device=device)

        # Generate transcript
        with torch.no_grad():
            with autocast(device, dtype=dtype, enabled=args.bf16):
                outputs = model.generate(
                    text_input_ids=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    voice_inputs=voice_input,
                    voice_lengths=voice_len,
                )

        gen_ids = outputs.get("generated_token_ids")
        if gen_ids is None:
            continue
        hypothesis = decode_tokens(gen_ids[0].tolist(), tokenizer).strip()

        if not hypothesis:
            hypothesis = "<empty>"

        # WER
        sample_wer = compute_wer(target_text, hypothesis)
        wer_scores.append(sample_wer)
        references.append(target_text)
        hypotheses.append(hypothesis)

        print(f"[{i}] WER={sample_wer:.4f}")
        print(f"     Generated: {hypothesis[:120]}")
        print(f"     Target:    {target_text[:120]}")
        print()

    if wer_scores:
        scores = torch.tensor(wer_scores)
        # Also compute corpus-level WER (more meaningful than mean of per-sample WERs)
        corpus_wer = compute_wer(references, hypotheses)
        print(f"\n{'='*60}")
        print(f"Voice Transcription Results ({len(wer_scores)} samples)")
        print(f"  Corpus WER:     {corpus_wer:.4f}")
        print(f"  Mean WER:       {scores.mean():.4f}")
        print(f"  Median WER:     {scores.median():.4f}")
        print(f"  Std:            {scores.std():.4f}")
        print(f"  Min WER:        {scores.min():.4f}")
        print(f"  Max WER:        {scores.max():.4f}")
        print(f"  Perfect (0.0):  {(scores == 0).sum().item()}/{len(wer_scores)}")
        print(f"{'='*60}")
    else:
        print("No voice samples found in dataset.")


if __name__ == "__main__":
    main()
