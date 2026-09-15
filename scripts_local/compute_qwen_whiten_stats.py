"""Compute per-dim Qwen3 target mean/std for the Z-Image adapter's Tier-0 whitened MSE.

Samples N captions from the training set, encodes them EXACTLY as the trainer target
(Qwen3TextTargetEncoder: chat template -> penultimate hidden_states -> resample to seq_len),
and accumulates per-dim (2560) mean/std over all tokens. Saves {mean, std} to a .pt for
--image_whiten_stats_path. Run ONCE; the stats are a fixed property of the Qwen3 target
space (independent of the adapter's weights).

Usage (K-slot resample, T0-T3):
    CUDA_VISIBLE_DEVICES=3 python scripts_local/compute_qwen_whiten_stats.py --cache_dir ./cached_datasets/Mekadrom/image_gen_captions_only/train --n_samples 8192 --seq_len 64 --output ./cached_datasets/qwen_whiten_stats_k64.pt

Usage (NATIVE length, T5 -- --native, real tokens only):
    CUDA_VISIBLE_DEVICES=3 python scripts_local/compute_qwen_whiten_stats.py --cache_dir ./cached_datasets/Mekadrom/image_gen_captions_only/train --n_samples 8192 --seq_len 128 --native --output ./cached_datasets/qwen_whiten_stats_native.pt
"""
import argparse
import os

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", type=str, required=True, help="train shard dir (text+image)")
    p.add_argument("--n_samples", type=int, default=8192)
    p.add_argument("--shuffle", action="store_true",
                   help="Sample captions RANDOMLY instead of taking the first n_samples in cache "
                        "order. Required for a representative estimate: the cache concatenates "
                        "caption sources of very different lengths without shuffling.")
    p.add_argument("--seed", type=int, default=0, help="shuffle seed")
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no_4bit", action="store_true", help="load Qwen3 bf16 instead of 4-bit")
    p.add_argument("--native", action="store_true",
                   help="accumulate over NATIVE-length targets (real tokens only, masked) "
                        "instead of the K-slot resample. Use for the T5 native-length head: "
                        "F.interpolate averages neighbouring hidden states, which SHRINKS "
                        "per-dim variance, so k64 stats would leave native targets whitened to "
                        "slightly above unit std.")
    return p.parse_args()


def main():
    args = parse_args()
    from megatransformer.utils.zimage_text_encoder import Qwen3TextTargetEncoder
    from megatransformer.scripts.data.world.dataset import MultimodalShardedDataset
    from megatransformer.scripts.data.world.data_collator import MultimodalDataCollator

    def resolve(d, s):
        for c in (d + "_" + s, d):
            if os.path.isdir(c):
                return c
        return None
    tdir = resolve(args.cache_dir, "train")
    dataset = MultimodalShardedDataset(text_shard_dir=tdir, image_shard_dir=tdir,
                                       cache_size=8, max_samples=args.n_samples)
    collator = MultimodalDataCollator(special_token_base=49152, eos_token_id=0)
    collator.force_direction = "synthesis"

    # gather captions
    # ⚠️ ORDER MATTERS. The cache is an UNSHUFFLED CONCATENATION of caption sources with very
    # different lengths (measured 2026-09-15: shards 0-39 average ~11.9 SmolLM2 tokens, shards
    # 150-164 average ~105, corpus mean 31.94). Walking indices 0..n_samples therefore samples ONLY
    # the shortest source. The ORIGINAL k64/native stats were built that way and describe the
    # shortest ~1.2% of the corpus. The trainer itself shuffles (ModalityGroupedSampler), so the
    # stats did not match the data they were applied to.
    caps = []
    order = list(range(len(dataset)))
    if args.shuffle:
        import random as _random
        _random.Random(args.seed).shuffle(order)
    for i in order:
        b = collator([dataset[i]])
        c = (b.get("text_texts") or [""])[0]
        if c:
            caps.append(c)
        if len(caps) >= args.n_samples:
            break
    print(f"gathered {len(caps)} captions", flush=True)

    enc = Qwen3TextTargetEncoder(seq_len=args.seq_len, device=args.device,
                                 load_in_4bit=not args.no_4bit)

    # streaming per-dim mean/std over all (B*seq_len) tokens, float64 accumulators on CPU
    dim = enc.seq_len and 2560  # Qwen3-4B hidden
    total = torch.zeros(2560, dtype=torch.float64)
    total_sq = torch.zeros(2560, dtype=torch.float64)
    count = 0
    for i in range(0, len(caps), args.batch_size):
        batch = caps[i:i + args.batch_size]
        if args.native:
            hs, m = enc.encode_native(batch, max_len=args.seq_len)
            t = hs[m.bool()].reshape(-1, 2560).double().cpu()      # real tokens only
        else:
            t = enc.encode(batch).reshape(-1, 2560).double().cpu()  # (B*K, 2560)
        total += t.sum(0)
        total_sq += (t * t).sum(0)
        count += t.shape[0]
        if (i // args.batch_size) % 10 == 0:
            print(f"  {count} tokens...", flush=True)

    mean = (total / count)
    var = (total_sq / count) - mean * mean
    std = var.clamp_min(0).sqrt()
    mean_f = mean.float(); std_f = std.float()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    torch.save({"mean": mean_f, "std": std_f, "n_tokens": count,
                "n_captions": len(caps), "seq_len": args.seq_len,
                "native": bool(args.native)}, args.output)
    print(f"\nsaved {args.output}")
    print(f"tokens={count}  per-dim std: min={std_f.min():.3f} median={std_f.median():.3f} "
          f"max={std_f.max():.3f}  (max/median = {float(std_f.max()/std_f.median()):.1f}x -> whitening lever)")
    print(f"per-dim |mean|: max={mean_f.abs().max():.3f} (massive-constant dims)")


if __name__ == "__main__":
    main()
