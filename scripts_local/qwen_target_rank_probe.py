"""How many dimensions does the whitened Qwen3 conditioning target actually occupy?

WHY. CondFlowHead's output is Linear(flow_dim -> seq_dim) = Linear(512 -> 2560), so everything the
head can WRITE lies in a rank-512 subspace. flow_x_skip / flow_x1_pred fix the NOISE half of that
problem (the ODE can now cancel its own initial noise), but the CONTENT is still rank-512. Whether
that is a real bottleneck depends entirely on the target's own spectrum:
  - if 512 principal components capture ~99% of the whitened target's variance, widening flow_dim
    buys nothing and the workarounds are sufficient;
  - if they capture ~85%, the head is structurally unable to express ~15% of the target and
    widening flow_dim is the highest-value change available.
This measures it directly, so that decision is made on evidence rather than intuition.

Whitening matters: the raw per-dim std spans 555x (median 4.85, max 2691), so a raw-space PCA
would just rediscover the loud dims. We whiten with the SAME stats the model trains against, then
take the covariance across the 2560 feature dims, pooling over all real tokens.

Usage (single line):
  CUDA_VISIBLE_DEVICES=3 uv run python scripts_local/qwen_target_rank_probe.py --cache_dir ./cached_datasets/Mekadrom/image_gen_captions_only/train --stats ./cached_datasets/qwen_whiten_stats_k64.pt --n_samples 2048
"""
import argparse, torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--stats", required=True, help="the whitening stats the model trains against")
    p.add_argument("--n_samples", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--native", action="store_true", help="native-length targets instead of K-slot resample")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_4bit", action="store_true")
    args = p.parse_args()

    import os
    from megatransformer.scripts.data.world.dataset import MultimodalShardedDataset
    from megatransformer.scripts.data.world.data_collator import MultimodalDataCollator
    from megatransformer.utils.zimage_text_encoder import Qwen3TextTargetEncoder

    def resolve(d, sub):                       # same convention as compute_qwen_whiten_stats.py
        for c in (d + "_" + sub, d):
            if os.path.isdir(c):
                return c
        return None
    tdir = resolve(args.cache_dir, "train")
    ds = MultimodalShardedDataset(text_shard_dir=tdir, image_shard_dir=tdir,
                                  cache_size=8, max_samples=args.n_samples)
    collator = MultimodalDataCollator(special_token_base=49152, eos_token_id=0)
    collator.force_direction = "synthesis"
    caps = []
    for i in range(len(ds)):
        caps.append((collator([ds[i]]).get("text_texts") or [""])[0] or "")
    print(f"gathered {len(caps)} captions", flush=True)

    st = torch.load(args.stats)
    mean = st["mean"].to(args.device).float()
    std = st["std"].to(args.device).float().clamp_min(1e-6)
    enc = Qwen3TextTargetEncoder(seq_len=args.seq_len, device=args.device,
                                 load_in_4bit=not args.no_4bit)

    # streaming covariance over the 2560 FEATURE dims, float64 on CPU
    D = 2560
    n = 0
    s1 = torch.zeros(D, dtype=torch.float64)
    s2 = torch.zeros(D, D, dtype=torch.float64)
    for i in range(0, len(caps), args.batch_size):
        b = caps[i:i + args.batch_size]
        if args.native:
            hs, m = enc.encode_native(b, max_len=128)
            t = hs[m.bool()]
        else:
            t = enc.encode(b).reshape(-1, D)
        t = ((t.to(args.device).float() - mean) / std).double().cpu()   # whitened space
        s1 += t.sum(0); s2 += t.T @ t; n += t.shape[0]
        if (i // args.batch_size) % 10 == 0:
            print(f"  {n} tokens...", flush=True)

    mu = (s1 / n)
    cov = s2 / n - torch.outer(mu, mu)
    ev = torch.linalg.eigvalsh(cov).flip(0).clamp_min(0)      # descending
    tot = ev.sum()
    cum = torch.cumsum(ev, 0) / tot
    print(f"\nwhitened target covariance over {n:,} tokens ({'native' if args.native else 'K=%d resampled' % args.seq_len})")
    print(f"  effective rank (participation ratio) = {(ev.sum()**2 / (ev**2).sum()).item():.1f} of {D}")
    for k in (64, 128, 256, 512, 768, 1024, 1536, 2048):
        print(f"  top {k:4d} components capture {100*cum[k-1].item():6.2f}% of variance")
    for frac in (0.90, 0.95, 0.99):
        k = int((cum < frac).sum().item()) + 1
        print(f"  {int(frac*100)}% of variance needs {k} components")
    print(f"\n=> flow_dim is 512. Compare against the rows above: if 512 already captures ~99%,")
    print(f"   widening it is pointless; if it captures ~85%, the head cannot express the rest.")


if __name__ == "__main__":
    main()
