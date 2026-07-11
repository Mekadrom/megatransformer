"""Generate speaker-embedding .pt files for probing the SMG's speaker space.

Writes a bank of [dim] float tensors (compatible with voice_clone.py's --content_encoder
uploader / load_embedding_file) representing canonical test points:

  Synthetic (no data):
    zeros, ones, neg_ones, randn_seed{S}
  Data-derived (from LibriTTS-R speaker embeddings in --cache_dir):
    mean_utt          mean over ALL utterances
    mean_speaker      mean over PER-SPEAKER means (the "average speaker", count-unbiased)
    std               per-dim std (a spread vector, not a real speaker)
    gauss_matched     ~N(per-dim mean, per-dim std) — a plausible-SCALE random point
    pc{K}_{±}{k}sigma  mean_speaker ± k·σ along principal identity axis K (SVD of
                       per-speaker means) — the MEANINGFUL directional probes
    real_spk{ID}       a few random real per-speaker centroids (on-manifold anchors)

NOTE on scale: if the SMG L2-normalizes the embedding internally (the vec256 run does),
only DIRECTION matters, so zeros is degenerate and magnitude is irrelevant. Pass
--match_norm to rescale the synthetic points to the mean real norm anyway (needed only
for a model trained with normalize_speaker_embedding=False).

Usage:
  python -m megatransformer.scripts.eval.smg.make_probe_embeddings \
    --cache_dir ./cached_datasets/smg_libritts_r_clean_contentvec_val \
    --output_dir eval_outputs/probe_embeddings_0
Then in voice_clone: Speaker source = "uploaded .pt", upload any of these files.
"""
import argparse
import json
import os

import numpy as np
import torch


def gather(cache_dirs, dim, max_per_cache, seed):
    from megatransformer.scripts.data.voice.dataset import VoiceShardedDataset
    E, S = [], []
    for cd in cache_dirs:
        ds = VoiceShardedDataset(cd, columns=["speaker_embeddings", "speaker_ids"])
        idx = range(len(ds))
        if max_per_cache and max_per_cache < len(ds):
            idx = np.random.RandomState(seed).choice(len(ds), max_per_cache, replace=False)
        for i in idx:
            s = ds[int(i)]
            e = s["speaker_embedding"].float().reshape(-1)
            if e.numel() == dim:
                E.append(e); S.append(int(s["speaker_id"]))
    if not E:
        return None, None
    return torch.stack(E), np.array(S)


def per_speaker_means(E, S):
    spk = sorted(set(S.tolist()))
    M = torch.stack([E[torch.from_numpy(S == s)].mean(0) for s in spk])
    return M, spk


def main():
    ap = argparse.ArgumentParser(description="Make speaker-embedding probe .pt files")
    ap.add_argument("--cache_dir", action="append", default=None,
                    help="Cache dir with speaker_embeddings (repeatable). More speakers = better mean/PCA.")
    ap.add_argument("--output_dir", default="eval_outputs/probe_embeddings_0")
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--max_per_cache", type=int, default=0, help="Cap utts read per cache (0 = all).")
    ap.add_argument("--pca_k", type=int, default=3, help="Number of principal identity axes.")
    ap.add_argument("--pca_sigmas", default="-2,-1,1,2", help="Comma sigma multiples along each PC.")
    ap.add_argument("--n_real", type=int, default=3, help="How many random real per-speaker centroids to dump.")
    ap.add_argument("--basis_dims", default="", help="Comma dim indices for unit basis vectors e_i (empty = none).")
    ap.add_argument("--match_norm", action="store_true",
                    help="Rescale synthetic points (zeros excepted) to the mean real norm.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    caches = args.cache_dir or ["./cached_datasets/smg_libritts_r_clean_contentvec_val"]
    dim = args.dim
    manifest = {}

    def dump(name, vec):
        vec = torch.as_tensor(vec).float().reshape(-1)
        assert vec.numel() == dim, (name, vec.numel())
        path = os.path.join(args.output_dir, f"{name}.pt")
        torch.save(vec.clone(), path)
        manifest[name] = {"norm": round(float(vec.norm()), 3), "mean": round(float(vec.mean()), 4)}

    # ---- data-derived stats ----
    E, S = gather(caches, dim, args.max_per_cache, args.seed)
    real_norm = float(E.norm(dim=1).mean()) if E is not None else 1.0
    unit = (lambda v: v)  # identity; scaling handled per-point below

    def maybe_scale(v):
        if args.match_norm and E is not None:
            n = v.norm()
            if n > 1e-8:
                return v / n * real_norm
        return v

    # ---- synthetic ----
    dump("zeros", torch.zeros(dim))
    dump("ones", maybe_scale(torch.ones(dim)))
    dump("neg_ones", maybe_scale(-torch.ones(dim)))
    for sd in (0, 1):
        g = torch.Generator().manual_seed(args.seed + sd)
        dump(f"randn_seed{args.seed + sd}", maybe_scale(torch.randn(dim, generator=g)))
    for d in [int(x) for x in args.basis_dims.split(",") if x.strip() != ""]:
        v = torch.zeros(dim); v[d] = 1.0
        dump(f"basis_e{d}", maybe_scale(v))

    if E is None:
        print(f"[warn] no embeddings loaded from {caches} — wrote synthetic points only.")
    else:
        print(f"Loaded {E.shape[0]} embeddings, {len(set(S.tolist()))} speakers; mean ||e||={real_norm:.2f}")
        mean_utt = E.mean(0)
        M, spk = per_speaker_means(E, S)
        mean_speaker = M.mean(0)
        dump("mean_utt", mean_utt)
        dump("mean_speaker", mean_speaker)
        dump("std", E.std(0))
        g = torch.Generator().manual_seed(args.seed)
        dump("gauss_matched", E.mean(0) + E.std(0) * torch.randn(dim, generator=g))

        # PCA on per-speaker means (speaker-to-speaker variation = identity axes)
        Mc = M - mean_speaker
        # SVD: rows = speakers; V columns = principal directions
        U, Sv, Vh = torch.linalg.svd(Mc, full_matrices=False)
        sigmas = Sv / max(1.0, (M.shape[0] - 1) ** 0.5)  # per-PC std
        sig_mults = [float(x) for x in args.pca_sigmas.split(",") if x.strip()]
        for k in range(min(args.pca_k, Vh.shape[0])):
            pc = Vh[k]
            for m in sig_mults:
                tag = f"pc{k}_{'m' if m < 0 else 'p'}{abs(m):g}sig"
                dump(tag, mean_speaker + m * float(sigmas[k]) * pc)

        # real per-speaker centroids (on-manifold anchors)
        rng = np.random.RandomState(args.seed)
        for sid in rng.choice(spk, size=min(args.n_real, len(spk)), replace=False):
            dump(f"real_spk{int(sid)}", M[spk.index(int(sid))])

    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(manifest)} embeddings -> {args.output_dir}")
    for n, m in manifest.items():
        print(f"  {n:24s} ||e||={m['norm']:8.2f}  mean={m['mean']:+.4f}")


if __name__ == "__main__":
    main()
