"""Export Mimi's semantic (codebook-0) VQ centroids into the repo codebook format,
so the SMG can (a) init its unit embedding from them and (b) carry per-speaker F0
stats for the speaker-normalized contour (injected later by
`preprocess_dataset.py stat-shards --fit_codebook_f0_stats`).

Mimi is a SPLIT residual VQ: codebook 0 is the WavLM-distilled *semantic* stream
(content), codebooks 1-31 are acoustic (timbre/speaker) — we take only cb0. Its
codebook is 2048x256; 256 == the SMG's default sive_encoder_dim, so id -> 256-d
centroid drops straight into the existing content-input width.

  uv run python -m megatransformer.scripts.data.voice.extract_mimi_codebook \
      --out ./cached_datasets/<smg_mimi_dir>/mimi_semantic_codebook.pt
"""
import argparse

import torch

from megatransformer.utils.codebook import load_codebook, save_codebook


def extract_mimi_semantic_centroids(model_id="kyutai/mimi") -> torch.Tensor:
    """Return the (2048, 256) normalized semantic codebook-0 centroid matrix."""
    from transformers import MimiModel
    m = MimiModel.from_pretrained(model_id)
    cb = m.quantizer.semantic_residual_vector_quantizer.layers[0].codebook
    # `embed` is the EMA-normalized centroid matrix (embed_sum / cluster_usage).
    centroids = cb.embed.detach().cpu().float()  # (K, D) = (2048, 256)
    return centroids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output codebook .pt path")
    ap.add_argument("--model_id", default="kyutai/mimi")
    a = ap.parse_args()

    centroids = extract_mimi_semantic_centroids(a.model_id)
    save_codebook(a.out, centroids, meta={
        "source": f"mimi:{a.model_id}:semantic_cb0",
        "frame_rate": 12.5,
        "codebook": "semantic (WavLM-distilled), split-RVQ layer 0",
    })
    # round-trip check
    got = load_codebook(a.out)
    print(f"saved Mimi semantic codebook -> {a.out}")
    print(f"  centroids {tuple(got.shape)}  (K={got.shape[0]} codes, dim={got.shape[1]})")
    print("  next: point preprocessing at this file and run "
          "`stat-shards --fit_codebook_f0_stats --codebook_name "
          f"{a.out.split('/')[-1]}` to inject per-speaker F0 stats.")


if __name__ == "__main__":
    main()
