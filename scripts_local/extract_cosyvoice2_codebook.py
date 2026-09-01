"""Extract the CosyVoice 2 voice codebook from the model checkpoint.

The codebook is NOT fit from data -- it is CosyVoice 2's own `flow.input_embedding.weight`
(6561 x 512), the frozen flow decoder's input table. Storing it means a predicted unit id
indexes straight into what the decoder expects, so there is no separate embedding to learn
or keep in sync. That also makes this file fully regenerable from the model snapshot, which
is why losing it (as happened when the LibriTTS-R cache was deleted on 2026-08-30) is
recoverable rather than fatal.

Note the off-by-one that trips people: the codebook has 6561 rows (ids 0..6560) but the
model's voice vocab is 6562, because EOV = 6561 sits OUTSIDE the codebook. Consumers bound
ids by `input_embedding.shape[0]` to strip EOV before decoding -- never by flow.input_size,
which is the 512-wide feature dim and would discard almost every unit.

  uv run python scripts_local/extract_cosyvoice2_codebook.py \
      --cosyvoice2_model_dir <CosyVoice2-0.5B snapshot> \
      --out cached_datasets/Mekadrom/libriheavy_cosyvoice2_smollm2/val/cosyvoice2_codebook.pt
"""
import argparse, os
import torch

from megatransformer.utils.codebook import save_codebook, load_codebook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosyvoice2_model_dir", required=True,
                    help="CosyVoice2-0.5B snapshot dir (the one holding flow.pt)")
    ap.add_argument("--out", required=True, help="Destination .pt path")
    ap.add_argument("--key", default="input_embedding.weight",
                    help="State-dict key holding the (K, D) table")
    a = ap.parse_args()

    flow_path = os.path.join(a.cosyvoice2_model_dir, "flow.pt")
    if not os.path.isfile(flow_path):
        raise SystemExit(f"no flow.pt in {a.cosyvoice2_model_dir}")

    sd = torch.load(flow_path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        sd = sd.state_dict()
    if a.key not in sd:
        cands = sorted(k for k in sd if "embedding" in k.lower())
        raise SystemExit(f"key '{a.key}' not in flow.pt. Embedding-ish keys: {cands}")

    w = sd[a.key].detach().cpu().float().clone()  # clone: a state-dict slice keeps the
    if w.ndim != 2:                               # whole storage alive through pickle
        raise SystemExit(f"expected a 2-D (K, D) table, got {tuple(w.shape)}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    save_codebook(a.out, w, meta={
        "source": "CosyVoice2-0.5B flow.pt",
        "source_key": a.key,
        "model_dir": a.cosyvoice2_model_dir,
        "note": "frozen flow-decoder input table; EOV id = K (outside the codebook)",
    })

    back = load_codebook(a.out)
    assert back.shape == w.shape and torch.equal(back, w), "round-trip mismatch"
    print(f"{a.out}: {tuple(w.shape)}  K={w.shape[0]} (EOV id = {w.shape[0]})  "
          f"dim={w.shape[1]}  {os.path.getsize(a.out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
