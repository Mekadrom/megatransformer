"""WHERE does world-image lose the prompt's meaning -- at the text encoder, or downstream?

text_dependency_probe.py measured the END of the path: on minimal pairs (identical bag of words,
inverted meaning) the emitted conditioning separates at ratio ~0.155, against ~0.366 for Qwen3's
own targets. So ~half the available meaning-sensitivity is gone. This localizes the loss by taking
the SAME measurement at each stage:

    llm_body        frozen SmolLM2 hidden states       (the encoder's own output)
    prelude_out     after input_proj + translator      (what the trunk actually receives)
    trunk_gen       recurrent-block output at the image gen-query positions
    conditioning    the emitted Qwen3 prediction       (measured already)

WHY IT DECIDES SOMETHING. If `llm_body` already sits near the final ratio, the ENCODER is the
bottleneck and a better/trainable text encoder (e.g. distilling SmolLM2 into a small trainable
prelude) targets the real problem. If `llm_body` is high and the ratio COLLAPSES downstream, the
encoder is already supplying more meaning than the trunk propagates, and spending a run on the
encoder cannot help -- no encoder can raise the path above what the trunk passes through.

Stages 1-3 are deterministic given the prompt (no flow sampler), so one pass each is enough.

⚠️ DO NOT MEAN-POOL. The minimal pairs hold the token SET fixed and change only which word sits in
which slot, so a mean over tokens is blind to them BY CONSTRUCTION -- an earlier version of this
probe mean-pooled and reported a flat ~0.119 at every stage, which measured the pooling, not the
model. Each stage is instead resampled to K slots (the same F.interpolate the Qwen target path
uses) and flattened, which preserves position and makes the numbers directly comparable to the
flattened GT reference of ~0.366.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts_local/text_signal_localization.py \
        --checkpoint_path runs/world/world_image_zimage_qwen_t3_xskip_0/checkpoint-10000 \
        --config small_sum_zimage_t3_xskip \
        --text_encoder_model HuggingFaceTB/SmolLM2-135M --bf16 \
        --output_dir eval_output/text_dependency/localize_t3_xskip_10000
"""
import argparse
import json
import os

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--text_encoder_model", type=str, default=None)
    p.add_argument("--include_modes", type=str, default="text,image")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seq_len", type=int, default=64,
                   help="K slots every stage is resampled to before flattening; matches the "
                        "Qwen target path so ratios compare to the GT reference")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def _as_tensor(o):
    if hasattr(o, "last_hidden_state"):
        return o.last_hidden_state
    while isinstance(o, (tuple, list)):
        o = o[0]
    return o


def main():
    import numpy as np
    from megatransformer.model.world.world_model import MegaTransformerWorldModel
    from megatransformer.utils import model_loading_utils, constants
    from transformers import AutoTokenizer
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from text_dependency_probe import PAIRS

    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)

    overrides = {"include_modes": [m.strip() for m in args.include_modes.split(",")]}
    if args.text_encoder_model:
        from transformers import AutoConfig
        _llm = AutoConfig.from_pretrained(args.text_encoder_model)
        _eos = _llm.eos_token_id
        if _eos is None:
            _eos = AutoTokenizer.from_pretrained(args.text_encoder_model).eos_token_id
        overrides["special_token_base"] = int(_llm.vocab_size)
        overrides["eos_token_id"] = int(_eos)
        overrides["text_encoder"] = {
            "model": args.text_encoder_model, "freeze": True,
            "translator_hidden_mult": 2.0, "n_special_tokens": constants.N_SPECIAL_TOKENS,
        }
    model = model_loading_utils.load_model(
        MegaTransformerWorldModel, args.config,
        checkpoint_path=args.checkpoint_path, overrides=overrides, device=device)
    model.to(device).eval()

    tfe = model.text_feature_extractor
    taps, caught = {}, {}
    if getattr(tfe, "llm_body", None) is not None:
        taps["llm_body"] = tfe.llm_body
    taps["prelude_out"] = tfe
    taps["trunk_gen"] = model.recurrent_block

    handles = []
    for name, mod in taps.items():
        def mk(n):
            def hook(_m, _i, out):
                caught[n] = _as_tensor(out).detach().float()
            return hook
        handles.append(mod.register_forward_hook(mk(name)))

    tok = AutoTokenizer.from_pretrained(args.text_encoder_model or "mistralai/Mistral-7B-v0.1")
    mcfg = model.config
    stb = int(getattr(mcfg, "special_token_base", 32000))
    sptok = constants.special_token_ids(stb)
    eos = int(getattr(mcfg, "eos_token_id", 2))
    n_gen = int(getattr(model, "_n_image_gen_positions", 0) or 0)

    @torch.no_grad()
    def stage_vecs(prompt):
        """{stage: mean-pooled vector} for one prompt."""
        caught.clear()
        ids = tok(prompt, add_special_tokens=False).input_ids
        seq = ids + [sptok.BOI, sptok.IMAGE_PLACEHOLDER, sptok.EOI, eos]
        text_input_ids = torch.tensor([seq], dtype=torch.long, device=device)
        image_inputs = torch.zeros(1, 1, 12, 32, 32, device=device)
        is_synth = torch.tensor([True], device=device)
        with torch.amp.autocast(device, dtype=dtype, enabled=args.bf16):
            model(text_input_ids=text_input_ids, image_inputs=image_inputs,
                  precomputed_latents=True, is_synthesis=is_synth, decode_outputs=False)
        out = {}
        for n, t in caught.items():
            if t.dim() != 3:
                continue
            # the trunk sees the interleaved sequence; the image gen queries are its TAIL
            if n == "trunk_gen" and n_gen and t.shape[1] > n_gen:
                t = t[:, -n_gen:, :]
            elif n in ("llm_body", "prelude_out"):
                t = t[:, :len(ids), :]      # real caption tokens only, no BOI/IPH/EOI/eos
            # resample to K slots and FLATTEN -- preserves which word is in which slot
            t = torch.nn.functional.interpolate(
                t.transpose(1, 2), size=args.seq_len, mode="linear",
                align_corners=False).transpose(1, 2)
            out[n] = t[0].flatten().cpu()
        return out

    def d(a, b):
        return float(1.0 - torch.nn.functional.cosine_similarity(a, b, dim=0))

    per_stage = {}
    for i, (base, swapped) in enumerate(PAIRS):
        other = PAIRS[(i + 1) % len(PAIRS)][0]
        vb, vs, vo = stage_vecs(base), stage_vecs(swapped), stage_vecs(other)
        for n in vb:
            if n not in vs or n not in vo:
                continue
            per_stage.setdefault(n, []).append(
                {"prompt": base, "d_swap": d(vb[n], vs[n]), "d_other": d(vb[n], vo[n])})
        print(f"[{i}] " + "  ".join(
            f"{n}: {d(vb[n], vs[n]):.4f}/{d(vb[n], vo[n]):.4f}" for n in vb), flush=True)

    for h in handles:
        h.remove()

    order = [n for n in ("llm_body", "prelude_out", "trunk_gen") if n in per_stage]
    summary = {"checkpoint": args.checkpoint_path, "n_gen_positions": n_gen, "stages": {}}
    print("\n| stage | mean d_swap | mean d_other | swap/other ratio |")
    print("|---|---|---|---|")
    for n in order:
        rows = per_stage[n]
        ms = float(np.mean([r["d_swap"] for r in rows]))
        mo = float(np.mean([r["d_other"] for r in rows]))
        # per-pair ratios then averaged: a pair with a tiny d_other must not dominate
        ratio = float(np.mean([r["d_swap"] / max(r["d_other"], 1e-9) for r in rows]))
        summary["stages"][n] = {"mean_d_swap": ms, "mean_d_other": mo, "ratio": ratio,
                                "rows": rows}
        print(f"| {n} | {ms:.4f} | {mo:.4f} | {ratio:.3f} |")
    print("\nreference: emitted conditioning ~0.155 (w=3) | Qwen3 ground truth ~0.366")
    print("ratio FLAT from llm_body to trunk_gen => the encoder is NOT the bottleneck;")
    print("  a better/trainable text encoder cannot raise what the trunk fails to propagate.")

    with open(os.path.join(args.output_dir, "localization.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {os.path.join(args.output_dir, 'localization.json')}")


if __name__ == "__main__":
    main()
