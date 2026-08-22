"""Does the world-image path actually USE word ORDER, or only the bag of words?

MOTIVATION. The frozen SmolLM2-135M text spine (--text_encoder_model) was added for world-VOICE's
text conditioning, where it barely moved the needle. world-IMAGE inherited it on the assumption it
was a crutch. Before paying for a re-tokenization + a full ablation run, ask the cheap question:
is the pretrained LM contributing anything a from-scratch {wte + prelude} would not?

A frozen pretrained LM's distinctive contribution over a small from-scratch embedding table is
CONTEXTUAL, order-sensitive structure. So: hold the bag of words FIXED and change only the
syntax/roles. If the predicted Qwen3 conditioning does not move, the path is consuming
bag-of-words and SmolLM2's pretraining is inert here.

WHY THIS IS FREE. It reads `image_clip_seq_pred` straight off the world model -- no Z-Image, no
Qwen3-4B, no CLIP. For fixed-K (T3) heads `cond_length` is None, so the Qwen tokenizer is not
needed either. A few GB and a couple of minutes.

THE NOISE FLOOR MATTERS. T3's flow head is a SAMPLER: two different flow seeds on the SAME prompt
give different conditioning. That spread is the floor any perturbation effect must clear. We
measure it explicitly (d_seed) and report every perturbation RELATIVE to it:

    sensitivity = (d_perturb - d_seed) / (d_other - d_seed)

  ~0.0 -> the perturbation is indistinguishable from resampling noise: BAG OF WORDS.
  ~1.0 -> the perturbation moves conditioning as much as a completely different prompt.

d_other (vs an unrelated prompt) is the natural upper reference. Both floors are measured, not
assumed, so a null here is a real null and not a missing-effect-size artifact.

Distances are cosines in the WHITENED space (the adapter de-whitens `image_clip_seq_pred` back to
Qwen3 space, where per-dim scale differences would otherwise dominate). Raw-space numbers are
printed alongside.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts_local/text_dependency_probe.py \
        --checkpoint_path runs/world/world_image_zimage_qwen_t3_xskip_0/checkpoint-10000 \
        --config small_sum_zimage_t3_xskip \
        --text_encoder_model HuggingFaceTB/SmolLM2-135M --bf16 \
        --output_dir eval_output/text_dependency/t3_xskip_10000
"""
import argparse
import json
import os

import torch


# Hand-authored minimal pairs: IDENTICAL bag of words, different meaning. Swapping roles or
# attributes rather than deleting anything keeps token statistics fixed, so any movement in the
# conditioning is attributable to ORDER/SYNTAX and nothing else.
PAIRS = [
    ("an astronaut riding a horse across a desert under a pink sky",
     "a horse riding an astronaut across a desert under a pink sky"),
    ("a vast empty desert landscape at golden hour, no people",
     "a golden empty desert landscape at vast hour, no people"),
    ("a single red door standing in an endless white void",
     "a single white door standing in an endless red void"),
    ("a bioluminescent forest at night with glowing blue mushrooms",
     "a bioluminescent mushroom at night with glowing blue forests"),
    ("a giant whale floating in the sky above a city skyline",
     "a giant city skyline floating in the sky above a whale"),
    ("a cozy empty library interior with tall bookshelves, no people",
     "a tall empty bookshelf interior with cozy libraries, no people"),
    ("a steaming bowl of ramen on a wooden table, top-down view",
     "a wooden bowl of ramen on a steaming table, top-down view"),
    ("A child holding a flowered umbrella and petting a yak.",
     "A yak holding a flowered umbrella and petting a child."),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--text_encoder_model", type=str, default=None,
                   help="frozen LM spine. OMIT to probe a from-scratch-prelude checkpoint.")
    p.add_argument("--include_modes", type=str, default="text,image")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--flow_guidance", type=float, nargs="+", default=[1.0, 3.0],
                   help="CFG scale(s) for the flow sampler, swept in ONE model load. MUST include "
                        "the operating point: unguided conditioning is noise-DOMINATED (the "
                        "rank-512 leak), so w=1 numbers say nothing about what gets rendered.")
    p.add_argument("--n_seeds", type=int, default=4,
                   help="flow-sampler draws per prompt variant; the seed spread IS the noise floor")
    p.add_argument("--scramble_seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    import numpy as np
    from megatransformer.model.world.world_model import MegaTransformerWorldModel
    from megatransformer.utils import model_loading_utils, constants

    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)

    overrides = {"include_modes": [m.strip() for m in args.include_modes.split(",")]}
    if args.text_encoder_model:
        from transformers import AutoConfig, AutoTokenizer
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

    adapter = model.image_generator
    if getattr(adapter, "ar_flow_head", None) is not None or \
            bool(getattr(adapter, "flow_native_length", False)):
        raise SystemExit("native-length / AR heads need a Qwen tokenizer for cond_length; "
                         "this probe supports fixed-K (T3) heads only")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.text_encoder_model or "mistralai/Mistral-7B-v0.1")
    mcfg = model.config
    stb = int(getattr(mcfg, "special_token_base", 32000))
    sptok = constants.special_token_ids(stb)
    eos = int(getattr(mcfg, "eos_token_id", 2))

    w_mean = adapter.whiten_mean.detach().float().cpu()
    w_std = adapter.whiten_std.detach().float().cpu().clamp_min(1e-6)
    whitening_on = bool(getattr(adapter, "whiten", False))

    @torch.no_grad()
    def cond_for(prompt, seed):
        """Predicted Qwen3 conditioning [L, 2560] for one prompt at one flow-sampler seed."""
        if getattr(adapter, "flow_head", None) is not None:
            g = torch.Generator(device=device)
            g.manual_seed(int(seed))
            adapter.flow_generator = g
        ids = tok(prompt, add_special_tokens=False).input_ids
        seq = ids + [sptok.BOI, sptok.IMAGE_PLACEHOLDER, sptok.EOI, eos]
        text_input_ids = torch.tensor([seq], dtype=torch.long, device=device)
        image_inputs = torch.zeros(1, 1, 12, 32, 32, device=device)
        is_synth = torch.tensor([True], device=device)
        with torch.amp.autocast(device, dtype=dtype, enabled=args.bf16):
            out = model(text_input_ids=text_input_ids, image_inputs=image_inputs,
                        precomputed_latents=True, is_synthesis=is_synth, decode_outputs=False)
        sp = out.get("image_clip_seq_pred")
        if sp is None:
            raise SystemExit("model produced no image_clip_seq_pred -- wrong config/checkpoint?")
        return sp[0].float().cpu()

    def dist(a, b, whiten=True):
        """1 - cosine over the flattened conditioning sequence."""
        if whiten and whitening_on:
            a = (a - w_mean) / w_std
            b = (b - w_mean) / w_std
        a, b = a.flatten(), b.flatten()
        return float(1.0 - torch.nn.functional.cosine_similarity(a, b, dim=0))

    rng = np.random.default_rng(args.scramble_seed)

    def scramble(prompt):
        w = prompt.split()
        for _ in range(32):
            p = list(rng.permutation(len(w)))
            if p != sorted(p):
                break
        return " ".join(w[i] for i in p)

    seeds = list(range(args.n_seeds))
    head = getattr(adapter, "flow_head", None)

    def measure(w):
        if head is not None:
            head.guidance = float(w)
        cache = {}

        def conds(prompt):
            if prompt not in cache:
                cache[prompt] = [cond_for(prompt, s_) for s_ in seeds]
            return cache[prompt]

        rows = []
        for i, (base, swapped) in enumerate(PAIRS):
            scr = scramble(base)
            other = PAIRS[(i + 1) % len(PAIRS)][0]
            cb, cs, cc, co = conds(base), conds(swapped), conds(scr), conds(other)
            d_seed = float(np.mean([dist(cb[a], cb[b]) for a in range(len(seeds))
                                    for b in range(a + 1, len(seeds))]))
            d_swap = float(np.mean([dist(cb[k], cs[k]) for k in range(len(seeds))]))
            d_scram = float(np.mean([dist(cb[k], cc[k]) for k in range(len(seeds))]))
            d_other = float(np.mean([dist(cb[k], co[k]) for k in range(len(seeds))]))
            span = d_other - d_seed
            rows.append({
                "prompt": base, "swapped": swapped, "scrambled": scr,
                "d_seed": d_seed, "d_swap": d_swap, "d_scramble": d_scram, "d_other": d_other,
                # normalized sensitivity is only meaningful when a DIFFERENT prompt moves the
                # conditioning further than resampling does; otherwise the prompt-conditional
                # signal is below the sampler's own noise and the ratio is meaningless.
                "sens_swap": ((d_swap - d_seed) / span) if span > 1e-6 else None,
                "sens_scramble": ((d_scram - d_seed) / span) if span > 1e-6 else None,
                "signal_over_noise": d_other / max(d_seed, 1e-9),
            })
            print(f"  [{i}] d_seed={d_seed:.4f} d_swap={d_swap:.4f} d_scram={d_scram:.4f} "
                  f"d_other={d_other:.4f}", flush=True)
        return rows

    print(f"whitening={'on' if whitening_on else 'OFF'}  seeds={seeds}  "
          f"guidance={args.flow_guidance}", flush=True)
    by_w = {}
    for w in args.flow_guidance:
        print(f"\n=== flow_guidance w={w} ===", flush=True)
        by_w[str(w)] = measure(w)

    summary = {
        "checkpoint": args.checkpoint_path, "config": args.config,
        "text_encoder_model": args.text_encoder_model, "n_seeds": args.n_seeds,
        "by_guidance": by_w,
    }
    with open(os.path.join(args.output_dir, "text_dependency.json"), "w") as f:
        json.dump(summary, f, indent=2)

    def fmt(x):
        return "  n/a " if x is None else f"{x:6.3f}"

    print("\n| w | d_seed (noise floor) | d_swap | d_scramble | d_other (ref) | "
          "d_other/d_seed | sens_swap | sens_scram |")
    print("|---|---|---|---|---|---|---|---|")
    for w, rows in by_w.items():
        m = lambda k: float(np.mean([r[k] for r in rows]))
        ss = [r["sens_swap"] for r in rows if r["sens_swap"] is not None]
        sc = [r["sens_scramble"] for r in rows if r["sens_scramble"] is not None]
        print(f"| {w} | {m('d_seed'):.4f} | {m('d_swap'):.4f} | {m('d_scramble'):.4f} | "
              f"{m('d_other'):.4f} | {m('signal_over_noise'):.2f} | "
              f"{fmt(float(np.mean(ss)) if ss else None)} | "
              f"{fmt(float(np.mean(sc)) if sc else None)} |")
    print("\nd_other/d_seed < 1 => the sampler draw moves conditioning MORE than the prompt does;")
    print("  at that guidance the conditioning is noise-dominated and sensitivity is unreadable.")
    print("sens ~0 => bag-of-words (SmolLM2's contextual structure inert); ~1 => order matters")
    print("  as much as a whole different prompt.")
    print(f"\nwrote {os.path.join(args.output_dir, 'text_dependency.json')}")


if __name__ == "__main__":
    main()
