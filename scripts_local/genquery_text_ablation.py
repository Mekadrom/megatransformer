"""Do the image gen queries READ the text at all, or read it and land in the same place?

trunk_compression_probe.py showed the gen-query output is ~96% prompt-invariant and two
unrelated prompts sit 0.4% cosine apart. Two very different causes, needing different fixes:

  ROUTING   the gen queries barely attend to the text positions at all. Only injecting text
            into the queries directly can fix that.
  DILUTION  they read the text fine, but the attention write is tiny next to the norm-83
            constant, and/or the trunk maps different prompts to nearly the same point.
            Then the init scale / output normalisation / depth-init are the levers.

Attention weights are not recoverable here (SDPA never materialises them, and rebuilding
RoPE-applied q/k by hand is error-prone), so this measures the CAUSAL contribution instead:
zero the text prelude's output and see how far the gen-query states move.

    text_dep = ||h_full - h_notext|| / ||h_full||     over the gen-query positions

  text_dep ~0     => ROUTING: the text is not reaching the gen queries at all.
  text_dep large  => DILUTION: text IS read; the small PROMPT-TO-PROMPT spread (0.4%) then
                     means the trunk maps different prompts to nearly the same place, which
                     is a collapse in the mapping, not a failure to look.

Reported next to the same quantity at the TEXT positions, which must be large (they are the
text), and against the prompt-to-prompt distance for scale.
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
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    import numpy as np
    from megatransformer.model.world.world_model import MegaTransformerWorldModel
    from megatransformer.utils import model_loading_utils, constants
    from transformers import AutoTokenizer, AutoConfig
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from text_dependency_probe import PAIRS

    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)

    overrides = {"include_modes": [m.strip() for m in args.include_modes.split(",")]}
    if args.text_encoder_model:
        _llm = AutoConfig.from_pretrained(args.text_encoder_model)
        _eos = _llm.eos_token_id or AutoTokenizer.from_pretrained(
            args.text_encoder_model).eos_token_id
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

    n_gen = int(getattr(model, "_n_image_gen_positions", 0) or 0)
    tok = AutoTokenizer.from_pretrained(args.text_encoder_model or "mistralai/Mistral-7B-v0.1")
    mcfg = model.config
    sptok = constants.special_token_ids(int(getattr(mcfg, "special_token_base", 32000)))
    eos = int(getattr(mcfg, "eos_token_id", 2))

    # zero the text prelude's output -> the trunk sees text positions carrying no content,
    # while sequence length, positions and the gen queries are all unchanged.
    ablate = {"on": False}

    def _kill_text(_m, _i, out):
        if not ablate["on"]:
            return out
        if isinstance(out, tuple):
            return (torch.zeros_like(out[0]),) + tuple(out[1:])
        return torch.zeros_like(out)

    model.text_feature_extractor.register_forward_hook(_kill_text)

    cap = {}
    model.recurrent_block.register_forward_hook(
        lambda _m, _i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach().float()))

    @torch.no_grad()
    def run(prompt, no_text):
        ablate["on"] = no_text
        ids = tok(prompt, add_special_tokens=False).input_ids
        seq = ids + [sptok.BOI, sptok.IMAGE_PLACEHOLDER, sptok.EOI, eos]
        with torch.amp.autocast(device, dtype=dtype, enabled=args.bf16):
            model(text_input_ids=torch.tensor([seq], dtype=torch.long, device=device),
                  image_inputs=torch.zeros(1, 1, 12, 32, 32, device=device),
                  precomputed_latents=True,
                  is_synthesis=torch.tensor([True], device=device),
                  decode_outputs=False)
        ablate["on"] = False
        h = cap["h"][0]
        return h[-n_gen:, :].cpu(), h[:len(ids), :].cpu()

    rows = []
    prompts = [b for b, _ in PAIRS]
    for i, prompt in enumerate(prompts):
        g_full, t_full = run(prompt, no_text=False)
        g_none, t_none = run(prompt, no_text=True)
        other = prompts[(i + 1) % len(prompts)]
        g_other, _ = run(other, no_text=False)
        gen_dep = float((g_full - g_none).norm() / g_full.norm().clamp_min(1e-9))
        txt_dep = float((t_full - t_none).norm() / t_full.norm().clamp_min(1e-9))
        prompt_spread = float((g_full - g_other).norm() / g_full.norm().clamp_min(1e-9))
        rows.append({"prompt": prompt, "gen_text_dep": gen_dep,
                     "text_pos_text_dep": txt_dep, "gen_prompt_spread": prompt_spread})
        print(f"[{i}] gen_text_dep={gen_dep:.4f}  text_pos_dep={txt_dep:.4f}  "
              f"gen_prompt_spread={prompt_spread:.4f}", flush=True)

    m = lambda k: float(np.mean([r[k] for r in rows]))
    summary = {"checkpoint": args.checkpoint_path, "n_gen": n_gen, "rows": rows,
               "mean_gen_text_dep": m("gen_text_dep"),
               "mean_text_pos_text_dep": m("text_pos_text_dep"),
               "mean_gen_prompt_spread": m("gen_prompt_spread")}
    with open(os.path.join(args.output_dir, "genquery_text_ablation.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nmean ||dh||/||h|| at GEN queries when text is ablated : {m('gen_text_dep'):.4f}")
    print(f"mean ||dh||/||h|| at TEXT positions (sanity, must be big): {m('text_pos_text_dep'):.4f}")
    print(f"mean ||dh||/||h|| at GEN queries across DIFFERENT prompts: {m('gen_prompt_spread'):.4f}")
    print("\ngen_text_dep ~0        => ROUTING: text never reaches the gen queries.")
    print("gen_text_dep >> spread => DILUTION: text IS read, but different prompts land in")
    print("                          nearly the same place -- a collapsed mapping.")


if __name__ == "__main__":
    main()
