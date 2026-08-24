"""WHY are the trunk's image gen-query outputs nearly prompt-invariant?

text_signal_localization.py found the recurrent trunk compresses prompt variation ~100x: two
COMPLETELY different prompts move the gen-query output by d_other=0.0038 cosine, vs 0.44 at the
prelude. This asks WHERE that comes from, because two mechanisms imply opposite fixes:

  (a) BORN SMALL. The learned gen queries are initialized at std=3.0 (world_model.py:195) -- a
      deliberate 3x scale-up, added as a large per-position CONSTANT. If the conditional part is
      already a tiny fraction of that constant at the trunk INPUT, the trunk is not suppressing
      anything; the constant is drowning the prompt. Fix = the query scale / injection, not the
      recurrence.
  (b) SUPPRESSED. If the conditional fraction starts healthy at x_0 and DECAYS with each
      recurrent iteration, the recurrent dynamics are washing it out. Fix = the recurrence
      (injection type, depth, iteration count).

⭐ HISTORY WORTH KNOWING. `gen_query_mode='positional_only'` once collapsed ALL gen positions to
near-identical outputs, diagnosed via `image_seq_var` and fixed with LEARNED queries at std=3.0.
But `image_seq_var` measures spread ACROSS POSITIONS, which learned per-position constants raise
BY CONSTRUCTION -- so that diagnostic could read "fixed" while ACROSS-PROMPT collapse persisted
untouched. This probe reports both quantities separately so they cannot be confused again.

DECOMPOSITION. For N prompts, at each stage, split the gen-query representation into
    mu   = mean over prompts        (the prompt-INVARIANT part: learned queries + biases)
    r_i  = h_i - mu                 (the prompt-CONDITIONAL part)
and report mean||r|| / ||mu||. Text positions are carried as a control: they ARE the prompt, so
their conditional fraction should be large at every stage. If the text control stays high while
the gen queries do not, the compression is specific to the gen-query path.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts_local/trunk_compression_probe.py \
        --checkpoint_path runs/world/world_image_zimage_qwen_t3_xskip_0/checkpoint-10000 \
        --config small_sum_zimage_t3_xskip \
        --text_encoder_model HuggingFaceTB/SmolLM2-135M --bf16 \
        --output_dir eval_output/text_dependency/trunk_compression_10000
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

    rb = model.recurrent_block
    n_gen = int(getattr(model, "_n_image_gen_positions", 0) or 0)

    # The adapter does NOT see the raw trunk output: world_model.py applies
    # image_coda_input_norm (a LayerNorm) first. LayerNorm normalises each token across its
    # FEATURE axis, which cannot subtract a PER-POSITION constant vector -- but it does rescale
    # it, so measure how much of the constant actually survives to the adapter's input.
    post = {}
    _icn = getattr(model, "image_coda_input_norm", None)
    if _icn is not None:
        _icn.register_forward_hook(
            lambda _m, _i, o: post.__setitem__("h", o.detach().float()))
    tok = AutoTokenizer.from_pretrained(args.text_encoder_model or "mistralai/Mistral-7B-v0.1")
    mcfg = model.config
    sptok = constants.special_token_ids(int(getattr(mcfg, "special_token_base", 32000)))
    eos = int(getattr(mcfg, "eos_token_id", 2))

    # learned gen-query scale, for the "born small" hypothesis
    q = getattr(model, "image_gen_queries", None)
    q_info = None
    if q is not None:
        qd = q.detach().float()
        q_info = {"shape": list(qd.shape), "std": float(qd.std()),
                  "mean_row_norm": float(qd.reshape(-1, qd.shape[-1]).norm(dim=-1).mean())}
        print(f"learned gen queries: shape={q_info['shape']} std={q_info['std']:.3f} "
              f"mean row norm={q_info['mean_row_norm']:.2f}", flush=True)

    captured = {}
    orig_run = rb._run_iteration

    def patched(x_0, thought, iteration, *a, **kw):
        if "x_0" not in captured:
            captured["x_0"] = x_0.detach().float().cpu()
        out = orig_run(x_0, thought, iteration, *a, **kw)
        captured.setdefault("iters", []).append(out.detach().float().cpu())
        return out

    rb._run_iteration = patched

    @torch.no_grad()
    def run(prompt):
        captured.clear()
        ids = tok(prompt, add_special_tokens=False).input_ids
        seq = ids + [sptok.BOI, sptok.IMAGE_PLACEHOLDER, sptok.EOI, eos]
        tii = torch.tensor([seq], dtype=torch.long, device=device)
        with torch.amp.autocast(device, dtype=dtype, enabled=args.bf16):
            model(text_input_ids=tii,
                  image_inputs=torch.zeros(1, 1, 12, 32, 32, device=device),
                  precomputed_latents=True,
                  is_synthesis=torch.tensor([True], device=device),
                  decode_outputs=False)
        pn = post.get("h")
        return (len(ids), captured["x_0"][0], [t[0] for t in captured["iters"]],
                None if pn is None else pn[0].cpu())

    prompts = [p for pair in PAIRS for p in pair]
    n_text, x0, iters, _pn = run(prompts[0])
    print(f"seq_len={x0.shape[0]} n_text={n_text} n_gen={n_gen} "
          f"iterations={len(iters)}", flush=True)

    stages = ["x_0"] + [f"iter{i}" for i in range(len(iters))]
    gen_acc = {s: [] for s in stages}
    txt_acc = {s: [] for s in stages}
    post_acc = []
    for p in prompts:
        nt, x0, its, pn = run(p)
        if pn is not None:
            post_acc.append(pn[-n_gen:, :])
        for s, t in zip(stages, [x0] + its):
            gen_acc[s].append(t[-n_gen:, :])
            txt_acc[s].append(t[:nt, :].mean(0, keepdim=True))   # pooled text control

    def decompose(mats):
        """mean||h_i - mu|| / ||mu||, plus the ACROSS-POSITION spread of mu itself."""
        H = torch.stack(mats)                       # (P, pos, d)
        mu = H.mean(0)                              # (pos, d)  prompt-invariant part
        cond = (H - mu).norm(dim=-1).mean()         # mean conditional magnitude
        const = mu.norm(dim=-1).mean()
        pos_center = mu.mean(0, keepdim=True)
        pos_spread = (mu - pos_center).norm(dim=-1).mean()
        return {"cond_over_const": float(cond / const.clamp_min(1e-9)),
                "cond_norm": float(cond), "const_norm": float(const),
                # what image_seq_var tracks: spread ACROSS POSITIONS of the invariant part
                "pos_spread_over_const": float(pos_spread / const.clamp_min(1e-9))}

    def cosines(mats):
        """Legible form: cosine similarity, same position across prompts vs across positions."""
        H = torch.stack(mats)                                  # (P, pos, d)
        P, npos, _ = H.shape
        Hn = H / H.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        # same POSITION, different PROMPTS -> "does this slot react to the prompt?"
        cross_prompt = []
        for a in range(P):
            for b in range(a + 1, P):
                cross_prompt.append((Hn[a] * Hn[b]).sum(-1).mean())
        # different POSITIONS, same PROMPT -> "do the slots differ from each other?"
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(npos, generator=g)[:min(npos, 32)]
        cross_pos = []
        for a in range(P):
            M = Hn[a][idx]
            sim = M @ M.T
            off = ~torch.eye(len(idx), dtype=torch.bool)
            cross_pos.append(sim[off].mean())
        return (float(torch.stack(cross_prompt).mean()),
                float(torch.stack(cross_pos).mean()))

    print("\n| stage | cos(same position, DIFFERENT prompts) | cos(DIFFERENT positions, same prompt) |")
    print("|---|---|---|")
    for s in (stages[0], stages[1], stages[-1]):
        cp, cq = cosines(gen_acc[s])
        print(f"| {s} | {cp:.4f} | {cq:.4f} |")

    rows = []
    for s in stages:
        g, t = decompose(gen_acc[s]), decompose(txt_acc[s])
        rows.append({"stage": s, "gen": g, "text": t})
        print(f"{s:8s} gen cond/const={g['cond_over_const']:.4f} "
              f"(pos_spread/const={g['pos_spread_over_const']:.4f})  "
              f"text cond/const={t['cond_over_const']:.4f}", flush=True)

    if post_acc:
        d_post = decompose(post_acc)
        print(f"\nAFTER image_coda_input_norm (what the adapter actually receives): "
              f"cond/const={d_post['cond_over_const']:.4f}  "
              f"pos_spread/const={d_post['pos_spread_over_const']:.4f}")
        print("  -> compare with the final iter row above (BEFORE the LayerNorm).")

    rb._run_iteration = orig_run
    summary = {"checkpoint": args.checkpoint_path, "n_gen": n_gen,
               "gen_queries": q_info, "n_prompts": len(prompts), "stages": rows}
    with open(os.path.join(args.output_dir, "trunk_compression.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n| stage | gen cond/const | gen pos_spread/const | text cond/const |")
    print("|---|---|---|---|")
    for r in rows:
        print(f"| {r['stage']} | {r['gen']['cond_over_const']:.4f} | "
              f"{r['gen']['pos_spread_over_const']:.4f} | {r['text']['cond_over_const']:.4f} |")
    first, last = rows[0]["gen"]["cond_over_const"], rows[-1]["gen"]["cond_over_const"]
    print(f"\nx_0 -> final conditional fraction: {first:.4f} -> {last:.4f} "
          f"({last / max(first, 1e-9):.2f}x)")
    print("flat & already tiny at x_0 => BORN SMALL: the std=3.0 constant drowns the prompt.")
    print("healthy at x_0, decaying per iteration => SUPPRESSED by the recurrence.")
    print(f"\nwrote {os.path.join(args.output_dir, 'trunk_compression.json')}")


if __name__ == "__main__":
    main()
