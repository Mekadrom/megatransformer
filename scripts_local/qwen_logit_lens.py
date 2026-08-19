"""Decode the adapter's predicted Qwen3 conditioning back into TOKENS (logit lens).

The Z-Image adapter emits 64 x 2560 vectors that are supposed to live where Qwen3-4B's
penultimate hidden states live. Those are normally opaque -- we only ever judged them by
what the frozen DiT renders. Qwen3 ties its embeddings (`tie_word_embeddings: True`), so
the unembedding IS `embed_tokens.weight`, and we can read the conditioning back as text.

This answers directly what render outcomes could only imply: is the word "door" / "yak" /
"empty" actually PRESENT in the conditioning our model produces, or is the DiT recovering
it from context?

TWO CORRECTIONS THE NAIVE VERSION GETS WRONG:
  1. Qwen3-4B has 36 layers, so output_hidden_states returns 37 entries and the LAST is
     already final-norm'd. `hidden_states[-2]` -- our target -- is the output of layer 34,
     i.e. layer 35 AND the final norm still have not run. `--mode faithful` (default) runs
     both before unembedding; `--mode lens` skips them (the logit lens proper, approximate).
  2. The trainer resamples a variable-length Qwen sequence to exactly K=64 with a LINEAR
     interpolation, which BLENDS adjacent positions. Decoded slots are therefore mixtures
     of neighbouring tokens, not clean ones. Short prompts get upsampled (mild); long
     captions get downsampled and smear badly.

SANITY CHECK, always run first: decoding the GT conditioning must reproduce the chat
template + caption almost verbatim. If it doesn't, the decode is wrong and nothing else
this script prints can be trusted.

Example:
  uv run python scripts_local/qwen_logit_lens.py --checkpoint_path runs/world/world_image_zimage_qwen_t3_2/checkpoint-20000 --config small_sum_zimage_t3 --text_encoder_model HuggingFaceTB/SmolLM2-135M --bf16 --flow_guidance 3.0 --topk 5
"""

import argparse
import os

import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--config", type=str, default="small_sum_zimage_t3")
    p.add_argument("--include_modes", type=str, default="text,image")
    p.add_argument("--text_encoder_model", type=str, default=None)
    p.add_argument("--prompts_file", type=str,
                   default="src/megatransformer/scripts/eval/world/zimage_probe_prompts.txt")
    p.add_argument("--keywords", type=str, default=None,
                   help="Comma-separated words to hunt for per prompt (default: auto -- the "
                        "content words of each prompt).")
    p.add_argument("--mode", type=str, default="faithful", choices=["faithful", "lens"],
                   help="faithful = run the remaining layer + final norm before unembedding; "
                        "lens = unembed the penultimate state directly (logit lens proper).")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--flow_guidance", type=float, default=None)
    p.add_argument("--flow_seed", type=int, default=4242)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="eval_output/zimage_logit_lens")
    return p.parse_args()


def main():
    from megatransformer.model.world.world_model import MegaTransformerWorldModel, ZImageConditioningAdapter
    from megatransformer.utils import model_loading_utils, constants
    from megatransformer.utils.zimage_text_encoder import Qwen3TextTargetEncoder

    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)

    overrides = {"include_modes": [m.strip() for m in args.include_modes.split(",")]}
    if args.text_encoder_model:
        from transformers import AutoConfig, AutoTokenizer
        _llm = AutoConfig.from_pretrained(args.text_encoder_model)
        _eos = _llm.eos_token_id or AutoTokenizer.from_pretrained(args.text_encoder_model).eos_token_id
        overrides["special_token_base"] = int(_llm.vocab_size)
        overrides["eos_token_id"] = int(_eos)
        overrides["text_encoder"] = {"model": args.text_encoder_model, "freeze": True,
                                     "translator_hidden_mult": 2.0,
                                     "n_special_tokens": constants.N_SPECIAL_TOKENS}
    model = model_loading_utils.load_model(
        MegaTransformerWorldModel, args.config, checkpoint_path=args.checkpoint_path,
        overrides=overrides, device=device)
    model.to(device).eval()
    gen = model.image_generator
    assert isinstance(gen, ZImageConditioningAdapter), "not a Z-Image adapter config"
    if args.flow_guidance is not None and gen.flow_head is not None:
        gen.flow_head.guidance = float(args.flow_guidance)
        g = torch.Generator(device=device); g.manual_seed(args.flow_seed)
        gen.flow_generator = g
        print(f"[lens] flow guidance w={args.flow_guidance}, seed {args.flow_seed}", flush=True)
    seq_len = int(gen.config.seq_len)

    from transformers import AutoTokenizer
    stok = AutoTokenizer.from_pretrained(args.text_encoder_model)
    stb = int(getattr(model.config, "special_token_base", 32000))
    sp = constants.special_token_ids(stb)
    eos = int(getattr(model.config, "eos_token_id", 2))

    prompts = [l.strip() for l in open(args.prompts_file) if l.strip()]

    @torch.no_grad()
    def predict(caption):
        ids = stok(caption, add_special_tokens=False).input_ids
        seq = ids + [sp.BOI, sp.IMAGE_PLACEHOLDER, sp.EOI, eos]
        with torch.amp.autocast(device, dtype=dtype, enabled=args.bf16):
            out = model(text_input_ids=torch.tensor([seq], dtype=torch.long, device=device),
                        image_inputs=torch.zeros(1, 1, 12, 32, 32, device=device),
                        precomputed_latents=True,
                        is_synthesis=torch.tensor([True], device=device), decode_outputs=False)
        return out["image_clip_seq_pred"][0].float()      # (K, 2560) Qwen3 space

    preds = {c: predict(c) for c in prompts}
    del model
    torch.cuda.empty_cache()

    enc = Qwen3TextTargetEncoder(
        model_name=getattr(gen.config, "target_model", "Tongyi-MAI/Z-Image-Turbo"),
        seq_len=seq_len, device=device,
        load_in_4bit=getattr(gen.config, "target_load_in_4bit", True))
    qm, qtok = enc.model, enc.tokenizer
    W = qm.embed_tokens.weight                            # tied unembedding (vocab, 2560)
    last_layer, final_norm = qm.layers[-1], qm.norm
    n_layers = len(qm.layers)
    print(f"[lens] Qwen3: {n_layers} layers, unembed {tuple(W.shape)}, mode={args.mode}", flush=True)

    @torch.no_grad()
    def decode(h):
        """h (K, 2560) penultimate-space -> (K, topk) token strings + the argmax string."""
        x = h.unsqueeze(0).to(W.dtype).to(W.device)
        if args.mode == "faithful":
            pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
            emb = qm.rotary_emb(x, pos)
            out = last_layer(x, position_embeddings=emb, attention_mask=None, position_ids=pos)
            x = out[0] if isinstance(out, tuple) else out
        x = final_norm(x)
        logits = x[0].to(W.dtype) @ W.T                   # (K, vocab)
        top = logits.topk(args.topk, dim=-1).indices
        rows = [[qtok.decode([int(t)]) for t in row] for row in top]
        return rows, "".join(r[0] for r in rows)

    def kws(prompt):
        if args.keywords:
            return [w.strip() for w in args.keywords.split(",") if w.strip()]
        stop = {"a","an","the","of","in","on","at","and","with","no","is","to","over","above","across","under"}
        return [w.strip(".,").lower() for w in prompt.split() if w.strip(".,").lower() not in stop]

    report = []
    for c in prompts:
        gt = enc.encode([c])[0].to(device)                # (K, 2560), same resampling as training
        gt_rows, gt_str = decode(gt)
        pr_rows, pr_str = decode(preds[c].to(device))
        print("\n" + "=" * 100)
        print(f"PROMPT: {c}")
        print(f"  GT   -> {gt_str[:220]!r}")
        print(f"  OURS -> {pr_str[:220]!r}")
        # keyword presence anywhere in the top-k of any slot
        line = []
        for w in kws(c):
            ing = any(w in t.lower() for row in gt_rows for t in row)
            inp = any(w in t.lower() for row in pr_rows for t in row)
            line.append(f"{w}[GT {'Y' if ing else '.'}|ours {'Y' if inp else '.'}]")
        print("  keywords: " + "  ".join(line))
        report.append((c, gt_str, pr_str))

    with open(os.path.join(args.output_dir, "decoded.txt"), "w") as f:
        for c, g_, p_ in report:
            f.write(f"PROMPT: {c}\nGT  : {g_}\nOURS: {p_}\n\n")
    print(f"\nwrote {args.output_dir}/decoded.txt", flush=True)


if __name__ == "__main__":
    main()
