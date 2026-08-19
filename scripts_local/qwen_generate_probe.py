"""What does Qwen3 ACTUALLY say about each probe prompt -- and does our conditioning match it?

The logit lens showed the Z-Image conditioning is not the caption: it is Qwen3's ANTICIPATED
RESPONSE (the decodes contain <think>, 描述/描绘, 画面). So the meaningful reference for our
adapter is not the prompt text but what Qwen3 would generate. This runs that generation and
scores content-word overlap against the GT and OURS decodes produced by qwen_logit_lens.py.

Uses Qwen3ForCausalLM so one 4-bit model serves both generation and (via .lm_head) any
unembedding, with the SAME chat template the conditioning uses (add_generation_prompt=True,
enable_thinking=True). Only the first tokens matter for the comparison: the conditioning is
the state at the END of the prompt, which anticipates the START of the response.
"""

import argparse, json, os, re

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Tongyi-MAI/Z-Image-Turbo")
    p.add_argument("--prompts_file", type=str,
                   default="src/megatransformer/scripts/eval/world/zimage_probe_prompts.txt")
    p.add_argument("--decoded", type=str, default="eval_output/zimage_logit_lens/decoded.json",
                   help="decoded.json from qwen_logit_lens.py (JSON, because decoded strings "
                        "contain literal newlines that a line-oriented format truncates)")
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="eval_output/zimage_logit_lens")
    return p.parse_args()


STOP = set("""a an the of in on at and with no is are to over above across under it its this that
for from by as be was were will would can could there their they them he she his her you your i we
our us not but or if then than so such very more most some any each other another one two""".split())


def content_words(text):
    ws = re.findall(r"[a-zA-Z][a-zA-Z\-']{2,}", text.lower())
    return {w for w in ws if w not in STOP}


def main():
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    kw = dict(subfolder="text_encoder", torch_dtype=torch.bfloat16)
    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        kw["device_map"] = {"": device}
    except Exception:
        pass
    m = Qwen3ForCausalLM.from_pretrained(args.model, **kw).eval()

    prompts = [l.strip() for l in open(args.prompts_file) if l.strip()]

    # parse the lens output: PROMPT/GT/OURS triples
    dec = {}
    if os.path.exists(args.decoded):
        for r in json.load(open(args.decoded)):
            dec[r["prompt"]] = {"gt": r.get("gt", ""), "ours": r.get("ours", "")}
    if not dec:
        raise SystemExit(f"no decodes parsed from {args.decoded} -- run qwen_logit_lens.py first")

    rows = []
    for p in prompts:
        s = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=True)
        ids = tok(s, return_tensors="pt").to(m.device)
        with torch.no_grad():
            out = m.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=False)
        gw = content_words(gen)
        d = dec.get(p, {})
        gtw, ow = content_words(d.get("gt", "")), content_words(d.get("ours", ""))
        def ov(a, b):
            return (len(a & b), f"{100*len(a & b)/max(len(a | b),1):.0f}%")
        n_gt, j_gt = ov(gw, gtw)
        n_ou, j_ou = ov(gw, ow)
        print("=" * 100)
        print(f"PROMPT: {p}")
        print(f"  QWEN3 SAYS: {gen[:300]!r}")
        print(f"  overlap with GT decode  : {n_gt} words ({j_gt} jaccard)  shared={sorted(gw & gtw)[:8]}")
        print(f"  overlap with OURS decode: {n_ou} words ({j_ou} jaccard)  shared={sorted(gw & ow)[:8]}", flush=True)
        rows.append({"prompt": p, "generation": gen, "overlap_gt": n_gt, "overlap_ours": n_ou,
                     "shared_gt": sorted(gw & gtw), "shared_ours": sorted(gw & ow)})
    json.dump(rows, open(os.path.join(args.output_dir, "qwen_generations.json"), "w"), indent=2)
    tg = sum(r["overlap_gt"] for r in rows) / len(rows)
    to = sum(r["overlap_ours"] for r in rows) / len(rows)
    print("=" * 100)
    print(f"MEAN content-word overlap with Qwen3's own generation:  GT decode {tg:.2f}   OURS decode {to:.2f}")
    print(f"wrote {args.output_dir}/qwen_generations.json")


if __name__ == "__main__":
    main()
