"""Is resampling the Qwen3 conditioning to K=64 costing us anything?

The trainer squashes a variable-length Qwen3 sequence (~18-23 tokens for the probe prompts)
to exactly K=64 with a linear interpolation -- a ~3x UPSAMPLE. The measured "GT ceiling"
(0.368) was itself produced with that same resampling, so interpolation is not a handicap
RELATIVE TO THAT NUMBER. What is unknown is whether it depresses the ceiling itself.

This renders the GROUND-TRUTH conditioning two ways, same prompts, same seeds:
  native   -- Qwen3's real masked hidden states at their natural length (no resampling)
  interp64 -- the same states resampled to K=64, i.e. exactly what training targets
and CLIPScores both. No world model is loaded; this is purely about the target pipeline.

If native > interp64, the resampling costs everyone and native-length (masked) targets are
worth building. If they tie, the resampling is harmless and only loss-masking the constant
chat-template positions has any value.
"""

import argparse, json, os, statistics as st

import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Tongyi-MAI/Z-Image-Turbo")
    p.add_argument("--prompts_file", type=str,
                   default="src/megatransformer/scripts/eval/world/zimage_probe_prompts.txt")
    p.add_argument("--seq_len", type=int, default=64, help="the K the trainer resamples to")
    p.add_argument("--gen_steps", type=int, default=8)
    p.add_argument("--guidance", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--n_seeds", type=int, default=2, help="renders per prompt per variant")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="eval_output/zimage_native_length")
    return p.parse_args()


def main():
    import numpy as np
    from PIL import Image
    from diffusers import ZImagePipeline
    import open_clip

    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    prompts = [l.strip() for l in open(args.prompts_file) if l.strip()]

    pipe = ZImagePipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    cm, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    ctok = open_clip.get_tokenizer("ViT-B-32")
    cm = cm.to(device).eval()
    from torchvision import transforms as T
    cnorm = T.Compose([T.Resize(224), T.CenterCrop(224),
                       T.Normalize((0.48145466, 0.4578275, 0.40821073),
                                   (0.26862954, 0.26130258, 0.27577711))])

    @torch.no_grad()
    def cond(caption):
        """-> (native (L,2560), interp64 (K,2560)) -- identical to the trainer's target path."""
        s = pipe.tokenizer.apply_chat_template([{"role": "user", "content": caption}],
                                               tokenize=False, add_generation_prompt=True,
                                               enable_thinking=True)
        ti = pipe.tokenizer(s, padding="longest", return_tensors="pt")
        ids = ti.input_ids.to(pipe.text_encoder.device)
        attn = ti.attention_mask.to(pipe.text_encoder.device)
        hs = pipe.text_encoder(input_ids=ids, attention_mask=attn,
                               output_hidden_states=True).hidden_states[-2][0]
        real = hs[attn.bool()[0]]
        x = real.transpose(0, 1).unsqueeze(0).float()
        r = F.interpolate(x, size=args.seq_len, mode="linear", align_corners=False)
        return real.float(), r[0].transpose(0, 1)

    @torch.no_grad()
    def render(seq, seed):
        g = torch.Generator(device=device).manual_seed(seed)
        return pipe(prompt_embeds=[seq.to(device, torch.bfloat16)],
                    num_inference_steps=args.gen_steps, guidance_scale=args.guidance,
                    height=1024, width=1024, generator=g).images[0]

    @torch.no_grad()
    def score(img, txt):
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
        im = F.normalize(cm.encode_image(cnorm(x)), dim=-1)
        tx = F.normalize(cm.encode_text(ctok([txt[:77]]).to(device)), dim=-1)
        return float((im @ tx.T).item())

    nat_all, int_all, rows = [], [], []
    for i, p in enumerate(prompts):
        nat, itp = cond(p)
        ns, is_, imgs = [], [], []
        for k in range(args.n_seeds):
            sd = args.seed + 100 * k
            a, b = render(nat, sd), render(itp, sd)
            ns.append(score(a, p)); is_.append(score(b, p)); imgs += [a, b]
        nat_all += ns; int_all += is_
        rows.append(np.concatenate([np.asarray(im.resize((288, 288))) for im in imgs], 1))
        print(f"  [{i}] native {st.mean(ns):.3f}  interp64 {st.mean(is_):.3f}  "
              f"({st.mean(ns)-st.mean(is_):+.3f})  L={nat.shape[0]} | {p[:44]}", flush=True)
    Image.fromarray(np.concatenate(rows, 0)).save(
        os.path.join(args.output_dir, "native_vs_interp64.png"))
    mn, mi = st.mean(nat_all), st.mean(int_all)
    print(f"\n  GT native   : {mn:.4f}")
    print(f"  GT interp64 : {mi:.4f}")
    print(f"  delta       : {mn-mi:+.4f}   ({'resampling COSTS the ceiling' if mn-mi > 0.008 else 'resampling is ~harmless'})")
    json.dump({"native": mn, "interp64": mi, "delta": mn - mi,
               "n_seeds": args.n_seeds, "n_prompts": len(prompts)},
              open(os.path.join(args.output_dir, "result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
