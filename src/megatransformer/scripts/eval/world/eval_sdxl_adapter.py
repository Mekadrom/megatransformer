"""Eval + TB viz for the SDXL-adapter image path (generated vs target via frozen SDXL).

Standalone (NOT an in-training callback): loading the full ~7GB SDXL pipeline
alongside a training run OOMs a 4090, so image-generation eval for the adapter path
runs separately. For each val image-synthesis sample:

  - forward the world model -> the adapter's predicted CLIP conditioning
    (image_clip_seq_pred 77x2048 + image_clip_pooled_pred 1280)
  - "generated" = frozen SDXL(predicted conditioning)
  - "target"    = frozen SDXL(true CLIP(caption))   [what perfect conditioning yields]
  - CLIPScore(generated, caption) vs CLIPScore(target, caption)
  - log both panels to TB + save a montage

Requires the world model's image_coda_config to be an SDXLAdapterConfig.

Usage:
    CUDA_VISIBLE_DEVICES=N python -m megatransformer.scripts.eval.world.eval_sdxl_adapter \
        --checkpoint_path runs/world/<run>/checkpoint-N --config small_sum_sdxl \
        --cache_dir ../cached_datasets/Mekadrom/sdxl_train/val --include_modes text,image \
        --max_samples 8 --bf16 --log_dir runs/world/<run>
"""

import argparse
import os

import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--config", type=str, default="small_sum_sdxl")
    p.add_argument("--include_modes", type=str, default="text,image")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=8)
    p.add_argument("--sdxl_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--gen_steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.0)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--text_encoder_model", type=str, default=None,
                   help="Pretrained text spine (e.g. HuggingFaceTB/SmolLM2-135M) — MUST match how "
                        "the checkpoint was trained, or the prelude weights won't load.")
    p.add_argument("--tie_word_embeddings", action="store_true")
    p.add_argument("--output_dir", type=str, default="eval_output/sdxl_adapter_eval")
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    import numpy as np
    from PIL import Image
    from megatransformer.model.world.world_model import MegaTransformerWorldModel, SDXLConditioningAdapter
    from megatransformer.utils import model_loading_utils
    from megatransformer.scripts.data.world.dataset import MultimodalShardedDataset
    from megatransformer.scripts.data.world.data_collator import MultimodalDataCollator

    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)
    include_modes = [m.strip() for m in args.include_modes.split(",")]

    # ── world model ──
    overrides = {"include_modes": include_modes}
    if args.tie_word_embeddings:
        overrides["tie_word_embeddings"] = True
    if args.text_encoder_model:
        # Mirror training: pretrained-LLM spine -> special_token_base=native vocab, native eos,
        # + the text_encoder dict. from_config reconstructs the dataclass (re-running __post_init__),
        # so the interleaver placeholder ids are re-derived for the new base.
        from transformers import AutoConfig, AutoTokenizer
        from megatransformer.utils import constants
        _llm = AutoConfig.from_pretrained(args.text_encoder_model)
        _eos = _llm.eos_token_id
        if _eos is None:
            _eos = AutoTokenizer.from_pretrained(args.text_encoder_model).eos_token_id
        overrides["special_token_base"] = int(_llm.vocab_size)
        overrides["eos_token_id"] = int(_eos)
        overrides["text_encoder"] = {
            "model": args.text_encoder_model,
            "freeze": True,
            "translator_hidden_mult": 2.0,
            "n_special_tokens": constants.N_SPECIAL_TOKENS,
        }
    model = model_loading_utils.load_model(
        MegaTransformerWorldModel, args.config,
        checkpoint_path=args.checkpoint_path, overrides=overrides, device=device)
    model.to(device).eval()
    if not isinstance(getattr(model, "image_generator", None), SDXLConditioningAdapter):
        raise SystemExit(f"--config {args.config} does not use the SDXL adapter "
                         f"(image_generator is {type(getattr(model,'image_generator',None)).__name__})")

    # ── val data (force image-synthesis direction) ──
    def resolve(d, s):
        if d is None:
            return None
        for c in (d + "_" + s, d):
            if os.path.isdir(c):
                return c
        return None
    image_dir = resolve(args.cache_dir, "val") if "image" in include_modes else None
    text_dir = resolve(args.cache_dir, "val") if "text" in include_modes else None
    dataset = MultimodalShardedDataset(text_shard_dir=text_dir, image_shard_dir=image_dir,
                                       cache_size=8, max_samples=args.max_samples)
    # special_token_base/eos MUST match the model (49152 for SmolLM2) so the injected
    # IMAGE_PLACEHOLDER id matches what the interleaver scans for.
    _mcfg = model.module.config if hasattr(model, "module") else model.config
    collator = MultimodalDataCollator(
        special_token_base=getattr(_mcfg, "special_token_base", 32000),
        eos_token_id=getattr(_mcfg, "eos_token_id", 2),
    )
    collator.force_direction = "synthesis"

    # ── SDXL ── (fp16-safe VAE: SDXL's stock VAE can decode to black/NaN in fp16)
    from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DPMSolverMultistepScheduler
    _vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.sdxl_model, vae=_vae, torch_dtype=torch.float16, use_safetensors=True).to(device)
    # DPM++ 2M Karras: consistently crisper than the stock Euler at equal steps
    # (validated GT A/B: +0.007 CLIPScore, sharper edges) — the project default.
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    pipe.set_progress_bar_config(disable=True)
    neg_pe, _, neg_pp, _ = pipe.encode_prompt(prompt="", device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)

    @torch.no_grad()
    def sdxl_render(seq, pool, seed):
        g = torch.Generator(device=device).manual_seed(seed)
        return pipe(prompt_embeds=seq.half(), pooled_prompt_embeds=pool.half(),
                    negative_prompt_embeds=neg_pe.half(), negative_pooled_prompt_embeds=neg_pp.half(),
                    num_inference_steps=args.gen_steps, guidance_scale=args.guidance,
                    height=1024, width=1024, generator=g).images[0]

    # CLIPScore
    import open_clip
    from torchvision import transforms
    cm, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
    cm.eval(); ctok = open_clip.get_tokenizer("ViT-B-32")
    cnorm = transforms.Compose([transforms.Resize((224, 224), antialias=True),
                                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                     (0.26862954, 0.26130258, 0.27577711))])

    @torch.no_grad()
    def clipscore(img, txt):
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
        im = F.normalize(cm.encode_image(cnorm(x)), dim=-1)
        tx = F.normalize(cm.encode_text(ctok([txt[:77]]).to(device)), dim=-1)
        return float((im @ tx.T).item())

    # ── loop ──
    grid, caps, sc_gen, sc_tgt = [], [], [], []
    n = 0
    for i in range(len(dataset)):
        batch = collator([dataset[i]])
        if "image_images" not in batch or "is_synthesis" not in batch:
            continue
        caption = (batch.get("text_texts") or [""])[0] or ""
        text_input_ids = batch["text_token_ids"].to(device)
        image_inputs = batch["image_images"].unsqueeze(1).to(device)
        is_synth = batch["is_synthesis"].to(device)
        with torch.no_grad():
            with torch.amp.autocast(device, dtype=dtype, enabled=args.bf16):
                out = model(text_input_ids=text_input_ids, image_inputs=image_inputs,
                            precomputed_latents=True, is_synthesis=is_synth, decode_outputs=False)
        seq_pred = out.get("image_clip_seq_pred")
        pool_pred = out.get("image_clip_pooled_pred")
        if seq_pred is None:
            continue
        seq_true, _, pool_true, _ = pipe.encode_prompt(prompt=caption, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)
        gen = sdxl_render(seq_pred[:1].float(), pool_pred[:1].float(), 1000 + n)
        tgt = sdxl_render(seq_true.float(), pool_true.float(), 1000 + n)
        sg, st = clipscore(gen, caption), clipscore(tgt, caption)
        sc_gen.append(sg); sc_tgt.append(st)
        grid.append([tgt, gen]); caps.append(caption)
        print(f"[{n}] CLIP target={st:.3f} generated={sg:.3f} | {caption[:50]}", flush=True)
        n += 1
        if n >= args.max_samples:
            break

    if not grid:
        raise SystemExit("no image-synthesis samples produced conditioning; check the dataset/config")

    thumb = 320
    canvas = Image.new("RGB", (2 * thumb, len(grid) * thumb), (20, 20, 20))
    for r, (tgt, gen) in enumerate(grid):
        canvas.paste(tgt.resize((thumb, thumb)), (0, r * thumb))
        canvas.paste(gen.resize((thumb, thumb)), (thumb, r * thumb))
    montage_path = os.path.join(args.output_dir, "montage_target_vs_generated.png")
    canvas.save(montage_path)

    import statistics as st_
    print(f"\nmean CLIPScore: target={st_.mean(sc_tgt):.3f}  generated={st_.mean(sc_gen):.3f}")
    print(f"montage: {montage_path}  (left=target SDXL(true caption), right=generated SDXL(adapter pred))")

    # TB
    if args.log_dir:
        from megatransformer.scripts.eval.world.eval_utils import infer_step_from_checkpoint, init_eval_metrics, log_eval_scalars
        from megatransformer.utils import metrics as _m
        step = args.step if args.step is not None else infer_step_from_checkpoint(args.checkpoint_path)
        init_eval_metrics(args.log_dir, args.checkpoint_path)
        # Log under text_to_image/* to match the in-loop viz scenario's tag convention,
        # so the standalone eval's panels land in the same TB group.
        log_eval_scalars({"text_to_image/clipscore_generated": st_.mean(sc_gen),
                          "text_to_image/clipscore_target": st_.mean(sc_tgt)}, step)
        logger = _m.get_logger()
        if logger is not None:
            def _chw(pil):  # HWC uint8 -> CHW float [0,1] (SummaryWriter.add_image expects CHW)
                return np.asarray(pil).astype(np.float32).transpose(2, 0, 1) / 255.0
            for r, (tgt, gen) in enumerate(grid):
                cap = caps[r][:500] if r < len(caps) else ""
                _m.log_image(f"text_to_image/image/{r}/generated", _chw(gen), step,
                             context={"prompt": cap})
                _m.log_image(f"text_to_image/image/{r}/target", _chw(tgt), step,
                             context={"prompt": cap})
            _m.flush()


if __name__ == "__main__":
    main()
