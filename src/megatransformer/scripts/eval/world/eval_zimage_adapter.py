"""Eval + TB viz for the Z-Image-adapter image path (generated vs target via frozen Z-Image).

Standalone (NOT an in-training callback): the full ~20GB Z-Image pipeline alongside a
training run OOMs a 4090, so image-generation eval for the adapter path runs separately.
For each val image-synthesis sample:

  - forward the world model -> the adapter's predicted Qwen3 conditioning
    (image_clip_seq_pred, shape (seq_len, 2560))
  - "generated" = frozen Z-Image(predicted conditioning)
  - "target"    = frozen Z-Image(Qwen3(caption) resampled to seq_len)   [the adapter's ceiling]
  - CLIPScore(generated, caption) vs CLIPScore(target, caption)
  - log both panels to TB + save a montage

The target uses the SAME resample-to-seq_len the trainer regresses toward (utils/
zimage_text_encoder.py), so "target" is the honest ceiling for the fixed-K adapter.

Requires the world model's image_coda_config to be a ZImageAdapterConfig.

Usage:
    CUDA_VISIBLE_DEVICES=N python -m megatransformer.scripts.eval.world.eval_zimage_adapter \
        --checkpoint_path runs/world/<run>/checkpoint-N --config small_sum_zimage \
        --text_encoder_model HuggingFaceTB/SmolLM2-135M \
        --cache_dir ../cached_datasets/Mekadrom/image_gen_captions_only/val \
        --include_modes text,image --max_samples 8 --bf16 --log_dir runs/world/<run>
"""

import argparse
import os

import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--config", type=str, default="small_sum_zimage")
    p.add_argument("--include_modes", type=str, default="text,image")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=8)
    p.add_argument("--train_cache_dir", type=str, default=None,
                   help="ALSO eval on TRAIN samples, logged under train_data/ (the training viz's "
                        "train_data path is latent-based and empty for the adapter). Lets you inspect "
                        "the ACTUAL training captions + their target/generated renders for data issues.")
    p.add_argument("--train_max_samples", type=int, default=8)
    p.add_argument("--zimage_model", type=str, default="Tongyi-MAI/Z-Image-Turbo")
    p.add_argument("--gen_steps", type=int, default=8)          # Turbo
    p.add_argument("--guidance", type=float, default=0.0)       # Turbo: no CFG
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--text_encoder_model", type=str, default=None,
                   help="Pretrained text spine (e.g. HuggingFaceTB/SmolLM2-135M) — MUST match how "
                        "the checkpoint was trained, or the prelude weights won't load.")
    p.add_argument("--tie_word_embeddings", action="store_true")
    p.add_argument("--output_dir", type=str, default="eval_output/zimage_adapter_eval")
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    import numpy as np
    from PIL import Image
    from megatransformer.model.world.world_model import MegaTransformerWorldModel, ZImageConditioningAdapter
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
    if not isinstance(getattr(model, "image_generator", None), ZImageConditioningAdapter):
        raise SystemExit(f"--config {args.config} does not use the Z-Image adapter "
                         f"(image_generator is {type(getattr(model,'image_generator',None)).__name__})")
    seq_len = int(model.image_generator.config.seq_len)

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
    _mcfg = model.module.config if hasattr(model, "module") else model.config
    collator = MultimodalDataCollator(
        special_token_base=getattr(_mcfg, "special_token_base", 32000),
        eos_token_id=getattr(_mcfg, "eos_token_id", 2),
    )
    collator.force_direction = "synthesis"

    # ── Z-Image ── (offload so the ~20GB stack fits with the world model + CLIP scorer)
    from diffusers import ZImagePipeline
    pipe = ZImagePipeline.from_pretrained(args.zimage_model, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def target_cond(caption):
        """Qwen3(caption) penultimate states, masked, resampled to seq_len — the trainer target."""
        s = pipe.tokenizer.apply_chat_template(
            [{"role": "user", "content": caption}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
        ti = pipe.tokenizer(s, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        ids = ti.input_ids.to(pipe.text_encoder.device)
        attn = ti.attention_mask.to(pipe.text_encoder.device)
        hs = pipe.text_encoder(input_ids=ids, attention_mask=attn,
                               output_hidden_states=True).hidden_states[-2][0]
        real = hs[attn.bool()[0]]
        if real.shape[0] == 0:
            real = hs[:1]
        x = real.transpose(0, 1).unsqueeze(0).float()
        r = F.interpolate(x, size=seq_len, mode="linear", align_corners=False)
        return r[0].transpose(0, 1).to(hs.dtype)          # (seq_len, 2560)

    @torch.no_grad()
    def zimage_render(seq, seed):
        g = torch.Generator(device=device).manual_seed(seed)
        return pipe(prompt_embeds=[seq], num_inference_steps=args.gen_steps,
                    guidance_scale=args.guidance, height=1024, width=1024, generator=g).images[0]

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

    # ── metrics init (once; shared by val + train splits) ──
    import statistics as st_
    from megatransformer.scripts.eval.world.eval_utils import infer_step_from_checkpoint, init_eval_metrics, log_eval_scalars
    from megatransformer.utils import metrics as _m
    step = args.step if args.step is not None else infer_step_from_checkpoint(args.checkpoint_path)
    if args.log_dir:
        init_eval_metrics(args.log_dir, args.checkpoint_path)

    def _chw(pil):
        return np.asarray(pil).astype(np.float32).transpose(2, 0, 1) / 255.0

    def eval_split(ds, tag, montage_name, max_samples):
        """Render generated-vs-target for up to max_samples of `ds`; save a montage + log
        images/captions/CLIPScore under `tag` (e.g. 'text_to_image' for val,
        'train_data/text_to_image' for train). Prints per-sample + mean; returns the means."""
        grid, caps, sc_gen, sc_tgt = [], [], [], []
        n = 0
        for i in range(len(ds)):
            batch = collator([ds[i]])
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
            if seq_pred is None:
                continue
            gen = zimage_render(seq_pred[0].float(), 1000 + n)
            tgt = zimage_render(target_cond(caption).float(), 1000 + n)
            sg, st = clipscore(gen, caption), clipscore(tgt, caption)
            sc_gen.append(sg); sc_tgt.append(st)
            grid.append([tgt, gen]); caps.append(caption)
            print(f"[{tag} {n}] CLIP target={st:.3f} generated={sg:.3f} | {caption[:50]}", flush=True)
            n += 1
            if n >= max_samples:
                break
        if not grid:
            print(f"[{tag}] no image-synthesis samples produced conditioning; skipping")
            return None

        thumb = 320
        canvas = Image.new("RGB", (2 * thumb, len(grid) * thumb), (20, 20, 20))
        for r, (tgt, gen) in enumerate(grid):
            canvas.paste(tgt.resize((thumb, thumb)), (0, r * thumb))
            canvas.paste(gen.resize((thumb, thumb)), (thumb, r * thumb))
        montage_path = os.path.join(args.output_dir, montage_name)
        canvas.save(montage_path)
        mg, mt = st_.mean(sc_gen), st_.mean(sc_tgt)
        print(f"[{tag}] mean CLIPScore: target={mt:.3f}  generated={mg:.3f}  montage: {montage_path}", flush=True)

        if args.log_dir:
            log_eval_scalars({f"{tag}/clipscore_generated": mg, f"{tag}/clipscore_target": mt}, step)
            logger = _m.get_logger()
            if logger is not None:
                for r, (tgt, gen) in enumerate(grid):
                    cap = caps[r][:500] if r < len(caps) else ""
                    # context={"prompt": cap} logs the caption as a sibling tag -> the actual
                    # training caption is visible alongside each render (data diagnosis).
                    _m.log_image(f"{tag}/image/{r}/generated", _chw(gen), step, context={"prompt": cap})
                    _m.log_image(f"{tag}/image/{r}/target", _chw(tgt), step, context={"prompt": cap})
                _m.flush()
        return mg, mt

    # ── val (always) ──
    if eval_split(dataset, "text_to_image", "montage_target_vs_generated.png", args.max_samples) is None:
        raise SystemExit("no val image-synthesis samples produced conditioning; check the dataset/config")

    # ── train (optional): inspect the ACTUAL training captions + renders under train_data/ ──
    if args.train_cache_dir:
        tr_img = resolve(args.train_cache_dir, "train") if "image" in include_modes else None
        tr_txt = resolve(args.train_cache_dir, "train") if "text" in include_modes else None
        train_ds = MultimodalShardedDataset(text_shard_dir=tr_txt, image_shard_dir=tr_img,
                                             cache_size=8, max_samples=args.train_max_samples)
        eval_split(train_ds, "train_data/text_to_image", "montage_train_target_vs_generated.png",
                   args.train_max_samples)


if __name__ == "__main__":
    main()
