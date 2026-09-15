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

# Standing diverse probe set (committed) — the DEFAULT eval prompts for the Z-Image adapter
# (outlandish / atmospheric / non-human / object, better than random COCO captions for
# tracking the collapse->breakout). Override with --prompts_file <other>, or pass
# --prompts_file "" to fall back to val dataset captions (--cache_dir).
_DEFAULT_PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zimage_probe_prompts.txt")


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
    p.add_argument("--prompts_file", type=str, default=_DEFAULT_PROMPTS_FILE,
                   help="Prompts (one per line) to eval, built as [prompt]+[BOI,IPH,EOI]+eos synthesis "
                        "inputs (requires --text_encoder_model). DEFAULT = the committed diverse probe "
                        "set (zimage_probe_prompts.txt). Pass \"\" to use val dataset captions instead.")
    p.add_argument("--skip_targets", action="store_true",
                   help="Never render/log target images (generated only). Targets are deterministic.")
    p.add_argument("--force_targets", action="store_true",
                   help="Always render targets, ignoring the 'already done' marker.")
    p.add_argument("--zimage_model", type=str, default="Tongyi-MAI/Z-Image-Turbo")
    # Offload is a DISCRETE-GPU optimisation: it keeps the ~20GB Z-Image stack on the host and
    # pages modules to the device on demand so the world model can coexist in limited VRAM. On
    # UNIFIED-MEMORY hardware (Ryzen AI, Apple silicon) host and device are the SAME physical
    # memory, so it buys nothing and costs transfer overhead. It also needs accelerate>=0.17,
    # which used to be declared only in the `training` group -- a demo-only install crashed here.
    p.add_argument("--zimage_offload", choices=["auto", "on", "off"], default="auto",
                   help="Z-Image CPU offload. 'auto' = offload if accelerate is present, else "
                        "load straight to the device. 'off' skips it -- the right choice on "
                        "unified-memory systems. 'on' requires accelerate and errors if missing.")
    p.add_argument("--gen_steps", type=int, default=8)          # Turbo
    p.add_argument("--guidance", type=float, default=0.0)       # Turbo: no CFG
    p.add_argument("--output_gain", type=float, default=1.0,
                   help="Z-Image adapter: inference-only dispersion gain on the whitened prediction. An MSE-trained point estimate is shrunk toward the target mean by 1-R^2, which the DiT renders as washed-out/generic; ~1/alpha undoes it. Measured best ~1.2-1.5 (gain 1.34: CLIPScore 0.287->0.303 vs 0.345 GT). Estimate per-checkpoint with scripts_local/zimage_shrinkage_probe.py. 1.0 = off.")
    p.add_argument("--flow_project", choices=["none", "proj", "renorm"], default="none",
                   help="Flow heads only: drop the component of the sample that the head could "
                        "never have written. out is Linear(flow_dim -> seq_dim) with "
                        "flow_dim << seq_dim, so velocities live in a flow_dim-dim subspace and "
                        "the initial noise ORTHOGONAL to it survives the ODE untouched "
                        "(measured 59%% of emitted energy at w=1, 43%% at w=3 on t3_2). "
                        "'proj' = hard projection; 'renorm' = project then rescale to unit "
                        "per-dim std (whitened targets are unit by construction, and the raw "
                        "projection lands under-dispersed). Inference-only, no retraining.")
    p.add_argument("--n_samples", type=int, default=1,
                   help="T3 only: draw N flow samples per prompt instead of 1. The head emits a "
                        "DISTRIBUTION, so a single draw conflates sampling variance with training "
                        "progress -- N>1 reports mean/std/best-of-N and renders all N side by side. "
                        "Noise is SEEDED from --flow_seed_base, so sample k uses the same noise at "
                        "every checkpoint and differences are attributable to the model.")
    p.add_argument("--tag_suffix", type=str, default="",
                   help="Appended to the TB tag root (e.g. '_w3' -> text_to_image_w3/...). Lets one "
                        "run log several eval variants (e.g. unguided vs guided) per checkpoint "
                        "without them overwriting each other's scalars and images.")
    p.add_argument("--trunk_gain", type=float, default=None,
                   help="Amplify the PROMPT-CONDITIONAL part of the trunk's image gen-query output "
                        "by this factor at inference. The gen-query representation is mu + r, where "
                        "mu is the across-prompt mean (learned queries + biases, norm ~83) and r is "
                        "the prompt-conditional residual (norm ~3.5, i.e. ~4%%). This feeds "
                        "mu + gain*r to the head. gain=1.0 is a no-op. Calibrates mu over the "
                        "prompt list in one extra pass. See scripts_local/trunk_compression_probe.py.")
    p.add_argument("--flow_steps", type=int, default=None,
                   help="Override the conditioning sampler's Euler step count (config default 8). "
                        "Applies to the T3/T5 parallel head and the T4 AR head alike. More steps "
                        "reduce ODE discretisation error, a variance source -- and variance, not "
                        "ceiling, is what separates these arms. Errors out rather than no-opping "
                        "if the checkpoint has no flow head.")
    p.add_argument("--flow_guidance", type=float, default=None,
                   help="T3 classifier-free guidance weight w. v = v_uncond + w*(v_cond - v_uncond). "
                        "1.0 = off. >1 trades diversity for fidelity (2x sampling cost). Only "
                        "meaningful for checkpoints TRAINED with flow_cfg_dropout > 0 -- otherwise "
                        "the null context is untrained and guidance extrapolates from noise.")
    p.add_argument("--flow_seed_base", type=int, default=4242,
                   help="Base seed for T3 flow sampling; sample k of prompt n uses base + 1000*k.")
    p.add_argument("--flow_bypass", action="store_true",
                   help="Bypass the T3 flow head and surface the auxiliary POINT head instead. Diagnostic: if point-head renders are healthy while sampled ones are not, the shared Q-Former/trunk is intact and only the flow head is undertrained/broken; if BOTH are bad, the fresh head's gradients damaged the warm-started trunk.")
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
    def _gen_head(m):
        """The generative head, whichever kind: T3 parallel `flow_head` or T4 `ar_flow_head`."""
        return (getattr(m.image_generator, "flow_head", None)
                or getattr(m.image_generator, "ar_flow_head", None))

    if args.flow_guidance is not None and _gen_head(model) is not None:
        _gen_head(model).guidance = float(args.flow_guidance)
        print(f"[flow] classifier-free guidance w={args.flow_guidance}", flush=True)
    if getattr(args, "flow_bypass", False) and _gen_head(model) is not None:
        model.image_generator.flow_head = None
        model.image_generator.ar_flow_head = None
        print("[t3] flow head BYPASSED -> surfacing the auxiliary point head", flush=True)
    if args.flow_project != "none":
        if _gen_head(model) is None:
            raise SystemExit("--flow_project needs a flow head (T3/T4/T5)")
        model.image_generator.flow_project = args.flow_project
        print(f"[flow] projecting samples onto the head's reachable subspace "
              f"(mode={args.flow_project})", flush=True)
    if args.output_gain != 1.0:
        model.image_generator.output_gain = args.output_gain
        print(f"[zimage] output_gain={args.output_gain} (dispersion correction on the "
              f"whitened prediction; whiten={model.image_generator.whiten})", flush=True)

    # ── val data (force image-synthesis direction) ──
    def resolve(d, s):
        if d is None:
            return None
        for c in (d + "_" + s, d):
            if os.path.isdir(c):
                return c
        return None
    # val-caption dataset is only needed when NOT probing custom prompts.
    dataset = None
    if not args.prompts_file:
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
    _has_accel = True
    try:
        import accelerate  # noqa: F401
    except ImportError:
        _has_accel = False
    _mode = getattr(args, "zimage_offload", "auto")
    if _mode == "on" and not _has_accel:
        raise SystemExit("--zimage_offload on requires accelerate>=0.17 "
                         "(install the `demo` or `image` extra), or pass --zimage_offload off")
    if _mode == "off" or not _has_accel:
        why = "--zimage_offload off" if _mode == "off" else "accelerate not installed"
        print(f"[zimage] no CPU offload ({why}); loading pipeline onto {device}", flush=True)
        pipe = pipe.to(device)
    else:
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

    # ── metrics init (once; shared by all splits) ──
    import statistics as st_, hashlib, json
    from megatransformer.scripts.eval.world.eval_utils import infer_step_from_checkpoint, init_eval_metrics, log_eval_scalars
    from megatransformer.utils import metrics as _m
    from megatransformer.utils import constants
    step = args.step if args.step is not None else infer_step_from_checkpoint(args.checkpoint_path)
    if args.log_dir:
        init_eval_metrics(args.log_dir, args.checkpoint_path)

    def _chw(pil):
        return np.asarray(pil).astype(np.float32).transpose(2, 0, 1) / 255.0

    # ── custom-prompt synthesis input: [SmolLM2(prompt)] + [BOI, IPH, EOI] + eos, dummy latent
    # trigger (verified against the dataset tokenization). Lets us probe arbitrary/diverse prompts. ──
    _stb = int(getattr(_mcfg, "special_token_base", 32000))
    _sptok = constants.special_token_ids(_stb)
    _eos = int(getattr(_mcfg, "eos_token_id", 2))
    _text_tok = None
    if args.prompts_file:
        if not args.text_encoder_model:
            raise SystemExit("--prompts_file requires --text_encoder_model (to tokenize prompts)")
        from transformers import AutoTokenizer
        _text_tok = AutoTokenizer.from_pretrained(args.text_encoder_model)

    def _iter_info(out):
        """(num_iterations, kl_per_iteration) from a forward — effective recurrent depth.
        num_iters = when ALL tokens converged (KL early-exit) or the cap; kl = mean-over-token
        KL curve. Eval forward is image-synthesis (short caption + ~64 gen queries dominate)."""
        ni = out.get("recurrent_num_iterations")
        kl = out.get("recurrent_kl_per_iteration") or []
        return (int(ni) if ni is not None else -1), [float(x) for x in kl]

    _adapter = model.image_generator

    if args.flow_steps is not None:
        # Euler steps for the conditioning sampler. Both heads read self.steps when sample() is
        # called with steps=None (cond_flow_head.py:322, ar_cond_flow_head.py:173), so setting the
        # attribute here IS the override -- there is no separate plumbing to thread.
        # Cost asymmetry worth knowing: the parallel head pays `steps` forwards total, the AR head
        # pays L x steps. Measured at steps=8 the AR arm's whole eval was only ~3% slower than the
        # parallel one, because Z-Image's 1024x1024 decode dominates -- so raising steps is far
        # cheaper here than the forward counts suggest.
        # NB: do NOT name these `_m` -- that is bound at the top of this function to the metrics
        # module (`from megatransformer.utils import metrics as _m`), and shadowing it here made
        # every --flow_steps eval die later at `_m.get_logger()`.
        _n_patched = 0
        for _head_name in ("flow_head", "ar_flow_head"):
            _head_mod = getattr(_adapter, _head_name, None)
            if _head_mod is not None and hasattr(_head_mod, "steps"):
                print(f"[flow_steps] {_head_name}.steps {_head_mod.steps} -> {args.flow_steps}")
                _head_mod.steps = int(args.flow_steps)
                _n_patched += 1
        if _n_patched == 0:
            raise SystemExit("--flow_steps was passed but this checkpoint has no flow head with a "
                             "`steps` attribute; the flag would have silently done nothing.")

    def _set_flow_seed(seed):
        """Pin the flow head's sampling noise so the SAME draw is compared across checkpoints."""
        if seed is not None and (getattr(_adapter, "flow_head", None) is not None
                                 or getattr(_adapter, "ar_flow_head", None) is not None):
            g = torch.Generator(device=device)
            g.manual_seed(int(seed))
            _adapter.flow_generator = g

    def _ar_length(prompt):
        """Native-length heads (T4 AR and T5 parallel): emit exactly as many conditioning tokens
        as Qwen3 would produce for this caption. Deterministic from the caption, so no stop
        token / length head is needed. Fixed-K heads (T3) return None and keep seq_len."""
        if (getattr(_adapter, "ar_flow_head", None) is None
                and not bool(getattr(_adapter, "flow_native_length", False))):
            return None
        return len(pipe.tokenizer(pipe.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True)).input_ids)

    @torch.no_grad()
    def seq_pred_for_prompt(prompt, seed=None):
        _set_flow_seed(seed)
        if args.trunk_gain is not None:
            # the thought state inits from noise (like-init, std 0.02). Pin it so that noise is
            # IDENTICAL across prompts and therefore lands in mu instead of inflating r.
            torch.manual_seed(12345)
        ids = _text_tok(prompt, add_special_tokens=False).input_ids
        seq = ids + [_sptok.BOI, _sptok.IMAGE_PLACEHOLDER, _sptok.EOI, _eos]
        text_input_ids = torch.tensor([seq], dtype=torch.long, device=device)
        image_inputs = torch.zeros(1, 1, 12, 32, 32, device=device)   # dummy synthesis trigger
        is_synth = torch.tensor([True], device=device)
        with torch.amp.autocast(device, dtype=dtype, enabled=args.bf16):
            out = model(text_input_ids=text_input_ids, image_inputs=image_inputs,
                        precomputed_latents=True, is_synthesis=is_synth, decode_outputs=False,
                        cond_length=_ar_length(prompt))
        sp = out.get("image_clip_seq_pred")
        if sp is None:
            return None, None, None
        ni, kl = _iter_info(out)
        return sp[0].float(), ni, kl

    @torch.no_grad()
    def items_from_dataset(ds, max_samples):
        """-> list of (caption, seq_pred, num_iters, kl_curve) for up to max_samples samples."""
        items, n = [], 0
        for i in range(len(ds)):
            batch = collator([ds[i]])
            if "image_images" not in batch or "is_synthesis" not in batch:
                continue
            caption = (batch.get("text_texts") or [""])[0] or ""
            with torch.amp.autocast(device, dtype=dtype, enabled=args.bf16):
                out = model(text_input_ids=batch["text_token_ids"].to(device),
                            image_inputs=batch["image_images"].unsqueeze(1).to(device),
                            precomputed_latents=True, is_synthesis=batch["is_synthesis"].to(device),
                            decode_outputs=False)
            sp = out.get("image_clip_seq_pred")
            if sp is None:
                continue
            ni, kl = _iter_info(out)
            items.append((caption, sp[0].float(), ni, kl))
            n += 1
            if n >= max_samples:
                break
        return items

    def items_from_prompts(prompts):
        # Seeded even at n_samples=1: an unseeded draw makes every checkpoint use different
        # noise, so single-draw trends mix training progress with sampling variance.
        out = []
        for p in prompts:
            sp, ni, kl = seq_pred_for_prompt(p, seed=args.flow_seed_base)
            if sp is not None:
                out.append((p, sp, ni, kl))
        return out

    # ── targets are deterministic (caption -> fixed render); render ONCE per (log_dir, prompt
    # set). A marker records the set-key so later checkpoints skip target render+log. ──
    _marker = os.path.join(args.log_dir, ".zimage_eval_targets.json") if args.log_dir else None

    def _skip_targets(key):
        if args.force_targets:
            return False
        if args.skip_targets:
            return True
        if _marker and os.path.exists(_marker):
            try:
                return json.load(open(_marker)).get("key") == key
            except Exception:
                return False
        return False

    def render_and_log_multi(prompts, tag, montage_name, n_samples):
        """T3 multi-sample mode: N seeded draws per prompt -> mean / std / best-of-N.

        Judging a stochastic head by ONE draw per checkpoint conflates sampling variance with
        training progress (a detail present at 19k and absent at 20k may be one model sampling
        twice, not two models). Seeded noise makes draw k identical across checkpoints, so the
        spread WITHIN a checkpoint measures the conditional's entropy and the change in the mean
        ACROSS checkpoints measures learning.
        """
        rows, per_prompt, sc_tgt, iters_list = [], [], [], []
        key = tag + ":" + hashlib.md5("|".join(prompts).encode()).hexdigest()[:8]
        skip_t = _skip_targets(key)
        for n, prompt in enumerate(prompts):
            imgs, scores = [], []
            for k in range(n_samples):
                sp, ni, _ = seq_pred_for_prompt(prompt, seed=args.flow_seed_base + 1000 * k)
                if sp is None:
                    continue
                if k == 0:
                    iters_list.append(ni)
                im = zimage_render(sp, 1000 + n)
                imgs.append(im); scores.append(clipscore(im, prompt))
            if not scores:
                continue
            row = list(imgs)
            if not skip_t:
                tgt = zimage_render(target_cond(prompt).float(), 1000 + n)
                sc_tgt.append(clipscore(tgt, prompt))
                row = [tgt] + row
            rows.append(row)
            mu = st_.mean(scores)
            sd = st_.pstdev(scores) if len(scores) > 1 else 0.0
            per_prompt.append((prompt, mu, sd, max(scores), min(scores)))
            print(f"[{tag} {n}] mean={mu:.3f} sd={sd:.3f} best={max(scores):.3f} "
                  f"worst={min(scores):.3f} | {prompt[:44]}", flush=True)
        if not rows:
            print(f"[{tag}] no samples produced conditioning; skipping")
            return None
        thumb, ncol = 288, max(len(r) for r in rows)
        canvas = Image.new("RGB", (ncol * thumb, len(rows) * thumb), (20, 20, 20))
        for r, row in enumerate(rows):
            for c, im in enumerate(row):
                canvas.paste(im.resize((thumb, thumb)), (c * thumb, r * thumb))
        montage_path = os.path.join(args.output_dir, montage_name)
        canvas.save(montage_path)
        mean_mu = st_.mean([p[1] for p in per_prompt])
        mean_sd = st_.mean([p[2] for p in per_prompt])
        mean_best = st_.mean([p[3] for p in per_prompt])
        tail = f"  target={st_.mean(sc_tgt):.3f}" if sc_tgt else "  (targets skipped)"
        print(f"[{tag}] N={n_samples} mean CLIPScore generated={mean_mu:.3f} "
              f"(within-prompt sd={mean_sd:.3f})  best-of-N={mean_best:.3f}{tail}"
              f"  montage: {montage_path}", flush=True)
        if args.log_dir:
            scal = {f"{tag}/clipscore_generated": mean_mu,
                    f"{tag}/clipscore_sample_sd": mean_sd,
                    f"{tag}/clipscore_best_of_n": mean_best}
            if sc_tgt:
                scal[f"{tag}/clipscore_target"] = st_.mean(sc_tgt)
            for r, (_, mu, sd, bst, wst) in enumerate(per_prompt):
                scal[f"{tag}/sample_sd/{r}"] = sd
                scal[f"{tag}/best_of_n/{r}"] = bst
            for r, ni in enumerate(iters_list):
                scal[f"{tag}/recurrent_iters/{r}"] = float(ni)
            log_eval_scalars(scal, step)
            # Log EVERY draw as its own image tag, so the step slider shows the sample
            # spread per prompt (sample0/sample1/...) instead of a single unlabelled draw.
            # Without this the multi-sample path wrote scalars only and the renders lived
            # nowhere but the on-disk montage.
            logger = _m.get_logger()
            if logger is not None:
                off = 0 if skip_t else 1
                for r, row in enumerate(rows):
                    cap = per_prompt[r][0][:500]
                    for k in range(len(row) - off):
                        _m.log_image(f"{tag}/image/{r}/sample{k}", _chw(row[off + k]), step,
                                     context={"prompt": cap})
                    if not skip_t:
                        _m.log_image(f"{tag}/image/{r}/target", _chw(row[0]), step,
                                     context={"prompt": cap})
                _m.flush()
        return mean_mu

    def render_and_log(items, tag, montage_name):
        if not items:
            print(f"[{tag}] no samples produced conditioning; skipping")
            return None
        key = tag + ":" + hashlib.md5("|".join(it[0] for it in items).encode()).hexdigest()[:8]
        skip_t = _skip_targets(key)
        grid, caps, sc_gen, sc_tgt, iters_list = [], [], [], [], []
        for n, (caption, seq_pred, num_iters, kl_curve) in enumerate(items):
            gen = zimage_render(seq_pred, 1000 + n)
            sg = clipscore(gen, caption); sc_gen.append(sg); iters_list.append(num_iters)
            # KL "elbow": first iteration whose mean-token KL falls below 10% of the initial
            # = the EFFECTIVE refinement depth (where the thought vector stops changing).
            elbow = num_iters
            if kl_curve and kl_curve[0]:
                thr = 0.1 * abs(kl_curve[0])
                elbow = next((j + 1 for j, k in enumerate(kl_curve) if abs(k) <= thr), num_iters)
            klr = (f" iters={num_iters} elbow~{elbow} kl0={kl_curve[0]:.2g} klN={kl_curve[-1]:.2g}"
                   if kl_curve else f" iters={num_iters}")
            if skip_t:
                grid.append([gen])
                print(f"[{tag} {n}] gen={sg:.3f}{klr} | {caption[:42]}", flush=True)
            else:
                tgt = zimage_render(target_cond(caption).float(), 1000 + n)
                stv = clipscore(tgt, caption); sc_tgt.append(stv); grid.append([tgt, gen])
                print(f"[{tag} {n}] tgt={stv:.3f} gen={sg:.3f}{klr} | {caption[:42]}", flush=True)
            caps.append(caption)

        thumb, ncol = 320, (1 if skip_t else 2)
        canvas = Image.new("RGB", (ncol * thumb, len(grid) * thumb), (20, 20, 20))
        for r, row in enumerate(grid):
            for c, im in enumerate(row):
                canvas.paste(im.resize((thumb, thumb)), (c * thumb, r * thumb))
        montage_path = os.path.join(args.output_dir, montage_name)
        canvas.save(montage_path)
        mg = st_.mean(sc_gen)
        valid_it = [x for x in iters_list if x >= 0]
        miters = st_.mean(valid_it) if valid_it else -1
        tail = f"  target={st_.mean(sc_tgt):.3f}" if sc_tgt else "  (targets skipped)"
        print(f"[{tag}] mean CLIPScore generated={mg:.3f}{tail}  mean recurrent iters={miters:.1f}"
              f"  montage: {montage_path}", flush=True)

        if args.log_dir:
            scal = {f"{tag}/clipscore_generated": mg, f"{tag}/recurrent_iters_mean": float(miters)}
            if sc_tgt:
                scal[f"{tag}/clipscore_target"] = st_.mean(sc_tgt)
            for r, ni in enumerate(iters_list):
                scal[f"{tag}/recurrent_iters/{r}"] = float(ni)
            log_eval_scalars(scal, step)
            logger = _m.get_logger()
            if logger is not None:
                for r, row in enumerate(grid):
                    cap = caps[r][:500]
                    _m.log_image(f"{tag}/image/{r}/generated", _chw(row[-1]), step, context={"prompt": cap})
                    if not skip_t:
                        _m.log_image(f"{tag}/image/{r}/target", _chw(row[0]), step, context={"prompt": cap})
                _m.flush()
        if not skip_t and _marker:
            try:
                json.dump({"key": key}, open(_marker, "w"))
            except Exception:
                pass
        return mg

    # ── val / custom prompts (always) ──
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [ln.strip() for ln in f if ln.strip()]
        print(f"custom prompts: {len(prompts)} from {args.prompts_file}", flush=True)
    else:
        prompts = None

    # ── trunk conditional-gain: mu + gain*r at the image gen-query positions ──
    if args.trunk_gain is not None:
        n_gen = int(getattr(model, "_n_image_gen_positions", 0) or 0)
        if n_gen <= 0:
            raise SystemExit("--trunk_gain needs image gen queries (n_image_gen_positions)")
        if prompts is None:
            prompts = [c for c, _, _, _ in items_from_dataset(dataset, args.max_samples)]

        _cap = {"on": False, "acc": [], "mu": None}

        def _trunk_hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            if _cap["on"]:
                _cap["acc"].append(t[:, -n_gen:, :].detach().float().cpu())
                return out
            if _cap["mu"] is None:
                return out
            mu = _cap["mu"].to(t.device, t.dtype)
            tail = t[:, -n_gen:, :]
            t = torch.cat([t[:, :-n_gen, :], mu + args.trunk_gain * (tail - mu)], dim=1)
            return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t

        _h = model.recurrent_block.register_forward_hook(_trunk_hook)
        _cap["on"] = True
        for _p in prompts:
            seq_pred_for_prompt(_p, seed=0)
        _cap["on"] = False
        if not _cap["acc"]:
            raise SystemExit("--trunk_gain calibration captured nothing from recurrent_block")
        _cap["mu"] = torch.cat(_cap["acc"]).mean(0, keepdim=True)
        _r = torch.cat(_cap["acc"])
        _rel = float((_r - _cap["mu"]).norm(dim=-1).mean() / _cap["mu"].norm(dim=-1).mean())
        print(f"[trunk_gain] mu over {len(prompts)} prompts; conditional fraction "
              f"||r||/||mu||={_rel:.4f}; applying gain={args.trunk_gain}", flush=True)
        _cap["acc"] = []

    if args.n_samples > 1:
        if _gen_head(model) is None:
            raise SystemExit("--n_samples > 1 needs a generative head (flow_head or ar_flow_head); "
                             "the point head is deterministic so extra draws would be identical")
        if prompts is None:
            prompts = [c for c, _, _, _ in items_from_dataset(dataset, args.max_samples)]
        if render_and_log_multi(prompts, "text_to_image" + args.tag_suffix,
                                "montage_samples.png", args.n_samples) is None:
            raise SystemExit("no image-synthesis samples produced conditioning; check dataset/config/prompts")
    else:
        val_items = items_from_prompts(prompts) if prompts is not None \
            else items_from_dataset(dataset, args.max_samples)
        if render_and_log(val_items, "text_to_image" + args.tag_suffix,
                          "montage_target_vs_generated.png") is None:
            raise SystemExit("no image-synthesis samples produced conditioning; check the dataset/config/prompts")

    # ── train (optional): the ACTUAL training captions + renders under train_data/ ──
    if args.train_cache_dir:
        tr_img = resolve(args.train_cache_dir, "train") if "image" in include_modes else None
        tr_txt = resolve(args.train_cache_dir, "train") if "text" in include_modes else None
        train_ds = MultimodalShardedDataset(text_shard_dir=tr_txt, image_shard_dir=tr_img,
                                             cache_size=8, max_samples=args.train_max_samples)
        render_and_log(items_from_dataset(train_ds, args.train_max_samples),
                       "train_data/text_to_image", "montage_train_target_vs_generated.png")


if __name__ == "__main__":
    main()
