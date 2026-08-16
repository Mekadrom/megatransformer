# Cleanup & Tech-Debt Backlog

Engineering cleanups, deferred wiring, and missing tests — distinct from
`RESEARCH_TODO.md` (the research / ablation roadmap). Living doc; any session
(world-image, world-tts, …) may add or check off items. Each entry notes *what*,
*why*, and the *files/how* so it's actionable later.

## Image / SDXL-adapter thread

- [ ] **Retire the dummy-latent mechanism** (make the model image-input-optional for
  synthesis). The image-input requirement during synthesis is vestigial — it only
  supplies a trigger + batch/n_images dims; the gen queries are learned params sized
  by config, and the interleaver already locates image slots from `IMAGE_PLACEHOLDER`
  tokens. Change `world_model.forward` so `image_inputs=None` + `IMAGE_PH` tokens ⇒
  build gen queries (batch from `text_input_ids`, count from `image_num_patches`),
  transcription still requires a real image. Deletes `dummy_latent_shape` end-to-end:
  no dummy in preprocess (`image/vae/preprocess.py`), no zero-synthesis in
  `world/dataset.py::_get_image_sample`, no `image_images` in the collator for
  generation. *Deferred — the dummy hack works & is tested; do deliberately + smoke-test.*
  Touches: `world_model.forward` image branch, `data_collator.py`, `dataset.py`, preprocess.

- [x] **End-to-end smoke test of the SDXL-adapter training path.** DONE — `world_image_sdxl_smoke_0`
  (300 steps) ran clean: `small_sum_sdxl` + `--text_encoder_model SmolLM2` compose, `image_clip_loss`
  flows, no OOM. Surfaced + fixed the DiT-assuming in-loop viz (commit a8b485a).

- [x] **Validate `eval_sdxl_adapter.py` against a real trained checkpoint.** DONE — ran vs
  checkpoint-300 (commit 5d42fd5 added `--text_encoder_model` + collator special_token_base).
  End-to-end pass: target CLIP 0.417, generated 0.048 (near-random at 300 steps, expected).

- [x] **Add `diffusers` (+ `bitsandbytes`) to the uv `training` dependency group.** DONE —
  `diffusers>=0.39` and `bitsandbytes` (4-bit Qwen3-4B target encoder) added to the training
  group in `pyproject.toml`. `accelerate` was already present. NOTE: not yet `uv sync`'d /
  `uv.lock`+`requirements.txt` not regenerated (avoided disturbing a live training env) —
  do `uv sync` before the first Z-Image run.

- [x] **Update `multimodal_chat.py` (Gradio) for the SDXL-adapter image path.** DONE.
  (a) `--text_encoder_model` added — builds the same overrides as `eval_sdxl_adapter.py`
  (special_token_base=native vocab, native eos, text_encoder dict) and loads the SmolLM2
  tokenizer; (b) control-token ids re-derived from the loaded model's `special_token_base`
  via `constants.special_token_ids()` (`sp`/`placeholder_triplet` threaded through
  `parse_prompt`/`render_generated_text`/`on_submit`), replacing the 32000-hardcoded imports;
  (c) `generate()` already surfaces `outputs["image_clip_cond"]` (List[List[(seq 77x2048,
  pooled 1280)]]) — the chat now loads a frozen SDXL pipe (fp16-fix VAE) when the image
  generator is an `SDXLConditioningAdapter` and renders the predicted conditioning via
  `render_sdxl_cond` instead of LiteVAE-decoding a latent. New args: `--sdxl_model`,
  `--sdxl_gen_steps`, `--sdxl_guidance`. Text/voice/audio arms untouched; DiT/LiteVAE path
  preserved as the `elif` fallback. py_compile clean; not yet run against a live checkpoint.

## Lower priority / conditional

- [ ] **Semantic input encoder (SigLIP2 / DINOv2) for the transcription arm** instead of
  LiteVAE latents. VAE latents are reconstruction-optimized, weak for understanding;
  a pretrained semantic encoder completes the frozen-experts-at-boundaries pattern.
  Only when the image→text arm is actually built. (See discussion; also makes the
  LiteVAE-in path fully retire-able.)

- [ ] **Optional diffusion-loss polish phase** (ELLA-style *timestep-aware* connector,
  frozen UNet in the training graph). Only if the static CLIP-regression adapter's
  fidelity plateaus. Expensive (SDXL in the training graph → OOM risk on a 4090); a
  distinct training regime, not a drop-in. Deferred by design.

- [ ] **Caption grayscale scan on jackyhate** — cheap text-only check for grayscale
  over-representation in the caption distribution (matters only via captions, since
  images are unused for generation). Optional diagnostic before a real run.

- [ ] **`--repack N` mode for `merge_shard_dirs.py`** — only if fewer shard files are
  wanted. Keep N small (≤~16k samples) to preserve shuffle granularity; larger shards
  hurt shuffle + memory. Currently *not recommended* (3MB lean shards are fine on nvme).

## Cross-cutting (coordinate with world-tts)

- [ ] **SmolLM2 intermediate-layer tap.** Reading the LLM's *final* layer may be
  suboptimal (final states are next-token-specialized; LLM2Vec favors mid-to-late layers).
  Ablate which layer the translator reads — affects both the voice and image text spine.
