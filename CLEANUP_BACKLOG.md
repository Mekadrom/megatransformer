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

- [ ] **End-to-end smoke test of the SDXL-adapter training path.** Steps 1–4 (CLIP-target
  loss, weight, preset, eval) are unit-tested in isolation but never run through the full
  trainer. Run `train world --config small_sum_sdxl --include_modes text,image` for a few
  hundred steps on a small slice; confirm `image_clip_loss` appears and drops. **Do this
  before any real SDXL-adapter run.**

- [ ] **Validate `eval_sdxl_adapter.py` against a real trained checkpoint.** It compiles,
  but the synthesis-forward → SDXL-render → CLIPScore path has only been reasoned through,
  not executed on a trained model. Shake out once the smoke run produces a checkpoint.

- [ ] **Add `diffusers` + `accelerate` to the uv `training` dependency group** once SDXL is
  committed to. Currently transient (`uv pip install`), so `uv sync` would drop them.
  Touches: `pyproject.toml`, `uv.lock`, `requirements.txt`.

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
