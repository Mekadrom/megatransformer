---
name: sive-diagnostics
description: Run speaker-leakage / disentanglement diagnostics on frozen SIVE checkpoints. Use when the user wants to measure how much speaker (or gender) identity survives in SIVE features, compare runs (GRL vs no-GRL, regularizer variants), or audit a SIVE checkpoint before/after a training change.
---

# SIVE diagnostics

Measure how much speaker / gender identity leaks into the frozen SIVE feature
(the "speaker-invariant" representation SMG and the world model consume). Low
leakage = good disentanglement.

## Environment

- Run from the repo root with `PYTHONPATH=src` (the `-m` module path needs it).
- Pin the GPU with `CUDA_VISIBLE_DEVICES=N` (check `nvidia-smi` for a free one).
- Default datasets: train `./cached_datasets/voice_sive_gender_train_merged/`,
  val `./cached_datasets/voice_sive_gender_val_merged/`. The val set is ~24.7k
  utterances / ~699 speakers (heavily imbalanced — one speaker ~3.5k utts).
- Pass the `--config` that MATCHES the checkpoint: `small_deep_3xdownsample_conv2d_layernorm_attentive`
  (256-dim) for the standard runs, `tiny_deep_3xdownsample_conv2d_layernorm_attentive`
  (128-dim) for the 128-dim ablation. **Always pass `--num_speakers 3610`** — the
  config default is stale.
- **Norm-variant checkpoints:** the config preset defaults `final_norm_type` to
  `layernorm`. If the run was trained with `--final_norm_type rmsnorm` (or `none`),
  the eval model MUST be built the same way or the final norm loads wrong and ALL
  features (hence every number) are silently garbage. Both `per_speaker_leakage.py`
  and `synthesis_usability.py` take **`--final_norm_type rmsnorm|none`** — pass it
  to match the checkpoint (omit for standard layernorm runs).
- Checkpoints are HF Trainer dirs: `runs/sive/<run_name>/checkpoint-<step>/` with
  `pytorch_model.bin`.

## Primary tool — per-speaker leakage (the one to reach for)

`scripts/eval/audio/sive/per_speaker_leakage.py` — closed-set speaker-ID probe
(linear + MLP) on the FINAL mean-pooled SIVE feature, speaker-stratified
utterance-disjoint split, per-speaker top-1/5 recall (micro + macro),
ratio-over-chance, Gini/CV + top-k share, gender-structured-confusion stat,
linear-vs-MLP per-speaker deltas, hot-spot ids, markdown report + 2 plots/run +
cross-checkpoint table. Repeat `--checkpoint name=path` for a multi-run contrast
(always include a no-GRL run as the baseline). Per-utterance features are cached
to npz, so re-running only the probes is instant.

```
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 -m megatransformer.scripts.eval.audio.sive.per_speaker_leakage --config small_deep_3xdownsample_conv2d_layernorm_attentive --num_speakers 3610 --val_cache_dir ./cached_datasets/voice_sive_gender_val_merged/ --output_dir ./eval_output/per_speaker_leakage --device cuda --max_samples 0 --min_utts 5 --test_split 0.3 --probe_max_epochs 600 --probe_patience 40 --checkpoint nogrl=runs/sive/<nogrl_run>/checkpoint-224000 --checkpoint grl=runs/sive/<grl_run>/checkpoint-224000
```

Outputs land in `--output_dir`: `per_speaker_leakage_report.md` (read this),
`per_speaker_leakage_results.json`, and per-run `per_speaker_recall_*.png` +
`hot_speakers_*.png`.

Bonus from the cached npz (no GPU): a binary **gender probe** (speaker-disjoint
split, balanced accuracy vs 0.5 chance) and **effective dimensionality** are both
computable directly from `_features_*.npz`. Gender leaks ~0.91 balanced acc in
every run measured so far (nothing targets it — there's no gender GRL); low
effective_dim flags feature collapse (covreg-style or a bunk run).

## Synthesis-usability (tiny-SMG) — does the feature decode back to speech?

`scripts/eval/audio/sive/synthesis_usability.py` — the metric that actually
predicts SMG usability (CTC is blind to prosody/reconstructability). Trains a
small SMG-shaped FiLM-conditioned conv decoder (fixed budget, comparable across
checkpoints, NOT to convergence) to reconstruct mel from the frozen feature + the
stored ECAPA embedding, and reports:
- recon L1 with the **true** embedding = content sufficiency (lower better);
- disentanglement Δ = L1(shuffled emb) − L1(true) (higher better);
- **gender-stratified** recon + **cross-gender re-pitch** (F0 via torchcrepe) +
  per-run mel-comparison figures, vocoded `target`/`recon_true`/`recon_wrong_<g>`
  WAVs, and `f0_contours.png`.

```
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 -m megatransformer.scripts.eval.audio.sive.synthesis_usability --config small_deep_3xdownsample_conv2d_layernorm_attentive --num_speakers 3610 --val_cache_dir ./cached_datasets/voice_sive_gender_val_merged/ --output_dir ./eval_output/synthesis_usability --device cuda --subset_size 1024 --probe_steps 5000 --dec_width 256 --dec_blocks 6 --num_render 8 --checkpoint name=runs/sive/<run>/checkpoint-300000
```

For a quick bunk-check, add `--no_f0 --num_render 2 --probe_steps 3000` (skips the
slow vocoder/F0 pass). A bunk run shows a much higher recon L1 than a healthy one
plus collapsed effective_dim. Validated finding: this metric ranks runs in line
with human listening where CTC/L1-magnitude alone can mislead (e.g. covreg had
best CTC but worst, mushiest reconstruction).

## Secondary tool — per-layer sweep

`scripts/eval/audio/sive/speaker_leakage_probe.py` — linear+MLP probe across ALL
encoder layers + CKA / silhouette / cosine-ratio. Use to find which intermediate
layer leaks least. **CAVEAT: it reads `sample["mel_spec"]` and will KeyError on
the waveform-only `voice_sive_gender_*_merged` shards** — only run it on shards
that stored precomputed mels, or extend it to extract mels from waveforms the
way per_speaker_leakage.py does.

## Tertiary — feature timelapse

`scripts/eval/audio/sive/timelapse_sive_encoder_features.py` — MP4 of how a
sample's per-layer features evolve across checkpoints. Qualitative only.

## Reading the results — non-negotiable caveats

1. **Confirm the probe plateaued.** An under-trained probe UNDER-reports leakage
   and can invent false stories (e.g. a phantom "nonlinear residual" when the
   linear probe simply hadn't converged). Check `plateaued=True` for every probe
   in the report; if any say False, bump `--probe_max_epochs` / `--probe_patience`
   and re-run (cheap — features are cached). 600 / 40 is usually enough.
2. **Macro is the honest metric, not micro.** The val set is wildly
   utterance-imbalanced, so micro@1 is dominated by a few high-count speakers.
   Judge leakage on **macro@1 ×chance** (typical speaker); use the micro-vs-macro
   gap + Gini to tell concentrated-vs-broad leakage.
3. **Always express accuracy as ratio-over-chance** (chance = 1/K top-1), never
   raw — K depends on the `--min_utts` filter (~467 speakers at min_utts=5).
4. **Mel params must match training**: sr=16000, n_mels=80, n_fft=1024,
   hop_length=256 (the script defaults). Wrong params → garbage features.
5. The split depends only on `speaker_ids` + `--seed`, so it's identical across
   checkpoints — a clean GRL-vs-no-GRL contrast.

## Notes

- Gender GRL is NOT implemented in the SIVE model/trainer (only a speaker GRL
  head exists); `gender_ids` are consumed only by these offline probes.
- Findings as of the 224k checkpoints live in memory `project_sive_speaker_leakage`.
- The SIVE eval-time t-SNE viz callback had an unbounded-scan hang; see memory
  `project_sive_eval_hang`.
