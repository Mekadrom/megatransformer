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
- Config for the current 12-layer runs: `small_deep_3xdownsample_conv2d_layernorm_attentive`
  (encoder_dim=256). **Always pass `--num_speakers 3610`** — the config default
  is stale and the encoder load needs the right head size (the encoder features
  load fine regardless, but be explicit).
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
