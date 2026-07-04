---
name: sive-diagnostics
description: Run speaker-leakage / disentanglement diagnostics on frozen SIVE checkpoints. Use when the user wants to measure how much speaker (or gender) identity survives in SIVE features, compare runs (GRL vs no-GRL, regularizer variants), or audit a SIVE checkpoint before/after a training change.
---

# SIVE diagnostics

Measure how much speaker / gender identity leaks into the frozen SIVE feature
(the "speaker-invariant" representation SMG and the world model consume). Low
leakage = good disentanglement.

## ALWAYS run BOTH probes — leakage AND synthesis

A complete SIVE diagnosis is **never just the leakage classifier**. Always run
**both**:
1. `per_speaker_leakage.py` — how much identity survives (the classifier view).
2. `synthesis_usability.py` — does the feature decode back to intelligible
   speech, **with vocoded WAVs to LISTEN to** (recon-L1, disentangle Δ, F0).

The **human ear on the rendered audio is one of the most crucial metrics** — it
has repeatedly caught things the leakage classifier *and* mel-L1 both miss
(pitch/texture collapse, cross-gender conversion quality, "mushy" reconstruction
that scores fine numerically). Never report a leakage number without running the
synthesis probe and surfacing its audio alongside. When comparing runs, render
all checkpoints at the **same fixed `--seed`** (e.g. 7; 42 has diverged before)
so the WAVs are matched and comparable, then have the user listen.

## Environment

- Run from the repo root with `PYTHONPATH=src` (the `-m` module path needs it).
- Pin the GPU with `CUDA_VISIBLE_DEVICES=N` (check `nvidia-smi` for a free one).
- Default datasets: train `./cached_datasets/voice_sive_gender_train_merged/`,
  val `./cached_datasets/voice_sive_gender_val_merged/`. The val set is ~24.7k
  utterances / ~699 speakers (heavily imbalanced — one speaker ~3.5k utts).
- Pass the `--config` that MATCHES the checkpoint: there is now a single SIVE
  preset, `small_deep_3xdownsample_conv2d_attentive` (256-dim). Older size/norm
  presets (incl. the 128-dim `tiny_deep_*` ablation) were pruned 2026-06-29 —
  recover one from the run's tagged commit if you need to eval a structural
  variant. **Always pass `--num_speakers 3610`** — the config default is stale.
- **Norm-variant checkpoints:** the four norm levers (frontend / block pre-norm /
  conformer conv / final) are now CLI-overridable at TRAIN time and default to the
  config's values (instancenorm / layernorm / instancenorm / layernorm). If a run
  was trained with a non-default norm, the eval model MUST be built the same way
  or features (hence every number) are silently garbage. `per_speaker_leakage.py`
  and `synthesis_usability.py` currently only expose **`--final_norm_type
  rmsnorm|none`** — pass it to match the checkpoint (omit for layernorm-final
  runs). NOTE: these eval scripts do NOT yet expose `--downsample_norm_type /
  --block_norm_type / --conv_norm_type`, so a checkpoint trained with a non-default
  frontend/block/conv norm cannot be matched here yet — add those flags before
  diagnosing such a run.
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
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 -m megatransformer.scripts.eval.audio.sive.per_speaker_leakage --config small_deep_3xdownsample_conv2d_attentive --num_speakers 3610 --val_cache_dir ./cached_datasets/voice_sive_gender_val_merged/ --output_dir ./eval_output/per_speaker_leakage --device cuda --max_samples 0 --min_utts 5 --test_split 0.3 --probe_max_epochs 600 --probe_patience 40 --checkpoint nogrl=runs/sive/<nogrl_run>/checkpoint-224000 --checkpoint grl=runs/sive/<grl_run>/checkpoint-224000
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
- **`xref_spk<j>_<g>` WAV** = the ACTUAL target speaker `j` the cross/wrong
  embedding came from (vocoded GT mel). The listening A/B is `recon_wrong_<g>` vs
  `xref_spk<j>_<g>`: did the cross-conversion land on *that* speaker, not just a
  gender? Same `j` is used across all runs at a given sample index (seed-fixed
  subset), so one shared reference lines up against every run's conversion.

```
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 -m megatransformer.scripts.eval.audio.sive.synthesis_usability --config small_deep_3xdownsample_conv2d_attentive --num_speakers 3610 --val_cache_dir ./cached_datasets/voice_sive_gender_val_merged/ --output_dir ./eval_output/synthesis_usability --device cuda --subset_size 1024 --probe_steps 5000 --dec_width 256 --dec_blocks 6 --num_render 8 --checkpoint name=runs/sive/<run>/checkpoint-300000
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
   and re-run (cheap — features are cached). 600 / 40 is usually enough, but
   slow-converging features (no-GRL, covreg, late checkpoints) can need 1500 / 60
   — when in doubt, use the higher numbers.
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
6. **A recon-L1 outlier or NaN is a diverged probe, not a feature verdict.** The
   decoder is now hardened (2026-06-30): a learnable per-frame **input-norm** makes
   it scale-robust (so `final_norm_type=none`'s unnormalized features decode fine —
   previously they diverged to ~3.1 across *every* seed), plus **grad-clip** and a
   **NaN-retry** (bumped seed) guard. This measures content/structure, not raw
   scale — faithful to what a real SMG does — and is uniform across all final-norm
   types, so the numbers stay comparable. Divergence should now be rare; if a
   recon-L1 outlier (≫ the ~1.0–1.1 norm) or NaN still appears, re-run with a
   different `--seed` before trusting it. (Historical: an un-hardened bad seed once
   nearly read as a "2× worse" run.)
7. **For a perceptual A/B, render both checkpoints at the SAME `--seed`.** The
   seed fixes the eval subset *and* decoder init, so cross-seed audio compares
   different utterances entirely. Compare matched `recon_true.wav` pairs only
   (same `sampleN_spkID`). The user's ear is the tiebreaker — but only on matched
   audio.
8. **eff-dim does NOT compare across final-norm types.** It's the entropy of
   per-dim variance, which the final norm's affine/centering directly shapes —
   rmsnorm vs layernorm eff-dim can even disagree in *sign*. Also the offline
   (mean-pooled, from the leakage cache) eff-dim ≠ the TB
   `feature_health/dim_utilization_ratio` (computed over all frames). Use eff-dim
   only within the same norm type + same pooling.
9. **Judge disentanglement at the 300k endpoint, not mid-training.** The slow GRL
   purge does the real work in the back half; mid-training snapshots mislead —
   e.g. grlwup30k's 80k content/eff-dim edge over stdhinge11 fully washed out by
   the 300k endpoint (tied leakage + recon), reversing the mid-training read.
10. **Verify config-identity before trusting any A/B.** Diff the *resolved* config
   (not the run name) — a run renamed without actually changing the variable is a
   pure noise generator. A "rmsnorm-final" run that never set `final_norm_type`
   was config-identical to the baseline yet showed 184x-vs-139x macro leakage at
   300k: that gap was run-to-run trajectory noise, which also sets a sobering
   floor — treat sub-~1.3x single-run leakage gaps as possibly noise, not signal.
11. **Don't call a "falsetto" from the synthesis F0 scalar without the ear.** The
   per-gender F0 (`F0 male true-emb=...`) is torchcrepe's median-over-voiced-frames,
   median-over-`--f0_samples` (default 30) utts — robust to single outliers, but
   torchcrepe can systematically **latch onto a harmonic** of a run's spectral
   structure and report an *elevated* pitch (e.g. a 132Hz male read as 192Hz) even
   when the *perceived fundamental* is fine. This happened for covreg @100k (F0 said
   falsetto; the user's ear said as-good-or-better, esp. cross-speaker) — it was a
   spectral difference mis-mapped to pitch, not a real falsetto. So a high true-emb
   F0 is a FLAG to LISTEN, not a verdict; confirm on the rendered WAVs, and if you
   must quantify, bump `--f0_samples 100` and eyeball `f0_contours.png`. (Real
   falsetto — rmsnorm-final, grllr3x — was confirmed by ear; the metric alone isn't.)
   The `cross-emb` F0 is worse: NaN utts (torchcrepe found no confident-voiced frame)
   are **dropped before the median**, so a low **`cross voiced%`** column (new; also
   in the console line as `voiced N% of M`) means that F0 is a survivorship-biased
   median over a voiced minority — the typical cross recon had NO extractable pitch
   (an empty red line in `f0_contours.png`), which is exactly how covreg @100k read
   192. Low `cross voiced%` ⇒ treat the cross-emb F0 as noise and trust the ear.

## Notes

- Gender GRL is NOT implemented in the SIVE model/trainer (only a speaker GRL
  head exists); `gender_ids` are consumed only by these offline probes.
- Current experiment state, best base, ruled-out runs, and next levers live in
  memory `project_sive_experiment_status` (the entry point); detailed findings in
  `project_sive_speaker_leakage` + `project_sive_synthesis_usability`.
- The SIVE eval-time t-SNE viz callback had an unbounded-scan hang; see memory
  `project_sive_eval_hang`.
