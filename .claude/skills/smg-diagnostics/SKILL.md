---
name: smg-diagnostics
description: Diagnose a trained SMG (SIVE-Mel Generator) checkpoint. Two metric families — (1) speaker-embedding control / cross-speaker "collapse" (does the FiLM embedding control the output or does the decoder ignore it and reconstruct the source speaker from the leaky SIVE features), and (2) over-smoothing / "robotic" naturalness (GV + HF variance, the signature a no-GAN recon leaves and that GAN repairs). Use when the user reports voice-conversion failing, output sounding like the source speaker, output sounding robotic/muffled, or wants a quantitative before/after read across SMG checkpoints (e.g. baseline vs GAN).
---

# SMG diagnostics

The SMG decodes frozen SIVE content features + an ECAPA speaker embedding (FiLM) +
F0 conditioning into a mel spectrogram. The critical failure mode is **speaker-
embedding collapse**: the decoder learns to read speaker identity straight from the
SIVE features (which leak ~140× macro; see [[project_sive_experiment_status]]) and
*ignores* the embedding, so cross-speaker / voice-conversion output reproduces the
SOURCE speaker regardless of the target embedding.

**Why it happens (mechanism, not a bug):** recon-only SMG training pairs each
utterance's features with *its own* embedding, so there is zero gradient pressure
for the embedding to *control* speaker identity — the embedding is redundant for
reconstruction. The collapse *worsens as the decoder converges* (better recon → more
reliance on the feature shortcut). The intended fix is the **FiLM contrastive loss**
(`--film_contrastive_loss_weight`, ramped), which forces different-embedding →
different-output. This metric is how you watch it recover.

## Standard run procedure — DO THIS EVERY TIME "run smg-diagnostics" is invoked

This is the canonical sweep (reproduces the baseline-vs-identity @11k run). Follow it
verbatim unless the user narrows the scope.

**0. Same-step rule (HARD).** When comparing two runs, evaluate BOTH at the *same*
checkpoint step — pick a step both have on disk (HF prunes to last N). Never mix steps.

**1. Match config + dim to the checkpoint from the run name.** `..._1d3x_...` →
`--config medium_decoder_only_1d_3x`; a `stdhinge11`/SIVE-256 base → `--sive_encoder_dim
256`; a 50 Hz ContentVec SMG (`..._1d_1x`, dim 768) → `--config medium_decoder_only_1d_1x
--sive_encoder_dim 768`. Wrong config/dim = garbage load. Val cache defaults to
`./cached_datasets/smg_libritts_r_clean_stdhinge11-300k_val` (ContentVec SMGs →
`..._clean_contentvec_val`).

**2. GPU discipline.** The user runs training on these GPUs. Do NOT infer "free" from one
`nvidia-smi` util reading. Run GPU jobs **one at a time**, low-profile, pinned with
`CUDA_VISIBLE_DEVICES=` to whichever GPU has headroom (or the one the user designates).
`conditioning_health` is CPU (`--device cpu`) — run it freely alongside a GPU job.

**3. Output convention.** One parent dir per comparison:
`eval_outputs/<task>_<step>_<idx>/` (e.g. `eval_outputs/baseline-identity_step11000_0/`).
These scripts print to stdout (no `--output_dir`), so **`tee` each into that folder** as
`<tool>_<runtag>.log`, and write a `SUMMARY.md` with a matched-step comparison table + a
verdict at the end. Keep this structure going forward.

**4. The four-tool sweep** (run all four; on a two-run comparison, run each on BOTH at the
same step). Reference reads and full caveats are in the per-tool sections below.
- `embedding_control.py` (`--n 400`) — **the headline collapse metric.** Judge success by
  `disentangle` rising (baseline collapse floor ≈ +0.03 / rel_influence ≈ 0.36–0.40;
  healthy conversion +0.2–0.4 / rel_influence >0.5). This is the one that decides "did the
  embedding take control."
- `conditioning_health.py` (`--device cpu`) — is the FiLM path structurally alive/redirected
  (trained/init, RMS-vs-backbone). CPU, safe next to training.
- `oversmoothing.py --wrong_emb` (`--n 400`) — same-spk `gv_ratio` + conversion
  `gv_ratio_wrong` (the extrapolative regime; a real converter drops here = the GAN target).
- `mos.py --wrong_emb` (`--n 200`) — perceptual UTMOS `mos_recon` + converted `mos_wrong` +
  `conv_gap` (the naturalness cost of real conversion = the GAN decision point).
- `content_preservation.py` (`--n 300`, needs `--sive_checkpoint … --sive_layer 10`) — **run
  this whenever a conversion/identity loss is active.** It DECONFOUNDS `disentangle`, which
  rises for BOTH real conversion AND content-breakage. Reports two independent axes: SPEAKER
  `conversion margin = convert(→target) − residual(→source)` (want > 0) and CONTENT
  `content_drift = content_wrong − content_true` (frozen SIVE on the decoded mel vs source
  features; want ≈ 0). High margin + high `content_drift` = "right speaker, wrong words"
  (the content-blind-identity-loss failure; identity run @16k = margin +0.42, drift +5.71).
  The fix is `--identity_content_loss_weight` (SIVE-perceptual on the swap); success = drift
  falls toward `content_true` while margin stays positive.

**5. Interpretation frame.** On a COLLAPSED model, `mos_wrong ≈ mos_recon` and
`gv_ratio_wrong ≈ gv_ratio` are ARTIFACTS (converted == source), not good conversion — read
them together with `disentangle`. A big `disentangle` jump + a big `conv_gap`/`gv_ratio_wrong`
drop = collapse broken, quality cost moved onto converted output = hand off to the GAN arm
(`--use_gan`). Present a matched table (baseline | run @ same step) and end with a verdict.
⚠️ **`disentangle` alone is CONFOUNDED once a conversion loss is on:** `l1_wrong` rises for
real speaker conversion AND for content-breakage, so a rising `disentangle` is NOT proof of
clean conversion — always pair it with `content_preservation.py` to split the two (a content-
blind identity loss can show a great `disentangle` while garbling words). After a content fix,
EXPECT `disentangle` to DROP (its content-drift inflation is removed); judge by margin +
`content_drift` + the ear, not `disentangle`.

Example (fills the parent dir + logs; adapt run names / step / config):
```
mkdir -p eval_outputs/<task>_step<STEP>_0 && OUT=eval_outputs/<task>_step<STEP>_0
# CPU, both runs:
PYTHONPATH=src python3 -m megatransformer.scripts.eval.smg.conditioning_health --checkpoint runs/smg/<run>/checkpoint-<STEP> --config <cfg> --sive_encoder_dim <dim> --device cpu 2>&1 | tee $OUT/conditioning_health_<tag>.log
# GPU, one at a time (pin CUDA_VISIBLE_DEVICES):
CUDA_VISIBLE_DEVICES=<G> PYTHONPATH=src python3 -m megatransformer.scripts.eval.smg.embedding_control --checkpoint runs/smg/<run>/checkpoint-<STEP> --config <cfg> --sive_encoder_dim <dim> --cache_dir <val> --n 400 --device cuda 2>&1 | tee $OUT/embedding_control_<tag>.log
CUDA_VISIBLE_DEVICES=<G> PYTHONPATH=src python3 -m megatransformer.scripts.eval.smg.oversmoothing --checkpoint runs/smg/<run>/checkpoint-<STEP> --config <cfg> --sive_encoder_dim <dim> --cache_dir <val> --n 400 --wrong_emb --device cuda 2>&1 | tee $OUT/oversmoothing_<tag>.log
CUDA_VISIBLE_DEVICES=<G> PYTHONPATH=src python3 -m megatransformer.scripts.eval.smg.mos --checkpoint runs/smg/<run>/checkpoint-<STEP> --config <cfg> --sive_encoder_dim <dim> --cache_dir <val> --n 200 --wrong_emb --device cuda 2>&1 | tee $OUT/mos_<tag>.log
```

## Primary tool — embedding-control metric

`scripts/eval/smg/embedding_control.py` — decodes a val subset twice per utterance
(TRUE embedding vs a DIFFERENT-speaker embedding) and reports, on VALID (unpadded)
frames only:

- `l1_true` — recon L1 vs GT mel (normal recon quality; should track training L1).
- `l1_wrong` — recon L1 using a wrong speaker's embedding.
- `disentangle = l1_wrong − l1_true` — **~0 ⇒ the wrong embedding reconstructs the
  source just as well ⇒ embedding IGNORED (collapse).** Healthy conversion wants this
  large (~0.2–0.4): a wrong speaker's rendering should be far from the GT source.
- `output_diff = L1(recon_true, recon_wrong)` — how much swapping the embedding
  changes the output at all.
- `rel_influence = output_diff / l1_true` — **<~0.3 ⇒ the embedding barely matters
  (collapse).** Healthy conversion wants >~0.5.

Swapping the embedding changes BOTH the predicted F0 and the FiLM conditioning
(`decode()` predicts F0 from the given embedding), so this is the full conversion
signal, not just FiLM.

```
CUDA_VISIBLE_DEVICES=N PYTHONPATH=src python3 -m megatransformer.scripts.eval.smg.embedding_control \
  --checkpoint runs/smg/<run>/checkpoint-10000 \
  --config medium_decoder_only_1d_3x --sive_encoder_dim 256 \
  --cache_dir ./cached_datasets/smg_libritts_r_clean_stdhinge11-300k_val --n 400
```

## Naturalness tool — over-smoothing (GV + HF)

`scripts/eval/smg/oversmoothing.py` — objectively quantifies how "robotic" the
reconstructions are. Robotic quality = over-smoothing (L1/MSE predicts the mel's
conditional MEAN → flattened harmonics, reduced frame-to-frame variance). Reconstructs a
val subset (TRUE embedding) and reports, on valid frames:

- `gv_ratio` = mean-per-dim var(recon) / var(GT), all mel bins. **<1 ⇒ over-smoothed/
  robotic; GAN pushes it back toward 1** — the ideal before/after number for the GAN arm.
- `hf_gv_ratio` = same variance ratio restricted to HIGH mel bins (`>= --hf_bin`, default
  53 ≈ 3.3 kHz). HF detail is smoothed first, so this is usually lower than `gv_ratio`.
- `hf_mean_gap` = mean(recon HF band) − mean(GT HF band), mel units. Negative ⇒ recon is
  duller/muffled in the high band.
- `--wrong_emb` → also computes `gv_ratio_wrong` on each utt decoded with a DIFFERENT
  speaker's embedding (the CONVERSION regime). `gv_ratio_wrong` below `gv_ratio` ⇒ the
  extrapolative conversion over-smooths *more* than same-speaker recon = GAN's target. On a
  collapsed model it ≈ `gv_ratio` (converted == source; verified 50k baseline 0.957 vs 0.963).

All from mels (no vocoder) — cheap, isolates the SMG.

```
CUDA_VISIBLE_DEVICES=N PYTHONPATH=src python3 -m megatransformer.scripts.eval.smg.oversmoothing \
  --checkpoint runs/smg/<run>/checkpoint-<step> \
  --config medium_decoder_only_1d_3x --sive_encoder_dim 256 \
  --cache_dir ./cached_datasets/smg_libritts_r_clean_stdhinge11-300k_val --n 400
```

⚠️ **GV is NOT a standalone "roboticness" meter** — it measures variance *magnitude*, not
whether the variance is in the right place. Verified 2026-07-04: across the baseline run,
GV was NON-monotonic (0.995 @5k robotic → 0.895 @10k → 0.962 @15k smooth) — the *robotic*
checkpoint had the *highest* GV, because early roboticness is a FIDELITY problem (wrong
harmonic placement, full variance), which GV can't see. Use GV for **over-smoothing** and
the **GAN before/after** (GAN raises GV); use recon-L1 and/or UTMOS (below) for perceived
roboticness across training.

## Perceptual tool — UTMOS (`mos.py`)

`scripts/eval/smg/mos.py` — the perceptual naturalness metric that DOES track the ear.
Vocodes the SMG recon AND the GT mel (copy-synthesis) through the same HiFi-GAN and scores
both with **UTMOS** (torch.hub `tarepan/SpeechMOS` `utmos22_strong`, 16 kHz, MOS ~1-5;
already cached in `~/.cache/torch/hub`). Reports:

- `mos_recon` — UTMOS of the SMG's true-emb reconstruction (moves with perceived quality).
- `mos_gt_voc` — UTMOS of the GT mel vocoded = the **vocoder ceiling** (~3.82 for HiFi-GAN
  here; checkpoint-independent — to beat it you need a better *vocoder*, not a better SMG).
- `mos_gap` = ceiling − recon = the SMG's own naturalness cost.
- `--wrong_emb` → also decodes each utt with a DIFFERENT speaker's embedding, vocodes+scores
  it, reports `mos_wrong` + `conv_gap = mos_recon − mos_wrong`. The CONVERSION (extrapolative)
  regime — where an L1/MSE decoder is most prone to over-smooth = the **GAN decision point that
  same-speaker recon can't see.** Collapsed model ⇒ `mos_wrong ≈ mos_recon`, `conv_gap ≈ 0`
  (verified 50k baseline −0.002); once conversion is real (post-contrastive) a positive
  `conv_gap` is GAN's target.

⚠️ **UTMOS is a 16 kHz model.** Its `forward(wave, sr)` resamples the input to 16 kHz before
scoring (`model.py:32`), so it is BLIND to anything above 8 kHz. We pass `16000` because the
HiFi-GAN is 16 kHz (no-op). Consequence: UTMOS **cannot see a 24 kHz upgrade's 8–12 kHz gain**
— a 24 kHz pipeline would score ~flat/slightly-lower on UTMOS despite sounding crisper. Use
the ear or a wideband metric (NISQA wideband / Audiobox-Aesthetics) to measure a 24 kHz move.

Validated 2026-07-04: `mos_recon` 1.71 @5k (robotic) → 3.43 @15k → 3.73 @50k; `mos_gap`
0.39 → 0.19 → **0.097 @50k** (near the ceiling). GV `gv_ratio` 0.963 @50k, `hf_mean_gap` +0.023
(not muffled). So the recon-only baseline is essentially at the vocoder ceiling ⇒ **GAN's
same-speaker headroom is ~nil**; its real (untested) value is on CONVERTED outputs (`--wrong_emb`,
post-contrastive). Heavier than GV (vocoding pass + UTMOS model); use `--n 200`.

```
CUDA_VISIBLE_DEVICES=N PYTHONPATH=src python3 -m megatransformer.scripts.eval.smg.mos \
  --checkpoint runs/smg/<run>/checkpoint-<step> --n 200
```

## Structural tool — conditioning-path weight health (`conditioning_health.py`)

`scripts/eval/smg/conditioning_health.py` — asks **what, structurally, is dying** in the
collapse: it splits the model's parameters into the **speaker/FiLM path**
(`speaker_embedding_projection`, `early_film_projection`, `speaker_projections_2d`), the
**F0 path**, and the **backbone**, and reports each group's weight RMS against both the rest
of the model and a **freshly-initialized model of the same config** (`trained/init` ratio),
plus a per-tensor breakdown. No forward pass / data — pure weights, runs on **CPU** (`--device
cpu`), so it never touches the training GPUs.

- FiLM output layers init at `std=0.02` (`smg.py _init_film_projection`), so `trained/init ≈ 1`
  on those ⇒ the FiLM never learned to condition; `≫ 1` ⇒ it did.
- The script prints a **data-driven verdict**: `speaker_film grew x_ vs backbone x_`.

**KEY FINDING (50k baseline, 2026-07-04): the conditioning path is NOT dead — it's the most-
trained part of the model.** FiLM output layer `early_film_projection.2.weight` = **1.55× init**,
speaker_film group **1.40×** vs backbone **1.03×**. So the cross-speaker collapse is
**FUNCTIONAL, not structural**: the FiLM is heavily trained and *does* modulate the output
(`output_diff`/`rel_influence` ≈ 0.36), but that modulation carries **non-identity** attributes
(prosody/energy) — because the leaky SIVE features supply identity for free, there was never
gradient pressure for FiLM to encode identity. Two consequences: (1) the fix is NOT "revive a
dead path" — the weights are already grown; it's redirecting an active path toward identity, or
removing identity from the features (SIVE-side). (2) **A pure `output_diff` contrastive loss is
suspect** — the FiLM already produces large `output_diff`, so pushing it risks amplifying the
existing non-identity modulation rather than creating identity control. Prefer an identity-
targeting contrastive loss (speaker-encoder on the converted output → target embedding) or the
features-side levers (GRL-at-layer-10 / ECAPA-regression adversary).

```
PYTHONPATH=src python3 -m megatransformer.scripts.eval.smg.conditioning_health \
  --checkpoint runs/smg/<run>/checkpoint-<step> \
  --config medium_decoder_only_1d_3x --sive_encoder_dim 256 --device cpu
```

## Reading the results — caveats

1. **Mask to valid frames.** The metric trims to `mel_length`; comparing against the
   full zero-padded mel inflates every L1 ~8× and muddies the ratios. The script does
   this — if you adapt it, keep the `mel_length` trim.
2. **`--config` / `--sive_encoder_dim` MUST match the checkpoint** or the load is
   garbage. Baseline runs use `medium_decoder_only_1d_3x` + 256.
3. **It's light** — a few hundred small conv decodes, fits next to a training run;
   run on whichever GPU has headroom (`nvidia-smi`). Uses `@torch.no_grad()`.
4. **Reference points (recon-only baseline).** @10k: `l1_true ≈ 0.44`, `disentangle ≈ +0.035`,
   `rel_influence ≈ 0.35`. @50k (final, recon AT the vocoder ceiling): `l1_true 0.356`,
   `disentangle +0.025`, `rel_influence 0.360`, `gv_ratio 0.963`, `mos_recon 3.73` (gap 0.097).
   disentangle PLATEAUED at ~+0.025 by ~26k — the collapse forms early then sits. **+0.025 /
   0.36 is the floor the contrastive arm must move; judge success by `disentangle` rising, NOT
   `output_diff`/`rel_influence`** (those can rise cosmetically — the FiLM already modulates
   non-identity attributes, see conditioning_health).
5. **The fix is SMG-side OR SIVE-side — both are live.** SIVE leakage floors ~140–200× among
   *stable* variants, so you can't just swap to an existing lower-leakage SIVE. Features-side
   lever = new SIVE work: GRL-at-layer-10 (align the adversary to the SMG's layer-10 tap) +
   ECAPA-regression / MHASP adversary (both implemented). SMG-side lever = an *identity-
   targeting* contrastive loss — the naive `output_diff` hinge is suspect (conditioning_health
   shows FiLM already produces large output_diff without carrying identity). Both compound.
   Tapping SIVE layer 12 was ruled out (~24% worse recon on clean data).

## Notes / future extensions

- Recon quality itself is best read from the training TB curves (`smg_l1_loss`,
  `smg_mse_loss` — logged RAW/pre-scale; `smg_total_loss` is fully post-scale). The
  "robotic" quality of a no-GAN baseline is the recon-only ceiling — GAN territory.
- F0 conversion (does →male/→female re-pitch land) is measured on the SIVE side by
  `synthesis_usability.py`; an SMG-native F0-conversion + vocoded-audio gamut is a
  natural extension of this tool if needed.
- The SMG eval/visualization callback renders cross-speaker figures/audio but logs
  NO scalar for the collapse — this script fills that gap; consider wiring the
  `disentangle`/`rel_influence` scalars into the callback so they trend in TB.
