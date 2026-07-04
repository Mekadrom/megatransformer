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

Validated 2026-07-04: `mos_recon` 1.71 @5k (robotic) → 3.43 @15k (smooth) — clean and
monotonic, unlike GV. At 15k the baseline sits ~0.39 MOS below the ceiling, so **GAN's
naturalness headroom is small** (it can't exceed the vocoder ceiling). Heavier than GV
(needs a vocoding pass + the UTMOS model); use `--n 200`.

```
CUDA_VISIBLE_DEVICES=N PYTHONPATH=src python3 -m megatransformer.scripts.eval.smg.mos \
  --checkpoint runs/smg/<run>/checkpoint-<step> --n 200
```

## Reading the results — caveats

1. **Mask to valid frames.** The metric trims to `mel_length`; comparing against the
   full zero-padded mel inflates every L1 ~8× and muddies the ratios. The script does
   this — if you adapt it, keep the `mel_length` trim.
2. **`--config` / `--sive_encoder_dim` MUST match the checkpoint** or the load is
   garbage. Baseline runs use `medium_decoder_only_1d_3x` + 256.
3. **It's light** — a few hundred small conv decodes, fits next to a training run;
   run on whichever GPU has headroom (`nvidia-smi`). Uses `@torch.no_grad()`.
4. **Reference points (recon-only baseline, ~epoch 20 / step 10k):** `l1_true ≈ 0.44`,
   `disentangle ≈ +0.035`, `rel_influence ≈ 0.35` = weak, non-controlling embedding
   (collapse). Track these rising when the FiLM contrastive loss is on; that's the
   sign conversion is being restored.
5. **The fix is SMG-side, not SIVE-side.** SIVE leakage floors at ~140× across every
   variant tried (covreg tied, grllr3x worse), so you can't swap to a lower-leakage
   SIVE — the contrastive loss is the lever.

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
