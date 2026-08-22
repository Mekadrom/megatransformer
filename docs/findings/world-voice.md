# world-voice (text -> voice)

Speech synthesis from the recurrent trunk into CosyVoice 2 speech tokens, decoded by the
frozen CosyVoice 2 flow decoder. Formerly "world-tts".

Stack: text -> frozen SmolLM2-135M -> recurrent trunk (trainable) -> voice coda ->
CosyVoice 2 FSQ tokens (25Hz, vocab 6562, EOV = 6561) -> frozen flow decoder + campplus
speaker embedding -> audio.

---

## ESTABLISHED

### The pipeline ceiling is 0.8937 LCS recall / 0.1056 WER
Ground-truth units through the same frozen decoder and Whisper. Everything else is read
against this: the decoder and ASR are not the limitation. `cosyvoice_wer_eval.py` reports
gen / ceiling / gap.

### NAR learned bidirectional inpainting instead of reading text (2026-08-21)
Masked-parallel (MaskGIT-style) voice, cosine mask schedule, measured at step 23000
teacher-forced across four mask ratios:

| mask ratio | acc_real | text_delta | text-attributed |
|---|---|---|---|
| 1.00 | 0.0169 | +0.0037 | 0.220 |
| 0.75 | 0.0491 | +0.0033 | 0.068 |
| 0.50 | 0.0701 | +0.0023 | 0.032 |
| 0.25 | 0.0860 | +0.0025 | 0.029 |

Revealing 25% of frames raises accuracy 5x and drops perplexity 20x (1832 -> 91) while the
text contribution FALLS. Context is worth ~28x more to this model than the transcript.

### At matched step, AR reads text ~10x more than NAR (2026-08-21)
Both at step 23000, same data, same trunk config:

| | AR (full context) | NAR (r=0.25) |
|---|---|---|
| acc_real | 0.0887 | 0.0860 |
| ppl_real | 162.3 | 90.8 |
| text_delta | +0.0271 | +0.0025 |
| text-attributed | 0.305 | 0.029 |

NAR predicts units as well as AR — better calibrated, even — and derives 3% of that from the
transcript against AR's 31%. Bidirectional context is a strong enough predictor that text
became unnecessary. Removing the autoregressive crutch installed a better one.

### The models are NOT mode-collapsed; repetition comes from self-conditioning (2026-08-21)
Direct measurement of the unit head's per-step predictive distribution
(`unit_confidence_probe.py`, n=256):

| | raw top1 | raw entropy (bits) | frac top1 > 0.9 |
|---|---|---|---|
| NAR r=1.0 | 0.0163 | 10.96 | 0.000 |
| NAR r=0.25 | 0.0862 | 8.01 | 0.000 |
| AR @23k TF | 0.1188 | 6.47 | 0.000 |

Max entropy is 12.68 bits. Sampling is nowhere near argmax anywhere. At 8.4 bits under
T=0.6, independent draws would collide on adjacent frames well under 1% of the time, yet
observed adjacent repeat is ~0.6 — so repetition is produced by conditioning on the model's
own output, not by peaked predictions.

Both arms are also well calibrated: NAR at r=1.0 has mean top-1 probability 0.0163 against
top-1 accuracy 0.0169. It is not confidently wrong, it is accurately uncertain.

### Generating from text alone, the model is near-uninformed
10.96 of 12.68 bits at r=1.0 — 86% of maximum entropy, ~2500 effective choices out of 6562.
This is a more legible progress metric than `text_delta`: one number, no shuffled-text
control, with a known floor (AR reaches 6.47 with full context).

### The duration token solves length; the AR arm fails on long utterances specifically
32 log-spaced buckets over [25,250] frames, emitted between BOV and the voice placeholder,
predicted from the BOV hidden state. Measured at step 23000:

| | NAR + duration token | AR |
|---|---|---|
| text-len -> gen-len r | +0.933 (GT ceiling 0.960) | +0.624 |
| hyp/ref words | 1.00 | 0.58 |
| duration bucket +-1 accuracy | 0.538 | — |

AR's failure is not uniform: short utterances land near GT, and every long one (GT 154-181)
runs to the 250-frame cap without firing EOV. NAR produced 183/196/196 on those same three.

Caveats: the model fills the allocated block and never fires EOV, so length is capped by
duration-head accuracy with no self-correction, and a wrong bucket halves intelligibility
(0.0742 -> 0.0358). Bucket range measured from 8102 train utterances: min 25, median 102,
p90 204, only 0.16% at the cap.

### Decode variance floor: std 0.0089, range 0.018 at n=64
Same config, same checkpoint, three seeds: truncated LCS 0.0320 / 0.0498 / 0.0396. A
single-run difference must exceed ~0.025 to mean anything. **Use three seeds per config or n
well above 64 for any decode comparison.**

### LCS recall is partly a length metric
When two checkpoints fail on length in opposite directions it prefers the verbose one. AR at
23k over-runs (167/250/472 frames) and scores 0.1620; the same arm at 44k collapses to
near-empty (3-43 frames) and scores 0.0949. That gap is termination behaviour, not quality.
Always read it beside hyp/ref.

---

## OPEN

### Does training at mask ratio 1.0 build text conditioning the cosine schedule never did?
`--voice_nar_mask_schedule high --voice_nar_mask_ratio_min 1.0`, constant LR after warmup so
plateau is attributable to the model rather than LR decay. Run
`world_tts_cosyvoice2_smollm2_nar_r1_flat_lr-flat_0` started 2026-08-21.

Rationale: inference starts fully masked, so r~1 fixes the first commitments and anchors
every later round — and cosine trains it least (a third of its mass lands below r=0.5, where
inpainting suffices). At r=1.0 the model's text-attributed fraction is highest (0.220),
i.e. it reaches for text when nothing else is available.

Reference points at r=1.0: entropy 11.29 bits at step 1000 (start), 10.96 for the
cosine-trained model at 23k, 6.47 for AR with full context. Read `text_delta` and entropy at
r=1.0, NOT `acc_real` (lower by construction).

Decision at ~23k, matched against the cosine model's r=1.0 numbers. Kill only on a DECLINING
trend across three checkpoints spanning 20k+ — that is the standard the gen-query verdict
met (+0.0156 -> +0.0127 -> +0.0112 across 23k/35k/47k), and a 5k snapshot proves nothing.

### Is the crutch fixable, or is it a trunk property?
Gen-query NAR failed the same way on 2026-08-13 (nine days earlier, Mimi/ContentVec features,
homegrown SMG decoder): removing the local crutch collapsed early frames rather than forcing
text. If the r=1.0 run also flatlines, that is the same result twice on the same trunk with
different feature spaces, which points at the trunk rather than the token space.

### Does refinement help once text conditioning exists?
Currently no measurable refinement gain, but every rounds/reveal-order comparison sits inside
the seed floor. Prediction worth testing: in a model that reads text, committed tokens carry
real constraints and refinement should pay. If it still does not, the rounds duplicate the
recurrent loop and one of them should go.

### Is CFG available and does it transfer?
`null_text_embed` requires `--voice_cfg_text_dropout_prob > 0` AT CONSTRUCTION — it cannot be
retrofitted to a checkpoint. Guidance was the entire 77% -> 98% win on the image side and
applies to categorical logits unchanged. The prior "CFG is closed" verdict was measured on
the AR path pre-M-RoPE-fix and should be treated as untested, not settled.

---

## RETRACTED

### "M-RoPE eval numbers" — the eval path never enabled M-RoPE (found 2026-08-21)
`load_world_model` rebuilt the config from a name and never set `use_mrope`, and `load_model`
runs `strict=False`, so `rotary_global`/`rotary_local` were silently dropped and the trunk ran
single-axis RoPE with sequential positions. **Every post-hoc diagnostic ever run on an M-RoPE
checkpoint was affected**: the scale_side ablation (both arms), the WER/LCS figures, the
position-resolved horizon, and the per-head attention result.

Specifically retracted: "text horizon ~3-5 words and NOT widening" (delta halving every ~8
frames). Re-measured with the geometry engaged, text_delta runs +0.0503/.0430/.0429/.0396/
.0369/.0306 across the whole utterance and never drops below threshold. Also retracted:
`eov_position_acc` "degrading" to 0.218 — it is 0.818.

Fixed by auto-detecting `use_mrope` from the weights and REQUIRING `--mrope_scale_side`.

### "Eval decode temperature" — scripts sampled at T=1.0 while training viz used 0.6
The eval scripts hardcoded `voice_temperature=1.0`; `train.py` renders at
`viz_voice_temperature=0.6`. Every free-running statistic described a regime nobody listens
at, and the audio was audibly worse than TensorBoard's. Correcting it inverted the reading:
at 0.6 the model loops (adj_repeat 0.453) where 1.0 looked healthy (0.081) only because tail
noise was escaping the loop.

### "The distribution is peaked, so sampling is argmax"
Asserted twice from indirect evidence (identical-looking runs, ineffective sampler knobs) and
contradicted by direct measurement: `frac top1 > 0.9` is 0.000 everywhere. Both premises were
bugs — a NaN Gumbel and a seeding-order error.

### "Confidence-ordered reveal is catastrophic (1.7x)" and the mode-seeking story built on it
The comparison was 0.0320 (seed 0, the unlucky draw) against a single sequential run, a gap of
0.015 against a 0.018 noise floor. Not established. Likewise "sequential beats random" (0.006)
and "refinement loses to single-shot" (0.003).

### "Generation is deterministic"
Claimed after three seeds produced identical output. The seeding ran before the CosyVoice 2
decoder load, which fixes the RNG, so `--nar_seed` never took effect. With seeding moved
after, seeds diverge (0.0320 / 0.0498 / 0.0396).

### "AR@23k is the best AR checkpoint"
It is the checkpoint that talks the most. See "LCS recall is partly a length metric".

### "AR + duration token combines both wins"
The duration token buys evaluation hygiene and usability, not capability. Training is
teacher-forced on GT-length sequences, so runaway generation never degraded training or any
teacher-forced metric — it only contaminated free-running WER/LCS. Poor length control was a
SYMPTOM of weak progress-tracking, and an explicit duration token fixes the readout rather
than the cause.

---

## Tooling

`scripts_local/`: `world_voice_ar_diagnostics.py` (TF + text ablation + bootstrap CIs +
position buckets + free-running; `--nar_mask_ratio` REQUIRED for NAR checkpoints),
`cosyvoice_wer_eval.py` (WER/CER/LCS + ceiling; `--nar_seed`, `--nar_reveal`, `--nar_rounds`),
`unit_confidence_probe.py` (per-step predictive distribution), `wer_arm_compare.py` (paired
bootstrap over arms), `render_distill_audio.py`, `run_nar_eval.sh` (the NAR suite).

Text-free baselines for CosyVoice 2 (NOT the Mimi-era script defaults 0.211/0.229/0.117):
n-gram ceiling 0.0545, asymptote 0.2261, repeat crutch 0.0314.
