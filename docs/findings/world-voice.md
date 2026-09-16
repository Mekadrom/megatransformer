# world-voice (text -> voice)

Speech synthesis from the recurrent trunk into CosyVoice 2 speech tokens, decoded by the
frozen CosyVoice 2 flow decoder. Formerly "world-tts".

✅ **VAL CACHE RESTORED (2026-08-31) — evals are unblocked.**
The LibriTTS-R cache (`libritts_r_cosyvoice2_smollm2`) was deleted 2026-08-30 and is NOT
coming back. Its replacement is **LibriHeavy `large`, 1,985.5 h**, at
`cached_datasets/Mekadrom/libriheavy_cosyvoice2_smollm2/{train,val}`.

⚠️ **`train`/`val` are SUBDIRECTORIES, not `_train`/`_val` siblings.** `--voice_cache_dir`
appends `_train`/`_val` (`train.py:216`) and will NOT resolve this layout — pass
`--voice_train_cache_dir .../libriheavy_cosyvoice2_smollm2/train` and
`--voice_val_cache_dir .../libriheavy_cosyvoice2_smollm2/val` explicitly. Eval scripts that
take `--cache_dir <...>/val` point straight at the val dir and are fine.

| | value |
|---|---|
| train | 480 shards, 959,663 utts |
| val | 3 shards, 6,000 utts (utterance-disjoint; **speakers overlap train**) |
| total | 965,663 utts, 1,985.5 h, 178.7M voice tokens, 2,465 speakers |
| tokens/param | **~1.40** (was 0.078 on the 146 h corpus — 18x) |

Built by 3 parallel `preprocess_dataset voice --content_encoder cosyvoice2` workers over
disjoint parquet ranges, then `merge_shards --shuffle` (global shuffle across all three —
the worker dirs had disjoint speaker pools of 940/1,167/358, and `ShardAwareSampler` groups
batches by shard, so an unshuffled merge would have made every batch one narrow speaker
pool), then `stat-shards --speaker_id_column speaker_ids --additional_shard_dirs <val>` to
densify speaker ids to [0,2464] across both splits jointly.

**Verified before the `train_p*` sources were deleted** (2026-08-31): per-sample fingerprint
multiset over (unit_ids ⊕ token_ids ⊕ speaker_emb ⊕ speaker_id) is IDENTICAL to the sources;
train∩val fingerprint overlap 0; 0 unreadable shards / NaN embeddings / all-zero embeddings /
bad lengths / nonzero padding past `feature_lengths`. Aggregate-only checks would not have
caught a field-permutation bug, which is why the fingerprint check was run.

Two quirks, neither a defect: the corpus contains **26 exact-duplicate utterances**
(0.003%, present in the LibriHeavy source, all pairs landed within one split); and
`token_ids` width is a uniform 71 post-merge (merge pads to the per-output-shard max),
costing ~130 MB — `text_lengths` preserves truth.

⚠️ **NOT comparable across the corpus boundary.** Kept LibriHeavy segments average 7.46 s
against LibriTTS-R's 4.5 s, so length-sensitive results (hyp/ref ratio, budget-capping rate,
EOV rate) break at 2026-08-30. `--max_speaker_id 2338` was a LibriSpeech filter and is
meaningless here.

**Still UNMEASURED:** `ar_cos_0`'s finished 40000 checkpoint — the open question on which LR
schedule wins. It was blocked on this cache; it no longer is.

📁 **RUN PATHS MOVED 2026-08-26.** Runs now live in `runs/world_voice/` (split out from
`runs/world/`, which mixed image and voice), and the redundant `world_tts_` / `world_voice_`
prefixes were stripped from every directory:
`runs/world/world_voice_cosyvoice2_smollm2_ar_flat_lr_1` ->
`runs/world_voice/cosyvoice2_smollm2_ar_flat_lr_1`.

**Entries below use the NEW stripped names**, so a name in this file can be pasted straight
at `runs/world_voice/`. Two consequences:
- Each run's TensorBoard `training/command_line` still records the ORIGINAL prefixed
  `--run_name`. That is launch history, not a path, so it was not rewritten. A TB command line
  and a directory name therefore differ by the prefix; strip it to map between them.
- The git log before this date quotes the old names.

⚠️ **`docs/findings/world-image.md` made the OPPOSITE choice** — it kept the OLD full run names
in its entries and maps forward by dropping the prefix. So the two direction files are
internally consistent but differ from each other. When moving between them, check which
convention the file declares at its top rather than assuming.

**Stack:** text -> frozen SmolLM2-135M -> recurrent trunk (trainable) -> voice coda ->
CosyVoice 2 FSQ tokens (25 Hz, vocab 6562, EOV = 6561) -> frozen flow decoder + campplus
speaker embedding -> audio. Only the trunk and its per-mode adapters train.

**Data (current):** `cached_datasets/Mekadrom/libriheavy_cosyvoice2_smollm2/{train,val}` —
959,663 train / 6,000 val, <=10 s, 25 Hz, feature_channels 512. Holds `unit_ids` +
SmolLM2 `token_ids` + campplus-192 speaker embeddings; no mel, no F0. See the restored-cache
block at the top for the full build + verification record.

**Codebook:** `val/cosyvoice2_codebook.pt` (6561 x 512) is CosyVoice 2's own
`flow.input_embedding.weight`, the frozen decoder's input table — so a predicted id indexes
straight into what the decoder expects. It is NOT fit from data, so it regenerates exactly
from the model snapshot: `scripts_local/extract_cosyvoice2_codebook.py` (written 2026-08-31,
after the original was lost with the LibriTTS-R cache). Bound ids by the codebook's 6561 rows,
never by `flow.input_size` (512, the feature width). **EOV = 6561 sits OUTSIDE the codebook**;
stored `unit_ids` span exactly [0, 6560].

⚠️ **`--save_text` defaults to FALSE and its absence fails SILENTLY (found 2026-08-31).**
The LibriHeavy cache was built without it, so the shards carry `token_ids` but no raw `text`.
`world/dataset.py:405` only sets `voice_text` when `"text" in shard`, and
`cosyvoice_wer_eval.py:129` does `ref = str(s.get("voice_voice_text","")).strip()` then
`continue`s on empty — so **every sample is skipped and the eval reports zero rows rather
than erroring**. Add `--save_text` to any future voice preprocessing run. For an existing
cache the transcripts are recoverable exactly, no re-preprocessing needed:
`scripts_local/backfill_shard_text.py` decodes `token_ids[:text_lengths]` with SmolLM2
(verified `re-tokenize(decode(ids)) == ids` on 100% of 2,000 val rows; the old cache stored
the NORMALIZED transcript too, so this reproduces it rather than approximating it).

**Old data (pre-2026-08-30, deleted):** `libritts_r_cosyvoice2_smollm2` — 118,102 train /
4,708 val. Utterance lengths: min 25, median 102, mean 112, p90 204 frames; 0.16% hit the
250 cap. Numbers measured against it are length-incomparable with LibriHeavy (7.46 s mean
segment vs 4.5 s).

Provenance note: entries dated before 2026-08-21 are recorded from prior sessions of this
project. Entries dated 2026-08-21 were measured in that session. Contamination status for the
older ones is in its own section below — read it before trusting any pre-08-21 number.

---

## Sampling: RAS w=10 is the whole fix for "dragged-out" speech (2026-09-02, ESTABLISHED)

`cosyvoice2_smollm2_libriheavy_ar_flat_lr_ema_0` **checkpoint-40000**, val n=128 TF / gen_n=48,
three arms differing ONLY in the sampler. Reports: `eval_output/world_voice_ras_ab/{A,B,C}/report.md`.

⚠️ **`--viz_voice_ras_win` DEFAULTS TO 0**, so training-time TB renders have RAS **OFF** unless
the run passes it. Every render judged by ear before this date was arm A. This is not a
model failure being heard, it is an unset flag.

| metric | A: T=0.6 no RAS | **B: T=0.6 RAS w=10 tau=0.1** | C: T=1.0 top-p 0.8 | GT |
|---|---|---|---|---|
| adj_repeat_rate | 0.4506 | **0.0093** | 0.1734 | 0.0792 |
| longest_run | 179 | **4** | 79 | 18 |
| len_mean | 229.7 | **173.5** | 193.8 | 179.7 |
| EOV fired /48 | 18 | **42** | 40 | — |
| budget-capped /48 | 30 | **6** | 8 | — |
| unit_entropy_bits | 7.51 | **10.22** | 9.63 | 10.40 |
| distinct_bigram_ratio | 0.478 | **0.872** | 0.755 | 0.869 |
| text->length r | 0.076 | 0.465 | 0.570 | 0.744 |

`longest_run 179` at 25 Hz = **7.2 s of one repeated unit** — that IS the "dragging words out"
the ear reports. RAS takes length from +28% over GT to -3% and termination from 18/48 to 42/48.
Nucleus (C) is strictly worse than RAS on every degeneration axis.

**AMENDED 2026-09-15 — RAS is most of the fix, not the whole fix; GREEDY closes the rest.**
Measured through ASR (not units) at step 68000, n=12 paired samples, `--ras_win 10` in every
arm, `eval_output/world_voice_bistream_audio/`. CADENCE = generated frames per ASR-transcribed
word, divided by the same quantity for GROUND-TRUTH units through the SAME decoder, so a
value of 1.00 is the pipeline's own speaking rate and >1 is genuinely slow per spoken word:

| arm | WER | LCS-recall | cadence | hyp/ref words |
|---|---|---|---|---|
| unistream T=0.6 | 0.330 | 0.709 | 1.16x | 0.94 |
| bistream T=0.6 | 0.490 | 0.515 | 1.31x | 0.82 |
| unistream greedy | 0.355 | **0.723** | **0.99x** | 1.02 |
| bistream greedy | **0.325** | 0.695 | **0.99x** | 0.94 |
| CEILING (GT units) | 0.095 | 0.915 | 1.00x | 1.00 |

With RAS already on, T=0.6 still costs 16% (unistream) to 31% (bistream) in speaking rate.
Greedy lands both models at 0.99x — indistinguishable from the ceiling. The 2026-09-02 entry
stands (RAS removes the catastrophic 7.2 s repeat runs), but "the whole fix" was too strong:
residual drawl survived it and is a TEMPERATURE effect.

⚠️ **Do not read the LCS spread 0.723 / 0.695 / 0.709 as a ranking** — n=12, and a 0.03
difference there is noise. What n=12 supports is (a) cadence collapsing to 0.99x under greedy
and (b) bistream-at-T=0.6 being genuinely bad. Note also that greedy's unit-space
`adj_repeat_rate` is 0.0096-0.0127 against GT's 0.0792, i.e. it under-repeats badly; ASR word
overlap is blind to that, so greedy's cadence win should be confirmed BY EAR before it becomes
the default.

⚠️ **The bistream arms here decode UNISTREAM**, which is out-of-distribution for a
50%-bistream checkpoint. Chunked-protocol arms are the pending measurement; an n=3 chunked
smoke scored WORSE (LCS 0.325), but n=3 including the Latin sample is not a measurement.

**The decoder is NOT the bottleneck.** Ground-truth units through the frozen CosyVoice 2
decoder score WER 0.095 / LCS 0.915 on this val set, so ~0.20 of LCS gap is the world model's
own. (An earlier n=2 smoke suggested a much worse ceiling, WER 0.250 — that was one Latin
utterance, "Quos vult perdere dementat", which the ceiling renders as "Kwas would perdere the
mentored". At n=12 the ceiling is clean.)

**RAS fixes over-length HERE, contradicting the prior-era note** in `render_distill_audio.py`'s
docstring ("RAS fixed repetition ... but did NOT fix over-length, 2.54x -> 2.32x GT"). That
observation belongs to the pre-LibriHeavy era/checkpoint; do not carry it forward. Not marked
RETRACTED because it was never a findings entry — but treat the docstring as era-scoped.

**Position-resolved repeat is FRONT-LOADED at this checkpoint, not rising** (A: 0.569 / 0.362 /
0.432 / 0.453 over frame buckets 1-32 / 32-64 / 64-128 / 128+). Per the probe's own rubric that
means "repetitive from the start" = conditioning, NOT exposure bias. The old LibriTTS-R run at
step 11076 WAS rising (0.064 -> 0.330); that diagnosis does not transfer. Arm C's profile does
rise (0.187 / 0.087 / 0.133 / 0.231) — nucleus permits the late drift RAS blocks outright.

⭐ **The wall is text conditioning, and sampling cannot touch it.** Teacher-forced metrics are
IDENTICAL across all three arms (acc_real 0.1769 / 0.1765 / 0.1763 — the clean control proving
only the sampler varied):

| | value | teacher ceiling |
|---|---|---|
| early_text_delta | **+0.0264** [+0.0078, +0.0459] | +0.0538 [+0.0480, +0.0597] |
| text-attributed fraction | **0.344** | 0.463 |

RAS makes output SOUND right — length, diversity, termination all land on GT — while text->content
binding stays at ~half the teacher's. Judging by render alone will now hide this, because the
renders improve a lot without this number moving at all.

Two open items on B: `len_min` 4 frames vs GT 74 (RAS may convert droning into premature EOV on
some prompts — 6/48 still budget-cap, so both failure modes coexist), and `adj_repeat_rate`
0.0093 is well BELOW GT's 0.0792, i.e. RAS bans legitimate repeats. Untested: `--ras_tau 0.2`
or a shorter window.

**Per-utterance render lengths** (`eval_output/world_voice_ras_ab/audio_ck40000/`, n=8, frames):

| # | GT | RAS | plain | plain/GT |
|---|---|---|---|---|
| 0 | 130 | 155 | 245 | 1.88x |
| 1 | 246 | 250 | 250 | 1.02x |
| 2 | 155 | **62** | 250 | 1.61x |
| 3 | 136 | 96 | 244 | 1.79x |
| 4 | 156 | 207 | 248 | 1.59x |
| 5 | 205 | 162 | 231 | 1.13x |
| 6 | 172 | **39** | 250 | 1.45x |
| 7 | 242 | 235 | 250 | 1.03x |

**Plain hits 231-250 on ALL 8** regardless of whether the text wants 130 or 246 frames — it does
not track text length at all, it runs until the budget stops it. This is the single clearest
statement of the "dragging" the ear reports.

⭐ **The bimodal "either it works or it doesn't" is RAS's failure mode, not plain's.** With RAS,
6/8 are reasonable but #6 gives 39 frames for text needing 172 and #2 gives 62 for 155 — TRUNCATED,
not dragged. Plain never truncates (it never terminates at all). So enabling RAS trades a
guaranteed-mild failure on every utterance for a severe failure on ~25% of them. Ear must
arbitrate that trade; the degeneration metrics all favour RAS and cannot see it.

### CosyVoice 2 sampler alignment: three fixes, one clear effect (2026-09-11, ESTABLISHED)

Our voice sampler diverged from CosyVoice 2's in three ways; all fixed in `0f1e429`, each
checked against the reference in `~/dev/projects/cosyvoice-runtime`.

1. **Resample scaling (a real bug).** The RAS resample read RAW `logits` while the initial pick
   read `logits/temperature`, so at T=0.6 every ban silently resampled at **T=1.0** — flatter,
   handing EOV and the tail more mass at exactly the moments RAS fires. CV2 passes the same
   `weighted_scores` to `nucleus_sampling` and `random_sampling`, so one tensor is also what the
   reference does. The resample stays UNTRUNCATED: CV2's `random_sampling` is
   `weighted_scores.softmax(0).multinomial(1)` over the full vocab. (An earlier same-session
   claim that omitting top-k/top-p on resample was ALSO a bug was WRONG — it matches CV2.)
2. **Nucleus semantics.** Now one prefix over the FULL softmax while `cum < top_p and
   len < top_k`. Ours applied top_k first and measured top_p against the RENORMALISED top-k
   mass, a different and smaller set. Verified the kept sets are now identical to CV2's on
   random logits.
3. **`voice_min_frame_ratio`** — CV2's text-proportional EOS floor
   (`min_len = text_len * min_token_text_ratio`). CV2's ratio 2 is against its 3:1 token rate;
   scaled to our 7.5 frames/token that is 5.0. Calibrated on the 6,000-utterance val set:
   ratio 3.0 -> mean floor 73 frames, **0%** of GT violates; 5.0 -> 122 frames, 1.07%; 6.0 -> 8.92%.

**MATCHED pre/post control at ck78000** (same checkpoint, rate, flags; only `world_model.py` +
the probe swapped to `6af9337`). `eval_output/world_voice_cv2sampling/`:

| | PRE-FIX | POST-FIX | vs measured sd |
|---|---|---|---|
| adj_repeat | 0.0100 | **0.0362** | **+22 sd** (GT 0.0792) |
| collapsed <50% GT | 4.2% | 0.0% | 2 utterances |
| len_min | 22 | 49 | — |
| hit rate | 79.2% | 83.3% | +1.2 sd (ns) |
| duration r | 0.716 | 0.702 | -0.5 sd (ns) |
| acc_real | 0.1825 | 0.1822 | flat |

⭐ **The one unambiguous effect is REPETITION NATURALNESS**, not the cutoffs: adj_repeat 22 sd
toward GT. **Every prior RAS arm's 0.009-0.014 adj_repeat was the BUG over-suppressing**, not
RAS working as designed. Cutoff improvements are directionally right on every measure but
individually within noise at gen_n=48.

⚠️ **"The fix eliminated collapses" would OVERSTATE it.** Pre-fix at THIS checkpoint was 4.2%,
not the 6-11% seen at other checkpoints — that comparison crossed checkpoints, runs AND M-RoPE
rates. The control is what caught it.

### ⭐⭐ BEST CHECKPOINT + DECODE: 12-arm post-fix sweep (2026-09-12, ESTABLISHED)

All arms post-sampler-fix, n=128 TF / gen_n=48, RAS w=10, each checkpoint evaluated at the rate
it was TRAINED at. `eval_output/world_voice_bestsweep/` + the four earlier post-fix arms.

| arm | EOV | coll% | hit% | dur_r | len_mean | len_min | adj_rep | ent |
|---|---|---|---|---|---|---|---|---|
| r6.0 ck60000 T=0.6 | 44 | 4.2 | 85.4 | 0.410 | 185.9 | 12 | 0.0435 | 9.878 |
| r6.0 ck60000 greedy | 48 | 14.6 | 72.9 | 0.624 | 165.4 | 7 | 0.0077 | 10.041 |
| r6.0 ck78000 T=0.6 | 41 | **0.0** | 75.0 | 0.644 | 205.4 | **75** | 0.0387 | 10.027 |
| r6.0 ck78000 greedy | 47 | 2.1 | **89.6** | 0.674 | 177.3 | 28 | 0.0109 | 10.006 |
| r6.0 ck90000 T=0.6 | 43 | 4.2 | 62.5 | 0.480 | 198.0 | 33 | **0.0598** | 9.760 |
| r6.0 ck90000 greedy | 48 | 8.3 | 81.2 | 0.562 | 170.0 | 11 | 0.0113 | 10.061 |
| **r7.5 ck78000 T=0.6** | **47** | **0.0** | **83.3** | **0.702** | 192.0 | 49 | 0.0362 | 10.056 |
| r7.5 ck78000 greedy | 47 | 4.2 | **89.6** | 0.741 | 178.4 | 17 | 0.0097 | 10.090 |
| r7.5 ck90000 T=0.6 | 45 | 4.2 | 72.9 | 0.516 | 200.8 | 35 | 0.0386 | 10.010 |
| r7.5 ck90000 greedy | 46 | 2.1 | 83.3 | 0.746 | **179.67** | 8 | 0.0129 | 10.025 |
| r7.5 ck100000 T=0.6 | 42 | 4.2 | 75.0 | 0.332 | 196.4 | 30 | 0.0473 | 9.917 |
| r7.5 ck100000 greedy | 47 | 4.2 | 83.3 | 0.458 | 177.2 | 28 | 0.0135 | 10.020 |
| GT | — | — | — | 0.744 | 179.67 | 74 | 0.0792 | 10.398 |

⭐ **THE PICK: `mrope75_0/checkpoint-78000` with T=0.6 + RAS w=10.** 0.0% collapsed, len_min 49,
hit 83.3%, dur_r 0.702. Use **greedy** on the same checkpoint only if length-TRACKING matters
more than cutoffs (hit 89.6%, dur_r 0.741, but 4.2% collapsed and len_min 17).

⭐ **A CLEAN DECODE TRADE, consistent across all 6 pairs.** greedy+RAS: higher hit rate and
duration r, LOWER len_min, MORE collapses, adj_repeat ~0.01. T=0.6: fewer collapses, higher
len_min, adj_repeat ~0.04 (closer to GT's 0.0792), lower hit rate and dur_r. Greedy optimises
length-tracking; T=0.6 optimises not-cutting-off. There is no arm that wins both.

⭐⭐ **78k is the free-running optimum in BOTH runs** — the only two arms at 0.0% collapsed are
r6 ck78000 T=0.6 and r7.5 ck78000 T=0.6. **Eval metrics peak at 90k, so selecting a checkpoint
by eval CE overshoots the generation optimum by ~12k steps.** Fourth instance today of an effect
visible only in free-running.

⭐ **RATE 7.5 DOES BEAT 6.0 — amends the "bought nothing" verdict.** That call rested on eval CE
and unit accuracy being identical to five decimals; they were, and they were blind to this. At
matched steps, four comparisons all favour 7.5: hit 83.3 vs 75.0 (2.4 sd) and dur_r 0.702 vs
0.644 (2.1 sd) at 78k; hit 72.9 vs 62.5 (3.1 sd) and dur_r 0.516 vs 0.480 (1.3 sd) at 90k.
None decisive alone; the consistency across two steps and two metrics is the evidence.

⚠️ **Do not over-read two numbers.** r7.5 ck90000 greedy hit len_mean 179.6667 against GT's
179.6667 and dur_r 0.746 against the 0.744 "ceiling" — that ceiling is GT's OWN correlation at
n=48, so exceeding it is chance, not a result.

⚠️ **hit rate and duration r DISAGREE about the optimum.** Hit rate declines monotonically over
training (r6: 85.4 -> 75.0 -> 62.5) while dur_r, collapses and len_min peak at 78k. They measure
different things: hit rate is per-utterance "within +-30% of GT length", dur_r is whether length
TRACKS THE TEXT across utterances. ck60000 produces plausible lengths that are not text-driven
(hit 85.4 with r 0.410 and len_min 12). For cutoffs, read collapsed% and len_min.

### mrope75 run: the rate fix bought NOTHING, and 100k is worse than 78k (2026-09-12, ESTABLISHED)

From-scratch run at the corrected `--mrope_voice_rate 7.5`, 100k steps, otherwise identical to
the rate-6.0 run. **Statistically identical to the baseline — the same numbers to five decimals:**

| | rate 6.0 | rate 7.5 |
|---|---|---|
| eval/loss @90k | 0.41527 | 0.41527 |
| eval/loss @100k | 0.41566 | 0.41566 |
| unit acc @90k | 0.19096 | 0.19093 |
| unit acc @100k | 0.19073 | 0.19073 |

⭐ **A 25% M-RoPE rate miscalibration is fully absorbed by training.** Predicted by the
position-resolved data (text_delta does not decay with position); now confirmed end-to-end.
Both runs also overfit at the SAME place — eval bottoms at 90k, turns by 100k, epoch 6.67.
**Use 7.5 for BISTREAM** (where the chunk ratio and positional rate are the same quantity and a
mismatch is structural, not absorbable), not as an improvement in its own right.

**Free-running degrades 78k -> 100k on BOTH decodes**, while teacher-forced accuracy stays flat
(acc_real 0.1832) — the overfit is visible only in generation:

| | EOV | coll% | hit% | dur_r | len_mean | len_min | adj_rep |
|---|---|---|---|---|---|---|---|
| ck78000 T=0.6+RAS | 47 | **0.0** | 83.3 | 0.702 | 192.0 | **49** | 0.0362 |
| ck78000 greedy+RAS | 47 | 4.2 | **89.6** | **0.741** | 178.4 | 17 | 0.0097 |
| ck100000 T=0.6+RAS | 42 | 4.2 | 75.0 | **0.332** | 196.4 | 30 | 0.0473 |
| ck100000 greedy+RAS | 47 | 4.2 | 83.3 | 0.458 | 177.2 | 28 | 0.0135 |
| GT | — | — | — | 0.744 | 179.67 | 74 | 0.0792 |

⭐⭐ **greedy+RAS REPLICATED as the better decode on an independent checkpoint**: +6.3 / +8.3
points of hit rate and +0.039 / +0.126 duration r at 78k / 100k. The earlier 1.9 sd result
survived replication, with a LARGER margin on the weaker checkpoint. At ck100000 it also
restores termination (47/48 vs 42/48) and pulls len_mean to 177.2 vs GT 179.67 (T=0.6: 196.4).
Mechanism is consistent: it bans the degenerate mode every step, and overfitting strengthens
that mode.

⚠️ **greedy+RAS does NOT fix early EOS.** Collapses stay 4.2% and len_min ~28 under both decodes
at ck100000. The owner's report of clips cutting off abruptly is real at that checkpoint and is
NOT a sampling artifact. ck78000 T=0.6 remains the only arm measured at 0.0% collapsed.

⚠️ ck100000's duration r of 0.332 is 13 sd below ck78000's against a 0.027 within-checkpoint
noise floor, and far outside the 0.565-0.707 range the rate-6.0 run showed across late
checkpoints. Direction is corroborated by the greedy arm; treat the MAGNITUDE as unreplicated.

### ⭐⭐ The argmax IS the repeat: greedy collapses, greedy+RAS is the best decode (2026-09-11)

ck78000, n=48, same protocol as the sampler map below.

| | hit% | dur_r | len_mean | len_min | adj_rep | longest_run | entropy | bigram | coll% |
|---|---|---|---|---|---|---|---|---|---|
| **T=0 greedy, no RAS** | 45.8 | nan | 250.0 | 250 | **0.9932** | **249** | **1.59** | 0.0043 | 0.0 |
| **T=0 greedy + RAS** | **89.6** | **0.741** | **178.4** | 17 | 0.0097 | 7 | 10.090 | 0.8482 | 4.2 |
| T=0.6 + RAS (arm A) | 83.3 | 0.702 | 192.0 | **49** | **0.0362** | 10 | 10.056 | 0.8167 | **0.0** |
| GT | — | 0.744 | 179.67 | 74 | 0.0792 | 18 | 10.398 | 0.8694 | — |

⭐⭐⭐ **Pure greedy is total degeneration: 99.3% adjacent repeats, longest_run 249/250, and
42 distinct units across ALL 48 generations** (entropy 1.59 bits vs GT's 10.398). EOV never
fires; every output is exactly the 250-frame cap, which is why duration r is `nan` — zero
variance in length. The in-code note recorded 96% at step 44k on the old run; at 78k post-fix
it is 99.3%, so this is not a stage the model trains out of.

**The model's top-1 prediction in free-running is "repeat the previous unit", essentially
always.** Consequences, which reframe the whole sampler thread:
- **Sampling is LOAD-BEARING, not a refinement.** This model cannot generate at all without it.
- The drawl exists because repeat is the MODE — even sampled decoding dwells.
- RAS is worth +0.20 LCS because it explicitly bans that mode.
- The useful signal lives in the TAIL, which is why truncation hurts: nucleus at T=0.6 (arm D)
  cut away the part carrying the content and duration r collapsed to 0.435.
- Contrast teacher-forced `acc_real` 0.18: given GT history the top-1 is right 18% of the time,
  but conditioned on its own output the argmax degenerates completely. That gap is a sharper
  statement of the exposure problem than either scheduled-sampling arm produced.

⭐⭐ **Greedy + RAS is the best decode measured**: duration r **0.741 against the 0.744 GT
ceiling**, len_mean 178.4 vs GT 179.67, hit rate 89.6%. Mechanism: since the argmax is the
repeat ~99% of the time, RAS fires at nearly every step, making this "ban the mode, then
sample" at essentially every position. Compare T=1.0+RAS at 0.545 — there the initial pick is
itself a sample, so RAS only fires when that sample happens to repeat; greedy makes the ban
near-universal.

⚠️ **NOT yet adopted.** Margins over arm A are 1.9 sd (hit rate) and 1.4 sd (duration r) at
n=48 — exactly the size that fails to replicate — and it regresses on the cutoff axis
(4.2% collapsed, len_min 17, vs A's 0.0% and 49). It also still under-repeats badly
(adj_repeat 0.0097 vs GT 0.0792). **Replicate (>=3 arms) and confirm by ear before switching.**

**Sampler map at ck78000 (all post-fix), n=48 — a 2x2 of temperature x nucleus, plus floors:**

| | hit% | dur_r | len_mean | len_min | adj_rep | entropy | bigram | coll% |
|---|---|---|---|---|---|---|---|---|
| **A: T=0.6, RAS** | **83.3** | **0.702** | 192.0 | 49 | **0.0362** | 10.056 | 0.8167 | 0.0 |
| D: T=0.6 + nucleus | 72.9 | 0.435 | 190.4 | 66 | 0.0554 | 9.694 | 0.7597 | 2.1 |
| E: T=1.0 | 75.0 | 0.545 | 183.9 | 23 | 0.0044 | **10.939** | 0.9517 | 4.2 |
| C: T=1.0 + nucleus + floor 5 | 81.2 | 0.601 | 188.0 | 66 | 0.0096 | 10.334 | 0.8836 | 0.0 |
| B: T=0.6 + floor 3 | 81.2 | 0.583 | 200.1 | 55 | 0.0414 | 9.967 | — | 2.1 |
| GT | — | 0.744 | 179.67 | 74 | 0.0792 | 10.398 | 0.8694 | — |

⭐ **Entropy orders by TOTAL concentration**, temperature and nucleus being two of the same kind
of knob: D (both) 9.69 < A (temp only) 10.06 < C (temp, offset by truncation) 10.33 < E
(neither) 10.94. **T=1.0 alone OVERSHOOTS ground truth** — entropy 10.94 vs 10.40, bigram 0.952
vs 0.869, adj_repeat 0.0044 vs 0.0792. It is not diverse, it is noisy. Nucleus in arm C is
CORRECTING that overshoot, not contributing anything of its own: bolted onto T=0.6 (arm D) the
same truncation over-concentrates and duration r collapses to 0.435.

**KEEP A: T=0.6 + RAS, no nucleus, no floor.** Its duration-r margin over C is +0.101 = ~3.7 sd
against the measured 0.027 — the only difference in the table that clearly clears noise. A also
has the closest adj_repeat to GT of all five. Its one real weakness is being slightly
UNDER-diverse (entropy 10.06 vs 10.40); C fixes that but pays 3.7 sd of duration conditioning.

**Untested and cheap:** T=0.7-0.8 with no nucleus. Temperature moves entropy monotonically here
(0.6 -> 10.06, 1.0 -> 10.94, GT 10.40), so ~0.7 should land on GT's entropy without the
truncation that wrecked D. Strictly better than A if duration r holds.

**(superseded framing) Arm comparison at ck78000 (all post-fix), n=48:**

| arm | EOV | coll% | hit% | dur r | len_mean | len_min | adj_rep | entropy |
|---|---|---|---|---|---|---|---|---|
| **A: T=0.6 RAS, no floor** | 47 | 0.0 | **83.3** | **0.702** | 192.0 | 49 | **0.0362** | 10.056 |
| B: T=0.6 RAS, ratio 3 | 42 | 2.1 | 81.2 | 0.583 | 200.1 | 55 | 0.0414 | 9.967 |
| C: full CV2 parity | 45 | 0.0 | 81.2 | 0.601 | 188.0 | 66 | 0.0096 | 10.334 |
| GT | — | — | — | 0.744 | 179.67 | 74 | 0.0792 | 10.398 |

**Keep A: T=0.6 + RAS, no nucleus, no floor.** The EOS floor is REDUNDANT once the resample
scaling is fixed — there is no ratchet left to prevent, and it only adds length and defers
termination. Full CV2 parity (C) buys a tighter length distribution and higher entropy but
loses duration correlation (0.601 vs 0.702) and over-suppresses repetition (0.0096 vs GT
0.0792). CV2's config is tuned for their model and their 3:1 rate; the parts that were
unambiguously BUGS are fixed, and the rest are tuning choices our measurements do not favour.

⚠️⚠️ **`early_acc_real`'s noise floor is CHECKPOINT-SPECIFIC — do not reuse the +-0.0005
figure.** Across four ck78000 arms that CANNOT affect a teacher-forced metric it read
0.2969 / 0.2988 / 0.2998 / 0.3027 — spread **0.0058**, against the 0.0009 range measured at
ck60000. `acc_real` over the same four arms was 0.1825 / 0.1822 / 0.1827 (stable, consistent
with its 0.0002 sd). early_acc_real is computed over a much smaller position subset and its
variance is not constant across checkpoints. **Re-measure it at the checkpoint in question
before quoting an effect size** — this affects the pending mrope75-vs-baseline comparison.

### Cadence decomposed: the drawl is uniform, not a termination artifact (2026-09-09, ESTABLISHED)

First WER/LCS measurement on the LibriHeavy corpus. **ck90000**, n=64, T=0.6, both RAS settings.
`eval_output/world_voice_cadence/`. New metric `frames_per_hyp_word` in `cosyvoice_wer_eval.py`
divides rendered frames by ASR-transcribed words and compares against the CEILING row, which
isolates speaking rate from content loss — `hyp_ref_word_ratio` counts words and `len_mean`
counts frames, and NEITHER can see cadence.

| | RAS 0 (the TB-render default) | RAS 10 | ceiling (GT units) |
|---|---|---|---|
| **cadence** | **1.63x** (16.5 f/word, 0.659 s) | **1.01x** (10.2 f/word) | 10.1 f/word |
| hyp/ref words | 0.76 | 0.99 | 0.98 |
| LCS-recall | 0.5004 | 0.6862 | 0.8576 |
| WER | 0.5153 | 0.3735 | 0.1520 |
| CER | 0.3993 | 0.2354 | 0.0668 |

⭐ **Without RAS the model speaks 63% slower per word AND emits 24% fewer words.** Duration goes
into drawl rather than content. RAS removes essentially all of it: cadence 1.01x, word count 0.99.

⭐⭐ **The drawl is NOT a termination artifact.** Bucketing by generated/GT length ratio (the
owner's observation: "even well-conditioned examples with the RIGHT length say only a portion"):

| len ratio | n | LCS | frames/word | hyp/ref w |
|---|---|---|---|---|
| RAS 0, <0.7 | 3 | 0.044 | 19.5 | 0.32 |
| **RAS 0, 0.7-1.3** | **34** | **0.555** | **15.0** | **0.77** |
| RAS 0, 1.3-2.0 | 24 | 0.497 | 17.8 | 0.79 |
| RAS 0, >2.0 | 3 | 0.375 | 31.6 | 0.75 |
| **RAS 10, 0.7-1.3** | **49** | **0.754** | **9.7** | **1.03** |
| CEILING | 64 | 0.859 | 10.1 | — |

Renders that land at CORRECT duration still deliver only 56% of reference words at 1.5x the
ceiling's frames each. Length ratio barely predicts coverage (0.555 at 0.7-1.3 vs 0.497 at
1.3-2.0), so over-running is a SYMPTOM of the drawl, not its cause. Fixing termination would
not fix this.

⚠️ **`--viz_voice_ras_win` defaults to 0, so every ear-judged render since the corpus switch has
been the 1.63x-drawl arm.** In the like-for-like length bucket RAS is worth +0.20 LCS
(0.555 -> 0.754). This flag is the largest single gap between model capability and what gets
heard. Pass `--viz_voice_ras_win 10` on every run.

**The residual deficit, cleanly isolated:** with RAS and correct length, LCS 0.754 vs the
ceiling's 0.862. Cadence and duration are both accounted for; the remaining 0.11 is content
binding. That is a much narrower target than "text conditioning is weak".

**Dead end recorded so the family is not retried:** unit-space smoothness metrics cannot
diagnose articulation on this codebook. Measured on real GT speech, adjacent-frame cosine is
mean +0.094 / median +0.024 against +0.016 for random same-utterance pairs, and the fraction of
adjacent pairs above cosine 0.5 (0.0763) EQUALS the fraction above 0.2 — nothing lies between.
Adjacent units are either identical or orthogonal, so there is no "acoustically nearby unit"
and any cosine-threshold transition rate reduces to `1/(1-adj_repeat_rate)`. Matches the earlier
Mimi result that its codebook is not locally smooth. ASR is the only route to cadence here.

### Scheduled sampling is a net negative at BOTH p=0.25 and p=0.5 (2026-09-07/08, ESTABLISHED)

Two doses, each 10k steps from the control's **checkpoint-50000** to 60000, every other flag
byte-identical, 2 diagnostic replicates each, quoted against the CE control's 4 replicates.
Both verified live (`train/scheduled_sampling_prob` = 0.250 / 0.500 from step 50000, ramp 0).
`eval_output/world_voice_ss_ab/`, `eval_output/world_voice_ss025_ab/`.

| metric | CTL | sd | p=0.25 | sd | p=0.50 | sd |
|---|---|---|---|---|---|---|
| acc_real | 0.1799 | 0.0002 | 0.1763 | **-15.1** | 0.1709 | **-37.8** |
| early_acc_real | 0.3100 | 0.0005 | 0.3076 | **-4.7** | 0.2891 | **-40.4** |
| early_acc_shuffled | 0.2773 | 0.0018 | 0.2710 | -3.6 | 0.2515 | -14.6 |
| duration r | 0.6895 | 0.0271 | 0.5825 | **-3.9** | 0.4565 | **-8.6** |
| len_mean | 170.95 | 1.244 | 168.70 | -1.8 | 152.08 | **-15.2** |
| hit rate % | 81.23 | 3.389 | 80.20 | -0.3 (ns) | 78.15 | -0.9 (ns) |
| EOV /48 | 47.5 | 0.577 | 48.0 | +0.9 (ns) | 47.5 | 0.0 |
| collapsed % | 8.83 | 2.642 | 9.35 | +0.2 (ns) | 11.45 | +1.0 (ns) |

⭐ **The DOSE hypothesis was half right and it does not rescue the direction.** The trade does
improve at lower p — real-accuracy lost per unit of crutch removed is **0.39 at p=0.25 vs 0.81
at p=0.50** — but the absolute result is negative at BOTH. p=0.25 is worse than control on
every metric that moved. Lowering p shrinks both effects together and converges on the control;
there is no tested dose where the crutch reduction converts into a gain.

**The mechanism does fire at both doses**: early_acc_shuffled falls 3.6 / 14.6 sd, so the model
really can predict less from voice history alone. It just never turns that into text use within
this window.

⚠️ **Alternative explanation NOT excluded: 10k steps may be too short.** Scheduled sampling is
normally applied across a long run, not as a short finetune; the model has to re-learn
prediction from a corrupted-history regime, and at 10k the disruption may simply dominate any
eventual payoff. These arms establish that SS does not help IN A 10k WINDOW from a competent
checkpoint. They do not establish that SS is useless in a from-scratch run. OPEN.

⭐⭐ **SECOND confirmation of the `early_text_delta` trap.** It reads **0.0366 (p=0.25) and
0.0376 (p=0.50) against the control's 0.0327 — it would score BOTH arms as wins**, because in
each case the shuffled baseline fell further than early_acc_real. Two independent runs, same
inversion. Use `early_acc_real`.

**Consequence for on-policy KD:** it REQUIRES `--voice_scheduled_sampling_prob > 0`, so it
inherits a 4.7-40 sd onset penalty before the teacher contributes anything — and the teacher
already measured as the student's PEER (0.1771 vs 0.1799) and lost off-policy. Running it now
would stack a negative on a negative. Not recommended without a reason to expect the teacher to
overcome the SS cost.

---

### (superseded) Scheduled sampling at p=0.5: NET NEGATIVE (2026-09-07, ESTABLISHED)

Arm A of the on-policy plan — exposure-bias training with NO teacher, to establish whether
scheduled sampling alone buys anything before paying for distillation. Resumed from the
control's **checkpoint-50000**, 10k steps to 60000, `--voice_scheduled_sampling_prob 0.5
--voice_scheduled_sampling_ramp_steps 0 --voice_scheduled_sampling_start_step 50000`, every
other flag byte-identical. Control = the CE run's own 50k->60k segment, which has FOUR
diagnostic replicates, so effects are quoted in measured sd. Reports:
`eval_output/world_voice_ss_ab/` (2 replicates).

**Verified live:** `train/scheduled_sampling_prob` = 0.500 from step 50000 (ramp 0 working).

| metric | CTL (n=4) | sd | SS (n=2) | delta | sd units |
|---|---|---|---|---|---|
| acc_real | 0.1799 | 0.0002 | 0.1709 | -0.0089 | **-37.8** |
| early_acc_real | 0.3100 | 0.0005 | 0.2891 | -0.0210 | **-40.4** |
| early_acc_shuffled | 0.2773 | 0.0018 | 0.2515 | -0.0259 | -14.6 |
| duration r | 0.6895 | 0.0271 | 0.4565 | -0.2330 | **-8.6** |
| len_mean | 170.95 | 1.244 | 152.08 | -18.86 | **-15.2** |
| hit rate % | 81.23 | 3.389 | 78.15 | -3.08 | -0.9 (ns) |
| EOV /48 | 47.5 | 0.577 | 47.5 | 0.00 | 0.0 |
| adj_repeat | 0.0106 | 0.0012 | 0.0101 | -0.0005 | -0.4 (ns) |
| collapsed % | 8.83 | 2.642 | 11.45 | +2.63 | +1.0 (ns) |

Teacher-forced eval agrees: eval/loss 0.4358 vs the control's 0.4193, unit accuracy 0.1797 vs
0.1881 — and SS is worse than its OWN 50k starting point (0.4226 / 0.1859), rising to a plateau
by 58k.

⚠️ **The apparent wins are under-generation artifacts.** Output is far too SHORT (len_mean 152
vs GT 179.7), so EOV 48/48 with 0 budget-capped, low adj_repeat and high distinct-bigram all
follow trivially from stopping early. Do not read them as quality.

⭐⭐ **CONCRETE CASE FOR `early_acc_real` OVER `early_text_delta`.** The delta reads **+0.0376
for SS vs +0.0327 for the control — it would call this run a WIN.** It rises only because the
shuffled baseline fell FURTHER (-0.0259) than early_acc_real did (-0.0210). Both terms dropped;
the ratio improved while the model got worse. Any conclusion drawn from the delta alone here
would have been backwards. This is the failure mode the noise-floor entry predicted.

**Not a pure loss — the mechanism SS targets did fire.** early_acc_shuffled falling 14.6 sd
means the model can predict materially LESS from voice history alone, i.e. the AR crutch really
was weakened. It just lost more absolute capability than it converted into text use. That points
at DOSE, not direction: p=0.5 with no ramp removes too much history at once. **Untested: p=0.25,
and any p with a ramp.** p=0.5/no-ramp was chosen deliberately (ramp premise — "early
predictions are noise" — does not hold at a competent 50k checkpoint) and that reasoning still
stands; the magnitude was the error.

**Status of the exposure-bias programme: two arms, both negative.** Off-policy KD at 0.3 and
scheduled sampling at 0.5 each made things worse from the same ck50000 start. On-policy KD
(scheduled sampling + teacher, `--voice_onpolicy_distill`) remains untested, but it inherits
whatever dose problem this arm has, since it REQUIRES `--voice_scheduled_sampling_prob > 0`.
Fix the dose before spending a run on it.

### MEASURED noise floor of the AR diagnostics probe (2026-09-06, ESTABLISHED)

**n=4 runs of ck60000 with a byte-identical command** (`eval_output/world_voice_noise_floor/`
plus the original `CTL_ras10`). Nothing varies but RNG/GPU non-determinism. Quote effect sizes
against THESE numbers, not against the bootstrap CIs, which are marginal and utterance-dominated.

| metric | mean | **sd** | range |
|---|---|---|---|
| acc_real | 0.1799 | **0.0002** | 0.0005 |
| early_acc_real | 0.3100 | **0.0005** | 0.0009 | ⚠️ ck60000-SPECIFIC; ck78000 gave spread 0.0058 — see the sampler-alignment entry |
| early_acc_shuffled | 0.2773 | 0.0018 | 0.0039 |
| early_text_delta | 0.0327 | **0.0023** | 0.0048 |
| duration r | 0.6895 | **0.0271** | 0.0600 |
| length hit rate % | 81.23 | **3.39** | 8.30 |
| len_mean | 170.95 | 1.24 | 2.73 |
| adj_repeat_rate | 0.0106 | 0.0012 | 0.0026 |

⭐⭐ **USE `early_acc_real`, NOT `early_text_delta`, TO COMPARE CHECKPOINTS.** The delta is 4.6x
noisier (sd 0.0023 vs 0.0005) for one reason: the shuffled baseline it subtracts has sd 0.0018.
That baseline is a CONTROL that should not vary with the model, so at fixed protocol subtracting
it adds noise without adding information. Keep the delta only to establish that text matters at
all. This is a free 4.6x precision gain — the 60k->100k onset decline reads **15 sd** on
early_acc_real and would read ~3 sd on the delta.
(Better still: seed the ablation shuffle, or average several shuffles, and the delta becomes
usable too. Currently unseeded.)

**Effect sizes against this floor** — every prior conclusion survives:

| claim | effect | sd |
|---|---|---|
| 100k onset decline vs 60k (early_acc_real) | 0.0078 | **15.0** |
| KD duration r deficit | 0.208 | **7.7** |
| ck90000 duration r drop | 0.142 | **5.2** |
| KD early_text_delta deficit | 0.0112 | **4.9** |

⚠️ **`length hit rate` is the WEAKEST generation metric** (sd 3.39 points): 81.2 vs 68.8 at 90k
is 3.7 sd (real), but 81.2 vs 72.9 at 100k is only 2.4 sd (marginal). Do not read hit-rate
differences under ~7 points. `len_mean` sd 1.24 and `adj_repeat` sd 0.0012 are both usable.

**Method note:** two samples are NOT a noise floor. The +-0.0010 figure first reported for
early_text_delta came from two arms that happened to land close; the third and fourth samples
were 4-5x further apart. n>=4 for any reproducibility claim.

### Late-checkpoint generation check: the overfit shows at the ONSET (2026-09-06, ESTABLISHED)

ck90000 and ck100000 run at protocol byte-identical to the ck60000 control arm
(`eval_output/world_voice_late_ckpt/`, n=128 TF / gen_n=48, T=0.6, RAS w=10).

| | 60k | 90k | 100k |
|---|---|---|---|
| acc_real (TF, val) | 0.1799 | 0.1816 | **0.1859** |
| early_acc_real | **0.3105** | 0.3086 | **0.3027** |
| early_acc_shuffled | 0.2764 | 0.2725 | 0.2734 |
| early_text_delta | +0.0342 | +0.0361 | +0.0293 |
| EOV fired /48 | 47 | 47 | 43 |
| budget-capped /48 | 1 | 1 | 5 |
| text->length r | 0.707 | 0.565 | 0.676 |
| length hit rate | 81.2% | 68.8% | 72.9% |

⭐ **acc_real RISES monotonically while early_acc_real FALLS monotonically.** Overall next-unit
prediction improves (0.1799 -> 0.1859) while the utterance ONSET — the one place with no voice
history, where text must carry the prediction — degrades (0.3105 -> 0.3027). The model is
shifting weight OFF the text and ONTO the local unit prior. That is what the train/eval gap
looks like in the place it hurts most, and it is invisible in aggregate accuracy, which
improves throughout.

**Generation did NOT collapse**: repetition, entropy and n-gram diversity steady; ck100000
len_mean 179.56 vs GT 179.67. Termination mildly worse (43/48, 5 capped, vs 47/48, 1).

~~**GENERATION METRICS AT gen_n=48 ARE TOO NOISY TO READ SMALL DIFFERENCES** — claimed
2026-09-06 from the r sequence being non-monotonic (0.465/0.595/0.707/0.565/0.676), asserting
a ~+-0.1 spread and that differences below ~0.15 were uninterpretable; and that this weakened
the generation half of the KD verdict.~~ **RETRACTED the same day by direct measurement.**
n=4 identical runs of ck60000 give duration r **sd 0.0271** (not +-0.1) — see the noise-floor
entry below. The non-monotonic sequence is REAL SIGNAL at 4-8x the noise band. The ck90000 drop
is 5.2 sd and the KD r deficit is 7.7 sd; both stand. The error was inferring a noise magnitude
from non-monotonicity instead of measuring it.

### The 100k run DID overfit — flat eval was only half the signature (2026-09-06, ESTABLISHED)

`..._ar_flat_lr_ema_0` completed to step 100000 = **epoch 6.67** on the 1985 h corpus.
Eval alone looks merely FLAT, which is why this was nearly missed; the train side is what
identifies it. Train metrics are per-batch and swing 0.14-0.29 between adjacent points — they
are meaningless pointwise and MUST be binned.

| window | train acc | eval acc | gap | train CE | eval CE | gap |
|---|---|---|---|---|---|---|
| 40-50k | 0.1810 | ~0.1846 | -0.0036 | 0.4285 | ~0.4243 | +0.0042 |
| 50-60k | 0.1841 | ~0.1868 | -0.0027 | 0.4230 | ~0.4212 | +0.0018 |
| 60-70k | 0.1913 | ~0.1885 | +0.0028 | 0.4149 | ~0.4187 | -0.0038 |
| 70-80k | 0.1936 | ~0.1899 | +0.0037 | 0.4125 | ~0.4170 | -0.0045 |
| 80-90k | 0.1938 | ~0.1902 | +0.0036 | 0.4108 | ~0.4164 | -0.0056 |
| 90-100k | **0.1989** | ~0.1904 | **+0.0085** | **0.4044** | ~0.4162 | **-0.0118** |

⭐ **The train/eval gap CROSSES OVER at ~60k and then widens ~3x.** In the final window train
improved faster than in the two before it (+0.0050 acc, -0.0064 CE) while eval went BACKWARDS
(acc 0.19096@90k -> 0.19073@100k; CE 0.41526 -> 0.41564). Flat eval + still-descending train =
overfitting; neither half alone is diagnostic.

⚠️ **The gap is UNDERSTATED by this table.** Eval runs on EMA weights (`training.py:2257`)
while train logs RAW, and EMA measured ~+0.004 accuracy BETTER than raw at ck40000. On matched
weights the final-window gap is nearer +0.012. Any future train-vs-eval comparison in this repo
has the same bias and must correct for it.

**Keeper = checkpoint-90000, not 100000**: best eval accuracy (0.19096) and lowest eval CE
(0.41526), and it precedes the window where train pulled away hardest.

**Epoch count, not hours, predicts the wall.** The 146 h corpus overfit at ~epoch 8; this
1985 h corpus (13.6x the data) overfits at ~epoch 6.7. Scaling data 13.6x did NOT buy 13.6x the
steps — it bought a better model at a similar epoch budget. Plan future runs in EPOCHS.

**Accuracy trajectory for reference** (eval, EMA): 0.1202@4k, 0.1685@20k, 0.1830@40k,
0.1881@60k, 0.1895@72k, 0.1910@90k, 0.1907@100k. A 40k->100k stretch bought +0.0077.

OPEN: whether the overfit is visible in GENERATION. Teacher-forced eval has dissociated from
free-running repeatedly in this direction (see the plain-sampling saturation note), so a
ck100000-vs-ck60000 diagnostics run at matched protocol is the outstanding check.

### Off-policy CosyVoice 2 KD at weight 0.3: NET NEGATIVE (2026-09-04, ESTABLISHED)

Falsification run as designed: `..._distill_0` resumed from the control's **checkpoint-50000**
and ran 10k steps to 60000 with `--voice_distill_weight 0.3 --voice_distill_temperature 2.0`,
EVERY other flag byte-identical. The control's own 50k->60k segment is therefore a perfectly
matched baseline (same start, same steps, same flat 1e-4 — verified flat across all 100 logged
LR points). Reports: `eval_output/world_voice_distill_ab/`.

**Distillation was verifiably LIVE** — `voice_distill_kl` fell 0.568 -> 0.402 and
`voice_distill_agreement` rose 0.515 -> 0.562 over the window. This is not an inert-flag result.

| | control 60k | distill 60k |
|---|---|---|
| acc_real (TF) | **0.1799** | 0.1769 |
| eval unit accuracy (TB) | **0.1881** | 0.1829 |
| early_acc_real | **0.3105** | 0.2988 |
| early_acc_shuffled | 0.2764 | 0.2764 (identical — no baseline confound) |
| early_text_delta | **+0.0342** | +0.0225 |
| text_delta all-pos | **+0.0653** | +0.0613 |
| EOV fired /48 | **47** | 40 |
| budget-capped /48 | **1** | 8 |
| length hit rate +-30% | **81.2%** | 70.8% |
| collapsed / overrun | 8.3% / 2.1% | 4.2% / **8.3%** |
| text->length r | **0.707** | 0.499 |

⭐ **Every text-conditioning measure moved the WRONG way**, plus accuracy, termination, duration
correlation and length hit rate. KD's only win (collapse 8.3% -> 4.2%) was bought by converting
truncations into overruns, so the net hit rate still fell.

⭐ **The deficit is ~12x the measurement noise floor.** Re-running the control at the SAME
checkpoint in a second arm gives +-0.0010 — **but that was two samples that happened to land
close; the measured n=4 sd is 0.0023, so the KD gap is 4.9 sd, NOT the "12x" first claimed.**
Corrected 2026-09-06 (see the noise-floor entry). The four arms:

| arm | acc_real | early_acc_real | early_acc_shuffled | early_text_delta |
|---|---|---|---|---|
| CTL_ras10 | 0.1799 | 0.3105 | 0.2764 | +0.0342 |
| CTL_noras | 0.1800 | 0.3086 | 0.2754 | +0.0332 |
| KD_ras10 | 0.1769 | 0.2988 | 0.2764 | +0.0225 |
| KD_noras | 0.1769 | 0.2979 | 0.2764 | +0.0215 |

Control spans [0.0332, 0.0342], KD spans [0.0215, 0.0225]. Against the MEASURED
early_text_delta sd of 0.0023 the 0.0112 gap is **4.9 sd** — solid, but a third the strength
originally claimed. On duration r the KD deficit is **7.7 sd**, so the generation half of the
verdict is in fact the STRONGER half, the opposite of what the retracted warning said.
**Do NOT read the overlapping bootstrap CIs as "not significant"**: those are MARGINAL CIs that
resample utterances and are dominated by per-utterance variance, whereas control and KD were
scored on the SAME 128 utterances, so the comparison is PAIRED and its error is the +-0.001
reproducibility, not the +-0.02 marginal interval. (A proper paired bootstrap over per-utterance
differences would be better still; the probe does not currently emit per-utterance values.)

**Why, and it was PREDICTED**: the open-questions section already argued "off-policy KD only
supervises states reachable from a perfect prefix — where this model is already near the
teacher. Its failure is off that manifold." Confirmed: `voice_distill_teacher_acc` = **0.1771**
against the student's 0.1799 — the teacher is a PEER on this data, not a superior — so
mode-covering forward KL spent capacity matching a distribution that was no better, and paid in
sharpness.

**Scope of the refutation:** off-policy KD, weight 0.3, T=2.0, 10k steps. Does NOT refute
on-policy distillation (`--voice_onpolicy_distill`, requires `--voice_scheduled_sampling_prob
> 0`) or a much smaller weight.

⚠️ **TEACHER-ACCURACY DISCREPANCY, UNRESOLVED.** The diagnostics report
`TEACHER acc_real (ceiling) 0.1283` while the training harness logs
`voice_distill_teacher_acc 0.1771` — same teacher, same val set, 38% apart. One path conditions
the teacher wrongly (the diagnostics' is the more suspect, since the harness number is what
actually shaped training). Any argument resting on the 0.1283 ceiling — including the
teacher-ceiling rows in every report above — should be held loosely until reconciled. OPEN.

**Control-only result worth keeping:** plain CE took duration r from 0.595 (50k) to **0.707**
(60k) against a 0.744 ceiling, with 47/48 terminating and an 81.2% length hit rate. Duration
conditioning is close to solved by CE alone.

**Probe change:** `world_voice_ar_diagnostics.py` now emits a per-utterance length hit rate
(fraction within +-30% of GT, plus collapsed/overrun splits). Added after `len_mean` was shown
to hide a bimodal distribution at ck50000; first emission verified here.

### ck50000 check: duration conditioning improves, content binding does NOT (2026-09-02, ESTABLISHED)

Protocol identical to the ck40000 arms (raw weights, n=128 TF / gen_n=48, T=0.6).
Reports: `eval_output/world_voice_ck50000/`.

| | 40k plain | 50k plain | 40k RAS | **50k RAS** | GT |
|---|---|---|---|---|---|
| ppl_real (TF) | 46.54 | 44.31 | — | — | — |
| early_text_delta | +0.0273 | **+0.0273** | +0.0264 | **+0.0254** | — |
| EOV fired /48 | 18 | 17 | 42 | **46** | — |
| budget-capped /48 | 30 | 31 | 6 | **2** | — |
| len_mean | 229.7 | 231.8 | 173.5 | **179.06** | 179.67 |
| text->length r | 0.076 | 0.073 | 0.465 | **0.595** | 0.744 |
| adj_repeat_rate | 0.4506 | 0.4247 | 0.0093 | 0.0145 | 0.0792 |

⭐⭐ **The dissociation, stated precisely: the model is learning HOW LONG to speak from the text
and NOT WHAT to say.** Duration r climbed 0.465 -> 0.595 in 10k steps (ceiling 0.744) and length
landed on GT; `early_text_delta` did not move at all (+0.0273 at BOTH 40k and 50k under plain).
Per the probe's own rubric, duration r is "structural conditioning, independent of content
alignment" — so structural conditioning is progressing while content binding is frozen.

⚠️ **PLAIN SAMPLING IS A SATURATED MEASUREMENT — do not judge progress from it.** 31/48 sit at
the 250-frame cap, so the metric cannot move even when the model improves. A same-session claim
that "free-running is not improving between 40k and 50k" was drawn from the plain arm and is
WRONG; the RAS arm shows clear improvement over the same interval. Always read free-running
progress from the RAS arm.

⚠️ **`len_mean` is a BAD summary here — the length distribution is BIMODAL.** 50k RAS len_mean
179.06 vs GT 179.67 looks exact, but the 8-utterance render shows 3/8 collapsing to 8, 4 and 22
frames (0.16-0.9 s) while the survivors run long; the two errors cancel in the mean. At 40k the
same 8 texts failed 2/8 at 62 and 39 frames — so the failures got MORE SEVERE even as aggregates
improved. (Rate is uncertain: the 48-sample arm reported len_min 13 and 46/48 EOV, so n=8 caught
an unlucky subset. The EXISTENCE of sub-25-frame collapses is not noise.) **A per-utterance hit
rate (fraction within +-30% of GT length) is the metric this needs; the probe does not yet
compute one.**

### EMA-vs-raw probe: noise reduction does NOT fix free-running (2026-09-02, ESTABLISHED)

ck40000 EMA shadow (decay 0.9995, ~2k-step horizon) materialized into a loadable checkpoint
(`scripts_local/materialize_ema_checkpoint.py`) and run through the SAME arm-A regime
(T=0.6, RAS off). EMA = "same weights, less optimizer noise", so this is a cheap proxy for
what an LR anneal would buy. Report: `eval_output/world_voice_ema_probe/EMA_t06_noras/`.

| | raw | EMA | RAS (arm B) | GT |
|---|---|---|---|---|
| acc_real (TF) | 0.1769 | **0.1810** | 0.1765 | — |
| ppl_real (TF) | 46.54 | **42.90** | — | — |
| EOV fired /48 | 18 | **18** | 42 | — |
| budget-capped /48 | 30 | **30** | 6 | — |
| len_mean | 229.7 | 233.9 | 173.5 | 179.7 |
| adj_repeat_rate | 0.4506 | 0.3709 | 0.0093 | 0.0792 |
| longest_run | 179 | 156 | 4 | 18 |
| early_text_delta | +0.0273 | +0.0254 | +0.0264 | — |

⭐ **Termination is IDENTICAL (18/48 EOV, 30 capped) and len_mean is slightly WORSE.** Removing
optimizer noise buys ~1/5 of the repetition gap (0.451 -> 0.371, still 4.7x GT) and NONE of the
termination gap. Teacher-forced quality does improve (ppl 46.5 -> 42.9, acc +0.004).

**Consequence for the LR anneal:** expect it to buy MODEL QUALITY (loss/accuracy), NOT a fix for
the dragging/termination. An earlier same-session claim that annealing would likely fix the
free-running pathology (reasoning from `project_world_tts_lr_volatility`, "late low-LR ckpts
free-run clean") is WEAKENED by this. Not RETRACTED, because EMA's 2k-step horizon is a weak
proxy for a full 1e-4->0 decay and the tension may be era-specific — but do not plan the decay
expecting it to fix free-running. OPEN.

⭐⭐ **`early_text_delta` is invariant to EVERYTHING tried so far**: +0.0273 raw / +0.0254 EMA /
+0.0264 RAS / +0.0264 nucleus — all CIs overlapping, against a teacher ceiling of +0.0538.
Not sampling, not noise reduction, not weight averaging. This is the load-bearing deficit and
only a training-side change addressing text->content binding can move it.

**Eval uses EMA weights** (`training.py:2257-2270`): `apply_shadow()` wraps `super().evaluate()`
with `restore()` in a finally, and the viz callback fires INSIDE that, so eval metrics AND TB
audio renders are EMA. The pre-eval checkpoint stays RAW so a resume cannot restart from
averaged weights. ⚠️ Consequence: `pytorch_model.bin` is RAW, so any eval script loading a
checkpoint directly is measuring DIFFERENT weights than the TB renders show.

**Training health at 40k:** eval/loss fell monotonically at all 19 eval points, 0.5822 -> 0.4268,
unit accuracy 0.1202 -> 0.1830. No overfitting anywhere — the 1985 h corpus removed the epoch-8
wall the 146 h corpus hit. Gains decelerate log-linearly (10k->20k +0.0227 acc; 30k->40k +0.0075).

## Reference numbers

Quote results against these rather than against adjectives.

**Pipeline ceiling** (GT units through the same frozen decoder + Whisper): LCS recall 0.8937,
WER 0.1056, CER 0.0416, hyp/ref 1.00. The decoder and ASR are not the limitation.

**Teacher ceiling** (CosyVoice 2's own speech LM, run on our val data, teacher-forced, n=1024,
CLEAN — no M-RoPE, no sampling): acc_real 0.1283 | text_delta all-pos +0.0594
[+0.0576,+0.0612] | early_text_delta +0.0538 [+0.0480,+0.0597] | text-attributed fraction
0.463. A model that solves this task sits here, not at 1.0.

**Text-free baselines for CosyVoice 2** (NOT the Mimi-era script defaults 0.211/0.229/0.117):
n-gram ceiling 0.0545, asymptote 0.2261, repeat crutch 0.0314.

**Decode variance floor:** std 0.0089, range 0.018 at n=64 (three seeds, same config). Any
single-run decode difference below ~0.025 means nothing.

---

## ESTABLISHED

### The architecture independently converged on CosyVoice 2's design
Their `Qwen2LM` bolts a new speech embedding table and a speech-only head onto a pretrained
Qwen2-0.5B and discards the text head; sos/task_id map to our BOV/BOV; **their eos token 6561
is exactly our EOV**; their loss is masked to speech positions, matching
`--mask_text_loss_in_synthesis`. Two real differences remain: (1) ~51x more English data,
(2) their backbone is FULLY FINE-TUNED while ours is a frozen SmolLM2 with a from-scratch
trunk asked to bridge into a token space the encoder was never adapted toward.

### The failure is text->content binding, not degeneration, sampling, or the token space
Three-way free-running comparison (student / teacher / GT) on identical degeneration stats
found the student indistinguishable from teacher and GT on every one: length ratio 1.21x vs
1.14x vs 1.0, adjacent repeat 0.0407 vs 0.0070 vs 0.0306, distinct bigram 0.943 vs 0.911 vs
0.958. It emits well-formed speech-like streams that say the wrong thing.

Corollary that has held up: **duration conditioning works while content conditioning does
not.** Text-length -> generated-length correlation was already +0.509 when content was near
chance. The text pathway is intact and carrying signal; the content mapping is what failed.

### Distillation from the CosyVoice 2 speech LM works (measured @44k)
Logit-KL on the teacher's 6564-way softmax at speech positions, weight 0.3, temperature 2.0,
teacher forward-only in the training loop. Vocab mapping is clean — slice teacher logits
`[0:6562]`; ids 0-6560 are content, 6561 is eos == our EOV.

| metric | nocurric @20k | distill @44k | teacher |
|---|---|---|---|
| early_text_delta | +0.0244 | **+0.0522** [+0.0461,+0.0583] | +0.0538 |
| text_delta all-pos | +0.0086 | +0.0349 | +0.0594 |
| text-attributed | 0.106 | **0.371** | 0.463 |
| acc_real | 0.0814 | 0.0940 | 0.1283 |

Early text conditioning reached the teacher (CIs overlap). Eval unit accuracy 0.0817 -> 0.0991.
**But it still sounded bad by ear at 44k** — the metrics above are teacher-forced while the
audio is free-running.

### The English data gap is 51x, and clean labels are not required
From the CosyVoice 2 paper, verified against its tables: the tokenizer saw 200,000 h; the TTS
LM saw ~167k h of which only **30k h is English** (78% is Chinese). Transcripts are
**pseudo-labels** — their pipeline runs Paraformer and SenseVoice over scraped audio, and the
FSQ tokenizer is explicitly "robust to data noise."

So the gap versus our 585 h is ~51x on English, not the 200x that gets quoted, and it is
reachable: LibriHeavy alone (~50k h labeled English) exceeds their English portion, MLS adds
~44k. Tokenizing ~50k h is roughly 36M utterances, ~12 GPU-days (about 3 days on 4 cards) and
~100 GB at ~3 KB/utt. This supersedes any note claiming voice datasets need clean transcripts.

### Removing a local crutch does not create text conditioning — measured twice
**2026-08-13, gen-query (Mimi/ContentVec features, homegrown SMG decoder).** Replacing the
trunk's AR voice input with learned per-position queries:

| checkpoint | early_acc_real | early_text_delta |
|---|---|---|
| baseline @100k | 0.321 | **+0.103** |
| gen-query @23k | 0.021 | +0.0156 |
| gen-query @35k | 0.016 | +0.0127 |
| gen-query @47k | 0.015 | **+0.0112** |

Declining across 24k steps, which is what ruled out "needs more steps." Mechanism: the coda's
`voice_coda_prev` rescued later frames, so aggregate accuracy looked near-baseline while the
early frames — the ones that cannot be crutched — collapsed to near chance.

**2026-08-21, NAR (CosyVoice 2 tokens, frozen decoder).** Same shape, different mechanism:
bidirectional inpainting replaced the AR crutch and proved to be a *better* one. See the NAR
entries below.

Also from the 2026-08-13 work, and still true: **baseline text conditioning is strong on early
frames (+0.103) and near zero late**, so any all-position average understates it. Report early
and late separately.

### M-RoPE: why the trunk has two position axes
Under a single global RoPE axis, aligning voice frame t with text token j ~= t/6 requires
relative offset `L_text + 0.83t` — large, growing with t, and different per utterance. RoPE
encodes differences, not ratios, so that relationship is expensive to represent. Measured
consequence: exactly ONE head of 72 (block 0, head 4) became a sharp monotonic aligner
(entropy 9% of uniform, correlation +0.85) and was load-bearing — ablating it cost acc -0.0069
and moved perplexity 142 -> 197, while three random heads cost ~0.

M-RoPE splits the rotary dims: a GLOBAL half increasing across the whole interleaved sequence
(needed so multiple media segments stay distinguishable) and a LOCAL half indexing within the
current same-modality segment, with one stream scaled by the frame rate so aligned pairs sit
at relative distance ~0. Flags: `--use_mrope --mrope_voice_rate 6.0 --mrope_scale_side
{voice,text}`. Off by default and byte-identical when off.

### Speaker identity is a partial, embedding-only conditioner (accepted limitation)
Decoding the SAME GT units with the correct versus most-dissimilar speaker moves median F0
only **36.3 Hz** (a real male/female swap is 60-90 Hz), and every swap moved pitch DOWN. So
campplus is a partial conditioner being out-voted by the decoder's lower-pitched prior.

Root cause: we supply only `flow_embedding`. Real CosyVoice 2 zero-shot also supplies
`flow_prompt_speech_token` and `prompt_speech_feat` — the ear-perfect clone test went through
`inference_vc` with all three. **The owner chose to stay embedding-only.** Expect drift to
improve as units get more on-manifold. Eval renders log `*_static_speaker_audio` siblings: if
the static render also drifts, it is the decoder's prior; if only the GT-speaker render
drifts, the embedding is not landing.

### NAR learned bidirectional inpainting instead of reading text (2026-08-21)
Masked-parallel (MaskGIT-style) voice, cosine mask schedule, step 23000, teacher-forced:

| mask ratio | acc_real | text_delta | text-attributed |
|---|---|---|---|
| 1.00 | 0.0169 | +0.0037 | 0.220 |
| 0.75 | 0.0491 | +0.0033 | 0.068 |
| 0.50 | 0.0701 | +0.0023 | 0.032 |
| 0.25 | 0.0860 | +0.0025 | 0.029 |

Revealing 25% of frames raises accuracy 5x and drops perplexity 20x (1832 -> 91) while the
text contribution FALLS. Context is worth ~28x more than the transcript.

### At matched step, AR reads text ~10x more than NAR (2026-08-21)
Both at step 23000, same data, same trunk config:

| | AR (full context) | NAR (r=0.25) |
|---|---|---|
| acc_real | 0.0887 | 0.0860 |
| ppl_real | 162.3 | 90.8 |
| text_delta | +0.0271 | +0.0025 |
| text-attributed | 0.305 | 0.029 |
| LCS recall (trunc) | 0.1620 | 0.0742 |

NAR predicts units as well as AR — better calibrated, even — and derives 3% of that from the
transcript against AR's 31%. **Removing the autoregressive crutch installed a better one.**

### The models are NOT mode-collapsed; repetition comes from self-conditioning (2026-08-21)
Direct measurement of the unit head's per-step predictive distribution
(`unit_confidence_probe.py`, n=256, max entropy 12.68 bits):

| | raw top1 | raw entropy | frac top1 > 0.9 |
|---|---|---|---|
| NAR r=1.0 | 0.0163 | 10.96 | 0.000 |
| NAR r=0.25 | 0.0862 | 8.01 | 0.000 |
| AR @23k TF | 0.1188 | 6.47 | 0.000 |

Sampling is nowhere near argmax anywhere. At 8.4 bits under T=0.6, independent draws would
collide on adjacent frames well under 1% of the time, yet observed adjacent repeat is ~0.6 —
so repetition is produced by conditioning on the model's own output, not by peaked
predictions. Both arms are also well calibrated: NAR at r=1.0 has mean top-1 probability
0.0163 against top-1 accuracy 0.0169. Not confidently wrong — accurately uncertain.

**Generating from text alone the model is near-uninformed:** 10.96 of 12.68 bits is 86% of
maximum, ~2500 effective choices out of 6562. This is a better progress metric than
`text_delta` — one number, no shuffled-text control, known floor (AR reaches 6.47).

### The duration token solves length; AR fails on long utterances specifically (2026-08-21)
32 log-spaced buckets over [25,250] frames, emitted between BOV and the voice placeholder,
predicted from the BOV hidden state (the last position with full causal multimodal context and
the last before gen queries must be allocated). Rides the existing trainable
`special_embed`/`special_head` extension, so the frozen LLM is untouched.

| | NAR + duration token | AR |
|---|---|---|
| text-len -> gen-len r | +0.933 (ceiling 0.960) | +0.624 |
| hyp/ref words | 1.00 | 0.58 |
| duration bucket +-1 | 0.538 | — |

AR's failure is not uniform: short utterances land near GT; every long one (GT 154-181) runs
to the 250-frame cap without firing EOV. NAR produced 183/196/196 on those same three.

Caveats: the model fills the allocated block and never fires EOV, so length is capped by
duration-head accuracy with no self-correction, and a wrong bucket halves intelligibility
(0.0742 -> 0.0358).

### LCS recall is partly a length metric (2026-08-21)
When two checkpoints fail on length in opposite directions it prefers the verbose one. The AR
arm at 23k over-runs (167/250/472 frames) and scores 0.1620; the same arm at 44k collapses to
near-empty (3-43 frames) and scores 0.0949. That gap is termination behaviour, not quality.
Always read it beside hyp/ref.

---

## CONTAMINATED — pre-2026-08-21 numbers, and which to trust

Two bugs found on 2026-08-21 invalidate large classes of earlier measurement. Before citing
any pre-08-21 number, check it against this list.

**VOID — every post-hoc diagnostic run on an M-RoPE checkpoint.** `load_world_model` never
re-enabled M-RoPE and `load_model` runs `strict=False`, so the trunk silently ran single-axis
RoPE with sequential positions. This voids: the scale_side ablation (BOTH arms), all M-RoPE
WER/LCS figures, the position-resolved horizon, and the per-head attention result. Fixed by
detecting `use_mrope` from the weights and requiring `--mrope_scale_side`.

**WRONG REGIME — every free-running number from the eval scripts.** They hardcoded
`voice_temperature=1.0` while the training viz renders at 0.6. Free-running statistics
described a regime nobody listens at. Fixed; default is now 0.6.

**CLEAN — teacher-forced metrics on non-M-RoPE checkpoints.** The teacher ceilings, the
distillation table, and the gen-query trend were all teacher-forced on checkpoints without
M-RoPE. Trust them.

**UNVERIFIED — `--mrope_scale_side text` was chosen on void numbers.** The verdict may still
be right; it has not been re-measured with the geometry engaged. The design rationale above is
unaffected.

**SUSPECT — "CFG is closed."** Measured on the AR path pre-M-RoPE-fix and plausibly at the
wrong temperature. Treat as untested.

---

## ESTABLISHED (continued)

### Training exclusively at mask ratio 1.0 does NOT build text conditioning (2026-08-23)
`cosyvoice2_smollm2_nar_r1_flat_lr-flat_0`, killed at ~63k steps = **34 epochs over
LibriTTS-R clean**. Train loss kept falling; eval did not improve; **no signs of
memorization.**

Trajectory measured at r=1.0 while it ran:

| step | predictive entropy | text_delta | acc_real |
|---|---|---|---|
| 1000 | 11.291 | — | — |
| 5000 | 11.072 | +0.0017 [+0.0007,+0.0026] | 0.0197 |
| 10000 | 11.045 | +0.0015 [+0.0007,+0.0022] | 0.0133 |

Entropy fell 0.219 bits over the first 4k steps and then 0.027 over the next 5k — an 8x
collapse in rate, flattening at 87% of maximum. `text_delta` never moved.

**The absence of memorization is the informative part.** Data scarcity predicts overfitting: a
~180M model given 34 passes over ~13M target tokens should memorize something if the mapping
is learnable. It did not. The falling train loss is consistent with learning the unit MARGINAL
(corpus statistics), which lowers loss slightly, transfers to eval, and is not memorization.
So the model cannot fit text -> unit even on data it has seen 34 times, which points at
representation, capacity or framing rather than at data quantity.

This is the third independent result with the same shape: gen-query (2026-08-13), the NAR
crutch finding (2026-08-21), and now training with no crutch available at all. Removing or
withholding the local shortcut does not cause text to carry content.

⚠️ Also unresolved from that run: the r=1.0 probe reported acc_real FALLING (0.0197 -> 0.0133)
while TensorBoard's own `eval/voice_synthesis/voice_unit_accuracy` rose monotonically
(0.0232 -> 0.0282 by 14.8k) and was ~2x higher in absolute terms. Both claim to measure
masked-position unit accuracy at r=1.0 on held-out data. They disagree and it was never
reconciled — suspect the probe before the trainer. Fix this before either number is used again.

---

### The 32-utterance memorization run ANSWERS the prerequisite: text is read, as a KEY (2026-08-24)
`memorize32_0` (NAR, mask ratio pinned 1.0, duration token on, LR 1e-4 constant,
32 train samples) reaches **train `voice_unit_accuracy` 1.0 by step ~1400** and holds it to
20000, with `voice_unit_ce_loss_norm` at 2e-5. At r=1.0 there is no voice context at all —
every unit is predicted from text + duration alone — so this is complete memorization of all
32 utterances from the text side.

**That number alone proves nothing, because of a layout confound.** Over those 32 samples
`(voice_frame_len, text_len)` is a *unique index*: 30 of 32 exact frame lengths are distinct,
the single collision (3 samples at 173 frames) is broken by text length, and all 32 are the
same speaker (730), so speaker is not a key either. The model could have reproduced all 32
while ignoring every text token, keyed on two integers the sequence layout hands it for free.

`scripts_local/memorization_text_dependence.py` settles it on the finished checkpoint, no
training required, by editing the transcript at the SAMPLE level (before collation, so control
tokens, placeholder positions and the duration bucket are byte-identical across arms) and
re-scoring at r=1.0, n=32:

| arm | what changes | unit acc | CE |
|---|---|---|---|
| real | nothing (the memorized condition) | **1.0000** | 0.0000 |
| roll | content + text_len (the standard ablation) | 0.0478 | 19.91 |
| matched | content only — **layout key preserved** | **0.0689** | 18.90 |
| within | token ORDER only (same multiset, same length) | 0.3866 | 8.28 |
| constant | all cross-sample text information removed | 0.0557 | 16.63 |

`matched` is the decisive arm: the (frame_len, text_len) key is intact and accuracy still
falls 1.0000 -> 0.0689. **The transcript is not decorative.** The text pathway is live.

**But it is being used as a retrieval key, not read compositionally.** Under `roll`, sample i
is handed sample i-1's transcript while keeping i's frame budget and i's duration bucket. The
predictions match **sample i-1's stored units at 0.8250** (n=3629 overlapping positions). The
text selects which memorized utterance to replay, and it replays it at 82.5% fidelity while
overriding both length signals. `within` = 0.3866 is consistent: an order-scrambled bag of
tokens is still a partial key over 32 items.

What this licenses and what it does not:
- **Licensed:** the text->trunk->coda path carries enough bandwidth to select among 32
  utterances and drive ~200 frames of output. The month of weak text conditioning is a
  GENERALIZATION failure, not a dead or low-bandwidth wiring. Data scale and alignment
  (bistream) stay the live questions; capacity and the frozen text representation do not.
- **Not licensed:** any claim that the model reads text phonetically. Retrieval at 0.825 is
  the degenerate solution memorization always admits. n=32 cannot distinguish
  "reads phonemes" from "hashes the sentence"; only a length-decorrelated set or a corpus-scale
  measurement can.

Protocol note: measured on `checkpoint-20000`, train split, the same 32 samples the run
trained on, mask ratio 1.0, argmax (no sampling), duration token emitted, M-RoPE
`scale_side=text` rate 6.0. Report: `eval_output/world_voice_memorization/text_dependence/`.

### Teacher-forced unit accuracy does NOT predict free-running quality (2026-08-24, replication)
Stated plainly because it keeps having to be re-derived. Two direct demonstrations in this
project, on the same weights at the same step:

- `memorize32_0` (NAR): train unit accuracy **1.0** while free-running renders were
  nonsense for some samples at 17.5k.
- `distill_0` @44k: early_text_delta +0.0522 against the teacher's +0.0538, text-attributed
  0.371 — and it **still sounded bad by ear**.

The mechanism is not subtle: teacher-forced accuracy conditions every prediction on a PERFECT
prefix. Free-running conditions on the model's own drifting output, and this model's known
failures — self-conditioned repetition, and termination — live entirely in that gap. A rising
`eval/voice_synthesis/voice_unit_accuracy` says the conditional is being learned. It cannot
say whether generation terminates, and it is structurally blind to over-length degeneration.
Read free-running from `world_voice_ar_diagnostics.py` section 3 (adj_repeat, longest_run,
EOV firing rate, length vs GT) and `cosyvoice_wer_eval.py`, and finally by ear.

### ⚠️ A constant LR may never leave the free-running collapse regime (2026-08-24, derived)
`cosyvoice2_smollm2_ar_flat_lr_0` runs `constant_with_warmup` at 1e-5. An earlier
finding established that greedy EOV-collapse in this project was a **mid-cosine, high-LR
artifact** and that late low-LR checkpoints free-ran clean — hence "judge free-running from
late checkpoints only". A constant-LR run has no late: it stays at the same LR forever, so if
that finding holds, this run may never produce a clean free-running checkpoint no matter how
long it trains, while its teacher-forced metrics keep improving. **Derived, not measured** —
but it predicts exactly the reported symptom (good and rising eval accuracy, no EOV, long
degenerate renders) and should be checked before concluding anything about AR from this run.

### Eval-path bug: generated voice was never trimmed to its real length (fixed 2026-08-24)
`voice_latent_preds` is (B, max_n, C, max_T), padded across every utterance a generate() call
produced. The training viz read `voice_preds[0, 0]` with no slice by `outputs["voice_lengths"]`,
so when the model emitted more than one voice block the FIRST one was zero-padded out to the
longest, and the frozen decoder rendered the zero tail as babble — the same "N real seconds
then nonsense to the cap" artifact already documented for the transcription-input render. A
short generation therefore sounded like a long degenerate one, and the model got the blame.
Fixed with `_trim_generated_voice` at both sites (voice_to_voice and text_to_voice).

### AR flat-LR @11076: text conditioning is at the TEACHER, free-running is not (2026-08-24)
`cosyvoice2_smollm2_ar_flat_lr_0/checkpoint-11076`, teacher-forced, held-out
n=512, M-RoPE engaged (`scale_side=text`):

| metric | this run @11k | AR @44k | teacher |
|---|---|---|---|
| acc_real | **0.1081** | 0.0940 | 0.1283 |
| text_delta (all-pos) | **+0.0487** [+0.0462,+0.0511] | +0.0349 | +0.0594 |
| early_text_delta | **+0.0640** [+0.0549,+0.0732] | +0.0522 | +0.0538 |
| **text-attributed fraction** | **0.450** | 0.397 | 0.463 |
| eov_position_acc | 0.8848 | — | — |

Text attribution **0.450 against the teacher's 0.463** at a QUARTER of the steps the 0.397
figure took, and early_text_delta is nominally above the teacher (CIs overlap, so read it as
"at the teacher", not "beyond"). The position-resolved horizon **never decays below 0.01**
(+0.0640 at frames 0-8 down to +0.0401 at 128+), where earlier AR runs showed a horizon of a
few words. Top-1 0.108 -> top-5 0.303 -> top-10 0.424 says the remaining top-1 gap is target
multimodality, not a conditioning wall.

⚠️ **Do not attribute the gain.** This run differs from the 44k baseline in FOUR ways at once:
AR-vs-flat-LR schedule (constant_with_warmup vs cosine), no CFG text dropout, no distillation,
and `--bucket_by_length`. It is a better model on this axis; which change bought it is
unmeasured. The older 0.305 @23k figure is additionally suspect — it predates the M-RoPE eval
fix, and only teacher-forced metrics on NON-M-RoPE checkpoints are in the CLEAN class.

**Free-running, CORRECTED (per-utterance, n=32, gen at 0.6):** EOV fired 12/32, budget-capped
20, and **disjoint utterances per prompt mean 2.03, max 5, with 26/32 prompts producing more
than one**. First-utterance statistics:

| metric | generated | GT |
|---|---|---|
| len_mean | 196.3 | 164.2 |
| len_min / max | 12 / 250 | 28 / 222 |
| **text-len → gen-len r** | **+0.887** | +0.960 (ceiling) |
| adj_repeat_rate | 0.2645 | 0.0306 |
| longest_run | 124 | 16 |
| distinct_units | 1720 | 2225 |
| unit_entropy_bits | 8.70 | 10.51 |
| distinct_bigram_ratio | 0.671 | 0.958 |

⭐ **Duration conditioning is near the ceiling: r=+0.887 against GT's +0.960.** The first pass
reported +0.535 — the concatenation was destroying the correlation by adding a second block's
frames to a length the prompt never asked for. Text drives DURATION well; this is structural
conditioning working.

**What remains is repetition, and it is severe and independent of the block issue:**
adj_repeat 8.6x GT, longest_run 124 vs 16, bigram diversity 0.671 vs 0.958, entropy 8.70 vs
10.51 bits. Fixing termination will NOT touch this.

**The original (void) numbers, for the record:** Reported: EOV
5/32, len_mean 373.9 vs GT 164.2, len_max 500, adj_repeat 0.2850 vs GT 0.0306, longest_run 87
vs 16, distinct-bigram 0.633 vs 0.958, entropy 8.75 vs 10.51 bits. But len_mean 373.9 and
len_max 500 against a **250-frame budget** are arithmetically impossible for one block — so
those statistics were computed over the FLAT unit trace spanning several blocks, the same
concatenation flaw found in the render path the same day. `len(trace) >= budget` then fires
for any multi-block generation, and "EOV fired" degrades to "the LAST block ended with EOV".
Fixed to score the FIRST utterance and to report disjoint-utterance counts; the repetition
figures are largely within-block and so probably survive, but **the length statistics above
are void pending the re-run**.

### ⭐⭐ `ar_flat_lr_1` OVERFITS: eval loss is a clean U with its minimum at 14768 (2026-08-25)
Owner observation, confirmed. `eval/voice_synthesis/voice_unit_ce_loss_norm`:

| step | 1846 | 5538 | 9230 | **14768** | 18460 | 22152 | 25844 | 29536 |
|---|---|---|---|---|---|---|---|---|
| eval CE | 0.664 | 0.570 | 0.534 | **0.5143** | 0.522 | 0.539 | 0.565 | 0.595 |
| eval acc | 0.0696 | 0.0947 | 0.1066 | **0.1127** | 0.1101 | 0.1065 | 0.1017 | 0.0980 |

Minimum eval CE and peak eval accuracy BOTH land at 14768 (epoch 8), then reverse
monotonically — eval CE +16% and accuracy −13% by 29536 — while train CE keeps falling
(1.032 → ~0.29-0.43 at 30400). Falling train, rising eval, widening gap: **memorization**.

This is the same 14768 inflection the trend table found independently via `acc_real`
(0.1118 peak → 0.1019), and it retro-explains the "conditioning plateau" — conditioning did
not plateau, the model passed its optimum and started regressing.

⭐ **THE EAR DISAGREES WITH EVAL CE, AND THE EAR IS THE ARBITER.** Owner, 2026-08-25, on the
LATEST checkpoints (~28-30k, i.e. deep into the eval-loss rebound): *"the latest outputs are
the best TTS we've gotten ever."* So the U-shaped eval curve does NOT mean the model is
getting worse at the task anyone cares about. It is consistent with the degeneration trend,
where 25844 beats 14768 on every free-running metric while its eval CE is far worse.

Two things are moving in opposite directions and they are not the same thing:
- **teacher-forced next-unit CE** peaks at 14768 and then regresses (memorization);
- **free-running speech quality** keeps improving well past it.

Do NOT select checkpoints on eval CE for this direction. A model that memorizes the training
distribution can still free-run better, and free-running is the product.

⚠️ **CONSEQUENCE FOR THE INTELLIGIBILITY RESULT: it was measured at 28000, ~13k steps PAST
the eval optimum.** LCS 0.62 may understate this run. It is NOT safe to assume 14768 is
better, because teacher-forced eval and free-running degeneration DIVERGE here — the trend
shows 25844 beating 14768 on every degeneration metric (adj_repeat 0.236 vs 0.303, entropy
9.02 vs 8.57, bigram 0.663 vs 0.602) while its eval CE is much worse. Which checkpoint
actually renders best is an open, cheap, and unmeasured question.

**LR vs data is not yet settled.** Constant 1e-4 with no decay is the obvious suspect (this
run anneals never), but 16 epochs over 118k utterances with a ~168M model could overfit on
data grounds alone. A WSD/cosine decay tail from ~12k distinguishes them cheaply: if the eval
minimum moves later and lower, it was the schedule; if the U simply repeats at the same place,
it is data and LibriHeavy/MLS is the answer.

### ⭐⭐⭐ WER TREND across `ar_flat_lr_1`: 28000 is the peak; keep it (2026-08-25)
30 runs — 5 checkpoints x 2 arms x 3 seeds, n=128, temperature 0.6, ceiling LCS 0.884:

| ckpt | RAS LCS | plain LCS | RAS WER | plain WER | plain hyp/ref |
|---|---|---|---|---|---|
| 14768 | 0.4701 ±0.0069 | 0.3683 ±0.0051 | 0.6210 | 0.6778 | 0.785 |
| 24000 | 0.5646 ±0.0102 | 0.4471 ±0.0124 | 0.4924 | 0.5878 | 0.836 |
| 26000 | 0.5655 ±0.0118 | 0.4609 ±0.0047 | 0.5006 | 0.5760 | 0.870 |
| **28000** | **0.5785 ±0.0021** | **0.5252 ±0.0069** | **0.4869** | **0.5181** | **0.946** |
| 30000 | 0.5792 ±0.0045 | 0.4743 ±0.0096 | 0.4942 | 0.5760 | 0.850 |

**14768 — the eval-CE OPTIMUM — is the WORST checkpoint measured**, by 0.16 LCS against seed
spreads of ±0.007. Selecting on eval CE would have picked a model ~30% worse at the task.
This is the ear-vs-eval-CE dissociation made quantitative; treat eval CE as diagnostic only.

**28000 is a genuine peak on the PLAIN arm** (0.5252 vs 0.4471/0.4609 before and 0.4743
after, 6-8σ), not the start of a plateau. An earlier partial read of the RAS-only data said
"plateau from 24k" — **wrong**, corrected when the plain arm landed.

⚠️ **RAS COMPRESSES CHECKPOINT DIFFERENCES and will mislead checkpoint selection.** Across
24000-30000 the RAS arm spans 0.015 while plain spans 0.078. Judge checkpoints on the PLAIN
arm; use RAS for rendering only.

`hyp/ref` tracks the plain arm exactly (0.836/0.870/0.946/0.850), so most of the plain
difference is HOW MUCH the model says — 28000 says about the right amount, its neighbours
under-speak. The advantage is narrower in kind than the LCS gap implies.

⚠️ A sharp single-point peak flanked by dips at 2000-step spacing may be a favourable point
in a noisy walk rather than a true optimum. If an anneal from 28000 underperforms, suspect
this first.

**RAS value grows as the model degrades**: +0.053 at 28000, +0.105 at 30000, +0.102 at 14768,
+0.118 at 24000. It is a compensator for a worse model, not a fixed bonus.
⚠️ It can also KILL a short utterance outright — utterance 02 renders 0.0 s under RAS at
28000 (target 1.1 s, plain 0.7 s). One empty render in twelve barely moves a mean, so the
aggregate hides it.

### ⭐⭐⭐ Flat LR PLATEAUS at ~34k; the WSD decay beats the plateau (2026-08-28)
Full WER trend, all identical protocol (n=128, temp 0.6, 3 seeds, first-utterance scoring):

| checkpoint | RAS LCS | RAS trunc | RAS WER | plain LCS | plain hyp/ref |
|---|---|---|---|---|---|
| flat @24000 | 0.5646 ±0.0102 | 0.5199 | 0.4924 | 0.4471 | 0.836 |
| flat @28000 | 0.5785 ±0.0021 | 0.5444 | 0.4869 | 0.5252 | 0.946 |
| flat @30000 | 0.5792 ±0.0045 | 0.5317 | 0.4942 | 0.4743 | 0.850 |
| flat @34000 | 0.5928 ±0.0169 | 0.5581 | 0.4840 | 0.5580 | 1.001 |
| flat @37000 | 0.5887 ±0.0090 | 0.5596 | 0.4761 | 0.5448 | 0.976 |
| flat @40612 | 0.5810 ±0.0165 | 0.5467 | 0.4820 | 0.5602 | 0.991 |
| **WSD @34000** | **0.6138 ±0.0065** | **0.5772** | **0.4424** | 0.5592 | 0.964 |

**Flat LR plateaus from ~34000.** 34000/37000/40612 are 0.5928/0.5887/0.5810 with spreads of
±0.009-0.017 — indistinguishable. Training past 34k at constant 1e-4 buys nothing.

**The WSD decay beats the whole plateau**: 0.6138 vs a best flat of 0.5928 (+0.021, marginal
against the combined spread) and WER 0.4424 vs a best flat of 0.4761 (**−0.034, clear**). So
annealing is worth roughly one WER point beyond anything constant LR reaches, and
`ar_wsd_0/checkpoint-34000` is the best model this direction has produced.

⚠️ **Two of my own earlier claims corrected by this table.** (1) "28000 is a genuine peak,
6-8σ" — false; it was the peak of a sweep that stopped at 30000, and 34000 is higher. (2) "I
annealed a model that was still improving and stopped it while it was still improving" —
also false; flat plateaus at ~34000, so decaying from 26000 to 34000 was reasonably timed.
Both errors came from extrapolating a trend past its last measured point.

**Anchor for a cosine run:** flat saturates ~34k and decay adds on top, so a total in the
34-40k range is supported by measurement rather than guessed. The earlier argument AGAINST
cosine was built on eval CE accelerating during decay — and eval CE anti-correlates with
speech quality here, so that argument should not have been made.

### ⭐⭐⭐⭐ THE WALL WAS THE LEARNING RATE, not the architecture (2026-08-27)
All measured with IDENTICAL current tooling and protocol — n=128, temperature 0.6, 3 seeds,
first-utterance scoring — so this is model-vs-model, not instrument-vs-instrument:

| run | step | LR | EOS loss | RAS LCS | RAS trunc | plain hyp/ref |
|---|---|---|---|---|---|---|
| `cosyvoice2_smollm2_mrope_scale_text_0` | 44000 | 1e-5 cosine | no | 0.1238 | 0.1120 | 0.276 |
| `cosyvoice2_smollm2_distill_0` | 44000 | 1e-5 cosine | no | 0.1966 | 0.1721 | 0.427 |
| `cosyvoice2_smollm2_ar_flat_lr_0` | **12000** | **1e-4 const** | **no** | **0.4267** | **0.3796** | 0.717 |
| `cosyvoice2_smollm2_ar_flat_lr_1` | 28000 | 1e-4 const | yes | 0.5785 | 0.5444 | 0.946 |

**`ar_flat_lr_0` at 12000 steps more than DOUBLES the old arms at 44000, with no EOS
supervision and no distillation** — a quarter of the training and 2.2x the truncated LCS. The
only material difference is 1e-4 constant vs 1e-5 cosine. **The learning rate was the wall.**

EOS supervision adds on top (0.3796 -> 0.5444 truncated), though that comparison also spans
12000 -> 28000 steps and is therefore confounded with training length.

⚠️ **This retroactively undermines every "structural ceiling" conclusion measured at 1e-5** —
the 0.267 unit-accuracy plateau, "voice conditioning develops over 20k-45k steps", and the
framing that text->content binding was an architectural limitation. Those were measured on an
under-optimised model. They are not disproven, but they cannot be cited as properties of the
architecture until re-measured at a working LR.

**The old numbers were also FLATTERED by the old tooling, not penalised by it.** Documented
truncated LCS was 0.1620; re-measured at the operating point with correct tooling the old arms
score 0.112-0.172. The earlier figure was inflated by temperature 1.0 and by flat-trace
concatenation (more emitted text -> more matched words). The prediction that eval fixes would
make the old runs look BETTER was wrong; they look worse, and the ~3x improvement stands.

**Dominant old failure: UNDER-SPEAKING.** `hyp/ref` 0.276-0.427 against `_1`'s 0.946. Those
models said a quarter to a half of what they should and then started another utterance.

### ⭐⭐⭐ PERCEPTUAL ANCHOR: what LCS 0.62 / CER 0.32 actually SOUNDS like (2026-08-25)
Owner, on `ar_flat_lr_1` at ~28-30k: **"the first time I can hear words all the way through
output examples, even if some of the sounds aren't quite right."**

Record this, because the project has accumulated metrics with no perceptual reference and the
mapping is what makes them usable:

| measurement | perceptual counterpart |
|---|---|
| LCS recall 0.62 (ceiling 0.89) | words are recoverable throughout |
| hyp/ref 0.978 | speech runs the FULL length — no truncation, no drift into silence |
| CER 0.32 vs WER 0.48 | phonemes roughly right, words imperfect = "sounds aren't quite right" |
| position text_delta never < 0.01 | conditioning holds to the END of the utterance |

CER well below WER is the signature of the described failure: the articulation is
approximately correct and the word identities are not quite landing. Contrast the entire prior
history of this direction — "correct onset that falls off quickly", "coherent bursts unrelated
to the prompt", "text horizon 3-5 words" — which is what LCS ~0.16 sounded like.

**The position-resolved horizon PREDICTED this.** At 11076 the metric first showed text_delta
never decaying below 0.01 across all position buckets, where earlier runs decayed within a few
words. The ear confirmed the same change independently. That is a validated leading indicator:
watch the horizon buckets, not just `early_text_delta`.

### ⭐⭐⭐ INTELLIGIBLE SPEECH: LCS recall 0.62 vs 0.89 ceiling at 28k (2026-08-25)
`ar_flat_lr_1/checkpoint-28000`, n=64, temperature 0.6, Whisper via `cosyvoice_wer_eval.py`:

| metric | plain | RAS win=10 | ceiling |
|---|---|---|---|
| LCS recall | 0.5438 | **0.6205** | 0.8943 |
| WER | 0.5181 | 0.4826 | 0.1049 |
| CER | 0.3718 | 0.3202 | 0.0416 |
| hyp/ref word ratio | 0.9783 | 1.0731 | 1.0022 |
| — truncated LCS | 0.4817 | 0.5782 | 0.8943 |

**60.8% of ceiling plain, 69.4% with RAS.** For scale, the previous best AR arm was truncated
LCS **0.1620** @23k — this is a **3.0-3.6x improvement**, and the first time this direction has
produced speech that is substantially intelligible rather than "correct onset then drift".

`hyp/ref` 0.978 plain says the LENGTH is right, matching the text->duration correlation
(r +0.86-0.92 vs GT 0.96). So the model says about the right amount, in about the right
places, with about 60% of the reference words recoverable.

**What the RAS A/B actually answered.** The prediction was binary — sharp WER drop = units
right / drift masking them, versus no drop = units wrong. The answer is BOTH, and the middle
case is the informative one: RAS buys **+0.077 LCS recall** (4x the 0.018 decode range at
n=64, so real), but plain is already at 0.544. **Repetition is a moderate tax, not the wall.**
The remaining 0.27 gap to ceiling is neither repetition nor termination — it is genuine
content error, and no decode-side trick will recover it.

Protocol: single seed, n=64. Decode floor is std 0.0089 / range 0.018, so the absolute numbers
carry ~±0.02 and the RAS delta is solid. Re-measure with >=3 seeds before quoting these as
final.

### ⭐ `--unmask_eos_in_synthesis` ELIMINATES the extra-utterance failure (2026-08-24)
THE one-flag comparison: `ar_flat_lr_0` vs `ar_flat_lr_1` at **checkpoint-11076**, identical
LR/schedule/seed/data, n=512 teacher-forced + gen_n=32 free-running, eval-time ban OFF in both.

| metric | `_0` (no EOS loss) | `_1` (EOS loss) | verdict |
|---|---|---|---|
| **disjoint utterances / prompt** | **2.16 (29/32 >1, max 4)** | **1.00 (0/32 >1, max 1)** | **ELIMINATED** |
| acc_real | 0.1083 | 0.1077 | unchanged |
| early_text_delta | +0.0581 [+0.0491,+0.0674] | +0.0588 [+0.0500,+0.0679] | unchanged |
| text-attributed | 0.454 | 0.438 | unchanged |
| len_mean (first utt) | 211.1 | 193.3 | within noise |
| adj_repeat_rate | 0.3486 | 0.3712 | **unchanged** (GT 0.0306) |
| longest_run | 92 | 98 | unchanged (GT 16) |
| unit_entropy_bits | 7.95 | 7.73 | unchanged (GT 10.51) |
| distinct_bigram_ratio | 0.595 | 0.579 | unchanged (GT 0.958) |
| EOV fired | 9/32 | 11/32 | unchanged |

**0 of 32 prompts produced a second utterance**, against 29 of 32 without the flag. All three
predictions recorded in advance held: utterances → 1.0, conditioning untouched, repetition
untouched. Supervising ONE token at ONE position removed the failure completely.

### Repetition, conditioning and onset ALL plateau together ~14-18k (2026-08-25)
Trend across `ar_flat_lr_1`, 7 epoch-boundary checkpoints, n=512 TF + gen_n=64:

| step | EOV | adj_repeat | entropy | bigram | early_text_delta | acc_real | attributed | 1-32 | 128+ |
|---|---|---|---|---|---|---|---|---|---|
| 3692 | 20/64 | 0.429 | 7.01 | 0.421 | +0.0376 | 0.0797 | 0.165 | 0.134 | 0.526 |
| 7384 | 25/64 | 0.371 | 7.90 | 0.518 | +0.0564 | 0.0976 | 0.373 | 0.112 | 0.475 |
| 11076 | 21/64 | 0.292 | 8.64 | 0.607 | +0.0605 | 0.1074 | 0.439 | 0.092 | 0.330 |
| 14768 | 22/64 | 0.303 | 8.57 | 0.602 | +0.0586 | **0.1118** | 0.485 | 0.067 | 0.406 |
| 18460 | 17/64 | 0.236 | 9.18 | 0.672 | +0.0581 | 0.1104 | 0.519 | 0.065 | 0.296 |
| 22152 | 31/64 | 0.234 | 9.18 | 0.673 | +0.0591 | 0.1071 | 0.534 | 0.063 | 0.316 |
| 25844 | 28/64 | 0.236 | 9.02 | 0.663 | +0.0588 | 0.1019 | 0.547 | 0.064 | 0.330 |
| GT | — | 0.038 | 10.51 | 0.958 | — | — | — | 0.030 | 0.030 |

⚠️ **The rising "text-attributed" column is an ARTIFACT past 14768.** `early_text_delta` is
FLAT at ~+0.0585 from 11076 on, while `acc_real` PEAKS at 0.1118 (14768) and then declines to
0.1019. Attribution is delta/acc_real, so 0.485 -> 0.547 is a shrinking denominator. **Do not
quote 0.547 as "past the teacher's 0.463".** Report `early_text_delta` and `acc_real`
separately.

Three things plateau together around 14-18k: repetition (0.234-0.236, flat), onset repetition
(0.063-0.065, flat, ~2x GT), and conditioning (+0.0585, flat) — while `acc_real` DECLINES and
training loss keeps falling. That co-plateau plus a falling accuracy is the first concrete
evidence for the **constant-LR concern**: 1e-4 held forever with no decay, past the point of
usefulness. An LR decay / WSD tail from ~18k is the obvious untested lever.

**Position-resolved repetition is RISING with position at every checkpoint** (0.064 at frames
1-32 vs 0.330 at 128+, against a flat GT ~0.030), which is the signature of self-conditioning
drift / exposure bias, NOT "repetitive from the start". That rules IN scheduled sampling and
on-policy distillation (both built, both unused) and rules OUT conditioning/decoder causes.

**EOV never improves** — 17-31/64 across the whole span, no trend, while repetition halves.
That kills the earlier hypothesis that looping is what prevents termination; they are
independent.

### The free-running metrics have a large noise floor at gen_n=32 (2026-08-24)
`ar_flat_lr_0/checkpoint-11076` was run through section 3 TWICE the same day, identical
weights, different RNG:

| metric | run A | run B | spread |
|---|---|---|---|
| adj_repeat_rate | 0.2645 | 0.3486 | **0.084** |
| longest_run | 124 | 92 | 32 |
| EOV fired | 12/32 | 9/32 | 3 |
| utterances / prompt | 2.03 | 2.16 | 0.13 |
| len_mean | 196.3 | 211.1 | 15 |

**A repetition difference under ~0.08 at gen_n=32 means NOTHING.** This is the degeneration
analogue of the documented LCS decode floor (std 0.0089, range 0.018 at n=64). The
arm_0-vs-arm_1 repetition delta (+0.023) sits well inside it, which is what licenses calling
it "unchanged" rather than "slightly worse". Raise gen_n or repeat-and-average before claiming
any repetition result.

(The teacher-forced numbers in run A are NOT a noise estimate — it used `--n 64` against run
B's `--n 512`, visible in the CI widths [+0.0215,+0.0820] vs [+0.0491,+0.0674].)

### ⭐ THE WALL IS NOW REPETITION (2026-08-24)
With termination fixed, what remains at 11076 is unambiguous and large:

| | generated | GT | ratio |
|---|---|---|---|
| adj_repeat_rate | 0.3712 | 0.0306 | **12x** |
| longest_run | 98 | 16 | 6x |
| unit_entropy_bits | 7.73 | 10.51 | −2.8 bits |
| distinct_bigram_ratio | 0.579 | 0.958 | |
| coverage | 0.234 | 0.339 | |

And it explains the residual length failure: **21 of 32 generations hit the 250-frame budget**
rather than emitting EOV. The model falls into a loop and never reaches a natural end. So
repetition is likely the ROOT cause of the remaining over-length, not a separate problem —
which also means "fix termination" is not the next lever; "fix repetition" is.

Teacher-forced conditioning is meanwhile essentially at the teacher: early_text_delta +0.0588
vs +0.0538, text-attributed 0.438 vs 0.463, text→duration r +0.856-0.922 vs 0.960. **The model
knows what to say and how long to say it, and cannot stop repeating while saying it.**

### The EOS exemption is ORTHOGONAL to voice conditioning — matched-step null (2026-08-24)
`ar_flat_lr_0` vs `ar_flat_lr_1` at **checkpoint-5538**, one flag apart, teacher-forced,
held-out n=512:

| metric | `_0` (no EOS loss) | `_1` (`--unmask_eos_in_synthesis`) | Δ |
|---|---|---|---|
| acc_real | 0.0918 | 0.0921 | +0.0003 |
| text_delta (all-pos) | +0.0288 [+0.0269,+0.0307] | +0.0290 [+0.0269,+0.0310] | +0.0002 |
| early_text_delta | +0.0430 [+0.0352,+0.0510] | +0.0442 [+0.0359,+0.0525] | +0.0012 |
| text-attributed | 0.313 | 0.315 | +0.002 |

Every difference is an order of magnitude inside its own CI. **The exemption supervises the
TEXT head's stop position and leaves the VOICE head's dependence on the transcript untouched**,
which is what it was supposed to do — and it means an EOS-flag run can be compared to a
non-EOS run on conditioning metrics without adjustment.

This was run as a PASS/FAIL on the flag, not to find an effect: a difference here would have
meant the added text loss was perturbing the shared trunk, which had to be known before
attributing anything at 11076.

**Answers "is 5538 too early to measure early_text_delta?" — no.** The probe is teacher-forced,
so free-running immaturity (1-unit renders, budget-capped blocks) does not touch it, and at
n=512 the CI is ±0.008. Conditioning trajectory for `_0`, both post-M-RoPE-fix and clean:

| step | early_text_delta | text-attributed |
|---|---|---|
| 5538 | +0.0430 | 0.313 |
| 11076 | +0.0640 | 0.450 |

Roughly +0.02 early-delta per 5.5k steps at this stage, against a teacher ceiling of +0.0538
(already passed at 11076) and 0.463 attributed.

### `--unmask_eos_in_synthesis` works, and shows no harm at matched step (2026-08-24)
`ar_flat_lr_1`, step 1846, **with the eval-time ban OFF** (`--no_viz_suppress_media_tokens`):
**zero `utt1+` tags across all 8 prompts** — every prompt produced exactly ONE utterance. The
model is emitting EOS after the voice block because it was trained to, not because sampling
was prevented. `train/text_loss_norm` bottoms out (1.0155 -> 2.8e-4 by step 100), which is the
expected floor for a one-class objective, so it confirms the gradient path is live and nothing
more.

The renders at 1846 are tiny — 1, 1, 1, 1, 2, 9, 11, 11 units (0.04-0.44 s). **That is
undertraining, not the fix.** Step-matched `_0` at 1846, which has no EOS supervision at all
and whose numbers are SUMS across every block: 3, 1, 33, 1, 224, 27, 3, 1. Equally tiny. `_0`
then grows over training (@3692: 2, 27, 250, 250, 3, 16, 250, 7; @5538: 15, 43, 26, 431, 14,
244, 250, 250).

So the earlier prediction that this fix would make renders shorter **cannot be evaluated at
1846** — the control is just as short there. Judge at `_1`@11076 against `_0`@11076, which is
a one-flag comparison. n=8 with stochastic sampling either way, so read the distribution, not
individual prompts.

Practical consequence: **leave the eval-time ban OFF for voice.** It is now redundant (the
model stops on its own), it hides the utterance count that would detect a regression, and the
A/B showed it truncates to a false start when one occurs.

### BO* suppression works — and reveals that PREMATURE EOV is the bigger failure (2026-08-24)
A/B on `ar_flat_lr_0/checkpoint-11076`, same 8 val prompts, voice_temperature 0.6
(`scripts_local/voice_render_length_audit.py --suppress_media_tokens / --no_...`):

| prompt | suppression ON | suppression OFF |
|---|---|---|
| 0 | 1 utt [95] | 1 utt [75] |
| 1 | 1 utt [20] | 4 utts [4, 69, 88, 91] |
| 2 | 1 utt [18] | 6 utts [47, 88, 82, 95, 81, 62] |
| 3 | 1 utt [16] | 3 utts [32, 238, 91] |
| 4 | 1 utt [41] | 4 utts [7, 184, 191, 83] |
| 5 | 1 utt [225] | 2 utts [250, 39] |
| 6 | 1 utt [250] | 3 utts [250, 231, 17] |
| 7 | 1 utt [250] | 3 utts [222, 219, 61] |

**The ban is completely effective: 1 utterance on every prompt, versus mean 3.25 / max 6
without it.** So the mechanism is confirmed — a second block requires a sampled BOV and
nothing else produces one.

⚠️ **But the ban is NOT a neutral fix, and this is the important part.** Look at the first
utterance with suppression OFF: prompt 1 emits **4 frames** then a real 69/88/91-frame
utterance; prompt 4 emits **7 frames** then 184/191. The model frequently emits a tiny FALSE
START, terminates it with EOV, and only then speaks properly. Suppressing BO* makes that false
start **final** — prompts 1/2/3 render at 20/18/16 frames (0.6-0.8 s) instead of reaching the
real utterance. The ban converts "too much audio" into "truncated audio".

So there are **two independent termination failures**, and only one of them was diagnosed:

1. **Premature EOV from the VOICE head** — 5 of 8 prompts end their first utterance under ~50
   frames against a val median of ~102. This is supervised (`eov_position_acc` 0.8848) and is
   the DOMINANT failure by audible impact.
2. **No EOS supervision on the TEXT head**, so after a block ends the model starts another.
   This is what `--unmask_eos_in_synthesis` fixes.

**Prediction for `ar_flat_lr_1`:** fixing (2) alone should drive utterances per prompt toward
1.0 while making renders SHORTER and possibly worse by ear, because the false-start blocks
will no longer be followed by the real utterance. A drop in utterance count is therefore NOT
by itself evidence of improvement — check first-utterance length distribution beside it.

### `ar_flat_lr_0` vs `ar_flat_lr_1` IS a clean A/B (2026-08-24, corrected)
~~An earlier version of this entry claimed `_0` ran at LR 1e-5 for its first 11000 steps and
was therefore confounded with `_1`.~~ **Wrong** — that was read off the CLI *handed over*, not
the one launched. `_0`'s own event file records `--learning_rate 1e-4
--lr_scheduler_type constant_with_warmup --warmup_steps 2000`, identical to `_1`, and the
resume at 11000 kept it. Verify launch commands from `training/command_line`, never from what
was recommended.

So the two runs differ in **exactly one thing**:

| | `_0` | `_1` |
|---|---|---|
| `--unmask_eos_in_synthesis` | no | **yes** |
| LR / schedule / warmup / batch / accum / seed / data | identical | identical |

`_0` is a usable step-matched control. Its measured baselines at 11076: text-attributed
**0.450**, text_delta **+0.0487**, acc_real **0.1081**, disjoint utterances per prompt
**2.03** (diagnostics n=32) / **2.75** (viz n=8), adj_repeat **0.2645**, longest_run **124**.

`_1` confirmed live: `train/text_loss_norm` appears and looks healthy, so the exemption is
NOT inert (the failure mode that hit two other flags the same day).

The primary readout is **disjoint utterances per prompt**, which should fall toward 1.0. The
repetition metrics should NOT move — nothing about EOS supervision touches them, so if they
improve, something else changed and the attribution is wrong.

### Why it starts a second utterance: EOS after a media block is never trained (2026-08-24)
Owner's diagnosis, confirmed in code and pinned by a test. The synthesis layout is
`[text][BOV][PH][EOV][eos]`, so after placeholder-stripping and the causal shift **the text
target AT the EOV position is EOS** — "the utterance is over". `--mask_text_loss_in_synthesis`
masks it along with the transcript, so the model receives **zero gradient on the one position
generation actually samples at**: immediately after a media block.

So this was never a preference for starting another block; the position was untrained. It
explains the whole shape of the failure — EOV fires correctly on 12 of 14 blocks (the VOICE
head is supervised), and then the TEXT head, which is not, does something arbitrary. Measured
at 11076: mean 2.03 utterances per prompt (max 5) on the diagnostics set, 2.75 (max 6) on the
viz set.

`--unmask_eos_in_synthesis` keeps exactly that target, sharing the `_keep` mechanism with the
duration-token and bistream exemptions. Opt-in so in-flight runs are unaffected. It is the
third flag to need the `include_text=False` gate at `training.py:638` — that gate is now a
standing trap for anything supervising the TEXT stream from a voice-only run.

⚠️ This fixes TERMINATION only. The repetition failure (adj_repeat 8.6x GT, longest_run 124)
is independent and untouched by it.

Generalizes beyond voice: world-image has the same layout (`[text][BOI][PH][EOI][eos]`) and
the same masking, so an image-synthesis run cannot learn to stop either.

### EOV FIRES — the over-length failure is repeated BOV, not missing termination (2026-08-24)
Measured, `cosyvoice2_smollm2_ar_flat_lr_0/checkpoint-11076`, n=8 val prompts,
voice_temperature 0.6, budget 250 frames (`scripts_local/voice_render_length_audit.py`):

| prompt | utterances | real frames each | ended by |
|---|---|---|---|
| 0 | 1 | [22] | EOV |
| 1 | 2 | [28, 48] | EOV, EOV |
| 2 | 1 | [21] | EOV |
| 3 | 2 | [26, 20] | EOV, EOV |
| 4 | 1 | [29] | EOV |
| 5 | 2 | [221, 181] | EOV, EOV |
| 6 | 2 | [226, 205] | EOV, EOV |
| 7 | 3 | [250, 250, 6] | budget, budget, EOV |

**EOV fired for 12 of 14 blocks.** ⚠️ This CORRECTS the in-session reading (owner's and mine)
that "the model does not output EOV". It does. What it then does is sample **BOV again** and
start a second utterance, which the flat unit trace concatenated into one long clip. The
failure is a repeated media-block hallucination, not a missing terminator.

The length failure is also **bimodal, not uniformly long**: 5 of 8 prompts end utterance 0 at
21-29 frames (~1 s) against a val median of ~102, i.e. severe UNDER-speaking; two land at
221/226 (plausible); one runs two full 250-frame blocks. Any single "renders are too long"
or "too short" summary of this run is wrong.

### Over-length renders: the flat unit trace concatenates EVERY voice block (fixed 2026-08-24)
Owner observation that cracked it: 20-second renders appeared only after the Mimi/SMG ->
CosyVoice 2 switch; Mimi runs were correctly capped at 10 s. The rate was never wrong — the
TB context string reads **`437 units -> 17.48s @ 24000Hz`**, i.e. exactly 25.0 Hz. 437 units
against a 250-frame budget is the whole story.

What happens: the model fails to emit EOV, block 1 runs to the 250-frame budget, finalizes,
hands back to text, the model samples BOV **again** and speaks a second time. `generate()`
returns `voice_unit_id_trace` as a FLAT trace across every block, so the render concatenated
250 + 187 into one 17.5 s clip.

Why the modality switch exposed it: the SMG path renders `voice_latent_preds[0, 0]` —
utterance 0 alone, therefore capped at the budget. The CosyVoice 2 path renders unit ids and
took them from the flat trace. **The second block was always being generated; the old render
simply could not show it.** `_generated_unit_ids` now prefers `voice_unit_id_segments[idx][0]`,
matching the latent path's utterance-0 convention, so an over-length render is a TERMINATION
finding again rather than a render artifact.

Two lessons worth carrying: a per-block budget bounds a BLOCK, not a call; and when a
diagnostic changes at the same time as a component, suspect the diagnostic.

**Structural fix (owner's proposal, 2026-08-24):** `generate(suppress_media_control_tokens=True)`
bans BOA/BOV/BOI and the three placeholders from the text sampler. At eval the prompt already
supplies BO*, so anything further is a hallucination, and banning it makes a second block
IMPOSSIBLE rather than merely unlikely — the right shape of fix given the measurement above,
where every block terminated correctly and the model simply started again. Off by default: a
speak-at-will model must be able to emit BO*. Placeholders ride along because sampling one is
meaningless in any regime — the interleaver consumes them at training time and at generation
there is nothing to replace them with.

**And the `n` dimension is now honoured at render.** Multiple generated utterances are logged
as SEPARATE clips (`{tag}/{i}` then `{tag}/{i}/utt1`, ...) with an `utterance_count` note,
rather than collapsed into one. That is what `n` is for: `<text><voice_0><text><voice_1>` is
two disjoint clips.

### Bistream trains and decodes; it ties unistream on memorization, as predicted (2026-08-24)
`memorize32_bistream_0` (identical to the unistream arm plus
`--bistream_text_chunk 5 --bistream_voice_chunk 30 --bistream_prob 1.0`): train
`voice_unit_accuracy` 0.787 @100, 0.991 @200, **1.000 @400**, CE 6.3e-4. The unistream arm
was 0.819 / 0.984 / **1.000 @400**, CE 1.4e-4. **A tie**, which is what the retirement note
above predicted: a task solved by retrieval does not exercise alignment. Read this only as
"the bistream path trains end to end on real data", never as evidence about alignment.

Chunk continuation at decode, `scripts_local/bistream_continuation_probe.py` on
checkpoint-500, greedy, generated frames vs target:

| idx | target | without continuation | with continuation |
|---|---|---|---|
| 0 | 73 | 30 | **73** |
| 1 | 165 | 30 | **165** |
| 2 | 129 | 30 | 7 |
| 3 | 85 | 30 | **85** |

Without the transcript continuation every utterance is exactly 30 frames — one chunk — with
the correct content. With it, 3 of 4 match the target length EXACTLY, meaning the model
reproduced every fill_token and the final EOV at precisely the right frames under
free-running greedy decoding. idx 2 collapsing to 7 frames is free-running divergence at a
partially trained checkpoint (train accuracy is teacher-forced), not a mechanism failure;
worth re-checking at a later checkpoint before drawing anything from it.

⚠️ **The one-chunk symptom was a bug, and a well-disguised one.** `forced_token_queue` was
declared beside `forced_next_token`, INSIDE the token loop, which is right for that variable
and wrong for this one -- so it was wiped every iteration, the continuation never fired, and
every render came out as its first chunk with correct content. That looks exactly like "the
model only learned one chunk" and not at all like a decode bug. Training was unaffected
throughout.

Separately, `_sample_tokens` never guarded `temperature <= 0`: it divided, produced infs, and
`torch.multinomial` failed with a device-side assert pointing at the sampler rather than the
argument, so greedy TEXT decoding presented as a CUDA fault. The voice sampler always took 0
as greedy; the text one now agrees.

### Unistream AR memorizes 32 utterances in ~400 steps AND renders them correctly (2026-08-24)
`memorize32_uni_ar_0` (AR, unistream, no duration token, LR 1e-4 constant, 32
samples, batch 8): train `voice_unit_accuracy` 0.819 @100, 0.984 @200, **1.000 @400**, held
to 1100; `voice_unit_ce_loss_norm` 1.028 -> 1.4e-4. Owner reports all 4 TB renders sound
perfect. Held-out val is unmoved and drifting the wrong way (unit acc 0.0124, CE 1.19 ->
1.24 between 500 and 1000) — pure memorization with zero generalization, which is the
expected and correct outcome at n=32.

Note on epochs: batch 8 over 32 samples is 4 steps/epoch, so step 400 is **100 epochs**, not
a handful. It is fast in wall-clock (minutes), not fast in passes over the data.

**This retires the open generation-path question.** The NAR arm (`memorize32_0`)
also reached train accuracy 1.0 — but at step ~1400-4000, and its free-running renders were
still nonsense for some samples at 17.5k (idx 3). AR reaches the same teacher-forced ceiling
3.5-10x sooner AND its renders are correct. So the earlier train-1.0-but-audio-nonsense
discrepancy was specific to the NAR MaskGIT generation path, not to what was learned. It
invalidates NAR generation numbers, not NAR training, exactly as suspected.

### ~~Stage 5: does bistream memorize faster than unistream?~~ RETIRED before running (2026-08-24)
Do not run the bistream memorization arm as a discriminating test. Two independent reasons:

1. **No headroom.** The unistream control is at accuracy 1.000 by step 400 with perfect
   renders. Nothing can beat that by enough to read.
2. **Wrong instrument, and this was knowable in advance.** The text-dependence probe earlier
   the same day established that at n=32 the model solves the task by RETRIEVAL — under
   `roll` its predictions match the SOURCE utterance's units at 0.825, with text acting as a
   key. Retrieval does not need alignment. Bistream's claim is about ALIGNMENT helping
   GENERALIZATION, so on a task solvable by retrieval it should be expected to tie, and a
   tie would have taught nothing. The memorization race was proposed (in the plan, and
   endorsed in-session) before that probe existed, and should have been withdrawn when it
   landed.

The discriminating measurement is the one bistream is actually claimed to move:
teacher-forced `text_delta` / text-attributed fraction at matched step on the full corpus,
against AR unistream @23k (+0.0271 / 0.305) and the teacher ceiling (+0.0594 / 0.463).

### Bistream chunk interleaving is BUILT and smoke-tested; nothing measured yet (2026-08-24)
Commits 685364e, fc0697e, ec45e6a, 90d5461, f44c3c1. Ten work items from
`docs/plans/bistream-inner-monologue.md`, all landed, with 11 new tests beside the existing
122. **No claim about whether it helps** — that needs the matched memorization arms
(stage 5) and then a full run.

Design decisions that turned out to matter more than expected:

- **Chunks are a SLICING of the utterance's stream, not a new batch axis.** The collator
  emits the utterance as one expanded stream (content frames with terminals inline) plus a
  `(utt_idx, start, length)` map. That single choice made work items 2 and 7 no-ops: the
  uninterleaver's ascending-order pack already reassembles the utterance, and fill_token is
  just another class in the existing (B, T) target. The alternative — chunks on the `n`
  axis — would have broken EOV, speaker conditioning, and emitted one utterance as N clips.
- **The unistream layout is the m=1 case**, byte-for-byte, rather than a parallel path.
- **CosyVoice 2's eligibility gate is wrong for our corpus.** They require
  `speech_len/text_len > s/k`; at s/k = 6 against our measured 5.9 frames/token that rejects
  ~half the data. The exact condition is `n_frames > (n_text_chunks - 1) * s`. Measured at
  k=5/s=30 on 64 samples: ~75% eligible, interior chunks exactly 30 frames, last chunk 10-60
  (median 27).

Two train/inference parity details are recorded in the plan because they are silent if wrong:
the chunk boundary must feed a zero FEATURE (not None, which skips the prelude), and the
prelude/coda KV caches must NOT reset across a boundary.

⚠️ **`--bistream_text_loss` was inert on the first try** — the same `include_text=False`
trap that silently killed the NAR duration head. Implemented correctly in `compute_loss`,
but text targets are never built for a voice-only run, so there was nothing to un-mask and
`train/text_loss_norm` never appeared while every other metric looked healthy. Caught only
because the smoke run checked for the metric rather than assuming it. Any future flag
supervising the TEXT stream from a voice-only run has to be added to the gate at
`training.py:638`.

### The shuffled-text TRAINING control was confounded by design — do not run it (2026-08-24)
`docs/plans/bistream-inner-monologue.md` named a shuffled-transcript training arm as the
prerequisite gating the whole bistream build. It would have been **uninformative**, and worse,
uninformative in the direction of a false negative. Shuffling pairs each voice sample with a
*fixed* wrong transcript, so `(frame_len, text_len)` stays a unique index and the arm memorizes
too — reported as "text is decorative", which the ablation above shows is false. The
inference-time ablation replaces it: same question, minutes instead of hours, and it isolates
content from layout in a way the training arm structurally cannot.

---

## Bistream buys DURATION, not content (2026-09-15, ESTABLISHED)

Matched-step four-arm probe at **step 68000**, `world_voice_ar_diagnostics.py`, n=128 TF /
gen_n=48, `--voice_ras_win 10 --voice_ras_tau 0.1`, M-RoPE 7.5/text. Arms:
`cosyvoice2_smollm2_libriheavy_ar_flat_lr_ema_mrope75_0` (unistream, the prior best) vs
`cosyvoice2_smollm2_libriheavy_bistream_scratch_0` (from-scratch, 50% bistream), each at
T=0.6 and greedy. Raw reports: `eval_output/world_voice_bistream_ab/`.

⚠️ **PROTOCOL: all four arms decode UNISTREAM.** The bistream checkpoint was evaluated in
unistream layout, because the chunked path makes the text-shuffle ablation structurally
invalid (a different transcript yields a different chunk plan and different voice padding, so
the two forwards stop being positionally comparable). The TensorBoard renders the ear judges
are CHUNKED. An n=8 chunked smoke gave acc_real 0.1925 vs 0.1734 unistream, so the chunked
numbers are probably better than what is tabulated here — **the content conclusion below is
therefore conservative, not confounded, but the chunked arm is still unmeasured at n=48.**

| metric | uni T=.6 | bi T=.6 | uni greedy | bi greedy | GT / ceiling |
|---|---|---|---|---|---|
| acc_real (TF) | 0.1791 | 0.1734 | 0.1791 | 0.1729 | 0.2110 n-gram |
| ppl_real | 42.51 | 46.56 | 42.51 | 46.65 | — |
| text_delta all-pos | +0.0642 | +0.0561 | +0.0639 | +0.0543 | +0.0594 teacher |
| text-attributed frac | 0.358 | 0.323 | 0.357 | 0.314 | 0.463 teacher |
| eov_position_acc | 0.4766 | 0.4453 | 0.4688 | 0.4453 | — |
| length r | +0.653 | +0.727 | +0.671 | **+0.763** | +0.744 |
| length hit rate | 75.0% | 83.3% | 83.3% | 83.3% | — |
| collapsed (<50% GT) | 4.2% | 4.2% | 4.2% | **0.0%** | — |
| len_min | 5 | 32 | 38 | **67** | 74 |
| adj_repeat_rate | 0.0378 | 0.0639 | 0.0127 | 0.0096 | 0.0792 |
| distinct_bigram_ratio | 0.806 | 0.735 | 0.849 | 0.815 | 0.869 |

**Content: bistream is no better and slightly worse.** Lower acc_real, higher perplexity,
lower all-position `text_delta`, lower text-attributed fraction — consistently, at both
temperatures. Chunk-interleaving made text->speech alignment local and monotonic without
making text bind harder.

**The text horizon did not widen, which was the entire reason to try it.** Neither arm's
position-resolved delta ever falls below the 0.01 threshold, so the horizon metric does not
discriminate; but bistream's TAIL deltas are *lower* (+0.0543 vs +0.0673 at 128+ frames).
Whatever bistream fixed, it was not the decay of text influence into the utterance.

**Duration is now essentially solved, and `bi_greedy` is the arm that does it**: r=+0.763
*above* the GT ceiling of +0.744, zero collapsed utterances, and a shortest generation of 67
frames against a GT minimum of 74. The unistream T=0.6 arm by contrast produced a 5-frame
utterance. This corroborates the earlier read that duration control was nearly solved by CE
and closes the remaining gap.

**Greedy beats T=0.6 on duration for BOTH models** (uni hit rate 83.3% vs 75.0%, median
length error 0.09 vs 0.17, zero overruns). The cost is naturalness: greedy `adj_repeat_rate`
is 0.0096-0.0127 against GT's 0.0792 — it essentially never repeats a unit, and real speech
does. Sampling buys repetition structure and spends length discipline.

**The user-audible "natural stutter" of the bistream run has a number.** Adjacent-repeat in
the first 32 frames: bistream 0.0793 vs unistream 0.0457 (GT 0.0894). Bistream's aggregate
repeat rate is the closest of any arm to GT, but its bigram diversity is well BELOW GT (0.735
vs 0.869). It repeats locally in a GT-plausible way while drawing from a narrower vocabulary,
and the excess is concentrated at ONSET, not drift — so it is a conditioning/decoder effect,
not exposure bias.

### Accidental replicate: early-frame TF metrics carry ~0.006 of jitter (2026-09-15, ESTABLISHED)

Sections 1/1b/2 are teacher-forced and temperature-independent, so the T=0.6 and greedy arms
of the SAME checkpoint are exact replicates of those metrics. They did not reproduce exactly:

| | uni T=.6 | uni greedy | bi T=.6 | bi greedy |
|---|---|---|---|---|
| acc_real (n~24k positions) | 0.1791 | 0.1791 | 0.1734 | 0.1729 |
| early_acc_real (n=1024) | 0.2969 | 0.3047 | 0.3066 | 0.3086 |
| early_text_delta | +0.0293 | +0.0352 | +0.0332 | +0.0332 |
| eov_position_acc | 0.4766 | 0.4688 | 0.4453 | 0.4453 |

All-position `acc_real` and `ppl_real` reproduce to 4 and 2 decimals; the n=1024 early-frame
subset moves by up to 0.008 in `early_acc_real` and **0.0059 in `early_text_delta`**.
Consistent with nondeterministic GPU reductions flipping near-tie argmaxes: stable in
aggregate, visible at n=1024.

**Consequence: the uni-vs-bi `early_text_delta` difference (~0.004) is SMALLER than this
jitter and must not be read as a result.** The stated 95% CIs are sampling CIs and do not
include this term. Any future `early_text_delta` comparison needs replicates, not one run.

### RETRACTION CANDIDATE: "free-running argmax repeats the previous unit ~99% of the time"

Both greedy arms here show `adj_repeat_rate` of 0.0096-0.0127 — about 1%, the opposite of the
earlier claim, and BELOW ground truth rather than above it. Either the behaviour is
checkpoint-specific and has since changed, or the original measurement was wrong. Not struck
yet because the original entry has not been re-located and re-read; do that before editing it.
Flagged here so the claim is not cited again in the meantime.

## The RAS resample inherits the pick's temperature — and is DISCONTINUOUS at T=0 (2026-09-15, ESTABLISHED)

**This is a decode bug, not a model property, and it has been contaminating every
temperature comparison in this direction.**

In `world_model.py` the voice sampler builds ONE tensor and uses it twice:

```python
scaled = (logits.float() / voice_temperature
          if voice_temperature > 0.0 else logits.float())   # <-- T=0 keeps RAW logits
...
banned = scaled.clone()        # the RAS resample draws from the SAME tensor
banned[int(unit_id)] = -inf
unit_id = torch.multinomial(torch.softmax(banned, -1), 1)[0]
```

So the RAS resample temperature is not independent of the pick temperature, and it is
discontinuous at zero: at **T=0 the resample runs at an effective temperature of 1.0**
(raw logits), while at **T=0.2 it runs 5x SHARPER than any pick ever is**. Resample
sharpness is therefore non-monotonic in T — maximal just above zero, relaxing back toward
1.0 as T rises.

A near-deterministic second-best pick locks into cycles that never reach EOV. Termination
tracks resample sharpness exactly. Step 68000, EMA weights, unistream decode, n=12 paired
samples, `--ras_win 10 --ras_tau 0.1`, budget 250 frames
(`eval_output/world_voice_temp_sweep/`):

| T | uni capped | uni LCS | bi capped | bi LCS |
|---|---|---|---|---|
| 0.0 | 0/12 | 0.6188 | 0/12 | 0.7270 |
| 0.2 | **12/12** | 0.2016 | **10/12** | 0.1785 |
| 0.4 | 7/12 | 0.4422 | 5/12 | 0.4524 |
| 0.5 | 5/12 | 0.4598 | 0/12 | 0.6621 |
| 0.6 | 3/12 | 0.5920 | 0/12 | 0.6398 |
| 0.7 | 0/12 | 0.6076 | 0/12 | 0.5973 |
| 0.8 | 0/12 | 0.5477 | 0/12 | 0.7206 |
| 1.0 | 2/12 | 0.4724 | 0/12 | 0.4952 |

**"Greedy" on this codebase is NOT greedy decoding.** It is argmax picks plus a T=1.0
multinomial resample on every RAS trigger. With `ras_win=10 tau=0.1` the trigger is
`rep >= win*tau` = 1, i.e. ANY repeat within the last 10 units, so the resample fires
constantly. That is why greedy looks so strong, and why it under-repeats badly
(adj_repeat 0.0096-0.0127 vs GT 0.0792 — RAS at tau=0.1 bans legitimate repetition).

**CAUSAL CONFIRMATION.** Built `voice_ras_temperature` (generate()) / `--ras_temperature`
(cosyvoice_wer_eval), which decouples the resample. Default None keeps the inherited
behaviour. Unistream EMA, T=0.2, everything else identical:

| T=0.2 arm | capped | LCS | WER | hyp/ref | median len |
|---|---|---|---|---|---|
| default (resample inherits 0.2) | 12/12 | 0.2016 | 0.8150 | 0.375 | 250 |
| `--ras_temperature 1.0` | **1/12** | **0.5868** | 0.4450 | 0.805 | 208 |
| (T=0.0 reference) | 0/12 | 0.6188 | 0.4000 | 0.850 | 188 |

One argument takes the worst arm in the sweep to within noise of greedy. The trough is
fully explained.

**Where termination is ALREADY fine, the fix is modest.** Bistream EMA at T=0.6 (0/12 capped
either way), both arms sharing `--skip_ceiling` so the RNG streams match:

| bi T=0.6 | capped | LCS | WER | median len |
|---|---|---|---|---|
| inherited resample | 0/12 | 0.6598 | 0.3850 | — |
| `--ras_temperature 1.0` | 0/12 | **0.7051** | **0.3050** | 175 |

+0.045 LCS at ~2x the applicable noise (see the RNG-stream figure below), with WER moving
the same direction. Real but modest. **The fix is decisive where termination is BROKEN and
marginal where it is not** — so for a run already at 0/12 caps it is a small positive, not
the large recovery the unistream T=0.2 numbers imply on their own.

⚠️ **The training default T=0.6 sits INSIDE the trough** (uni 3/12 capped). Every TB render
judged by ear at T=0.6 has been sampling from the pathological band. This is a plausible
contributor to the long-standing "early stopping" complaint, independent of architecture.

⚠️ **Retroactive contamination.** Any prior comparison of greedy against a mid-range
temperature compared out-of-band against in-band, NOT temperature per se. That includes the
2026-09-15 `bi_t06` vs `bi_greedy` audio arms. Unit-space teacher-forced metrics are
sampler-free and unaffected.

**Default inertness verified EMPIRICALLY, not just by reading the branch**: `ema_bi_uni_06`
ran before the edit (14:17) and `bi_t06` after it (14:54) at identical config — LCS
0.639817, WER 0.400000, CER 0.263355 on both. Identical to 6 dp.

### NOISE FLOOR: LCS at n=12 carries ~0.12 of trajectory spread across temperature (2026-09-15, ESTABLISHED)

Re-running an identical config reproduces to 6 decimals (above), so there is no RNG noise
between processes. But ADJACENT TEMPERATURES follow different sampling trajectories, and
among bistream's four clean-termination points T=0.5/0.6/0.7/0.8 the LCS runs
0.662 / 0.640 / 0.597 / 0.721 — a **0.124 spread with zero caps throughout**.

**Consequence: a single-point LCS difference below ~0.12 at n=12 is not a result**, even
though the arms are paired on the same utterances and each is individually deterministic.
Determinism under replication does NOT imply smoothness in T. Cap counts (integers over 12
with monotonic structure) are the more trustworthy readout at this n.

⚠️ **`--skip_ceiling` IS NOT A PURE SPEED FLAG — it changes what you measure.** The CosyVoice 2
flow decoder draws `torch.randn_like(mu)` per call (`cosyvoice/flow/flow_matching.py:56`, and
HiFiGAN draws again), so EVERY `dec.decode()` advances the global RNG. The per-sample loop is
`generate -> decode(gen) -> decode(trunc) -> decode(ceiling)`, so sample i+1's GENERATION
consumes the stream left by sample i's CEILING decode. Dropping the ceiling pass therefore
shifts the sampling trajectory of every later sample. An A/B where one arm has it and the
other does not is confounded — this was caught only because a determinism control failed to
reproduce (0.659834 vs 0.639817 at identical model/config/temperature).

**Three distinct noise scales, measured 2026-09-15, n=12:**

| comparison | LCS spread |
|---|---|
| identical config, identical RNG stream | 0.000000 (6 dp) |
| identical config, DIFFERENT RNG stream | **0.020 - 0.055** |
| adjacent temperature | ~0.124 |

Pick the yardstick that matches the perturbation: an RNG-stream shift for a same-temperature
A/B, the adjacent-temperature spread for anything that moves T.

⚠️ **The RNG-stream figure was first recorded as ~0.020 from ONE pair and is understated.** A
second pair — `bi_t00` (0.7270, with ceiling) vs `bi_t00_tau01_rtnone` (0.6719, identical
except `--skip_ceiling`) — differs by **0.055**. Treat 0.02-0.055 as the band. Consequence:
the `--ras_temperature` result at T=0.6 (+0.045) sits INSIDE it and is NOT established at
n=12; the unistream T=0.2 result (12/12 -> 1/12 capped) is untouched, because an integer cap
count moving by 11 of 12 is not a noise-scale effect.

### REVISION to "Bistream buys DURATION, not content" (2026-09-15)

~~The generation-side half of that entry is **STRUCK**.~~ It concluded bistream was worse on
content from raw-weight arms at T=0.6 — a single point inside the trough, on non-EMA
weights. With the full curve:

- **Bistream terminates over a far wider band — SOLID.** 0/12 caps at T>=0.5; unistream
  still caps at 0.5 (5/12) and 0.6 (3/12). Totals across the sweep: uni 29 caps, bi 15.
- **Bistream's content advantage — SUGGESTIVE ONLY.** Mean LCS over clean-termination
  points ~0.67 vs ~0.56, but no single comparison clears the 0.12 spread, and unistream
  actually edges bistream at T=0.7 (0.6076 vs 0.5973).
- **Teacher-forced unit metrics still favour unistream slightly** (acc_real 0.1734 vs
  0.1791, ppl 46.56 vs 42.51). These are sampler-free and stand.

The reading that fits both: chunk-interleaving did not make the trunk better at PREDICTING
units — it made generation less likely to DERAIL when a sampled unit is wrong. TF accuracy
is blind to that; cap counts are not.

The DURATION half of the original entry (length r=+0.763 above the GT ceiling, zero
collapsed utterances at bi_greedy) is unaffected and stands.

## campplus embedding alone does NOT preserve speaker identity — the PROMPT does (2026-09-15, ESTABLISHED)

**Ear-confirmed by the user, who is the arbiter for speaker identity.** Same ground-truth
units, same frozen CosyVoice 2 decoder, same 192-d campplus embedding — the ONLY difference
is whether `prompt_ids`/`prompt_feat` were supplied:

- **embedding-only**: rendered a MALE reference speaker as **entirely female**.
- **with prompt** (`logs/speaker_male.wav`, 131 units + (260,80) mel): correct speaker.

Renders: `eval_output/world_voice_prompt_check/{embed_only,with_prompt}.wav`. Mean |diff|
0.064 over the common span, identical duration; prompted render is 2.5x quieter
(rms 0.105 -> 0.041), consistent with the prompt pulling the output toward the reference's
own channel/level.

The 192-d vector is a GLOBAL SUMMARY that modulates through `spk_embed_affine_layer`. It is
evidently not enough to pin identity: the flow decoder falls back toward its own prior for
everything the vector does not specify, and that fallback can cross sex.

⚠️ **This undermines the `_static_speaker` control render.** That second render exists to
separate speaker drift from off-manifold units — "a wandering GT render with a STABLE static
render means the units are at fault, not the embedding". It has always been embedding-only,
so a wandering static render may have been the CONDITIONING failing, not the units. Do not
cite pre-2026-09-15 static-speaker drift observations as evidence about unit quality.

⚠️ **Cache-based evals CANNOT do this today.** The val shards store
`unit_ids / speaker_embeddings / speaker_ids / text / token_ids / feature_lengths` — no
waveform and no mel — so `prompt_feat` cannot be built from the cache. Every WER/LCS number
in this file is embedding-only. **Content metrics are unaffected** (Whisper transcribes the
words regardless of the rendered voice's identity), but any claim about speaker similarity
or timbre from a cache-based eval is void. Fixing it needs a preprocessing change storing one
reference mel + its unit ids per speaker.

✅ **BUILT**: `--viz_voice_prompt_audio <wav>` on the world trainer. Resolves `prompt_ids`
(CosyVoice 2 `speech_tokenizer_v2.onnx`) and `prompt_feat` (`decoder.prompt_mel()`) lazily,
caches them, and latches failures so a viz problem cannot kill a run. When
`--static_speaker_embedding_path` is absent the campplus embedding is DERIVED FROM THE SAME
WAV, which also removes the footgun of an embedding and a prompt clip that are different
speakers (nothing checked this before). Verified end-to-end: 2:1 mel-to-unit ratio honoured,
derived norm 13.377 matching an independently extracted `.pt`, and the render demonstrably
differs from embedding-only.

Requiring a reference clip costs nothing: one is needed to produce the embedding anyway.

## RAS tau: the axis is NOT flat — it interacts with the resample temperature (2026-09-15, ESTABLISHED)

Bistream EMA ck68000, n=12, `ras_win=10`, all arms `--skip_ceiling` so the RNG streams match
(`eval_output/world_voice_tau_sweep/`). Two regimes: greedy (resample inherits an effective
1.0) and T=0.6 with the resample pinned to 1.0.

| tau | greedy capped / LCS | T=0.6 + rt1.0 capped / LCS |
|---|---|---|
| 0.0 | **12/12 / 0.0000** | 3/12 / 0.5327 |
| 0.1 | **0/12** / 0.6719 | **0/12** / 0.7051 |
| 0.2 | 2/12 / 0.6705 | 1/12 / 0.6820 |
| 0.3 | 2/12 / 0.6376 | 1/12 / **0.7442** |
| 0.5 | 4/12 / 0.6547 | 0/12 / 0.5978 |

**RAS OFF (tau=0) at greedy is TOTAL COLLAPSE**: 12/12 budget-capped, LCS 0.0000, WER 1.0000,
and EMPTY Whisper transcripts — the degenerate repeat-the-previous-unit loop rendering as a
held tone. This **VINDICATES the old "greedy decoding puts 96% adjacent repeats" finding**,
which an earlier 2026-09-15 note flagged as a retraction candidate on the strength of
measuring adj_repeat 0.0096-0.0127. Both are correct and describe DIFFERENT SAMPLERS: the
original is greedy WITHOUT RAS, every 2026-09-15 measurement had `--ras_win 10` on. **The
retraction candidate is withdrawn; the original entry stands as written.**

RAS is therefore not a mild anti-repetition nudge — it is the only thing between this model
and total collapse under greedy. "Greedy is incredibly good" is entirely RAS's doing.

⚠️ **CORRECTION to an intermediate conclusion in this session.** After the greedy regime alone
(tau 0.1/0.2/0.3 = 0.672/0.671/0.638, all inside noise) the tau axis was declared "flat" and
"closed". That was an artifact of holding the wrong variable fixed. With the resample
decoupled at 1.0, **tau=0.3 goes from worst (0.6376) to best (0.7442)**. tau sets how OFTEN
RAS fires; ras_temperature sets HOW it resamples. Sweeping either alone reads as flat.
Never sweep one without the other.

**Practical picks.** tau=0.1 is the only value giving 0/12 caps in BOTH regimes, so it is the
safe default. Content differences across tau 0.1-0.3 span 0.062 within the rt1.0 regime and
0.034 within greedy — both inside the 0.02-0.055 RNG band, so the content ranking is NOT
resolved at n=12. tau extremes (0.0 and 0.5) are genuinely worse in both regimes.

### OPEN: greedy vs T=0.6 + ras_temperature 1.0 is UNRESOLVED at n=12

Both reach 0/12 caps at tau=0.1, so the reliable metric cannot separate them. LCS favours the
decoupled regime by +0.033 (tau 0.1) to +0.072 (tau 0.3) — straddling the noise band.

Current viz default is **greedy, tau 0.1** (`c9d9a1d`), chosen because it is the simpler
configuration and ties on the metric that is trustworthy at this n.

An argument the metrics cannot see: greedy under-repeats badly (adj_repeat 0.0096-0.0127 vs
GT 0.0792) and ASR word-overlap is blind to repetition naturalness, which the user HAS
reacted to by ear. If the decoupled regime is confirmed at larger n, it would also restore
sampled variation at equal-or-better content. **Resolve with n=48 on the two candidates
before changing the default** — this session already produced one wrong conclusion (the tau
axis) from reasoning past the noise floor.

## ⚠️ SEED VARIANCE IS THE DOMINANT TERM: single-seed LCS comparisons are worthless (2026-09-15, ESTABLISHED)

**Read this before running or citing any world-voice generation comparison.**

Decision run: bistream EMA ck68000, n=48, two seeds per config, all arms `--skip_ceiling`
(`eval_output/world_voice_n48_decision/`, analysis `scripts_local/summarize_n48_decision.py`).

| config | seed 1 | seed 2 | seed-averaged | WER | caps |
|---|---|---|---|---|---|
| greedy, tau 0.1 | 0.6904 | 0.7598 | **0.7251** | 0.3057 | 5/96 |
| T=0.6 + ras_temp 1.0, tau 0.1 | 0.7605 | 0.6980 | **0.7293** | 0.2943 | 3/96 |

**Seed-averaged difference: +0.0041 LCS. The two configs are EQUIVALENT.**

Per-seed spread is **0.069** (greedy) and **0.063** (decoupled) — each config's two seeds
straddle the other's. A single-seed comparison of these arms gave +0.0684 in favour of the
decoupled regime with a paired bootstrap 95% CI of [+0.0240, +0.1088] that EXCLUDED ZERO.
That "significant" result was pure seed luck: A's bad seed met B's good seed.

⚠️ **The paired bootstrap CI is NOT a valid significance test here.** It captures variance
across UTTERANCES and is blind to variance across SEEDS, and the seed term dominates. Proof
from the same run: the seed replicate A2-A gives +0.0700 CI [+0.0266, +0.1124] and B2-B gives
-0.0622 CI [-0.1116, -0.0127] — two runs of an IDENTICAL config, each "significant", in
OPPOSITE directions.

**Protocol going forward:** a generation comparison needs >= 2 seeds per arm and must compare
SEED-AVERAGED means. n alone does not fix this — going from n=12 to n=48 did not shrink the
seed term. Cap counts (integers, mechanistic) remain the more trustworthy readout.

**Casualties — every single-seed LCS claim from 2026-09-15 is void**, including:
- "bistream buys duration not content" generation-side numbers (already struck)
- the `--ras_temperature` +0.045 at T=0.6 (already downgraded)
- "tau 0.3 is best" at n=12 (already refuted at n=48)
- "bistream leads at every temperature" from the temperature sweep
The unistream T=0.2 cap result (12/12 -> 1/12) SURVIVES: an integer count moving 11 of 12 is
not a noise-scale effect.

### Bistream vs unistream on CONTENT: NULL (2026-09-15, ESTABLISHED)

Paired at n=48, greedy tau 0.1, EMA, same 48 utterances: **-0.0038, 95% CI
[-0.0685, +0.0652]**. Aggregate LCS 0.6904 (bi) vs 0.6940 (uni); caps 2 vs 1. Unistream's
0.6940 sits INSIDE bistream's own seed range (0.6904-0.7598).

This session produced three different answers to this question before getting here: raw
weights at T=0.6 said bistream was much WORSE (LCS 0.515 vs 0.709), EMA at greedy said much
BETTER (0.727 vs 0.619), n=48 seed-aware says IDENTICAL. The first two were n=12 single-seed.

**What survives about bistream:** duration control (length r=+0.763 vs GT ceiling +0.744,
zero collapsed utterances) is a large effect and stands. Termination robustness across the
temperature band (0/12 caps at T=0.5/0.6 vs unistream's 5/12 and 3/12) is n=12 single-seed
and NOT yet verified. Content: no advantage.

## Bistream buys SAMPLER INSENSITIVITY, not better speech (2026-09-15, ESTABLISHED)

Full 2x2, ck68000 EMA, n=48, **2 seeds per cell**, tau 0.1, `--skip_ceiling` throughout
(`eval_output/world_voice_n48_decision/`).

| model | config | seed1 | seed2 | mean LCS | WER | caps | seed spread |
|---|---|---|---|---|---|---|---|
| bistream | greedy | 0.6904 | 0.7598 | 0.7251 | 0.3057 | 5/96 | 0.069 |
| bistream | T=0.6 + rt1.0 | 0.7605 | 0.6980 | 0.7293 | 0.2943 | 3/96 | 0.063 |
| unistream | greedy | 0.6940 | 0.7178 | 0.7059 | 0.3092 | 4/96 | 0.024 |
| unistream | T=0.6 + rt1.0 | 0.6397 | 0.6562 | **0.6480** | **0.3971** | 4/96 | 0.017 |

| | greedy -> T=0.6+rt1.0 |
|---|---|
| bistream | **+0.0041** (no detectable change) |
| unistream | **-0.0579** (~3x its own seed spread; WER +0.088 agrees) |
| interaction | **+0.0620** |

**Unistream degrades under the decoupled-resample regime; bistream does not.** The unistream
effect is readable because that row's seed spread is 0.017-0.024 — THREE TIMES TIGHTER than
bistream's 0.063-0.069, consistently across both cells. Two independent metrics move together
(LCS -0.058, WER +0.088).

**Bistream's margin over unistream is entirely conditional on the sampler:** +0.019 at greedy
(nothing) vs +0.081 at T=0.6+rt1.0 (where unistream falls apart). Combined with the
established content NULL at greedy, the reading is that chunk-interleaving did not make the
model better at producing speech — it made the model **insensitive to how it is sampled**.
That is independently consistent with the n=12 temperature sweep, where bistream held 0/12
caps at T=0.5/0.6 while unistream capped 5/12 and 3/12.

⚠️ **OBSERVATION, NOT A FINDING — the variance asymmetry runs the other way.** Bistream is
~3x LESS stable seed-to-seed at a FIXED config while being MORE stable ACROSS configs. Two
seeds cannot establish a variance ratio; the consistency across both cells is what makes it
worth checking. If real it is an odd result for an architecture whose claimed virtue is
robustness, and it also means bistream needs MORE seeds than unistream to measure at all.

**Practical:** greedy remains the default — it is the configuration where BOTH models do
well, so it is the one that does not depend on the bistream/unistream choice being settled.

### EMA vs raw on generation: NULL (2026-09-15, ESTABLISHED)

Bistream greedy, n=48: raw 0.7339 / WER 0.3011 / 0 caps, vs EMA seed-averaged 0.7251 /
0.3057 / 5 per 96. Paired EMA-raw = -0.0399, 95% CI [-0.0972, +0.0180]. Raw's value sits
INSIDE EMA's own seed range (0.6904-0.7598).

**Closes the loose end** left by the n=12 arms, which had EMA helping +0.125 at T=0.6 and
hurting -0.104 at greedy — both seed noise. It also defuses the protocol worry raised earlier
that day: eval scripts load RAW `pytorch_model.bin` while the training viz renders under EMA.
For CONTENT that mismatch is immaterial. (It remains real for anything measuring the weights
themselves.)

Asterisk: raw posted 0/48 caps against EMA's 2 and 3 — the cleanest termination of any arm —
but that is one seed and cap counts this small need a replicate.

## BEST MEASURED CONFIG (2026-09-15): unistream ck90000-ema + greedy

```
checkpoint: cosyvoice2_smollm2_libriheavy_ar_flat_lr_ema_mrope75_0/checkpoint-90000-ema
sampling:   --voice_temperature 0.0 --ras_win 10 --ras_tau 0.1
            LCS 0.7519   WER 0.2621   (n=48, 2 seeds, --skip_ceiling)
```

ck90000 is that run's eval-loss minimum (0.41527, epoch 6.00). Sweep:
`eval_output/world_voice_ck90k/`.

| config | LCS | WER | hyp/ref | caps | collapsed | seed spread |
|---|---|---|---|---|---|---|
| **greedy** | **0.7519** | **0.2621** | 0.953 | 5/96 | 7/96 | 0.017 |
| T=0.6 + rt1.0 | 0.7103 | 0.3201 | 0.944 | **1/96** | 6/96 | 0.057 |
| T=0.7 | 0.6933 | 0.3592 | **0.974** | 3/96 | **4/96** | 0.031 |

Greedy wins on BOTH metrics; the margin over T=0.7 is 2-3x seed spread. Unistream preferring
greedy over the decoupled-resample regime REPLICATES the ck68000 result (0.7059 vs 0.6480),
so it is a property of the model, not one checkpoint.

**Supersedes the earlier "best config" answer.** Full ranking of everything measured at n=48
with 2 seeds:

| rank | checkpoint + config | LCS | WER |
|---|---|---|---|
| 1 | **unistream ck90000 greedy** | **0.7519** | **0.2621** |
| 2 | bistream ck68000 T=0.6+rt1.0 | 0.7293 | 0.2943 |
| 3 | bistream ck68000 greedy | 0.7251 | 0.3057 |
| 4 | unistream ck68000 greedy | 0.7059 | 0.3092 |

⚠️ **The bistream-vs-unistream comparison earlier that day was handicapped**: both models were
evaluated at ck68000, the BISTREAM run's eval minimum. Unistream's own minimum is ck90000, and
at it unistream beats every bistream arm. ck90000 vs ck68000 greedy is +0.046 against a
unistream seed spread of 0.017-0.024, so the checkpoint difference is real. This does NOT
revive a bistream content advantage (the paired null at matched step stands) but it does mean
**no bistream arm has been measured at ITS best step past 68000** — the run stopped at 75000
having turned at 68000.

⚠️ **Greedy wins the MEAN while failing hardest on the TAIL**: most caps (5/96) and most
collapsed utterances (7/96) of the three. Per-utterance inspection of `greedy_s1` shows the
distribution is bimodal, not centred — 2 of 46 produced <30 frames against 100+ frame targets
("Indeed." for a 183-frame reference). LCS 0.75 is an average over near-perfect and
catastrophic, which is what the ear has always reported as "either works or it doesn't".

⚠️ LCS-recall ignores insertions: `gen_0026` scores a perfect 1.00 while audibly stuttering
"in times in times past". These numbers rank configs; they do not describe quality.

## Premature collapse is a COHERENCE failure, not a confident stop (2026-09-15, ESTABLISHED)

Measured with `scripts_local/voice_eov_entropy_probe.py` at ck90000-ema, greedy + RAS w=10
tau=0.1, n=96 (2 seeds). Distribution shape AT THE STEP EOV FIRED:

| | collapsed (n=5) | normal (n=89) |
|---|---|---|
| entropy | **5.87 nats** | 2.59 |
| p_max | **0.064** | 0.287 |
| p(EOV) | **0.031** | 0.219 |
| EOV was argmax | 3/5 | 63/89 |

5.87 nats is ~350 effective candidates out of 6562. The model has NOT decided to stop -- it
has lost the plot, everything is roughly equally unlikely, and EOV wins by accident. In 3 of
5 it is "argmax" only in that degenerate sense. `max_p_eov_before_end` agrees: collapsed
utterances never had a confident stop moment (median 0.021 vs 0.066).

⚠️ **This RULES OUT min-p and top-a for this failure.** Their threshold is `min_p * p_max`,
and p_max is itself tiny exactly when collapse happens (0.064), so the cutoff lands near
0.006 and EOV at 0.031 passes. Every RELATIVE truncation scheme is blind here by
construction. An ABSOLUTE floor is what discriminates.

### The EOV confidence guard: collapses 7/96 -> 1/96 (2026-09-15, ESTABLISHED — but read the caveat)

`voice_eov_min_prob` + `voice_eov_max_entropy` (committed 584c569, flags 229270c) ban EOV
when p(EOV) is below the first AND entropy above the second. Threshold sizing on the n=5
collapses: p(EOV)<0.10 alone blocks 5/5 but breaks 27/89 good stops; entropy>3.0 alone blocks
5/5 and breaks 22/89; **both together block 5/5 and break 13/89**. They are not redundant --
some good stops are low-confidence-but-sharp, others diffuse-but-confident, only collapses
are both.

A/B at ck90000-ema greedy, n=48 x 2 seeds (`eval_output/world_voice/`):

| | LCS | WER | hyp/ref | caps | collapsed |
|---|---|---|---|---|---|
| baseline | 0.7519 | 0.2621 | 0.953 | 5/96 | **7/96** |
| eov_guard (0.10 / 3.0) | 0.7667 | 0.2569 | 0.986 | 5/96 | **1/96** |

Replicated in both seeds (4->1, 3->0). **Budget-caps unchanged at 5/96** — the guard does not
convert collapses into runaways in aggregate. LCS +0.0147 / WER -0.0052 are INSIDE the seed
spread (0.013-0.017) and are NOT claimed.

⚠️ **The collapse count OVERSTATES the win — per-utterance inspection of seed 1's four
collapses:**

| idx | ref | baseline | guard | verdict |
|---|---|---|---|---|
| 34 | 109f | 16f "I hand turned a" | 109f "The announcement was re-scvering caft pi." | genuine rescue |
| 31 | 183f | 26f "Indeed." | 250f "Indeed no You know I understand it..." | better content, ran to budget |
| 22 | 202f | 56f "Gara Sutjiks, which he touched." | 144f "улиbnil-gil ... Aladhey Copus" | **longer GIBBERISH** |
| 32 | 168f | 58f "Look, my old Duke!" | 44f "Nook, my lord." | still collapsed |

One clean rescue, one partial, one lateral-into-garbage, one null. That is why the LCS gain is
only +0.015. **The collapse metric counts LENGTH, and length is exactly what the guard
manipulates**, so it reads closer to "did the intervention fire" than "did quality improve".

**Mechanistic reading:** the guard removes the EXIT without restoring COHERENCE. idx 22 shows
what that means -- with EOV blocked the model keeps generating from the flat distribution and
emits multilingual junk. The underlying failure (losing the plot mid-utterance) is NOT a
sampling problem and no sampler addresses it.

**Status:** keep the guard (nothing got worse in aggregate; the tail improved), but thresholds
0.10/3.0 are tuned on n=5 and are provisional. The MECHANISM is the durable part. Ear check on
the A/B pairs is pending.

**Generality:** reads only the current distribution -- no reference transcript, no text-length
prior -- so unlike ASR-rerank and `voice_min_frame_ratio` (which multiplies `_n_text_tokens`,
`world_model.py:1894`) it carries to instruction-following voice targets unchanged.

## ⭐⭐⭐ THE RECURRENT EXIT CRITERION WAS BROKEN DURING GENERATION (2026-09-16, ESTABLISHED)

**Read this before citing ANY world-voice generation number measured on or before
2026-09-15.** All of them were produced under a criterion that freezes arbitrary positions
mid-computation.

`logit_kl` was wired only into `forward()`. Every `generate()` call site passed no readout,
so generation fell through to the legacy `kl_divergence` path, which compares POST-NORM
ACTIVATIONS with `F.kl_div`. That is a SIGNED quantity, so `value < threshold` passes
whenever it goes negative -- by sign accident, not convergence. The damage is not that it
runs shallow; it is that it halts computation on positions that have NOT converged while
letting others continue.

Fixed in 64861ba (`_trunk_readout_voice` + generate wiring). ck90000-ema unistream, greedy,
RAS w=10 tau=0.1, n=48 x 2 seeds (`eval_output/world_voice/reports/crit_*`):

| criterion | depth | LCS | WER | hyp/ref | caps | collapsed | seed spread |
|---|---|---|---|---|---|---|---|
| legacy kl_divergence | 14.92 | 0.7519 | 0.2621 | 0.953 | 5/96 | 7/96 | 0.0168 |
| logit_kl 5e-4 | 19.50 | 0.8849 | 0.1333 | 1.007 | 6/96 | **0/96** | 0.0007 |
| **none (full 32)** | 32.00 | **0.9103** | **0.0879** | 0.988 | **1/96** | **0/96** | 0.0012 |
| CEILING (GT units, n=12) | -- | 0.9146 | 0.0950 | -- | -- | -- | -- |

**Quality rises MONOTONICALLY with depth all the way to the trained budget.** The exit
criterion is therefore a pure compute knob, not a quality feature: `logit_kl` buys ~40% fewer
iterations for -0.025 LCS, and the legacy criterion was destroying the model. **At the depth
it was TRAINED at, the model sits within 0.004 LCS of the pipeline ceiling and scores BETTER
WER than ground-truth units through the same decoder.**

✅ **RESOLVED 2026-09-16 — the model BEATS the ceiling at matched n.** Same run, same 48
utterances (`reports/ceiling_n48`): model 0.9134 LCS / 0.0920 WER / 0.0424 CER vs ceiling
0.8601 / 0.1460 / 0.0649. **+0.053 LCS, -0.054 WER.** The n=12 ceiling (0.9146) was an easy
sample; the matched one is WORSE, so the lead is larger than first measured, not smaller.

Not an error: the ceiling is real human speech round-tripped through tokenizer + decoder, and
it carries disfluencies, accent and noise that Whisper mis-transcribes. The model emits
canonical, evenly-paced speech that ASR finds easier (cadence 0.95x -- slightly FASTER than
the reference). This is `project_dispersion_over_smoothing` seen from the other side: here
canonicalisation HELPS the metric.

⚠️⚠️ **THE ASR EVALUATION IS THEREFORE SATURATED.** LCS/WER against Whisper can no longer
measure world-voice progress -- the best config has passed the reference. Consequences:
- every ranking on 2026-09-15/16 was made on a metric the winner now exceeds; further sampler
  or checkpoint tuning measured this way will return noise
- the remaining gap lives where ASR is blind: prosody, speaker similarity, naturalness. That
  is exactly what the EAR kept reporting while the numbers could not see it
- speaker similarity needs prompt conditioning, which is viz-only today because the val cache
  stores no waveforms (see the campplus/prompt entry)
- next metrics should be speaker-similarity (campplus cosine vs the reference speaker),
  prosody (F0/duration correlation), and ear/MOS

~~The "at ceiling" claim is not yet matched-n.~~ The published ceiling (0.9146 / 0.0950) is
**n=12**; these arms are n=96. A matched n=48 ceiling is being measured
(`reports/ceiling_n48`). WER beating the ceiling is exactly the pattern a sample-set mismatch
produces, so treat "beats the teacher" as unverified until that lands -- the plausible
mechanism (the model emits smoother, more canonical speech than GT units, which carry
disfluencies the decoder renders awkwardly) is a story, not evidence.

**Practical:** use `none` for quality. Use `logit_kl` only when generation throughput matters,
and know it costs 0.025 LCS.

### `latent_diff` — Huginn's own readout-free criterion — is CATASTROPHIC here (2026-09-16, ESTABLISHED)

Same protocol, n=48 x 2 seeds:

| criterion | depth | LCS | WER | hyp/ref | collapsed |
|---|---|---|---|---|---|
| none (full 32) | 32.00 | 0.9103 | 0.0879 | 0.988 | 0/96 |
| logit_kl 5e-4 | 19.50 | 0.8849 | 0.1333 | 1.007 | 0/96 |
| legacy (broken) | 14.92 | 0.7519 | 0.2621 | 0.953 | 7/96 |
| **latent_diff 0.01** | 24.98 | **0.2007** | 0.7948 | 0.301 | **65/96** |
| **latent_diff 0.03** | 23.90 | **0.0964** | 0.9115 | 0.250 | **75/96** |

It does not merely underperform -- it INDUCES collapse, 65-75 of 96, against ZERO for
logit_kl, and is far worse than the numerically broken legacy criterion despite running ~10
iterations DEEPER.

**This settles that depth is not the variable.** Ordered by depth: legacy 14.9 < logit_kl 19.5
< latent_diff 23.9-25.0 < none 32. Ordered by quality: none > logit_kl > legacy >> latent_diff.
What matters is WHICH positions freeze, not how many iterations run.

**Mechanism (inferred, not measured):** relative latent movement is a poor proxy for the
output distribution having settled. A voice frame's trunk state can stabilise while its unit
distribution is still moving; freezing there locks in an unconverged distribution, and
empirically that favours EOV. Note also that the two thresholds give nearly identical depth
(24.98 vs 23.90) despite being 3x apart, which suggests it fires on a few positions almost
regardless of threshold rather than tracking convergence smoothly. The implementation is a
faithful port (per-position rather than Huginn's per-batch mean, identical at seq_len 1) --
this is not a porting bug.

⚠️ **Do not recommend `latent_diff` to other directions as "the readout-free option" without
measuring it.** world-image cannot use `logit_kl` (its DiT coda has no categorical head), and
the obvious substitution is exactly this criterion. On voice it is the worst of four. Start
those directions on `none`, which needs no readout at all.

**LCS +0.133, WER -0.129 (49% relative). The gap to the pipeline ceiling falls from 0.163 to
0.030 -- ~82% of it closed by a comparison operator.**

Depth rises only 14.92 -> 19.50, so this is NOT "more compute". It is computing the RIGHT
positions. ~4.5 extra iterations cannot explain a 0.133 LCS move; freezing the wrong tokens
can.

### Consequence 1: "seed variance dominates" was measuring THIS BUG

Seed spread falls **0.0168 -> 0.0007, 24x tighter**. The 2026-09-15 finding that single-seed
LCS comparisons are worthless and that generation carries ~0.07 of seed noise was TRUE FOR
THAT REGIME and is an artifact of the broken criterion: freezing arbitrary positions is the
noise source. Under a correct criterion the model is near-deterministic across seeds.
⚠️ Do NOT relax the 2-seed protocol on the strength of one checkpoint -- but expect single-seed
comparisons to become viable again, which changes the cost of every future sweep.

### Consequence 2: premature collapse was a SYMPTOM of it

Collapse goes 7/96 -> **0/96**. The 2026-09-15 diagnosis (collapse = flat distribution,
entropy 5.87 vs 2.59 nats at the EOV step) was ACCURATE but identified a symptom: a trunk
state frozen mid-computation is what produces those flat distributions. The EOV confidence
guard (584c569) therefore targets a failure that does not occur under a correct criterion.
Its continued value is being measured (`guard_on_logitkl_*`); expect a no-op at best.

### ⚠️ EVERYTHING from 2026-09-15 needs re-measurement

Under re-measurement now (`scripts_local/world_voice_recheck_campaign.sh`, results under
`eval_output/world_voice/{reports,audio,logs}/`): sampling config (greedy vs T=0.7 vs
T=0.6+rt1.0), checkpoint ranking (uni90k vs uni68k vs bi68k), the EOV guard, and
`latent_diff` as a readout-free alternative.

Specifically suspect, because they rest on failure modes this bug may have CAUSED:
- best sampling config (greedy won partly on collapse frequency)
- ck90000 vs ck68000 (+0.046, and the eval-loss minima that nominated both were themselves
  computed under the broken criterion in forward())
- bistream vs unistream (its surviving claim was "sampler insensitivity")
- the 0.7519 headline, already superseded

Probably survives, being a code-path fact rather than an empirical one: the T=0 discontinuity
in the RAS resample. Also expected to survive: tau=0 causing total degenerate collapse.

⚠️ **`bf4769e` (2026-09-15 21:53) changed `forward()` to restrict convergence to TEXT
positions**, so voice now runs full budget in eval. Eval losses from before that commit and
after it are NOT comparable, and every checkpoint-selection decision to date predates it.

## RE-MEASUREMENT of the 2026-09-15 findings under a working criterion (2026-09-16)

Full campaign, ck90000-ema unless stated, n=48 x 2 seeds, `logit_kl` 5e-4 unless stated.
Runner: `scripts_local/world_voice_recheck_campaign.sh`. Results:
`eval_output/world_voice/{reports,audio,logs}/`.

### Transfer scorecard

| 2026-09-15 conclusion | verdict under a working criterion |
|---|---|
| sampling ranking greedy > T=0.6+rt1.0 > T=0.7 | **SURVIVES** — identical order, similar gaps |
| bistream vs unistream on CONTENT = null | **SURVIVES** — +0.011 vs ~0.011 spread |
| bistream buys TERMINATION robustness | **STRENGTHENED** — 0/96 caps vs uni 2/96 |
| ck90000 > ck68000 | **WEAKENED** — +0.046 -> +0.016, ~1.4x spread |
| seed variance dominates; >=2 seeds required | **DEAD** — spread was 24x inflated by the bug |
| premature collapse needs an EOV guard | **DEAD** — 0/96 collapse without it; guard neutral |
| best config LCS 0.7519 | **SUPERSEDED** — 0.9103 at full depth |

### Sampling (2 seeds, n=96)

| sampler | LCS | WER | caps | spread | LCS under the broken criterion |
|---|---|---|---|---|---|
| **greedy** | **0.8849** | **0.1333** | 6/96 | 0.0007 | 0.7519 |
| T=0.6+rt1.0 | 0.8619 | 0.1626 | 2/96 | 0.0110 | 0.7103 |
| T=0.7 | 0.8384 | 0.1868 | 4/96 | 0.0031 | 0.6933 |

Same ordering as under the broken criterion, so the criterion degraded all three roughly
uniformly rather than interacting with temperature. Greedy's margins are now 20-30x the seed
spread instead of 1.4x. The "greedy fails harder but occasionally nails it" tradeoff the ear
reported is GONE: all three collapse 0/96, so greedy simply wins.

### Checkpoints (2 seeds, n=96, greedy)

| checkpoint | LCS | WER | caps | spread |
|---|---|---|---|---|
| uni ck90000 | 0.8849 | 0.1333 | 6/96 | 0.0007 |
| uni ck68000 | 0.8690 | 0.1511 | 2/96 | 0.0113 |
| bi ck68000 | 0.8581 | 0.1598 | **0/96** | 0.0107 |

⚠️ **ck90000 over ck68000 is now MARGINAL** (+0.016 against 0.011). On 2026-09-15 it was
+0.046 at ~2x spread and was called real; most of that gap was the broken criterion hurting
ck68000 more. 22k extra steps bought little, and ck68000 has fewer budget-caps.

**Bistream vs unistream at matched step 68000 is STILL A NULL** (+0.011, spreads ~0.011),
replicating the 2026-09-15 paired result in a completely different regime. Bistream's
termination edge shows again: 0/96 caps, the best of any arm.

### EOV confidence guard: NEUTRAL, not needed

| arm | LCS | WER | caps | spread |
|---|---|---|---|---|
| guard OFF | 0.8849 | 0.1333 | 6/96 | 0.0007 |
| guard ON (0.10/3.0) | 0.8946 | 0.1241 | 3/96 | 0.0109 |

+0.0097 against a 0.0109 spread — inside noise. Seed 1 alone showed +0.0155 and was briefly
read as a real benefit; seed 2 gave ~+0.004. **Keep the flag, do not enable it by default.**
Full depth with NO guard beats both (0.9103 / 0.0879 / 1 cap per 96).

### RECOMMENDED CONFIG (2026-09-16)

```
--exit_criteria none          # full 32 iterations; the depth the model was trained at
--voice_temperature 0.0       # greedy
--ras_win 10 --ras_tau 0.1    # RAS is load-bearing: tau=0 is total collapse
checkpoint: ck90000-ema       # ck68000 within ~1.4x noise; fewer caps
# no EOV guard
```

## logit_kl threshold curve: a KNEE at ~24 iterations, and a U-shaped seed spread (2026-09-16, ESTABLISHED)

ck90000-ema, greedy, RAS w=10 tau=0.1, n=48 x 2 seeds
(`scripts_local/world_voice_logitkl_threshold_sweep.sh`, `reports/thr_logitkl_*`).

| threshold | depth | trunk compute | LCS | WER | seed spread | marginal LCS/iter |
|---|---|---|---|---|---|---|
| 1e-3 | 17.55 | 54.8% | 0.8711 | 0.1592 | 0.0016 | -- |
| 5e-4 | 19.50 | 60.9% | 0.8849 | 0.1333 | 0.0007 | 0.00707 |
| **1e-4** | 23.79 | **74.4%** | 0.9051 | 0.0989 | 0.0086 | 0.00472 |
| 5e-5 | 25.53 | 79.8% | 0.9063 | 0.0989 | 0.0094 | 0.00067 |
| 1e-5 | 28.30 | 88.4% | 0.8986 | 0.1011 | 0.0106 | -0.00278 |
| **none** | 32.00 | 100% | **0.9103** | **0.0879** | 0.0012 | 0.00317 |

**The curve rises steeply to ~24 iterations then plateaus.** Marginal return runs 0.005-0.007
per iteration up to 1e-4, then collapses: 1e-4 / 5e-5 / 1e-5 span 0.8986-0.9063 with no trend,
all inside their own ~0.01 spreads, and WER is pinned at 0.0989 across three of them. Anything
at or tighter than 1e-4 is ONE FLAT REGION -- the threshold choice there is compute, not
quality.

### ⭐ Seed spread is U-SHAPED in depth

**0.0016, 0.0007** (shallow) -> **0.0086, 0.0094, 0.0106** (plateau) -> **0.0012** (full depth).
Six to eight times wider through the plateau, monotonic in, monotonic out.

Reading: on the plateau the EXIT DECISION ITSELF is marginal -- positions sit near the
threshold, so small trajectory differences flip whether a position stops at 23 or 28
iterations, and realised depth varies run to run. Far from the boundary the decision is
decisive; at `none` there is no decision at all. This is a property of adaptive exit, not of
the model.

**Consequence: `none` is not just nominally best, it is 8x more REPRODUCIBLE.** Any future
comparison run under an adaptive criterion inherits ~0.01 of extra noise for free.

### Practical

- **`--exit_criteria none`** — best mean (0.9103), best WER (0.0879), 8x tighter spread. Use
  for quality, and for any measurement that will be compared against another.
- **`--exit_criteria logit_kl --exit_criteria_threshold 1e-4`** — 74% of trunk compute, ~0.005
  LCS behind, 7x noisier. Use for throughput.
- ⚠️ The trunk saving is NOT necessarily a wall-clock saving: `logit_kl` invokes the voice coda
  (20.7M params) every recurrent iteration as its readout, against a 127.5M trunk. Plausibly
  still a win, but UNPROFILED.

### Reference: parameter counts (2026-09-16)

| component | params |
|---|---|
| CosyVoice 2 decoder (frozen; the ASR ceiling's own renderer) | **133.4M** (flow 112.5M + HiFT 20.8M) |
| world model checkpoint, total | 362.1M |
| — text_feature_extractor (frozen SmolLM2) | 165.6M |
| — recurrent trunk | 127.5M |
| — text_generator | 30.4M |
| — voice_generator (coda) | 20.7M |
| — voice_feature_extractor | 14.6M |

CosyVoice 2's own `llm.pt` (2.0 GB) is NOT loaded — the world model replaces it. So a 127.5M
trunk substitutes for a ~0.5B speech LM and drives a 133M decoder. Relevant to
[[project_leanness_and_open_data]]: unlike the 6.2B Z-Image departure on the image side, the
voice stack is already lean, and a future own-decoder distillation target is 133M.

## OPEN

### ~~Can it memorize 32 utterances?~~ ANSWERED 2026-08-24: yes, in ~1400 steps (see above)
The partition below is kept because it is what the run was launched to decide, and the answer
picked the first branch: it memorizes, so the 63k failure is about GENERALIZATION. The third
branch (shuffled text) was never runnable as stated — see the confound entry above.
`--use_memorization_dataset --max_samples 32`, LR 1e-4, constant. Judge on train loss and
whether it reproduces those 32 utterances; there is no meaningful held-out set at n=32.

- **Memorizes** -> the mapping is learnable and the architecture is capable; the 63k failure
  is about GENERALIZATION, so data scale and alignment (bistream) become the live questions.
- **Cannot memorize 32** -> the wall is upstream of data: capacity, the frozen text
  representation, or the framing. LibriHeavy would not help and the data-scaling slope would
  waste a week.
- **Memorizes equally well with SHUFFLED text** -> it is memorizing unit sequences
  positionally and learning no text->unit map at all. Run this control; it is the outcome that
  would explain the whole month.

⚠️ The memorization dataset was three-ways stale until 2026-08-23 (no discrete-unit support,
no text pairing, no `_direction`); two of the three failed SILENTLY, one by training on
nothing. Sanity-check `train/voice_unit_accuracy` climbs fast in the first few hundred steps —
on 32 samples it should — before trusting a negative result from it.

### ~~Does training at mask ratio 1.0 build text conditioning?~~ ANSWERED: no (see above)
Run `cosyvoice2_smollm2_nar_r1_flat_lr-flat_0`, started 2026-08-21, killed at 63k:
`--voice_nar_mask_schedule high --voice_nar_mask_ratio_min 1.0`, constant LR after warmup so
plateau is attributable to the model rather than LR decay, `--voice_cfg_text_dropout_prob 0.1`
so CFG is testable later.

Rationale: inference starts fully masked, so r~1 fixes the first commitments and anchors every
later round — and cosine trains it least (a third of its mass lands below r=0.5, where
inpainting suffices). At r=1.0 the model's text-attributed fraction is highest (0.220), i.e.
it reaches for text when nothing else is available.

Reference points at r=1.0: entropy 11.29 bits at step 1000 (start), 10.96 for the
cosine-trained model at 23k, 6.47 for AR with full context. Read `text_delta` and entropy at
r=1.0, NOT `acc_real` (lower by construction).

Decide at ~23k against the cosine model's r=1.0 numbers. **Kill only on a DECLINING trend
across three checkpoints spanning 20k+** — that is the standard the gen-query verdict met; a
5k snapshot proves nothing, and voice conditioning in this project develops over 20k-45k steps
(AR text-attributed went 0.305 @23k -> 0.397 @44k).

### Follow-up already built: anneal, and trunk-text-only
`--voice_nar_mask_anneal_steps` / `--voice_nar_mask_ratio_floor` ramp the mask floor from 1.0
down, to form the text pathway while no shortcut exists and then admit context.
`--voice_nar_trunk_text_only` routes revealed units DIRECTLY to the coda (zero-init
projection) so the trunk never sees them and any inpainting shortcut is confined to the head.

The two are **identical at mask ratio 1.0** (nothing to route), so the r=1.0 run does not need
restarting to adopt the flag — add it when the anneal begins, where the projection arrives
zero-init and has to earn its contribution. Honest caveat: gradient descent has no loyalty to
features it built earlier, so if context is still the easier predictor once admitted, the
crutch can reassert. Readout is `text_delta` at r=1.0 across the anneal.

### Bistream + inner monologue — THE CURRENT PLAN (scoped 2026-08-24)

**The layout.** CosyVoice 2 trains 50/50 on two layouts. Unistream is
`[sos][all text][task_id][all speech]`, what we do today. Bistream is
`[sos][5 text][15 speech][5 text][15 speech]...[remaining text][task_id][remaining speech]`,
where the 15th speech position of each chunk targets `fill_token` — the model's learned way of
saying "this text chunk is consumed, send the next."

**Why it should help.** In unistream the text a given frame realizes can be hundreds of
positions back, so alignment is long-range, non-local, and example-dependent. Bistream makes
the relevant text ADJACENT and the alignment roughly monotonic — the inductive bias a duration
model gives classical TTS, baked into the sequence layout rather than the architecture. It is
a DATA-LAYOUT change, not an architecture change.

⚠️ **Recalibrate the ratio.** Theirs is 3 frames per text token (15/5), gated to samples with
`speech_len/text_len > 3`. Our corpus is **5.9 frames/token** (n=6102: mean 6.015, median
5.906, std 1.239, p5 4.19, p95 8.22). At 5:15 unchanged, text would be exhausted after ~3N
frames of a 6N-frame utterance, so only the first HALF of each utterance would get local
alignment. Roughly **5:30** matches our rate.

**What it does and does not fix.** The fixed cadence still assumes a constant rate WITHIN an
utterance, the same assumption a global or per-utterance M-RoPE rate makes. What differs is
that it RE-ANCHORS every chunk, so drift is bounded to one chunk instead of accumulating.
Bounded error, not eliminated error.

### Inner monologue: how bistream extends to the full multimodal model
**The problem** (raised by the owner, 2026-08-24): bistream gets you TTS, where a transcript
exists to chunk. It does not obviously get you an instruct model that emits voice AT WILL —
"say your previous message aloud" has no transcript to interleave, and the text that
corresponds to the voice is not in the input.

**The resolution: the model generates its own transcript, interleaved, and it is filtered from
the output.** Precedent is Moshi's "Inner Monologue" — a text stream generated alongside the
audio, with text LEADING the audio it describes; they report it substantially improves the
linguistic quality of generated speech.

⭐ **The key consequence: bistream TRAINING already trains this.** Training teacher-forces the
text chunks either way. Only inference differs:
- **TTS mode** — feed the real transcript's chunks.
- **Speak-at-will mode** — let the model GENERATE each text chunk, then continue.
Same weights, same layout, same run. The multimodal capability is a decoding mode, not a
separate finetune.

⚠️ **One change this REQUIRES, and it fails silently otherwise:** the text loss must NOT be
masked on the interleaved text chunks. `--mask_text_loss_in_synthesis` zeroes text loss on
synthesis examples, which is right when the transcript is pure conditioning but leaves the
model unable to GENERATE it. Same shape as the duration-token exemption already in the
codebase, and it needs the same explicit carve-out.

**Why this is the right decomposition, beyond solving the no-transcript problem.** "Say my
previous message aloud" needs long-range reasoning over the conversation, through an
intervening user turn. That is a TEXT-space problem where a pretrained LM is strong. Rendering
a decided chunk of text as speech is a LOCAL problem bounded to a few frames. The current
architecture asks the trunk to do both at once across hundreds of positions, and the
measurements say it does neither (text-attributed 0.029; cannot uniformly fit even 32
memorized utterances). Inner monologue splits them: reason in text, render locally.

⚠️ **Design detail worth copying rather than rediscovering:** text should LEAD the audio it
describes, not sit adjacent to it. If text and audio are emitted at the same position, the
model must commit to semantics and acoustics simultaneously, which loses most of the benefit.

### This returns the voice path to AR, and the evidence supports that
Bistream is autoregressive at the chunk level (chunk k+1 conditions on chunk k), and inner
monologue requires AR text generation. That is a return to AR after the NAR detour, and it is
what the measurements point to: at matched step 23000 AR beat NAR 2.2x on intelligibility
(truncated LCS 0.1620 vs 0.0742) and ~10x on text attribution (0.305 vs 0.029), and the NAR
premise — that removing the AR crutch would force text conditioning — was falsified twice
(gen-query 2026-08-13, the mask-ratio curve 2026-08-21).

Kept from the NAR work: the duration token (architecture-independent, one flag), the
mask-ratio probe, the seed-floor discipline, and a clean negative result. Note the duration
token's role SHRINKS under bistream — length emerges from the chunk cadence plus EOV — so it
becomes optional rather than load-bearing.

### Bistream implementation scope (CORRECTED 2026-08-24)

⚠️ An earlier version of this section said the scope was "smaller than feared" because the
interleaver already supports multiple voice placeholders per example. **That was wrong**, and
the error is worth recording because it is the kind that survives a code read.

`voice_positions[batch_idx]` is indeed a list, and the `n` dimension of `voice_inputs`
(B, n, C, T) is indeed a segment count. But **`n` means DISJOINT UTTERANCES within one
example** -- the `<text><voice_0><text><voice_1>` case the M-RoPE global axis exists to keep
distinguishable. Chunks of a SINGLE utterance are a different axis, and overloading `n` with
them collides with it.

The concrete breakage, verified in world_model.py: `completed_voice[b].append(feat)` fires per
SEGMENT (line ~1891 and the end-of-generation flush ~2095). Treat chunks as segments and one
utterance is emitted as N separate audio clips instead of one -- structurally the same failure
as the multi-segment trace concatenation found on 2026-08-21, except by design. Speaker
conditioning and EOV are also per-utterance concepts that would be shredded.

**The fix: make the placeholder -> media mapping EXPLICIT rather than positional.**

| | today | needed |
|---|---|---|
| mapping | placeholder i -> `batch_voice[i]`, full length | placeholder i -> `(utt_idx, start, length)` |
| default | — | `(i, 0, voice_lens[i])`, byte-identical to today |
| bistream | — | several placeholders sharing one `utt_idx`, consecutive slices |
| composed | impossible | two utterances, each chunked |

Work required:
1. **Interleaver contract** — per-placeholder `(utt_idx, start, length)` instead of positional
   full-segment lookup. Backward-compatible default keeps existing runs identical.
2. **Collator** — emit `[BOV][k text][VOICE_PH][k text][VOICE_PH]...[EOV]` and the chunk map.
3. **Uninterleaver** — must gather voice positions back GROUPED BY UTTERANCE, not by segment.
4. **`_finalize_voice` and the generation flush** — accumulate across chunks of the same
   utterance; emit one clip per `utt_idx`.
5. **Two terminators** — EOV ends an UTTERANCE, `fill_token` ends a CHUNK. Only the former
   exists today. (v1 can skip fill_token with fixed chunk sizes and a counter.)
6. **Loss assembly** — voice targets are (B, T_total) against a coda output of
   (B*n_chunks, chunk_T, V); the mapping back must respect utterance grouping.
7. **Text-loss carve-out** for the interleaved text chunks (see inner monologue above).
8. **`generate()`** — alternate k text tokens and one voice chunk, tracking which utterance and
   how far into it; KV-cache layout must mirror training exactly.
9. **M-RoPE check** — the local axis resets per contiguous same-modality run, so it re-anchors
   per CHUNK (desirable). But that leaves the GLOBAL axis as the only thing separating
   utterance 0 chunk 3 from utterance 1 chunk 3 -- exactly what it was introduced for, so it
   should hold, but verify rather than assume.

Simplification for v1: fixed chunk sizes remove the need for `fill_token`. Copy CosyVoice 2's
50/50 unistream/bistream mix rather than going pure bistream -- one model does both, and the
unistream half stays directly comparable to what is trained today.

### Is the crutch fixable, or is it a trunk property?
If the r=1.0 run also flatlines, that is the same failure twice on the same trunk with
different feature spaces (Mimi/ContentVec in August, CosyVoice 2 now), which points at the
trunk rather than the token space. The cheapest disambiguation is a plain-stack control at
matched parameters — the trunk is the one component with no external validation.

### Does refinement help once text conditioning exists?
Currently no measurable refinement gain, but every rounds/reveal-order comparison sits inside
the seed floor. Prediction worth testing: in a model that reads text, committed tokens carry
real constraints and refinement should pay. If it still does not, the rounds duplicate the
recurrent loop and one of them should go.

### Data scaling slope
`--data_fraction` trains on a seeded RANDOM subset (distinct from `--max_samples`, a prefix cap
that would confound less data with a biased speaker subset). Train 0.25 / 0.5 / 1.0 at matched
steps and read val `acc_real` and `text_delta`. Steep and un-saturating means data is the
answer and the 51x English gap is worth closing; already flattening at 585 h means the
bottleneck is the frozen encoder or the trunk. Note distillation may flatten the slope, since
it exists partly to substitute for data — a `--voice_distill_weight 0` arm disambiguates.

### Unfreeze the text encoder
`--text_encoder_unfreeze` exists. The biggest architectural delta from the system known to
solve this task: CosyVoice 2 fully fine-tuned its backbone. A frozen semantic LM may not make
graphemic/phonetic detail linearly accessible — and the transcripts show phonetic near-misses
("a sorely breathing that no" for "a softly-breathing air, that no"), which is what a lossy
text representation produces. 1e-5 is already a sane fine-tuning LR for a 135M model.

### On-policy distillation
`--voice_onpolicy_distill` scores the teacher on the model's own scheduled-sampling history
instead of ground truth. Off-policy KD only supervises states reachable from a perfect prefix —
where this model is already near the teacher. Its failure is off that manifold. **Does not
apply to NAR** (no AR history). Requires `--voice_scheduled_sampling_prob > 0` and
`--voice_distill_weight > 0`; raises rather than silently no-opping.

---

## RETRACTED

### "Text horizon ~3-5 words and not widening" (retracted 2026-08-21)
Measured with M-RoPE silently disabled. Re-measured with the geometry engaged, `text_delta`
runs +0.0503/.0430/.0429/.0396/.0369/.0306 across the whole utterance and never drops below
threshold. A sustained horizon is what a correctly-evaluated model of this family looks like —
it is not an M-RoPE property. Also retracted: `eov_position_acc` "degrading" to 0.218; it is
0.818.

### "No head is a sharp aligner under M-RoPE" (retracted 2026-08-21)
Same cause. Wrong positions would scramble exactly that statistic. Untested either way now.

### "The text-scaled arm does not need RAS" (retracted 2026-08-21)
Rested on adj_repeat 0.0257, measured with M-RoPE off at T=1.0. At the operating point the
arm loops (0.453).

### "The distribution is peaked, so sampling is argmax" (retracted 2026-08-21)
Asserted twice from indirect evidence and contradicted by direct measurement: `frac top1 > 0.9`
is 0.000 everywhere. Both premises were bugs — a NaN Gumbel expression and a seeding-order
error.

### "Confidence-ordered reveal is catastrophic (1.7x)" and the mode-seeking story on it
The comparison was 0.0320 (the unlucky seed) against a single sequential run — a gap of 0.015
against a 0.018 noise floor. Likewise "sequential beats random" (0.006) and "refinement loses
to single-shot" (0.003). None established.

### "Generation is deterministic" (retracted 2026-08-21)
Claimed after three seeds produced identical output. Seeding ran BEFORE the CosyVoice 2
decoder load, which fixes the RNG, so `--nar_seed` never took effect. With seeding moved
after, seeds diverge (0.0320 / 0.0498 / 0.0396).

### "AR@23k is the best AR checkpoint" (retracted 2026-08-21)
It is the checkpoint that talks the most. See "LCS recall is partly a length metric."

### "AR + duration token combines both wins" (retracted 2026-08-21)
The duration token buys evaluation hygiene and usability, not capability. Training is
teacher-forced on GT-length sequences, so runaway generation never degraded training or any
teacher-forced metric — it only contaminated free-running WER/LCS. Poor length control was a
SYMPTOM of weak progress-tracking; an explicit duration token fixes the readout, not the cause.

---

## Tooling and operational traps

`scripts_local/`: `world_voice_ar_diagnostics.py` (TF + text ablation + bootstrap CIs +
position buckets + free-running; **`--nar_mask_ratio` REQUIRED for NAR checkpoints** — at ratio
0 a masked model can satisfy the task by copying its input, giving ~1.0 accuracy and ~0
text_delta, which reads as collapsed conditioning), `cosyvoice_wer_eval.py` (WER/CER/LCS +
ceiling; `--nar_seed`, `--nar_reveal`, `--nar_rounds`, `--duration_shuffle`),
`unit_confidence_probe.py`, `wer_arm_compare.py` (paired bootstrap over arms),
`render_distill_audio.py`, `run_nar_eval.sh`, `voice_text_horizon_sweep.py`,
`teacher_text_ablation.py`, `teacher_freerun_compare.py`, `ngram_unit_baseline.py`.

**Traps, each of which has cost time:**
- **`--mrope_scale_side` is required** for an M-RoPE checkpoint and raises rather than guessing.
  It carries no weights, so it cannot be detected.
- **transformers 5.x breaks CosyVoice's incremental decode.** `inference_wrapper` recomputes
  masks from `lm_input.shape[1]`, which is 1 after step 0, so the attention mask is (1,1)
  against a full KV cache. Under 4.44 the teacher free-runs sanely; under 5.13 the same
  weights degenerate. **Teacher-forced single-forward is unaffected, so distillation is fine.**
  Use `~/dev/projects/cosyvoice-runtime/venv` for anything incremental.
- **Judge decode comparisons across >=3 seeds** or n well above 64.
- **Read LCS recall beside hyp/ref**, always.
- **`--compile_model` and `--use_gradient_checkpointing` both break the recurrent world model.**
  `--compile_recurrent_block` is fine.
- **With RAS on, unit entropy stops being a valid conditioning diagnostic** (the sampler forces
  diversity).
- `eval_output/` is gitignored scratch at REPO ROOT, one parent dir per task.

---

## Run inventory (as of 2026-08-21)

| run | latest | status |
|---|---|---|
| `cosyvoice2_smollm2_nar_r1_flat_lr-flat_0` | training | **LIVE** — the r=1.0 probe |
| `cosyvoice2_smollm2_nar_0` | 23000 | stopped; the cosine-masked NAR baseline |
| `cosyvoice2_smollm2_mrope_scale_text_0` | 44000+ | stopped; best AR arm, all checkpoints kept |
| `cosyvoice2_smollm2_distill_0` | 55000 | stopped; the distillation result above |
| `cosyvoice2_smollm2_mrope_0` | 28000 | killed (voice-scaled M-RoPE) |
| `cosyvoice2_smollm2_0` / `_nocurric_0` | 27000 / 23000 | stopped; pre-distill baselines |
| `..._0__baseline_ngram_*` | — | synthetic text-free baselines, not models |
