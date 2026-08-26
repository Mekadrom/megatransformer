# Plan: bistream chunk-interleaving + inner monologue (world-voice)

Status: NOT STARTED. Written 2026-08-24 for a fresh session to execute.
Read `docs/findings/world-voice.md` first — especially CONTAMINATED and RETRACTED — then this.

## Why (one paragraph)

Unistream puts all text before all speech, so the text a given frame realizes can be hundreds
of positions back: alignment is long-range, non-local and example-dependent. Measured
consequence: text-attributed fraction 0.305 (AR) / 0.029 (NAR) against a teacher's 0.463, and
a run that reached 34 epochs without fitting. Bistream makes the relevant text ADJACENT and
the alignment roughly monotonic. Inner monologue (the model generating its own transcript,
interleaved, filtered from output) extends that from TTS to the full multimodal model, and
costs nothing extra at training time. Precedent: CosyVoice 2 (bistream), Moshi (inner
monologue).

## PREREQUISITE — CLEARED 2026-08-24, do not re-run it

This plan originally gated itself on a shuffled-transcript memorization arm. **That control was
confounded and must not be run.** Over the 32 memorization samples `(voice_frame_len, text_len)`
is a unique index (30/32 distinct exact frame lengths; the one 173-frame collision broken by
text length; all 32 the same speaker), so a shuffled arm memorizes by keying on the layout and
reports "text is decorative" whether or not that is true.

The question was answered instead on the finished `memorize32_0/checkpoint-20000`, by
ablating the transcript at the sample level with the layout key held fixed
(`scripts_local/memorization_text_dependence.py`):

- real 1.0000 -> **matched (content changed, `(frame_len, text_len)` preserved) 0.0689.**
  The transcript is NOT decorative; the text pathway is live and high-bandwidth.
- Under `roll`, predictions match the SOURCE utterance's units at **0.8250** — the text is a
  RETRIEVAL KEY selecting which memorized utterance to replay, overriding both length signals.

**What that means for this plan:** it does not rest on nothing, and the 63k failure is a
generalization failure rather than dead wiring — which is exactly the failure bistream targets.
It does *not* establish that the model reads text phonetically; n=32 cannot separate "reads
phonemes" from "hashes the sentence". Proceed, but do not cite memorization as evidence of
compositional reading.

Still outstanding, cheap, and worth having (NOT blocking): a per-utterance probe over all 32 to
explain the generation-side discrepancy — train accuracy is 1.0 on every sample, yet TB
free-running renders were still nonsense for some (idx 3 at 17.5k). Teacher-forced 1.0 with a
broken render means the gap is in the GENERATION path, not in what was learned, and that would
invalidate NAR generation numbers rather than NAR training.

## The layout

Per example, synthesis direction:

    [text_0..k-1] [VOICE_PH] [text_k..2k-1] [VOICE_PH] ... [remaining text] [VOICE_PH] [EOV]

Each `VOICE_PH` expands to ONE CHUNK of the utterance's frames (not the whole utterance).

- **k = 5 text tokens per chunk, 30 frames per chunk.** NOT CosyVoice 2's 5:15. Theirs assumes
  3 frames/token; our corpus is **5.9** (n=6102: mean 6.015, median 5.906, std 1.239, p5 4.19,
  p95 8.22). At 5:15 the text would be exhausted after ~3N frames of a 6N-frame utterance,
  leaving the back half unaligned.
- Copy CosyVoice 2's **50/50 unistream/bistream mix**, chosen per sample. One model does both;
  the unistream half stays directly comparable to what is trained today.
- Gate bistream to samples with enough frames per token (they gate on
  `speech_len/text_len > mix_ratio[1]/mix_ratio[0]`); with 5:30 that gate is rate > 6.

## Two terminators — separate from the start (owner's call)

| token | ends | emitted at |
|---|---|---|
| `EOV` (unit id 6561) | the UTTERANCE | after the last chunk |
| `fill_token` (NEW, unit id 6562) | a CHUNK | last frame of each chunk |

**Unit head grows from 6562-way to 6563-way.** Today `unit_vocab_size = K + 1` (6561 content +
EOV) at `training.py:2426` and `visualize.py:245`. It becomes `K + 2`. Both sites must change,
and eval-side detection must read the width from the checkpoint the way M-RoPE/NAR detection
does, or a bistream checkpoint silently loads into a 6562-wide head.

`fill_token` is a TARGET, not an input marker: the model learns to emit it, and at inference
you generate until it appears and then feed/generate the next text chunk. That is what makes
chunk boundaries adaptive later; v1 may also just count frames.

## The axis mistake — read this before touching the interleaver

`voice_inputs` is (B, n, C, T) and **`n` means DISJOINT UTTERANCES within one example**
(`<text><voice_0><text><voice_1>`), which is what the M-RoPE global axis exists to keep
distinguishable. **Chunks are a different axis.** Overloading `n` with chunks breaks:

- `world_model.py` ~1891 and ~2095: `completed_voice[b].append(feat)` fires per SEGMENT, so
  one utterance would be emitted as N separate audio clips.
- speaker conditioning and EOV, both per-utterance concepts.

**Fix: make the placeholder -> media mapping EXPLICIT rather than positional.**

| | today | needed |
|---|---|---|
| mapping | placeholder i -> `batch_voice[i]`, full length | placeholder i -> `(utt_idx, start, length)` |
| default | — | `(i, 0, voice_lens[i])` — byte-identical to today |
| bistream | — | several placeholders sharing one `utt_idx`, consecutive slices |

## Work items — ALL LANDED 2026-08-24 (685364e, fc0697e, ec45e6a, 90d5461, f44c3c1)

Kept for the record of what was touched; every item below is done and tested.

1. **`TokenInterleaver.forward`** — optional `voice_chunk_map` (B, n_placeholders, 3) =
   (utt_idx, start, length). None reproduces the historical mapping; a test asserts the
   explicit equivalent map gives bit-identical output. ✅
2. **`TokenUninterleaver.forward`** — **no change needed.** Its masked left-pack already
   gathers voice positions in ascending order, which for one utterance's chunks laid out
   consecutively IS the utterance grouping. Verified by round-trip, not assumed. ✅
3. **`_build_token_sequence`** — per chunk emits `[text_j][BOV][PH][EOV]`, so a 1-chunk plan
   is byte-identical to unistream: unistream is literally the m=1 case. ✅
4. **`_collate_audio_like`** — the discrete path now builds everything from one description,
   a list of `(content_start, content_len, terminal_or_None)` segments laid end to end with
   terminals INLINE. That is what keeps the unit target at (B, T_expanded) per utterance, so
   the coda, the CE, EOV and speaker conditioning stay per-utterance and chunks are a
   slicing of the stream rather than a new batch axis. ✅
5. **`world_model.forward`** — chunk map threaded through, and dropped by the
   mixed-modality null-out along with the voice it indexes. ✅
6. **Chunk accumulation in `generate`** — fill tears down the voice block but keeps frames,
   F0 and the prelude/coda KV caches; ONE clip per utterance. ✅
7. **Voice loss** — **no change needed**, because of the design in (4): fill_token is just
   another class in the existing (B, T) target. Only the head width changed. ✅
8. **Text-loss carve-out** — `--bistream_text_loss`, a per-ROW re-target sharing the
   mechanism with the duration-token exemption. ✅ **It failed silently on the first try
   exactly as this plan predicted** — see the traps section. ✅
9. **`generate()`** — chunk continuation, forced-token FIFO for the next transcript chunk,
   and the zero-FEATURE parity detail below. Viz wired so a bistream checkpoint decodes as
   one was trained. ✅
10. **M-RoPE check** — done as a test: local re-anchors per chunk (desirable — a frame's
    coordinate is relative to the text chunk it was just given), so global is the only thing
    separating utterance 0 chunk j from utterance 1 chunk j, and it stays strictly
    increasing. ✅

### Two parity details worth re-reading before changing any of this

- **The zero FEATURE at a chunk boundary.** The next chunk's first trunk input is
  `prelude(feature at the fill slot)`, and that feature is ZERO. So generation sets
  `last_voice_pred` to a zero FEATURE, not `None` — `None` takes the position-0 branch,
  which is a zero in d_model space WITHOUT running the prelude, and only an utterance's very
  first position looks like that in training.
- **Caches do NOT reset at a chunk boundary.** In training the coda is causal over the whole
  expanded stream with no gap there; resetting would give inference a shorter acoustic
  history than training ever had.

### Feasibility gate, corrected

CosyVoice 2 gates on `speech_len/text_len > s/k`. With s/k = 6 and our corpus at 5.9
frames/token that rejects about half the data for no reason. The exact condition is only
that the chunks before the last one fit: `n_frames > (n_text_chunks - 1) * s`. Measured on
64 LibriTTS-R samples at k=5/s=30: ~75% eligible, interior chunks all exactly 30 frames,
last chunk 10–60 (median 27) — no pathological tails.

## Traps that have already cost time on this codebase

- **`forced_next_token` is re-created every token-loop iteration, and that is correct for it**
  — it is set and consumed within one iteration. Anything declared beside it inherits that
  lifetime. The bistream `forced_token_queue` must OUTLIVE the iteration that fills it (that
  iteration emits the forced text-EOV; the queue drains over the ones after), so it lives
  outside the loop. Declared inside, it is wiped every step, the continuation silently never
  happens, and **every utterance renders as exactly its first chunk with the correct content**
  — which reads as "the model only learned one chunk" rather than as a decode bug. Cost a
  live run's worth of misleading TB audio on 2026-08-24.
- **Two collate-text paths.** `__call__` uses `_collate_text`, not `_collate_text_per_sample`.
- **A voice-only run has `include_text=False`**, so text targets are not built unless the
  gating includes your flag (see the `emit_duration_token` clause in `compute_loss`).
  ⚠️ **This one bit again on 2026-08-24.** `--bistream_text_loss` was implemented correctly
  in `compute_loss` and was completely inert: targets were never built for it to un-mask, so
  `train/text_loss_norm` simply never appeared while every other metric looked healthy. Any
  future flag that supervises the TEXT stream from a voice-only run must be added to the
  gate at `training.py:638`. Caught only because the smoke run CHECKED the metric.
- **`_collate_text` reads `ex["_direction"]`.** Datasets that omit it get a 50/50 coin flip.
- **Eval must re-derive config from weights.** `load_model` runs `strict=False`, so a widened
  unit head or a missing flag loads silently. Follow `detect_world_mrope` /
  `detect_world_nar` / `detect_n_special_tokens` in `model_loading_utils.py`.
- **`--compile_model` and `--use_gradient_checkpointing` break the recurrent world model.**
  `--compile_recurrent_block` is fine.
- **Decode comparisons need >=3 seeds** (floor: std 0.0089, range 0.018 at n=64) and LCS recall
  must be read beside hyp/ref, because it is partly a length metric.

## Test plan — stages 1-4 PASSED 2026-08-24; stage 5 is the next thing to run

1. **Collator unit test.** Build a batch with `emit_bistream`; assert the token sequence
   alternates as specified, chunk lengths sum to the utterance length, `fill_token` sits at
   each chunk's last frame, `EOV` only after the last, and unistream samples are BYTE-IDENTICAL
   to today.
2. **Interleaver round-trip.** Interleave then uninterleave a two-utterance, chunked example;
   assert the recovered per-utterance frames equal the input exactly.
3. **M-RoPE inspection.** Print position ids for that example; confirm the local axis resets
   per chunk and the global axis separates the two utterances.
4. **Smoke run.** `--max_steps 10 --eval_strategy steps --eval_steps 10 --max_samples 64`.
   Confirm no crash, grad_norm finite, `fill_token` appearing in targets, and text loss NON-ZERO
   on interleaved chunks (item 8 — check the metric, do not assume).
5. **Memorization sanity.** 32 samples, LR 1e-4: should memorize FASTER than unistream did if
   the alignment hypothesis is right. This is the first real signal.
6. **Full run** and the diagnostics below.

Stages 1-4 are green: `tests/test_bistream_collator.py` (8) and
`tests/test_bistream_interleave.py` (3) alongside the existing 122, and two smoke runs (one
at `bistream_prob 0.5` with generation, one at 1.0) complete with finite grad norms, unit CE
falling, and `train/text_loss_norm` non-zero on chunked rows once the `include_text` gate was
fixed.

⚠️ **Stage 5 needs a MATCHED control, and `memorize32_0` is not one** — it is NAR
at mask ratio 1.0 with the duration token. Run a unistream **AR** arm and a bistream AR arm
that differ only in the bistream flags, and compare steps-to-memorize. Keep
`--bistream_text_loss` OFF for both: the sanity test is about ALIGNMENT, and fill_token is
already supervised through the voice CE, so adding a text objective changes what is being
compared.

## What to measure, and against what

Primary: `text_delta` and text-attributed fraction, teacher-forced.

| baseline | text_delta | text-attributed |
|---|---|---|
| AR unistream @23k | +0.0271 | 0.305 |
| NAR @23k (r=0.25) | +0.0025 | 0.029 |
| TEACHER (ceiling) | +0.0594 | 0.463 |

Also: LCS recall (trunc) against AR's 0.1620 @23k, pipeline ceiling 0.8937; per-step
predictive entropy (`unit_confidence_probe.py`) against AR's 6.47 bits and the 12.68 max;
hyp/ref beside every LCS number.

Tools: `world_voice_ar_diagnostics.py`, `cosyvoice_wer_eval.py`, `unit_confidence_probe.py`,
`wer_arm_compare.py` (paired bootstrap), `render_distill_audio.py`.

## Decisions already made (do not relitigate)

- **AR, not NAR.** At matched step AR beat NAR 2.2x on intelligibility and ~10x on text
  attribution; the NAR premise was falsified twice.
- **Duration token OFF for this run.** Its motivation was NAR pre-allocation. Under
  speak-at-will the model has not decided what to say at BOV, so predicting duration there is
  ill-posed. Keep the flag; expect bistream's text-consumption signal to supply the progress
  cue that AR termination was missing.
- **EOV and fill_token separate from the start** (owner, 2026-08-24).
- **5:30, not 5:15.**
- **50/50 unistream/bistream mix.**
