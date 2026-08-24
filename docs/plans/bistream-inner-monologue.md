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

The question was answered instead on the finished `world_tts_memorize32_0/checkpoint-20000`, by
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

## Work items

1. **`token_alignment.py:198` `TokenInterleaver.forward`** — accept an optional per-placeholder
   `(utt_idx, start, length)` map. Today it does `all_placeholders[pos] = ("voice", ex_idx)`
   then `batch_voice[ex_idx, :batch_voice_lens[ex_idx]]`. Default map preserves that exactly.
2. **`token_alignment.py:441` `TokenUninterleaver.forward`** — gather voice positions back
   GROUPED BY UTTERANCE, not by contiguous run. Never exercised with scattered voice.
3. **`data_collator.py:108` `_build_token_sequence`** — emit the chunk-alternating layout for
   bistream samples; keep `[BOV][VOICE_PH][EOV]` for unistream. Note `_collate_text` (line 185)
   is the path `__call__` uses; `_collate_text_per_sample` (232) is NOT. (Cost 20 minutes on
   2026-08-21 — patching the wrong one looks like the flag doing nothing.)
4. **`data_collator.py:286` `_collate_audio_like`** — build the chunk map and per-chunk
   lengths alongside the existing per-utterance tensors. `voice_eov_id` (line 51/67) inserts
   EOV today; `fill_token` insertion per chunk goes here too.
5. **`world_model.py:500` `forward`** — thread the chunk map from inputs to
   `self.token_interleaver(...)` (line 770) and back through `token_uninterleaver` (805).
6. **`world_model.py` ~1891 / ~2095 `_finalize_voice` + end-of-generation flush** — accumulate
   across chunks of one `utt_idx`; emit ONE clip per utterance.
7. **`training.py:1059-1094` voice loss** — targets are (B, T_total) against a coda output of
   (B*n_chunks, chunk_T, V). Map back respecting utterance grouping. Add `fill_token` to the
   targets at each chunk's last frame.
8. **Text-loss carve-out** — `--mask_text_loss_in_synthesis` currently zeroes ALL text loss on
   synthesis examples. Under inner monologue the interleaved text chunks MUST receive gradient
   or the model can never generate its own transcript. Mirror the duration-token exemption
   already in `training.py` (search `duration_only`). **This fails silently**: training loss
   looks healthy and the model simply cannot speak at will.
9. **`world_model.py:generate`** — alternate k text tokens and one voice chunk, tracking
   `utt_idx` and position within the utterance. KV-cache layout MUST mirror training exactly.
   TTS mode feeds real transcript chunks; speak-at-will mode generates them.
10. **M-RoPE check** — the local axis resets per contiguous same-modality run, so it re-anchors
    per chunk (desirable). The GLOBAL axis then becomes the only thing separating utterance 0
    chunk 3 from utterance 1 chunk 3. Verify with `build_mrope_position_ids`
    (`token_alignment.py:512`) on a two-utterance chunked example.

## Traps that have already cost time on this codebase

- **Two collate-text paths.** `__call__` uses `_collate_text`, not `_collate_text_per_sample`.
- **A voice-only run has `include_text=False`**, so text targets are not built unless the
  gating includes your flag (see the `emit_duration_token` clause in `compute_loss`).
- **`_collate_text` reads `ex["_direction"]`.** Datasets that omit it get a 50/50 coin flip.
- **Eval must re-derive config from weights.** `load_model` runs `strict=False`, so a widened
  unit head or a missing flag loads silently. Follow `detect_world_mrope` /
  `detect_world_nar` / `detect_n_special_tokens` in `model_loading_utils.py`.
- **`--compile_model` and `--use_gradient_checkpointing` break the recurrent world model.**
  `--compile_recurrent_block` is fine.
- **Decode comparisons need >=3 seeds** (floor: std 0.0089, range 0.018 at n=64) and LCS recall
  must be read beside hyp/ref, because it is partly a length metric.

## Test plan — verify each stage before the next

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
