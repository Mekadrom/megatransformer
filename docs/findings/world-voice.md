# world-voice (text -> voice)

Speech synthesis from the recurrent trunk into CosyVoice 2 speech tokens, decoded by the
frozen CosyVoice 2 flow decoder. Formerly "world-tts".

**Stack:** text -> frozen SmolLM2-135M -> recurrent trunk (trainable) -> voice coda ->
CosyVoice 2 FSQ tokens (25 Hz, vocab 6562, EOV = 6561) -> frozen flow decoder + campplus
speaker embedding -> audio. Only the trunk and its per-mode adapters train.

**Data:** `cached_datasets/Mekadrom/libritts_r_cosyvoice2_smollm2` — 118,102 train / 4,708 val,
<=10 s, 25 Hz, feature_channels 512. Holds `unit_ids` + SmolLM2 text + campplus-192 speaker
embeddings; no mel, no F0. Codebook is CosyVoice 2's own `flow.input_embedding` (6561x512) at
`val/cosyvoice2_codebook.pt` — the flow decoder's input table, so predicted ids index straight
into what the decoder expects. Utterance lengths: min 25, median 102, mean 112, p90 204
frames; 0.16% hit the 250 cap.

Provenance note: entries dated before 2026-08-21 are recorded from prior sessions of this
project. Entries dated 2026-08-21 were measured in that session. Contamination status for the
older ones is in its own section below — read it before trusting any pre-08-21 number.

---

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
`world_tts_cosyvoice2_smollm2_nar_r1_flat_lr-flat_0`, killed at ~63k steps = **34 epochs over
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

## OPEN

### Can it memorize 32 utterances?
THE NEXT DIAGNOSTIC, and it partitions the hypothesis space cheaply (hours, not days).
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
Run `world_tts_cosyvoice2_smollm2_nar_r1_flat_lr-flat_0`, started 2026-08-21, killed at 63k:
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

### Bistream implementation scope
Smaller than feared: **the interleaver already supports multiple voice placeholders per
example.** `voice_positions[batch_idx]` is a list and `ex_idx` enumerates it, indexing
`batch_voice[ex_idx]` and `batch_voice_lens[ex_idx]` — so the `n` dimension of
`voice_inputs` (B, n, C, T) IS the segment count. It has simply always been 1.

Work required:
1. **Collator** — emit `[BOV][k text][VOICE_PH][k text][VOICE_PH]...[EOV]` and split the voice
   features into that `n` dimension with per-chunk lengths (B, n). This is the bulk of it.
2. **Uninterleaver** — verify it handles n > 1; it should be symmetric with the interleaver
   but has never been exercised that way.
3. **Loss assembly** — voice targets are currently (B, T_total) against a coda output of
   (B*n, chunk_T, V); the mapping back needs care.
4. **Text-loss carve-out** for the interleaved text chunks (see inner monologue above).
5. **generate()** — alternate k text tokens and one voice chunk, mirroring the training layout
   exactly. KV-cache layout must follow.
6. **M-RoPE** — the local axis resets per contiguous same-modality segment, so it re-anchors
   every chunk automatically. Verify `build_mrope_position_ids` behaves sensibly with many
   small segments.

Simplification available for v1: with FIXED chunk sizes, `fill_token` is not strictly needed —
inference can alternate deterministically. CosyVoice 2 needs it because their setting is
streaming. Adding it later is what buys variable-rate chunking.

Also copy the 50/50 unistream/bistream mix rather than going pure bistream: one model does
both, and the unistream half is exactly what is trained today, which keeps the comparison
clean.

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
| `world_tts_cosyvoice2_smollm2_nar_r1_flat_lr-flat_0` | training | **LIVE** — the r=1.0 probe |
| `world_tts_cosyvoice2_smollm2_nar_0` | 23000 | stopped; the cosine-masked NAR baseline |
| `world_tts_cosyvoice2_smollm2_mrope_scale_text_0` | 44000+ | stopped; best AR arm, all checkpoints kept |
| `world_tts_cosyvoice2_smollm2_distill_0` | 55000 | stopped; the distillation result above |
| `world_tts_cosyvoice2_smollm2_mrope_0` | 28000 | killed (voice-scaled M-RoPE) |
| `world_tts_cosyvoice2_smollm2_0` / `_nocurric_0` | 27000 / 23000 | stopped; pre-distill baselines |
| `..._0__baseline_ngram_*` | — | synthetic text-free baselines, not models |
