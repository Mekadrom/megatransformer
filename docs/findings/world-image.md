# world-image (text -> image)

Image synthesis from the recurrent trunk into a frozen diffusion decoder (Z-Image / SDXL) via
a trainable conditioning adapter.

⚠️ The **ESTABLISHED (owner-reported)** section below was reported in conversation by the
project owner during the 2026-08-21 world-voice session. Those numbers come from the WARM-START
lineage (whiten -> t3_1 -> t3_2 -> xskip, ~63k cumulative steps) and have not been re-measured
here — treat them as owner-reported.

Everything from **"world-image session, 2026-08-21"** onward was measured directly by the
world-image session, on the FROM-SCRATCH runs, with the protocol stated per entry.

---

## ESTABLISHED (owner-reported)

### Best configuration is NAR, not autoregressive
Qwen features interpolated to a fixed 64-length sequence and diffused in parallel with a
flow-matching head, conditioned via a Q-Former whose input is the recurrent trunk's OUTPUT at the
64 image gen-query positions (not the learned query embeddings themselves — the distinction is
load-bearing, see the compression entry below). The head and Q-Former see NO text; text reaches
them only through those trunk states. T4
(per-token autoregressive flow head, MAR-style) performed badly by comparison.

The 64 is not structural: SDXL's CLIP conditioning is 77 tokens, and 64 covers a majority of
Qwen condition sequences for the training mixture, so it interpolates UP rather than squeezing
down. It also equals the gen-query count, making the Q-Former a 64->64 cross-attention module.

### Guidance is the win, not the flow head per se

⚠️ **Convention, verified in `eval_zimage_adapter.py:261-265`:** every CLIPScore in this file is a
RAW cosine — `F.normalize` on both encodings, ViT-B-32 `laion2b_s34b_b79k`, no `2.5x` rescale and
no `max(0,.)`. So GT 0.368 is not a bad score; on the published `w=2.5` convention it is ~0.92. Do
not compare these figures against papers without multiplying by 2.5.

⚠️ **Two different gain measurements exist on two different probe sets — do not mix them.** This
table's `0.266 -> 0.283` is on the probe whose GT ceiling is **0.368**. The `--output_gain` help
text in `eval_zimage_adapter.py:73` quotes `0.287 -> 0.303` against a GT of **0.345**, a different
probe. Both are believed correct; quoting a gain delta without its ceiling is what makes them look
contradictory.

| config | inference | CLIPScore | % of GT |
|---|---|---|---|
| GT (true Qwen3 conditioning) | — | 0.368 | 100% |
| naive MSE, no whitening (4k) | deterministic | 0.034 | 9% |
| MSE + whitening (20k) | deterministic | 0.266 | 72% |
| MSE + whitening + gain 1.33 | deterministic | 0.283 | 77% |
| T3 flow + x1pred, unguided | w=1 | 0.224 | 61% |
| T3 flow (control), unguided | w=1 | 0.277 | 75% |
| T3 flow + xskip, unguided | w=1 | 0.287 | 78% |
| T3 flow + x1pred + min_snr | w=3 | 0.351 | 95% |
| T3 flow (control) | w=3 | 0.356 | 97% |
| T3 flow + xskip | w=3 | 0.359 | 98% |

Readings:
- **Unguided flow ties gain-corrected regression** (0.277-0.287 vs 0.283). A generative head
  buys nothing over a variance-calibrated point estimate at w=1.
- **Guidance takes it 77% -> 98%**, and guidance requires a generative model — a regression
  head has no unconditional branch to guide against. Gain amplifies isotropically; CFG
  amplifies the departure from the unconditional, which is direction- and prompt-dependent.
- **Whitening is the largest single lever** (9% -> 72%), and it is preprocessing, not
  architecture. Any write-up ordering these honestly puts whitening first.
- **x1pred looks falsified**: worst unguided (61% vs 75%) and still below control with min_snr
  and guidance (95% vs 97%).
- **xskip is small and shrinks under guidance** (+0.010 unguided, +0.003 guided), consistent
  with de-noising and guidance repairing overlapping failures.

### Prior art
The closest published ancestors are unCLIP / DALL-E 2's diffusion "prior" (generate the
conditioning embedding, feed a frozen decoder; their diffusion prior beat their AR prior) and
MAR (a diffusion head modelling continuous tokens rather than regressing them). The
surrounding family — GILL, ELLA, Emu, NExT-GPT, SUR-adapter — maps LLM features to frozen
diffusion conditioning by REGRESSION. Sequence-level generative conditioning for a frozen DiT
is the part that may be unclaimed; verify against current literature before asserting novelty.

---

## OPEN

### Is the Q-Former necessary?
A from-scratch ablation removing it is planned. Note that world-voice's NAR head is natively
Q-Former-free (the coda reads trunk states directly), so a working voice result would be
independent evidence on the same question.

### Why is the ceiling only ~0.368 CLIPScore?
Whether the low ceiling is the data mixture, caption quality, or CLIPScore's own insensitivity is
unresolved. See the convention note above first — 0.368 raw cosine is ~0.92 rescaled, so the
ceiling is less alarming than it reads.

⚠️ **Corrected 2026-08-21 (world-image session):** the original entry argued that "matches the
GT-prompt ceiling" shows the conditioner is not the bottleneck. That inference is not safe. The
SDXL and Z-Image arms fail by OPPOSITE mechanisms (Z-Image under-dispersed, rho .86 / R^2 .74;
SDXL over-dispersed, rho .60 / R^2 .21) yet both render ~78% of their GT — which suggests 78% is
simply what "right subject, wrong details" SCORES on CLIPScore, not a capacity wall.
**Do not use the CLIPScore ratio as a capacity readout.**

---

# world-image session, 2026-08-21

Measured on the two live from-scratch runs (`world_image_zimage_qwen_t3_xskip_0` and
`..._t3_xskip_trunkctx_0`, identical but for `flow_ctx`), not the warm-start lineage above.

## ESTABLISHED

### Eval protocol: repeat noise is ~0.004, checkpoint-to-checkpoint variance is ~10x that
**Load-bearing, read this before trusting any table here.** `t3_xskip/checkpoint-11000` scored
**0.201 and 0.205** on two independent runs of the same eval (w=3, n_samples 2, flow seeds pinned)
— repeat noise ~0.004, attributable to unpinned thought-init (`like-init`, std 0.02). But the same
arm went **0.205 @ 11k -> 0.148 @ 12k**, a 0.055 swing, with NO training-loss anomaly (per-step
loss scatters 0.57–1.32 throughout; 12k is indistinguishable from 11k).

The probe is 8 prompts x 2 samples, and per-prompt means range 0.12–0.29 — a prompt either snaps
into a coherent image or it does not. So the *measurement* is precise and the *checkpoint* is not.
**Read trends across several checkpoints; never read adjacent pairs.** The `within-prompt sd`
printed on every eval line (~0.03) is prompt/sample spread, NOT an error bar on the mean.

### The from-scratch guidance optimum is w=4.5, not w=3
Sweep on `t3_xskip/checkpoint-8000`, n_samples 2, same eval path as the monitor:

| w | 1.0 | 1.5 | 2.0 | 3.0 | **4.5** | 6.0 |
|---|---|---|---|---|---|---|
| CLIPScore | 0.088 | 0.116 | 0.150 | 0.158 | **0.180** | 0.169 |

w=3 was inherited from `t3_2/ckpt-7000`, which carried ~40k steps of warm-start lineage. A
from-scratch arm carrying `x_skip` from step 0 is less converged and MORE guidance-dependent, so
the optimum moved right — as the noise-leak model predicts. Consequences: every from-scratch number
recorded at w=3 is a floor understating the arm by ~0.02; the arm-vs-arm comparison is unaffected
(identical w on both). Note 2.0->3.0 is only +0.008 — sweeping only around the incumbent would have
read a shoulder and hidden the peak.

### The trunk emits gen-query states that are nearly prompt-INVARIANT
`scripts_local/trunk_compression_probe.py`, `t3_xskip/checkpoint-11000`, 16 prompts, cosine on
K-slot-resampled flattened states:

| stage | cos(same position, DIFFERENT prompts) | cos(DIFFERENT positions, same prompt) |
|---|---|---|
| `x_0` (learned queries, PRE-recurrence) | 0.9989 | 0.0447 |
| iter0 (POST) | 0.9965 | 0.0276 |
| iter25 (POST, fed to the head) | **0.9963** | 0.0268 |

Positions are near-orthogonal (~88.5°) — that axis is healthy. Two unrelated prompts move the same
slot by **~4.9°**, up from 2.7° at `x_0` (where the only variation is prompt LENGTH shifting
absolute positions, since the queries are prompt-independent constants). Position identity gets
~88° of angular separation; all of the prompt conditioning lives in ~2.2°.

In magnitudes: decomposing `h_i = mu + r_i` over prompts, `mean||r||/||mu|| = 0.046`. The
prompt-varying component is ~4.6% of the constant it rides on.

**Mechanism — two multiplying suppressors, both measured:**
1. The learned gen queries are a norm-**83.1** constant (`std 3.0 * sqrt(768)`), ~3x a unit-std
   text token's norm (~27.7). `scale_embeddings` is False for this config, so no extra factor.
2. Depth-scaled init sets every recurrent `o_proj` / `ffn.condense` output to
   `std = 1/sqrt(5*d*l_eff) = 0.00285` (d=768, mean_thinking_steps=32) — Huginn's trick including
   its factor-of-5 margin. Each iteration writes ~nothing at init and must learn upward.

A tiny write rate into a large constant.

### `image_syn_seq_var` cannot detect this, and reads healthy while it happens
At step 11400 the live run logs `recurrent_out/image_syn_seq_var = 0.974` — HIGHER than
`text_seq_var` (0.776) — while across-prompt conditionality is 0.042. It is a variance across
SEQUENCE POSITIONS, which learned per-position constants make large BY CONSTRUCTION. Every
image-side TB metric (`image_syn_seq_var`, `image_token_var`, `image_entropy`) reduces over `dim=1`
or `dim=-1` and then means over the batch — the batch axis, where different prompts live, is
averaged and never varianced.

Added `recurrent_out/image_syn_cond_frac` (`world_model.py`, guarded on >=2 synthesis samples in
the batch) to log `||r||/||mu||` directly. **Never certify conditioning with a spread-across-
positions statistic.**

### Amplifying the trunk's conditional component substitutes for CFG, and adds nothing once CFG is on
`--trunk_gain` in `eval_zimage_adapter.py` feeds `mu + gamma*r` to the head (calibrates `mu` over
the prompt list in one extra pass; pins thought-init so `like-init` noise is common-mode).
`t3_xskip/checkpoint-10000`:

| | g=1.0 | g=2.0 | g=3.0 | g=5.0 |
|---|---|---|---|---|
| w=1.0 | 0.105 | — | **0.163** | 0.154 |
| w=3.0 | 0.198 | **0.206** | 0.189 | 0.169 |

+0.058 at w=1 (~55% of the unguided->w3 distance); +0.008 at w=3, inside the noise. The peak moves
LEFT as w rises (g~3 at w=1, g~2 at w=3) — the redundancy signature.

**Why this fails where the adapter-output gain succeeded (+30%, 0.266->0.283):** that gain corrected
a SYSTEMATIC shrinkage with a known signature (alpha ~= R^2, the conditional-mean fingerprint) —
real signal uniformly under-emitted. The trunk residual `r` has no such signature; it mixes prompt
signal with thought-init noise and mis-imported attention, so scaling amplifies error alongside
signal. **Conditioning the trunk never imported cannot be rescaled back in.**

### The frozen SmolLM2 spine is NOT the conditioning bottleneck
Minimal pairs holding the bag of words fixed while inverting meaning ("a single **red** door in an
endless **white** void" -> "**white** door / **red** void"), measured at every stage
(`scripts_local/text_signal_localization.py`), as swap/other distance ratio:

| stage | ratio |
|---|---|
| SmolLM2 hidden states | **0.318** |
| after `input_proj` + translator | 0.310 |
| trunk output @ gen queries | 0.179 |
| emitted conditioning | 0.155 |
| Qwen3 ground truth (the bar) | 0.366 |

The encoder supplies ~87% of GT meaning-sensitivity and the prelude preserves it; the TRUNK is where
it collapses. Independently, closed-form ridge onto whitened k64 Qwen targets (2048 captions,
best-lambda holdout R^2): pretrained SmolLM2 **0.2887**, same architecture with RANDOM weights
0.2014, embedding table only 0.1767. Pretraining buys +0.087 over random same-width features — real,
but 70% of it is matched by random projection, and the from-scratch alternative is TRAINED rather
than random.

**Therefore: distilling SmolLM2 into a smaller trainable prelude is not worth GPU time for
conditioning fidelity** — no encoder can raise the path above what the trunk propagates.
Adaptability remains a separate, weaker motivation. Note the ablation is not a flag flip: SmolLM2
mode sets `special_token_base = llm.vocab_size` and uses SmolLM2's tokenizer, so shards must be
re-tokenized — the same blocker `world-text.md` has.

⚠️ Protocol note that cost a re-run: **any text-conditioning probe on this path MUST run at the
operating guidance.** At w=1 the conditioning is noise-dominated — resampling the flow head moves it
7x more than changing the prompt entirely (d_seed 0.881 vs d_other 0.131). A second protocol trap:
these minimal pairs hold the token SET fixed, so **mean-pooling over tokens is blind to them by
construction** — a first version pooled and reported a flat 0.119 at every stage, measuring the
pooling rather than the model. Resample to K slots and flatten.

---

## OPEN (world-image session)

### Q-Former ablation is tracking EVEN, four gap-closures in
w=3, from-scratch, matched steps where available:

| step | t3_xskip | trunkctx (`flow_ctx="trunk"`, no Q-Former) |
|---|---|---|
| 8000 | 0.158 | 0.134 |
| 10000 | 0.195 | 0.155 |
| 11000 | 0.201–0.205 | 0.137 |
| 12000 | 0.148 | 0.186 |
| 13000 | — | 0.205 |

`t3_xskip` leads at matched steps but `trunkctx` reaches the same 0.205 about 2k steps later, which
is what a convergence-RATE difference looks like — unsurprising given `t3_xskip` carries 33M more
active parameters. The gap has looked real and then closed four times. ⚠️ Given the variance entry
above, no single pair of steps here means anything. Settle it at the matched-step comparison against
the `whiten_0` baseline at 20k (that run has every checkpoint 1000–20000 on disk). Note `whiten_0`
is a POINT head, so there is no matched-w comparison against it — it is flow-arm-at-some-w vs
point-head-at-no-guidance, which is exactly why the w=3-vs-4.5 choice is not cosmetic.

### Both from-scratch arms PEAK at 0.205 and then decline while training loss does not
Replicated across two independent runs, offset by the ~2k steps that separates them:

| | peak | +1k | +2k |
|---|---|---|---|
| `t3_xskip` | **0.205** @ 11k | 0.148 | 0.128 |
| `trunkctx` | **0.205** @ 13k | 0.187 | 0.151 |

Training loss over the same windows shows NO corresponding degradation: `t3_xskip` IMPROVED
(mean ~0.99 over 10.6k-12.3k -> ~0.86 over 12.8k-13.2k) and `trunkctx` is flat (~1.07 over
13.8k-15.2k). Loss down, render down.

Two independent arms peaking at the same value and declining the same way is much harder to
attribute to the 8-prompt probe's variance than a single arm's dip was — though note it is still
only 2 post-peak checkpoints each, so a shared transient is not excluded. The next two
checkpoints per arm settle it.

This is the anti-correlation already documented on the GAIN axis (training loss and render
disagree; the DiT wants dispersion over L2), appearing here as a function of TRAINING TIME: the
flow objective rewards an increasingly conditional-mean-like prediction, which a frozen decoder
renders blander. See the dispersion entry in `cross-modal.md` if it has been promoted there.

⚠️ **Consequences for the 20k comparison:** by 20k both arms may be well past peak, so a
matched-step reading at 20k could compare two degraded checkpoints and understate both. Compare
PEAK-to-peak as well as step-to-step, and do not prune old checkpoints — for `t3_xskip` the best
checkpoint so far is 11000, not the latest.

### Late recurrent iterations now DECAY conditioning (new at 11k, absent at 10k)
`cond/const` peaks 0.0472 at iteration 2, holds ~0.046 through iteration 18, then falls to 0.0327 by
iteration 25 (-29%); text positions do the same (1.03 -> 0.57). At 10k the curve was flat to the end.
Also note ~96% of the conditioning arrives in iteration 0 — 23+ further iterations add ~5%. Whether
this is a transient or a trend matters for recurrent depth generally; re-measure at 20k and 40k
before drawing any conclusion for the refinement thesis.

### Untested levers for the gen-query compression, cheapest first
- **Cap the iterations** (~18). Free, inference-only: `max_iterations_override` is plumbed through
  `recurrent.forward` and `world_model.py:1918`, and `multimodal_chat.py:129` exposes
  `--image_iteration_override`; the zimage eval needs the same flag.
- **Lower `image_gen_query_init_std` 3.0 -> ~1.0.** De-risked: positional identity comes from the
  query DIRECTIONS being distinct and cosine is SCALE-INVARIANT (measured 0.027 — near-orthogonal at
  any magnitude). Init-time only, so it needs a fresh run.
- **Normalize at the trunk output** before the head (RMSNorm at gen positions, or a learned
  per-position offset). Also a precision argument: 4% signal on a 96% constant burns ~4–5 of bf16's
  8 mantissa bits.
- **Inject text into the gen queries directly** (FiLM / cross-attn / pooled-text concat). Attacks
  the root — the queries start prompt-blind by construction (0.0019 at `x_0`) and the recurrence
  must import 100% of the conditioning through attention.
- ⚠️ **Relaxing the depth-scaled init** is the obvious lever and should be held back; recurrent
  stability is LR-sensitive in this project and that init exists for a reason.
- **Diagnostic that picks between them:** attention mass from gen-query positions to text positions.
  Heavy attention + small write => dilution (init std / norm / depth-init). No attention => routing,
  and only direct injection fixes it.

---

## RETRACTED

### "Learned gen queries fixed the image gen-query collapse" (believed 2026-04-15 -> 2026-08-21)
The original finding stands for the axis it was measured on: with `gen_query_mode='positional_only'`
all 64 gen positions collapsed to near-identical outputs, and learned queries at std=3.0 fixed that.
`image_seq_var ~= 6e-6` -> 0.974 is real.

What was wrong is the scope of the conclusion. The diagnostic used to certify the fix
(`image_seq_var`, a spread ACROSS POSITIONS) rises **by construction** the moment each position gets
its own learned constant, so it could never have detected the ACROSS-PROMPT collapse, which was left
untouched: cos 0.9963 between unrelated prompts at step 11000, four months later. The 2026-04 entry
also recorded `image_recurrent_tokens |sample0 - sample1| ~= 0.003` across different prompts as a
collapse signature — that quantity was never re-checked after the fix.

Also retracted: the implication that std=3.0 is what fixed it. Distinct LEARNABLE DIRECTIONS fixed
it; the magnitude is separable and, per the mechanism entry above, is now actively suppressing
conditioning.
