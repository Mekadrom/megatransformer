# world-image (text -> image)

Image synthesis from the recurrent trunk into a frozen diffusion decoder (Z-Image / SDXL) via
a trainable conditioning adapter.

📁 **RUN PATHS MOVED 2026-08-26.** Runs now live in `runs/world_image/` (split out from
the old `runs/world/`, which mixed image and voice), and the redundant `world_image_` prefix was
stripped from every directory. Run names in this file have been UPDATED to match, matching how
`world-voice.md` records its own move:

    old: runs/world/world_image_zimage_qwen_<name>
    new: runs/world_image/zimage_qwen_<name>

⚠️ Commit messages and older memory entries still quote the OLD names, so when cross-referencing
git history, add `world_image_` back.

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

Measured on the two from-scratch runs (`zimage_qwen_t3_xskip_0` and
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

### ~~The from-scratch guidance optimum is w=4.5, not w=3~~ — RETRACTED 2026-09-11
~~Sweep on `t3_xskip/checkpoint-8000`, n_samples 2:~~ 0.088 / 0.116 / 0.150 / 0.158 / **0.180** /
0.169 at w = 1.0 / 1.5 / 2.0 / 3.0 / 4.5 / 6.0. ~~The optimum moved right; every from-scratch number
recorded at w=3 is a floor understating the arm by ~0.02.~~

⛔ **WRONG, and the error was the CHECKPOINT.** ckpt-8000 is a barely-trained arm (its own w=3 score
was 0.158, less than half the converged 0.346), measured at n_samples=2 where the noise floor is
~0.008 — the same size as the 2.0->3.0 step the entry itself flagged. Peak-w drifts as conditioning
strengthens, which the entry ALSO said, and then applied the number to converged arms anyway.
The "every number is a floor, understating by ~0.02" caveat was carried on several tables in this
file for three weeks. **It was never true of the converged runs. Drop it wherever it appears.**
Superseded by the sweep below.

### Guidance is FLAT from w=3 to w=6 on a converged baseline
`zimage_qwen_t3_xskip_lr1e-4_1e-4_1e-4_cosine_0/checkpoint-97000` (the best from-scratch checkpoint
at w=3), **N=8**, `--flow_seed_base 4242` so every w sees IDENTICAL draws, one checkpoint across all
w so per-checkpoint variance is a constant offset and cannot move the peak:

| w | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 | 4.5 | 5.0 | 6.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean | 0.308 | 0.335 | 0.335 | 0.344 | 0.346 | **0.350** | 0.349 | **0.350** | **0.350** | 0.347 |
| best-of-8 | 0.362 | 0.380 | 0.370 | 0.380 | 0.377 | 0.385 | 0.384 | **0.389** | 0.383 | 0.379 |

⭐⭐ **w=3 to w=6 spans 0.346-0.350 — a 0.004 range against ~0.004/point residual noise (unpinned
thought-init). There is no peak, there is a PLATEAU.** All the action is below w=3: 1.0 -> 2.5 gains
+0.036, and 2.5 -> 6.0 gains +0.003.
⭐ **Therefore w=3 is within noise of optimal for a converged arm, and every matched comparison in
this file was made at a sensible operating point.** No table needs re-running or re-qualifying.
⭐ best-of-8 peaks slightly higher and later (0.389 at w=4.5): if you SAMPLE AND SELECT, a bit more
guidance still helps even though the mean is flat. The mean and the best-of-N optima are not the
same quantity.
⚠️ One checkpoint, one seed base, 8 prompts. This establishes the SHAPE for a converged t3_xskip
arm; it does not transfer to a different architecture. T4 (AR) in particular has a structural
reason to differ — guidance compounds along an autoregressive sample in a way it cannot for a
one-shot parallel head — and has NOT been swept.


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

### MATCHED-STEP 20k: the point head OUT-RENDERS the flow head, at equal conditioning quality
The comparison the from-scratch runs were held for. `whiten_0` has every checkpoint 1000-20000, so
this is same-step, same 8-prompt probe, same harness.

| arm | step | eval MSE | R^2 | CLIPScore |
|---|---|---|---|---|
| `whiten_0` — point head, pure MSE w=1.0 | 20000 | **0.275** | **0.725** | 0.266 unguided / **0.283** +gain |
| `t3_xskip` — flow | 20000 | 0.297 | 0.703 | **0.246** (w=3) |
| `trunkctx` — flow | 20000 | 0.307 | 0.693 | 0.175 (w=3) |

⭐ **CONDITIONING QUALITY IS ESSENTIALLY MATCHED** (R^2 0.725 vs 0.691-0.693, gap ~0.03). Pure MSE
at weight 1.0 for 20k steps teaches the trunk barely better than flow-with-0.1-aux does. Same
result at 14k (0.312 vs 0.337). **So the mean-regression phase is NOT what the warm-start recipe
buys** -- it is not a better trunk teacher.

⭐⭐ **RENDER: `t3_xskip` has CAUGHT UP; `trunkctx` has not.** 0.246 (w=3) vs the point head's
0.266 unguided is a 0.020 gap, well inside this probe's +-0.05 per-checkpoint spread -- a tie.
Against 0.283 (point + free gain correction) the point head keeps a 0.037 edge, still under one
SE. `trunkctx` at 0.175 IS clearly behind. And `t3_xskip` is climbing fast (0.211 -> 0.231 ->
0.231 -> 0.234 -> 0.246 over 16k-20k), so it should pass the point head shortly.

⭐ **IT DID, at 21k: 0.274** — above the point head's raw 0.266 and at parity with its
gain-corrected 0.283, at 74% of GT. **And it passed the GAIN-CORRECTED point head too, confirmed
at 32k**: smoothed 0.2825 over 29k-32k (0.292 / 0.263 / 0.282 / 0.293), three of four checkpoints
at or above 0.28, 80% of GT. Confirmed on a window rather than the single 0.292 at 29k, which
did not hold at 30k (0.263) — this probe produces threshold-crossing outliers readily. So the flow
head needs ~20k steps to MATCH a regression head and ~21k to BEAT it.

⛔ **~~From-scratch is a viable recipe; warm-starting buys a one-time ~20k head start, not a
structural requirement.~~ — RETRACTED 2026-08-27.** That compared from-scratch against the
POINT-HEAD baseline only, and read "passed the baseline" as "matches the lineage". It does not.
`ws_xskip_0`/ckpt-3000 RE-MEASURED in the current harness (N=4, same probe — this number had never
been verified here): **0.355 at w=3 / 0.268 at w=1**, against the historical quote of 0.359. It
verifies to 0.004.

| | w=3.0 | w=1.0 | cumulative steps |
|---|---|---|---|
| `ws_xskip_0` (whiten -> t3_1 -> t3_2 -> xskip) | **0.355** | **0.268** | ~63k |
| `t3_xskip_0` from-scratch, final | 0.326 | 0.245 | 100k |
| warm restarts ON TOP of from-scratch | 0.3235-0.3245 | 0.240-0.251 | 103k |

**The staged lineage beats from-scratch by ~0.03 on BOTH axes with ~37k FEWER cumulative steps.**
⭐⭐ And NOT because its trunk learned the conditioning better — from-scratch ends with BETTER
conditioning (eval MSE 0.233 / R^2 0.767 vs `whiten_0`'s 0.275 / R^2 0.725). Whatever staging
buys, it is in the HEAD or the trajectory, not the trunk.
⚠️ It is also not a restart artifact: three 3000-step warm restarts from the from-scratch
checkpoint (control / auxdetach / offset) land at 0.322-0.329, matching the `ws_*` restart-only
confound of +0.009.

⚠️ **This CORRECTS what this entry said an hour earlier**, when it was written from `trunkctx`@20k
plus `t3_xskip`@18k (0.231): "the point head renders much better" and "a from-scratch flow run
spends its first ~20k steps STRICTLY WORSE than the baseline". `t3_xskip`@20k falsifies the strong
form. The claim holds for `trunkctx` only, and reading a 2k-early checkpoint as if it were the
matched one is exactly the adjacent-step error the protocol entry warns about -- committed here
while quoting that same entry.

⭐⭐ **What survives: the warm-start recipe's value is the FLOW HEAD's convergence, not the
trunk's.** Conditioning R^2 is matched throughout, so mean-regression is not a better trunk
teacher; what MSE-pretraining buys is skipping the stretch where the flow head has not yet caught
up. That stretch now measures as roughly 20k steps in the `t3_xskip` config -- not indefinite, and
shorter than the ~63k the warm-start lineage suggested.

---

## OPEN (world-image session)

### The flow head's context MATTERS: Q-Former output beats raw trunk states
`t3_xskip` (flow head reads the Q-Former output `q`) vs `trunkctx` (reads the self_enc'd trunk
states `x`). One config field apart, same seed, same schedule.

| | matched step 20k | 4-ckpt window | unguided (w=1) band |
|---|---|---|---|
| `t3_xskip` | **0.246** | 0.246 (18k-21k) | 0.156-0.177 |
| `trunkctx` | 0.175 | 0.195 (20k-23k) | 0.097-0.147 |

The gap grew monotonically across seven windows (0.016 -> 0.025 -> 0.041 -> 0.044 -> 0.051) to
~2 SE, and `trunkctx` reads lower than `t3_xskip` even when it is 2k steps FURTHER along
(0.207 @22k vs 0.246 @20k).

⛔ **THE "FINAL READ" BELOW WAS WRONG TOO. At 85k both gaps collapsed to ~+0.013 and the arms
have essentially converged.** Full history:

| window | w=3.0 | w=1.0 |
|---|---|---|
| 20k | +0.071 | — |
| 53k-60k | +0.0255 (7/8) | +0.0396 (8/8) |
| 61k-70k | +0.0237 (8/10) | +0.0448 (10/10) |
| **71k-86k** | **+0.0132 (13/15)** | **+0.0131 (11/15)** |

The w=1 gap I recorded as "stabilized at +0.045, 10/10, measure unguided" fell 70% in the NEXT
window. Both arms are also deep in cosine decay by then (LR ~10-12% of peak, plateaued since
~45k), which is exactly when a convergence-RATE difference would close.

⭐ **ARC OF THIS FINDING, worth more than the finding:** "tracking even" (8-13k) -> recorded as a
real ~0.05 effect (23k) -> downgraded to 0.024 (60k) -> +0.013 and converging (85k). **The FIRST
reading was closest to right.** The middle-period separation was a transient of differing
convergence rates — named in the original single-seed caveat, then argued past TWICE. When a
caveat names an alternative and the data later fits it, that is the alternative, not a coincidence.

**Practical conclusion:** `cross_dec` in the flow head's context buys CONVERGENCE SPEED, not
endpoint quality — `trunkctx` reaches the same place ~2k steps later and finishes within noise.
The one thing that has NOT converged is the qualitative failure mode (occasional unguided mode
collapse, below), which the mean score is a poor instrument for; that remains the live reason to
prefer the Q-Former or a cheaper constant-remover.

~~**REVISED TWICE. Read at 70k: the gap SHRANK, then STABILIZED — it is not closing.**~~

| matched-step window | w=3.0 | w=1.0 |
|---|---|---|
| 20k (single) | +0.071 | — |
| 53k-60k (n=8) | +0.0255 (7/8) | +0.0396 (8/8) |
| **61k-70k (n=10)** | **+0.0237 (8/10)** | **+0.0448 (10/10)** |

So it is neither the ~0.05 capability wall first recorded NOR a convergence-rate advantage that
closes: it decayed to ~0.024 guided / ~0.045 unguided and held there for 20k steps.
⚠️ I called convergence at the 70k pair alone (+0.003 at w=3) — a single matched pair inside a
window averaging +0.024. Same adjacent-step error as before; the window is the unit.
⭐⭐ **MEASURE THIS UNGUIDED.** The effect is ~2x larger at w=1 (+0.045 vs +0.024) and unambiguous
there (10/10 vs 8/10), because CFG partially substitutes for what `cross_dec` provides. Guided
evaluation understates it by half AND is the noisier half.

⭐⭐⭐ **THE UNGUIDED DEFICIT IS OCCASIONAL MODE COLLAPSE, NOT UNIFORM DEGRADATION.** The owner
spotted `trunkctx` at 74k rendering a PORTRAIT OF A PERSON for "an astronaut riding a horse" at
w=1. The logs carry the signature (steps >=50k):

| | w=1.0 within-prompt sd | w=3.0 within-prompt sd |
|---|---|---|
| t3_xskip | 0.0249 (max 0.037) | 0.0238 (max 0.039) |
| trunkctx | **0.0325 (max 0.057)** | 0.0216 (max 0.033) |

`trunkctx`'s draw-to-draw variance is +31% vs `t3_xskip` UNGUIDED and slightly TIGHTER guided --
the excess is entirely guidance-removable. Most draws are fine; some abandon the prompt, which
inflates spread while barely moving the mean. A portrait is plausibly the dominant mode of a web
image-text corpus, so this is falling to the DATA PRIOR -- the textbook failure for a conditional
signal too weak to escape it without amplification, i.e. exactly ~10%-of-constant with no CFG.
⚠️ **A CLIPScore mean over 8 prompts is close to the worst instrument for this.** Track w=1
within-prompt sd (or eyeball w=1 renders) when judging conditioning strength.
⚠️ At N=2, `best-of-N - mean` is ALGEBRAICALLY IDENTICAL to the reported sd -- not independent
evidence; do not quote both.
⭐ **The advantage is LARGER UNGUIDED than guided (0.040 vs 0.026)** — consistent with the
constant-remover story plus the guidance-redundancy pattern seen everywhere else in this file: CFG
amplifies the conditional component, partially doing for `trunkctx` what `cross_dec` does for
`t3_xskip`, so the arms look most similar exactly where guidance is strongest. Same shape as the
gamma-gain result (trunk gain substituted for CFG and added nothing once CFG was on).

⭐ **Mechanism, and it was predicted before the gap appeared:** the raw trunk states hand the flow
head conditioning at ~4% of a norm-83 constant (see the compression entry). Cross-attention with
its own learned queries can put whatever it likes in the constant component, so `cross_dec` can act
as a **learned constant-remover** — which is exactly the advantage this measures. It also explains
the ~15k-step delay before separation: the flow head can partially compensate on its own first.

⚠️ **Caveats.** Single seed. And this is NOT the "is the Q-Former redundant" result — `cross_dec`
is built, executed and trained in BOTH arms (see the scope note below); only the flow head's input
differs. If the constant-remover reading is right, **lever D (normalise at the trunk output) should
recover most of this for a fraction of 18.9M params**, and `..._nocrossdec` becomes a test of the
MECHANISM rather than of redundancy.

⚠️ Supersedes this file's earlier "tracking even / cross_dec can be dropped for free" reading,
which was written at 8k-13k before the arms separated.

### ~~Q-Former ablation is tracking EVEN, four gap-closures in~~ — superseded (see above)
w=3, from-scratch, matched steps where available:

⚠️ **The arm name is misleading and so was my description of it.** `zimage_adapter.py:294-295`
runs `q = self.cross_dec(q, x)` UNCONDITIONALLY and `seq_pred = self.seq_head(self.seq_norm(q))`
always reads `q`, whatever `flow_ctx` is. So in `trunkctx` the Q-Former is still present, still
executed, and still TRAINED -- by the 0.1 aux MSE, whose gradient runs
seq_head -> cross_dec -> self_enc -> trunk. What is ablated is only whether the FLOW HEAD is
conditioned on `q` or on the raw trunk states `x`. This arm therefore does NOT answer "can
cross_dec's 18.9M params be dropped"; that needs `flow_ctx="trunk"` AND
`flow_aux_mse_weight=0` (or `seq_head` rewired to read `x`). It also gives the two arms' matched
trajectories a duller explanation than the learned-constant-remover hypothesis: both train the
same Q-Former on the same auxiliary objective.

| step | t3_xskip | trunkctx (`flow_ctx="trunk"` — flow head only; Q-Former still trained) |
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

### ~~Both from-scratch arms PEAK at 0.205 and then decline~~ — WITHDRAWN 2026-08-22, same day
**The next checkpoint falsified this.** `t3_xskip` went 0.205 @ 11k -> 0.148 -> 0.128 -> **0.197
@ 14k**. It recovered; there was no decline, just the +-0.05 per-checkpoint oscillation already
documented in the eval-protocol entry above. `trunkctx`'s 0.205 -> 0.187 -> 0.151 is the same
oscillation and should not be read as a trend either.

**How this got recorded:** three consecutive down-steps on one arm, then three on the other,
looked like replication across independent runs. It was the SAME 8-prompt variance twice. The
protocol entry above says not to read adjacent pairs; three points in a row is not enough more
to overturn it, and "it replicated" is not independent evidence when both arms share a probe
this small. Keeping the original claim per the RETRACTED convention.

The loss observation below still stands on its own and is what made the story attractive:

| | peak | +1k | +2k | +3k |
|---|---|---|---|---|
| `t3_xskip` | 0.205 @ 11k | 0.148 | 0.128 | **0.197** |
| `trunkctx` | 0.205 @ 13k | 0.187 | 0.151 | — |

Training loss over the same windows shows no corresponding degradation: `t3_xskip` mean ~0.99
over 10.6k-12.3k -> ~0.86 over 12.8k-13.2k, `trunkctx` flat ~1.07 over 13.8k-15.2k. With the
decline withdrawn this is no longer evidence of anything — flat loss against oscillating render
is exactly what you expect when the render metric is the noisy one.

Two independent arms peaking at the same value and declining the same way is much harder to
attribute to the 8-prompt probe's variance than a single arm's dip was — though note it is still
only 2 post-peak checkpoints each, so a shared transient is not excluded. The next two
checkpoints per arm settle it.

This is the anti-correlation already documented on the GAIN axis (training loss and render
disagree; the DiT wants dispersion over L2), appearing here as a function of TRAINING TIME: the
flow objective rewards an increasingly conditional-mean-like prediction, which a frozen decoder
renders blander. See the dispersion entry in `cross-modal.md` if it has been promoted there.

⚠️ **What survives the withdrawal:** do not prune old checkpoints, and quote the BEST
checkpoint alongside the latest — not because the arms are degrading, but because per-checkpoint
variance is +-0.05 and any single reading is a coin flip. For the 20k comparison, take several
checkpoints per arm and compare distributions, not endpoints.

### Late recurrent iterations now DECAY conditioning (new at 11k, absent at 10k)
`cond/const` peaks 0.0472 at iteration 2, holds ~0.046 through iteration 18, then falls to 0.0327 by
iteration 25 (-29%); text positions do the same (1.03 -> 0.57). At 10k the curve was flat to the end.
Also note ~96% of the conditioning arrives in iteration 0 — 23+ further iterations add ~5%. Whether
this is a transient or a trend matters for recurrent depth generally; re-measure at 20k and 40k
before drawing any conclusion for the refinement thesis.

### RECURRENT DEPTH: monotone but SATURATED BY 8 iterations
`scripts_local/iteration_sweep.py` on the lr1e-4 run's ckpt-46000, 20 UNSEEN prompts x 2 PAIRED
draws (same flow seed at every depth), w=3, KL early-exit DISABLED so the counts are exact:

| iterations | 1 | 2 | 4 | 8 | 16 | 32 | GT |
|---|---|---|---|---|---|---|---|
| CLIPScore | 0.3196 | 0.3281 | 0.3310 | 0.3329 | 0.3332 | **0.3334** | 0.3613 |
| delta | — | +0.0085 | +0.0029 | +0.0019 | +0.0003 | +0.0002 | |
| % of per-prompt GT | 88.9 | 91.1 | 92.0 | 92.5 | 92.6 | 92.7 | |

Each octave buys about half the previous one. **8 -> 32 iterations is worth +0.0005 for 4x the
trunk compute** — far below this eval's noise. The knee is at 8; the KL exit already converges
near 13 unprompted.
⭐ **This RECONCILES the compression probe rather than contradicting it.** That probe found
cond/const flat from iteration 0 — conditioning MAGNITUDE saturates immediately — yet the render
improves through 8. So iterations after the first change the conditioning's DIRECTION usefully
while its magnitude stays put; the probe measured the wrong quantity to see it.
⚠️ **Only 12/20 prompts improve from 1->32** (median +0.008, range -0.020 to +0.089). The smooth
mean averages over prompts that behave quite differently; a third are flat or slightly worse.
⚠️ A 3-prompt x 1-draw smoke test of the same script said 1 iteration BEAT 4 (0.337 vs 0.317).
The full 20-prompt run reverses it cleanly. This probe needs ~20 prompts before its ORDERING is
trustworthy, not just its magnitude.
⭐ **OPEN, and the largest efficiency lever on the board:** this measured INFERENCE depth on a
model TRAINED at mean_thinking_steps=32 (Huginn samples depth, so it is explicitly trained to be
depth-robust). It does NOT establish that TRAINING at 8 gives the same model — a shallow-trained
recurrent model is generally a shallow model, not a shallow-and-deep one. `backprop_depth` is
already 8, so gradients only flow through <=8 steps either way; what changes is the forward state
those gradients see. The trunk is the throughput bottleneck ("launch-bound, 32 iters small kernels
@ batch8"), so if training at 8 holds up it is a ~4x speedup on every future image run — worth
more than any of the +0.02 quality levers queued. One arm at mean_thinking_steps=8, matched steps.

### LR 1e-4 x3: 5x FASTER to the same plateau, not a higher one
`zimage_qwen_t3_xskip_lr1e-4_1e-4_1e-4_cosine_0` (lr/lr_dit/lr_flow all 1e-4) vs the original
from-scratch run (1e-5 / 7e-5 / 1e-4), same config otherwise. No loss spikes or NaN — the failure
[[project_peak_lr_default]] warns about for 1e-4 did not appear.

| milestone | prev run | lr1e-4 run |
|---|---|---|
| pass point head (0.266) | ~21k | **9k** |
| pass point+gain (0.283) | ~32k | **11k** |
| reach ~0.32 | ~100k (its final) | **20k** |
| @50k | 0.330 / 0.274 | 0.337 / 0.275 |
| @82k | — | 0.347 / 0.308 |

⭐ **The early advantage is enormous and then vanishes**: 2-5x faster to any given level up to ~30k,
but by 50k the two runs are within 0.007 — the higher LR bought SPEED TO a plateau, not a higher
plateau. Both settle near 0.33-0.35.
⭐ Its UNGUIDED column is the standout: 0.308 at 82k vs the staged lineage's 0.268 and the previous
run's 0.245. The guided/unguided gap narrowed from ~0.09 early to ~0.04, i.e. it depends on
guidance less — consistent with the "peak-w drifts left as conditioning strengthens" prediction,
here as a run-level effect. ~~**A guidance re-sweep on this run would likely find its optimum below
w=3, so the w=3 column understates it.**~~
⛔ **PREDICTION TESTED AND WRONG (2026-09-11).** The sweep was run on THIS run's own ckpt-97000:
w=3 sits on a flat plateau running to w=6 (0.346-0.350), and everything BELOW w=3 is worse
(w=2.5 0.344, w=2.0 0.335, w=1.0 0.308). The optimum did not move left, and the w=3 column does not
understate this run. A narrowing guided/unguided gap evidently does NOT imply the peak shifts —
only that the unguided floor rose. See the sweep entry near the top of this file.

**RUN COMPLETE 2026-09-01, 100k/100k, all 100 checkpoints evaluated** (8-prompt committed probe,
N=4, GT ceiling 0.368). Final: w=3 last-10-ckpt mean **0.3475**, best 0.351 @97k; w=1 last-10 mean
**0.3047**, best 0.311 @90k.

⚠️ **CORRECTION to the "plateau" reading above — at block resolution the run never stopped
improving.** Per-checkpoint noise (individual points bounce over a ~0.02 band) hid a slow monotone
climb that only appears when 10 checkpoints are averaged:

| steps | w=1.0 block mean | w=3.0 block mean |
|---|---|---|
| 21000-30000 | 0.2631 | 0.3261 |
| 31000-40000 | 0.2753 | 0.3305 |
| 41000-50000 | 0.2788 | 0.3359 |
| 51000-60000 | 0.2820 | 0.3345 |
| 61000-70000 | 0.2903 | 0.3376 |
| 71000-80000 | 0.2985 | 0.3424 |
| 81000-90000 | 0.3029 | 0.3453 |
| 91000-100000 | **0.3047** | **0.3475** |

Monotone in w=1 throughout, and in w=3 but for one -0.0014 step. 30k->100k is **+0.042 (w=1)** and
**+0.021 (w=3)** — an order of magnitude larger than the block-mean standard error (~0.002). The
increments decelerate (+0.0018 / +0.0022 in the final block) but had NOT reached zero when the
schedule ended: 100k steps was not enough to converge this arm.
⚠️ **Confounded with the cosine tail.** The LR anneals to ~0 at 100k, so the late climb cannot be
separated from an annealing bump. Distinguishing them needs a WSD or held-LR arm — the same
untested lever [[project_world_tts_ceiling_noise_floor]] flags for world-TTS.
⚠️ I called this run "flat" from single checkpoints several times while it was live. That was
wrong at block resolution and is the third premature trend call of this session; **read block
means, not consecutive checkpoints, on an eval whose per-point noise is 10x its per-10k drift.**

### Constant vs cosine LR: the late climb IS annealing, but it is VARIANCE, not level
`zimage_qwen_t3_xskip_lr1e-4_1e-4_1e-4_constant_0` — identical to the cosine run in every respect
but `--lr_scheduler_type constant_with_warmup`. Both ran to 100k; all 100 checkpoints evaluated on
both arms at w=1.0 and w=3.0, N=4, same probe. 200 evals, 0 failures. This is the held-LR arm the
cosine entry said was needed.

| steps | cosine LR | w=3.0 delta | w=1.0 delta |
|---|---|---|---|
| 1000-10000 | 100% | +0.0033 | -0.0032 |
| 11000-20000 | 96% | +0.0045 | +0.0080 |
| 21000-30000 | 87% | +0.0020 | +0.0010 |
| 31000-40000 | 75% | -0.0003 | -0.0032 |
| 41000-50000 | 60% | **-0.0220** | **-0.0261** |
| 51000-60000 | 44% | -0.0136 | -0.0228 |
| 61000-70000 | 28% | -0.0187 | -0.0262 |
| 71000-80000 | 15% | -0.0060 | -0.0140 |
| 81000-90000 | 6% | -0.0094 | -0.0155 |
| 91000-100000 | 1% | -0.0118 | -0.0193 |

(delta = constant - cosine at matched steps.) EARLY (<=40k): +0.0024 +- 0.0018 SE at w=3, 20/40 —
a coin flip, as it must be while the schedules are within 25 LR points. LATE (>40k): **-0.0136 +-
0.0019 (10/60)** at w=3 and **-0.0206 +- 0.0019 (5/60)** at w=1. The crossover is at ~40k, i.e.
where cosine falls below ~75% of peak.

⭐⭐⭐ **BUT THE GAP IS ENTIRELY VARIANCE. The two arms reach the SAME ceiling.**

| late window (>40k) | cosine | constant |
|---|---|---|
| mean w=3 | 0.3405 | 0.3270 |
| **sd** w=3 | **0.0071** | **0.0151 (2.12x)** |
| top-5 mean w=3 | 0.3502 | 0.3478 (**-0.0024**) |
| best single w=3 | 0.351 | **0.351 (tied)** |
| best single w=1 | 0.311 | **0.313 (constant HIGHER)** |

Constant is 1.7-2.1x noisier checkpoint-to-checkpoint and its mean is dragged down by the troughs,
but its PEAK is identical — tied at w=3 and marginally ahead at w=1. So annealing is buying
STABILITY, not capability.
⭐ **Practical consequence: if you select the best checkpoint, the schedule is nearly free
(~0.002).** If you deploy whatever the run ends on, cosine is clearly safer. The right reason to
anneal here is variance reduction, not a better optimum.

⚠️ **This QUALIFIES the cosine run's "it never plateaued" entry above.** Late climb from the
31k-40k block to the final-10 mean: cosine **+0.0170**, constant **+0.0055** (w=3); +0.0294 vs
+0.0133 (w=1). So ~2/3 of the cosine late climb is the schedule. Combined with the variance result,
the honest reading of that climb is **the mean converging onto a ceiling both arms can already
touch**, not continued learning — the max barely moves (cosine's best is 0.351, first reached at
97k but matched by constant's noisy peak). Do not cite the cosine tail as evidence that more steps
were buying more quality.
⚠️ One seed per arm. The 41k-50k block is the sharpest drop (-0.022/-0.026) and partially recovers
by 71k-80k, so some of that specific block is likely high-LR instability rather than the schedule
effect proper.

### From scratch, the aux point-head MSE is a CONVERGENCE ACCELERANT, not a quality term
`zimage_qwen_t3_xskip_lr1e-4_1e-4_1e-4_noaux_0` — the cosine baseline's CLI plus
`--image_flow_aux_mse_weight 0.0` (the flag added 2026-09-01, `9d3e9be`), so `flow_aux_mse_weight`
0.1 -> 0.0 and NOTHING else. Both ran 100k; all 100 checkpoints evaluated on both arms at
w=1.0/3.0, N=4. This is the from-scratch version of the question the warm-start `auxdetach` arm
could only ask weakly.

| steps | w=3.0 delta | w=1.0 delta |
|---|---|---|
| 1000-20000 | **-0.0472** | **-0.0645** |
| 21000-40000 | -0.0141 | -0.0266 |
| 41000-60000 | -0.0079 | -0.0040 |
| 61000-80000 | +0.0002 | -0.0009 |
| 81000-100000 | -0.0020 | -0.0015 |

(delta = noaux - baseline at matched steps.) **LATE (>60k): -0.0009 +- 0.0010 SE at w=3 and
-0.0012 +- 0.0012 at w=1 — both under 1 SE.** Final-10 means -0.0036 / -0.0026; best single
checkpoint 0.349 vs 0.351 (w=3) and 0.311 vs 0.311 (w=1, exactly tied). The user independently
called the 100k renders "indiscernible to the eye" before seeing these numbers.

⭐⭐ **So the 0.1-weight aux MSE buys EARLY CONVERGENCE and contributes nothing at the endpoint.**
Removing it costs ~0.05-0.06 through the first 20k and is paid back in full by ~60k.
⚠️ This CORRECTS the reading of the warm-start `auxdetach` arm above, which measured -0.0103
unguided and was recorded as "the aux term is mildly load-bearing". It is not load-bearing for
quality; that arm was 3000 steps on a trunk already shaped by 100k steps of aux gradient, which is
exactly where removing an accelerant looks like removing a contributor.
⭐⭐⭐ **THIRD instance of the same pattern in this direction**: `cross_dec` buys speed not endpoint;
LR 1e-4 x3 buys speed not endpoint; the aux MSE buys speed not endpoint. Every intervention that
has looked promising here has turned out to move the CONVERGENCE RATE while the ceiling stays put
— which is itself evidence the ceiling is set by something none of them touch (the trunk
gen-query compression). Treat "arm X is ahead at 20k" as uninformative about the endpoint by
default; this direction has now paid for that lesson three times.
⚠️ One seed per arm.

### LONG-CAPTION PAIR: native/AR loses by 36x LESS when captions are long
Two from-scratch arms, 40k steps, on a long-caption subset built from the 21 t2i shards
(`image_gen_captions_long`: 320k samples, median 93 SmolLM2 tokens, 81.4% over 64, matched
held-out val). Identical but for the conditioning layout:
`rope` = `small_sum_zimage_t4_ar_rope` (AR per-token flow head, NATIVE length, RoPE so there is no
length cap, whitened on long_native = 84.4 mean Qwen tokens) vs `k64` = `small_sum_zimage_t3_xskip`
(parallel head, every caption F.interpolate'd to 64, whitened on long_k64).
Final comparison: 3 late checkpoints x 2 seeds, paired, N=4.

| probe | w | rope | k64 | paired delta | rope higher |
|---|---|---|---|---|---|
| LONG (~48 words, ceiling 0.289) | 3.0 | 0.2603 | 0.2620 | **-0.0017 +- 0.0010** | 1/6 |
| LONG | 1.0 | 0.2342 | 0.2395 | -0.0053 +- 0.0021 | 1/6 |
| SHORT (~10 words, ceiling 0.368) | 3.0 | 0.2720 | 0.3330 | **-0.0610 +- 0.0021** | 0/2 |
| SHORT | 1.0 | 0.2060 | 0.2465 | -0.0405 +- 0.0018 | 0/2 |

⭐⭐⭐ **The gap is 36x smaller on long prompts than short ones at w=3 (0.0017 vs 0.0610).** As a
fraction of each probe's own GT ceiling: on long captions rope reaches 90.1% against k64's 90.7%;
on short ones 73.9% against 90.5%. First evidence in this direction that conditioning FORMAT
interacts with caption length at all.
⛔ **k64 still wins both.** -0.0017 is 1.7 SE with 1/6 paired comparisons favouring rope. "No
longer decisively behind" is the honest reading; "tied" is not.
⭐⭐ **The SHORT deficit is the more interesting half.** Both arms trained on the SAME long corpus,
so rope is not worse from seeing different data -- it generalises worse to short prompts.
Suspected cause, NOT tested: the AR unroll emits its full budget regardless, so a 10-word prompt
gets conditioning tokens the caption never justified. `image_cond_length` (built 2026-09-17, wired
through generate()) sets the count per prompt and would test this directly.
⚠️ **Confounded three ways.** rope changes the factorisation (AR vs parallel), the length handling
(native vs K=64) AND the coda (position-wise vs Q-Former) at once -- the same bundling that made
the original T4 verdict uninterpretable. It says the three TOGETHER are competitive on long
captions, not which one is responsible.
⚠️ The long probe's 8 prompts average ~48 words ~= 60-70 CLIP tokens, near CLIP's 77-token
truncation. A longer probe would be silently scored on a truncated prompt.

### Euler steps are NOT the variance lever: +0.006 for both arms, equally
`--flow_steps` sweep, `cosine_0/ckpt-97000` and `t4_ar/ckpt-100000`, N=8, seeds pinned, 20 cells.

| w=3.0 | steps 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| baseline (K=64) | 0.338 | 0.346 | 0.351 | 0.352 | 0.355 |
| t4 (AR) | 0.326 | 0.327 | 0.327 | 0.321 | 0.324 |

⛔ **Raising 8 -> 32 buys the baseline +0.006 and T4 +0.000-0.006 -- no differential benefit.** So
T4's -0.0235 deficit is NOT Euler discretisation error, which was the top item on the
variance-mitigation list. Kills that lever.
⭐ Below 8 costs real quality for the parallel head (4 -> 8 is +0.008 at w=3) but NOT for the AR one
(0.326 -> 0.327), whose per-token ODE is apparently easier to integrate.
⚠️ Cost: the flow head is a tiny share of eval wall-clock (Z-Image's 1024 decode dominates), so
steps are nearly free at this scale and 16 is a reasonable default. Do not budget by forward count.

### The broken exit criterion does NOT reach world-image
The world-voice session found `exit_criteria: "kl_divergence"` applies `F.kl_div` to post-norm
ACTIVATIONS -- a signed quantity, so `value < threshold` passes on any negative value -- and that
fixing it was worth LCS 0.7519 -> 0.8849 there. Checked here across 24 cells (2 checkpoints x
{kl_divergence, latent_diff, none} x w in {1,3} x 2 seeds):

| arm | w | kl_divergence | latent_diff | none |
|---|---|---|---|---|
| baseline | 3.0 | 0.3470 | 0.3445 | 0.3450 |
| baseline | 1.0 | 0.3050 | 0.3070 | 0.3045 |
| t4 | 3.0 | 0.3245 | 0.3220 | 0.3225 |
| t4 | 1.0 | 0.3035 | 0.3055 | 0.3015 |

⛔ **Null. Max spread 0.0025, inside the 0.004 noise floor, and seed spread is unchanged** (t4 w=1
stays 0.013-0.015 under all three) -- none of voice's 24x variance collapse appears.
⭐ **Structural reason, not luck:** the image eval calls `forward()`, which routes through
`_exit_eligibility()` -- a dict of text/voice masks only. Image appears in NO key, so image
positions were never eligible and the criterion could not reach them. The unmasked call site the
voice session flagged is in `generate()`, which only the Gradio path uses for image.
⭐⭐ **THEREFORE the variance conclusions survive**: T4's wider spread and the per-checkpoint
variance that forced the block-mean protocol are real, not criterion artifacts.
⚠️ `converge_eligible=None` does NOT mean exempt -- `recurrent.py` sets `eligible_any =
converge_eligible`, so None makes EVERY position eligible. Exempt is an all-FALSE mask. My first
"fix" to generate() had this inverted; corrected in 263a5ce, verified 32 iters/step off vs 9 on.
⭐ **Free inference win, untaken:** with `image_exit_eligible=True` + `latent_diff` 0.03 the trunk
runs 13.2 iterations instead of 32 with CLIPScore held (0.355 vs 0.359, inside noise) -- 2.4x less
trunk compute. Lands exactly at the knee the iteration sweep found. Opt-in, default unchanged.
⚠️ The voice session separately found `latent_diff` INDUCES collapse for voice (65-75/96,
`fa5d316`). Do not carry this recommendation across directions.

### FOUR-ARM RESULT at 100k: AR ties the baseline UNGUIDED and loses only on guidance response
All four from-scratch arms complete at 100k on the same recipe (1e-4 x3, cosine, seed 42), every
checkpoint evaluated at w=1.0/3.0, N=4. Final-10 means:

| arm | w=3.0 | w=1.0 | best (w=3) | w3-w1 GAIN |
|---|---|---|---|---|
| baseline `t3_xskip` (K=64) | **0.3475** | **0.3047** | 0.351 | 0.0428 |
| `noaux` (K=64, aux 0) | 0.3439 | 0.3021 | 0.349 | 0.0418 |
| `t4_ar` (AR, native) | 0.3209 | 0.2972 | 0.329 | **0.0237** |
| `nopos` (parallel, native) | 0.2972 | 0.2258 | 0.314 | 0.0714 |

Paired over all 100 matched checkpoints:

| | w=3.0 | w=1.0 |
|---|---|---|
| t4 - baseline | -0.0235 +- 0.0022 (**1**/100 t4 higher) | **-0.0068 +- 0.0024 (46/100)** |
| t4 - noaux | -0.0093 +- 0.0022 (22/100) | **+0.0127 +- 0.0019 (67/100)** |
| t4 - nopos | +0.0260 +- 0.0024 (92/100) | +0.0583 +- 0.0026 (95/100) |

⭐⭐⭐ **AT THE OPERATING POINT THE K=64 BASELINE WINS** (-0.0235, 99/100 checkpoints). That is the
headline and it should not be softened.
⭐⭐⭐ **BUT UNGUIDED, AR TIES IT** — 46/100 is a coin flip — **and BEATS the noaux arm (+0.0127,
67/100).** T4's entire deficit is guidance response: it extracts **0.0237** from w1->w3 where both
K=64 arms extract ~0.042, barely half. So AR-at-native-length is NOT worse at CONDITIONING; it is
worse at EXPLOITING CFG.
⭐ Mechanism (predicted in advance of this data, still not directly measured): guidance COMPOUNDS
along an autoregressive sample — each position is guided conditioned on tokens that were themselves
guided, so over-sharpening accumulates along the sequence instead of being applied once as it is
for a one-shot parallel head.
⚠️ **w=3 may therefore be an unfair operating point for T4**, and the converged baseline sweep
(flat w=3..6) explicitly does not transfer to it. A T4 sweep on ckpt-100000 over
w = 0.5 .. 4.5 is RUNNING 2026-09-13; until it lands, treat the -0.0235 as "T4 at the baseline's
operating point", not "T4 at its own".
⚠️ Single seed per arm. `nopos` is last on every axis and has the LARGEST guidance gain (0.0714),
consistent with it having the weakest conditioning of the four — the unordered-set handicap.

### AR (T4) at native length TRACKS the K=64 baseline; the collapse was `flow_pos_embed`
Three native-length arms now exist on the SAME recipe (from scratch, 1e-4 x3, cosine, 100k, native
whiten stats), differing only in how the output slots are ordered. IN FLIGHT: t4 at 27k, nopos at
53k. Matched-step block means, w=3.0:

| steps | t4 (AR) | nopos (parallel, no pos) | native+pos | K=64 baseline |
|---|---|---|---|---|
| 11000-15000 | **0.2508** | 0.2056 | 0.0182 | 0.2932 |
| 21000-25000 | 0.3108 | 0.2598 | 0.0200 | 0.3244 |
| 26000-30000 | **0.3180** | 0.2694 | 0.0184 | 0.3278 |
| 51000-55000 | — | 0.2925 | 0.0166 | 0.3344 |

⭐⭐⭐ **THE COLLAPSE WAS `flow_pos_embed`, NOT NATIVE LENGTH.** Same config minus the positional
table goes from 0.02 (noise) to 0.29-0.30. The positional-shortcut hypothesis is confirmed at the
level of WHICH KNOB; the mechanism (chat-template tokens make position j near-constant across
captions, so `pos[j]` alone satisfies the objective and the context is ignored) remains inferred,
not directly measured.
⭐⭐⭐ **THE T4 VERDICT IS OVERTURNED.** From scratch at the standard LR, T4 passes the old run's
"plateau" of 0.248 by ~13k steps and reaches 0.3180 by 26-30k — within 0.010 of the K=64 baseline
at w=3, and AHEAD of it unguided at 21-25k (0.2548 vs 0.2542). The old 0.248 was an artifact of a
20k warm start at 5e-6 with a randomly-initialised AR head and mismatched whiten stats. AR
factorisation does NOT lose.
⭐⭐ **AR BEATS THE PARALLEL SET AT NATIVE LENGTH** — 0.3180 vs 0.2694 at matched steps, +0.049.
Mechanism (consistent, not proven): without `pos` the parallel head is permutation-equivariant and
emits an unordered SET, but the Qwen3 target is an ordered causal sequence. AR gets ordering
intrinsically and pays no positional-shortcut tax. So the three arms separate exactly along "how is
order supplied": shortcut -> collapse; no order -> handicapped; AR -> tracks baseline.
⚠️ **CORRECTION to a prediction I recorded:** T4's sequential decode (L x flow_steps ~ 160 forwards
vs 8) was expected to make its evals much slower. Measured medians: **t4 456s vs nopos 444s, ~3%**.
Z-Image's own 1024x1024 decode dominates the eval, so the flow head's step count is nearly free at
this scale. Do not budget eval time by flow-head forwards.
⚠️ Both arms IN FLIGHT and single-seed; T4 has only 27 checkpoints. Given this direction has three
times seen early leads vanish by 60k, do NOT call the endpoint yet.

### ⚠️ The T4 "AR factorisation loses" verdict rests on an UNDER-TRAINED arm — reopened
The recorded claim (in `world_model.py`, and the stated reason T5 exists) was that T4 "lost
decisively (w=3 plateau ~0.248 vs T3's 0.344), which indicts the factorisation, not the length
handling." Re-examined 2026-09-09 from `zimage_qwen_t4_0`'s own launch command and loss curve,
**that arm cannot support the claim.** Three independent defects, any one of which would be
disqualifying:

| | T4 run (`zimage_qwen_t4_0`) | every from-scratch arm we trust |
|---|---|---|
| init | **warm start** off `t3_2/ckpt-20000`, which contains NO `ar_flow_head` — the AR head began RANDOM | from scratch |
| LR | **5e-6** | 1e-4 / 1e-4 / 1e-4 |
| steps | **20000** (`--fresh_schedule`) | 100000 |
| whiten stats | **`qwen_whiten_stats_k64.pt`** — but `ar_flow_head` takes the `encode_native` path, so native-length targets were whitened by RESAMPLED stats | matched to the target layout |

Its eval loss fell monotonically to the end — 1.355 @2k -> 1.196 @20k, still decreasing on the
final interval — so it had **not converged**; 0.248 is where it got to, not a plateau.
⭐⭐ **Why this matters beyond one arm:** this direction has now measured THREE interventions that
looked clearly worse early and were level by ~60k (`cross_dec`, LR 1e-4x3, the aux MSE — see the
accelerant finding). A 20k warm start at 1/20th LR is precisely the regime that cannot separate
"wrong factorisation" from "barely trained". **The AR factorisation is OPEN, not settled.**
⚠️ T5 was justified in-comment as "the part of T4's premise that survives its result". T5 is still
worth running on its own merits, but it is not downstream of a settled T4 result, and the native
+pos collapse below is a separate failure that says nothing about AR either.
⚠️ Not re-derived: the 0.248 figure itself. The reader timed out walking that run's 74 event files,
so the trajectory argument rests on the loss curve and the launch command, both verified.
The honest test is T4 on the standard recipe — from scratch, 1e-4 x3, cosine, 100k, native whiten
stats. Not yet run.

### Native length + slot positions COLLAPSES to prompt-independent generation
`zimage_qwen_t3_xskip_lr1e-4_1e-4_1e-4_native_0`, `small_sum_zimage_t5_native_pos_xskip`, correct
`whiten_stats_native.pt` (a first attempt with k64 stats was discarded). Stopped by the user at
~96k. **This is NOT a native-length quality result — the arm is BROKEN, and must not be cited as
one.**

| steps | native w=3.0 | noaux w=3.0 |
|---|---|---|
| 1000-5000 | 0.1066 | 0.1170 |
| 11000-15000 | 0.0182 | 0.2252 |
| 56000-60000 | 0.0194 | 0.3340 |

Native tracks the control for ~5k, then DECAYS to ~0.02 and flatlines there for 90k steps. Raw
cosine ~0.02 is what unrelated images score; several per-prompt means are NEGATIVE.
⭐ **Training is healthy the whole time** — eval loss falls smoothly 0.758 -> 0.710 (control:
0.694 -> 0.666). So this is an inference/conditioning failure, not divergence.
⭐⭐ **The montage is unambiguous** (`eval_output/native/ckpt-93000_w3.0/`, 8 prompts x [target,
sample0-3]): the target column matches every prompt, and the four sample columns are COHERENT,
photographic, and completely unrelated to the prompt — with **the same four subjects repeating
down every row** (curly-haired portrait / man handling bread / uniformed group in a field / aerial
forest canopy). Sample INDEX determines content; the prompt does not. The head emits valid
Qwen-space embeddings of generic dataset-like captions, i.e. it has stopped reading its context and
samples its unconditional prior. Consistent with w=1 and w=3 being equally dead: CFG amplifies
`v_cond - v_uncond`, which is ~0.
❓ **HYPOTHESIS, NOT MEASURED — a positional shortcut.** Every caption goes through the same Qwen3
chat template, so at native length position j is nearly the same token across captions and position
alone predicts much of the target; `F.interpolate` at K=64 smears captions of different lengths
across the same slots, weakening position->content and forcing reliance on context. If true, the
failure belongs to native+pos TOGETHER, not to native length.
Two cheap tests, neither run: `small_sum_zimage_t5_native_xskip` (native WITHOUT pos, preset
exists, set up precisely as the independent ablation) and `scripts_local/text_dependency_probe.py`
on ckpt-94000 to confirm prompt-independence numerically rather than by reading a montage.
⚠️ The 2026-08-20 warm-start native result (-0.045, blob artifacts) is a DIFFERENT failure signature
(off-manifold texture vs coherent-but-wrong). Do not merge them.
⚠️ ckpt-96000 is truncated (run stopped mid-write); its eval fails with PytorchStreamReader.

### The three warm-start finetune arms are ALL NULL (control included)
Warm restarts from `t3_xskip_0/ckpt-100000`, 3000 steps, identical CLI, one config field apart.
Late-window means (1500-3000, n=4):

| arm | w=3.0 | vs control | w=1.0 | vs control |
|---|---|---|---|---|
| `control` (nothing changed) | 0.3255 | — | 0.2505 | — |
| `offset` (lever D) | 0.3245 | -0.0010 | 0.2510 | +0.0005 |
| `auxdetach` | 0.3235 | -0.0020 | 0.2402 | **-0.0103** |

⛔ **LEVER D DOES NOTHING.** The learned per-position offset — built on a measured mechanism
(conditioning at ~10% of a norm-83 constant; a LayerNorm provably cannot remove a per-position
vector; measured at ckpt-85000 the existing norm takes cond/const only 0.0991 -> 0.1136) — lands
on top of the control at both guidance settings. A correct mechanism story is not a prediction.
⚠️ `auxdetach` is -0.0103 unguided, the only reading outside noise and NEGATIVE: detaching the
0.1-weight aux MSE COSTS a little unguided quality, so that term is mildly load-bearing as a
training signal, contradicting the "purely diagnostic" reading of its own code comment. One seed.
⭐ **THE CONTROL EARNED ITS SLOT.** Without it, `offset` at 0.3245 against the source's 0.322 reads
as a small win; against the control it is -0.001. Never run a warm-start arm without one — the
restart alone is worth ~+0.009.

### Was the MSE baseline just under-trained? NO -- and the decoupling test PASSED
`whiten_0` was stopped at 20k while the flow runs went to 100k, and the flow runs reach R^2 0.787
against its 0.725 — backwards, since R^2 is what the point head directly optimises and the flow
head carries only as a 0.1-weight auxiliary. So "flow beats MSE" rested on a 20k baseline vs 100k
arms. `zimage_qwen_whiten_cont_0` continues it (warm restart; a true optimiser resume was
impossible, see the gotcha below).
**First 12 checkpoints (steps 21k-32k): 0.262, 0.279, 0.272, 0.262, 0.269, 0.264, 0.266, 0.273,
0.272, 0.283, 0.263, 0.267 — mean 0.269 vs `whiten_0`'s 0.266. FLAT.** Two apparent highs (0.279
@22k, 0.283 @30k) both fell back the next checkpoint.
⭐⭐⭐ **RESOLVED 2026-09-01 — the sharper test this entry asked for has now RUN, and it separates
cleanly. R^2 climbed while the render did not move.**

| | steps 21k-38k (18 evaluated ckpts) |
|---|---|
| eval MSE (whitened) | **0.2734 -> 0.2438** (min 0.2399 @50k) |
| => R^2 = 1 - MSE | **0.727 -> 0.760** (min-MSE ckpt: 0.760) |
| CLIPScore, block means of 6 | 0.2680 -> 0.2707 -> **0.2715** |
| CLIPScore drift vs block-mean SE | **+0.0035 vs SE 0.0036 = 1.0 SE** |

The point head spent 30k steps closing most of the R^2 gap it was launched to close (0.727 -> 0.760,
against the flow runs' 0.787) and bought **exactly nothing** in render: 1 SE of drift, and the run
sits at 0.270 vs `whiten_0`'s 0.266. The original stopping decision at 20k was correct.
⭐ ~~This is the cleanest demonstration in the project of the loss/render decoupling~~ — see the
2026-09-02 amendment below: the decoupling held over 21k-38k and then stopped holding. Do NOT cite
this run as the clean decoupling case. It generalises the [[project_dispersion_over_smoothing]] result from "the gain axis" to
"the training axis": R^2 is not a proxy for render quality for a POINT head, in either direction.
⛔ **Therefore "flow beats MSE" does NOT rest on a 20k-vs-100k artefact.** The worry that opened
this entry is retired: the baseline is trained out, not under-trained.
⚠️ Do not read this as "R^2 never matters" — it is a statement about a point head near its own
ceiling, one seed, 8-prompt probe. What it rules out is the under-training explanation.

⚠️⚠️ **AMENDED 2026-09-02 — the "render does not move" half is WINDOW-SPECIFIC and does not hold
past ~40k.** Continuing to stride-5000 eval over steps 40k-75k:

| window | eval MSE (=> R^2) | CLIPScore mean |
|---|---|---|
| 21k-38k (n=18, stride 1000) | 0.2734 -> 0.2438 (0.727 -> 0.756) | 0.2701 (sd 0.0088) |
| 40k-75k (n=8, stride 5000) | 0.2438 -> 0.2336, min 0.2298 @66k (0.756 -> **0.770**) | **0.2812** (sd 0.0107) |

+0.0112 render, combined SE 0.0043 => ~2.6 SE. The newer window's WORST point (0.265) is about the
older window's MEAN. So render did eventually move, and in window B it moved roughly in step with
R^2 (+0.010 R^2, +0.011 render) rather than decoupling from it as in window A (+0.029 R^2, +0.0035
render). **The clean decoupling above was a property of steps 21k-38k, not of the point head.**
⚠️ 2.6 SE is an OVERSTATEMENT: consecutive checkpoints on one trajectory are autocorrelated, not
independent draws, so the true SE is larger. Read this as "a real level shift, magnitude uncertain",
not as a p-value. This is the same error class the run-level entries above were corrected for.
⛔ What SURVIVES: the baseline is not under-trained (that question is still answered), and the point
head remains far below the flow arms — 0.281 vs the cosine run's 0.347 guided / 0.305 unguided. The
"flow beats MSE" conclusion is untouched. What does NOT survive is using this run as the clean
demonstration that R^2 and render are decoupled; it shows both behaviours in different windows.
Per-checkpoint eval was dropped to stride 5000 on 2026-09-01.

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
### It is DILUTION, not routing — measured 2026-08-22
`scripts_local/genquery_text_ablation.py`, ckpt-14000, 8 prompts. Attention weights are not
recoverable (SDPA never materialises them; rebuilding RoPE-applied q/k by hand is error-prone), so
this measures the CAUSAL contribution: zero the text prelude's output and see how far the
gen-query states move.

| measurement | mean ||dh||/||h|| |
|---|---|
| gen queries, TEXT ABLATED | **0.201** |
| gen queries, DIFFERENT PROMPT | 0.112 |
| text positions, text ablated (sanity) | 1.303 |

Text reaches the gen queries and accounts for ~20% of their norm, and 0.112/0.201 means **~56% of
that contribution is PROMPT-SPECIFIC** rather than a generic "a prompt exists" signal. So neither
the routing nor the mapping is broken — the conditioning arrives, at low SNR against the norm-83
constant.

⭐ **This CUTS lever C (direct text injection into the gen queries).** It targets routing, and
routing is not what is wrong; it would add a second path for information that already arrives.
The live levers are the ones aimed at the constant-to-signal ratio: **A (lower
`image_gen_query_init_std`)** and **D (normalise at the trunk output)**.
⭐ It also sharpens the Q-Former question: if `cross_dec` is acting as a learned constant-remover
(cross-attention with its own learned queries can put whatever it likes in the constant component),
it is doing D implicitly — which would explain the two arms tracking so closely, and predicts
`trunkctx` takes longer to learn the same subtraction.

⚠️ **Corrects the earlier phrasing "conditioning the trunk never imported".** It imports ~20%, over
half prompt-specific, and the aux point head extracts R^2 ~= 0.5 from it (`image_clip_mse_loss`
0.545 against 1.0 for predicting the mean). A 4% MAGNITUDE ratio is not 4% of the information —
a learned head can amplify a small direction. The accurate claim is low SNR plus a bf16 precision
tax, and that a SCALAR at the output cannot fix it (the gamma sweep).

---

## PRACTICAL GOTCHAS (world-image session)

### Old checkpoints can no longer be optimiser-resumed
Resuming `whiten_0/ckpt-20000` with the current code fails:
`ValueError: loaded state dict contains a parameter group that doesn't match the size of
optimizer's group`. The saved optimiser has **4** groups (23/44/56/95 tensors — a decay x DiT
split); `create_optimizer` now builds **2** (71/203). The model also gained parameters since
(`contrastive_proj`, +4 in image_generator, +66 elsewhere). Model weights still load fine
(`load_model` defaults to `strict=False`), so a WARM RESTART works — but the optimiser moments
and the schedule position are gone, which reintroduces the ~+0.009 restart confound.
⚠️ **And `--fresh_schedule` restarts HF's `global_step` at 0, so checkpoints are written as
`checkpoint-1000`, `checkpoint-2000`… — straight OVER the original run's checkpoints if you reuse
its run_name.** `whiten_0` is the baseline every matched-step comparison rests on; that would have
destroyed it. ALWAYS give a fresh_schedule continuation a NEW run name. `whiten_cont_0`'s
checkpoint N is therefore actual step N+20000.

### Dataset scale
**2,639,029 train captions** (165 shards), 25,010 held-out COCO val. At 100k steps x batch 8 x
grad-accum 8 = 6.4M samples, a full run sees **2.43 epochs**. For reference ELLA — the closest
published analogue, mapping LLM features to a frozen diffusion model's conditioning — used ~30M
image-text pairs, ~10x this. Caption QUALITY is already flagged as a limiter in
[[project_image_caption_quality]].

### The flow head is emphatically NOT a disguised deterministic map
Two independent measurements. Cosine distance between two sampler draws of the SAME prompt
(`trunk_compression_probe`, ckpt-11000): **d_seed 0.881 at w=1**, 0.767 at w=3, 0.728 at w=4.5 —
against d_other (different prompts, same seed) of 0.131 / 0.393 / 0.471. Unguided, the SEED moves
the conditioning 7x more than changing the prompt entirely. And the shrinkage probe on the flow
output reports alpha 0.061 / R^2 -0.659, which is what a point-estimate metric says about a
genuine sample. Meanwhile the CLIPScore across those same seeds varies by only ~+-0.03 (the
`within-prompt sd` on every eval line IS the across-seed spread). **Different samples, similarly
good** — a working sampler.

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

### The validation split does not match the training distribution (full corpus)
`image_gen_captions_only/val` is 25,010 samples of **mean 11.9 SmolLM2 tokens, max 53, ZERO
captions >= 64**, i.e. entirely the short COCO source. Training data across all 165 shards is mean
**31.9**, median 23, p95 101, max 256, with 11.8% over 64 tokens. The cache is an unshuffled
concatenation of caption sources (shards 0-39 ~11.9 tokens, 36-41 and 150-164 ~90-105, the rest
mixed); the val split was cut from the short end only.
⚠️ Every `eval/loss` curve in this file was therefore measured OUT OF DISTRIBUTION. That includes
the train-vs-eval gap used to argue the corpus is large enough for ~215M params (constant -0.17,
never widening) and the whiten point head's converged eval loss. Those statements hold for THAT val
set; they are not evidence about the training distribution.
⚠️ It does NOT affect any arm-vs-arm comparison: every arm used the same val split, and CLIPScore —
which all the headline results rest on — is computed on the 8-prompt probe, not on val.
✅ `image_gen_captions_only/val_mixed` (24,915 samples sampled evenly from all 165 shards: mean
31.8, median 23, p95 102, max 249, 11.2% >= 64) now matches train. The original val is left in
place so the finished runs' logged curves stay interpretable.
✅ The long-caption subset ships its own matched val (one held-out long shard, 15,029 samples,
median 100, 98.5% >= 64).
