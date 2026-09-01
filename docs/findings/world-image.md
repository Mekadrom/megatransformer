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
here as a run-level effect. **A guidance re-sweep on this run would likely find its optimum below
w=3, so the w=3 column understates it.**

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

### Was the MSE baseline just under-trained? (whiten continuation, IN FLIGHT)
`whiten_0` was stopped at 20k while the flow runs went to 100k, and the flow runs reach R^2 0.787
against its 0.725 — backwards, since R^2 is what the point head directly optimises and the flow
head carries only as a 0.1-weight auxiliary. So "flow beats MSE" rested on a 20k baseline vs 100k
arms. `zimage_qwen_whiten_cont_0` continues it (warm restart; a true optimiser resume was
impossible, see the gotcha below).
**First 12 checkpoints (steps 21k-32k): 0.262, 0.279, 0.272, 0.262, 0.269, 0.264, 0.266, 0.273,
0.272, 0.283, 0.263, 0.267 — mean 0.269 vs `whiten_0`'s 0.266. FLAT.** Two apparent highs (0.279
@22k, 0.283 @30k) both fell back the next checkpoint.
So far this SUPPORTS the original stopping decision ("plateaued 16k-20k, loss dropping steeply but
renders frozen") and is another instance of the loss/render anti-correlation: more training buys
feature accuracy the decoder does not use. ⚠️ Only 12k of an 80k run; the flow runs took ~20k to
show their trajectory. The sharper test is whether R^2 climbs to ~0.787 while render stays at
0.266 — that would demonstrate the decoupling directly.

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
