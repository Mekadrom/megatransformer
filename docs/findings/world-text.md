# world-text (text -> text)

Language modelling / continuation from the recurrent trunk.

Stack today: text -> frozen SmolLM2-135M prelude -> recurrent trunk (trainable) -> text coda
-> next-token logits. No world-text run has ever been launched; every entry below is either a
measurement of the code/data as it stands or arithmetic from measured quantities. Nothing here
is a training result yet.

---

## ESTABLISHED

### The text corpus is 5.25B tokens, Mistral-tokenized (2026-08-21)
`cached_datasets/text_train_merged` = 51 full shards of 100,000 samples + one partial shard of
26,516, all at seq 1024:

| | samples | tokens | disk |
|---|---|---|---|
| train | 5,126,516 | **5,249,552,384** | 42.04 GB |
| val | 54,429 | 55,735,296 | 0.45 GB |

Packing mode, so there is no padding waste — every stored token is real. Stored as **int64**;
max id observed 31999, so the tokenizer is `mistralai/Mistral-7B-v0.1` (the preprocessing
commands in `logs/huginn_dataset_mixture.md` pass no `--tokenizer_name` and take that default).
No `config.json` and no `shard_index.json` in the directory, and no raw text is preserved in
pack mode. ~~So the corpus cannot be re-tokenized in place — it has to be rebuilt from
source.~~ **Superseded 2026-09-17** — see "the corpus converts to any tokenizer in an hour"
below. The inference was that no stored text means no re-tokenization; decoding recovers it
exactly, which was not tested at the time.

At uint16 the same 5.25B tokens would be 10.5 GB rather than 42 GB; vocab 49161 fits with room
to spare. Untaken 4x disk saving, noted for whenever the corpus is rebuilt.

### Text-only param counts for the three coda configurations (2026-08-21)
Built with `--include_modes text` (confirmed text-only: totals equal the sum of the three text
modules, so no image/voice/audio modules are instantiated):

| config | total | trainable | text FE | trunk | text coda |
|---|---|---|---|---|---|
| from-scratch (Mistral 32k) | 195.6M | 195.6M | 29.3M | 127.5M | 38.8M |
| frozen SmolLM2, shared head | 266.9M | 132.4M | 137.3M (2.8M tr.) | 127.5M | 30.4M (2.1M tr.) |
| frozen SmolLM2, trainable head | 305.0M | 170.5M | 137.3M (2.8M tr.) | 127.5M | 40.2M (all tr.) |

In shared-head mode the entire text readout is **2.1M trainable params** — an MLP translating
768 -> 576 into a frozen tied head.

### The trunk is SMALLER than SmolLM2 in parameters (2026-08-21)
Trunk 127.5M vs SmolLM2-135M's frozen stack 134.5M (106M body + 28.3M tied embed/head). The
model is larger only in **effective depth**: 6 blocks x `mean_thinking_steps=32` = 192 layer
applications at d=768, against SmolLM2's 30 layers at d=576. Any argument of the form "our
model is bigger than the LLM it reads out through, so the frozen head must be limiting it" is
unsupported — the parameter comparison goes the other way.

### `--text_encoder_trainable_head` is byte-identical when off (2026-08-21, commit 9dbe097)
Gives the coda its own readout at trunk width (`Linear(768, 49161)` = native vocab + 9 control
tokens as native rows) instead of the LLM's tied head. Verified by reconstructing the previous
coda from `git show HEAD:` and comparing seeded `state_dict`s: **7/7 keys bit-identical** with
the gate off. Module creation order was preserved deliberately so RNG draws land identically.
122/122 tests pass. Costs +38.1M params (266.9M/132.4M -> 305.0M/170.5M).

Protocol note: the equality test is construction-time only. It shows the default path is
unchanged; it is not a training-equivalence claim.

### Per-token training cost is ~40x a dense 135M model (2026-08-21, derived)
Effective compute-relevant params per token, counting `backprop_depth=8` truncated backprop
(forward over all 32 iterations, backward over 8):

| | non-recurrent | recurrent | effective params/token |
|---|---|---|---|
| this model | 149M | 127.5M x 32 | **4.23B** |
| Huginn-0125 | 1.5B | 1.5B x 32 | 49.5B |
| SmolLM2-135M | 106M | — | 106M |

=> **1.31e10 FLOPs/token** to train this model. The recurrence that buys depth at fixed
parameter count is paid for with ~40x the per-token compute of a dense model of the same
parameter count. Over-training — the thing that makes SmolLM2-135M punch above its weight at
14,815 tok/param, 741x Chinchilla — is therefore ~40x more expensive here than for the model
it is being compared against. Derived from the measured param counts above, not measured
end-to-end; the same estimator reproduces Huginn's reported total within 1.4x.

### The corpus is at Chinchilla, not below it (2026-08-21, derived)
5.25B tokens / ~300M params = **17.5 tok/param** against Chinchilla's 20. This model is not
data-starved for its parameter count. Any capability gap to SmolLM2 is a consequence of
SmolLM2 being deliberately over-trained, not of this corpus being small.
See `project_recurrent_chinchilla` — recurrent paths are expected to saturate data *earlier*
than 20 tok/param, which if true means the corpus is already at or past saturation.

### Reference scale: Huginn-0125 (2026-08-21, external)
795B tokens, 47,000 steps, global batch 16M tokens, seq 4096, on **4096 AMD MI250X GCDs**
(= 2048 cards = 512 nodes = 5.44% of Frontier) over 21 segments of up to 12 hours, ~201 hours
of compute, ~0.82M GCD-hours, ~1.7e23 FLOPs. Compute was an award, not a purchase: "An award
for computer time was provided by the U.S. Department of Energy's INCITE Program."
[arXiv 2502.05171](https://arxiv.org/abs/2502.05171)

Against 4x RTX 4090 (~140 TFLOP/s sustained bf16), reaching Huginn's 800B-token budget with
*this* model is ~3.3 years of the whole box; SmolLM2's 2T budget is ~8.3 years. The existing
5.25B-token corpus is ~8 days. Reproducing published pretraining scale is not reachable here;
the corpus on disk is roughly one epoch's worth of what this box can do in a working week.

### Logit-KL distillation forces the tokenizer, which forces the prelude (2026-08-21)
Verified vocab sizes: Qwen3-0.6B / 1.7B / 4B all report **151936**; SmolLM2-135M reports
**49152**. KL between student and teacher distributions requires a shared vocabulary, so the
teacher choice fixes the tokenizer, and because prelude and head consume the *same* token
sequence in a causal LM, it fixes the prelude family too. "Qwen3 teacher + SmolLM2 prelude" is
not an available configuration. Consequences:
- The corpus must be preprocessed with the *teacher's* tokenizer, so the teacher has to be
  chosen **before** the re-preprocessing pass, not after.
- All Qwen3 sizes share one vocab, so preprocessing once for Qwen3 leaves teacher size AND
  prelude size free to ablate later at no additional data cost.
- If the teacher is not SmolLM2, the shared-head mode is unusable by construction and
  `--text_encoder_trainable_head` stops being an ablation arm and becomes mandatory.

### Cost of a Qwen3 readout at trunk width (2026-08-21, derived)
`Linear(768, 151945)` = **116.7M params**, 3.1x the SmolLM2 head's 37.8M and ~92% the size of
the entire recurrent trunk. Model total would be ~384M, of which ~30% is a readout matrix.
The full logits tensor at batch 8 x seq 1024 is **4.98 GB in fp32** before gradients — a fused
or chunked linear-cross-entropy is a hard requirement on a 24GB card, not an optimization.

### Z-Image's Qwen3-4B is stock EXCEPT for a pruned final-layer MLP (2026-08-23)
`Tongyi-MAI/Z-Image-Turbo/text_encoder` vs `Qwen/Qwen3-4B`, tensor-by-tensor over all 398
tensors / 4.022B params: **396 are bit-identical**. Two differ, both in the last layer (35 of
0..35), and the modification is structured zeroing, not a finetune:

| tensor | shape | change |
|---|---|---|
| `layers.35.mlp.up_proj.weight` | 9728 x 2560 | **5770 / 9728 rows entirely zeroed** (59.3%) |
| `layers.35.mlp.down_proj.weight` | 2560 x 9728 | **404 / 2560 rows entirely zeroed** (15.8%) |

Every differing element is exactly 0 in the Z-Image copy; `layers.35.mlp.gate_proj` is
untouched, which is consistent — zeroing a neuron's `up_proj` row forces
`silu(gate) * up = 0` regardless of gate. Norm ratio matches cosine to 3 decimals
(86.502/129.345 = 0.6688 vs cos 0.6690), the signature of a projection. Config is stock
Qwen3-4B (hybrid-instruct, `eos 151645` = `<|im_end|>`, `tie_word_embeddings: true`, no
separate `lm_head` tensor).

Functional impact, measured on one caption, fp32, CPU:

| readout | Z-Image copy vs stock |
|---|---|
| `hidden_states[-2]` (what world-image regresses) | **bit-identical**, max abs diff 0.000e+00 |
| `hidden_states[-1]` | cos 0.798, max abs diff 36.1 |
| next-token logits | KL(stock‖zimage) **0.111 nats**, top-10 overlap 8/10, logit cos 0.195 |

=> The pruning sits entirely **outside** the path world-image uses: `hidden_states[-2]` is the
output of layer 34, which layer 35's MLP cannot affect. That conditioning target is exactly
stock Qwen3-4B and no world-image result is contaminated.

=> **But this checkpoint is a degraded language model.** A text teacher needs logits, which run
through layer 35. Distilling from the Z-Image copy would silently distil from a lobotomised
Qwen3-4B. **Use stock `Qwen/Qwen3-4B` for any text-teacher role** — it is a separate 7.6 GB
download and is already in `HF_HOME`; reusing the Z-Image copy because it is "already on disk"
is the trap.

### Logit distillation is basis-free; hidden-state regression is basis-locked (2026-08-23)
A KL between student and teacher distributions is over **token ids**, so it is invariant to the
teacher's internal basis. A regression onto `hidden_states[-2]` is a fit to **one checkpoint's
coordinate system**. Consequences for this project, which does both:

- **world-text (logits):** teacher size is a free choice and a free ablation. Qwen3-0.6B /
  1.7B / 4B share vocab 151936, so all three can be tried on one corpus, and teachers could
  even be mixed or ensembled.
- **world-image (hidden states):** the teacher is not a choice at all — Z-Image was trained
  against Qwen3-4B conditioning, so the target is dictated by the decoder. Swapping to another
  Qwen3, *even one with the same `d`*, invalidates every learned adapter weight.

Equal `hidden_size` across the lineage (14B and 32B both d=5120; 1.7B and 30B-A3B both d=2048;
8B and 235B-A22B both d=4096) does **not** imply a shared or compatible embedding space. Those
pairs differ in depth and head count (14B is 40L/40H, 32B is 64L/64H), and the
[Qwen3 report](https://arxiv.org/abs/2505.09388)'s strong-to-weak distillation is a
**post-training** procedure — off-policy response distillation, then on-policy logit-KL against
Qwen3-32B / Qwen3-235B-A22B — which constrains output distributions, not representation bases.
NOT independently verified here: only Qwen3-4B and 0.6B weights are cached locally, so the
equal-`d` embedding matrices were never compared numerically. Treat "unrelated bases" as the
architecturally-motivated default, not a measurement.

### ⚠️ The recurrent exit criterion reads the THOUGHT STATE, and it is not a KL (2026-09-15)
`recurrent_criteria.KLDivergenceCriteria` is fed `last_thought_state` / `new_thought` — raw
`(B, T, d_model)` trunk hidden states straight out of `_run_iteration`. It never sees the coda
or the LM head, so the answer to "does it exit on the output distribution?" is **no**: it exits
on the trunk's internal state between iterations. That alone is a design choice (Huginn's
adaptive-compute criterion uses successive *output distributions*). The bug is what it computes:

`F.kl_div(a, b, log_target=True)` evaluates `exp(b) * (b - a)`, which is a KL only when both
arguments are log-probabilities. Thought states are post-norm activations, so the quantity is
`sum_d exp(h_new_d) * (h_new_d - h_old_d)` — **signed**, and the test is `value < 1e-4`, so
every negative value passes as "converged".

Measured (`small_sum`, text-only, fresh init, batch 2 x 24, `mean_thinking_steps=32`):

| | result |
|---|---|
| iterations run, **train** mode | 43 (stochastic n+k sampling — criterion inactive) |
| iterations run, **eval** mode | **17** |
| reported "kl" trace, first 8 iters | 1254.1, 0.249, 0.015, **-0.083, -0.128, -0.148, -0.158, -0.162** |
| `converged_mask` on a 1%-perturbed state | **0.500 of tokens** declared converged |
| `converged_mask` on identical states | 1.000 (correct) |

The trace going negative at iteration 4 is the tell — a real KL is non-negative. Because
`converged` is sticky (`converged = converged | newly_converged`) and ~half of the remaining
tokens pass on each iteration by sign accident, essentially every token freezes within ~5-10
iterations and `forward()` returns early.

**Scope.** Gated on `not self.training`, so **training is unaffected** — no trained weights are
compromised. But it is live in **every eval, generation, and diagnostic**, for every modality:
all seven `self.recurrent_block(...)` call sites in `world_model.py` use `forward()`, and the
default config is `exit_criteria="kl_divergence"`, `exit_criteria_threshold=1e-4`. The
`generate_step()` path (line 466) has a second copy of the flaw — `should_exit` uses `.any()`,
exiting the whole batch when a *single* token trips — but no caller reaches it.

Consequence: eval has been running the trunk at roughly half its trained depth, with each token
frozen at an iteration **uncorrelated with how much refinement that token needed**. This is a
train/eval mismatch in the one component the whole architecture is built around, and it is a
candidate confound for any eval-time result in ANY direction.

**Stopping-time distribution, measured 2026-09-16** (fresh `small_sum` block, 12 trials x 4 x 32
positions, cap 32). "Random iteration" (my first phrasing) and "roughly the same per token" (the
world-voice reading) are both wrong; it is **bimodal**:

| | |
|---|---|
| mean / std / CV | 8.07 / 6.43 / **0.797** |
| exit within 3 iterations | **42.6%** (38% of all positions retire at the SECOND check) |
| exit at >= 10 iterations | 49.5% |
| range | 1 to 18 |

The spike at the second check is structural, not incidental: the logged trace goes
`1315 -> -0.009 -> -0.137 -> ...`, so whichever positions happen to be negative the moment it
crosses zero retire instantly, having had essentially no recurrence at all. The rest stay
positive a while and run 10+.

So the variance is real but it is **noise, not signal** — a hard token is no likelier to get
more iterations than an easy one. That is a fixed budget with jitter, which is the exact
opposite of adaptive compute, and it explains the observed symptom directly: a token that drew
an early exit is sampled mid-trajectory, while its distribution is still near the uniform the
iteration starts from. See the flat-logit entry below.
(Caveat: measured on a randomly-initialised block with random input, so the exact shape need
not match a trained checkpoint. The sign-flip mechanism that produces it is structural and does.)

### Flat logits at eval were unfinished recurrence, not model uncertainty (2026-09-16)
Diagnosed by the world-voice session and mechanistically consistent with the above. Voice
generation intermittently produced a near-uniform distribution over units — no confident choice
anywhere in the vocabulary — and considerable sampler engineering went into recovering from a
unit history containing one such state.

That symptom is what an early exit looks like. The recurrent trajectory starts near uniform (it
is why Huginn's `KLExitEvaluator` *initialises* `prev_log_probs` to uniform), so a position
retired at the second check is read out before its distribution has sharpened. With ~38% of
positions retiring there, flat logits were not rare events to be sampled around — they were the
expected output for a large minority of tokens.

Fixing the criterion collapsed the symptom. World-voice reports **WER 0.0989 vs a GT ceiling of
0.1460, at 74% of the compute** under `logit_kl` — better numbers AND less compute, because the
budget now follows the need instead of being spent uniformly. (GT ceiling = real audio through
the CV2 encoder to units, then back through the same decoder and speaker embedding. Generated
speech scoring *below* that is expected and is not "better than ground truth": real recordings
carry disfluency and noise that a TTS model does not reproduce. Judge quality by ear, per
[[feedback_recurring_reminders]].)

⚠️ The sampler that was tuned to work around flat logits is still the best config under the
fixed criterion. That the RANKING held is weak evidence; the informative quantity is the
**margin**. If the clever sampler's advantage over greedy has narrowed, it was a workaround and
can be simplified away; if the margin is unchanged, it is doing independent work. Not yet
measured.

**Relay to world-voice and world-image** — measured here, but it is not a world-text finding.

### `logit_kl` implements Huginn's criterion exactly (2026-09-15, commit e4074ea)
Ported from `tomg-group-umd/huginn-0125`, `raven_modeling_minimal.py`, `KLExitEvaluator`.
The three details that were wrong in the legacy version and are right here:

1. **The readout runs every iteration.** Huginn's `predict_from_latents` is `ln_f` +
   `lm_head`, so the comparison is between post-head *distributions*. The analogue here is
   the text coda, passed into `forward()` as `readout` with `use_cache=False`.
2. **`KL(prev ‖ current)`.** `F.kl_div(input, target, log_target=True)` is
   `exp(target) * (target - input)`; the reference passes current as `input` and previous
   as `target`. Non-negative by construction.
3. **`prev_log_probs` starts UNIFORM**, not at the first step's output — which is what
   guarantees one real iteration before any exit is possible.

Threshold: paper says 5e-4 ("if this divergence falls below 5x10^-4, we stop iterating");
the reference code's `"auto"` is 1e-3. Default here is 5e-4.

Verified: uniform init matches `log(1/V)` exactly; iteration-1 values all >= 0; identical
logits two steps running give exactly 0.0; on a fresh `small_sum`, `none` runs the full 32
and legacy `kl_divergence` runs 16-20 with a trace that goes negative.

⚠️ **The iteration COUNT is not yet validated on a trained model.** On random init the
criterion exits at iteration 2 (values 4.247e-02 then 6.642e-06 against a 5e-4 threshold) —
correct behaviour, because an untrained model's output distribution is near-uniform and
barely moves between iterations, so it genuinely has converged. It means a random-init test
can confirm the *math* but says nothing about whether 5e-4 is the right threshold here.
**Settled by:** running `logit_kl` vs `none` on a real checkpoint and reading the iteration
histogram. Huginn's threshold was tuned for a 3.5B model with a 4096-wide recurrent core
and `padded_vocab_size` 65536; nothing guarantees it transfers to a 127.5M trunk.

Deviation from the reference, deliberate: Huginn scores only the last position
(`logits[:, -1, :]`) because its adaptive compute runs during single-token generation.
`forward()` here processes whole sequences, so the quantity is computed per position,
(B, T). At seq_len 1 — generation, Huginn's actual regime — they are identical.

Scoping decision: `world_model` passes `converge_eligible = (modality_map == MODALITY_TEXT)`,
so only text positions can exit early. A text head's distribution is meaningless at
voice/audio/image positions; those run the full budget. Also note the readout is *exact*
only in pretrained/trainable-head mode, where the coda is stateless (norm -> MLP -> head).
With a from-scratch transformer coda it self-attends across the interleaved sequence, which
training never does, so the scored distribution is an approximation. It never contributes to
the output either way.

Cost: `logit_kl` holds (B, T, V) fp32 log-probs and runs the readout once per iteration —
negligible at seq_len 1, but ~5 GB at batch 8 x seq 1024 x vocab 152k. Use small eval
batches, or `none`, for full-sequence evaluation.

The broken `kl_divergence` is left in place **and left as the default** deliberately:
silently fixing it would change every historical eval that produced a number. Verified
untouched by rebuilding the old block from `git show HEAD:` — same 16 iterations, zero
output difference, identical trace.

### Measure in OUTPUT space, not latent space — the divergence is the easy half (2026-09-15)
Two choices get conflated and are independent:

1. **Which space** the criterion measures in — the trunk latent, or the post-readout output.
2. **Which divergence** on that space — KL on a simplex, relative L2 on a continuous space
   (they are the same Bregman divergence under different generators: negative entropy vs
   `1/2|x|^2`, so L2 is what KL becomes off the simplex).

(2) is settled and cheap. **(1) is the one that decides quality, and output space wins.**

Reasoning that "every readout is a deterministic function of the trunk state, so latent
convergence implies output convergence" is true but WEAK: deterministic continuity gives
convergence *in the limit*, and a thresholded criterion is not a limit statement. At a
threshold what matters is the Jacobian. Per the `huginn-exit-criteria-sweep` session:

> "not pointwise: the coda contains attention, so logits at position t depend on latents at
> all positions <= t. And the map isn't an isometry — ‖Δlogits‖ ≈ ‖J·Δx‖, so a latent step
> of a given size produces wildly different logit changes depending on its direction, then
> softmax weights that by probability mass."

Two distinct failures there: freezing token t's latent does not freeze token t's *output*
while other positions still move, and equal-norm latent steps produce unequal output steps.
Measured by that session on **huginn-0125**: relative-L2 in latent space is **~3x worse**
than KL on post-coda logits. NOT replicated on this model — external, different scale
(3.5B / 4096-wide core), recorded as the prior to beat, not as a local result.

Practical reading: `latent_diff` is the correct *readout-free* criterion and the right
fallback when a readout is unavailable or too expensive, but `logit_kl` should be expected
to beat it wherever the readout is strongly expansive and nonlinear — which is exactly the
text case (768 -> 49k-152k, then a softmax).

### For a continuous output space, measure the CONDITIONING, not a sample (2026-09-15)
Relevant once world-image goes autoregressive (see [[project_zimage_t4_ar]]) and image
positions want an exit criterion after all — the "single-shot, so no latency to amortise"
argument only holds for one-shot gen queries.

- **Do not quantise** the Qwen features or the flow-head conditioning to manufacture a
  distribution. That is the same error the original `kl_divergence` made and the reason
  `latent_kl` was dropped: inventing structure to fit the tool.
- **Do not diffuse per iteration.** The flow head maps `(conditioning, noise, t) ->
  velocity`, so with the conditioning fixed the induced output distribution is fixed. The
  conditioning is a *sufficient statistic* for what a sample-space criterion would measure.
  Sampling to check convergence measures the same quantity through an expensive stochastic
  channel and adds sampling noise to the test.
- **Do** apply the output-space rule above: relative L2 on the **Q-Former adapter output**
  (the flow-head conditioning) — the last deterministic representation before the generative
  head, which is the structural analogue of where `logit_kl` measures for text, with the
  divergence that matches a continuous space. One adapter forward per iteration, no diffusion.

### The recurrent block receives no timestep (2026-09-15)
`_run_iteration` passes `iteration` only to `kv_cache.get_layer_at_iteration(...)` for cache
slot selection; the blocks themselves get `(h, attention_mask, kv_cache, position_offset,
use_cache, additive_attn_bias, position_ids)` and no `t`. The trunk is therefore an
**autonomous** (time-invariant) iterated map, `dx/dtau = f(x)`.

Consequence for the "recurrent loop as sampler" direction ([[project_recurrent_loop_as_sampler]]):
flow matching needs a time-DEPENDENT velocity field `v(x, t)`. An autonomous field cannot
represent most probability paths (its trajectories cannot cross in state space), so
timestep conditioning into the recurrent block — AdaLN-style, which the image DiT already
does and is a pattern to copy — is a hard prerequisite, not a refinement.
`initialize_thinking_state` already supports noise init (`"normal"` / `"embed"`), so that
half is in place.

### The corpus converts to any tokenizer in an hour, so the teacher choice is reversible (2026-09-17)
`scripts/data/text/retokenize_shards.py` (commit f5b2bcb). Pack mode stores no text, but
decoding recovers it: split the stream on the source EOS to get documents, decode with the
source tokenizer, re-encode with the target, re-pack. Measured on the real cache,
Mistral -> Qwen3:

| | |
|---|---|
| decoded text round-trip | **exact**, 194/194 through both tokenizers |
| source id round-trip | 0/194 — *only* the first token of each document differs (`hip` vs `▁hip`) |
| token-count ratio | **0.900x** (Qwen3 is denser on English), so 5.25B -> ~4.7B |
| throughput | ~1.4M tok/s single process, **~1 hour** for the whole corpus |
| end-to-end check | 51,636 documents both sides, decoded text identical 300/300 |

The id mismatch is SentencePiece writing a word-boundary marker when encoding fresh text, so a
document that was cut mid-word loses that fact. One token per document, text unchanged.

⭐ **This removes the blocker that has been gating this whole direction, and weakens the
constraint above it.** "The teacher must be chosen BEFORE the re-preprocessing pass" assumed
the pass was a multi-terabyte re-download. It is not: the **Mistral cache is the master copy**
and a tokenizer-specific corpus is a cheap derived artifact. Choosing the wrong teacher now
costs an hour of CPU, not a re-download of the ~30-source mixture. Tokenizer-family choice is
still load-bearing for what a run MEANS — do not train an arm on a corpus tokenized for a
different model — but it is no longer an expensive one-way door.

Consequence for the shared-head ablation in OPEN below: its stated cost was "a full
re-download-and-preprocess pass". That cost is now an hour, so the decision rule there should
be re-read — the ablation is much cheaper than recorded, though the Qwen3-teacher argument that
it answers itself by fiat is unaffected.

### The Qwen3 vocabulary, not the trunk, dominates the parameter count (2026-09-17)
Measured by instantiation, `small_sum` text-only, d_model 768:

| corpus | tied | student total | embed+head share |
|---|---|---|---|
| Mistral (32000) | no | 195.6M | 25% |
| Mistral (32000) | yes | 171.0M | 14% |
| **Qwen3 (151936)** | no | **380.0M** | **61%** |
| **Qwen3 (151936)** | yes | **263.3M** | 44% |

A 151,945-wide table at d=768 is 116.7M, and an untied model carries two of them. So moving to
the Qwen3 corpus nearly doubles the model without adding any computation, and 44-61% of it is
lookup. Two consequences: `--tie_word_embeddings` is worth 116.7M here where on the Mistral
corpus it was a rounding error, and **"my student is <200M" is not true on this corpus** — any
capacity-gap reasoning should use 263M/380M, not 196M.

### Text distillation is built (2026-09-17, commit 8c8ddf3)
`model/text/distill_teacher.py` + `--text_distill_model / _weight / _temperature / _device /
_fp32`. Frozen `AutoModelForCausalLM` held outside the module tree (`object.__setattr__`, so it
is never optimized, checkpointed, or moved by the student's `.to()`), run under `no_grad` once
per batch, forward KL(teacher ‖ student) with the standard `T^2` scaling, ADDED to the CE rather
than replacing it. Mirrors the CosyVoice 2 voice path, which already proved the pattern.

Three things that are load-bearing rather than defensive:
- **The vocab guard RAISES.** A KL over mismatched vocabularies does not fail loudly — every id
  indexes a row, the loss falls, the run looks healthy. `assert_vocab_matches()` runs at
  construction. Same reason a failed teacher load is fatal here rather than a warning: a
  distillation arm silently running CE-only is a baseline wearing the wrong run name.
- **`valid_ctx` masks the KL after the first control token.** Control tokens sit above the
  teacher's vocab and must be clamped, but a clamped id is a WRONG token in a causal model's
  context and the corruption propagates rightward. Verified: a control token at position 10 of
  32 leaves `valid_ctx` 10/32. Pure text never trips it.
- **`--text_tokenizer` was a prerequisite, not a nicety.** `special_token_base` was only
  settable inside the `--text_encoder_model` branch, so a FROM-SCRATCH prelude was pinned to
  base 32000 / vocab 32009 and could not read a Qwen3 corpus at all.

Verified on CPU against a real Qwen3-0.6B: guard passes on a match and raises on 32000, KL 4.03
nats, backward clean, teacher `requires_grad=False` throughout. Not yet run on GPU or at scale.

**WHERE the KL applies, and why (2026-09-17, commits 6883bd6 + f978f3e).** A text-only teacher
cannot supervise text conditioned on something it never saw. For `image_transcription` /
`voice_transcription` its distribution over the caption is not a worse estimate of the right
answer — it answers a DIFFERENT question, "what text plausibly follows this text?" — so
distilling it would teach the student to ignore the very conditioning those tasks exist to
learn. Gated on `task_type`, which is exact because batches are homogeneous under
`ModalityGroupedSampler`; `valid_ctx` is a second line of defence.

`--text_distill_all_tasks` (off by default) extends the KL to the pre-media PREFIX of any
batch. That case is not merely harmless, it is *warranted*: text before BOV/BOI has context
identical to the student's, because no media has entered the sequence yet. Measured eligibility
against a real Qwen3-0.6B:

| layout | eligible positions |
|---|---|
| `text_continuation` (pure text) | **40/40** |
| `voice_synthesis` (transcript, BOV@20, placeholders after) | **20/40** — prefix only |
| interleaved text -> image -> text | **10/40** — leading text only |
| `image_transcription` (media first, caption after) | **0/40** — fully excluded |

⚠️ **The flag was inert when first written, and silently so.** `--mask_text_loss_in_synthesis`
sets `skip_text_loss` on synthesis batches, and the KL block was nested inside
`if ... and not skip_text_loss:` — so on exactly the batches the flag was meant to unlock, the
whole block was skipped. It failed by producing no `text_distill_kl` entry, which reads as "no
synthesis batches this window". Fixed by hoisting the KL out of the CE guard and keying it off
a pre-mask snapshot of the targets: *which positions hold real text* and *which positions CE
should train on* stop being the same question once synthesis masking is on.

This is defensible against the objection that `--mask_text_loss_in_synthesis` exists to REMOVE
transcript supervision: that is an argument about what to put a CE on, not about whether the
text is text. Under the old nesting those positions produced no text gradient at all, so the
KL is signal recovered from batches contributing nothing — not a reintroduced distraction.
Note the consequence: with the flag on, synthesis batches carry a text gradient they did not
have, so such runs are NOT step-comparable to earlier ones.

---

## OPEN

### Does the frozen shared head actually cost anything?
The whole point of commit 9dbe097 and unanswered. Shared-head mode caps the logit matrix at
rank <= llm_d+1 = 577 (vs 769 for a trunk-width head) and pins every output direction to the
frozen token geometry. Whether that binds at this scale is unmeasured.
**Settled by:** frozen prelude + shared head vs frozen prelude + trainable head, same seed,
same corpus, matched step. Compare in **bits-per-byte**, never nats/token — arms on different
tokenizers are not comparable per-token, and the trainer's `text_loss_norm = text_loss_raw /
log(vocab_size)` normalizes the uniform baseline but not tokenization granularity.

**Cost, and whether it is worth paying (2026-08-21).** Both arms need a **SmolLM2-tokenized
corpus, which does not exist** — the 5.25B tokens on disk are Mistral, pack mode, unrecoverable
in place. So this ablation is not one flag and two runs; it is one flag, two runs, **and a full
re-download-and-preprocess pass** over the mixture in `logs/huginn_dataset_mixture.md`. If the
direction then moves to a Qwen3 teacher, that is a *second* full pass, and the question is
answered by fiat anyway: a frozen SmolLM2 head cannot emit a 151936-entry vocab, so the
trainable head becomes mandatory and there is no frozen arm left to compare against.

=> **Worth a corpus pass only if the direction stays on SmolLM2.** Under a Qwen3 teacher, drop
the comparison; `--text_encoder_trainable_head` is still required, by force rather than by
evidence. The residual value — a prior on whether readout capacity binds at this scale, which
bears on whether a 116.7M Qwen3 readout is buying anything — is obtainable more cheaply and
more relevantly *inside* the Qwen3 setup, as full head vs a low-rank factorized head, on the
corpus that has to be built regardless.

### Is there an identity shortcut that makes text-only uninformative?
Predicted, not measured. In text-only, frozen prelude + frozen shared head form a complete
SmolLM2 forward path with the trunk as a residual in the middle, and
`out_translator o trunk o (input_proj + translator) ~ identity` is reachable. If the trunk
takes it, the run's floor is SmolLM2's own distribution and the run measures nothing about the
trunk. Voice and image have no such shortcut — their targets are a foreign space.
**Settled by:** the trunk's marginal contribution — same checkpoint, trunk output zeroed or
bypassed vs live — which is the text analogue of `early_text_delta`. If the delta is near zero
the arm is uninformative regardless of its absolute loss.

### Is Qwen3-4B the right teacher size for a ~300M student?
[Distillation Scaling Laws](https://arxiv.org/abs/2502.08606) (Busbridge et al., ICML 2025)
reports a **capacity gap** — student loss follows a power law in teacher cross-entropy that
transitions between regimes, and a teacher that is too strong makes the student worse. The
same work says distillation beats supervised pretraining "in settings involving many students
or an existing teacher", which is this project's situation (off-the-shelf teacher, many
ablation students on one corpus). Whether 4B over ~300M is past the gap is unmeasured.
**Evidence against the worry (2026-09-17):** Qwen3-0.6B was itself distilled from Qwen3-32B and
Qwen3-235B-A22B — ratios of ~50x and ~390x. A 4B teacher over a ~263-380M student is ~13x, far
inside what Qwen themselves shipped. Caveat: theirs was POST-TRAINING distillation onto an
already-pretrained student, not pretraining-scale distillation into a from-scratch one, so this
is evidence rather than proof. Note also that the student is 263M tied / 380M untied on this
corpus, not the <200M it is at Mistral vocab — see the parameter entry above.

**Recommendation for the FIRST run: Qwen3-0.6B, on cost rather than capacity.** Teacher forward
is +9% per step vs +61% for 4B, and 1.2GB vs 8.0GB bf16. Exercise the plumbing cheaply, then
ablate up.

**Settled by:** distil from Qwen3-0.6B / 1.7B / 4B at a small matched token budget. Shared
vocab means this ablation costs **zero** additional preprocessing.

### Does distillation change the data requirement enough to matter here?
Untested in this project for text. The mechanism is bits per token — a one-hot label carries
at most log2(49152) = 15.6 bits, a full teacher distribution carries the answer and the shape
of the uncertainty at every position. Expected to raise the ceiling of the existing 5.25B
corpus rather than to reduce the token count needed, since the corpus is already at Chinchilla
(above). Teacher forward adds +9% / +26% / +61% per step for Qwen3-0.6B / 1.7B / 4B in-loop,
or ~0 amortized if top-k logits are cached (one-time ~3.5 days on 4x4090 for a 4B pass over
the corpus; 0.50 TB at top-16, 1.01 TB at top-32).
**Settled by:** hard-label CE vs logit-KL from the same teacher, matched tokens, bits-per-byte.

---

## RETRACTED

*(none yet)*
