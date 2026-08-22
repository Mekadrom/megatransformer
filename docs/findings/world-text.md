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
pack mode, so the corpus cannot be re-tokenized in place — it has to be rebuilt from source.

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

---

## OPEN

### Does the frozen shared head actually cost anything?
The whole point of commit 9dbe097 and unanswered. Shared-head mode caps the logit matrix at
rank <= llm_d+1 = 577 (vs 769 for a trunk-width head) and pins every output direction to the
frozen token geometry. Whether that binds at this scale is unmeasured.
**Settled by:** frozen prelude + shared head vs frozen prelude + trainable head, same seed,
same corpus, matched step. Compare in **bits-per-byte**, never nats/token — arms on different
tokenizers are not comparable per-token, and the trainer's `text_loss_raw / log(vocab_size)`
normalizes the uniform baseline but not tokenization granularity.

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
