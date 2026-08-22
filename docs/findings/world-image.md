# world-image (text -> image)

Image synthesis from the recurrent trunk into a frozen diffusion decoder (Z-Image / SDXL) via
a trainable conditioning adapter.

⚠️ The entries below were REPORTED IN CONVERSATION by the project owner during the
2026-08-21 world-voice session and are recorded here so they are not lost. They have not been
independently re-measured by whoever wrote this file — treat the numbers as owner-reported
rather than verified, and correct them from the world-image session's own records.

---

## ESTABLISHED (owner-reported)

### Best configuration is NAR, not autoregressive
Qwen features interpolated to a fixed 64-length sequence and diffused in parallel with a
flow-matching head, conditioned via a Q-Former whose input is 64 learned image gen queries
from the recurrent trunk. The head and Q-Former see NO text; only the trunk does. T4
(per-token autoregressive flow head, MAR-style) performed badly by comparison.

The 64 is not structural: SDXL's CLIP conditioning is 77 tokens, and 64 covers a majority of
Qwen condition sequences for the training mixture, so it interpolates UP rather than squeezing
down. It also equals the gen-query count, making the Q-Former a 64->64 cross-attention module.

### Guidance is the win, not the flow head per se

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
"Matches the GT-prompt ceiling" says the conditioner is not the bottleneck; it does not say
the output is good. Whether the low ceiling is the data mixture, caption quality, or
CLIPScore's own insensitivity is unresolved.
