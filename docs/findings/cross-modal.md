# Cross-modal principles

Only for findings REPLICATED IN MORE THAN ONE DIRECTION. Something that merely feels general
belongs in the file for the direction it was measured in.

---

## ESTABLISHED

### Point-estimate predictors are structurally under-dispersed, and frozen decoders render that as degraded output
Measured on the image side: predicted conditioning has alpha ~= 0.745 against R^2 ~= 0.738 —
the conditional-mean signature — and rescaling by 1/alpha recovers ~30% of the gap with no
retraining. Replicated across two decoders, two feature spaces, and two opposite regimes.

Calibrate the correction by RENDER or EAR, never by loss: on SDXL the feature-space MSE
preferred a gain of 0.607 while the render preferred 1.45. The loss can have the sign wrong.

⚠️ What does NOT follow, and was wrongly inferred once: that a generative head therefore beats
a regression head. On the image side, unguided flow matched gain-corrected regression
(0.277-0.287 vs 0.283). The generative head's value there was that it makes GUIDANCE possible,
which a point estimate cannot express at any calibration.

## OPEN

### Does "model the distribution, not the mean" hold on the voice side?
Image models continuous conditioning with a flow head; voice models discrete tokens with a
categorical head. Both are distribution-modelling rather than mean-regression, and the AR
voice failure (greedy putting 96% of its mass on repeats) looks like the same disease in a
third form. But the voice-side per-step distributions turned out NOT to be peaked, so the
mechanism there is self-conditioning rather than mode collapse — which may mean these are two
different problems that happen to rhyme. Not promoted until measured as one.
