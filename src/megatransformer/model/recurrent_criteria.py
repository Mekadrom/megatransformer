import torch
import torch.nn.functional as F


from typing import Optional, Tuple


class RecurrentExitCriteria:
    # True when the criterion needs output LOGITS (a readout applied to the latent),
    # rather than the latent itself. The recurrent block checks this to decide whether
    # it must call the readout each iteration.
    needs_readout = False

    def should_exit(self, last_thought_state: torch.Tensor, current_thought_state: torch.Tensor):
        raise NotImplementedError


class NoOpCriteria(RecurrentExitCriteria):
    """Never exits early -- the trunk always runs its full iteration budget.

    This is the honest control arm, not a no-op: `mean_thinking_steps` (32 by default)
    is the depth the model was TRAINED at, so running to the cap is what matches
    training. Use it as the baseline any adaptive criterion has to beat.
    """

    # -inf so the generic `exit_values < threshold` test can never fire, while still
    # letting the block log a real per-iteration diagnostic (see exit_values below).
    threshold = float("-inf")

    def should_exit(self, last_thought_state, current_thought_state):
        return False

    def exit_values(self, last_thought_state: torch.Tensor, current_thought_state: torch.Tensor) -> torch.Tensor:
        """Relative latent movement -- logged for diagnostics, never used to exit."""
        delta = (current_thought_state - last_thought_state).norm(dim=-1)
        return delta / current_thought_state.norm(dim=-1).clamp_min(1e-6)

    def converged_mask(self, last_thought_state: torch.Tensor, current_thought_state: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            last_thought_state.shape[:-1], dtype=torch.bool, device=last_thought_state.device
        )


class KLDivergenceCriteria(RecurrentExitCriteria):
    """LEGACY, NUMERICALLY BROKEN -- kept only so old eval numbers stay reproducible.

    `F.kl_div(a, b, log_target=True)` evaluates `exp(b) * (b - a)`, which is a KL
    divergence only when both arguments are log-probabilities. The thought states passed
    here are post-norm activations, so this computes a SIGNED quantity, and the test
    `value < threshold` therefore passes for every negative value -- roughly half of all
    tokens on a near-converged step, by sign accident rather than by convergence. Measured
    2026-09-15: eval ran 17 of 32 iterations on a fresh `small_sum`, with the logged trace
    going negative from iteration 4. See docs/findings/world-text.md.

    Use `logit_kl` (Huginn's actual criterion) or `none` instead. This class is not fixed
    in place on purpose: changing it would silently alter every historical eval it produced.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold

    def should_exit(self, last_thought_state: Optional[torch.Tensor], current_thought_state: Optional[torch.Tensor]):
        if last_thought_state is None or current_thought_state is None:
            return False

        kl_divergence = F.kl_div(last_thought_state, current_thought_state, reduction="none", log_target=True).sum(dim=-1)
        return (kl_divergence < self.threshold).any()

    def converged_mask(self, last_thought_state: torch.Tensor, current_thought_state: torch.Tensor) -> torch.Tensor:
        """Per-token convergence mask. Shape: (batch, seq_len). True = converged."""
        kl = F.kl_div(last_thought_state, current_thought_state, reduction="none", log_target=True).sum(dim=-1)
        return kl < self.threshold


class LatentDiffCriteria(RecurrentExitCriteria):
    """Huginn's `LatentDiffExitEvaluator`: normalised relative distance in latent space.

    Ported from `tomg-group-umd/huginn-0125`, `raven_modeling_minimal.py`:

        exit_values = ((latents - self.prev_latents).norm(dim=-1) / latents.norm(dim=-1)).mean(dim=-1)
        return exit_values < self.exit_threshold      # exit_threshold "auto" = 0.03

    This is the reference implementation's own readout-free criterion, and the paper
    describes it as "the simplest adaptive exit criterion ... normalized distance in latent
    space". If the goal is "Huginn's criterion but without the readout", this is literally
    it -- `latent_kl` is a corrected version of what THIS codebase was doing, which is a
    different thing.

    Scale-free by construction (it divides by the state norm), so unlike `latent_kl` its
    threshold does transfer across widths and normalisation settings. That is the main
    reason to prefer it.

    Deviation from the reference: Huginn takes `.mean(dim=-1)` over the sequence, giving one
    value per batch row, because its adaptive compute runs during single-token generation.
    Kept per position here, (B, T), to match the rest of this codebase's per-token freeze.
    At seq_len 1 the two are identical.
    """

    def __init__(self, threshold: float = 0.03):
        self.threshold = threshold

    def exit_values(self, last_thought_state: torch.Tensor, current_thought_state: torch.Tensor) -> torch.Tensor:
        delta = (current_thought_state - last_thought_state).norm(dim=-1)
        return delta / current_thought_state.norm(dim=-1).clamp_min(1e-6)

    def converged_mask(self, last_thought_state: torch.Tensor, current_thought_state: torch.Tensor) -> torch.Tensor:
        return self.exit_values(last_thought_state, current_thought_state) < self.threshold


class LogitKLCriteria(RecurrentExitCriteria):
    """Huginn's `KLExitEvaluator`: KL between successive OUTPUT distributions.

    Ported verbatim from the reference implementation
    (`tomg-group-umd/huginn-0125`, `raven_modeling_minimal.py`, `KLExitEvaluator`):

        def init(self, initial_latents):
            self.prev_log_probs = ((1 / self.V) * torch.ones(batch_size, self.V, ...)).log()

        def check(self, model, latents, aux_inputs):
            outputs = model.predict_from_latents(latents, **aux_inputs)
            log_probs = F.log_softmax(outputs.logits[:, -1, :].float(), dim=-1)
            exit_values = F.kl_div(log_probs, self.prev_log_probs,
                                   reduction="none", log_target=True).sum(dim=-1)
            self.prev_log_probs = log_probs
            return exit_values < self.exit_threshold, outputs, exit_values

    Three details that matter and are easy to get wrong:

    1. The readout IS applied every iteration. `predict_from_latents` is `ln_f` + `lm_head`,
       so the comparison is between post-head probability distributions -- NOT latents.
    2. Argument order gives **KL(prev || current)**. `F.kl_div(input, target,
       log_target=True)` computes `exp(target) * (target - input)`; here `input` is the
       current step and `target` is the previous one. Non-negative by construction, unlike
       the legacy latent-space version.
    3. `prev_log_probs` is initialised to the **uniform** distribution, not to the first
       step's output. That makes iteration 1's divergence large (~log V minus the model's
       entropy) and so guarantees at least one real iteration before any exit is possible.

    Threshold: the reference code's `"auto"` is 1e-3; the paper reports 5e-4 for its
    experiments ("the KL-divergence between two successive steps. If this divergence falls
    below 5x10^-4, we stop iterating"). Default here is the paper's 5e-4.

    Deviation from the reference, deliberate: Huginn evaluates only the last position
    (`logits[:, -1, :]`) because its adaptive compute runs during single-token generation.
    This codebase's `forward()` processes a whole sequence, so the same quantity is
    computed per position, shape (batch, seq_len). At seq_len 1 -- i.e. generation, the
    regime Huginn actually uses -- the two are identical.

    Memory: holds the previous log-probs, (batch, seq_len, vocab) in fp32. Negligible for
    generation (seq_len 1) but ~5 GB at batch 8 x seq 1024 x vocab 152k. Prefer small eval
    batches, or `none`, for full-sequence evaluation.
    """

    needs_readout = True

    def __init__(self, threshold: float = 5e-4):
        self.threshold = threshold

    def init_state(self, logits: torch.Tensor) -> torch.Tensor:
        """Uniform log-probabilities, matching the reference's `init`, shaped like `logits`."""
        vocab = logits.shape[-1]
        return torch.full(
            logits.shape, 1.0 / vocab, device=logits.device, dtype=torch.float32
        ).log()

    def step(
        self, prev_log_probs: torch.Tensor, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One iteration's check.

        Returns (converged_mask (B,T), new_prev_log_probs, exit_values (B,T)).
        """
        log_probs = F.log_softmax(logits.float(), dim=-1)
        exit_values = F.kl_div(
            log_probs, prev_log_probs, reduction="none", log_target=True
        ).sum(dim=-1)
        return exit_values < self.threshold, log_probs, exit_values
