"""Frozen causal-LM teacher for text logit distillation.

Mirrors `model/voice/cosyvoice2_teacher.py` in role: a frozen model held OUTSIDE the student's
module tree, run under `no_grad` once per batch, whose logits supply a KL term alongside the
hard-label CE. The voice teacher (CosyVoice 2) already proved the pattern; this is the text
analogue.

WHY KL ON TOP OF CE. Hard labels collapse a one-to-many target onto a single id. The teacher's
full distribution carries the "which continuations are plausible" structure that CE discards,
which is worth far more per token than a one-hot: at most log2(V) bits of label versus a dense
distribution over the whole vocabulary. That matters here specifically because this corpus is
~17.5 tokens/param -- about Chinchilla-optimal, i.e. NOT data-rich -- so extracting more signal
per token is the lever, not collecting more tokens. See docs/findings/world-text.md.

⚠️ THE FAILURE THIS CLASS EXISTS TO PREVENT. A KL is only meaningful between distributions over
the SAME vocabulary. If the corpus was tokenized with tokenizer A and the teacher speaks
tokenizer B, every id still indexes *something* in B's embedding table, so nothing raises and
the loss still falls -- it just trains against noise. Measured on this repo's own cache: Mistral
ids decoded under SmolLM2 turn "hip and artistic sensibilities of their time" into " area n
Adding solution hazardsstific): bak cour...". `assert_vocab_matches()` is therefore called at
construction and RAISES rather than warning. Do not downgrade it.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class TextDistillTeacher(nn.Module):
    """Frozen `AutoModelForCausalLM`, exposing per-position logits over its native vocab.

    Not a submodule of the student: the trainer stores it with `object.__setattr__` so it is
    never optimized, never checkpointed, and never moved by the student's `.to()`.
    """

    def __init__(self, model, vocab_size: int, device: str, dtype: torch.dtype):
        super().__init__()
        self.model = model
        self.vocab_size = int(vocab_size)
        self._device = device
        self._dtype = dtype
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @classmethod
    def from_pretrained(cls, model_name: str, device: str = "cuda",
                        dtype: torch.dtype = torch.bfloat16,
                        load_in_4bit: bool = False) -> "TextDistillTeacher":
        """Load the frozen teacher, optionally NF4-quantized.

        4-bit matters when several ranks each hold a teacher on one card: under DeepSpeed
        every rank builds its own, so 3 ranks x Qwen3-4B is 24GB in bf16 (overflows a 24GB
        card) against ~7.5GB in NF4. It buys little for a 0.6B teacher, which is ~1.2GB
        either way.

        ⚠️ Quantization error lands DIRECTLY in the supervision signal here -- unlike an
        inference use where it only perturbs a sample, a distillation teacher's logits ARE
        the target. Measure before trusting it; see the 4-bit fidelity entry in
        docs/findings/world-text.md. Same NF4 + bf16-compute setup the Z-Image text encoder
        uses, so the two are consistent.
        """
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = AutoConfig.from_pretrained(model_name)
        kw = {"dtype": dtype}
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype)
                kw["device_map"] = {"": device}   # bnb places at load; .to() is not allowed
            except Exception as e:
                print(f"WARNING: 4-bit teacher unavailable ({type(e).__name__}: {e}); "
                      f"falling back to {dtype}.", flush=True)
                load_in_4bit = False
        model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
        if not load_in_4bit:
            model = model.to(device)
        return cls(model, vocab_size=int(cfg.vocab_size), device=device, dtype=dtype)

    def assert_vocab_matches(self, student_base: int, model_name: str = "?"):
        """Hard-fail when the teacher's vocabulary differs from the corpus's.

        `student_base` is the world model's `special_token_base` -- by construction the size of
        the real vocabulary, with the multimodal control tokens living above it. A teacher whose
        native vocab differs is speaking a different language than the data.
        """
        if int(student_base) != self.vocab_size:
            raise ValueError(
                f"TOKENIZER MISMATCH: teacher {model_name!r} has vocab_size {self.vocab_size}, "
                f"but the model's special_token_base is {student_base}. These must be equal -- "
                f"the base IS the tokenizer's vocabulary size, and a KL between distributions "
                f"over different vocabularies is meaningless. This does NOT fail loudly on its "
                f"own (ids stay numerically in range and the loss still falls), so it is checked "
                f"here. Re-tokenize the corpus for this teacher with "
                f"`python -m megatransformer.scripts.data.text.retokenize_shards` (about an hour "
                f"for 5B tokens), or pick a teacher matching the corpus."
            )

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score a batch of student inputs.

        Returns:
            logits: (B, T, V_teacher) on the teacher's device, in the teacher's dtype.
            valid_ctx: (B, T) bool. False from the first out-of-vocabulary id in a row onward.

        `valid_ctx` exists because of the control tokens. They live at ids >= vocab_size, which
        the teacher's embedding has no row for, so they are clamped before the forward. Clamping
        keeps the call from throwing, but a clamped id is a WRONG token in the teacher's context,
        and because the teacher is causal that corruption propagates to every later position in
        the row. So the mask goes False at the first such id and stays False -- the teacher's
        opinion is only trustworthy on the clean prefix. For a pure text corpus no control token
        appears mid-sequence and the mask is all True.
        """
        ids = input_ids.to(self._device)
        ood = ids >= self.vocab_size
        valid_ctx = ood.cumsum(dim=1) == 0
        safe = ids.clamp(max=self.vocab_size - 1)
        out = self.model(input_ids=safe)
        return out.logits, valid_ctx.to(input_ids.device)


def text_distill_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    teacher_vocab: Optional[int] = None,
    max_positions: int = 0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Forward KL(teacher ‖ student) over the teacher's vocabulary, at masked positions.

    Direction and scaling follow the voice path and standard KD: forward KL is mode-COVERING
    (the student is penalised for putting no mass where the teacher does), and the `T**2` factor
    keeps the gradient magnitude independent of temperature so the weight means the same thing
    at any T.

    The student's head is WIDER than the teacher's -- it carries the control tokens above the
    native vocab -- so it is sliced to the teacher's width. That is a renormalisation over the
    native vocab, which is correct here: the control tokens are not part of the distribution the
    teacher is modelling, and the masked positions are ones where no control token is the target.

    ⚠️ MEMORY. This is the largest allocation in the step, and the reason `max_positions`
    exists. Over a 151,936-wide vocabulary every masked position costs ~0.6 MB in fp32, so a
    2x1024 micro-batch materialises ~2.9 GB student-side (gathered logits, the `.float()`
    copy, the log_softmax output) plus ~1.2 GB for `t_prob` -- measured, and it OOM'd a 24GB
    card at batch 2 even after the coda and CE had fit.

    Chunking does NOT fix this: every chunk's log_softmax is saved for backward, so the total
    retained is unchanged. Subsampling does. The KL is a MEAN over positions, so scoring a
    random subset is an unbiased estimator of it -- more variance per micro-batch, and with
    gradient accumulation the effective sample per optimizer step is `max_positions x
    accum_steps`, which is ample. Set `max_positions=0` to score every position.
    """
    if teacher_vocab is None:
        teacher_vocab = teacher_logits.shape[-1]
    temp = max(1e-3, float(temperature))
    if max_positions and int(max_positions) > 0:
        idx = mask.nonzero(as_tuple=False)
        n = idx.shape[0]
        if n > int(max_positions):
            # Sample WITHOUT replacement so no position is double-counted within a step.
            sel = torch.randperm(n, device=idx.device, generator=generator)[:int(max_positions)]
            keep = torch.zeros(n, dtype=torch.bool, device=idx.device)
            keep[sel] = True
            mask = torch.zeros_like(mask)
            mask[idx[keep, 0], idx[keep, 1]] = True
    s_logp = torch.log_softmax(student_logits[mask][:, :teacher_vocab].float() / temp, dim=-1)
    t_prob = torch.softmax(teacher_logits[mask].to(s_logp.device).float() / temp, dim=-1)
    return torch.nn.functional.kl_div(s_logp, t_prob, reduction="batchmean") * (temp ** 2)
