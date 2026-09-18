"""Run EleutherAI's lm-evaluation-harness against a world-model checkpoint.

WHY A SEPARATE SCRIPT. `eval_text_perplexity.py` measures the training objective on held-out
shards of our own corpus. That tracks optimisation but says nothing comparable to the outside
world: a perplexity in SmolLM2 token space over a bespoke mixture cannot be put next to any
published number. The harness tasks can, because every base-model report (GPT-3, Pythia, OLMo,
SmolLM2 itself) uses exactly these.

WHAT THIS NEEDS FROM THE MODEL. Every task below is scored by LOGLIKELIHOOD -- the harness
supplies (context, continuation) pairs and asks for log P(continuation | context), or for a
whole document under `loglikelihood_rolling`. Nothing here generates, so none of the KV-cache
and `generate()` machinery is involved and none of its failure modes can contaminate results.
`generate_until` therefore raises rather than pretending.

TWO ARCHITECTURAL FACTS THIS RELIES ON, both verified rather than assumed:
  1. The trunk is CAUSAL (`MegaTransformerBlockConfig.causal` defaults True and
     `recurrent_block_config.block_config` does not override it). So a batch may be RIGHT-padded:
     padding after the real tokens cannot influence any real position. `world_model.forward`
     exposes no attention_mask, so this is the only safe way to batch.
  2. ⚠ AN ACTIVE EXIT CRITERION MAKES SCORES DEPEND ON BATCH COMPOSITION. Convergence is
     tracked per token, `(B, T)`, and a converged token is frozen -- but the loop terminates on
     `converged.all()` (recurrent.py:524), which is GLOBAL ACROSS THE BATCH. Co-batching a
     harder sequence keeps the loop running, so the easier sequence's not-yet-converged tokens
     receive extra iterations and score differently. Measured on checkpoint-2000, the same
     request scored alone vs beside one longer sequence:
         exit_criteria=logit_kl   delta 1.48e-03 nats
         exit_criteria=none       delta 1.91e-06 nats   (fp32 noise)
     The second number is also the proof that right-padding itself is sound. For numbers you
     intend to report, use --batch_size 1 or --exit_criteria none; this script warns otherwise.
     Note the same coupling applies to the training-time eval at --eval_steps, where it is a
     function of --eval_batch_size.
  3. At eval the recurrent sampler returns `n = mean_thinking_steps, k = 0`
     (`recurrent.py:n_k_steps`, the `else` branch), i.e. a DETERMINISTIC iteration count with
     the exit criterion free to stop earlier. Scores are therefore reproducible without
     seeding anything.

ON "DOES THIS TASK MAKE SENSE FOR A PRETRAINING MODEL". All nine are base-model benchmarks;
none need instruction tuning. The distinction that actually matters at ~250M trainable
parameters and one 5B-token epoch is whether a task produces SIGNAL or sits at chance, so the
split below is by expected signal, not by instruct-vs-base:

  --tasks default (signal early):
    lambada_openai  last-word prediction; pure LM ability, moves from very early
    wikitext        word-level perplexity; always informative
    sciq            easy science MC; small base models land well above the 25% chance line
    piqa            physical commonsense; modest but real headroom over 50%

  --include_low_signal (off by default; near chance at this scale):
    hellaswag       ~29% vs 25% chance for a 160M base model
    winogrande      ~51% vs 50%; indistinguishable from noise below ~1B
    arc_easy        some signal, but high variance on 2376 items at this scale
    openbookqa      ~29% vs 25%
    boolq           ⚠ the majority class is 62.2%; small base models routinely score BELOW it,
                    so accuracy alone reads as "worse than a constant" -- the baseline is
                    logged alongside so the number cannot be misread

Neither list is a judgement about the tasks; it is about what a 5B-token run can resolve.
Pass --tasks explicitly to override both.

INSTALL (the harness is not in this project's lockfile; ask the user to run it):
    uv pip install "lm_eval==0.4.*"

USAGE
    python -m megatransformer.scripts.eval.world.eval_lm_harness --checkpoint_path runs/world_text/smollm2_ce_0/checkpoint-2000 --config small_sum --text_tokenizer HuggingFaceTB/SmolLM2-1.7B --tie_word_embeddings --exit_criteria logit_kl --exit_criteria_threshold 5e-4 --bf16 --batch_size 8 --log_dir runs/world_text/smollm2_ce_0
"""

import argparse
import json
import math
import os
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.amp import autocast

CORE_TASKS = ["lambada_openai", "wikitext", "sciq", "piqa"]
LOW_SIGNAL_TASKS = ["hellaswag", "winogrande", "arc_easy", "openbookqa", "boolq"]

# Published majority-class / chance baselines, logged next to accuracy so a number below the
# trivial baseline is visible as such instead of looking like a weak-but-positive result.
BASELINES = {
    "boolq": ("majority class", 0.622),
    "piqa": ("chance", 0.500),
    "winogrande": ("chance", 0.500),
    "hellaswag": ("chance", 0.250),
    "arc_easy": ("chance", 0.250),
    "openbookqa": ("chance", 0.250),
    "sciq": ("chance", 0.250),
    "lambada_openai": ("chance", 0.0),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--config", default="small_sum")
    p.add_argument("--text_tokenizer", default=None,
                   help="The CORPUS tokenizer, exactly as training was given it. This sets the "
                        "vocabulary, so a mismatch is not a warning -- the checkpoint simply "
                        "will not load (49161 vs the Mistral-era 32009).")
    p.add_argument("--text_encoder_model", default=None,
                   help="Only if training used a PRETRAINED prelude. smollm2_ce_0 does not.")
    p.add_argument("--tie_word_embeddings", action="store_true")
    p.add_argument("--exit_criteria", default=None,
                   help="Must match training: it decides how many recurrent iterations run at "
                        "eval, and therefore every logprob this script produces.")
    p.add_argument("--exit_criteria_threshold", type=float, default=None)
    p.add_argument("--extra_train_flags", default="",
                   help="Verbatim extra model-affecting flags, pasted from the run's "
                        "training/command_line text summary in TensorBoard.")
    p.add_argument("--tokenizer", default=None, help="Override the scoring tokenizer only.")
    p.add_argument("--tasks", default=None,
                   help="Comma-separated task list. Default: the core set; add the low-signal "
                        "set with --include_low_signal.")
    p.add_argument("--include_low_signal", action="store_true",
                   help="Also run the tasks that sit near chance at this scale (see docstring).")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only N docs per task -- for a smoke test, not for a result.")
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--max_length", type=int, default=1024,
                   help="Context window. The corpus is packed at 1024; going beyond what the "
                        "model saw in training measures extrapolation, not the model.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--log_dir", default=None, help="TensorBoard dir to log results into.")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--output_json", default=None, help="Write the full harness result here.")
    return p.parse_args()


def load_world_model(args, device):
    """Build through the REAL training loader, not a hand-rolled override dict.

    `--text_tokenizer` does more than set a number: it rewrites special_token_base, both
    vocab_size fields and eos_token_id, then calls `config.__post_init__()` to re-derive the
    interleaver's placeholder ids. Reproducing that here would drift from training the first
    time either side changed. So this builds the same argparse namespace training uses and
    calls `world.training.load_model`, which also loads the checkpoint.
    """
    import argparse as _ap
    from megatransformer.scripts.train import train as train_mod
    from megatransformer.scripts.train.world import training as world_training

    argv = ["world", "--config", args.config, "--include_modes", "text",
            "--resume_from_checkpoint", args.checkpoint_path,
            "--run_name", "_lm_harness", "--logging_base_dir", "/tmp"]
    if args.text_tokenizer:
        argv += ["--text_tokenizer", args.text_tokenizer]
    if args.text_encoder_model:
        argv += ["--text_encoder_model", args.text_encoder_model]
    if args.tie_word_embeddings:
        argv += ["--tie_word_embeddings"]
    if args.exit_criteria:
        argv += ["--exit_criteria", args.exit_criteria]
    if args.exit_criteria_threshold is not None:
        argv += ["--exit_criteria_threshold", str(args.exit_criteria_threshold)]
    if args.bf16:
        argv += ["--bf16"]
    if args.extra_train_flags:
        argv += args.extra_train_flags.split()

    p = _ap.ArgumentParser()
    train_mod.add_args(p)
    targs = p.parse_args(argv)
    m = world_training.load_model(targs, device=str(device))
    m.to(device)
    m.eval()
    return m


class WorldModelLM:
    """lm-eval adapter. Subclasses lm_eval.api.model.LM at construction time.

    Deliberately implements loglikelihood/loglikelihood_rolling directly rather than going
    through TemplateLM, which would couple this to the harness's internal tokenisation
    helpers and break on minor version bumps. The only harness API used is the three abstract
    methods and the request objects' `.args` tuples.
    """

    def __init__(self, model, tokenizer, device, max_length=1024, batch_size=8, bf16=False):
        self.model = model
        self.tok = tokenizer
        self.device = device
        self._max_length = max_length
        self.batch_size = batch_size
        self.bf16 = bf16
        self.eot_token_id = tokenizer.eos_token_id

    # ---- harness plumbing ---------------------------------------------------------------
    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError(
            "eval_lm_harness scores by loglikelihood only. A generative task was requested, "
            "which would need the world model's generate() path (KV caches for prelude, trunk "
            "and coda). None of the default tasks need it; if you add one that does, wire "
            "generate() in deliberately rather than letting it be exercised by accident.")

    # ---- scoring ------------------------------------------------------------------------
    @torch.no_grad()
    def _score_batch(self, seqs: List[List[int]]) -> List[torch.Tensor]:
        """Per-token log-probs of seq[1:] under seq[:-1], for a right-padded batch.

        Right-padding is safe ONLY because the trunk is causal; see the module docstring.
        """
        widths = [len(s) for s in seqs]
        w = max(widths)
        pad = self.eot_token_id if self.eot_token_id is not None else 0
        inp = torch.full((len(seqs), w - 1), pad, dtype=torch.long)
        for i, s in enumerate(seqs):
            inp[i, : len(s) - 1] = torch.tensor(s[:-1], dtype=torch.long)
        inp = inp.to(self.device)

        dtype = torch.bfloat16 if self.bf16 else torch.float32
        with autocast(self.device.type if hasattr(self.device, "type") else str(self.device),
                      dtype=dtype, enabled=self.bf16):
            out = self.model(text_input_ids=inp)
        logits = out["logits"] if isinstance(out, dict) else out.logits
        logprobs = F.log_softmax(logits.float(), dim=-1)

        res = []
        for i, s in enumerate(seqs):
            n = len(s) - 1
            tgt = torch.tensor(s[1:], dtype=torch.long, device=logprobs.device)
            lp = logprobs[i, :n, :]
            res.append((lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1), lp.argmax(dim=-1), tgt))
        return res

    def _run(self, jobs, reduce):
        """jobs: list of (key, seq, slice_start). Batches longest-first for even padding."""
        order = sorted(range(len(jobs)), key=lambda i: -len(jobs[i][1]))
        out = [None] * len(jobs)
        for b in range(0, len(order), self.batch_size):
            idxs = order[b: b + self.batch_size]
            scored = self._score_batch([jobs[i][1] for i in idxs])
            for i, (tok_lp, greedy, tgt) in zip(idxs, scored):
                out[i] = reduce(jobs[i], tok_lp, greedy, tgt)
        return out

    def loglikelihood(self, requests, disable_tqdm=False) -> List[Tuple[float, bool]]:
        jobs = []
        for r in requests:
            ctx, cont = r.args
            ctx_ids = self.tok.encode(ctx) if ctx else [self.eot_token_id]
            cont_ids = self.tok.encode(cont)
            if not cont_ids:                      # empty continuation scores as certain
                cont_ids = [self.eot_token_id]
            seq = (ctx_ids + cont_ids)[-(self._max_length + 1):]
            n_cont = min(len(cont_ids), len(seq) - 1)
            jobs.append((n_cont, seq, None))

        def reduce(job, tok_lp, greedy, tgt):
            n_cont = job[0]
            lp = tok_lp[-n_cont:]
            is_greedy = bool((greedy[-n_cont:] == tgt[-n_cont:]).all().item())
            return (float(lp.sum().item()), is_greedy)

        return self._run(jobs, reduce)

    def loglikelihood_rolling(self, requests, disable_tqdm=False) -> List[float]:
        """Whole-document logprob, in non-overlapping windows of max_length.

        Each window is prefixed with EOS so its first real token is scored under a context
        rather than being dropped -- otherwise a document's total omits one token per window
        and the resulting perplexity is quietly optimistic.
        """
        totals = []
        for r in requests:
            (doc,) = r.args
            ids = self.tok.encode(doc)
            jobs = []
            for i in range(0, len(ids), self._max_length):
                chunk = ids[i: i + self._max_length]
                if not chunk:
                    continue
                jobs.append((len(chunk), [self.eot_token_id] + chunk, None))
            if not jobs:
                totals.append(0.0)
                continue
            parts = self._run(jobs, lambda job, lp, g, t: float(lp.sum().item()))
            totals.append(sum(parts))
        return totals


def main():
    args = parse_args()
    try:
        import lm_eval
        from lm_eval.api.model import LM
    except ImportError:
        raise SystemExit(
            "lm_eval is not installed. Ask the user to run:\n"
            '    uv pip install "lm_eval==0.4.*"\n'
            "(Claude must not run uv commands.)")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tasks = ([t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks
             else CORE_TASKS + (LOW_SIGNAL_TASKS if args.include_low_signal else []))

    from transformers import AutoTokenizer
    tok_src = args.tokenizer or args.text_tokenizer or args.text_encoder_model
    if not tok_src:
        raise SystemExit("pass --text_tokenizer (the corpus tokenizer training was given)")
    tok = AutoTokenizer.from_pretrained(tok_src)
    model = load_world_model(args, device)

    # Bind the adapter to the harness base class here, so importing this module (for tests or
    # for the scoring core) does not require lm_eval to be installed.
    Adapter = type("WorldModelLMBound", (WorldModelLM, LM), {})
    lm = Adapter.__new__(Adapter)
    LM.__init__(lm)
    WorldModelLM.__init__(lm, model, tok, device, args.max_length, args.batch_size, args.bf16)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"checkpoint {args.checkpoint_path}  ({n_params/1e6:.1f}M trainable)")
    print(f"tokenizer  {tok_src}  (vocab {tok.vocab_size})")
    print(f"tasks      {', '.join(tasks)}")
    if args.limit:
        print(f"⚠ --limit {args.limit}: a smoke test, NOT a reportable result")
    if args.batch_size > 1 and (args.exit_criteria or "none") != "none":
        print(f"⚠ --batch_size {args.batch_size} with --exit_criteria {args.exit_criteria}: "
              f"scores depend on which requests share a batch, because the recurrent loop "
              f"exits on converged.all() across the batch (see the module docstring). "
              f"Use --batch_size 1 or --exit_criteria none for reportable numbers.")

    results = lm_eval.simple_evaluate(
        model=lm, tasks=tasks, num_fewshot=args.num_fewshot,
        limit=args.limit, batch_size=args.batch_size)

    scalars, lines = {}, []
    for task, res in sorted(results["results"].items()):
        for metric, value in sorted(res.items()):
            if not isinstance(value, (int, float)) or metric.endswith("_stderr"):
                continue
            key = metric.split(",")[0]
            scalars[f"lm_harness/{task}/{key}"] = float(value)
            base = BASELINES.get(task)
            note = ""
            if base and key in ("acc", "acc_norm"):
                name, b = base
                note = f"   ({name} {b:.3f}" + (
                    "  <-- BELOW BASELINE)" if value < b else ")")
            lines.append(f"  {task:<18}{key:<12}{value:>10.4f}{note}")

    print("\n" + "=" * 72)
    print("\n".join(lines))
    print("=" * 72)

    if args.log_dir:
        from megatransformer.scripts.eval.world.eval_utils import (
            infer_step_from_checkpoint, init_eval_metrics, log_eval_scalars)
        step = args.step if args.step is not None else infer_step_from_checkpoint(
            args.checkpoint_path)
        init_eval_metrics(args.log_dir, args.checkpoint_path)
        log_eval_scalars(scalars, step)
        print(f"logged {len(scalars)} scalars at step {step} -> {args.log_dir}")

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
