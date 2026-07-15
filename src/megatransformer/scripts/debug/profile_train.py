"""Wrap a training run in torch.profiler without changing the training code.

Takes the SAME arguments as scripts.train.train and passes them straight through, so
any run can be profiled by swapping the module path:

    uv run python -m megatransformer.scripts.debug.profile_train world \
        --run_name prof --config small_sum \
        --include_modes voice --include_tasks voice_synthesis \
        ... (every other train.py flag, unchanged) ...

Profiler-specific flags are prefixed --prof_ and are stripped before train.py sees argv.
Training is stopped as soon as the profiling window closes, so there is no need to pass a
small --max_steps (though it does no harm).

WHAT TO LOOK AT in the printed table:
  - If `aten::mm` / `aten::bmm` / `_scaled_mm` dominate CUDA time, you are compute-bound
    and the GEMMs are the cost -- that is the expected shape for this model.
  - If `aten::softmax`, `aten::masked_fill_`, `aten::mul`, or `aten::_softmax_backward`
    are near the top, the hand-rolled attention's full (B, H, T, T) elementwise passes
    are the cost, and switching to F.scaled_dot_product_attention would pay off.
  - If `cudaMemcpyAsync` or `cudaStreamSynchronize` show large CPU time, you are stalling
    on GPU->CPU syncs. `track_iteration_stats` does 6 syncs per recurrent iteration and
    is enabled on logging steps only (world/training.py:561-565) -- so whether these
    appear depends on whether the window straddles a logging step (see --prof_wait).
  - Compare `Self CUDA %` against wall time: a low total CUDA time relative to the step
    wall time means the GPU is idle waiting on the dataloader.

The default window (wait 5, warmup 3, active 8) profiles steps 8-15, which WILL include
step 10 if --logging_steps 10. To profile only ordinary steps, pass --logging_steps 100
to train.py, or shift the window with --prof_wait.
"""

import argparse
import runpy
import sys

import torch
from torch.profiler import ProfilerActivity, profile, schedule


class _ProfilingWindowClosed(Exception):
    """Unwinds out of Trainer.train() once the profiling window is done."""


def main():
    ap = argparse.ArgumentParser(
        add_help=False,  # let train.py own --help so passthrough stays honest
        description="Run scripts.train.train under torch.profiler.",
    )
    ap.add_argument("--prof_wait", type=int, default=5,
                    help="Steps to skip before profiling (lets shapes/allocator settle).")
    ap.add_argument("--prof_warmup", type=int, default=3,
                    help="Steps to warm up the profiler (recorded but discarded).")
    ap.add_argument("--prof_active", type=int, default=8,
                    help="Steps actually profiled.")
    ap.add_argument("--prof_output", type=str, default="profile_trace.json.gz",
                    help="Chrome trace path. Open in chrome://tracing or perfetto.dev.")
    ap.add_argument("--prof_rows", type=int, default=30,
                    help="Rows in the printed key_averages table.")
    ap.add_argument("--prof_memory", action="store_true",
                    help="Also profile allocations. Adds noticeable overhead.")
    ap.add_argument("--prof_stack", action="store_true",
                    help="Record python stacks — attributes kernels to source lines, but "
                         "is slow and inflates the trace.")
    prof_args, passthrough = ap.parse_known_args()

    total_steps = prof_args.prof_wait + prof_args.prof_warmup + prof_args.prof_active

    # Drive the profiler from HF's training_step so one profiler step == one training step,
    # regardless of gradient accumulation or which trainer subclass is in play.
    from transformers.trainer import Trainer

    original_training_step = Trainer.training_step
    state = {"prof": None, "steps": 0}

    def training_step_with_profiler(self, *args, **kwargs):
        out = original_training_step(self, *args, **kwargs)
        prof = state["prof"]
        if prof is not None:
            prof.step()
            state["steps"] += 1
            if state["steps"] >= total_steps:
                raise _ProfilingWindowClosed
        return out

    Trainer.training_step = training_step_with_profiler

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    print(f"[profile] window: wait={prof_args.prof_wait} warmup={prof_args.prof_warmup} "
          f"active={prof_args.prof_active} (stopping after step {total_steps})")
    print(f"[profile] passthrough argv: {' '.join(passthrough)}")

    with profile(
        activities=activities,
        schedule=schedule(
            wait=prof_args.prof_wait,
            warmup=prof_args.prof_warmup,
            active=prof_args.prof_active,
            repeat=1,
        ),
        record_shapes=True,          # needed to see GEMM shapes / arithmetic intensity
        profile_memory=prof_args.prof_memory,
        with_stack=prof_args.prof_stack,
    ) as prof:
        state["prof"] = prof
        sys.argv = ["train.py"] + passthrough
        try:
            runpy.run_module("megatransformer.scripts.train.train", run_name="__main__")
        except _ProfilingWindowClosed:
            print(f"\n[profile] window closed after {state['steps']} steps; training stopped.")
        except SystemExit as e:  # train.py may exit normally (e.g. --help, arg errors)
            if e.code not in (0, None):
                raise

    sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    print(f"\n[profile] top {prof_args.prof_rows} ops by {sort_key}:\n")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=prof_args.prof_rows))

    print(f"\n[profile] top {prof_args.prof_rows} ops by {sort_key}, GROUPED BY INPUT SHAPE:\n")
    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by=sort_key, row_limit=prof_args.prof_rows))

    prof.export_chrome_trace(prof_args.prof_output)
    print(f"\n[profile] chrome trace written to {prof_args.prof_output}")
    print("[profile] view it at https://ui.perfetto.dev or chrome://tracing")


if __name__ == "__main__":
    main()
