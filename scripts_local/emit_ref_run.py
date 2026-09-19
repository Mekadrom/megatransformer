"""Emit a flat reference line as its OWN TensorBoard run.

TB charts are keyed by TAG and overlay one line per RUN, so a reference curve has to reuse
the training run's exact tag names in a sibling run directory. A new tag would create a
separate chart instead, which defeats the point.

Two points (step 0 and step max_steps) are enough -- TB interpolates, giving a flat line
across the whole x-range.
"""
import argparse, math, os
from torch.utils.tensorboard import SummaryWriter

ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True, help="e.g. runs/world_text/REF_smollm2_135m")
ap.add_argument("--ce", type=float, required=True, help="corpus CE in nats/token")
ap.add_argument("--max_steps", type=int, default=76000)
ap.add_argument("--vocab", type=int, default=49161)
ap.add_argument("--task", default="text_continuation")
a = ap.parse_args()

ppl = math.exp(a.ce)
norm = a.ce / math.log(a.vocab)
os.makedirs(a.run_dir, exist_ok=True)
w = SummaryWriter(a.run_dir)
for step in (0, a.max_steps):
    w.add_scalar(f"eval/{a.task}/text_loss_raw", a.ce, step)
    w.add_scalar(f"eval/{a.task}/text_loss_norm", norm, step)
    w.add_scalar(f"eval/{a.task}/text_ppl", ppl, step)
    w.add_scalar("train/text_loss_raw", a.ce, step)
    w.add_scalar("train/text_ppl", ppl, step)
w.close()
print(f"wrote {a.run_dir}")
print(f"  CE {a.ce:.4f} nats/tok | PPL {ppl:.2f} | loss_norm {norm:.4f}")
print(f"  flat across steps 0..{a.max_steps}, tags match the training run")
