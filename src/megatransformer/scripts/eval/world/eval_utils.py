"""Shared utilities for world model eval scripts."""

import os
import re


def infer_step_from_checkpoint(checkpoint_path: str) -> int:
    """Extract step number from checkpoint path like '.../checkpoint-3000'."""
    match = re.search(r'checkpoint-(\d+)', checkpoint_path)
    return int(match.group(1)) if match else 0


def init_eval_metrics(log_dir: str = None, checkpoint_path: str = None):
    """Initialize TensorBoard metrics logging for eval scripts.

    If log_dir is provided, logs will be written there. Otherwise no logging.
    Returns the step number (inferred from checkpoint_path if not 0).
    """
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        from megatransformer.utils.metrics_backend import TensorBoardBackend
        from megatransformer.utils import metrics
        metrics.init_metrics(TensorBoardBackend(log_dir=log_dir))


def log_eval_scalars(scalars: dict, step: int):
    """Log a dict of {name: value} scalars to metrics if initialized."""
    try:
        from megatransformer.utils import metrics
        logger = metrics.get_logger()
        if logger is None:
            return
        for name, value in scalars.items():
            metrics.log_scalar(name, value, step)
        metrics.flush()
    except Exception:
        pass
