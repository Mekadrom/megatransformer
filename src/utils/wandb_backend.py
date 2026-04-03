"""Weights & Biases backend for MetricsLogger.

Logs scalars, images, audio, text, and figures to W&B. When context is
provided, related media items are grouped into a single ``wandb.Table``
row so they can be viewed together in the W&B UI.

Usage:
    from utils.wandb_backend import WandBBackend
    from utils.metrics import init_metrics

    backend = WandBBackend(project="megatransformer", run_name="my_run")
    init_metrics(backend)

Requires: ``pip install wandb``
"""

from typing import Optional

import numpy as np
from matplotlib.figure import Figure


class WandBBackend:
    """Weights & Biases metrics backend with context-aware media grouping."""

    supports_context = True

    def __init__(
        self,
        project: str = "megatransformer",
        run_name: Optional[str] = None,
        run_id: Optional[str] = None,
        config: Optional[dict] = None,
        log_dir: Optional[str] = None,
    ):
        import wandb
        self._wandb = wandb

        init_kwargs = {"project": project}
        if run_name is not None:
            init_kwargs["name"] = run_name
        if run_id is not None:
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = "allow"
        if config is not None:
            init_kwargs["config"] = config
        if log_dir is not None:
            init_kwargs["dir"] = log_dir

        self._run = wandb.init(**init_kwargs)
        # Cache table columns per tag prefix to accumulate rows across steps
        self._tables: dict[str, list] = {}

    def _convert_context_value(self, value):
        """Convert a context value to a W&B media type."""
        wandb = self._wandb
        if isinstance(value, str):
            return value
        elif isinstance(value, Figure):
            return wandb.Image(value)
        elif isinstance(value, np.ndarray):
            if value.ndim >= 2:
                # Image: ensure HWC format for wandb
                if value.ndim == 3 and value.shape[0] in (1, 3, 4) and value.shape[0] < value.shape[1]:
                    value = value.transpose(1, 2, 0)
                return wandb.Image(value)
            return value
        elif isinstance(value, (int, float)):
            return value
        elif isinstance(value, tuple) and len(value) == 2:
            audio, sr = value
            return wandb.Audio(audio, sample_rate=sr)
        return str(value)

    def _log_with_context(self, tag: str, primary_media, step: int, context: dict):
        """Log primary media + context as a W&B Table row."""
        wandb = self._wandb

        columns = ["step", "primary"]
        data = [step, primary_media]

        for key, value in context.items():
            columns.append(key)
            data.append(self._convert_context_value(value))

        table = wandb.Table(columns=columns, data=[data])
        self._run.log({f"{tag}/grouped": table}, step=step)

    def add_scalar(self, tag: str, value: float, step: int, context: Optional[dict] = None) -> None:
        self._run.log({tag: value}, step=step)
        if context:
            self._log_with_context(tag, value, step, context)

    def add_image(self, tag: str, img_array: np.ndarray, step: int, context: Optional[dict] = None) -> None:
        wandb = self._wandb
        # wandb expects HWC; our arrays are CHW
        if img_array.ndim == 3 and img_array.shape[0] in (1, 3, 4) and img_array.shape[0] < img_array.shape[1]:
            img_array = img_array.transpose(1, 2, 0)

        img = wandb.Image(img_array)

        if context:
            # Find caption from string context items
            caption_parts = [v for v in context.values() if isinstance(v, str)]
            if caption_parts:
                img = wandb.Image(img_array, caption=caption_parts[0])
            self._log_with_context(tag, img, step, context)

        self._run.log({tag: img}, step=step)

    def add_audio(self, tag: str, audio_array: np.ndarray, step: int, sample_rate: int, context: Optional[dict] = None) -> None:
        wandb = self._wandb

        # Find caption from string context items
        caption = None
        if context:
            caption_parts = [v for v in context.values() if isinstance(v, str)]
            if caption_parts:
                caption = caption_parts[0]

        audio = wandb.Audio(audio_array, sample_rate=sample_rate, caption=caption)
        self._run.log({tag: audio}, step=step)

        if context:
            self._log_with_context(tag, audio, step, context)

    def add_figure(self, tag: str, figure: Figure, step: int, context: Optional[dict] = None) -> None:
        wandb = self._wandb
        img = wandb.Image(figure)
        self._run.log({tag: img}, step=step)

        if context:
            self._log_with_context(tag, img, step, context)

    def add_text(self, tag: str, text: str, step: int, context: Optional[dict] = None) -> None:
        wandb = self._wandb

        if context:
            self._log_with_context(tag, text, step, context)
        else:
            # W&B doesn't have a native text panel like TensorBoard;
            # log as a Table with a single text column
            table = wandb.Table(columns=["text"], data=[[text]])
            self._run.log({tag: table}, step=step)

    def add_histogram(self, tag: str, values: np.ndarray, step: int, context: Optional[dict] = None) -> None:
        wandb = self._wandb
        self._run.log({tag: wandb.Histogram(values)}, step=step)

    def flush(self) -> None:
        pass  # W&B handles flushing internally

    def close(self) -> None:
        self._run.finish()
