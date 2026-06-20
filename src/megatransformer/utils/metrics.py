"""Central metrics logging module.

Provides a backend-agnostic API for logging scalars, images, audio, figures,
and text during training. All logging goes through the module-level singleton
MetricsLogger, which delegates to a pluggable MetricsBackend.

Every log method accepts an optional ``context`` dict of related media to
log alongside the primary item. For TensorBoard these are emitted as sibling
tags (``{tag}/{key}``); richer backends can render them in unified panels.

Usage:
    from megatransformer.utils import metrics

    # Initialize once at training start (done by train.py):
    metrics.init_metrics(backend)

    # Log from anywhere:
    metrics.log_scalar("train/loss", loss, global_step)

    # Log with context — audio + its mel spectrogram + transcription together:
    metrics.log_audio("eval/voice/0", waveform, global_step, sr, context={
        "mel": mel_figure,           # Figure → log_figure
        "transcription": "hello",    # str → log_text
    })
"""

from typing import Optional, Union

import numpy as np
import torch
from matplotlib.figure import Figure

from megatransformer.utils.metrics_backend import MetricsBackend


class MetricsLogger:
    """Wraps a MetricsBackend with convenience methods.

    Handles tensor→scalar conversion, skip_zero logic, context dispatch,
    and tensor stats expansion so callers don't need to worry about
    backend details.
    """

    def __init__(self, backend: MetricsBackend):
        self._backend = backend

    # -- context dispatch ----------------------------------------------------

    def _log_context(self, tag: str, global_step: int, context: dict) -> None:
        """Log each context item as a sibling under ``tag``."""
        for key, value in context.items():
            subtag = f"{tag}/{key}"
            if isinstance(value, str):
                self._backend.add_text(subtag, value, global_step)
            elif isinstance(value, Figure):
                self._backend.add_figure(subtag, value, global_step)
            elif isinstance(value, torch.Tensor):
                v = value.detach().cpu()
                img_min, img_max = v.min(), v.max()
                if img_max > img_min:
                    v = (v - img_min) / (img_max - img_min)
                else:
                    v = torch.zeros_like(v)
                self._backend.add_image(subtag, v.numpy(), global_step)
            elif isinstance(value, np.ndarray):
                self._backend.add_image(subtag, value, global_step)
            elif isinstance(value, (int, float)):
                self._backend.add_scalar(subtag, float(value), global_step)
            elif isinstance(value, tuple) and len(value) == 2:
                # (audio_array, sample_rate)
                audio, sr = value
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().float().cpu().numpy()
                self._backend.add_audio(subtag, audio, global_step, sr)

    def _dispatch_context(self, tag: str, global_step: int, context: Optional[dict]) -> None:
        """Route context to the backend or fall back to sibling tags."""
        if not context:
            return
        if getattr(self._backend, 'supports_context', False):
            return  # backend already received context via its add_* method
        self._log_context(tag, global_step, context)

    # -- primary log methods -------------------------------------------------

    def log_scalar(
        self, tag: str, value: Union[float, torch.Tensor], global_step: int,
        skip_zero: bool = True, context: Optional[dict] = None,
    ) -> None:
        if isinstance(value, torch.Tensor):
            value = value.detach().item()
        if skip_zero and value == 0.0:
            return
        self._backend.add_scalar(tag, value, global_step, context=context)
        self._dispatch_context(tag, global_step, context)

    def log_scalars(
        self, tag_value_dict: dict, global_step: int, skip_zero: bool = True,
    ) -> None:
        for tag, value in tag_value_dict.items():
            self.log_scalar(tag, value, global_step, skip_zero)

    def log_tensor_stats(self, tag: str, tensor: torch.Tensor, global_step: int) -> None:
        t = tensor.float()
        self._backend.add_scalar(f"{tag}/mean", t.mean().item(), global_step)
        self._backend.add_scalar(f"{tag}/std", t.std().item(), global_step)
        self._backend.add_scalar(f"{tag}/min", t.min().item(), global_step)
        self._backend.add_scalar(f"{tag}/max", t.max().item(), global_step)
        self._backend.add_scalar(f"{tag}/norm", t.norm().item(), global_step)
        self._backend.add_scalar(f"{tag}/any_nan", float(t.isnan().any().item()), global_step)
        self._backend.add_scalar(f"{tag}/any_inf", float(t.isinf().any().item()), global_step)

    def log_image(
        self, tag: str, image: Union[torch.Tensor, np.ndarray], global_step: int,
        context: Optional[dict] = None,
    ) -> None:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()
            img_min = image.min()
            img_max = image.max()
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min)
            else:
                image = torch.zeros_like(image)
            image = image.numpy()
        self._backend.add_image(tag, image, global_step, context=context)
        self._dispatch_context(tag, global_step, context)

    def log_audio(
        self, tag: str, audio: Union[np.ndarray, torch.Tensor], global_step: int,
        sample_rate: int, context: Optional[dict] = None,
    ) -> None:
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().float().cpu().numpy()
        self._backend.add_audio(tag, audio, global_step, sample_rate, context=context)
        self._dispatch_context(tag, global_step, context)

    def log_figure(
        self, tag: str, figure: Figure, global_step: int,
        context: Optional[dict] = None,
    ) -> None:
        self._backend.add_figure(tag, figure, global_step, context=context)
        self._dispatch_context(tag, global_step, context)

    def log_text(
        self, tag: str, text: str, global_step: int,
        context: Optional[dict] = None,
    ) -> None:
        self._backend.add_text(tag, text, global_step, context=context)
        self._dispatch_context(tag, global_step, context)

    def log_histogram(
        self, tag: str, values: Union[torch.Tensor, np.ndarray], global_step: int,
        context: Optional[dict] = None,
    ) -> None:
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        self._backend.add_histogram(tag, values, global_step, context=context)
        self._dispatch_context(tag, global_step, context)

    def flush(self) -> None:
        self._backend.flush()

    def close(self) -> None:
        self._backend.close()


# ---------------------------------------------------------------------------
# Module-level singleton and convenience functions
# ---------------------------------------------------------------------------

_logger: Optional[MetricsLogger] = None


def init_metrics(backend: MetricsBackend) -> None:
    global _logger
    _logger = MetricsLogger(backend)


def get_logger() -> Optional[MetricsLogger]:
    return _logger


def log_scalar(tag: str, value, global_step: int, skip_zero: bool = True, context: Optional[dict] = None) -> None:
    if _logger is not None:
        _logger.log_scalar(tag, value, global_step, skip_zero, context)


def log_scalars(tag_value_dict: dict, global_step: int, skip_zero: bool = True) -> None:
    if _logger is not None:
        _logger.log_scalars(tag_value_dict, global_step, skip_zero)


def log_tensor_stats(tag: str, tensor: torch.Tensor, global_step: int) -> None:
    if _logger is not None:
        _logger.log_tensor_stats(tag, tensor, global_step)


def log_image(tag: str, image, global_step: int, context: Optional[dict] = None) -> None:
    if _logger is not None:
        _logger.log_image(tag, image, global_step, context)


def log_audio(tag: str, audio, global_step: int, sample_rate: int, context: Optional[dict] = None) -> None:
    if _logger is not None:
        _logger.log_audio(tag, audio, global_step, sample_rate, context)


def log_figure(tag: str, figure: Figure, global_step: int, context: Optional[dict] = None) -> None:
    if _logger is not None:
        _logger.log_figure(tag, figure, global_step, context)


def log_text(tag: str, text: str, global_step: int, context: Optional[dict] = None) -> None:
    if _logger is not None:
        _logger.log_text(tag, text, global_step, context)


def log_histogram(tag: str, values, global_step: int, context: Optional[dict] = None) -> None:
    if _logger is not None:
        _logger.log_histogram(tag, values, global_step, context)


def flush() -> None:
    if _logger is not None:
        _logger.flush()
