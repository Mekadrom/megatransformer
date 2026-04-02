from typing import Protocol, Optional, runtime_checkable

import numpy as np
import torch
from matplotlib.figure import Figure


@runtime_checkable
class MetricsBackend(Protocol):
    """Backend protocol for metrics logging.

    Implementations wrap a specific logging framework (TensorBoard, W&B, etc.).
    All methods accept pre-processed Python/numpy data — tensor conversion
    happens in MetricsLogger before reaching the backend.
    """

    def add_scalar(self, tag: str, value: float, step: int) -> None: ...
    def add_image(self, tag: str, img_array: np.ndarray, step: int) -> None: ...
    def add_audio(self, tag: str, audio_array: np.ndarray, step: int, sample_rate: int) -> None: ...
    def add_figure(self, tag: str, figure: Figure, step: int) -> None: ...
    def add_text(self, tag: str, text: str, step: int) -> None: ...
    def add_histogram(self, tag: str, values: np.ndarray, step: int) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class TensorBoardBackend:
    """Wraps torch.utils.tensorboard.SummaryWriter."""

    def __init__(self, writer=None, log_dir: Optional[str] = None):
        if writer is not None:
            self._writer = writer
        else:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=log_dir)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self._writer.add_scalar(tag, value, step)

    def add_image(self, tag: str, img_array: np.ndarray, step: int) -> None:
        self._writer.add_image(tag, img_array, step)

    def add_audio(self, tag: str, audio_array: np.ndarray, step: int, sample_rate: int) -> None:
        self._writer.add_audio(tag, audio_array, step, sample_rate=sample_rate)

    def add_figure(self, tag: str, figure: Figure, step: int) -> None:
        self._writer.add_figure(tag, figure, step)

    def add_text(self, tag: str, text: str, step: int) -> None:
        self._writer.add_text(tag, text, step)

    def add_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        self._writer.add_histogram(tag, values, step)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


class NoOpBackend:
    """Silent backend for non-rank-zero processes in distributed training."""

    def add_scalar(self, tag, value, step): pass
    def add_image(self, tag, img_array, step): pass
    def add_audio(self, tag, audio_array, step, sample_rate): pass
    def add_figure(self, tag, figure, step): pass
    def add_text(self, tag, text, step): pass
    def add_histogram(self, tag, values, step): pass
    def flush(self): pass
    def close(self): pass
