import abc

import torch


class DataCollator(abc.ABC):
    """
    Base class for data collators.
    """
    @abc.abstractmethod
    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        pass
