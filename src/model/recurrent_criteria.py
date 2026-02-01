import torch
import torch.nn.functional as F


from typing import Optional


class RecurrentExitCriteria:
    def should_exit(self, last_thought_state: torch.Tensor, current_thought_state: torch.Tensor):
        raise NotImplementedError


class KLDivergenceCriteria(RecurrentExitCriteria):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def should_exit(self, last_thought_state: Optional[torch.Tensor], current_thought_state: Optional[torch.Tensor]):
        if last_thought_state is None or current_thought_state is None:
            return False

        kl_divergence = F.kl_div(last_thought_state, current_thought_state, reduction="none", log_target=True).sum(dim=-1)
        return (kl_divergence < self.threshold).any()
