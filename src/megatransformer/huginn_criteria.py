from typing import Optional

import torch

class HuginnExitCriteria:
    def should_exit(self, last_thought_state: torch.Tensor, current_thought_state: torch.Tensor):
        raise NotImplementedError

class KLDivergenceCriteria(HuginnExitCriteria):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def should_exit(self, last_thought_state: Optional[torch.Tensor], current_thought_state: Optional[torch.Tensor]):
        if last_thought_state is None or current_thought_state is None:
            return False

        kl_divergence = torch.kl_div(last_thought_state, current_thought_state)
        return kl_divergence < self.threshold
