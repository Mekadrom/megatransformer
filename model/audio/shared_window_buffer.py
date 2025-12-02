import torch


class SharedWindowBuffer:
    def __init__(self):
        self.cache = {}

    def get_window(self, window_size: int, device: torch.device) -> torch.Tensor:
        if window_size not in self.cache:
            window = torch.hann_window(window_size, device=device)
            self.cache[window_size] = window
        return self.cache[window_size].to(device)
