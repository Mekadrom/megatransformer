import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    """
    SpecAugment: A Simple Data Augmentation Method for ASR.

    Applies time and frequency masking to mel spectrograms during training.
    Reference: https://arxiv.org/abs/1904.08779

    Args:
        time_mask_param: Maximum width of time mask (T in paper)
        freq_mask_param: Maximum width of frequency mask (F in paper)
        num_time_masks: Number of time masks to apply
        num_freq_masks: Number of frequency masks to apply
        mask_value: Value to fill masked regions (0.0 or mean)
    """

    def __init__(
        self,
        time_mask_param: int = 50,
        freq_mask_param: int = 20,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
        mask_value: float = 0.0,
    ):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.mask_value = mask_value

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to mel spectrogram.

        Args:
            mel_spec: [B, n_mels, T] mel spectrogram

        Returns:
            Augmented mel spectrogram [B, n_mels, T]
        """
        if not self.training:
            return mel_spec

        # Clone to avoid modifying input
        mel_spec = mel_spec.clone()

        batch_size, n_mels, time_steps = mel_spec.shape

        # Apply frequency masks
        for _ in range(self.num_freq_masks):
            mel_spec = self._apply_freq_mask(mel_spec, n_mels)

        # Apply time masks
        for _ in range(self.num_time_masks):
            mel_spec = self._apply_time_mask(mel_spec, time_steps)

        return mel_spec

    def _apply_freq_mask(self, mel_spec: torch.Tensor, n_mels: int) -> torch.Tensor:
        """Apply frequency masking to each sample in batch."""
        batch_size = mel_spec.size(0)

        # Sample mask width for each batch element
        f = torch.randint(0, min(self.freq_mask_param, n_mels) + 1, (batch_size,), device=mel_spec.device)
        f0 = torch.stack([
            torch.randint(0, max(1, n_mels - f[i].item() + 1), (1,), device=mel_spec.device).squeeze()
            for i in range(batch_size)
        ])

        # Apply masks per batch element
        for i in range(batch_size):
            f_start = f0[i].item()
            f_end = f_start + f[i].item()
            mel_spec[i, f_start:f_end, :] = self.mask_value

        return mel_spec

    def _apply_time_mask(self, mel_spec: torch.Tensor, time_steps: int) -> torch.Tensor:
        """Apply time masking to each sample in batch."""
        batch_size = mel_spec.size(0)

        # Sample mask width for each batch element
        t = torch.randint(0, min(self.time_mask_param, time_steps) + 1, (batch_size,), device=mel_spec.device)
        t0 = torch.stack([
            torch.randint(0, max(1, time_steps - t[i].item() + 1), (1,), device=mel_spec.device).squeeze()
            for i in range(batch_size)
        ])

        # Apply masks per batch element
        for i in range(batch_size):
            t_start = t0[i].item()
            t_end = t_start + t[i].item()
            mel_spec[i, :, t_start:t_end] = self.mask_value

        return mel_spec