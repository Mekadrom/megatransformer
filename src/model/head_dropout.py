import torch
import torch.nn as nn


class HeadDropout(nn.Module):
    """
    DropHead: Randomly drops entire attention heads during training.

    Reference: "Scheduled DropHead" (Zhou et al., 2020)

    Args:
        num_heads: Number of attention heads
        head_dim: Dimension per head (d_model // num_heads)
        drop_prob: Probability of dropping each head (0.0 = no dropout)
    """

    def __init__(self, num_heads: int, head_dim: int, drop_prob: float = 0.0):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply head dropout to attention output.

        Args:
            x: [B, T, D] attention output where D = num_heads * head_dim

        Returns:
            [B, T, D] with randomly dropped heads (scaled appropriately)
        """
        if not self.training or self.drop_prob == 0.0:
            return x

        B, T, D = x.shape

        # Reshape to expose heads: [B, T, num_heads, head_dim]
        x = x.view(B, T, self.num_heads, self.head_dim)

        # Generate head mask: [B, 1, num_heads, 1] - same mask for all timesteps
        # Each head is independently dropped with probability drop_prob
        head_mask = torch.bernoulli(
            torch.full((B, 1, self.num_heads, 1), 1.0 - self.drop_prob, device=x.device)
        )

        # Ensure at least one head is kept (avoid all-zeros)
        # If all heads would be dropped, keep a random one
        all_dropped = (head_mask.sum(dim=2, keepdim=True) == 0)
        if all_dropped.any():
            # Pick a random head to keep for each sample where all were dropped
            random_head = torch.randint(0, self.num_heads, (B, 1, 1, 1), device=x.device)
            keep_mask = (torch.arange(self.num_heads, device=x.device).view(1, 1, -1, 1) == random_head)
            head_mask = torch.where(all_dropped, keep_mask.float(), head_mask)

        # Apply mask and scale (like standard dropout)
        # Scale by 1/(1-p) to maintain expected value, accounting for kept heads
        keep_prob = head_mask.mean(dim=2, keepdim=True).clamp(min=1e-6)
        x = x * head_mask / keep_prob

        # Reshape back: [B, T, D]
        return x.view(B, T, D)

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, head_dim={self.head_dim}, drop_prob={self.drop_prob}"
