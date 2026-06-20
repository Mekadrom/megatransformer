from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial training.
    Forward: identity
    Backward: negates gradients scaled by alpha
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.alpha * grad_output, None


class SpeakerClassifier(nn.Module):
    """
    Speaker classifier head with modern pooling strategies.
    Used with gradient reversal to push speaker info out of features.

    Pooling options:
    - "mean": Simple mean pooling
    - "statistics": Mean + std concatenation (2x input dim)
    - "attention": Learnable attention weights
    - "attentive_statistics": ASP from ECAPA-TDNN (attention-weighted mean + std)
    - "multi_head_attention": Multi-head self-attention pooling
    """

    def __init__(
        self,
        d_model: int,
        num_speakers: int,
        hidden_dim: Optional[int] = None,
        pooling: str = "attentive_statistics",
        dropout: float = 0.1,
        num_attention_heads: int = 4,
    ):
        super().__init__()

        hidden_dim = hidden_dim or d_model * 2
        self.pooling = pooling
        self.d_model = d_model

        # Determine pooled dimension based on pooling type
        if pooling in ("statistics", "attentive_statistics"):
            pooled_dim = d_model * 2  # mean + std
        else:
            pooled_dim = d_model

        # Pooling-specific layers
        if pooling == "attention":
            self.attn_pool = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Tanh(),
                nn.Linear(d_model, 1),
            )

        elif pooling == "attentive_statistics":
            # Attentive Statistics Pooling (ASP) from ECAPA-TDNN
            # Uses attention to compute weighted mean and std
            self.asp_linear = nn.Linear(d_model, d_model)
            self.asp_attention = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm1d(d_model),
                nn.Conv1d(d_model, d_model, kernel_size=1),
                nn.Softmax(dim=2),
            )

        elif pooling == "multi_head_attention":
            # Learnable query for pooling
            self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
            self.mha_pool = nn.MultiheadAttention(
                d_model, num_attention_heads, dropout=dropout, batch_first=True
            )
            self.mha_norm = nn.LayerNorm(d_model)

        # Classifier MLP
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_speakers),
        )

    def _statistics_pooling(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute mean and std pooling."""
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()  # [B, T, 1]
            lengths = mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]

            # Masked mean
            mean = (x * mask_expanded).sum(dim=1) / lengths

            # Masked std
            diff_sq = ((x - mean.unsqueeze(1)) ** 2) * mask_expanded
            var = diff_sq.sum(dim=1) / lengths.clamp(min=1)
            std = (var + 1e-6).sqrt()
        else:
            mean = x.mean(dim=1)
            std = x.std(dim=1) + 1e-6

        return torch.cat([mean, std], dim=-1)  # [B, 2*D]

    def _attentive_statistics_pooling(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Attentive Statistics Pooling (ASP) from ECAPA-TDNN.
        Computes attention-weighted mean and std.
        """
        # x: [B, T, D]
        h = self.asp_linear(x)  # [B, T, D]

        # Compute attention weights using Conv1d (expects [B, D, T])
        h_transposed = h.transpose(1, 2)  # [B, D, T]
        attn_weights = self.asp_attention(h_transposed)  # [B, D, T] with softmax over T
        attn_weights = attn_weights.transpose(1, 2)  # [B, T, D]

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()  # [B, T, 1]
            attn_weights = attn_weights * mask_expanded
            # Re-normalize after masking
            attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

        # Weighted mean
        weighted_mean = (x * attn_weights).sum(dim=1)  # [B, D]

        # Weighted std
        diff_sq = (x - weighted_mean.unsqueeze(1)) ** 2
        weighted_var = (diff_sq * attn_weights).sum(dim=1)
        weighted_std = (weighted_var + 1e-6).sqrt()  # [B, D]

        return torch.cat([weighted_mean, weighted_std], dim=-1)  # [B, 2*D]

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] features
            mask: [B, T] True for valid positions
        Returns:
            [B, num_speakers] speaker logits
        """
        if self.pooling == "mean":
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled = x.mean(dim=1)

        elif self.pooling == "max":
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            pooled = x.max(dim=1)[0]

        elif self.pooling == "statistics":
            pooled = self._statistics_pooling(x, mask)

        elif self.pooling == "attentive_statistics":
            pooled = self._attentive_statistics_pooling(x, mask)

        elif self.pooling == "attention":
            attn_weights = self.attn_pool(x).squeeze(-1)  # [B, T]
            if mask is not None:
                attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1)  # [B, T]
            pooled = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]

        elif self.pooling == "multi_head_attention":
            B = x.size(0)
            query = self.pool_query.expand(B, -1, -1)  # [B, 1, D]

            # Create key_padding_mask (True = ignore)
            key_padding_mask = ~mask if mask is not None else None

            pooled, _ = self.mha_pool(query, x, x, key_padding_mask=key_padding_mask)
            pooled = self.mha_norm(pooled.squeeze(1))  # [B, D]

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")

        return self.classifier(pooled)
