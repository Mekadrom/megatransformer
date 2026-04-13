from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn as nn

from config.text.generator import TextCodaClassifierConfig
from model.norms import create_norm
from model.transformer import MegaTransformerEncoderBlock
from utils.megatransformer_utils import (
    apply_depth_scaled_residual_init,
    linear_weight_init,
)


class TextCodaClassifierWithLoss(nn.Module):
    """
    Text output head for the multimodal world model.

    Processes hidden states from the main transformer through a modality-specific
    coda (a small transformer stack) and projects to vocabulary logits for next-token
    prediction. The coda allows the model to specialize its final processing for
    text generation separately from other modalities.

    The architecture uses a residual connection around the coda transformer.
    """

    def __init__(self, config: TextCodaClassifierConfig):
        super(TextCodaClassifierWithLoss, self).__init__()

        self.config = config
        coda_config = config.coda_config

        if config.use_input_norm:
            self.input_norm = create_norm(coda_config.d_model, config.input_norm_type, config.norm_epsilon)

        self.coda = nn.ModuleList([
            MegaTransformerEncoderBlock(coda_config)
            for _ in range(config.n_layers)
        ])

        self.lm_head = nn.Linear(coda_config.d_model, config.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        self._init_weights()

    def _init_weights(self):
        # Standard Xavier on every Linear (coda blocks + lm_head), then
        # depth-scaled init for the coda blocks' residual outputs.
        self.apply(linear_weight_init(gain=1.0))
        apply_depth_scaled_residual_init(self.coda)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List] = None,
        position_offset: int = 0,
        use_cache: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Takes inputs in the shape (batch_size, seq_length, d_model) and processes them through the Coda and classification head.

        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model).
            targets: Target token IDs for computing classification loss.
            kv_caches: Optional list of KVCache objects, one per coda layer.
                When provided, the coda's self-attention attends to cached
                representations from previous positions (inference only).
            position_offset: RoPE position offset for cached generation.
            use_cache: If True, return updated KV caches in the output dict.

        Returns:
            dict with "logits" and optionally "text_classification_loss" and "kv_caches".
        """

        if hasattr(self, 'input_norm'):
            x = self.input_norm(x)

        # MegaTransformerEncoderBlock.forward already adds residuals internally,
        # so the loop just chains layers without re-adding the input.
        h = x
        new_kv_caches = []
        for i, block in enumerate(self.coda):
            block_cache = kv_caches[i] if kv_caches is not None else None
            h, new_cache = block(
                h,
                kv_cache=block_cache,
                position_offset=position_offset,
                use_cache=use_cache,
            )
            if use_cache:
                new_kv_caches.append(new_cache)

        logits: torch.Tensor = self.lm_head(h)

        output = {"logits": logits}
        if use_cache:
            output["kv_caches"] = new_kv_caches

        if targets is not None:
            batch_size, seq_length, vocab_size = logits.size()
            loss = self.loss_fn(
                logits.view(batch_size * seq_length, vocab_size),
                targets.view(batch_size * seq_length)
            )
            output["text_classification_loss"] = loss

        return output
