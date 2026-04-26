from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from config.text.feature_extractor import TextPreludeFeatureExtractorConfig
from model.norms import create_norm
from model.sinusoidal_positional_encoding import SinusoidalPositionalEncoding
from model.transformer import MegaTransformerEncoderBlock
from utils import megatransformer_utils
from utils.megatransformer_utils import (
    apply_depth_scaled_residual_init,
    embedding_weight_init,
    linear_weight_init,
)


class TextPreludeFeatureExtractor(nn.Module):
    """
    Embeds text token IDs into dense vectors for the multimodal world model.

    Unlike audio and image modalities, text has no prelude transformer because
    processing text tokens in isolation (without interleaved modalities) provides
    limited benefit. The text embeddings are designed to be immediately interleaved
    with media embeddings via TokenInterleaver before entering the main transformer.

    The embedding includes optional layer normalization and dropout for regularization.
    """

    def __init__(self, config: TextPreludeFeatureExtractorConfig):
        super().__init__()

        self.config = config
        prelude_config = config.prelude_config

        self.wte = nn.Embedding(config.vocab_size, config.d_model)

        self.prelude = nn.ModuleList([
            MegaTransformerEncoderBlock(prelude_config)
            for _ in range(config.n_layers)
        ])

        if (not prelude_config.use_rotary_embedding) or config.use_pos_emb_ovr:
            self.pos_encoding = SinusoidalPositionalEncoding(
                d_model=prelude_config.d_model,
                max_len=config.max_position_embeddings * 2 + 1,
                dropout=0.0
            )

        if config.use_output_norm:
            self.output_norm = create_norm(config.d_model, config.output_norm_type, config.norm_epsilon)

        self.gradient_checkpointing = False
        self._init_weights()

    def _init_weights(self):
        # Embeddings: N(0, 1/sqrt(d_model)) so embedding magnitude is ~1.
        self.apply(embedding_weight_init(self.config.d_model))
        # Prelude transformer: standard Xavier on linears + depth-scaled
        # residual outputs. Previously this was missing entirely (linear
        # layers fell through to PyTorch's kaiming_uniform default).
        init_linear = linear_weight_init(gain=1.0)
        for block in self.prelude:
            block.apply(init_linear)
        apply_depth_scaled_residual_init(self.prelude)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[List] = None,
        position_offset: int = 0,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        Embed text token IDs into dense vectors.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            kv_caches: Optional list of KVCache objects, one per prelude layer.
                Used for autoregressive generation so token t's prelude
                self-attention sees tokens 0..t-1 (matching training).
            position_offset: Position offset for sinusoidal PE and RoPE when
                using KV caching.
            use_cache: If True, return (embeddings, kv_caches) tuple.

        Returns:
            Hidden states of shape (batch_size, seq_len, d_model).
            If use_cache=True, returns (hidden_states, new_kv_caches) tuple.
        """
        projected = self.wte(input_ids)

        if hasattr(self, 'pos_encoding'):
            projected = self.pos_encoding(projected, offset=position_offset)

        # MegaTransformerEncoderBlock.forward already adds residuals internally,
        # so the loop just chains layers without re-adding the input.
        x = projected
        new_kv_caches = []
        for i, block in enumerate(self.prelude):
            block_cache = kv_caches[i] if kv_caches is not None else None
            if self.gradient_checkpointing and self.training and not use_cache:
                x, new_cache = torch_checkpoint(
                    block, x, None, None, block_cache, position_offset, use_cache,
                    use_reentrant=False,
                )
            else:
                x, new_cache = block(
                    x,
                    kv_cache=block_cache,
                    position_offset=position_offset,
                    use_cache=use_cache,
                )
            if use_cache:
                new_kv_caches.append(new_cache)

        if hasattr(self, 'output_norm'):
            x = self.output_norm(x)

        if use_cache:
            return x, new_kv_caches
        return x
