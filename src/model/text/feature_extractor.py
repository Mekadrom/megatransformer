import torch
import torch.nn as nn

from model.norms import create_norm
from model.sinusoidal_positional_encoding import SinusoidalPositionalEncoding
from model.transformer import MegaTransformerBlock
from utils import megatransformer_utils
from utils.megatransformer_utils import embedding_weight_init


import torch
import torch.nn as nn

from config.text.feature_extractor import TextPreludeFeatureExtractorConfig
from utils.megatransformer_utils import embedding_weight_init


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
            MegaTransformerBlock(prelude_config)
            for _ in range(config.n_layers)
        ])

        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=prelude_config.d_model,
            max_len=config.max_position_embeddings * 2 + 1,
            dropout=0.0
        )

        self.norm = create_norm(config.d_model, config.norm_type, config.layer_norm_epsilon)

        self._init_weights()

    def _init_weights(self):
        self.apply(embedding_weight_init(self.config.d_model))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed text token IDs into dense vectors.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).

        Returns:
            Hidden states of shape (batch_size, seq_len, d_model).
        """
        x = self.wte(input_ids)

        megatransformer_utils.print_debug_tensor("embedding text prelude output", x)

        projected = self.pos_encoding(x)

        megatransformer_utils.print_debug_tensor("positional encoding text prelude output", projected)

        x = projected
        for block in self.prelude:
            hidden, _ = block(x)
            x = x + hidden

        megatransformer_utils.print_debug_tensor("prelude block text prelude output", x)
        
        normed = self.norm(x)

        megatransformer_utils.print_debug_tensor("normed text prelude output", normed)

        return normed
