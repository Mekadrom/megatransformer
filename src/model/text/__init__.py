from dataclasses import dataclass
import torch
import torch.nn as nn

from model.norms import RMSNorm
from utils.megatransformer_utils import embedding_weight_init
from utils.model_utils import create_norm


class TextFeatureExtractor(nn.Module):
    """
    Embeds text token IDs into dense vectors for the multimodal world model.

    Unlike audio and image modalities, text has no prelude transformer because
    processing text tokens in isolation (without interleaved modalities) provides
    limited benefit. The text embeddings are designed to be immediately interleaved
    with media embeddings via TokenInterleaver before entering the main transformer.

    The embedding includes optional layer normalization and dropout for regularization.
    """

    def __init__(self, config: TextFeatureExtractorConfig):
        super().__init__()

        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.norm = create_norm(config.d_model, config.norm_type, config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

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
        hidden_states = self.wte(input_ids)
        hidden_states = self.norm(hidden_states)
        return self.dropout(hidden_states)
