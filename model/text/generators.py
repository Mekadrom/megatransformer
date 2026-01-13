from dataclasses import dataclass
import torch
import torch.nn as nn

from model.world.transformer import MegaTransformerBlock
from utils import configuration
from utils.megatransformer_utils import transformer_weight_init


@dataclass
class TextCodaClassifierConfig:
    coda_config: configuration.TransformerBlockConfig
    vocab_size: int
    label_smoothing: float = 0.0


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
        self.coda = MegaTransformerBlock(coda_config)

        self.lm_head = nn.Linear(coda_config.d_model, config.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        self._init_weights()

    def _init_weights(self):
        self.apply(transformer_weight_init())

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None) -> dict[str, torch.Tensor]:
        """
        Takes inputs in the shape (batch_size, seq_length, d_model) and processes them through the Coda and classification head.
        It goes: Coda -> classification head -> logits over vocabulary.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).
            targets (torch.Tensor, optional): Target token IDs for computing classification loss.
        Returns:
            dict[str, torch.Tensor]: A dictionary containing the logits and classification loss if targets are provided.
        """
        coda_output = x + self.coda(x)

        logits: torch.Tensor = self.lm_head(coda_output)

        output = {"logits": logits}

        if targets is not None:
            batch_size, seq_length, vocab_size = logits.size()
            loss = self.loss_fn(
                logits.view(batch_size * seq_length, vocab_size),
                targets.view(batch_size * seq_length)
            )
            output["text_classification_loss"] = loss

        return output
