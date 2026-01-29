import torch
import torch.nn.functional as F


from scripts.data.preprocessor import BatchProcessor
from utils.text_encoder import TextEncoderType, get_text_encoder


class TextConditionsBatchProcessor(BatchProcessor):
    """Batched GPU processing for extracting F0 and VUV labels."""

    def __init__(
        self,
        text_embedding_model: TextEncoderType = "t5_small",
        device: str = "cuda",
    ):
        self.model = get_text_encoder(text_embedding_model, device=device)

    @torch.no_grad()
    def process_batch(
        self,
        text: list[str],
    ) -> dict[str, torch.Tensor]:
        """
        Process batch of waveforms to extract speaker embeddings, F0 and VUV labels.

        Args:
            text: list of strings

        Returns:
            Dict with:
                - conditions: [B, T, D] text embeddings
                - conditions_lengths: [B] lengths of text embeddings
        """
        embeddings = self.model.encode(text, return_attention_mask=False)
        conditions_lengths = torch.tensor([e.size(0) for e in embeddings], device=embeddings.device)

        return {
            "conditions": embeddings,  # [B, T, D]
            "conditions_lengths": conditions_lengths,  # [B]
        }
