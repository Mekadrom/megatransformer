import torch
import torch.nn.functional as F


from model.audio.sive.ctc_vocab import CTCVocab
from scripts.data.preprocessor import BatchProcessor
from utils.text_encoder import TextEncoderType, get_text_encoder


class TextCTCTokensBatchProcessor(BatchProcessor):
    """Batched GPU processing for extracting CTC tokens from text."""

    def __init__(self):
        self.vocab = CTCVocab()

    @torch.no_grad()
    def process_batch(
        self,
        texts: list[str],
    ) -> dict[str, torch.Tensor]:
        """
        Process batch of texts to extract CTC tokens.

        Args:
            text: list of strings

        Returns:
            Dict with:
                - conditions: [B, T, D] text embeddings
                - conditions_lengths: [B] lengths of text embeddings
        """
        text_tokens_list = []
        text_lengths = []
        max_text_len = 0

        for text in texts:
            tokens = self.vocab.encode(text)
            text_tokens_list.append(tokens)
            text_lengths.append(len(tokens))
            max_text_len = max(max_text_len, len(tokens))

        # Pad text tokens
        text_tokens = torch.zeros(len(texts), max_text_len, dtype=torch.long)
        for i, tokens in enumerate(text_tokens_list):
            text_tokens[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)

        return {
            "ctc_tokens": text_tokens,
            "ctc_lengths": torch.tensor(text_lengths, dtype=torch.long),
        }
