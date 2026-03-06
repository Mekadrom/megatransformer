import torch

from scripts.data.data_collator import DataCollator
from utils.megatransformer_utils import pad_and_mask, trim


class TextDataCollator(DataCollator):
    """
    Data collator for text-only training.

    Pads token_ids to same length within batch and creates masks.
    """

    def __init__(
        self,
        max_seq_len: int = 2048,
    ):
        self.max_seq_len = max_seq_len

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        valid_examples = [ex for ex in examples if ex is not None]
        if not valid_examples:
            return {}

        all_token_ids = []
        all_text_lengths = []
        all_texts = []

        for ex in valid_examples:
            all_token_ids.append(trim(ex["token_ids"], self.max_seq_len, dim=-1))
            all_text_lengths.append(ex["text_length"])
            all_texts.append(ex.get("text", None))

        padded_token_ids, token_masks = pad_and_mask(all_token_ids, all_text_lengths)

        batch = {
            "token_ids": torch.stack(padded_token_ids),      # [B, T]
            "text_lengths": torch.stack(all_text_lengths),    # [B]
            "token_masks": torch.stack(token_masks),          # [B, T]
            "texts": all_texts,                               # list of strings
        }

        return batch
