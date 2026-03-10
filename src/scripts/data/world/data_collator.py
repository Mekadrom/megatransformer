import random

import torch

from scripts.data.data_collator import DataCollator
from utils.constants import (
    BOA_TOKEN_ID, EOA_TOKEN_ID,
    BOV_TOKEN_ID, EOV_TOKEN_ID,
    BOI_TOKEN_ID, EOI_TOKEN_ID,
    AUDIO_PLACEHOLDER_TOKEN_ID,
    VOICE_PLACEHOLDER_TOKEN_ID,
    IMAGE_PLACEHOLDER_TOKEN_ID,
)
from utils.megatransformer_utils import pad_and_mask, trim


class MultimodalDataCollator(DataCollator):
    """
    Data collator for multimodal training that combines text, audio, voice, and image samples.

    Handles two critical tasks:
    1. Collates per-modality tensors with padding/masking
    2. Injects boundary tokens (BOA/EOA, BOV/EOV, BOI/EOI) and placeholder tokens
       into the text sequence so that the TokenInterleaver can replace placeholders
       with media embeddings during the forward pass

    Each sample is randomly assigned a direction:
    - Synthesis (text → media):  [text_tokens] [BO*] [PLACEHOLDER] [EO*]
    - Transcription (media → text):  [BO*] [PLACEHOLDER] [EO*] [text_tokens]

    For samples with multiple media types, media blocks are chained in fixed
    order: audio, voice, image.
    """

    def __init__(
        self,
        # text
        max_seq_len: int = 2048,
        # audio
        max_waveforms: int = 160000,
        max_mel_spec_frames: int = 625,
        max_sive_feature_frames: int = 157,
    ):
        self.max_seq_len = max_seq_len
        self.max_waveforms = max_waveforms
        self.max_mel_spec_frames = max_mel_spec_frames
        self.max_sive_feature_frames = max_sive_feature_frames

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        valid_examples = [ex for ex in examples if ex is not None]
        if not valid_examples:
            return {}

        batch = {}

        # Detect which modalities are present
        has_text = "text_token_ids" in valid_examples[0]
        has_audio = any(k.startswith("audio_") for k in valid_examples[0])
        has_voice = any(k.startswith("voice_") for k in valid_examples[0])
        has_image = "image_image" in valid_examples[0]

        # Collate non-text modalities first (text needs to know which are present)
        if has_audio:
            batch.update(self._collate_audio(valid_examples))
        if has_voice:
            batch.update(self._collate_voice(valid_examples))
        if has_image:
            batch.update(self._collate_image(valid_examples))

        # Collate text with boundary/placeholder token injection
        if has_text:
            batch.update(self._collate_text(
                valid_examples,
                has_audio=has_audio,
                has_voice=has_voice,
                has_image=has_image,
            ))

        return batch

    def _build_token_sequence(
        self,
        text_token_ids: torch.Tensor,
        text_length: int,
        has_audio: bool,
        has_voice: bool,
        has_image: bool,
    ) -> torch.Tensor:
        """Build a token sequence with boundary and placeholder tokens injected.

        Randomly chooses synthesis (text → media) or transcription (media → text)
        direction. For text-only samples, returns the original token IDs unchanged.

        Returns:
            1D tensor of token IDs with boundary/placeholder tokens inserted.
        """
        # Trim text to actual length
        text_tokens = text_token_ids[:text_length]

        has_any_media = has_audio or has_voice or has_image
        if not has_any_media:
            return text_tokens

        # Build media token blocks in fixed order: audio, voice, image
        media_blocks = []
        if has_audio:
            media_blocks.append(torch.tensor(
                [BOA_TOKEN_ID, AUDIO_PLACEHOLDER_TOKEN_ID, EOA_TOKEN_ID],
                dtype=text_tokens.dtype,
            ))
        if has_voice:
            media_blocks.append(torch.tensor(
                [BOV_TOKEN_ID, VOICE_PLACEHOLDER_TOKEN_ID, EOV_TOKEN_ID],
                dtype=text_tokens.dtype,
            ))
        if has_image:
            media_blocks.append(torch.tensor(
                [BOI_TOKEN_ID, IMAGE_PLACEHOLDER_TOKEN_ID, EOI_TOKEN_ID],
                dtype=text_tokens.dtype,
            ))

        media_sequence = torch.cat(media_blocks)

        # Randomly choose direction: synthesis or transcription
        if random.random() < 0.5:
            # Synthesis: [text] [media]
            return torch.cat([text_tokens, media_sequence])
        else:
            # Transcription: [media] [text]
            return torch.cat([media_sequence, text_tokens])

    def _collate_text(
        self,
        examples: list[dict],
        has_audio: bool = False,
        has_voice: bool = False,
        has_image: bool = False,
    ) -> dict:
        all_token_ids = []
        all_text_lengths = []
        all_texts = []

        for ex in examples:
            raw_token_ids = trim(ex["text_token_ids"], self.max_seq_len, dim=-1)
            text_length = ex["text_text_length"]

            # Inject boundary + placeholder tokens
            injected = self._build_token_sequence(
                raw_token_ids,
                text_length,
                has_audio=has_audio,
                has_voice=has_voice,
                has_image=has_image,
            )

            all_token_ids.append(injected)
            all_text_lengths.append(torch.tensor(injected.shape[0], dtype=torch.long))
            all_texts.append(ex.get("text_text", None))

        padded_token_ids, token_masks = pad_and_mask(all_token_ids, all_text_lengths)

        return {
            "text_token_ids": torch.stack(padded_token_ids),
            "text_lengths": torch.stack(all_text_lengths),
            "text_token_masks": torch.stack(token_masks),
            "text_texts": all_texts,
        }

    def _collate_audio_like(self, examples: list[dict], prefix: str) -> dict:
        """Collate audio-like modality (audio or voice) with the given key prefix."""
        all_waveforms = []
        all_waveform_lengths = []
        all_features = []
        all_feature_lengths = []
        all_mel_specs = []
        all_mel_lengths = []
        all_speaker_embeddings = []
        all_speaker_ids = []
        all_f0 = []
        all_vuv = []
        all_ctc_tokens = []
        all_ctc_lengths = []
        all_texts = []

        for ex in examples:
            all_waveforms.append(trim(ex.get(f"{prefix}_waveform", None), self.max_waveforms, dim=-1))
            all_waveform_lengths.append(ex.get(f"{prefix}_waveform_length", None))
            all_features.append(trim(ex.get(f"{prefix}_features", None), self.max_sive_feature_frames, dim=-1))
            all_feature_lengths.append(ex.get(f"{prefix}_feature_length", None))
            all_mel_specs.append(trim(ex.get(f"{prefix}_mel_spec", None), self.max_mel_spec_frames, dim=-1))
            all_mel_lengths.append(ex.get(f"{prefix}_mel_length", None))
            all_speaker_embeddings.append(ex.get(f"{prefix}_speaker_embedding", None))
            all_speaker_ids.append(ex.get(f"{prefix}_speaker_id", None))
            all_f0.append(trim(ex.get(f"{prefix}_f0", None), self.max_mel_spec_frames, dim=-1))
            all_vuv.append(trim(ex.get(f"{prefix}_vuv", None), self.max_mel_spec_frames, dim=-1))
            all_ctc_tokens.append(ex.get(f"{prefix}_ctc_tokens", None))
            all_ctc_lengths.append(ex.get(f"{prefix}_ctc_length", None))
            # Text key differs between audio and voice in dataset
            text_key = f"{prefix}_audio_text" if prefix == "audio" else f"{prefix}_voice_text"
            all_texts.append(ex.get(text_key, None))

        batch = {}

        if all_waveforms[0] is not None:
            padded, masks = pad_and_mask(all_waveforms, all_waveform_lengths)
            batch[f"{prefix}_waveforms"] = torch.stack(padded)
            batch[f"{prefix}_waveform_lengths"] = torch.stack(all_waveform_lengths)
            batch[f"{prefix}_waveform_masks"] = torch.stack(masks)

        if all_features[0] is not None:
            padded, masks = pad_and_mask(all_features, all_feature_lengths)
            batch[f"{prefix}_features"] = torch.stack(padded)
            batch[f"{prefix}_feature_lengths"] = torch.stack(all_feature_lengths)
            batch[f"{prefix}_feature_masks"] = torch.stack(masks)

        if all_mel_specs[0] is not None:
            padded, masks = pad_and_mask(all_mel_specs, all_mel_lengths)
            batch[f"{prefix}_mel_specs"] = torch.stack(padded)
            batch[f"{prefix}_mel_lengths"] = torch.stack(all_mel_lengths)
            batch[f"{prefix}_mel_spec_masks"] = torch.stack(masks)

        if all_speaker_embeddings[0] is not None:
            batch[f"{prefix}_speaker_embeddings"] = torch.stack(all_speaker_embeddings)

        if all_speaker_ids[0] is not None:
            batch[f"{prefix}_speaker_ids"] = torch.stack(all_speaker_ids)

        if all_f0[0] is not None:
            padded_f0, _ = pad_and_mask(all_f0, all_mel_lengths)
            padded_vuv, _ = pad_and_mask(all_vuv, all_mel_lengths)
            batch[f"{prefix}_f0"] = torch.stack(padded_f0)
            batch[f"{prefix}_vuv"] = torch.stack(padded_vuv)

        if all_ctc_tokens[0] is not None:
            padded, masks = pad_and_mask(all_ctc_tokens, all_ctc_lengths)
            batch[f"{prefix}_ctc_tokens"] = torch.stack(padded)
            batch[f"{prefix}_ctc_lengths"] = torch.stack(all_ctc_lengths)
            batch[f"{prefix}_ctc_masks"] = torch.stack(masks)

        batch[f"{prefix}_texts"] = all_texts

        return batch

    def _collate_audio(self, examples: list[dict]) -> dict:
        return self._collate_audio_like(examples, "audio")

    def _collate_voice(self, examples: list[dict]) -> dict:
        return self._collate_audio_like(examples, "voice")

    def _collate_image(self, examples: list[dict]) -> dict:
        images = [ex["image_image"] for ex in examples]
        return {"image_images": torch.stack(images)}
