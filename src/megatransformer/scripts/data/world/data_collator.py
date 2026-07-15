import random

import torch

from megatransformer.scripts.data.data_collator import DataCollator
from megatransformer.utils.constants import (
    BOA_TOKEN_ID, EOA_TOKEN_ID,
    BOV_TOKEN_ID, EOV_TOKEN_ID,
    BOI_TOKEN_ID, EOI_TOKEN_ID,
    AUDIO_PLACEHOLDER_TOKEN_ID,
    VOICE_PLACEHOLDER_TOKEN_ID,
    IMAGE_PLACEHOLDER_TOKEN_ID,
    EOS_TOKEN_ID,
)
from megatransformer.utils.megatransformer_utils import pad_and_mask, trim


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
        self.force_direction = None  # Set to "synthesis" or "transcription" to override random direction

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        valid_examples = [ex for ex in examples if ex is not None]
        if not valid_examples:
            return {}

        batch = {}

        # Detect which modalities are present. Check all samples (not just first)
        # to handle mixed batches from eval dataloaders.
        has_text = any("text_token_ids" in ex for ex in valid_examples)
        has_audio = any(any(k.startswith("audio_") for k in ex) for ex in valid_examples)
        has_voice = any(any(k.startswith("voice_") for k in ex) for ex in valid_examples)
        has_image = any("image_image" in ex for ex in valid_examples)

        # Collate non-text modalities (only from samples that have them)
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
                force_direction=self.force_direction,
            ))

        return batch

    def _build_token_sequence(
        self,
        text_token_ids: torch.Tensor,
        text_length: int,
        has_audio: bool,
        has_voice: bool,
        has_image: bool,
        force_direction: str = None,  # None=random, "synthesis", "transcription"
    ) -> torch.Tensor:
        """Build a token sequence with boundary and placeholder tokens injected.

        Randomly chooses synthesis (text → media) or transcription (media → text)
        direction. For text-only samples, returns the original token IDs unchanged.

        Returns:
            1D tensor of token IDs with boundary/placeholder tokens inserted.
        """
        # Trim text to actual length (strips any preprocessor padding)
        text_tokens = text_token_ids[:text_length]
        eos = torch.tensor([EOS_TOKEN_ID], dtype=text_tokens.dtype)

        # Only append EOS when the text wasn't truncated — truncated samples
        # were cut off mid-content and didn't genuinely end.
        text_truncated = text_length >= self.max_seq_len

        has_any_media = has_audio or has_voice or has_image
        if not has_any_media:
            if text_truncated:
                return text_tokens, False
            return torch.cat([text_tokens, eos]), False

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

        # Choose direction: synthesis or transcription
        if force_direction == "synthesis":
            is_synthesis = True
        elif force_direction == "transcription":
            is_synthesis = False
        else:
            is_synthesis = random.random() < 0.5
        if is_synthesis:
            # Synthesis: [text] [media] [EOS]
            return torch.cat([text_tokens, media_sequence, eos]), is_synthesis
        else:
            # Transcription: [media] [text] [EOS]
            return torch.cat([media_sequence, text_tokens, eos]), is_synthesis

    def _collate_text(
        self,
        examples: list[dict],
        has_audio: bool = False,
        has_voice: bool = False,
        has_image: bool = False,
        force_direction: str = None,
    ) -> dict:
        all_token_ids = []
        all_text_lengths = []
        all_texts = []
        all_is_synthesis = []

        for ex in examples:
            raw_token_ids = trim(ex["text_token_ids"], self.max_seq_len, dim=-1)
            text_length = ex["text_text_length"]

            # Use sample's _direction if available, otherwise use force_direction or random
            direction = force_direction or ex.get("_direction", None)

            # Inject boundary + placeholder tokens
            injected, is_synthesis = self._build_token_sequence(
                raw_token_ids,
                text_length,
                has_audio=has_audio,
                has_voice=has_voice,
                has_image=has_image,
                force_direction=direction,
            )

            all_token_ids.append(injected)
            all_text_lengths.append(torch.tensor(injected.shape[0], dtype=torch.long))
            all_texts.append(ex.get("text_text", None))
            all_is_synthesis.append(is_synthesis)

        padded_token_ids, token_masks = pad_and_mask(all_token_ids, all_text_lengths)

        return {
            "text_token_ids": torch.stack(padded_token_ids),
            "text_lengths": torch.stack(all_text_lengths),
            "text_token_masks": torch.stack(token_masks),
            "text_texts": all_texts,
            # Per-sample flag: True = synthesis (text→media), False = transcription (media→text)
            "is_synthesis": torch.tensor(all_is_synthesis, dtype=torch.bool),
        }

    def _collate_text_per_sample(
        self,
        examples: list[dict],
        per_sample_has_audio: list[bool],
        per_sample_has_voice: list[bool],
        per_sample_has_image: list[bool],
        force_direction: str = None,
    ) -> dict:
        """Collate text with per-sample modality awareness.

        Each sample gets placeholder tokens only for its own modalities.
        Text-only samples get no placeholders. Voice samples get BOV/VOICE_PH/EOV.
        Image samples get BOI/IMAGE_PH/EOI.
        """
        all_token_ids = []
        all_text_lengths = []
        all_texts = []
        all_is_synthesis = []

        for i, ex in enumerate(examples):
            if "text_token_ids" not in ex:
                # Skip samples without text (shouldn't happen, but defensive)
                continue

            raw_token_ids = trim(ex["text_token_ids"], self.max_seq_len, dim=-1)
            text_length = ex.get("text_text_length", raw_token_ids.shape[0])
            if isinstance(text_length, torch.Tensor):
                text_length = text_length.item()

            injected, is_synthesis = self._build_token_sequence(
                raw_token_ids,
                text_length,
                has_audio=per_sample_has_audio[i],
                has_voice=per_sample_has_voice[i],
                has_image=per_sample_has_image[i],
                force_direction=force_direction,
            )

            all_token_ids.append(injected)
            all_text_lengths.append(torch.tensor(injected.shape[0], dtype=torch.long))
            all_texts.append(ex.get("text_text", None))
            all_is_synthesis.append(is_synthesis)

        padded_token_ids, token_masks = pad_and_mask(all_token_ids, all_text_lengths)

        return {
            "text_token_ids": torch.stack(padded_token_ids),
            "text_lengths": torch.stack(all_text_lengths),
            "text_token_masks": torch.stack(token_masks),
            "text_texts": all_texts,
            "is_synthesis": torch.tensor(all_is_synthesis, dtype=torch.bool),
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
        all_unit_ids = []

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
            all_unit_ids.append(trim(ex.get(f"{prefix}_unit_ids", None), self.max_sive_feature_frames, dim=-1))
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

        if all_unit_ids[0] is not None:
            # Pad with -100, NOT 0: 0 is a real unit id. pad_and_mask() pads with 0
            # (it is the shared waveform/mel helper), which would silently supervise the
            # coda to predict unit 0 across every padded frame — the exact bug the text
            # targets had. Pad by hand so padding is the CE ignore_index.
            T = max(int(u.shape[-1]) for u in all_unit_ids)
            padded_units = []
            for u, n in zip(all_unit_ids, all_feature_lengths):
                n = int(n)
                out = torch.full((T,), -100, dtype=torch.long)
                out[:n] = u[:n].to(torch.long)
                padded_units.append(out)
            batch[f"{prefix}_unit_ids"] = torch.stack(padded_units)

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
        filtered = [ex for ex in examples if any(k.startswith("audio_") for k in ex)]
        return self._collate_audio_like(filtered, "audio") if filtered else {}

    def _collate_voice(self, examples: list[dict]) -> dict:
        filtered = [ex for ex in examples if any(k.startswith("voice_") for k in ex)]
        return self._collate_audio_like(filtered, "voice") if filtered else {}

    def _collate_image(self, examples: list[dict]) -> dict:
        images = [ex["image_image"] for ex in examples if "image_image" in ex]
        if not images:
            return {}
        return {"image_images": torch.stack(images)}
