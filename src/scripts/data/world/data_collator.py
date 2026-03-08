import torch

from scripts.data.data_collator import DataCollator
from utils.megatransformer_utils import pad_and_mask, trim


class MultimodalDataCollator(DataCollator):
    """
    Data collator for multimodal training that combines text, audio, and image samples.

    Each modality is optional. Produces a single batch dict with modality-prefixed keys:
        text_*   — token_ids, text_lengths, token_masks, texts
        audio_*  — features, feature_lengths, feature_masks, mel_specs, mel_lengths,
                   mel_spec_masks, speaker_embeddings, f0, vuv, ctc_tokens, ctc_lengths,
                   ctc_masks, waveforms, waveform_lengths, waveform_masks, texts
        image_*  — images
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
        has_image = "image_image" in valid_examples[0]

        if has_text:
            batch.update(self._collate_text(valid_examples))
        if has_audio:
            batch.update(self._collate_audio(valid_examples))
        if has_image:
            batch.update(self._collate_image(valid_examples))

        return batch

    def _collate_text(self, examples: list[dict]) -> dict:
        all_token_ids = []
        all_text_lengths = []
        all_texts = []

        for ex in examples:
            all_token_ids.append(trim(ex["text_token_ids"], self.max_seq_len, dim=-1))
            all_text_lengths.append(ex["text_text_length"])
            all_texts.append(ex.get("text_text", None))

        padded_token_ids, token_masks = pad_and_mask(all_token_ids, all_text_lengths)

        return {
            "text_token_ids": torch.stack(padded_token_ids),
            "text_lengths": torch.stack(all_text_lengths),
            "text_token_masks": torch.stack(token_masks),
            "text_texts": all_texts,
        }

    def _collate_audio(self, examples: list[dict]) -> dict:
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
            all_waveforms.append(trim(ex.get("audio_waveform", None), self.max_waveforms, dim=-1))
            all_waveform_lengths.append(ex.get("audio_waveform_length", None))
            all_features.append(trim(ex.get("audio_features", None), self.max_sive_feature_frames, dim=-1))
            all_feature_lengths.append(ex.get("audio_feature_length", None))
            all_mel_specs.append(trim(ex.get("audio_mel_spec", None), self.max_mel_spec_frames, dim=-1))
            all_mel_lengths.append(ex.get("audio_mel_length", None))
            all_speaker_embeddings.append(ex.get("audio_speaker_embedding", None))
            all_speaker_ids.append(ex.get("audio_speaker_id", None))
            all_f0.append(trim(ex.get("audio_f0", None), self.max_mel_spec_frames, dim=-1))
            all_vuv.append(trim(ex.get("audio_vuv", None), self.max_mel_spec_frames, dim=-1))
            all_ctc_tokens.append(ex.get("audio_ctc_tokens", None))
            all_ctc_lengths.append(ex.get("audio_ctc_length", None))
            all_texts.append(ex.get("audio_audio_text", None))

        batch = {}

        if all_waveforms[0] is not None:
            padded, masks = pad_and_mask(all_waveforms, all_waveform_lengths)
            batch["audio_waveforms"] = torch.stack(padded)
            batch["audio_waveform_lengths"] = torch.stack(all_waveform_lengths)
            batch["audio_waveform_masks"] = torch.stack(masks)

        if all_features[0] is not None:
            padded, masks = pad_and_mask(all_features, all_feature_lengths)
            batch["audio_features"] = torch.stack(padded)
            batch["audio_feature_lengths"] = torch.stack(all_feature_lengths)
            batch["audio_feature_masks"] = torch.stack(masks)

        if all_mel_specs[0] is not None:
            padded, masks = pad_and_mask(all_mel_specs, all_mel_lengths)
            batch["audio_mel_specs"] = torch.stack(padded)
            batch["audio_mel_lengths"] = torch.stack(all_mel_lengths)
            batch["audio_mel_spec_masks"] = torch.stack(masks)

        if all_speaker_embeddings[0] is not None:
            batch["audio_speaker_embeddings"] = torch.stack(all_speaker_embeddings)

        if all_speaker_ids[0] is not None:
            batch["audio_speaker_ids"] = torch.stack(all_speaker_ids)

        if all_f0[0] is not None:
            padded_f0, _ = pad_and_mask(all_f0, all_mel_lengths)
            padded_vuv, _ = pad_and_mask(all_vuv, all_mel_lengths)
            batch["audio_f0"] = torch.stack(padded_f0)
            batch["audio_vuv"] = torch.stack(padded_vuv)

        if all_ctc_tokens[0] is not None:
            padded, masks = pad_and_mask(all_ctc_tokens, all_ctc_lengths)
            batch["audio_ctc_tokens"] = torch.stack(padded)
            batch["audio_ctc_lengths"] = torch.stack(all_ctc_lengths)
            batch["audio_ctc_masks"] = torch.stack(masks)

        batch["audio_texts"] = all_texts

        return batch

    def _collate_image(self, examples: list[dict]) -> dict:
        images = [ex["image_image"] for ex in examples]
        return {"image_images": torch.stack(images)}
