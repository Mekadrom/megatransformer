import torch
import torch.nn.functional as F

from scripts.data.data_collator import DataCollator
from utils.megatransformer_utils import pad_and_mask, trim, print_debug_tensor

# Set to True to enable debug printing in data collator
DEBUG_COLLATOR = False  # Set True to debug data loading issues


class AudioDataCollator(DataCollator):
    """
    Data collator for any audio training.

    Pads features, waveforms, and mel specs to same length within batch and creates masks.
    """

    def __init__(
        self,
        max_waveforms: int = 160000,
        max_mel_spec_frames: int = 625,
        max_sive_feature_frames: int = 157,
        speaker_embedding_dim: int = 192,
    ):
        self.max_waveforms = max_waveforms
        self.max_mel_spec_frames = max_mel_spec_frames
        self.max_sive_feature_frames = max_sive_feature_frames
        self.speaker_embedding_dim = speaker_embedding_dim

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        # Filter out any None examples
        valid_examples = [ex for ex in examples if ex is not None]
        if not valid_examples:
            return {}

        all_waveforms = []
        all_waveform_lengths = []
        all_features = []
        all_feature_lengths = []
        all_mel_specs = []
        all_speaker_embeddings = []
        all_speaker_ids = []
        all_mel_lengths = []
        all_f0 = []
        all_vuv = []
        all_ctc_tokens = []
        all_ctc_lengths = []
        all_texts = []  # the only string column, we won't pad but will collate into a list with one entry per example

        for ex in valid_examples:
            all_waveforms.append(trim(ex.get("waveform", None), self.max_waveforms, dim=-1))
            all_waveform_lengths.append(ex.get("waveform_length", None))
            all_features.append(trim(ex.get("features", None), self.max_sive_feature_frames, dim=-1))
            all_feature_lengths.append(ex.get("feature_length", None))
            all_mel_specs.append(trim(ex.get("mel_spec", None), self.max_mel_spec_frames, dim=-1))
            all_mel_lengths.append(ex.get("mel_length", None))
            all_speaker_embeddings.append(ex.get("speaker_embedding", None))
            all_speaker_ids.append(ex.get("speaker_id", None))
            all_f0.append(trim(ex.get("f0", None), self.max_mel_spec_frames, dim=-1))
            all_vuv.append(trim(ex.get("vuv", None), self.max_mel_spec_frames, dim=-1))
            all_ctc_tokens.append(ex.get("ctc_tokens", None))
            all_ctc_lengths.append(ex.get("ctc_length", None))
            all_texts.append(ex.get("text", None))

        if all_waveforms[0] is not None:
            padded_waveforms, waveform_masks = pad_and_mask(all_waveforms, all_waveform_lengths)
        else:
            padded_waveforms, waveform_masks = None, None
        if all_features[0] is not None:
            padded_features, features_masks = pad_and_mask(all_features, all_feature_lengths)
        else:
            padded_features, features_masks = None, None
        if all_mel_specs[0] is not None:
            padded_mel_specs, mel_spec_masks = pad_and_mask(all_mel_specs, all_mel_lengths)
        else:
            padded_mel_specs, mel_spec_masks = None, None
        if all_f0[0] is not None:
            padded_f0, _ = pad_and_mask(all_f0, all_mel_lengths)
            padded_vuv, _ = pad_and_mask(all_vuv, all_mel_lengths)
        else:
            padded_f0 = None
            padded_vuv = None
        if all_ctc_tokens[0] is not None:
            padded_ctc_tokens, ctc_masks = pad_and_mask(all_ctc_tokens, all_ctc_lengths)
        else:
            padded_ctc_tokens, ctc_masks = None, None
        
        batch = {}

        if padded_waveforms is not None:
            batch["waveforms"] = torch.stack(padded_waveforms)  # [B, T]
            batch["waveform_lengths"] = torch.stack(all_waveform_lengths)  # [B]
            batch["waveform_masks"] = torch.stack(waveform_masks)  # [B, T] mask for waveforms

        if padded_features is not None:
            batch["features"] = torch.stack(padded_features)  # [B, encoder_dim, T'] or [B, num_layers, encoder_dim, T']
            batch["feature_lengths"] = torch.stack(all_feature_lengths)  # [B]
            batch["feature_masks"] = torch.stack(features_masks)  # [B, T'] mask for features

        if padded_mel_specs is not None:
            batch["mel_specs"] = torch.stack(padded_mel_specs)  # [B, num_mel_bins, T']
            batch["mel_lengths"] = torch.stack(all_mel_lengths)  # [B]
            batch["mel_spec_masks"] = torch.stack(mel_spec_masks)  # [B, T'] mask for mel specs

        if all_speaker_embeddings[0] is not None:
            batch["speaker_embeddings"] = torch.stack(all_speaker_embeddings)  # [B, speaker_embedding_dim]
            batch["speaker_ids"] = torch.stack(all_speaker_ids)  # [B]

        if padded_f0 is not None:
            batch["f0"] = torch.stack(padded_f0)  # [B, T]
            batch["vuv"] = torch.stack(padded_vuv)  # [B, T]

        if padded_ctc_tokens is not None:
            batch["ctc_tokens"] = torch.stack(padded_ctc_tokens)  # [B, T]
            batch["ctc_lengths"] = torch.stack(all_ctc_lengths)  # [B]
            batch["ctc_masks"] = torch.stack(ctc_masks)  # [B, T] mask for ctc tokens

        batch["texts"] = all_texts  # list of strings, one per example

        if DEBUG_COLLATOR and "ctc_tokens" in batch:
            print("\n--- DATA COLLATOR DEBUG ---")
            print_debug_tensor("batch mel_specs", batch.get("mel_specs"))
            print_debug_tensor("batch mel_lengths", batch.get("mel_lengths"))
            print_debug_tensor("batch ctc_tokens", batch.get("ctc_tokens"))
            print_debug_tensor("batch ctc_lengths", batch.get("ctc_lengths"))
            print(f"  mel_lengths: {batch['mel_lengths'].tolist() if 'mel_lengths' in batch else 'N/A'}")
            print(f"  ctc_lengths: {batch['ctc_lengths'].tolist() if 'ctc_lengths' in batch else 'N/A'}")
            # Check the ratio of mel frames to CTC tokens (need sufficient frames for CTC)
            if "mel_lengths" in batch and "ctc_lengths" in batch:
                mel_lens = batch["mel_lengths"]
                ctc_lens = batch["ctc_lengths"]
                for i in range(len(mel_lens)):
                    ratio = mel_lens[i].item() / max(ctc_lens[i].item(), 1)
                    print(f"  Sample {i}: mel_len={mel_lens[i].item()}, ctc_len={ctc_lens[i].item()}, ratio={ratio:.2f}")
            print("---\n")

        return batch
