import torch
import torch.nn.functional as F

from megatransformer.scripts.data.data_collator import DataCollator
from megatransformer.utils.megatransformer_utils import pad_and_mask, trim, print_debug_tensor

# Set to True to enable debug printing in data collator
DEBUG_COLLATOR = False  # Set True to debug data loading issues


class VoiceDataCollator(DataCollator):
    """
    Data collator for any voice training.

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
        all_unit_ids = []
        all_feature_lengths = []
        all_mel_specs = []
        all_speaker_embeddings = []
        all_speaker_ids = []
        all_gender_ids = []
        all_mel_lengths = []
        all_f0 = []
        all_f0_contour = []
        all_vuv = []
        all_ctc_tokens = []
        all_ctc_lengths = []
        all_texts = []  # the only string column, we won't pad but will collate into a list with one entry per example

        for ex in valid_examples:
            all_waveforms.append(trim(ex.get("waveform", None), self.max_waveforms, dim=-1))
            all_waveform_lengths.append(ex.get("waveform_length", None))
            all_features.append(trim(ex.get("features", None), self.max_sive_feature_frames, dim=-1))
            all_unit_ids.append(trim(ex.get("unit_ids", None), self.max_sive_feature_frames, dim=-1))
            all_feature_lengths.append(ex.get("feature_length", None))
            all_mel_specs.append(trim(ex.get("mel_spec", None), self.max_mel_spec_frames, dim=-1))
            all_mel_lengths.append(ex.get("mel_length", None))
            all_speaker_embeddings.append(ex.get("speaker_embedding", None))
            all_speaker_ids.append(ex.get("speaker_id", None))
            all_gender_ids.append(ex.get("gender_id", None))
            all_f0.append(trim(ex.get("f0", None), self.max_mel_spec_frames, dim=-1))
            all_f0_contour.append(trim(ex.get("f0_contour", None), self.max_mel_spec_frames, dim=-1))
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
            # Zero the feature pad. quantize() deliberately leaves padding frames as their
            # nonzero stored centroids (so unit_ids stay -1 without snapping to a spurious
            # unit near the origin), and pad_and_mask masks-but-does-not-zero. But the SMG
            # decoder applies no input mask, so its conv receptive field bleeds those
            # real-unit-like pad frames into the last VALID frames -> an end-of-utterance
            # energy burst (visible in the mel; GT decays to silence, so the model is
            # manufacturing it). Zeroing here removes the contamination AND makes the
            # training boundary match the eval path (which trims to length), so the model
            # can learn to render the stereotyped trailing-silence codes as actual silence.
            # Mask is [T'] and broadcasts against the last dim of [D, T'] or [L, D, T'].
            padded_features = [f * m.to(f.dtype) for f, m in zip(padded_features, features_masks)]
        else:
            padded_features, features_masks = None, None
        # Pre-quantized unit ids (Mimi): [T'] int. Padded into the "features" slot so the
        # SMG's _embed_ids sees integer ids. Only when there are no continuous features
        # (the SIVE-VQ path already yields float centroids). Pad id 0 is a valid code but
        # its frames are masked downstream by the mel length, so it never reaches the loss.
        if padded_features is None and all_unit_ids[0] is not None:
            padded_unit_ids, unit_id_masks = pad_and_mask(all_unit_ids, all_feature_lengths)
            # Mark every pad frame (the collator's batch pad AND the shard's internal 0-pad
            # past feature_length) with -1 so the SMG embeds it to ZERO instead of code 0's
            # real vector. Otherwise the decoder renders code-0 content in the padded tail =
            # an end-of-clip burst (invisible in the length-trimmed mel viz, audible in the
            # vocoded clip). Mirrors the continuous feature-pad zeroing above.
            padded_unit_ids = [torch.where(m.bool(), u, torch.full_like(u, -1))
                               for u, m in zip(padded_unit_ids, unit_id_masks)]
        else:
            padded_unit_ids, unit_id_masks = None, None
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
        elif padded_unit_ids is not None:
            batch["features"] = torch.stack(padded_unit_ids)  # [B, T'] integer unit ids
            batch["feature_lengths"] = torch.stack(all_feature_lengths)  # [B]
            batch["feature_masks"] = torch.stack(unit_id_masks)  # [B, T']

        if padded_mel_specs is not None:
            batch["mel_specs"] = torch.stack(padded_mel_specs)  # [B, num_mel_bins, T']
            batch["mel_lengths"] = torch.stack(all_mel_lengths)  # [B]
            batch["mel_spec_masks"] = torch.stack(mel_spec_masks)  # [B, T'] mask for mel specs

        if all_speaker_embeddings[0] is not None:
            batch["speaker_embeddings"] = torch.stack(all_speaker_embeddings)  # [B, speaker_embedding_dim]
            batch["speaker_ids"] = torch.stack(all_speaker_ids)  # [B]

        if all_gender_ids[0] is not None:
            batch["gender_ids"] = torch.stack(all_gender_ids)  # [B]

        if padded_f0 is not None:
            batch["f0"] = torch.stack(padded_f0)  # [B, T]
            if all_f0_contour[0] is not None:
                # Speaker-normalized contour. 0-pad is fine: unvoiced frames are already 0
                # and the F0 loss is voicing-weighted, so padding contributes nothing.
                padded_c, _ = pad_and_mask(all_f0_contour, all_mel_lengths)
                batch["f0_contour"] = torch.stack(padded_c)
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
