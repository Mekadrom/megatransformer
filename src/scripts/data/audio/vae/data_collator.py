import torch
import torch.nn.functional as F

from scripts.data.data_collator import DataCollator


class SIVEFeatureDataCollator(DataCollator):
    """
    Data collator for SIVE feature VAE training.

    Handles both single-layer and multi-layer feature formats:
    - Single layer: [encoder_dim, T'] -> batch: [B, encoder_dim, T']
    - Multi-layer: [num_layers, encoder_dim, T'] -> batch: [B, num_layers, encoder_dim, T']

    Pads features to same length within batch and creates masks.
    """

    def __init__(
        self,
        max_feature_frames: int = 469,
        max_audio_frames: int = 1875,
        speaker_embedding_dim: int = 192,
    ):
        self.max_feature_frames = max_feature_frames
        self.max_audio_frames = max_audio_frames
        self.speaker_embedding_dim = speaker_embedding_dim

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        # Filter out any None examples
        valid_examples = [ex for ex in examples if ex is not None]
        if not valid_examples:
            return {}

        raw_features = []
        feature_lengths = []
        speaker_embeddings = []
        mel_specs = []
        mel_lengths = []
        f0_list = []
        voiced_list = []

        # Detect multi-layer mode and F0 availability from first example
        first_features = valid_examples[0]["features"]
        num_layers = valid_examples[0].get("num_layers", 1)
        is_multi_layer = num_layers > 1 or first_features.dim() == 3
        has_f0 = "f0" in valid_examples[0]

        for ex in valid_examples:
            # Shape depends on mode:
            # - Single layer: [encoder_dim, T']
            # - Multi-layer:  [num_layers, encoder_dim, T']
            features = ex["features"]
            feature_length = ex.get("feature_length", features.shape[-1])

            # Clamp length
            feature_length = min(feature_length, features.shape[-1], self.max_feature_frames)

            raw_features.append(features)
            feature_lengths.append(feature_length)

            speaker_emb = ex.get("speaker_embedding", None)
            if speaker_emb is not None:
                if speaker_emb.dim() == 1:
                    speaker_emb = speaker_emb.unsqueeze(0)
                speaker_embeddings.append(speaker_emb)
            else:
                speaker_embeddings.append(torch.zeros(1, self.speaker_embedding_dim))

            mel_specs.append(ex.get("mel_specs"))
            mel_lengths.append(ex.get("mel_lengths"))

            # Collect F0 data if available
            if has_f0:
                f0_list.append(ex["f0"])
                voiced_list.append(ex["voiced"])

        # Compute batch max length (dynamic padding)
        batch_max_length = max(feature_lengths)

        # Pad/truncate to batch max length
        padded_features = []

        for feat, feat_length in zip(raw_features, feature_lengths):
            # Create mask (1 = valid, 0 = padding)
            feat_mask = torch.zeros(batch_max_length, dtype=torch.float32)
            feat_mask[:feat_length] = 1.0

            # Truncate to batch max length (along last dimension for both formats)
            feat = feat[..., :batch_max_length]

            # Pad if needed (along last dimension)
            if feat.shape[-1] < batch_max_length:
                feat = F.pad(feat, (0, batch_max_length - feat.shape[-1]), value=0)

            padded_features.append(feat)

        mel_spec_masks = []
        padded_mel_specs = []
        for m, mel in enumerate(mel_specs):
            # Truncate/pad mel specs to max length if needed
            mel = mel[..., :self.max_audio_frames]
            # print(f"Collation: Padding mel spec from {mel.shape[-1]} to {self.max_audio_frames}. Mask will have shape [{self.max_audio_frames}] with ones up to {mel_lengths[m]}.")
            if mel.shape[-1] < self.max_audio_frames:
                mel = F.pad(mel, (0, self.max_audio_frames - mel.shape[-1]), value=0)
            mel_spec_mask = torch.zeros(self.max_audio_frames, dtype=torch.float32)
            mel_spec_mask[:mel_lengths[m]] = 1.0
            padded_mel_specs.append(mel)
            mel_spec_masks.append(mel_spec_mask)

        # Stack features:
        # - Single layer: [B, encoder_dim, T']
        # - Multi-layer:  [B, num_layers, encoder_dim, T']
        batch = {
            "features": torch.stack(padded_features),
            'mel_specs': torch.stack(padded_mel_specs),
            "mel_lengths": torch.tensor(mel_lengths, dtype=torch.long),  # [B]
            "mel_spec_masks": torch.stack(mel_spec_masks),  # [B, T] mask for mel specs
            "feature_lengths": torch.tensor(feature_lengths, dtype=torch.long),  # [B]
            "speaker_embedding": torch.stack(speaker_embeddings),  # [B, 1, speaker_embedding_dim]
            "num_layers": num_layers,  # For VAE to know the format
        }

        # Add F0 data if available
        if has_f0 and f0_list:
            # Pad/truncate F0 and voiced to same length as mel specs
            padded_f0 = []
            padded_voiced = []
            for f0, voiced in zip(f0_list, voiced_list):
                # Truncate to max length
                f0 = f0[:self.max_audio_frames]
                voiced = voiced[:self.max_audio_frames]
                # Pad if needed
                if len(f0) < self.max_audio_frames:
                    f0 = F.pad(f0, (0, self.max_audio_frames - len(f0)), value=0)
                    voiced = F.pad(voiced, (0, self.max_audio_frames - len(voiced)), value=0)
                padded_f0.append(f0)
                padded_voiced.append(voiced)

            batch["target_f0"] = torch.stack(padded_f0)  # [B, T]
            batch["target_voiced"] = torch.stack(padded_voiced)  # [B, T]

        return batch
