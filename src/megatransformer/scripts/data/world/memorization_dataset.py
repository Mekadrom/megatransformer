"""
In-memory multimodal dataset for fast memorization experiments.

Loads a fixed number of samples from each modality's shard directory into RAM
at init time. No shard swapping, no LRU cache, no sampler overhead — just
direct tensor indexing. Drop-in replacement for MultimodalShardedDataset.

Usage:
    dataset = MultimodalMemorizationDataset(
        text_shard_dir="path/to/text",
        voice_shard_dir="path/to/voice",
        image_shard_dir="path/to/image",
        max_samples=32,
    )
"""

import os
import torch
from torch.utils.data import Dataset


class MultimodalMemorizationDataset(Dataset):
    """In-memory dataset that preloads a small number of samples per modality."""

    def __init__(
        self,
        text_shard_dir: str = None,
        audio_shard_dir: str = None,
        voice_shard_dir: str = None,
        image_shard_dir: str = None,
        text_columns: list[str] = None,
        audio_columns: list[str] = None,
        voice_columns: list[str] = None,
        cache_size: int = 3,  # ignored, kept for interface compat
        max_samples: int = 32,
        voice_codebook=None,
    ):
        self.samples = []
        self.modalities_present = set()

        # Determine per-modality sample count: split max_samples across modalities
        dirs = {}
        if text_shard_dir is not None:
            dirs["text"] = text_shard_dir
        if audio_shard_dir is not None:
            dirs["audio"] = audio_shard_dir
        if voice_shard_dir is not None:
            dirs["voice"] = voice_shard_dir
        if image_shard_dir is not None:
            dirs["image"] = image_shard_dir

        if not dirs:
            raise ValueError("At least one shard directory must be provided")

        n_modalities = len(dirs)
        # DISCRETE-UNIT VOICE. This dataset predates the VQ/CosyVoice 2 path: those caches
        # store `unit_ids` and NO `features` key, and the sharded dataset expands
        # centroids[unit_ids] on the fly. Without the codebook here, voice samples came back
        # with no features and no feature_length, and the collator died on int(None).
        from megatransformer.utils.codebook import load_codebook
        self.voice_centroids = load_codebook(voice_codebook) if voice_codebook is not None else None

        per_modality = max(1, max_samples // n_modalities)

        # Load raw samples per modality
        modality_samples = {}
        for name, shard_dir in dirs.items():
            self.modalities_present.add(name)
            modality_samples[name] = self._load_samples(name, shard_dir, per_modality)
            print(f"[MemorizationDataset] {name}: loaded {len(modality_samples[name])} samples from {shard_dir}")

        # Total length = max across modalities; shorter wrap via modulo
        self.total_samples = max(len(s) for s in modality_samples.values())
        if max_samples is not None and max_samples > 0:
            self.total_samples = min(self.total_samples, max_samples)

        # Precompute all samples into a flat list for O(1) access
        self._modality_samples = modality_samples
        print(f"[MemorizationDataset] Effective length: {self.total_samples}")

    def _load_samples(self, modality: str, shard_dir: str, n: int) -> list[dict]:
        """Load up to n samples from the first shard(s) in a directory."""
        shard_files = sorted([
            f for f in os.listdir(shard_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        ])
        if not shard_files:
            raise ValueError(f"No shard files found in {shard_dir}")

        samples = []
        for shard_file in shard_files:
            if len(samples) >= n:
                break
            shard = torch.load(
                os.path.join(shard_dir, shard_file),
                map_location="cpu",
                weights_only=True,
            )
            num_in_shard = shard.get("num_samples", 0)
            for i in range(min(num_in_shard, n - len(samples))):
                samples.append(self._extract_sample(modality, shard, i))

        return samples

    def _extract_sample(self, modality: str, shard: dict, idx: int) -> dict:
        """Extract a single sample dict from a shard."""
        if modality == "text":
            sample = {
                "token_ids": shard["token_ids"][idx],
                "text_length": shard["text_lengths"][idx],
            }
            if "text" in shard:
                sample["text"] = shard["text"][idx]
            elif "raw_text" in shard:
                sample["text"] = shard["raw_text"][idx]
            return sample

        elif modality in ("audio", "voice"):
            sample = {}
            if "features" in shard:
                sample["features"] = shard["features"][idx]
                sample["feature_length"] = shard["feature_lengths"][idx]
            elif "unit_ids" in shard:
                # Discrete path: mirror MultimodalShardedDataset._get_voice_sample.
                unit_ids = shard["unit_ids"][idx]
                sample["unit_ids"] = unit_ids
                sample["feature_length"] = shard["feature_lengths"][idx]
                if self.voice_centroids is not None:
                    sample["features"] = self.voice_centroids[
                        unit_ids.clamp(min=0)].transpose(0, 1).contiguous()   # (D, T)
            if "waveforms" in shard:
                sample["waveform"] = shard["waveforms"][idx]
                sample["waveform_length"] = shard["waveform_lengths"][idx]
            if "mel_specs" in shard:
                sample["mel_spec"] = shard["mel_specs"][idx]
                sample["mel_length"] = shard["mel_lengths"][idx]
            if "speaker_embeddings" in shard:
                sample["speaker_embedding"] = shard["speaker_embeddings"][idx]
            if "speaker_ids" in shard:
                sample["speaker_id"] = shard["speaker_ids"][idx]
            if "f0" in shard:
                sample["f0"] = shard["f0"][idx]
            if "vuv" in shard:
                sample["vuv"] = shard["vuv"][idx]
            if "ctc_tokens" in shard:
                sample["ctc_tokens"] = shard["ctc_tokens"][idx]
                sample["ctc_length"] = shard["ctc_lengths"][idx]
            if "text" in shard:
                sample["voice_text"] = shard["text"][idx]
            # The voice shard carries the PAIRED transcript. Without lifting it out, a
            # voice-only run produces samples with no text_token_ids at all, the collator
            # skips text entirely, and compute_loss returns zero loss -- training on nothing.
            if "token_ids" in shard:
                sample["token_ids"] = shard["token_ids"][idx]
                sample["text_length"] = shard["text_lengths"][idx]
            return sample

        elif modality == "image":
            if "latents" in shard:
                return {"image": shard["latents"][idx]}
            return {"image": shard["images"][idx]}

        return {}

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int) -> dict:
        # _direction/_task matter: the collator reads ex["_direction"], and without it the
        # synthesis/transcription choice falls back to a 50/50 coin flip, so half a
        # "text -> voice" memorization run would silently train transcription instead.
        sample = {"_modality": "voice", "_task": "voice_synthesis", "_direction": "synthesis"} \
            if "voice" in self.modalities_present else {}

        for modality, mod_samples in self._modality_samples.items():
            wrapped_idx = idx % len(mod_samples)
            mod_sample = mod_samples[wrapped_idx]

            if modality == "text":
                sample.update({f"text_{k}": v for k, v in mod_sample.items()})
            elif modality == "audio":
                sample.update({f"audio_{k}": v for k, v in mod_sample.items()})
            elif modality == "voice":
                sample.update({f"voice_{k}": v for k, v in mod_sample.items()})
                # Mirror MultimodalShardedDataset.__getitem__: promote the voice shard's own
                # transcript to the text slots the collator looks for.
                if "token_ids" in mod_sample:
                    sample["text_token_ids"] = mod_sample["token_ids"]
                    sample["text_text_length"] = mod_sample["text_length"]
                if "voice_text" in mod_sample:
                    sample["text_text"] = mod_sample["voice_text"]
            elif modality == "image":
                sample.update({f"image_{k}": v for k, v in mod_sample.items()})

        return sample
