from typing import Dict, List
import torch
import torch.nn.functional as F
import os
import json

from torch.utils.data import Dataset


class CachedAudioDiffusionDataset(Dataset):
    """
    Dataset that loads preprocessed audio diffusion data from cached .pt files.
    Much faster than processing audio at runtime.
    """
    
    def __init__(
        self,
        cache_dir: str,
        audio_max_frames: int = 1875,
    ):
        self.cache_dir = cache_dir
        self.audio_max_frames = audio_max_frames
        
        # Load config
        config_path = os.path.join(cache_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        # Find all cached files
        self.file_paths = sorted([
            os.path.join(cache_dir, f) 
            for f in os.listdir(cache_dir) 
            if f.endswith(".pt")
        ])
        
        print(f"Found {len(self.file_paths)} cached examples in {cache_dir}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx])
        # Return in expected format
        # mel_spec_length may not exist in older cached datasets
        mel_spec_length = data.get("mel_spec_length", data["mel_spec"].shape[-1])
        result = {
            "text_attention_mask": data["text_attention_mask"],
            "text_embeddings": data["text_embeddings"],
            "mel_spec": data["mel_spec"],
            "mel_spec_length": mel_spec_length,
            "speaker_embedding": data["speaker_embedding"],
        }
        # Include VAE latents if available (for latent diffusion)
        if "latent_mu" in data:
            result["latent_mu"] = data["latent_mu"]
            result["latent_shape"] = data.get("latent_shape", list(data["latent_mu"].shape))
        return result


class AudioDiffusionDataCollator:
    """Data collator for audio diffusion training."""
    def __init__(
        self,
        audio_max_frames: int,
        max_conditions: int,
        n_mels: int,
        use_latent_diffusion: bool = False,
        latent_max_frames: int = 25,  # audio_max_frames / time_compression (e.g., 1875/75=25)
    ):
        self.audio_max_frames = audio_max_frames
        self.max_conditions = max_conditions
        self.n_mels = n_mels
        self.use_latent_diffusion = use_latent_diffusion
        self.latent_max_frames = latent_max_frames

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        attention_masks = []
        text_embeddings = []
        mel_specs = []
        mel_spec_masks = []
        speaker_embeddings = []
        latent_mus = []
        has_latents = False

        for ex in examples:
            if ex is None:
                continue

            mel = ex["mel_spec"]
            mel_length = ex.get("mel_spec_length", mel.shape[-1])

            # Create mel spec padding mask (1 = valid, 0 = padding)
            mel_mask = torch.ones(self.audio_max_frames, dtype=torch.float32)
            if mel_length < self.audio_max_frames:
                mel_mask[mel_length:] = 0.0

            if mel.shape[-1] < self.audio_max_frames:
                mel = F.pad(mel, (0, self.audio_max_frames - mel.shape[-1]), value=0)
            elif mel.shape[-1] > self.audio_max_frames:
                mel = mel[..., :self.audio_max_frames]
                mel_mask[:] = 1.0  # All valid if truncated

            text_embedding = ex["text_embeddings"]
            attention_mask = ex["text_attention_mask"]
            if text_embedding.shape[0] < self.max_conditions:
                pad_length = self.max_conditions - text_embedding.shape[0]
                text_embedding = F.pad(text_embedding, (0, 0, 0, pad_length), value=0)
                attention_mask = F.pad(attention_mask, (0, pad_length), value=0)
            elif text_embedding.shape[0] > self.max_conditions:
                text_embedding = text_embedding[:self.max_conditions, :]
                attention_mask = attention_mask[:self.max_conditions]

            attention_masks.append(attention_mask)
            text_embeddings.append(text_embedding)
            mel_specs.append(mel)
            mel_spec_masks.append(mel_mask)
            speaker_embeddings.append(ex["speaker_embedding"])

            # Handle latent diffusion
            if "latent_mu" in ex:
                has_latents = True
                latent = ex["latent_mu"]
                # Pad or truncate latent time dimension
                if latent.shape[-1] < self.latent_max_frames:
                    latent = F.pad(latent, (0, self.latent_max_frames - latent.shape[-1]), value=0)
                elif latent.shape[-1] > self.latent_max_frames:
                    latent = latent[..., :self.latent_max_frames]
                latent_mus.append(latent)

        # Stack tensors
        batch = {
            "attention_mask": torch.stack(attention_masks),
            "text_embeddings": torch.stack(text_embeddings),
            "mel_spec": torch.stack(mel_specs),
            "mel_spec_mask": torch.stack(mel_spec_masks),  # [B, T] mask for mel spectrogram
            "speaker_embedding": torch.stack(speaker_embeddings),
        }

        # Add latents if available and requested
        if has_latents and self.use_latent_diffusion:
            batch["latent_mu"] = torch.stack(latent_mus)

        return batch
