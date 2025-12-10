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
        return {
            "text_attention_mask": data["text_attention_mask"],
            "text_embeddings": data["text_embeddings"],
            "mel_spec": data["mel_spec"],
        }


class AudioDiffusionDataCollator:
    """Data collator for audio diffusion training."""
    def __init__(
        self,
        audio_max_frames: int,
        max_conditions: int,
        n_mels: int,
    ):
        self.audio_max_frames = audio_max_frames
        self.max_conditions = max_conditions
        self.n_mels = n_mels

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        attention_masks = []
        text_embeddings = []
        mel_specs = []
        for ex in examples:
            if ex is None:
                continue

            mel = ex["mel_spec"]
            if mel.shape[-1] < self.audio_max_frames:
                mel = F.pad(mel, (0, self.audio_max_frames - mel.shape[-1]), value=0)
            elif mel.shape[-1] > self.audio_max_frames:
                mel = mel[..., :self.audio_max_frames]

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

        # Stack tensors
        batch = {
            "attention_mask": torch.stack(attention_masks),
            "text_embeddings": torch.stack(text_embeddings),
            "mel_spec": torch.stack(mel_specs),
        }

        return batch
