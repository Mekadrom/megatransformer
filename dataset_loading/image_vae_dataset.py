import json
import os
from typing import Dict, List
import torch
from torch.utils.data import Dataset


class CachedImageVAEDataset(Dataset):
    def __init__(
        self,
        cache_dir: str,
    ):
        self.cache_dir = cache_dir
        
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
            "images": data["images"],
        }


class ImageVAEDataCollator:
    def __init__(
        self,
        audio_max_frames: int,
        audio_max_waveform_length: int,
        n_mels: int,
        input_noise_std: float = 0.0,
        training: bool = True,
    ):
        self.audio_max_frames = audio_max_frames
        self.audio_max_waveform_length = audio_max_waveform_length
        self.n_mels = n_mels
        self.input_noise_std = input_noise_std
        self.training = training

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        images = []
        for ex in examples:
            if ex is None:
                continue

            image = ex["images"]
            images.append(image)

        # Stack tensors
        image_batch = torch.stack(images)

        batch = {
            "images": image_batch,
        }

        return batch
