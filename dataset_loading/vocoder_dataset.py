import torch
import os
import json

from torch.utils.data import Dataset


class CachedVocoderDataset(Dataset):
    """
    Dataset that loads preprocessed vocoder data from cached .pt files.
    Much faster than processing audio at runtime.
    """
    
    def __init__(
        self,
        cache_dir: str,
        audio_max_frames: int = 626,
        audio_max_waveform_length: int = 160000,
    ):
        self.cache_dir = cache_dir
        self.audio_max_frames = audio_max_frames
        self.audio_max_waveform_length = audio_max_waveform_length
        
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
            "mel_spec": data["mel_spec"],
            "waveform_labels": data["waveform_labels"],
            "target_complex_stfts": data["target_complex_stfts"],
        }
