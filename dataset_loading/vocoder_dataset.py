from typing import Dict, List
import torch
import torch.nn.functional as F
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


class VocoderDataCollator:
    """Data collator for vocoder training."""

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
        mel_specs = []
        waveforms = []
        target_complex_stfts = []
        for ex in examples:
            if ex is None:
                continue

            mel = ex["mel_spec"]
            if mel.shape[-1] < self.audio_max_frames:
                mel = F.pad(mel, (0, self.audio_max_frames - mel.shape[-1]), value=0)
            elif mel.shape[-1] > self.audio_max_frames:
                mel = mel[..., :self.audio_max_frames]

            wav = ex["waveform_labels"]
            if wav.shape[-1] < self.audio_max_waveform_length:
                wav = F.pad(wav, (0, self.audio_max_waveform_length - wav.shape[-1]), value=0)
            elif wav.shape[-1] > self.audio_max_waveform_length:
                wav = wav[..., :self.audio_max_waveform_length]
            
            target_complex_stft = ex["target_complex_stfts"]
            if target_complex_stft.shape[-1] < self.audio_max_frames:
                target_complex_stft = F.pad(target_complex_stft, (0, self.audio_max_frames - target_complex_stft.shape[-1]), value=0)
            elif target_complex_stft.shape[-1] > self.audio_max_frames:
                target_complex_stft = target_complex_stft[..., :self.audio_max_frames]

            assert target_complex_stft.shape[-1] == mel.shape[-1], f"Mismatch in mel frames and stft frames: {mel.shape} vs {target_complex_stft.shape}. Max frames: {self.audio_max_frames}"

            mel_specs.append(mel)
            waveforms.append(wav)
            target_complex_stfts.append(target_complex_stft)

        # Stack tensors
        mel_batch = torch.stack(mel_specs)

        # Add input noise for regularization (only to mel specs, not targets)
        if self.training and self.input_noise_std > 0.0:
            mel_batch = mel_batch + torch.randn_like(mel_batch) * self.input_noise_std

        batch = {
            "mel_spec": mel_batch,
            "waveform_labels": torch.stack(waveforms),
            "target_complex_stfts": torch.stack(target_complex_stfts),
        }

        return batch
