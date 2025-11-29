from typing import Dict, Optional, Iterator, Any
from datasets import load_from_disk, Audio, load_dataset
from torch.utils.data import IterableDataset, Dataset
import torch
from transformers import T5Tokenizer

from dataset_loading import audio_loading


class VocoderIterableDataset(IterableDataset):
    """
    Dataset for vocoder training that loads audio with transcriptions.
    Returns mel spectrograms, waveforms, and text for T5 conditioning.
    """
    def __init__(
        self,
        config,
        t5_tokenizer: Optional[T5Tokenizer],
        approximated_length: int,
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        audio_max_frames: int,
        cache_dir: str = "cached_datasets",
        split: str = "train",
        dataset_name: str = "fixie-ai/common_voice_17_0",
        dataset_config: str = "en",
        max_text_length: int = 256,
    ):
        self.config = config
        self.t5_tokenizer = t5_tokenizer
        self.approximated_length = approximated_length
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.audio_max_frames = audio_max_frames
        self.cache_dir = cache_dir
        self.split = split
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.max_text_length = max_text_length

        # Load the base audio dataset
        self.dataset = load_dataset(
            dataset_name,
            dataset_config,
            cache_dir=cache_dir,
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=sample_rate))

    def __len__(self) -> int:
        return self.approximated_length

    def __iter__(self) -> Iterator[Any]:
        for example in self.dataset:
            try:
                processed = self._process_example(example)
                if processed is not None:
                    yield processed
            except Exception as e:
                print(f"Error processing example: {e}")
                continue

    def _process_example(self, example: Dict) -> Optional[Dict]:
        """Process a single example into vocoder training format."""
        # Get text - handle different column names
        text = example.get("text") or example.get("sentence") or example.get("caption")
        if text is None:
            return None

        # Get audio
        audio = example["audio"]

        # Extract waveform and mel spectrogram
        waveforms, y, _ = audio_loading.extract_waveforms(audio, sr=self.sample_rate)

        # Skip low-energy audio
        if waveforms.abs().max() < 0.05 or waveforms.std() < 0.02:
            return None

        mel_spec = audio_loading.extract_mels(
            y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        # Skip if mel spectrogram is too long
        if mel_spec.shape[-1] > self.audio_max_frames:
            return None

        # Tokenize text for T5
        if self.t5_tokenizer is not None:
            text_encoding = self.t5_tokenizer(
                text,
                max_length=self.max_text_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        else:
            text_encoding = None

        return {
            "mel_spec": mel_spec,
            "waveform_labels": waveforms,
            "text_input_ids": text_encoding["input_ids"].squeeze(0) if text_encoding is not None else torch.tensor([]),
            "text_attention_mask": text_encoding["attention_mask"].squeeze(0) if text_encoding is not None else torch.tensor([]),
        }


class VocoderDataset(Dataset):
    """
    Dataset for vocoder training that loads audio with transcriptions.
    Returns mel spectrograms, waveforms, and text for T5 conditioning.
    """
    def __init__(
        self,
        config,
        t5_tokenizer: Optional[T5Tokenizer],
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        audio_max_frames: int,
        cache_dir: str = "cached_datasets",
        split: str = "train",
        dataset_name: str = "fixie-ai/common_voice_17_0",
        dataset_config: str = "en",
        max_text_length: int = 256,
    ):
        self.config = config
        self.t5_tokenizer = t5_tokenizer
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.audio_max_frames = audio_max_frames
        self.cache_dir = cache_dir
        self.split = split
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.max_text_length = max_text_length

        self.dataset = load_from_disk(
            "./cached_datasets/cv17_local"
        )
        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=sample_rate))

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self._process_example(self.dataset[idx])

    def _process_example(self, example: Dict) -> Optional[Dict]:
        """Process a single example into vocoder training format."""
        # Get text - handle different column names
        text = example.get("text") or example.get("sentence") or example.get("caption")
        if text is None:
            return None

        # Get audio
        audio = example["audio"]

        # Extract waveform and mel spectrogram
        waveforms, y, _ = audio_loading.extract_waveforms(audio, sr=self.sample_rate)

        # Skip low-energy audio
        if waveforms.abs().max() < 0.05 or waveforms.std() < 0.02:
            return None

        mel_spec = audio_loading.extract_mels(
            y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        # Skip if mel spectrogram is too long
        if mel_spec.shape[-1] > self.audio_max_frames:
            return None

        # Tokenize text for T5
        if self.t5_tokenizer is not None:
            text_encoding = self.t5_tokenizer(
                text,
                max_length=self.max_text_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        else:
            text_encoding = None

        return {
            "mel_spec": mel_spec,
            "waveform_labels": waveforms,
            "text_input_ids": text_encoding["input_ids"].squeeze(0) if text_encoding is not None else torch.tensor([]),
            "text_attention_mask": text_encoding["attention_mask"].squeeze(0) if text_encoding is not None else torch.tensor([]),
        }
