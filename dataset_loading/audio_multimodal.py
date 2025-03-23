from datasets import load_dataset, Audio, config
from typing import List, Literal, Optional 

import librosa
import logging
import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_audio_features(audio, sr=16000, n_mels=128, n_fft=1024, hop_length=512):
    """
    Extract audio features from loaded audio data.
    
    Args:
        audio: Audio data loaded by datasets library (contains 'array' and 'sampling_rate')
        sr: Target sampling rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length for feature extraction
        
    Returns:
        log_mel_spec: Log mel spectrogram features
    """
    # Check if audio is already processed
    if isinstance(audio, dict) and 'array' in audio:
        y = audio['array']
        orig_sr = audio['sampling_rate']
        
        # Resample if needed
        if orig_sr != sr:
            y = librosa.resample(y, orig_freq=orig_sr, target_freq=sr)
    else:
        # Fallback for direct file paths
        y, orig_sr = librosa.load(audio, sr=sr)
    
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    
    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec)
    
    return log_mel_spec

def load_audio_dataset(
    dataset_name,
    dataset_config_name: Literal["clean", "other", "all"], 
    split,
    tokenizer=None,
    batch_size: int = 100,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
    max_duration: Optional[float] = None,
    streaming: bool = False,
    cache_dir: Optional[str] = None,
):
    """
    Load and process the LibriSpeech dataset with feature extraction.
    
    Args:
        config_name: Dataset configuration ("clean" or "other" or "all")
        splits: List of dataset splits to load
        batch_size: Batch size for processing
        num_proc: Number of processes for parallel processing
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length for feature extraction
        max_duration: Maximum audio duration to keep (in seconds)
        
    Returns:
        processed_dataset: Dataset with processed features
    """
    logger.info(f"Loading dataset {dataset_name} with config {dataset_config_name}, split {split} for audio.")
    
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config_name, cache_dir=cache_dir, split=split, streaming=streaming, trust_remote_code=True)

    audio_placeholder_token = tokenizer.convert_tokens_to_ids("<|AUDIO|>")

    assert isinstance(audio_placeholder_token, int), f"Audio placeholder token should be an integer, got {type(audio_placeholder_token)}"
    
    # Define processing function for custom features
    def process_audio(examples):
        # Load and process audio

        text_column_name = "text" if "text" in examples else "caption" if "caption" in examples else "sentence" if "sentence" in examples else None
        if text_column_name is None:
            raise ValueError("No text column found in the dataset.")

        audios = examples["audio"]
        texts = examples[text_column_name]

        all_input_ids = []
        all_audio_raw_inputs = []
        for text, audio in zip(texts, audios):
            tokenized = tokenizer(text=text, add_special_tokens=True)
            input_ids = tokenized.input_ids

            # Skip long audio files if specified
            if max_duration is not None:
                duration = len(audio["array"]) / audio["sampling_rate"]
                if duration > max_duration:
                    continue
            
            transcription_input_ids = [audio_placeholder_token] + input_ids
            generation_input_ids = input_ids + [audio_placeholder_token]
            
            # Extract features
            audio_raw_inputs = extract_audio_features(
                audio,
                sr=16000,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length
            )

            all_input_ids.append(torch.tensor(transcription_input_ids))
            all_input_ids.append(torch.tensor(generation_input_ids))

            audio_raw_inputs = torch.tensor(audio_raw_inputs)
            all_audio_raw_inputs.append(audio_raw_inputs)
            all_audio_raw_inputs.append(audio_raw_inputs)

        # Pad sequences to the same length
        max_length = max([len(ids) for ids in all_input_ids])
        all_input_ids = [torch.nn.functional.pad(ids, (0, max_length - len(ids)), value=0) for ids in all_input_ids]

        # Return processed features and transcript
        return {
            "input_ids": torch.stack(all_input_ids),
            "audio_raw_inputs": all_audio_raw_inputs,
            "audio_labels": all_audio_raw_inputs,
        }
    
    logger.info("Processing dataset with custom feature extraction...")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    dataset = dataset.map(
        process_audio,
        batch_size=batch_size,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    logger.info("Dataset processing complete!")
    return dataset
