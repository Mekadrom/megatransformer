from datasets import load_dataset, Audio
import torchaudio
from transformers import PreTrainedTokenizer
from typing import Literal, Optional

import logging
import torch

from utils.audio_utils import configurable_mel_spectrogram, SharedWindowBuffer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_waveforms(audio, sr=16000):
    # Check if audio is already processed
    if isinstance(audio, dict) and 'array' in audio:
        orig_sr = audio['sampling_rate']
        
        waveforms = torch.tensor(audio['array'], dtype=torch.float32).unsqueeze(0)
    elif isinstance(audio, torch.Tensor):
        # waveform as tensor
        waveforms = audio if audio.dim() == 2 else audio.unsqueeze(0)
        # assume default sample rate
        orig_sr = sr
    else:
        # Fallback for direct file paths
        waveforms, orig_sr = torchaudio.load(audio)

    # Resample if needed
    if orig_sr != sr:
        waveforms = torchaudio.transforms.Resample(orig_sr, sr)(waveforms)
    waveforms = waveforms.squeeze(0)
    y = waveforms.numpy()

    return waveforms, y, orig_sr

def extract_mels(shared_window_buffer: SharedWindowBuffer, y, sr=16000, n_mels=80, n_fft=1024, hop_length=256):
    """
    Extract audio features from loaded audio data.

    Args:
        shared_window_buffer: SharedWindowBuffer for STFT windows
        y: Audio data as waveform. Supports:
           - 1D tensor [T] - single sample
           - 2D tensor [B, T] - batch of samples (same length)
           - 2D tensor [1, T] - single sample with channel dim (will be squeezed)
        sr: Target sampling rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length for feature extraction

    Returns:
        log_mel_spec: Log mel spectrogram features
           - [n_mels, T] for single sample input
           - [B, n_mels, T] for batched input
    """
    # Handle input dimensions:
    # - [T] -> [T] (single sample, 1D)
    # - [1, T] -> [T] (single sample with channel dim, squeeze it)
    # - [B, T] where B > 1 -> [B, T] (batch, keep as is)
    if y.dim() == 2 and y.shape[0] == 1:
        # Single sample with channel dim [1, T] -> [T]
        y = y.squeeze(0)
    # For [B, T] batches or [T] single samples, pass through as-is

    # Extract mel spectrogram
    mel_spec, _ = configurable_mel_spectrogram(
        audio=y,
        sample_rate=sr,
        hop_length=hop_length,
        win_length=n_fft,
        n_mels=n_mels,
        n_fft=n_fft,
        f_min=0.0,
        f_max=8000.0,
        power=1,
        normalized=False,
        min_max_energy_norm=False,
        norm="slaney",
        mel_scale="slaney",
        compression=False,
        window_provider=lambda win_size: shared_window_buffer.get_window(win_size, device=y.device)
    )

    # converts to log scale, with clipping to avoid log(0)
    log_mel_spec = torch.log(mel_spec.clamp(min=1e-5))

    return log_mel_spec

def remove_mains_hum(waveform, sample_rate, frequencies=[60, 120, 180, 240]):
    """Remove mains hum and harmonics."""
    for freq in frequencies:
        waveform = torchaudio.functional.bandreject_biquad(
            waveform, sample_rate, central_freq=freq, Q=30.0
        )
    return waveform

def load_audio_dataset(
    dataset_name,
    dataset_config_name: Literal["clean", "other", "all"], 
    split,
    tokenizer: PreTrainedTokenizer,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    max_frames: int,
    shared_window_buffer: SharedWindowBuffer,
    is_voice: bool = True,
    batch_size: int = 100,
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

    if is_voice:
        begin_audio_token_id = tokenizer.convert_tokens_to_ids("<|VOICE|>")
        end_audio_token_id = tokenizer.convert_tokens_to_ids("<|/VOICE|>")
    else:
        begin_audio_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
        end_audio_token_id = tokenizer.convert_tokens_to_ids("<|/AUDIO|>")

    assert isinstance(begin_audio_token_id, int), f"Audio placeholder token should be an integer, got {type(begin_audio_token_id)}"
    assert isinstance(end_audio_token_id, int), f"Audio placeholder token should be an integer, got {type(end_audio_token_id)}"
    
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
        all_audio_waveform_labels = []
        all_target_complex_stfts = []
        for text, audio in zip(texts, audios):
            tokenized = tokenizer(text=text, add_special_tokens=True)
            input_ids = tokenized.input_ids

            # model will interleave embeds between the begin and end tokens; they need to be appended in the text input beforehand so that the embedding process applies appropriately to each token
            transcription_input_ids = [begin_audio_token_id, end_audio_token_id] + input_ids
            generation_input_ids = input_ids + [begin_audio_token_id, end_audio_token_id]
            
            waveforms, y, _ = extract_waveforms(audio, sr=sample_rate)

            # Extract features
            audio_mels = extract_mels(
                shared_window_buffer,
                waveforms,
                sr=sample_rate,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
            )

            # filter mains hum
            waveforms = remove_mains_hum(waveforms.unsqueeze(0), sample_rate).squeeze(0)

            if audio_mels.shape[-1] > max_frames:
                continue
            all_audio_raw_inputs.append(audio_mels)
            all_audio_raw_inputs.append(audio_mels)

            all_input_ids.append(torch.tensor(transcription_input_ids))
            all_input_ids.append(torch.tensor(generation_input_ids))

            all_audio_waveform_labels.append(waveforms)
            all_audio_waveform_labels.append(waveforms)
            all_target_complex_stfts.append(torch.stft(
                waveforms, n_fft, hop_length,
                window=torch.hann_window(n_fft), return_complex=True
            ))

        # Pad sequences to the same length
        max_length = max([len(ids) for ids in all_input_ids])
        all_input_ids = [torch.nn.functional.pad(ids, (0, max_length - len(ids)), value=0) for ids in all_input_ids]

        # Return processed features and transcript
        return {
            "input_ids": torch.stack(all_input_ids),
            "audio_raw_inputs": all_audio_raw_inputs,
            "audio_mel_spec_labels": all_audio_raw_inputs,
            "audio_waveform_labels": all_audio_waveform_labels,
            "audio_waveform_labels_complex_stft": all_target_complex_stfts,
        }
    
    logger.info("Processing dataset with custom feature extraction...")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
    
    dataset = dataset.map(
        process_audio,
        batch_size=batch_size,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    logger.info("Dataset processing complete!")
    return dataset
