import os
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset, Audio
import json

from dataset_loading.audio_loading import extract_waveforms, extract_mels, remove_mains_hum


def preprocess_and_cache_dataset(
    output_dir: str,
    dataset_name: str = "openslr/librispeech_asr",
    dataset_config: str = "clean",
    split: str = "train.360",
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
    audio_max_frames: int = 312,
    max_text_length: int = 256,
    segment_length_sec: float = 10.0,
    segment_overlap_sec: float = 1.0,
):
    """
    Preprocess dataset and save as individual .pt files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset {dataset_name}/{dataset_config} split {split}...")
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
    
    # Track statistics
    stats = {
        "total": len(dataset),
        "saved_samples": 0,
        "saved_segments": 0,
        "skipped_silent": 0,
        "skipped_too_long": 0,
        "skipped_error": 0,
        "no_speaker_id": 0,
    }
    
    max_samples = int(segment_length_sec * sample_rate)
    overlap_samples = int(segment_overlap_sec * sample_rate)
    stride = max_samples - overlap_samples

    # Process each example
    print("Processing examples...")
    for idx in tqdm(range(len(dataset))):
        try:
            example = dataset[idx]
            
            # Extract waveform
            audio = example["audio"]
            waveforms, y, _ = extract_waveforms(audio, sr=sample_rate)
            
            # Skip low-energy audio
            if waveforms.abs().max() < 0.05 or waveforms.std() < 0.02:
                stats["skipped_silent"] += 1
                continue
            
            # Remove mains hum
            waveforms = remove_mains_hum(waveforms.unsqueeze(0), sample_rate).squeeze(0)

            if len(waveforms) > max_samples:
                segments = []
                for start in range(0, len(waveforms) - max_samples + 1, stride):
                    segments.append(waveforms[start:start + max_samples])
                # Include final segment if there's leftover
                if len(waveforms) % stride != 0:
                    segments.append(waveforms[-max_samples:])
            else:
                segments = [waveforms]
            
            for seg_idx, segment in enumerate(segments):
                # Extract mel spectrogram (use filtered waveform)
                mel_spec = extract_mels(
                    segment.numpy(),
                    sr=sample_rate,
                    n_mels=n_mels,
                    n_fft=n_fft,
                    hop_length=hop_length,
                )
                
                # Skip if too long
                if mel_spec.shape[-1] > audio_max_frames + 1:
                    stats["skipped_too_long"] += 1
                    continue
            
                speaker_id = example.get("speaker_id", None)
                if speaker_id is None:
                    stats["no_speaker_id"] += 1
                    speaker_id = -1
                
                # Save to file
                save_path = os.path.join(output_dir, f"{idx:08d}_{seg_idx:02d}.pt")
                torch.save({
                    "mel_spec": mel_spec,
                    "waveform_labels": segment,
                    "speaker_id": speaker_id,
                }, save_path)
                
                stats["saved_segments"] += 1
            stats["saved_samples"] += 1
            
        except Exception as e:
            print(f"Error processing {idx}: {e}")
            stats["skipped_error"] += 1
            continue
    
    # Save stats and config
    config = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "sample_rate": sample_rate,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "audio_max_frames": audio_max_frames,
        "max_text_length": max_text_length,
        "stats": stats,
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nPreprocessing complete!")
    print(f"  Saved samples: {stats['saved_samples']}")
    print(f"  Saved segments: {stats['saved_segments']}")
    print(f"  Skipped (silent): {stats['skipped_silent']}")
    print(f"  Skipped (too long): {stats['skipped_too_long']}")
    print(f"  Skipped (error): {stats['skipped_error']}")
    print(f"  No speaker ID: {stats['no_speaker_id']}")
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="openslr/librispeech_asr")
    parser.add_argument("--dataset_config", type=str, default="clean")
    parser.add_argument("--split", type=str, default="train.360")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--audio_max_frames", type=int, default=312)
    parser.add_argument("--max_text_length", type=int, default=256)
    args = parser.parse_args()
    
    preprocess_and_cache_dataset(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        audio_max_frames=args.audio_max_frames,
        max_text_length=args.max_text_length,
    )
