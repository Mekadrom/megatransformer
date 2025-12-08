import os
import random
import torch
import torchaudio.functional as F
import argparse
from tqdm import tqdm
from datasets import load_dataset, Audio
import json

from dataset_loading.audio_loading import extract_waveforms, extract_mels, remove_mains_hum
from model.audio.shared_window_buffer import SharedWindowBuffer


def apply_augmentations(
    waveform: torch.Tensor,
    sample_rate: int,
    speed_perturb: bool = False,
    speed_range: tuple = (0.9, 1.1),
    gain_perturb: bool = False,
    gain_range_db: tuple = (-6, 6),
    pitch_shift: bool = False,
    pitch_range_semitones: tuple = (-2, 2),
) -> torch.Tensor:
    """
    Apply audio augmentations to waveform.

    Args:
        waveform: 1D tensor of audio samples
        sample_rate: Sample rate of audio
        speed_perturb: Whether to apply speed perturbation (changes duration)
        speed_range: (min, max) speed factor
        gain_perturb: Whether to apply random gain
        gain_range_db: (min, max) gain in dB
        pitch_shift: Whether to apply pitch shifting
        pitch_range_semitones: (min, max) pitch shift in semitones

    Returns:
        Augmented waveform
    """
    # Speed perturbation (resampling trick)
    if speed_perturb:
        speed_factor = random.uniform(*speed_range)
        # Resample to change speed: faster = higher intermediate sr
        intermediate_sr = int(sample_rate * speed_factor)
        waveform = F.resample(waveform, intermediate_sr, sample_rate)

    # Pitch shift (using resampling + speed compensation would change duration,
    # so we use a simpler approach: small pitch shifts via resampling)
    if pitch_shift:
        semitones = random.uniform(*pitch_range_semitones)
        # Pitch ratio: 2^(semitones/12)
        pitch_ratio = 2 ** (semitones / 12)
        # To shift pitch up without changing duration:
        # 1. Resample down (makes audio shorter but higher pitch when played at original sr)
        # 2. Then stretch back (not implemented here - would need phase vocoder)
        # Simple approach: just do the resample, accept slight duration change
        intermediate_sr = int(sample_rate / pitch_ratio)
        if intermediate_sr != sample_rate:
            waveform = F.resample(waveform, sample_rate, intermediate_sr)
            waveform = F.resample(waveform, intermediate_sr, sample_rate)

    # Gain perturbation (simple amplitude scaling)
    if gain_perturb:
        gain_db = random.uniform(*gain_range_db)
        gain_linear = 10 ** (gain_db / 20)
        waveform = waveform * gain_linear
        # Clip to prevent clipping distortion
        waveform = waveform.clamp(-1.0, 1.0)

    return waveform


def preprocess_and_cache_dataset(
    output_dir: str,
    dataset_name: str = "openslr/librispeech_asr",
    dataset_config: str = "clean",
    split: str = "train.360",
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    audio_max_frames: int = 626,
    segment_length_sec: float = 10.0,
    segment_overlap_sec: float = 1.0,
    mel_window: str = "hann_window",
    # Augmentation options
    num_augmented_copies: int = 0,
    speed_perturb: bool = False,
    speed_range: tuple = (0.9, 1.1),
    gain_perturb: bool = False,
    gain_range_db: tuple = (-6, 6),
    pitch_shift: bool = False,
    pitch_range_semitones: tuple = (-2, 2),
    augmentation_seed: int = 42,
):
    """
    Preprocess dataset and save as individual .pt files.

    Augmentation options:
        num_augmented_copies: Number of augmented copies to create per segment (0 = no augmentation).
            The original unaugmented segment is always saved. Setting this to 2 means each segment
            produces 3 files: original + 2 augmented copies.
        speed_perturb: Apply random speed perturbation (changes duration slightly)
        speed_range: (min, max) speed factor, e.g., (0.9, 1.1) for +/- 10%
        gain_perturb: Apply random gain adjustment
        gain_range_db: (min, max) gain in dB, e.g., (-6, 6)
        pitch_shift: Apply random pitch shifting
        pitch_range_semitones: (min, max) semitones, e.g., (-2, 2)
        augmentation_seed: Random seed for reproducibility
    """
    random.seed(augmentation_seed)
    augment_enabled = (speed_perturb or gain_perturb or pitch_shift) and num_augmented_copies > 0
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
        "augmented": 0 if augment_enabled else None,
    }

    if augment_enabled:
        print(f"Augmentation enabled: {num_augmented_copies} copies per segment")
        print(f"  speed_perturb={speed_perturb}, gain_perturb={gain_perturb}, pitch_shift={pitch_shift}")
    
    max_samples = int(segment_length_sec * sample_rate)
    overlap_samples = int(segment_overlap_sec * sample_rate)
    stride = max_samples - overlap_samples

    shared_window_buffer = SharedWindowBuffer()

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
                speaker_id = example.get("speaker_id", None)
                if speaker_id is None:
                    stats["no_speaker_id"] += 1
                    speaker_id = -1

                # Determine how many versions to save: original + augmented copies
                num_versions = 1 + (num_augmented_copies if augment_enabled else 0)

                for aug_idx in range(num_versions):
                    if aug_idx == 0:
                        # Original (unaugmented) version
                        current_segment = segment
                    else:
                        # Augmented copy
                        current_segment = apply_augmentations(
                            segment.clone(),
                            sample_rate,
                            speed_perturb=speed_perturb,
                            speed_range=speed_range,
                            gain_perturb=gain_perturb,
                            gain_range_db=gain_range_db,
                            pitch_shift=pitch_shift,
                            pitch_range_semitones=pitch_range_semitones,
                        )
                        stats["augmented"] += 1

                    # Extract mel spectrogram
                    mel_spec = extract_mels(
                        shared_window_buffer,
                        current_segment,
                        sr=sample_rate,
                        n_mels=n_mels,
                        n_fft=n_fft,
                        hop_length=hop_length,
                    )

                    # Skip if too long
                    if mel_spec.shape[-1] > audio_max_frames + 1:
                        if aug_idx == 0:
                            print(f"Skipping sample {idx} segment {seg_idx} - too long ({mel_spec.shape[-1]} frames)")
                            stats["skipped_too_long"] += 1
                        continue

                    stft = torch.stft(
                        current_segment,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        window=shared_window_buffer.get_window(n_fft, current_segment.device),
                        return_complex=True,
                    )

                    # Save to file
                    if aug_idx == 0:
                        save_path = os.path.join(output_dir, f"{idx:08d}_{seg_idx:02d}.pt")
                    else:
                        save_path = os.path.join(output_dir, f"{idx:08d}_{seg_idx:02d}_aug{aug_idx}.pt")

                    torch.save({
                        "mel_spec": mel_spec,
                        "waveform_labels": current_segment,
                        "target_complex_stfts": stft,
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
        "stats": stats,
        "augmentation": {
            "enabled": augment_enabled,
            "speed_perturb": speed_perturb,
            "speed_range": speed_range if speed_perturb else None,
            "gain_perturb": gain_perturb,
            "gain_range_db": gain_range_db if gain_perturb else None,
            "pitch_shift": pitch_shift,
            "pitch_range_semitones": pitch_range_semitones if pitch_shift else None,
            "seed": augmentation_seed,
        } if augment_enabled else None,
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

    # Dataset options
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="openslr/librispeech_asr")
    parser.add_argument("--dataset_config", type=str, default="clean")
    parser.add_argument("--split", type=str, default="train.360")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--audio_max_frames", type=int, default=626)
    parser.add_argument("--mel_window", type=str, default="hann_window")

    # Augmentation options
    parser.add_argument("--num_augmented_copies", type=int, default=0,
                        help="Number of augmented copies per segment (0 = no augmentation). Original is always saved.")
    parser.add_argument("--speed_perturb", action="store_true", help="Apply random speed perturbation")
    parser.add_argument("--speed_range", type=float, nargs=2, default=[0.9, 1.1], metavar=("MIN", "MAX"),
                        help="Speed perturbation range (default: 0.9 1.1)")
    parser.add_argument("--gain_perturb", action="store_true", help="Apply random gain perturbation")
    parser.add_argument("--gain_range_db", type=float, nargs=2, default=[-6, 6], metavar=("MIN", "MAX"),
                        help="Gain range in dB (default: -6 6)")
    parser.add_argument("--pitch_shift", action="store_true", help="Apply random pitch shifting")
    parser.add_argument("--pitch_range_semitones", type=float, nargs=2, default=[-2, 2], metavar=("MIN", "MAX"),
                        help="Pitch shift range in semitones (default: -2 2)")
    parser.add_argument("--augmentation_seed", type=int, default=42, help="Random seed for augmentation reproducibility")

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
        mel_window=args.mel_window,
        num_augmented_copies=args.num_augmented_copies,
        speed_perturb=args.speed_perturb,
        speed_range=tuple(args.speed_range),
        gain_perturb=args.gain_perturb,
        gain_range_db=tuple(args.gain_range_db),
        pitch_shift=args.pitch_shift,
        pitch_range_semitones=tuple(args.pitch_range_semitones),
        augmentation_seed=args.augmentation_seed,
    )
