import os
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset, Audio
import json

from dataset_loading.audio_loading import extract_waveforms, extract_mels, remove_mains_hum
from model.audio.shared_window_buffer import SharedWindowBuffer
from transformers import T5Tokenizer, T5EncoderModel

# Optional: speechbrain for speaker embeddings
try:
    from speechbrain.inference.speaker import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    print("Warning: speechbrain not available. Speaker embeddings will not be computed.")


"""
Uses T5-small to produce and cache text embeddings alongside mel spectrograms.
Optionally uses ECAPA-TDNN (via SpeechBrain) for speaker embeddings.
"""

def preprocess_and_cache_dataset(
    output_dir: str,
    dataset_name: str = "openslr/librispeech_asr",
    dataset_config: str = "clean",
    huggingface_text_model: str = "t5-small",
    split: str = "train.360",
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    audio_max_frames: int = 1875,
    max_conditions: int = 1024,  # max text embeddings sequence length
    segment_length_sec: float = 30.0,
    segment_overlap_sec: float = 1.0,
    mel_window: str = "hann_window",
    enable_segmentation: bool = False,
    compute_speaker_embeddings: bool = True,
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
        "skipped_no_text": 0,
        "skipped_text_too_long": 0,
        "skipped_error": 0,
        "no_speaker_id": 0,
    }
    
    max_samples = int(segment_length_sec * sample_rate)
    overlap_samples = int(segment_overlap_sec * sample_rate)
    stride = max_samples - overlap_samples

    shared_window_buffer = SharedWindowBuffer()

    text_model = T5EncoderModel.from_pretrained(huggingface_text_model)
    text_tokenizer = T5Tokenizer.from_pretrained(huggingface_text_model)
    text_model.eval()

    # Initialize speaker encoder (ECAPA-TDNN)
    speaker_encoder = None
    if compute_speaker_embeddings and SPEECHBRAIN_AVAILABLE:
        print("Loading ECAPA-TDNN speaker encoder...")
        speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
        stats["speaker_embeddings_computed"] = 0
    elif compute_speaker_embeddings and not SPEECHBRAIN_AVAILABLE:
        print("Warning: Speaker embeddings requested but speechbrain not available. Skipping.")

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
                if enable_segmentation:
                    # Break into segments
                    segments = []
                    for start in range(0, len(waveforms) - max_samples + 1, stride):
                        segments.append(waveforms[start:start + max_samples])
                    # Include final segment if there's leftover
                    if len(waveforms) % stride != 0:
                        segments.append(waveforms[-max_samples:])
                else:
                    # skip too long
                    stats["skipped_too_long"] += 1
                    continue
            else:
                segments = [waveforms]
            
            for seg_idx, segment in enumerate(segments):
                # Extract mel spectrogram (use filtered waveform)
                mel_spec = extract_mels(
                    shared_window_buffer,
                    segment,
                    sr=sample_rate,
                    n_mels=n_mels,
                    n_fft=n_fft,
                    hop_length=hop_length,
                )

                # Skip if mel spec for segment too long (shouldn't happen)
                if mel_spec.shape[-1] > audio_max_frames + 1:
                    print(f"Skipping sample {idx} segment {seg_idx} - too long ({mel_spec.shape[-1]} frames)")
                    stats["skipped_too_long"] += 1
                    continue
            
                speaker_id = example.get("speaker_id", None)
                if speaker_id is None:
                    stats["no_speaker_id"] += 1
                    speaker_id = -1

                if "text" not in example or example["text"] is None:
                    stats["skipped_no_text"] += 1
                    continue

                text = example["text"]
                text_inputs = text_tokenizer(
                    text,
                    max_length=max_conditions,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                if text_inputs['input_ids'].shape[1] > max_conditions:
                    stats["skipped_text_too_long"] += 1
                    continue

                text_embeddings = text_model(**text_inputs).last_hidden_state.squeeze(0)
                text_attention_mask = text_inputs['attention_mask'].squeeze(0)  # Remove batch dim

                # Compute speaker embedding if available
                speaker_embedding = None
                if speaker_encoder is not None:
                    with torch.no_grad():
                        # ECAPA-TDNN expects [batch, time] waveform
                        speaker_embedding = speaker_encoder.encode_batch(
                            segment.unsqueeze(0)
                        ).squeeze(0).cpu()  # [192] embedding
                        stats["speaker_embeddings_computed"] += 1

                # Save to file
                save_path = os.path.join(output_dir, f"{idx:08d}_{seg_idx:02d}.pt")
                save_dict = {
                    "text_embeddings": text_embeddings,
                    "text_attention_mask": text_attention_mask,
                    "mel_spec": mel_spec,
                    "speaker_id": speaker_id,
                }
                if speaker_embedding is not None:
                    save_dict["speaker_embedding"] = speaker_embedding
                torch.save(save_dict, save_path)
                
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
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nPreprocessing complete!")
    print(f"  Saved samples: {stats['saved_samples']}")
    print(f"  Saved segments: {stats['saved_segments']}")
    print(f"  Skipped (silent): {stats['skipped_silent']}")
    print(f"  Skipped (too long): {stats['skipped_too_long']}")
    print(f"  Skipped (error): {stats['skipped_error']}")
    print(f"  Skipped (no text): {stats['skipped_no_text']}")
    print(f"  Skipped (text too long): {stats['skipped_text_too_long']}")
    print(f"  No speaker ID: {stats['no_speaker_id']}")
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="openslr/librispeech_asr")
    parser.add_argument("--dataset_config", type=str, default="clean")
    parser.add_argument("--split", type=str, default="train.360")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--audio_max_frames", type=int, default=1875)
    parser.add_argument("--mel_window", type=str, default="hann_window")
    parser.add_argument("--max_conditions", type=int, default=512)
    parser.add_argument("--enable_segmentation", action="store_true")
    parser.add_argument("--compute_speaker_embeddings", action="store_true", default=True)
    parser.add_argument("--no_speaker_embeddings", action="store_false", dest="compute_speaker_embeddings")

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
        max_conditions=args.max_conditions,
        mel_window=args.mel_window,
        enable_segmentation=args.enable_segmentation,
        compute_speaker_embeddings=args.compute_speaker_embeddings,
    )
