import os
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset, Audio
import json

from dataset_loading.audio_loading import extract_waveforms, extract_mels, remove_mains_hum
from model.audio.shared_window_buffer import SharedWindowBuffer
from transformers import T5Tokenizer, T5EncoderModel

from speechbrain.inference.speaker import EncoderClassifier


"""
Uses T5-small to produce and cache text embeddings alongside mel spectrograms.
Optionally uses ECAPA-TDNN (via SpeechBrain) for speaker embeddings.
Optionally encodes mel spectrograms to VAE latents for latent diffusion.
"""


def load_audio_vae(checkpoint_path: str, vae_config: str, latent_channels: int, device: str = "cuda"):
    """
    Load an audio VAE from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory (containing model.safetensors or pytorch_model.bin)
        vae_config: Config name from model_config_lookup (e.g., "mini", "tiny", "mini_deep")
        latent_channels: Number of latent channels the VAE was trained with
        device: Device to load the model on

    Returns:
        VAE model in eval mode
    """
    from model.audio.vae import model_config_lookup

    if vae_config not in model_config_lookup:
        raise ValueError(f"Unknown VAE config: {vae_config}. Available: {list(model_config_lookup.keys())}")

    # Create model with same config
    model = model_config_lookup[vae_config](
        latent_channels=latent_channels,
        perceptual_loss_type="none",  # Don't need loss for inference
    )

    # Try to load from safetensors first, then pytorch_model.bin
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict)
        print(f"Loaded VAE from {safetensors_path}")
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded VAE from {pytorch_path}")
    else:
        raise FileNotFoundError(
            f"No model checkpoint found at {checkpoint_path}. "
            f"Expected model.safetensors or pytorch_model.bin"
        )

    model = model.to(device)
    model.eval()

    return model


@torch.no_grad()
def encode_mel_to_latent(vae, mel_spec: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """
    Encode a mel spectrogram to VAE latent space.

    Args:
        vae: VAE model
        mel_spec: Mel spectrogram tensor [n_mels, T] or [1, n_mels, T]
        device: Device to run on

    Returns:
        Latent mu tensor [latent_channels, H', T']
    """
    # Ensure correct shape: [B, C, H, W] = [1, 1, n_mels, T]
    if mel_spec.dim() == 2:
        mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)  # [n_mels, T] -> [1, 1, n_mels, T]
    elif mel_spec.dim() == 3:
        mel_spec = mel_spec.unsqueeze(0)  # [N, n_mels, T] -> [N, 1, n_mels, T]

    mel_spec = mel_spec.to(device)

    # Get mu from encoder (don't sample, use deterministic mean)
    mu, _ = vae.encoder(mel_spec)

    # Remove batch dimension: [1, C, H', T'] -> [C, H', T']
    return mu.squeeze(0).cpu()

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
    # VAE encoding options
    vae_checkpoint: str = None,
    vae_config: str = "mini",
    latent_channels: int = 16,
):
    """
    Preprocess dataset and save as individual .pt files.

    If vae_checkpoint is provided, mel spectrograms are encoded to VAE latent space
    and saved as 'latent_mu' for latent diffusion training.
    """
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load VAE if checkpoint provided
    vae = None
    if vae_checkpoint is not None:
        print(f"Loading audio VAE from {vae_checkpoint}...")
        vae = load_audio_vae(vae_checkpoint, vae_config, latent_channels, device)
        print(f"  VAE config: {vae_config}, latent_channels: {latent_channels}")

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
    if vae is not None:
        stats["latents_encoded"] = 0

    max_samples = int(segment_length_sec * sample_rate)
    overlap_samples = int(segment_overlap_sec * sample_rate)
    stride = max_samples - overlap_samples

    shared_window_buffer = SharedWindowBuffer()

    text_model = T5EncoderModel.from_pretrained(huggingface_text_model)
    text_tokenizer = T5Tokenizer.from_pretrained(huggingface_text_model)
    text_model.eval()

    # Initialize speaker encoder (ECAPA-TDNN)
    speaker_encoder = None
    if compute_speaker_embeddings:
        print("Loading ECAPA-TDNN speaker encoder...")
        speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        stats["speaker_embeddings_computed"] = 0
    elif compute_speaker_embeddings:
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

                # Encode to VAE latent if VAE is provided
                latent_mu = None
                if vae is not None:
                    latent_mu = encode_mel_to_latent(vae, mel_spec, device)
                    stats["latents_encoded"] += 1

                # Save to file
                save_path = os.path.join(output_dir, f"{idx:08d}_{seg_idx:02d}.pt")
                save_dict = {
                    "text_embeddings": text_embeddings,
                    "text_attention_mask": text_attention_mask,
                    "mel_spec": mel_spec,
                    "mel_spec_length": mel_spec.shape[-1],  # Original length before any padding
                    "speaker_id": speaker_id,
                }
                if speaker_embedding is not None:
                    save_dict["speaker_embedding"] = speaker_embedding
                if latent_mu is not None:
                    save_dict["latent_mu"] = latent_mu
                    save_dict["latent_shape"] = list(latent_mu.shape)  # [C, H', T']
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
    if vae is not None:
        config["vae_config"] = vae_config
        config["vae_checkpoint"] = vae_checkpoint
        config["latent_channels"] = latent_channels

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
    if vae is not None:
        print(f"  Latents encoded: {stats['latents_encoded']}")
    
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

    # VAE encoding options for latent diffusion
    parser.add_argument("--vae_checkpoint", type=str, default=None,
                        help="Path to audio VAE checkpoint directory for latent encoding")
    parser.add_argument("--vae_config", type=str, default="mini",
                        help="VAE config name (tiny, mini, mini_deep)")
    parser.add_argument("--latent_channels", type=int, default=16,
                        help="Number of latent channels in the VAE")

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
        vae_checkpoint=args.vae_checkpoint,
        vae_config=args.vae_config,
        latent_channels=args.latent_channels,
    )
