#!/usr/bin/env python3
"""
Optimized Audio Diffusion Dataset Preprocessing with Sharding.

Produces sharded datasets containing:
- text_embeddings: T5 text embeddings [T_text, context_dim]
- text_attention_mask: Attention mask for text [T_text]
- mel_spec: Mel spectrogram [n_mels, T] (optional, for reference)
- mel_spec_length: Original length before padding
- speaker_embedding: ECAPA-TDNN speaker embeddings [192]
- latent_mu: VAE-encoded latent [C, H, T'] (if VAE checkpoint provided)

Designed for multi-GPU parallel preprocessing:
    GPU 0: CUDA_VISIBLE_DEVICES=0 python preprocess_audio_diffusion_dataset.py --gpu_id 0 --total_gpus 4 ...
    GPU 1: CUDA_VISIBLE_DEVICES=1 python preprocess_audio_diffusion_dataset.py --gpu_id 1 --total_gpus 4 ...
    etc.

Then merge with:
    python shard_utils.py merge-audio-diffusion --input_dir cached_datasets/audio_diffusion_raw --output_dir cached_datasets/audio_diffusion
"""

import os
import json
import argparse
import time
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import T5Tokenizer, T5EncoderModel

from dataset_loading.audio_loading import extract_mels, remove_mains_hum
from utils.audio_utils import SharedWindowBuffer


def load_speaker_encoder(device: str = "cuda"):
    """Load ECAPA-TDNN mel-spec speaker encoder from SpeechBrain."""
    try:
        from speechbrain.inference.speaker import EncoderClassifier

        encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb-mel-spec",
            savedir="pretrained_models/spkrec-ecapa-voxceleb-mel-spec",
            run_opts={"device": device},
        )
        return encoder
    except ImportError:
        print("Warning: speechbrain not available. Speaker embeddings will be zeros.")
        return None


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
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded VAE from {safetensors_path}")
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded VAE from {pytorch_path}")
    else:
        raise FileNotFoundError(
            f"No model checkpoint found at {checkpoint_path}. "
            f"Expected model.safetensors or pytorch_model.bin"
        )

    model = model.to(device)
    model.eval()

    return model


class AudioDiffusionBatchProcessor:
    """Batched GPU processing for audio diffusion preprocessing."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        audio_max_frames: int = 1875,
        max_text_length: int = 512,
        text_model_name: str = "t5-small",
        compute_speaker_embeddings: bool = True,
        vae_checkpoint: Optional[str] = None,
        vae_config: str = "mini",
        latent_channels: int = 16,
        device: str = "cuda",
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.audio_max_frames = audio_max_frames
        self.max_text_length = max_text_length
        self.device = device

        self.shared_window_buffer = SharedWindowBuffer()

        # Text encoder
        print(f"Loading T5 ({text_model_name})...")
        self.tokenizer = T5Tokenizer.from_pretrained(text_model_name)
        self.text_model = T5EncoderModel.from_pretrained(text_model_name)
        self.text_model.eval().to(device)
        self.context_dim = self.text_model.config.d_model

        # Speaker encoder
        self.speaker_encoder = None
        if compute_speaker_embeddings:
            print("Loading ECAPA-TDNN speaker encoder...")
            self.speaker_encoder = load_speaker_encoder(device)
            if self.speaker_encoder is not None:
                print("  Speaker encoder loaded successfully")

        # VAE for latent encoding
        self.vae = None
        if vae_checkpoint is not None:
            print(f"Loading audio VAE from {vae_checkpoint}...")
            self.vae = load_audio_vae(vae_checkpoint, vae_config, latent_channels, device)
            print(f"  VAE config: {vae_config}, latent_channels: {latent_channels}")

    @torch.no_grad()
    def process_batch(
        self,
        waveforms: List[torch.Tensor],
        texts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Process batch of waveforms and texts.

        Args:
            waveforms: List of [T] waveform tensors
            texts: List of text strings

        Returns:
            Dict with batched tensors
        """
        batch_size = len(waveforms)

        # Process mel spectrograms
        mel_specs = []
        mel_lengths = []

        for waveform in waveforms:
            # Extract mel spectrogram
            mel = extract_mels(
                self.shared_window_buffer,
                waveform,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )

            # Store original length
            mel_length = min(mel.shape[-1], self.audio_max_frames)
            mel_lengths.append(mel_length)

            # Pad or truncate to max frames
            if mel.shape[-1] < self.audio_max_frames:
                mel = F.pad(mel, (0, self.audio_max_frames - mel.shape[-1]), value=0)
            elif mel.shape[-1] > self.audio_max_frames:
                mel = mel[..., :self.audio_max_frames]

            mel_specs.append(mel)

        # Stack mel specs: [B, n_mels, T]
        mel_specs = torch.stack(mel_specs)
        mel_lengths = torch.tensor(mel_lengths, dtype=torch.long)

        # Encode text (batched)
        text_inputs = self.tokenizer(
            texts,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        text_embeddings = self.text_model(**text_inputs).last_hidden_state.cpu()  # [B, T_text, context_dim]
        text_attention_masks = text_inputs['attention_mask'].cpu()  # [B, T_text]

        # Compute speaker embeddings from mel specs
        if self.speaker_encoder is not None:
            # mel_specs is [B, n_mels, T]
            # ECAPA-TDNN mel-spec model expects [B, T, n_mels] and direct calls
            # (encode_batch assumes compute_features exists, which mel-spec model lacks)
            mel_for_ecapa = mel_specs.transpose(1, 2).to(self.device)  # [B, T, n_mels]

            # Compute relative lengths (1.0 for max length, proportional for others)
            max_len = mel_for_ecapa.shape[1]
            rel_lens = torch.tensor([l / max_len for l in mel_lengths.tolist()], device=self.device)

            # Call normalizer and embedding model directly
            normalized = self.speaker_encoder.mods.normalizer(mel_for_ecapa, rel_lens, epoch=1)
            speaker_embeddings = self.speaker_encoder.mods.embedding_model(
                normalized, rel_lens
            ).squeeze(1).cpu()  # [B, 192]
        else:
            # Zeros if no speaker encoder
            speaker_embeddings = torch.zeros(batch_size, 192)

        result = {
            "mel_specs": mel_specs,
            "mel_lengths": mel_lengths,
            "text_embeddings": text_embeddings,
            "text_attention_masks": text_attention_masks,
            "speaker_embeddings": speaker_embeddings,
        }

        # Encode to VAE latent if VAE is provided
        if self.vae is not None:
            # Add channel dimension: [B, n_mels, T] -> [B, 1, n_mels, T]
            mel_input = mel_specs.unsqueeze(1).to(self.device)
            latent_mu, _ = self.vae.encoder(mel_input)
            result["latent_mus"] = latent_mu.cpu()  # [B, C, H, T']
            result["latent_shapes"] = [list(latent_mu.shape[1:])] * batch_size

        return result


def save_shard(shard_data: Dict[str, torch.Tensor], shard_path: str):
    """Save a shard to disk."""
    torch.save(shard_data, shard_path)


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio dataset for diffusion training")

    # Dataset
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for shards")
    parser.add_argument("--dataset_name", type=str, default="openslr/librispeech_asr",
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, default="clean",
                        help="Dataset configuration")
    parser.add_argument("--split", type=str, default="train.360",
                        help="Dataset split")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Column name for text transcription")

    # Multi-GPU
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="This GPU's ID (0-indexed)")
    parser.add_argument("--total_gpus", type=int, default=1,
                        help="Total number of GPUs preprocessing in parallel")

    # Audio settings
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--audio_max_frames", type=int, default=1875,
                        help="Maximum mel spectrogram frames (1875 = ~30sec at 16kHz)")

    # Text settings
    parser.add_argument("--text_model", type=str, default="t5-small",
                        help="T5 model for text encoding")
    parser.add_argument("--max_text_length", type=int, default=512,
                        help="Maximum text sequence length")

    # Processing
    parser.add_argument("--gpu_batch_size", type=int, default=16,
                        help="Batch size for GPU processing")
    parser.add_argument("--shard_size", type=int, default=2000,
                        help="Number of samples per shard")

    # Speaker embeddings
    parser.add_argument("--compute_speaker_embeddings", action="store_true", default=True)
    parser.add_argument("--no_speaker_embeddings", action="store_false",
                        dest="compute_speaker_embeddings")

    # VAE encoding for latent diffusion
    parser.add_argument("--vae_checkpoint", type=str, default=None,
                        help="Path to audio VAE checkpoint directory for latent encoding")
    parser.add_argument("--vae_config", type=str, default="mini",
                        help="VAE config name (tiny, mini, mini_deep, small, etc.)")
    parser.add_argument("--latent_channels", type=int, default=16,
                        help="Number of latent channels in the VAE")

    # Filtering
    parser.add_argument("--min_audio_energy", type=float, default=0.05,
                        help="Minimum audio energy (skip silent samples)")
    parser.add_argument("--min_audio_std", type=float, default=0.02,
                        help="Minimum audio std (skip near-silent samples)")
    parser.add_argument("--remove_mains_hum", action="store_true", default=True,
                        help="Remove 50/60Hz mains hum from audio")
    parser.add_argument("--no_remove_mains_hum", action="store_false",
                        dest="remove_mains_hum")

    # Include mel specs in output (for debugging/reference)
    parser.add_argument("--include_mel_specs", action="store_true", default=False,
                        help="Include mel spectrograms in output (increases storage)")

    # Limits
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to process (for testing)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting index in dataset")

    args = parser.parse_args()

    # Create output directory with GPU suffix for parallel processing
    if args.total_gpus > 1:
        output_dir = os.path.join(args.output_dir, f"gpu_{args.gpu_id}")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Audio Diffusion Dataset Preprocessing")
    print(f"=====================================")
    print(f"GPU {args.gpu_id}/{args.total_gpus}")
    print(f"Processing every {args.total_gpus}th sample starting at offset {args.gpu_id}")
    print(f"Output: {output_dir}")
    print(f"")

    # Load dataset
    print(f"Loading dataset {args.dataset_name}/{args.dataset_config} split {args.split}...")
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
        trust_remote_code=True,
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=args.sample_rate))
    print(f"  Total samples in dataset: {len(dataset):,}")

    # Calculate samples for this GPU
    samples_for_this_gpu = len([
        i for i in range(args.start_idx, len(dataset))
        if i % args.total_gpus == args.gpu_id
    ])
    print(f"  Samples for this GPU: {samples_for_this_gpu:,}")

    # Initialize processor
    processor = AudioDiffusionBatchProcessor(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        audio_max_frames=args.audio_max_frames,
        max_text_length=args.max_text_length,
        text_model_name=args.text_model,
        compute_speaker_embeddings=args.compute_speaker_embeddings,
        vae_checkpoint=args.vae_checkpoint,
        vae_config=args.vae_config,
        latent_channels=args.latent_channels,
        device=device,
    )

    # Stats
    stats = {
        "processed": 0,
        "saved": 0,
        "skipped_silent": 0,
        "skipped_too_long": 0,
        "skipped_no_text": 0,
        "skipped_error": 0,
    }

    # Shard accumulators
    shard_mel_specs = []
    shard_mel_lengths = []
    shard_text_embeddings = []
    shard_text_attention_masks = []
    shard_speaker_embeddings = []
    shard_latent_mus = []
    shard_idx = 0
    has_latents = args.vae_checkpoint is not None

    def flush_shard():
        nonlocal shard_mel_specs, shard_mel_lengths, shard_text_embeddings
        nonlocal shard_text_attention_masks, shard_speaker_embeddings, shard_latent_mus, shard_idx

        if not shard_text_embeddings:
            return

        shard_data = {
            "text_embeddings": torch.cat(shard_text_embeddings, dim=0),
            "text_attention_masks": torch.cat(shard_text_attention_masks, dim=0),
            "mel_lengths": torch.cat(shard_mel_lengths, dim=0),
            "speaker_embeddings": torch.cat(shard_speaker_embeddings, dim=0),
            "num_samples": sum(x.shape[0] for x in shard_text_embeddings),
        }

        if args.include_mel_specs and shard_mel_specs:
            shard_data["mel_specs"] = torch.cat(shard_mel_specs, dim=0)

        if has_latents and shard_latent_mus:
            shard_data["latent_mus"] = torch.cat(shard_latent_mus, dim=0)

        shard_path = os.path.join(output_dir, f"shard_{shard_idx:06d}.pt")
        torch.save(shard_data, shard_path)

        print(f"  Saved shard {shard_idx} ({shard_data['num_samples']} samples)")

        shard_mel_specs = []
        shard_mel_lengths = []
        shard_text_embeddings = []
        shard_text_attention_masks = []
        shard_speaker_embeddings = []
        shard_latent_mus = []
        shard_idx += 1

    # Batch accumulators
    batch_waveforms = []
    batch_texts = []

    def process_and_accumulate():
        nonlocal batch_waveforms, batch_texts
        nonlocal shard_mel_specs, shard_mel_lengths, shard_text_embeddings
        nonlocal shard_text_attention_masks, shard_speaker_embeddings, shard_latent_mus

        if not batch_waveforms:
            return

        try:
            result = processor.process_batch(batch_waveforms, batch_texts)

            # Add to shard
            if args.include_mel_specs:
                shard_mel_specs.append(result["mel_specs"])
            shard_mel_lengths.append(result["mel_lengths"])
            shard_text_embeddings.append(result["text_embeddings"])
            shard_text_attention_masks.append(result["text_attention_masks"])
            shard_speaker_embeddings.append(result["speaker_embeddings"])

            if has_latents and "latent_mus" in result:
                shard_latent_mus.append(result["latent_mus"])

            stats["saved"] += len(batch_waveforms)

            # Flush shard if full
            current_size = sum(x.shape[0] for x in shard_text_embeddings)
            if current_size >= args.shard_size:
                flush_shard()

        except Exception as e:
            print(f"Batch processing error: {e}")
            stats["skipped_error"] += len(batch_waveforms)

        batch_waveforms = []
        batch_texts = []

    # Main processing loop
    start_time = time.time()
    pbar = tqdm(total=samples_for_this_gpu, desc=f"GPU {args.gpu_id}")

    for idx in range(args.start_idx, len(dataset)):
        # Skip if not our sample (multi-GPU distribution)
        if idx % args.total_gpus != args.gpu_id:
            continue

        # Check max samples limit
        if args.max_samples and stats["saved"] >= args.max_samples:
            break

        try:
            example = dataset[idx]

            # Extract waveform
            audio = example["audio"]
            waveform = torch.tensor(audio["array"], dtype=torch.float32)

            # Skip silent/near-silent audio
            if waveform.abs().max() < args.min_audio_energy or waveform.std() < args.min_audio_std:
                stats["skipped_silent"] += 1
                pbar.update(1)
                continue

            # Remove mains hum if enabled
            if args.remove_mains_hum:
                waveform = remove_mains_hum(waveform.unsqueeze(0), args.sample_rate).squeeze(0)

            # Check length (max_frames corresponds to ~30 sec at default settings)
            max_samples = args.audio_max_frames * args.hop_length
            if len(waveform) > max_samples:
                # Truncate to max length
                waveform = waveform[:max_samples]

            # Get text
            text = example.get(args.text_column, None)
            if text is None or not str(text).strip():
                stats["skipped_no_text"] += 1
                pbar.update(1)
                continue

            # Add to batch
            batch_waveforms.append(waveform)
            batch_texts.append(str(text))

            # Process batch when full
            if len(batch_waveforms) >= args.gpu_batch_size:
                process_and_accumulate()

            stats["processed"] += 1
            pbar.update(1)

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            stats["skipped_error"] += 1
            pbar.update(1)
            continue

    # Final flush
    process_and_accumulate()
    flush_shard()

    pbar.close()
    elapsed = time.time() - start_time

    # Save stats and config
    stats["elapsed_seconds"] = elapsed
    stats["samples_per_second"] = stats["saved"] / elapsed if elapsed > 0 else 0
    stats["gpu_id"] = args.gpu_id
    stats["total_gpus"] = args.total_gpus
    stats["num_shards"] = shard_idx

    config = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "sample_rate": args.sample_rate,
        "n_mels": args.n_mels,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "audio_max_frames": args.audio_max_frames,
        "text_model": args.text_model,
        "max_text_length": args.max_text_length,
        "compute_speaker_embeddings": args.compute_speaker_embeddings,
        "remove_mains_hum": args.remove_mains_hum,
        "include_mel_specs": args.include_mel_specs,
        "shard_size": args.shard_size,
        "has_latents": has_latents,
        "stats": stats,
    }
    if has_latents:
        config["vae_config"] = args.vae_config
        config["vae_checkpoint"] = args.vae_checkpoint
        config["latent_channels"] = args.latent_channels

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*50}")
    print(f"GPU {args.gpu_id} complete!")
    print(f"  Processed: {stats['processed']:,}")
    print(f"  Saved: {stats['saved']:,}")
    print(f"  Skipped (silent): {stats['skipped_silent']:,}")
    print(f"  Skipped (no text): {stats['skipped_no_text']:,}")
    print(f"  Skipped (error): {stats['skipped_error']:,}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Speed: {stats['saved']/elapsed:.1f} samples/sec")
    print(f"  Shards: {shard_idx}")
    print(f"  Output: {output_dir}")
    if has_latents:
        print(f"  Latent diffusion: enabled (VAE: {args.vae_config})")


if __name__ == "__main__":
    main()