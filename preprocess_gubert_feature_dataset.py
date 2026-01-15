#!/usr/bin/env python3
"""
GuBERT Feature Extraction for VAE Training

Extracts features from a trained GuBERT model for training a VAE on the feature space.

Produces sharded datasets containing:
- features: GuBERT features
    - Single layer: [encoder_dim, T']
    - Multi-layer: [num_layers, encoder_dim, T'] (layers stored separately for VAE to normalize)
- feature_lengths: Original lengths before padding
- speaker_embeddings: Optional speaker embeddings for decoder conditioning

Feature extraction options:
- --normalize: Apply LayerNorm to features (default: False for VAE-friendly features)
- --layers: Extract from multiple layers (e.g., --layers -1 -3 for last and third-to-last)
    When multiple layers are specified, they are stored SEPARATELY (not concatenated)
    so the VAE can apply learned per-layer normalization (LayerNorm, RMSNorm, etc.)

Designed for multi-GPU parallel preprocessing:
    GPU 0: python preprocess_gubert_feature_dataset.py --gpu_id 0 --total_gpus 4 ...
    GPU 1: python preprocess_gubert_feature_dataset.py --gpu_id 1 --total_gpus 4 ...
    etc.

Then merge with:
    python shard_utils.py merge-gubert-features --input_dir cached_datasets/gubert_features_raw --output_dir cached_datasets/gubert_features

Example usage:
    # Single layer (default, final layer):
    python preprocess_gubert_feature_dataset.py \
        --gubert_checkpoint runs/gubert/my_run/checkpoint-50000 \
        --gubert_config small \
        --output_dir cached_datasets/gubert_features \
        --split train.360

    # Multi-layer extraction:
    python preprocess_gubert_feature_dataset.py \
        --gubert_checkpoint runs/gubert/my_run/checkpoint-50000 \
        --gubert_config small \
        --output_dir cached_datasets/gubert_features_multilayer \
        --layers -1 -3 \
        --split train.360
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

from dataset_loading.audio_loading import extract_mels, remove_mains_hum
from utils.audio_utils import SharedWindowBuffer
from utils.speaker_encoder import (
    get_speaker_encoder,
    get_speaker_embedding_dim,
    get_speaker_encoder_input_type,
    SpeakerEncoderType,
)


def load_gubert_model(
    checkpoint_path: str,
    config_name: str,
    num_speakers: int = 992,
    device: str = "cuda",
):
    """
    Load a MaskedGuBERT model from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory (containing model.safetensors or pytorch_model.bin)
        config_name: Config name from MASKED_GUBERT_CONFIGS (e.g., "small", "medium")
        num_speakers: Number of speakers the model was trained with
        device: Device to load the model on

    Returns:
        MaskedGuBERTEncoder model in eval mode
    """
    from model.audio.gubert import MaskedGuBERTEncoder

    # Create model with same config
    model = MaskedGuBERTEncoder.from_config(config_name, num_speakers=num_speakers)

    # Try to load from safetensors first, then pytorch_model.bin
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded GuBERT from {safetensors_path}")
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded GuBERT from {pytorch_path}")
    else:
        raise FileNotFoundError(
            f"No model checkpoint found at {checkpoint_path}. "
            f"Expected model.safetensors or pytorch_model.bin"
        )

    model = model.to(device)
    model.eval()

    return model


class GuBERTFeatureBatchProcessor:
    """Batched GPU processing for extracting GuBERT features."""

    def __init__(
        self,
        gubert_model,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        max_audio_seconds: int = 30,
        compute_speaker_embeddings: bool = True,
        speaker_encoder_type: SpeakerEncoderType = "ecapa_tdnn",
        normalize: bool = False,
        layers: Optional[List[int]] = None,
        device: str = "cuda",
    ):
        self.gubert_model = gubert_model
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_audio_seconds = max_audio_seconds
        self.device = device
        self.speaker_encoder_type = speaker_encoder_type
        self.normalize = normalize
        self.layers = layers

        self.audio_max_frames = (max_audio_seconds * sample_rate) // hop_length
        self.shared_window_buffer = SharedWindowBuffer()

        # Get GuBERT encoder dimension
        # Note: we keep layers separate (not concatenated), so encoder_dim is always base_dim
        self.encoder_dim = gubert_model.config.encoder_dim
        self.num_layers = len(layers) if layers is not None else 1

        # Speaker encoder (uses centralized cached singleton)
        self.speaker_encoder = None
        self.speaker_embedding_dim = get_speaker_embedding_dim(speaker_encoder_type)
        self.speaker_encoder_input_type = get_speaker_encoder_input_type(speaker_encoder_type)
        if compute_speaker_embeddings:
            print(f"Loading {speaker_encoder_type} speaker encoder (cached singleton)...")
            self.speaker_encoder = get_speaker_encoder(
                encoder_type=speaker_encoder_type,
                device=device,
            )
            print(f"  Speaker encoder loaded (embedding_dim={self.speaker_embedding_dim}, input={self.speaker_encoder_input_type})")

    @torch.no_grad()
    def process_batch(
        self,
        waveforms: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Process batch of waveforms to GuBERT features and optional speaker embeddings.

        Args:
            waveforms: List of [T] waveform tensors

        Returns:
            Dict with:
                - features: GuBERT features (transposed)
                    - Single layer: [B, encoder_dim, T']
                    - Multi-layer: [B, num_layers, encoder_dim, T']
                - feature_lengths: [B] original lengths before padding
                - mel_lengths: [B] mel spectrogram lengths (before GuBERT subsampling)
                - speaker_embeddings: [B, embedding_dim] speaker embeddings
        """
        batch_size = len(waveforms)

        # Process mel spectrograms
        mel_specs = []
        mel_lengths = []
        waveform_lengths = []

        for waveform in waveforms:
            waveform_lengths.append(len(waveform))

            # Extract mel spectrogram
            mel = extract_mels(
                self.shared_window_buffer,
                waveform,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )

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
        mel_lengths_tensor = torch.tensor(mel_lengths, dtype=torch.long)

        # Extract GuBERT features
        mel_specs_gpu = mel_specs.to(self.device)

        if self.layers is not None and len(self.layers) > 1:
            # Multi-layer extraction: get each layer separately
            layer_features_list = []
            for layer_idx in self.layers:
                layer_features, feature_lengths = self.gubert_model.extract_features(
                    mel_specs_gpu,
                    lengths=mel_lengths_tensor.to(self.device),
                    normalize=self.normalize,
                    layers=[layer_idx],  # Single layer at a time
                )
                # layer_features: [B, T', encoder_dim]
                layer_features_list.append(layer_features)

            # Stack to [B, num_layers, T', encoder_dim]
            features = torch.stack(layer_features_list, dim=1)
            # Transpose to [B, num_layers, encoder_dim, T']
            features = features.permute(0, 1, 3, 2).cpu()
        else:
            # Single layer extraction
            features, feature_lengths = self.gubert_model.extract_features(
                mel_specs_gpu,
                lengths=mel_lengths_tensor.to(self.device),
                normalize=self.normalize,
                layers=self.layers,
            )
            # features: [B, T', encoder_dim]
            # Transpose to [B, encoder_dim, T'] for consistency with mel spec format
            features = features.permute(0, 2, 1).cpu()  # [B, encoder_dim, T']

        # Compute speaker embeddings
        if self.speaker_encoder is not None:
            if self.speaker_encoder_input_type == "waveform":
                # WavLM: needs waveforms padded to same length
                max_waveform_len = max(waveform_lengths)
                padded_waveforms = []
                for waveform in waveforms:
                    if len(waveform) < max_waveform_len:
                        waveform = F.pad(waveform, (0, max_waveform_len - len(waveform)), value=0)
                    padded_waveforms.append(waveform)

                waveform_batch = torch.stack(padded_waveforms).to(self.device)
                waveform_lengths_tensor = torch.tensor(waveform_lengths, dtype=torch.long)

                speaker_embeddings = self.speaker_encoder(
                    waveform=waveform_batch,
                    lengths=waveform_lengths_tensor,
                    sample_rate=self.sample_rate,
                ).cpu()
            else:
                # ECAPA-TDNN: needs mel specs
                speaker_embeddings = self.speaker_encoder(
                    mel_spec=mel_specs,
                    lengths=mel_lengths_tensor,
                ).cpu()
        else:
            speaker_embeddings = torch.zeros(batch_size, self.speaker_embedding_dim)

        result = {
            "features": features,  # [B, encoder_dim, T']
            "feature_lengths": feature_lengths.cpu(),  # [B]
            "mel_lengths": mel_lengths_tensor,  # [B]
            "speaker_embeddings": speaker_embeddings,  # [B, embedding_dim]
        }

        return result


def main():
    parser = argparse.ArgumentParser(description="Extract GuBERT features for VAE training")

    # GuBERT model
    parser.add_argument("--gubert_checkpoint", type=str, required=True,
                        help="Path to GuBERT checkpoint directory")
    parser.add_argument("--gubert_config", type=str, default="small",
                        help="GuBERT config name (tiny, small, medium, large)")
    parser.add_argument("--num_speakers", type=int, default=992,
                        help="Number of speakers the GuBERT model was trained with")

    # Dataset
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for shards")
    parser.add_argument("--dataset_name", type=str, default="openslr/librispeech_asr",
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, default="clean",
                        help="Dataset configuration")
    parser.add_argument("--split", type=str, default="train.360",
                        help="Dataset split")

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
    parser.add_argument("--max_audio_seconds", type=int, default=30,
                        help="Maximum audio length in seconds")

    # Processing
    parser.add_argument("--gpu_batch_size", type=int, default=32,
                        help="Batch size for GPU processing")
    parser.add_argument("--shard_size", type=int, default=2000,
                        help="Number of samples per shard")

    # Speaker embeddings
    parser.add_argument("--compute_speaker_embeddings", action="store_true", default=True)
    parser.add_argument("--no_speaker_embeddings", action="store_false",
                        dest="compute_speaker_embeddings")
    parser.add_argument("--speaker_encoder_type", type=str, default="ecapa_tdnn",
                        choices=["ecapa_tdnn", "wavlm"],
                        help="Speaker encoder type")

    # Feature extraction options
    parser.add_argument("--normalize", action="store_true", default=False,
                        help="Apply LayerNorm to features (default: False for VAE-friendly features)")
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Layer indices for multi-scale features (e.g., --layers -1 -3). "
                             "If not specified, uses final layer only.")

    # Filtering
    parser.add_argument("--min_audio_energy", type=float, default=0.05,
                        help="Minimum audio energy (skip silent samples)")
    parser.add_argument("--min_audio_std", type=float, default=0.02,
                        help="Minimum audio std (skip near-silent samples)")
    parser.add_argument("--remove_mains_hum", action="store_true", default=True,
                        help="Remove 50/60Hz mains hum from audio")
    parser.add_argument("--no_remove_mains_hum", action="store_false",
                        dest="remove_mains_hum")

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

    print(f"GuBERT Feature Extraction")
    print(f"=========================")
    print(f"GPU {args.gpu_id}/{args.total_gpus}")
    print(f"Processing every {args.total_gpus}th sample starting at offset {args.gpu_id}")
    print(f"Output: {output_dir}")
    print(f"")

    # Load GuBERT model
    print(f"Loading GuBERT model from {args.gubert_checkpoint}...")
    gubert_model = load_gubert_model(
        args.gubert_checkpoint,
        args.gubert_config,
        num_speakers=args.num_speakers,
        device=device,
    )
    print(f"  GuBERT config: {args.gubert_config}")
    print(f"  Encoder dimension: {gubert_model.config.encoder_dim}")
    print(f"  Total stride (subsampling): {gubert_model.conv_subsample.total_stride}x")

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

    # Feature extraction settings
    print(f"  Normalize features: {args.normalize}")
    if args.layers:
        print(f"  Multi-scale layers: {args.layers}")

    # Initialize processor
    processor = GuBERTFeatureBatchProcessor(
        gubert_model=gubert_model,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        max_audio_seconds=args.max_audio_seconds,
        compute_speaker_embeddings=args.compute_speaker_embeddings,
        speaker_encoder_type=args.speaker_encoder_type,
        normalize=args.normalize,
        layers=args.layers,
        device=device,
    )
    print(f"  Output feature dimension: {processor.encoder_dim}")

    # Stats
    stats = {
        "processed": 0,
        "saved": 0,
        "skipped_silent": 0,
        "skipped_error": 0,
    }

    # Shard accumulators
    shard_features = []
    shard_feature_lengths = []
    shard_mel_lengths = []
    shard_speaker_embeddings = []
    shard_idx = 0

    def flush_shard():
        nonlocal shard_features, shard_feature_lengths, shard_mel_lengths
        nonlocal shard_speaker_embeddings, shard_idx

        if not shard_features:
            return

        # Find max feature length in this shard for padding
        max_feature_len = max(f.shape[-1] for f in shard_features)

        # Pad features to same length
        padded_features = []
        for feat in shard_features:
            if feat.shape[-1] < max_feature_len:
                feat = F.pad(feat, (0, max_feature_len - feat.shape[-1]), value=0)
            padded_features.append(feat)

        # Shape depends on multi-layer mode:
        # - Single layer: [N, encoder_dim, T']
        # - Multi-layer:  [N, num_layers, encoder_dim, T']
        shard_data = {
            "features": torch.cat(padded_features, dim=0),
            "feature_lengths": torch.cat(shard_feature_lengths, dim=0),
            "mel_lengths": torch.cat(shard_mel_lengths, dim=0),
            "speaker_embeddings": torch.cat(shard_speaker_embeddings, dim=0),
            "num_samples": sum(f.shape[0] for f in shard_features),
        }

        shard_path = os.path.join(output_dir, f"shard_{shard_idx:06d}.pt")
        torch.save(shard_data, shard_path)

        print(f"  Saved shard {shard_idx} ({shard_data['num_samples']} samples)")

        shard_features = []
        shard_feature_lengths = []
        shard_mel_lengths = []
        shard_speaker_embeddings = []
        shard_idx += 1

    # Batch accumulators
    batch_waveforms = []

    def process_and_accumulate():
        nonlocal batch_waveforms
        nonlocal shard_features, shard_feature_lengths, shard_mel_lengths, shard_speaker_embeddings

        if not batch_waveforms:
            return

        try:
            result = processor.process_batch(batch_waveforms)

            # Add to shard
            shard_features.append(result["features"])
            shard_feature_lengths.append(result["feature_lengths"])
            shard_mel_lengths.append(result["mel_lengths"])
            shard_speaker_embeddings.append(result["speaker_embeddings"])

            stats["saved"] += len(batch_waveforms)

            # Flush shard if full
            current_size = sum(f.shape[0] for f in shard_features)
            if current_size >= args.shard_size:
                flush_shard()

        except Exception as e:
            print(f"Batch processing error: {e}")
            import traceback
            traceback.print_exc()
            stats["skipped_error"] += len(batch_waveforms)

        batch_waveforms = []

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

            # Truncate if too long
            max_samples = args.max_audio_seconds * args.sample_rate
            if len(waveform) > max_samples:
                waveform = waveform[:max_samples]

            # Add to batch
            batch_waveforms.append(waveform)

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
        "gubert_checkpoint": args.gubert_checkpoint,
        "gubert_config": args.gubert_config,
        "encoder_dim": gubert_model.config.encoder_dim,
        "total_stride": gubert_model.conv_subsample.total_stride,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "sample_rate": args.sample_rate,
        "n_mels": args.n_mels,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "max_audio_seconds": args.max_audio_seconds,
        "compute_speaker_embeddings": args.compute_speaker_embeddings,
        "speaker_encoder_type": args.speaker_encoder_type,
        "speaker_embedding_dim": processor.speaker_embedding_dim,
        "remove_mains_hum": args.remove_mains_hum,
        "shard_size": args.shard_size,
        # Feature extraction settings
        "normalize": args.normalize,
        "layers": args.layers,  # None for single layer (default), list for multi-layer
        "num_layers": processor.num_layers,  # 1 for single layer, >1 for multi-layer
        "stats": stats,
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*50}")
    print(f"GPU {args.gpu_id} complete!")
    print(f"  Processed: {stats['processed']:,}")
    print(f"  Saved: {stats['saved']:,}")
    print(f"  Skipped (silent): {stats['skipped_silent']:,}")
    print(f"  Skipped (error): {stats['skipped_error']:,}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Speed: {stats['saved']/elapsed:.1f} samples/sec")
    print(f"  Shards: {shard_idx}")
    print(f"  Encoder dimension: {gubert_model.config.encoder_dim}")
    print(f"  Num layers: {processor.num_layers}")
    if args.layers:
        print(f"  Layers: {args.layers}")
    print(f"  Normalize: {args.normalize}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
