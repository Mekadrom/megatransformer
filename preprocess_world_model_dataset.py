#!/usr/bin/env python3
"""
World Model Dataset Preprocessing with Sharding.

Produces sharded datasets for multimodal autoregressive world model training.
Each shard contains a list of samples, where each sample is a dict with:
- text_input_ids: Tokenized text with special tokens (always present)
- audio_mel_spec_latents: Audio VAE latents (optional)
- voice_mel_spec_latents: Voice VAE latents (optional)
- image_latents: Image VAE latents (optional)
- task_type: String describing the task (e.g., "audio_transcription", "audio_generation")

Supports multiple modality types:
- text_only: Text-only datasets (no duplication)
- text_audio: Text + audio pairs (duplicated for transcription and generation)
- text_voice: Text + voice pairs (duplicated for transcription and generation)
- text_image: Text + image pairs (duplicated for description and generation)

Designed for multi-GPU parallel preprocessing:
    GPU 0: python preprocess_world_model_dataset.py --gpu_id 0 --total_gpus 4 ...
    GPU 1: python preprocess_world_model_dataset.py --gpu_id 1 --total_gpus 4 ...
    etc.

Then merge multiple datasets with:
    python shard_utils.py merge-world-model \
        --input_dirs cached_datasets/wm_raw_librispeech cached_datasets/wm_raw_laion \
        --output_dir cached_datasets/world_model_merged
"""

import os
import json
import argparse
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Literal

import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import AutoTokenizer

from dataset_loading.audio_loading import extract_mels, remove_mains_hum
from utils.audio_utils import SharedWindowBuffer


# Special tokens for world model
SPECIAL_TOKENS = {
    "bos": "<BOS>",
    "eos": "<EOS>",
    "boa": "<BOA>",  # Begin of audio
    "eoa": "<EOA>",  # End of audio
    "bov": "<BOV>",  # Begin of voice
    "eov": "<EOV>",  # End of voice
    "boi": "<BOI>",  # Begin of image
    "eoi": "<EOI>",  # End of image
    "audio_placeholder": "<|AUDIO_PLACEHOLDER|>",
    "voice_placeholder": "<|VOICE_PLACEHOLDER|>",
    "image_placeholder": "<|IMAGE_PLACEHOLDER|>",
}


ModalityType = Literal["text_only", "text_audio", "text_voice", "text_image"]


@dataclass
class WorldModelSample:
    """A single sample for world model training."""
    text_input_ids: torch.Tensor
    task_type: str
    audio_mel_spec_latents: Optional[torch.Tensor] = None
    voice_mel_spec_latents: Optional[torch.Tensor] = None
    image_latents: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict:
        """Convert to dict for serialization."""
        d = {
            "text_input_ids": self.text_input_ids,
            "task_type": self.task_type,
        }
        if self.audio_mel_spec_latents is not None:
            d["audio_mel_spec_latents"] = self.audio_mel_spec_latents
        if self.voice_mel_spec_latents is not None:
            d["voice_mel_spec_latents"] = self.voice_mel_spec_latents
        if self.image_latents is not None:
            d["image_latents"] = self.image_latents
        return d


def load_audio_vae(checkpoint_path: str, vae_config: str, latent_channels: int, device: str = "cuda"):
    """
    Load an audio VAE encoder from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory
        vae_config: Config name from model_config_lookup
        latent_channels: Number of latent channels
        device: Device to load the model on

    Returns:
        VAE model in eval mode
    """
    from model.audio.vae import model_config_lookup

    if vae_config not in model_config_lookup:
        raise ValueError(f"Unknown audio VAE config: {vae_config}. Available: {list(model_config_lookup.keys())}")

    model = model_config_lookup[vae_config](
        latent_channels=latent_channels,
        perceptual_loss_type="none",
    )

    # Load checkpoint
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded audio VAE from {safetensors_path}")
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded audio VAE from {pytorch_path}")
    else:
        raise FileNotFoundError(f"No model checkpoint found at {checkpoint_path}")

    model = model.to(device)
    model.eval()
    return model


def load_image_vae(checkpoint_path: str, vae_config: str, latent_channels: int, device: str = "cuda"):
    """
    Load an image VAE encoder from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory
        vae_config: Config name from model_config_lookup
        latent_channels: Number of latent channels
        device: Device to load the model on

    Returns:
        VAE model in eval mode
    """
    from model.image.vae import model_config_lookup

    if vae_config not in model_config_lookup:
        raise ValueError(f"Unknown image VAE config: {vae_config}. Available: {list(model_config_lookup.keys())}")

    model = model_config_lookup[vae_config](
        latent_channels=latent_channels,
        perceptual_loss_type="none",
    )

    # Load checkpoint
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded image VAE from {safetensors_path}")
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded image VAE from {pytorch_path}")
    else:
        raise FileNotFoundError(f"No model checkpoint found at {checkpoint_path}")

    model = model.to(device)
    model.eval()
    return model


class WorldModelBatchProcessor:
    """Batched GPU processing for world model preprocessing."""

    def __init__(
        self,
        modality: ModalityType,
        tokenizer_name: str,
        device: str = "cuda",
        # Audio settings
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        audio_max_frames: int = 1875,
        audio_vae_checkpoint: Optional[str] = None,
        audio_vae_config: str = "small",
        audio_latent_channels: int = 32,
        # Voice settings (can share audio VAE or have separate)
        voice_vae_checkpoint: Optional[str] = None,
        voice_vae_config: Optional[str] = None,
        voice_latent_channels: Optional[int] = None,
        # Image settings
        image_size: int = 256,
        image_vae_checkpoint: Optional[str] = None,
        image_vae_config: str = "small",
        image_latent_channels: int = 4,
        # Processing
        remove_mains_hum: bool = True,
    ):
        self.modality = modality
        self.device = device
        self.remove_mains_hum_flag = remove_mains_hum

        # Audio settings
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.audio_max_frames = audio_max_frames

        # Image settings
        self.image_size = image_size

        # Load tokenizer and add special tokens
        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Add special tokens
        special_tokens_list = list(SPECIAL_TOKENS.values())
        num_added = self.tokenizer.add_special_tokens({
            "additional_special_tokens": special_tokens_list
        })
        print(f"  Added {num_added} special tokens")

        # Cache special token IDs
        self.special_token_ids = {
            name: self.tokenizer.convert_tokens_to_ids(token)
            for name, token in SPECIAL_TOKENS.items()
        }
        print(f"  Special token IDs: {self.special_token_ids}")

        # Shared window buffer for audio processing
        self.shared_window_buffer = SharedWindowBuffer()

        # Load VAE encoders based on modality
        self.audio_vae = None
        self.voice_vae = None
        self.image_vae = None

        if modality == "text_audio" and audio_vae_checkpoint:
            print(f"Loading audio VAE from {audio_vae_checkpoint}...")
            self.audio_vae = load_audio_vae(
                audio_vae_checkpoint, audio_vae_config, audio_latent_channels, device
            )

        if modality == "text_voice":
            # Voice can use same VAE as audio or a separate one
            voice_ckpt = voice_vae_checkpoint or audio_vae_checkpoint
            voice_cfg = voice_vae_config or audio_vae_config
            voice_ch = voice_latent_channels or audio_latent_channels
            if voice_ckpt:
                print(f"Loading voice VAE from {voice_ckpt}...")
                self.voice_vae = load_audio_vae(voice_ckpt, voice_cfg, voice_ch, device)

        if modality == "text_image" and image_vae_checkpoint:
            print(f"Loading image VAE from {image_vae_checkpoint}...")
            self.image_vae = load_image_vae(
                image_vae_checkpoint, image_vae_config, image_latent_channels, device
            )

    def _format_text_with_modality(
        self,
        text: str,
        modality: str,
        task: str,  # "transcription" or "generation"
    ) -> str:
        """
        Format text with special tokens for the given modality and task.

        For transcription: <BOS><BOX><PLACEHOLDER><EOX>text<EOS>
        For generation: <BOS>text<BOX><PLACEHOLDER><EOX><EOS>
        """
        bos = SPECIAL_TOKENS["bos"]
        eos = SPECIAL_TOKENS["eos"]

        if modality == "audio":
            bo = SPECIAL_TOKENS["boa"]
            eo = SPECIAL_TOKENS["eoa"]
            placeholder = SPECIAL_TOKENS["audio_placeholder"]
        elif modality == "voice":
            bo = SPECIAL_TOKENS["bov"]
            eo = SPECIAL_TOKENS["eov"]
            placeholder = SPECIAL_TOKENS["voice_placeholder"]
        elif modality == "image":
            bo = SPECIAL_TOKENS["boi"]
            eo = SPECIAL_TOKENS["eoi"]
            placeholder = SPECIAL_TOKENS["image_placeholder"]
        else:
            raise ValueError(f"Unknown modality: {modality}")

        if task == "transcription":
            # Media first, then text
            return f"{bos}{bo}{placeholder}{eo}{text}{eos}"
        elif task == "generation":
            # Text first, then media
            return f"{bos}{text}{bo}{placeholder}{eo}{eos}"
        else:
            raise ValueError(f"Unknown task: {task}")

    def _format_text_only(self, text: str) -> str:
        """Format text-only sample."""
        return f"{SPECIAL_TOKENS['bos']}{text}{SPECIAL_TOKENS['eos']}"

    @torch.no_grad()
    def process_audio_batch(
        self,
        waveforms: List[torch.Tensor],
        texts: List[str],
        modality_key: str,  # "audio" or "voice"
    ) -> List[WorldModelSample]:
        """
        Process batch of audio/voice waveforms and texts.

        Returns two samples per input: transcription and generation.
        """
        vae = self.audio_vae if modality_key == "audio" else self.voice_vae
        latent_key = f"{modality_key}_mel_spec_latents"

        samples = []

        for waveform, text in zip(waveforms, texts):
            # Extract mel spectrogram
            mel = extract_mels(
                self.shared_window_buffer,
                waveform,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )

            # Pad or truncate
            if mel.shape[-1] < self.audio_max_frames:
                mel = F.pad(mel, (0, self.audio_max_frames - mel.shape[-1]), value=0)
            elif mel.shape[-1] > self.audio_max_frames:
                mel = mel[..., :self.audio_max_frames]

            # Encode to latent if VAE provided
            latent = None
            if vae is not None:
                # [n_mels, T] -> [1, 1, n_mels, T]
                mel_input = mel.unsqueeze(0).unsqueeze(0).to(self.device)
                latent_mu, _ = vae.encoder(mel_input)
                latent = latent_mu.squeeze(0).cpu()  # [C, H, T']

            # Create transcription sample (media -> text)
            transcription_text = self._format_text_with_modality(text, modality_key, "transcription")
            transcription_ids = self.tokenizer.encode(
                transcription_text, add_special_tokens=False, return_tensors="pt"
            ).squeeze(0)

            transcription_sample = WorldModelSample(
                text_input_ids=transcription_ids,
                task_type=f"{modality_key}_transcription",
            )
            if latent is not None:
                setattr(transcription_sample, latent_key, latent)
            samples.append(transcription_sample)

            # Create generation sample (text -> media)
            generation_text = self._format_text_with_modality(text, modality_key, "generation")
            generation_ids = self.tokenizer.encode(
                generation_text, add_special_tokens=False, return_tensors="pt"
            ).squeeze(0)

            generation_sample = WorldModelSample(
                text_input_ids=generation_ids,
                task_type=f"{modality_key}_generation",
            )
            if latent is not None:
                setattr(generation_sample, latent_key, latent)
            samples.append(generation_sample)

        return samples

    @torch.no_grad()
    def process_image_batch(
        self,
        images: List[torch.Tensor],  # List of [3, H, W] tensors
        texts: List[str],
    ) -> List[WorldModelSample]:
        """
        Process batch of images and texts.

        Returns two samples per input: description and generation.
        """
        samples = []

        for image, text in zip(images, texts):
            # Encode to latent if VAE provided
            latent = None
            if self.image_vae is not None:
                # [3, H, W] -> [1, 3, H, W]
                image_input = image.unsqueeze(0).to(self.device)
                latent_mu, _ = self.image_vae.encoder(image_input)
                latent = latent_mu.squeeze(0).cpu()  # [C, H', W']

            # Create description sample (image -> text)
            description_text = self._format_text_with_modality(text, "image", "transcription")
            description_ids = self.tokenizer.encode(
                description_text, add_special_tokens=False, return_tensors="pt"
            ).squeeze(0)

            description_sample = WorldModelSample(
                text_input_ids=description_ids,
                task_type="image_description",
                image_latents=latent,
            )
            samples.append(description_sample)

            # Create generation sample (text -> image)
            generation_text = self._format_text_with_modality(text, "image", "generation")
            generation_ids = self.tokenizer.encode(
                generation_text, add_special_tokens=False, return_tensors="pt"
            ).squeeze(0)

            generation_sample = WorldModelSample(
                text_input_ids=generation_ids,
                task_type="image_generation",
                image_latents=latent,
            )
            samples.append(generation_sample)

        return samples

    def process_text_only_batch(self, texts: List[str]) -> List[WorldModelSample]:
        """Process batch of text-only samples (no duplication)."""
        samples = []
        for text in texts:
            formatted = self._format_text_only(text)
            input_ids = self.tokenizer.encode(
                formatted, add_special_tokens=False, return_tensors="pt"
            ).squeeze(0)

            samples.append(WorldModelSample(
                text_input_ids=input_ids,
                task_type="text_only",
            ))

        return samples


def save_shard(samples: List[WorldModelSample], shard_path: str):
    """Save a shard to disk."""
    shard_data = {
        "samples": [s.to_dict() for s in samples],
        "num_samples": len(samples),
    }
    torch.save(shard_data, shard_path)


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for world model training")

    # Dataset
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for shards")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset configuration")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Column name for text")
    parser.add_argument("--audio_column", type=str, default="audio",
                        help="Column name for audio (for audio/voice modalities)")
    parser.add_argument("--image_column", type=str, default="image",
                        help="Column name for image")

    # Modality
    parser.add_argument("--modality", type=str, required=True,
                        choices=["text_only", "text_audio", "text_voice", "text_image"],
                        help="Modality type to process")

    # Tokenizer
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.2-1B",
                        help="HuggingFace tokenizer name")

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
                        help="Maximum mel spectrogram frames")

    # Audio VAE
    parser.add_argument("--audio_vae_checkpoint", type=str, default=None,
                        help="Path to audio VAE checkpoint")
    parser.add_argument("--audio_vae_config", type=str, default="small",
                        help="Audio VAE config name")
    parser.add_argument("--audio_latent_channels", type=int, default=32,
                        help="Audio VAE latent channels")

    # Voice VAE (optional, defaults to audio VAE settings)
    parser.add_argument("--voice_vae_checkpoint", type=str, default=None,
                        help="Path to voice VAE checkpoint (defaults to audio VAE)")
    parser.add_argument("--voice_vae_config", type=str, default=None,
                        help="Voice VAE config name (defaults to audio VAE)")
    parser.add_argument("--voice_latent_channels", type=int, default=None,
                        help="Voice VAE latent channels (defaults to audio)")

    # Image settings
    parser.add_argument("--image_size", type=int, default=256,
                        help="Target image size")
    parser.add_argument("--image_vae_checkpoint", type=str, default=None,
                        help="Path to image VAE checkpoint")
    parser.add_argument("--image_vae_config", type=str, default="small",
                        help="Image VAE config name")
    parser.add_argument("--image_latent_channels", type=int, default=4,
                        help="Image VAE latent channels")

    # Processing
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for processing")
    parser.add_argument("--shard_size", type=int, default=2000,
                        help="Number of samples per shard")

    # Audio filtering
    parser.add_argument("--min_audio_energy", type=float, default=0.05,
                        help="Minimum audio energy (skip silent samples)")
    parser.add_argument("--min_audio_std", type=float, default=0.02,
                        help="Minimum audio std")
    parser.add_argument("--remove_mains_hum", action="store_true", default=True)
    parser.add_argument("--no_remove_mains_hum", action="store_false", dest="remove_mains_hum")

    # Limits
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to process (for testing)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting index in dataset")

    args = parser.parse_args()

    # Create output directory with GPU suffix
    if args.total_gpus > 1:
        output_dir = os.path.join(args.output_dir, f"gpu_{args.gpu_id}")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"World Model Dataset Preprocessing")
    print(f"==================================")
    print(f"Modality: {args.modality}")
    print(f"GPU {args.gpu_id}/{args.total_gpus}")
    print(f"Output: {output_dir}")
    print()

    # Load dataset
    print(f"Loading dataset {args.dataset_name}" +
          (f"/{args.dataset_config}" if args.dataset_config else "") +
          f" split {args.split}...")

    load_kwargs = {"trust_remote_code": True}
    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split, **load_kwargs)
    else:
        dataset = load_dataset(args.dataset_name, split=args.split, **load_kwargs)

    # Cast audio column if needed
    if args.modality in ["text_audio", "text_voice"]:
        dataset = dataset.cast_column(args.audio_column, Audio(sampling_rate=args.sample_rate))

    print(f"  Total samples in dataset: {len(dataset):,}")

    # Calculate samples for this GPU
    samples_for_this_gpu = len([
        i for i in range(args.start_idx, len(dataset))
        if i % args.total_gpus == args.gpu_id
    ])
    print(f"  Samples for this GPU: {samples_for_this_gpu:,}")

    # Initialize processor
    processor = WorldModelBatchProcessor(
        modality=args.modality,
        tokenizer_name=args.tokenizer,
        device=device,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        audio_max_frames=args.audio_max_frames,
        audio_vae_checkpoint=args.audio_vae_checkpoint,
        audio_vae_config=args.audio_vae_config,
        audio_latent_channels=args.audio_latent_channels,
        voice_vae_checkpoint=args.voice_vae_checkpoint,
        voice_vae_config=args.voice_vae_config,
        voice_latent_channels=args.voice_latent_channels,
        image_size=args.image_size,
        image_vae_checkpoint=args.image_vae_checkpoint,
        image_vae_config=args.image_vae_config,
        image_latent_channels=args.image_latent_channels,
        remove_mains_hum=args.remove_mains_hum,
    )

    # Stats
    stats = {
        "processed": 0,
        "saved": 0,
        "skipped_silent": 0,
        "skipped_no_text": 0,
        "skipped_error": 0,
    }

    # Shard accumulator
    shard_samples: List[WorldModelSample] = []
    shard_idx = 0

    def flush_shard():
        nonlocal shard_samples, shard_idx

        if not shard_samples:
            return

        shard_path = os.path.join(output_dir, f"shard_{shard_idx:06d}.pt")
        save_shard(shard_samples, shard_path)
        print(f"  Saved shard {shard_idx} ({len(shard_samples)} samples)")

        shard_samples = []
        shard_idx += 1

    # Batch accumulators
    batch_data = []  # List of (data, text) tuples

    def process_and_accumulate():
        nonlocal batch_data, shard_samples

        if not batch_data:
            return

        try:
            if args.modality == "text_only":
                texts = [t for _, t in batch_data]
                new_samples = processor.process_text_only_batch(texts)

            elif args.modality in ["text_audio", "text_voice"]:
                waveforms = [d for d, _ in batch_data]
                texts = [t for _, t in batch_data]
                modality_key = "audio" if args.modality == "text_audio" else "voice"
                new_samples = processor.process_audio_batch(waveforms, texts, modality_key)

            elif args.modality == "text_image":
                images = [d for d, _ in batch_data]
                texts = [t for _, t in batch_data]
                new_samples = processor.process_image_batch(images, texts)

            else:
                raise ValueError(f"Unknown modality: {args.modality}")

            shard_samples.extend(new_samples)
            stats["saved"] += len(new_samples)

            # Flush shard if full
            if len(shard_samples) >= args.shard_size:
                flush_shard()

        except Exception as e:
            print(f"Batch processing error: {e}")
            import traceback
            traceback.print_exc()
            stats["skipped_error"] += len(batch_data)

        batch_data = []

    # Main processing loop
    start_time = time.time()
    pbar = tqdm(total=samples_for_this_gpu, desc=f"GPU {args.gpu_id}")

    for idx in range(args.start_idx, len(dataset)):
        # Skip if not our sample
        if idx % args.total_gpus != args.gpu_id:
            continue

        # Check max samples limit
        if args.max_samples and stats["processed"] >= args.max_samples:
            break

        try:
            example = dataset[idx]

            # Get text
            text = example.get(args.text_column, None)
            if text is None or not str(text).strip():
                stats["skipped_no_text"] += 1
                pbar.update(1)
                continue
            text = str(text).strip()

            # Get data based on modality
            if args.modality == "text_only":
                batch_data.append((None, text))

            elif args.modality in ["text_audio", "text_voice"]:
                audio = example[args.audio_column]
                waveform = torch.tensor(audio["array"], dtype=torch.float32)

                # Skip silent audio
                if waveform.abs().max() < args.min_audio_energy or waveform.std() < args.min_audio_std:
                    stats["skipped_silent"] += 1
                    pbar.update(1)
                    continue

                # Remove mains hum
                if args.remove_mains_hum:
                    waveform = remove_mains_hum(waveform.unsqueeze(0), args.sample_rate).squeeze(0)

                # Truncate to max length
                max_samples = args.audio_max_frames * args.hop_length
                if len(waveform) > max_samples:
                    waveform = waveform[:max_samples]

                batch_data.append((waveform, text))

            elif args.modality == "text_image":
                image = example[args.image_column]
                # Assume image is already a tensor or PIL Image
                if not isinstance(image, torch.Tensor):
                    from torchvision import transforms
                    transform = transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.CenterCrop(args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ])
                    if hasattr(image, 'convert'):
                        image = image.convert('RGB')
                    image = transform(image)

                batch_data.append((image, text))

            # Process batch when full
            if len(batch_data) >= args.batch_size:
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
        "modality": args.modality,
        "tokenizer": args.tokenizer,
        "special_tokens": SPECIAL_TOKENS,
        "shard_size": args.shard_size,
        "stats": stats,
    }

    # Add modality-specific config
    if args.modality in ["text_audio", "text_voice"]:
        config["audio"] = {
            "sample_rate": args.sample_rate,
            "n_mels": args.n_mels,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "audio_max_frames": args.audio_max_frames,
            "vae_checkpoint": args.audio_vae_checkpoint or args.voice_vae_checkpoint,
            "vae_config": args.audio_vae_config or args.voice_vae_config,
            "latent_channels": args.audio_latent_channels or args.voice_latent_channels,
        }

    if args.modality == "text_image":
        config["image"] = {
            "image_size": args.image_size,
            "vae_checkpoint": args.image_vae_checkpoint,
            "vae_config": args.image_vae_config,
            "latent_channels": args.image_latent_channels,
        }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*50}")
    print(f"GPU {args.gpu_id} complete!")
    print(f"  Processed: {stats['processed']:,}")
    print(f"  Saved: {stats['saved']:,} (with duplication for paired modalities)")
    print(f"  Skipped (silent): {stats['skipped_silent']:,}")
    print(f"  Skipped (no text): {stats['skipped_no_text']:,}")
    print(f"  Skipped (error): {stats['skipped_error']:,}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Speed: {stats['saved']/elapsed:.1f} samples/sec")
    print(f"  Shards: {shard_idx}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()