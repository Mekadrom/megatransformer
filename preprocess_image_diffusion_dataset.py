import os
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
import json
from PIL import Image
from torchvision import transforms
from io import BytesIO
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple

from transformers import T5Tokenizer, T5EncoderModel


"""
Preprocesses image datasets for diffusion training.
Uses T5 to produce text embeddings for text-to-image generation.
Optionally encodes images using a VAE for latent diffusion.

Supports both embedded images and URL-based datasets (like LAION).
"""


def download_image(url: str, timeout: int = 10) -> Optional[Image.Image]:
    """
    Download image from URL.

    Args:
        url: Image URL
        timeout: Request timeout in seconds

    Returns:
        PIL Image or None if download failed
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=timeout, headers=headers, stream=True)
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            return None

        # Load image
        img = Image.open(BytesIO(response.content))

        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        return img
    except Exception:
        return None


def load_image_vae(checkpoint_path: str, vae_config: str, latent_channels: int, device: str = "cuda"):
    """
    Load an image VAE from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory (containing model.safetensors or pytorch_model.bin)
        vae_config: Config name from model_config_lookup (e.g., "mini", "tiny")
        latent_channels: Number of latent channels the VAE was trained with
        device: Device to load the model on

    Returns:
        VAE model in eval mode
    """
    from model.image.vae import model_config_lookup

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
        # strict=False ignores LPIPS weights if checkpoint was trained with perceptual loss
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            # Filter out expected missing keys (lpips)
            missing = [k for k in missing if "lpips" not in k.lower()]
            if missing:
                print(f"Warning: Missing keys: {missing}")
        print(f"Loaded VAE from {safetensors_path}")
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location=device)
        # strict=False ignores LPIPS weights if checkpoint was trained with perceptual loss
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            # Filter out expected missing keys (lpips)
            missing = [k for k in missing if "lpips" not in k.lower()]
            if missing:
                print(f"Warning: Missing keys: {missing}")
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
def encode_image_to_latent(vae, image: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """
    Encode an image to VAE latent space.

    Args:
        vae: VAE model
        image: Image tensor [C, H, W] or [B, C, H, W]
        device: Device to run on

    Returns:
        Latent mu tensor [latent_channels, H', W']
    """
    # Ensure correct shape: [B, C, H, W]
    if image.dim() == 3:
        image = image.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]

    image = image.to(device)

    # Get mu from encoder (don't sample, use deterministic mean)
    mu, _ = vae.encoder(image)

    # Remove batch dimension: [1, C, H', W'] -> [C, H', W']
    return mu.squeeze(0).cpu()


def get_image_transform(image_size: int, normalize: bool = True):
    """
    Create image transformation pipeline.

    Args:
        image_size: Target size (images will be resized and center-cropped)
        normalize: Whether to normalize to [-1, 1] range

    Returns:
        torchvision transforms
    """
    transform_list = [
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]

    if normalize:
        # Normalize to [-1, 1] range (common for diffusion models)
        transform_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    return transforms.Compose(transform_list)


def preprocess_and_cache_dataset(
    output_dir: str,
    dataset_name: str = "pixparse/cc3m-wds",
    dataset_config: str = None,
    split: str = "train",
    image_size: int = 256,
    max_samples: int = None,
    huggingface_text_model: str = "t5-small",
    max_text_length: int = 512,
    caption_column: str = "caption",
    image_column: str = "jpg",
    # VAE encoding options
    vae_checkpoint: str = None,
    vae_config: str = "mini",
    latent_channels: int = 4,
):
    """
    Preprocess image dataset and save as individual .pt files.

    If vae_checkpoint is provided, images are encoded to VAE latent space
    and saved as 'latent_mu' for latent diffusion training.

    Captions are encoded using T5 and saved as 'text_embeddings'.
    Samples without captions are skipped.
    """
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load VAE if checkpoint provided
    vae = None
    if vae_checkpoint is not None:
        print(f"Loading image VAE from {vae_checkpoint}...")
        vae = load_image_vae(vae_checkpoint, vae_config, latent_channels, device)
        print(f"  VAE config: {vae_config}, latent_channels: {latent_channels}")

    # Load dataset
    print(f"Loading dataset {dataset_name} split {split}...")
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True)
    else:
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Check if dataset has captions
    has_caption_column = caption_column in dataset.column_names
    if not has_caption_column:
        raise ValueError(f"Caption column '{caption_column}' not found in dataset. "
                         f"Available columns: {dataset.column_names}")

    # Load text encoder
    print(f"Loading text encoder {huggingface_text_model}...")
    text_model = T5EncoderModel.from_pretrained(huggingface_text_model)
    text_tokenizer = T5Tokenizer.from_pretrained(huggingface_text_model)
    text_model.eval()
    text_model.to(device)

    # Create image transform
    transform = get_image_transform(image_size, normalize=True)

    # Track statistics
    stats = {
        "total": len(dataset),
        "saved_samples": 0,
        "skipped_no_caption": 0,
        "skipped_invalid_image": 0,
        "skipped_error": 0,
    }
    if vae is not None:
        stats["latents_encoded"] = 0

    # Process each example
    print("Processing examples...")
    for idx in tqdm(range(len(dataset))):
        try:
            example = dataset[idx]

            # Check for caption first
            caption = example.get(caption_column, None)
            # Handle list of captions (e.g., flickr30k) - take the first one
            if isinstance(caption, list):
                caption = caption[0] if len(caption) > 0 else None
            if caption is None or (isinstance(caption, str) and caption.strip() == ""):
                stats["skipped_no_caption"] += 1
                continue

            # Get image
            image = example.get(image_column, None)
            if image is None:
                stats["skipped_invalid_image"] += 1
                continue

            # Convert to PIL if needed
            if not isinstance(image, Image.Image):
                if isinstance(image, dict) and "bytes" in image:
                    import io
                    image = Image.open(io.BytesIO(image["bytes"]))
                elif isinstance(image, dict) and "path" in image:
                    image = Image.open(image["path"])
                elif isinstance(image, bytes):
                    import io
                    image = Image.open(io.BytesIO(image))
                else:
                    stats["skipped_invalid_image"] += 1
                    continue

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Apply transforms
            image_tensor = transform(image)

            # Encode text
            text_inputs = text_tokenizer(
                caption,
                max_length=max_text_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

            with torch.no_grad():
                text_embeddings = text_model(**text_inputs).last_hidden_state.squeeze(0).cpu()
            text_attention_mask = text_inputs['attention_mask'].squeeze(0).cpu()

            # Encode to VAE latent if VAE is provided
            latent_mu = None
            if vae is not None:
                latent_mu = encode_image_to_latent(vae, image_tensor, device)
                stats["latents_encoded"] += 1

            # Save to file
            save_path = os.path.join(output_dir, f"{idx:08d}.pt")
            save_dict = {
                "image": image_tensor,
                "text_embeddings": text_embeddings,
                "text_attention_mask": text_attention_mask,
            }

            if latent_mu is not None:
                save_dict["latent_mu"] = latent_mu
                save_dict["latent_shape"] = list(latent_mu.shape)  # [C, H', W']

            # Add label if available (for class-conditional generation)
            if "label" in example:
                save_dict["label"] = example["label"]

            torch.save(save_dict, save_path)
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
        "image_size": image_size,
        "text_model": huggingface_text_model,
        "max_text_length": max_text_length,
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
    print(f"  Skipped (no caption): {stats['skipped_no_caption']}")
    print(f"  Skipped (invalid image): {stats['skipped_invalid_image']}")
    print(f"  Skipped (error): {stats['skipped_error']}")
    if vae is not None:
        print(f"  Latents encoded: {stats['latents_encoded']}")

    return stats


def preprocess_url_dataset(
    output_dir: str,
    dataset_name: str = "laion/relaion400m",
    dataset_config: Optional[str] = None,
    split: str = "train",
    image_size: int = 256,
    max_samples: Optional[int] = None,
    huggingface_text_model: str = "t5-small",
    max_text_length: int = 512,
    caption_column: str = "TEXT",
    url_column: str = "URL",
    min_image_size: int = 64,
    # VAE encoding options
    vae_checkpoint: Optional[str] = None,
    vae_config: str = "mini",
    latent_channels: int = 4,
    # Download options
    num_download_workers: int = 8,
    download_timeout: int = 10,
    # Resume options
    start_idx: int = 0,
    num_expected_examples: Optional[int] = None,
):
    """
    Preprocess URL-based image dataset (like LAION) and save as individual .pt files.
    Uses streaming mode for large datasets and parallel downloading.

    Args:
        output_dir: Directory to save preprocessed files
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration (if any)
        split: Dataset split
        image_size: Target image size (square)
        max_samples: Maximum number of samples to process (None = all)
        huggingface_text_model: Text encoder model name
        max_text_length: Maximum text token length
        caption_column: Column name containing captions/text
        url_column: Column name containing image URLs
        min_image_size: Minimum image dimension to accept
        vae_checkpoint: Path to VAE checkpoint for latent encoding
        vae_config: VAE config name
        latent_channels: Number of latent channels in VAE
        num_download_workers: Number of parallel download threads
        download_timeout: Timeout for image downloads
        start_idx: Index to start from (for resuming)
        num_expected_examples: Expected number of examples (for progress bar)
    """
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load VAE if checkpoint provided
    vae = None
    if vae_checkpoint is not None:
        print(f"Loading image VAE from {vae_checkpoint}...")
        vae = load_image_vae(vae_checkpoint, vae_config, latent_channels, device)
        print(f"  VAE config: {vae_config}, latent_channels: {latent_channels}")

    # Load dataset in streaming mode
    print(f"Loading dataset {dataset_name}" + (f"/{dataset_config}" if dataset_config else "") + f" split {split} (streaming)...")
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True, trust_remote_code=True)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)

    # Load text encoder
    print(f"Loading text encoder {huggingface_text_model}...")
    text_model = T5EncoderModel.from_pretrained(huggingface_text_model)
    text_tokenizer = T5Tokenizer.from_pretrained(huggingface_text_model)
    text_model.eval()
    text_model.to(device)

    # Create image transform
    transform = get_image_transform(image_size, normalize=True)

    # Track statistics
    stats = {
        "total_processed": 0,
        "saved_samples": 0,
        "skipped_no_caption": 0,
        "skipped_download_failed": 0,
        "skipped_too_small": 0,
        "skipped_error": 0,
    }
    if vae is not None:
        stats["latents_encoded"] = 0

    print(f"Processing images...")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Min source size: {min_image_size}")
    print(f"  Download workers: {num_download_workers}")
    print(f"  URL column: {url_column}")
    print(f"  Caption column: {caption_column}")

    def process_example(idx: int, example: dict) -> Optional[Tuple[int, dict]]:
        """Download image and return processed data, or None if failed."""
        # Check caption
        caption = example.get(caption_column, None)
        if isinstance(caption, list):
            caption = caption[0] if len(caption) > 0 else None
        if caption is None or (isinstance(caption, str) and caption.strip() == ""):
            return ("no_caption", None)

        # Download image
        url = example.get(url_column, None)
        if url is None:
            return ("no_url", None)

        img = download_image(url, timeout=download_timeout)
        if img is None:
            return ("download_failed", None)

        # Check minimum size
        if img.width < min_image_size or img.height < min_image_size:
            return ("too_small", None)

        return ("success", (idx, img, caption))

    # Batch processing with thread pool
    batch_size = num_download_workers * 4
    batch = []
    idx = 0
    saved_idx = 0  # Separate counter for saved files (no gaps)

    pbar = tqdm(desc="Processing images", total=num_expected_examples)

    for example in dataset:
        # Skip to start_idx
        if idx < start_idx:
            idx += 1
            continue

        if max_samples is not None and stats["saved_samples"] >= max_samples:
            break

        batch.append((idx, example))

        if len(batch) >= batch_size:
            # Process batch - download images in parallel
            with ThreadPoolExecutor(max_workers=num_download_workers) as executor:
                futures = {
                    executor.submit(process_example, b_idx, b_example): b_idx
                    for b_idx, b_example in batch
                }

                results = []
                for future in as_completed(futures):
                    status, data = future.result()
                    stats["total_processed"] += 1
                    pbar.update(1)

                    if status == "no_caption":
                        stats["skipped_no_caption"] += 1
                    elif status == "download_failed" or status == "no_url":
                        stats["skipped_download_failed"] += 1
                    elif status == "too_small":
                        stats["skipped_too_small"] += 1
                    elif status == "success":
                        results.append(data)

            # Process successful downloads (encode text and save)
            # This part is sequential to use GPU efficiently
            for orig_idx, img, caption in results:
                try:
                    # Apply image transform
                    image_tensor = transform(img)

                    # Encode text
                    text_inputs = text_tokenizer(
                        caption,
                        max_length=max_text_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

                    with torch.no_grad():
                        text_embeddings = text_model(**text_inputs).last_hidden_state.squeeze(0).cpu()
                    text_attention_mask = text_inputs['attention_mask'].squeeze(0).cpu()

                    # Encode to VAE latent if VAE is provided
                    latent_mu = None
                    if vae is not None:
                        latent_mu = encode_image_to_latent(vae, image_tensor, device)
                        stats["latents_encoded"] = stats.get("latents_encoded", 0) + 1

                    # Save to file (use saved_idx for contiguous numbering)
                    save_path = os.path.join(output_dir, f"{saved_idx:08d}.pt")
                    save_dict = {
                        "image": image_tensor,
                        "text_embeddings": text_embeddings,
                        "text_attention_mask": text_attention_mask,
                    }

                    if latent_mu is not None:
                        save_dict["latent_mu"] = latent_mu
                        save_dict["latent_shape"] = list(latent_mu.shape)

                    torch.save(save_dict, save_path)
                    stats["saved_samples"] += 1
                    saved_idx += 1

                except Exception as e:
                    stats["skipped_error"] += 1

            batch = []

        idx += 1

    # Process remaining batch
    if batch:
        with ThreadPoolExecutor(max_workers=num_download_workers) as executor:
            futures = {
                executor.submit(process_example, b_idx, b_example): b_idx
                for b_idx, b_example in batch
            }

            results = []
            for future in as_completed(futures):
                status, data = future.result()
                stats["total_processed"] += 1
                pbar.update(1)

                if status == "no_caption":
                    stats["skipped_no_caption"] += 1
                elif status == "download_failed" or status == "no_url":
                    stats["skipped_download_failed"] += 1
                elif status == "too_small":
                    stats["skipped_too_small"] += 1
                elif status == "success":
                    results.append(data)

        for orig_idx, img, caption in results:
            try:
                image_tensor = transform(img)

                text_inputs = text_tokenizer(
                    caption,
                    max_length=max_text_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

                with torch.no_grad():
                    text_embeddings = text_model(**text_inputs).last_hidden_state.squeeze(0).cpu()
                text_attention_mask = text_inputs['attention_mask'].squeeze(0).cpu()

                latent_mu = None
                if vae is not None:
                    latent_mu = encode_image_to_latent(vae, image_tensor, device)
                    stats["latents_encoded"] = stats.get("latents_encoded", 0) + 1

                save_path = os.path.join(output_dir, f"{saved_idx:08d}.pt")
                save_dict = {
                    "image": image_tensor,
                    "text_embeddings": text_embeddings,
                    "text_attention_mask": text_attention_mask,
                }

                if latent_mu is not None:
                    save_dict["latent_mu"] = latent_mu
                    save_dict["latent_shape"] = list(latent_mu.shape)

                torch.save(save_dict, save_path)
                stats["saved_samples"] += 1
                saved_idx += 1

            except Exception as e:
                stats["skipped_error"] += 1

    pbar.close()

    # Save config and stats
    config = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "image_size": image_size,
        "text_model": huggingface_text_model,
        "max_text_length": max_text_length,
        "url_column": url_column,
        "caption_column": caption_column,
        "min_image_size": min_image_size,
        "stats": stats,
    }
    if vae is not None:
        config["vae_config"] = vae_config
        config["vae_checkpoint"] = vae_checkpoint
        config["latent_channels"] = latent_channels

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nPreprocessing complete!")
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  Saved samples: {stats['saved_samples']}")
    print(f"  Skipped (no caption): {stats['skipped_no_caption']}")
    print(f"  Skipped (download failed): {stats['skipped_download_failed']}")
    print(f"  Skipped (too small): {stats['skipped_too_small']}")
    print(f"  Skipped (error): {stats['skipped_error']}")
    if vae is not None:
        print(f"  Latents encoded: {stats.get('latents_encoded', 0)}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess image dataset for diffusion training")

    # Dataset options
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save cached .pt files")
    parser.add_argument("--dataset_name", type=str, default="pixparse/cc3m-wds",
                        help="HuggingFace dataset name (default: pixparse/cc3m-wds)")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset config name (if applicable)")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to process")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Target image size (images will be resized and center-cropped)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (for testing)")

    # Column names (dataset-specific)
    parser.add_argument("--image_column", type=str, default="jpg",
                        help="Column name containing embedded images (for non-URL datasets)")
    parser.add_argument("--url_column", type=str, default=None,
                        help="Column name containing image URLs (for URL-based datasets like LAION)")
    parser.add_argument("--caption_column", type=str, default="caption",
                        help="Column name containing captions")

    # Text encoder options
    parser.add_argument("--text_model", type=str, default="t5-small",
                        help="HuggingFace text encoder model for captions")
    parser.add_argument("--max_text_length", type=int, default=512,
                        help="Maximum text token length")

    # VAE encoding options for latent diffusion
    parser.add_argument("--vae_checkpoint", type=str, default=None,
                        help="Path to image VAE checkpoint directory for latent encoding")
    parser.add_argument("--vae_config", type=str, default="mini",
                        help="VAE config name (tiny, mini)")
    parser.add_argument("--latent_channels", type=int, default=4,
                        help="Number of latent channels in the VAE")

    # URL-based dataset options
    parser.add_argument("--min_image_size", type=int, default=64,
                        help="Minimum source image dimension (for URL datasets)")
    parser.add_argument("--num_download_workers", type=int, default=8,
                        help="Number of parallel download threads (for URL datasets)")
    parser.add_argument("--download_timeout", type=int, default=10,
                        help="Timeout for image downloads in seconds (for URL datasets)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Index to start from, for resuming (for URL datasets)")
    parser.add_argument("--num_expected_examples", type=int, default=None,
                        help="Expected number of examples for progress bar (for URL datasets)")

    args = parser.parse_args()

    if args.url_column is not None:
        # URL-based dataset (like LAION)
        preprocess_url_dataset(
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.split,
            image_size=args.image_size,
            max_samples=args.max_samples,
            huggingface_text_model=args.text_model,
            max_text_length=args.max_text_length,
            caption_column=args.caption_column,
            url_column=args.url_column,
            min_image_size=args.min_image_size,
            vae_checkpoint=args.vae_checkpoint,
            vae_config=args.vae_config,
            latent_channels=args.latent_channels,
            num_download_workers=args.num_download_workers,
            download_timeout=args.download_timeout,
            start_idx=args.start_idx,
            num_expected_examples=args.num_expected_examples,
        )
    else:
        # Embedded image dataset
        preprocess_and_cache_dataset(
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.split,
            image_size=args.image_size,
            max_samples=args.max_samples,
            huggingface_text_model=args.text_model,
            max_text_length=args.max_text_length,
            caption_column=args.caption_column,
            image_column=args.image_column,
            vae_checkpoint=args.vae_checkpoint,
            vae_config=args.vae_config,
            latent_channels=args.latent_channels,
        )
