import os
import random
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
import json
from PIL import Image
from io import BytesIO
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchvision import transforms
from typing import Optional
import time


"""
Preprocesses image datasets for VAE training.
Downloads images from URLs (for LAION-style datasets) or uses local images,
applies optional augmentations, and saves as tensors.

For VAE training, we only need images - no captions/text required.
"""


def create_image_transforms(
    image_size: int = 256,
    augment: bool = False,
    center_crop: bool = True,
):
    """
    Create image transformation pipeline.

    Args:
        image_size: Target size for images (will be square)
        augment: Whether to apply data augmentation
        center_crop: If True, center crop; if False with augment, random crop

    Returns:
        torchvision.transforms.Compose pipeline
    """
    transform_list = []

    if augment:
        # Augmented pipeline (no color jitter - hurts VAE reconstruction fidelity)
        transform_list.extend([
            transforms.Resize(int(image_size * 1.1), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
    else:
        # Non-augmented pipeline
        transform_list.extend([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size) if center_crop else transforms.RandomCrop(image_size),
        ])

    # Common final transforms
    transform_list.extend([
        transforms.ToTensor(),  # Converts to [0, 1] range
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
    ])

    return transforms.Compose(transform_list)


def download_image(url: str, timeout: int = 1) -> Optional[Image.Image]:
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


def process_single_image(
    idx: int,
    image_source,  # URL string or PIL Image or path
    output_dir: str,
    transform: transforms.Compose,
    augment_transform: Optional[transforms.Compose] = None,
    num_augmented_copies: int = 0,
    min_size: int = 64,
    is_url: bool = True,
) -> dict:
    """
    Process a single image: download (if URL), transform, and save.

    Returns:
        Dict with processing stats
    """
    stats = {
        "saved": 0,
        "saved_augmented": 0,
        "skipped_download_failed": 0,
        "skipped_too_small": 0,
        "skipped_error": 0,
    }

    try:
        # Get image
        if is_url:
            img = download_image(image_source)
            if img is None:
                stats["skipped_download_failed"] = 1
                return stats
        elif isinstance(image_source, str):
            # File path
            img = Image.open(image_source).convert('RGB')
        else:
            # Already a PIL Image
            img = image_source
            if img.mode != 'RGB':
                img = img.convert('RGB')

        # Check minimum size
        if img.width < min_size or img.height < min_size:
            stats["skipped_too_small"] = 1
            return stats

        # Save original (non-augmented) version
        img_tensor = transform(img)
        save_path = os.path.join(output_dir, f"{idx:08d}.pt")
        torch.save({"image": img_tensor}, save_path)
        stats["saved"] = 1

        # Save augmented copies
        if augment_transform is not None and num_augmented_copies > 0:
            for aug_idx in range(num_augmented_copies):
                aug_tensor = augment_transform(img)
                aug_save_path = os.path.join(output_dir, f"{idx:08d}_aug{aug_idx + 1}.pt")
                torch.save({"image": aug_tensor}, aug_save_path)
                stats["saved_augmented"] += 1

    except Exception as e:
        stats["skipped_error"] = 1

    return stats


def preprocess_and_cache_dataset(
    output_dir: str,
    dataset_name: str = "laion/relaion400m",
    dataset_config: Optional[str] = None,
    split: str = "train",
    image_size: int = 256,
    url_column: str = "URL",
    image_column: Optional[str] = None,  # For datasets with embedded images
    max_samples: Optional[int] = None,
    min_image_size: int = 64,
    # Augmentation options
    augment: bool = False,
    num_augmented_copies: int = 0,
    augmentation_seed: int = 42,
    # Download options
    num_download_workers: int = 8,
    download_timeout: int = 10,
    # Resume options
    start_idx: int = 0,
    num_expected_examples: Optional[int] = None,
):
    """
    Preprocess image dataset and save as individual .pt files.

    For LAION-style datasets, images are downloaded from URLs.
    For datasets with embedded images, they are extracted directly.

    Args:
        output_dir: Directory to save preprocessed files
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration (if any)
        split: Dataset split
        image_size: Target image size (square)
        url_column: Column name containing image URLs
        image_column: Column name containing embedded images (if not URL-based)
        max_samples: Maximum number of samples to process (None = all)
        min_image_size: Minimum image dimension to accept
        augment: Whether to apply augmentation to original saves
        num_augmented_copies: Number of additional augmented copies per image
        augmentation_seed: Random seed for reproducibility
        num_download_workers: Number of parallel download threads
        download_timeout: Timeout for image downloads
        start_idx: Index to start from (for resuming)
    """
    random.seed(augmentation_seed)
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset {dataset_name}" + (f"/{dataset_config}" if dataset_config else "") + f" split {split}...")

    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=True)

    # Create transforms
    base_transform = create_image_transforms(image_size=image_size, augment=False)
    augment_transform = create_image_transforms(image_size=image_size, augment=True) if num_augmented_copies > 0 else None

    # Determine if URL-based or embedded images
    is_url_based = image_column is None

    # Track statistics
    stats = {
        "total_processed": 0,
        "saved": 0,
        "saved_augmented": 0,
        "skipped_download_failed": 0,
        "skipped_too_small": 0,
        "skipped_error": 0,
    }

    print(f"Processing images...")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Min source size: {min_image_size}")
    print(f"  Augmented copies per image: {num_augmented_copies}")
    print(f"  URL-based: {is_url_based}")
    if is_url_based:
        print(f"  Download workers: {num_download_workers}")

    # Process with threading for URL downloads
    if is_url_based and num_download_workers > 1:
        # Batch processing with thread pool
        batch_size = num_download_workers * 4
        batch = []
        idx = start_idx

        pbar = tqdm(desc="Processing images", total=num_expected_examples)

        for example in dataset:
            if start_idx > 0 and idx < start_idx:
                idx += 1
                continue

            if max_samples is not None and stats["total_processed"] >= max_samples:
                break

            url = example.get(url_column)
            if url:
                batch.append((idx, url))

            if len(batch) >= batch_size:
                # Process batch
                with ThreadPoolExecutor(max_workers=num_download_workers) as executor:
                    futures = {
                        executor.submit(
                            process_single_image,
                            b_idx, b_url, output_dir, base_transform,
                            augment_transform, num_augmented_copies,
                            min_image_size, True
                        ): b_idx for b_idx, b_url in batch
                    }

                    for future in as_completed(futures):
                        result = future.result()
                        stats["saved"] += result["saved"]
                        stats["saved_augmented"] += result["saved_augmented"]
                        stats["skipped_download_failed"] += result["skipped_download_failed"]
                        stats["skipped_too_small"] += result["skipped_too_small"]
                        stats["skipped_error"] += result["skipped_error"]
                        stats["total_processed"] += 1
                        pbar.update(1)

                batch = []

            idx += 1

        # Process remaining batch
        if batch:
            with ThreadPoolExecutor(max_workers=num_download_workers) as executor:
                futures = {
                    executor.submit(
                        process_single_image,
                        b_idx, b_url, output_dir, base_transform,
                        augment_transform, num_augmented_copies,
                        min_image_size, True
                    ): b_idx for b_idx, b_url in batch
                }

                for future in as_completed(futures):
                    result = future.result()
                    stats["saved"] += result["saved"]
                    stats["saved_augmented"] += result["saved_augmented"]
                    stats["skipped_download_failed"] += result["skipped_download_failed"]
                    stats["skipped_too_small"] += result["skipped_too_small"]
                    stats["skipped_error"] += result["skipped_error"]
                    stats["total_processed"] += 1
                    pbar.update(1)

        pbar.close()

    else:
        # Sequential processing (for embedded images or single-threaded)
        idx = start_idx

        for example in tqdm(dataset, desc="Processing images"):
            if start_idx > 0 and idx < start_idx:
                idx += 1
                continue

            if max_samples is not None and stats["total_processed"] >= max_samples:
                break

            if is_url_based:
                image_source = example.get(url_column)
            else:
                image_source = example.get(image_column)

            if image_source is None:
                idx += 1
                continue

            result = process_single_image(
                idx, image_source, output_dir, base_transform,
                augment_transform, num_augmented_copies,
                min_image_size, is_url_based
            )

            stats["saved"] += result["saved"]
            stats["saved_augmented"] += result["saved_augmented"]
            stats["skipped_download_failed"] += result["skipped_download_failed"]
            stats["skipped_too_small"] += result["skipped_too_small"]
            stats["skipped_error"] += result["skipped_error"]
            stats["total_processed"] += 1

            idx += 1

    # Save config and stats
    config = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "image_size": image_size,
        "url_column": url_column if is_url_based else None,
        "image_column": image_column,
        "min_image_size": min_image_size,
        "augmentation": {
            "enabled": num_augmented_copies > 0,
            "num_copies": num_augmented_copies,
            "seed": augmentation_seed,
        },
        "stats": stats,
        "normalization": {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "range": "[-1, 1]",
        },
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nPreprocessing complete!")
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  Saved (original): {stats['saved']}")
    print(f"  Saved (augmented): {stats['saved_augmented']}")
    print(f"  Total files: {stats['saved'] + stats['saved_augmented']}")
    print(f"  Skipped (download failed): {stats['skipped_download_failed']}")
    print(f"  Skipped (too small): {stats['skipped_too_small']}")
    print(f"  Skipped (error): {stats['skipped_error']}")

    return stats


def preprocess_local_images(
    output_dir: str,
    input_dir: str,
    image_size: int = 256,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".webp"),
    min_image_size: int = 64,
    augment: bool = False,
    num_augmented_copies: int = 0,
    augmentation_seed: int = 42,
):
    """
    Preprocess local image directory.

    Args:
        output_dir: Directory to save preprocessed files
        input_dir: Directory containing source images
        image_size: Target image size
        extensions: Tuple of valid image extensions
        min_image_size: Minimum image dimension
        augment: Apply augmentation to original saves
        num_augmented_copies: Number of augmented copies
        augmentation_seed: Random seed
    """
    random.seed(augmentation_seed)
    os.makedirs(output_dir, exist_ok=True)

    # Find all images
    image_paths = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(extensions):
                image_paths.append(os.path.join(root, f))

    print(f"Found {len(image_paths)} images in {input_dir}")

    # Create transforms
    base_transform = create_image_transforms(image_size=image_size, augment=augment)
    augment_transform = create_image_transforms(image_size=image_size, augment=True) if num_augmented_copies > 0 else None

    stats = {
        "total": len(image_paths),
        "saved": 0,
        "saved_augmented": 0,
        "skipped_too_small": 0,
        "skipped_error": 0,
    }

    for idx, img_path in enumerate(tqdm(image_paths, desc="Processing images")):
        result = process_single_image(
            idx, img_path, output_dir, base_transform,
            augment_transform, num_augmented_copies,
            min_image_size, is_url=False
        )

        stats["saved"] += result["saved"]
        stats["saved_augmented"] += result["saved_augmented"]
        stats["skipped_too_small"] += result["skipped_too_small"]
        stats["skipped_error"] += result["skipped_error"]

    # Save config
    config = {
        "source": "local",
        "input_dir": input_dir,
        "image_size": image_size,
        "augmentation": {
            "enabled": num_augmented_copies > 0,
            "num_copies": num_augmented_copies,
        },
        "stats": stats,
        "normalization": {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "range": "[-1, 1]",
        },
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nPreprocessing complete!")
    print(f"  Total images: {stats['total']}")
    print(f"  Saved: {stats['saved']}")
    print(f"  Augmented: {stats['saved_augmented']}")
    print(f"  Skipped (too small): {stats['skipped_too_small']}")
    print(f"  Skipped (error): {stats['skipped_error']}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for VAE training")

    # Required
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save preprocessed .pt files")

    # Dataset source (choose one)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--dataset_name", type=str,
                              help="HuggingFace dataset name (e.g., laion/relaion400m)")
    source_group.add_argument("--input_dir", type=str,
                              help="Local directory containing images")

    # Dataset options
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset configuration name")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split")
    parser.add_argument("--url_column", type=str, default="URL",
                        help="Column name containing image URLs")
    parser.add_argument("--image_column", type=str, default=None,
                        help="Column name for embedded images (if not URL-based)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Index to start from (for resuming)")

    # Image options
    parser.add_argument("--image_size", type=int, default=256,
                        help="Target image size (square)")
    parser.add_argument("--min_image_size", type=int, default=64,
                        help="Minimum source image dimension")

    # Augmentation options
    parser.add_argument("--augment", action="store_true",
                        help="Apply augmentation to original saves")
    parser.add_argument("--num_augmented_copies", type=int, default=0,
                        help="Number of augmented copies per image (0 = none)")
    parser.add_argument("--augmentation_seed", type=int, default=42,
                        help="Random seed for augmentation")

    # Download options
    parser.add_argument("--num_download_workers", type=int, default=8,
                        help="Number of parallel download threads")
    parser.add_argument("--download_timeout", type=int, default=10,
                        help="Timeout for image downloads (seconds)")
    
    parser.add_argument("--num_expected_examples", type=int, default=None,
                        help="Expected number of examples (for tqdm)")

    args = parser.parse_args()

    if args.input_dir:
        # Process local images
        preprocess_local_images(
            output_dir=args.output_dir,
            input_dir=args.input_dir,
            image_size=args.image_size,
            min_image_size=args.min_image_size,
            augment=args.augment,
            num_augmented_copies=args.num_augmented_copies,
            augmentation_seed=args.augmentation_seed,
        )
    else:
        # Process HuggingFace dataset
        preprocess_and_cache_dataset(
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.split,
            image_size=args.image_size,
            url_column=args.url_column,
            image_column=args.image_column,
            max_samples=args.max_samples,
            min_image_size=args.min_image_size,
            augment=args.augment,
            num_augmented_copies=args.num_augmented_copies,
            augmentation_seed=args.augmentation_seed,
            num_download_workers=args.num_download_workers,
            download_timeout=args.download_timeout,
            start_idx=args.start_idx,
            num_expected_examples=args.num_expected_examples,
        )