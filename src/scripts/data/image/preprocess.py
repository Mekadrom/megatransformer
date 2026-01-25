
import argparse
from io import BytesIO
import os
import requests
import torch
import torch.nn.functional as F
import traceback


from platform import processor
from typing import Optional

from datasets import load_dataset
from PIL.Image import Image
from torchvision import transforms

from scripts.data.audio.vae.preprocess import SIVEFeatureBatchProcessor
from scripts.data.preprocessor import BatchProcessor
from utils import audio_utils
from utils.model_loading_utils import load_model
from utils.text_encoder import extract_text_embedding, get_text_encoder


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


def download_image(url: str, timeout: int = 1) -> Optional[Image]:
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


class ConditionsBatchProcessor(BatchProcessor):
    """Batched GPU processing for preprocessing conditions for conditional training."""

    def __init__(
        self,
        text_embedding_model: str = "google/t5-v1_1-small",
        max_text_length: int = 1536,
        device: str = "cuda",
    ):
        self.text_embedding_model = text_embedding_model
        self.max_text_length = max_text_length
        self.device = device

        self.encoder = get_text_encoder(encoder_type="t5-small", device=device)


    @torch.no_grad()
    def process_batch(
        self,
        data: list[str],
    ) -> dict[str, torch.Tensor]:
        """
        Process batch of images to normalize and/or resize.
        """
        # alias for clarity
        conditions = data  # List of [T] tensors

        processed_conditions: list[torch.Tensor] = []
        for condition in conditions:
            # extract text embeddings
            processed_conditions.append(extract_text_embedding(text=condition, encoder=self.encoder))

        return {
            "conditions": torch.stack(processed_conditions),  # [B, 3, H, W]
        }


class URLImageBatchProcessor(BatchProcessor):
    """Batched GPU processing for preprocessing images for image training."""

    def __init__(
        self,
        image_size=256,
        num_augmented_copies: int = 0,
        device: str = "cuda",
    ):
        self.image_size = image_size
        self.num_augmented_copies = num_augmented_copies
        self.device = device

        self.base_transform = create_image_transforms(image_size=image_size, augment=False)
        self.augment_transform = create_image_transforms(image_size=image_size, augment=True) if num_augmented_copies > 0 else None

    @torch.no_grad()
    def process_batch(
        self,
        data: list[str],
    ) -> dict[str, torch.Tensor]:
        """
        Process batch of images to normalize and/or resize.
        """
        # alias for clarity
        urls = data  # List of [T] tensors

        processed_images: list[torch.Tensor] = []
        for url in urls:
            image = download_image(url)
            if image is None:
                continue

            # Extract mel spectrogram
            transformed = self.base_transform(image).to(self.device)  # [3, H, W]
            processed_images.append(transformed)
            if self.augment_transform is not None:
                for _ in range(self.num_augmented_copies):
                    augmented = self.augment_transform(image).to(self.device)
                    processed_images.append(augmented)

        return {
            "images": torch.stack(processed_images),  # [B, 3, H, W]
        }


class ImageBatchProcessor(BatchProcessor):
    """Batched GPU processing for preprocessing images for image VAE training."""

    def __init__(
        self,
        image_size=256,
        num_augmented_copies: int = 0,
        device: str = "cuda",
    ):
        self.image_size = image_size
        self.num_augmented_copies = num_augmented_copies
        self.device = device

        self.base_transform = create_image_transforms(image_size=image_size, augment=False)
        self.augment_transform = create_image_transforms(image_size=image_size, augment=True) if num_augmented_copies > 0 else None

    @torch.no_grad()
    def process_batch(
        self,
        data: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Process batch of images to normalize and/or resize.
        """
        # alias for clarity
        images = data  # List of [T] tensors

        processed_images: list[torch.Tensor] = []
        for image in images:
            # Extract mel spectrogram
            transformed = self.base_transform(image).to(self.device)  # [3, H, W]
            processed_images.append(transformed)
            if self.augment_transform is not None:
                for _ in range(self.num_augmented_copies):
                    augmented = self.augment_transform(image).to(self.device)
                    processed_images.append(augmented)

        return {
            "images": torch.stack(processed_images),  # [B, 3, H, W]
        }


class ImageDatasetPreprocessor:
    """Preprocess dataset to save images for training."""

    def __init__(
        self,
        args,
        dataset,
        output_dir,
        shard_fields,
        batch_accumulators,
        stats_accumulator,
        device,
        url_column=None,
        image_column=None,
        include_conditions=True,
        text_conditions_model="google/t5-v1_1-small",
        max_text_length=1536,
    ):
        self.args = args
        self.dataset = dataset
        self.output_dir = output_dir
        self.shard_fields = shard_fields
        self.batch_accumulators = batch_accumulators
        self.stats_accumulator = stats_accumulator
        self.include_conditions = include_conditions
        self.device = device

        if image_column is not None:
            print(f"  Total samples in dataset: {len(self.dataset):,}")

            self.batch_processor = ImageBatchProcessor(
                image_size=self.args.image_size,
                num_augmented_copies=self.args.num_augmented_copies,
                device=self.device,
            )
        elif url_column is not None:
            print(f"  Total samples in dataset: {len(self.dataset):,}")

            self.batch_processor = URLImageBatchProcessor(
                image_size=self.args.image_size,
                num_augmented_copies=self.args.num_augmented_copies,
                device=self.device,
            )
        else:
            raise ValueError("Either image_column or url_column must be specified for Image preprocessing.")
        
        shard_fields.update({
            'shard_images': [],
        })

        if include_conditions:
            self.conditions_processor = ConditionsBatchProcessor(
                text_embedding_model=text_conditions_model,
                max_text_length=max_text_length,
                device=self.device,
            )
            shard_fields.update({
                'shard_conditions': [],
            })
        else:
            self.conditions_processor = None

    @classmethod
    def add_cli_args(cls, subparsers):
        sub_parser = subparsers.add_parser("image-vae", help="Preprocess image dataset for unconditional or conditional image training.")
    
        # Image settings
        sub_parser.add_argument("--image_size", type=int, default=256,
                            help="Target size for images (will be square)")
        sub_parser.add_argument("--num_augmented_copies", type=int, default=0,
                            help="Number of augmented copies to create per image (0 = no augmentation)")
        sub_parser.add_argument("--url_column", type=str, default=None,
                            help="Dataset column containing image URLs (if using URLs)")
        sub_parser.add_argument("--image_column", type=str, default=None,
                            help="Dataset column containing images (if using images directly)")
        sub_parser.add_argument("--include_conditions", action="store_true",
                            help="Whether to include text conditions for conditional training")
        sub_parser.add_argument("--text_conditions_model", type=str, default="google/t5-v1_1-small",
                            help="Text embedding model for conditions (e.g., 'google/t5-v1_1-small')")
        sub_parser.add_argument("--max_text_length", type=int, default=1536,
                            help="Maximum text length for conditions")
        sub_parser.add_argument("--condition_column", type=str, default="text",
                            help="Dataset column containing text conditions")

        return sub_parser

    def flush_shard(self):
        if not self.shard_fields['shard_images']:
            return
        
        shard_data = {
            "images": torch.cat(self.shard_fields['shard_images'], dim=0),
            "num_samples": sum(f.shape[0] for f in self.shard_fields['shard_images']),
        }

        # Find max conditions length in this shard for padding
        if self.include_conditions:
            max_feature_len = max(f.shape[-1] for f in self.shard_fields['shard_conditions'])

            # Pad features to same length
            padded_features = []
            for feat in self.shard_fields['shard_conditions']:
                if feat.shape[-1] < max_feature_len:
                    feat = F.pad(feat, (0, max_feature_len - feat.shape[-1]), value=0)
                padded_features.append(feat)

            shard_data.update({"conditions": torch.cat(padded_features, dim=0)})


        shard_path = os.path.join(self.output_dir, f"shard_{self.shard_fields['shard_idx']:06d}.pt")
        torch.save(shard_data, shard_path)

        print(f"  Saved shard {self.shard_fields['shard_idx']} ({shard_data['num_samples']} samples)")

        self.shard_fields['shard_images'] = []
        if self.include_conditions:
            self.shard_fields['shard_conditions'] = []
        self.shard_fields['shard_idx'] += 1
    
    def process_and_accumulate(self):
        if not self.batch_accumulators['batch_images']:
            return

        try:
            image_result = self.batch_processor.process_batch(self.batch_accumulators['batch_images'])
            self.shard_fields['shard_images'].append(image_result["images"])

            if self.include_conditions:
                conditions_result = self.conditions_processor.process_batch(self.batch_accumulators['batch_conditions'])
                self.shard_fields['shard_conditions'].append(conditions_result["conditions"])

            self.stats_accumulator["saved"] += len(self.batch_accumulators['batch_images'])

            # Flush shard if full
            current_size = sum(f.shape[0] for f in self.shard_fields['shard_images'])
            if current_size >= self.args.shard_size:
                self.flush_shard()
        except Exception as e:
            print(f"Batch processing error: {e}")
            traceback.print_exc()
            self.stats_accumulator["skipped"]["error"] += len(self.batch_accumulators['batch_images'])

        self.batch_accumulators['batch_images'] = []
        if self.include_conditions:
            self.batch_accumulators['batch_conditions'] = []
    
    def preprocess_example(self, example):
        # Extract image or URL
        image_or_url = example[self.args.image_column] if self.args.image_column is not None else example[self.args.url_column]

        # Add to batch
        self.batch_accumulators['batch_images'].append(image_or_url)
        if self.include_conditions:
            text_condition = example.get(self.args.condition_column, "")  # Default to empty string if not present
            self.batch_accumulators['batch_conditions'].append(text_condition)

        # Process batch when full
        if len(self.batch_accumulators['batch_images']) >= self.args.gpu_batch_size:
            self.process_and_accumulate()

    def parse_config(self) -> dict:
        return {
            "dataset_name": self.args.dataset_name,
            "dataset_config": self.args.dataset_config,
            "split": self.args.split,
            "shard_size": self.args.shard_size,
            "stats": self.stats_accumulator,
            "include_conditions": self.include_conditions,
        }
