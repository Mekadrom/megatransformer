import argparse
import io
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Union

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image

from scripts.data.preprocessor import BatchProcessor, Preprocessor


def download_image(
    url: str,
    timeout: float = 10.0,
    max_retries: int = 2,
) -> Optional[Image.Image]:
    """
    Download an image from a URL with retry logic.

    Args:
        url: URL to download from
        timeout: Request timeout in seconds
        max_retries: Number of retry attempts

    Returns:
        PIL Image or None if download failed
    """
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ImagePreprocessor/1.0)"},
            )
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception:
            if attempt == max_retries:
                return None
    return None


def download_images_parallel(
    urls: list[str],
    num_workers: int = 8,
    timeout: float = 10.0,
) -> list[Optional[Image.Image]]:
    """
    Download multiple images in parallel.

    Args:
        urls: List of URLs to download
        num_workers: Number of parallel download workers
        timeout: Request timeout per image

    Returns:
        List of PIL Images (None for failed downloads)
    """
    results: list[Optional[Image.Image]] = [None] * len(urls)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(download_image, url, timeout): idx
            for idx, url in enumerate(urls)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = None

    return results


class T5TextEncoder:
    """Encode text using T5 encoder for conditioning."""

    def __init__(
        self,
        model_name: str = "google/t5-v1_1-small",
        max_length: int = 128,
        device: str = "cuda",
    ):
        from transformers import T5EncoderModel, T5Tokenizer

        self.device = device
        self.max_length = max_length

        print(f"Loading T5 tokenizer and encoder: {model_name}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name).to(device).eval()

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        print(f"  T5 encoder loaded (hidden_size={self.encoder.config.d_model})")

    @torch.no_grad()
    def encode_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Encode batch of texts through T5 encoder.

        Args:
            texts: List of text strings

        Returns:
            Dict with:
                - text_embeddings: [B, seq_len, hidden_size] T5 encoder outputs
                - attention_mask: [B, seq_len] attention mask (1 = valid, 0 = padding)
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Encode through T5
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = outputs.last_hidden_state

        return {
            "text_embeddings": text_embeddings.cpu(),
            "text_attention_mask": attention_mask.cpu(),
        }


LITEVAE_MODELS = {
    "litevae": {
        "repo_id": "RGenDiff/olitevaeB_im_f8c12",
        "filename": "olitevaeB_im_f8c12.safetensors",
        "config": {
            "target": "olvae.models.liteautoencoder.LiteAutoencoderKL",
            "params": {
                "use_ema": False,
                "embed_dim": 12,
                "use_quant": False,
                "encoder_config": {
                    "target": "olvae.modules.litevae.encoder_model.LiteVAE_Encoder",
                    "params": {
                        "in_channels": 3,
                        "dct_levels": 3,
                        "latent_dim": 12,
                        "image_size": 256,
                        "extractor_channels": 32,
                        "extractor_mult": [1, 2, 3],
                        "extractor_resblocks": 4,
                        "aggregate_channels": 32,
                        "aggregate_mult": [1, 2, 3],
                        "aggregate_resblocks": 4,
                    },
                },
                "decoder_config": {
                    "target": "olvae.modules.litevae.decoder_model.LiteVAE_Decoder",
                    "params": {
                        "channels": 128,
                        "z_channels": 12,
                        "out_channels": 3,
                        "channel_mult": [1, 2, 4, 4],
                        "num_res_blocks": 2,
                    },
                },
                "lossconfig": {"target": "torch.nn.Identity"},
            },
        },
    },
}


def _load_litevae(model_name: str = "litevae", device: str = "cuda"):
    """Load a pretrained LiteVAE model, downloading the checkpoint if needed.

    Requires open-litevae to be installed: pip install -e . from the open-litevae repo.

    Args:
        model_name: Key in LITEVAE_MODELS (currently only "litevae")
        device: Device to load model on
    """
    try:
        from omegaconf import OmegaConf
        from safetensors.torch import load_file
        from olvae.util import instantiate_from_config
    except ImportError:
        raise ImportError(
            "open-litevae is required for LiteVAE support. "
            "Install it from https://github.com/RGenDiff/open-litevae"
        )

    from huggingface_hub import hf_hub_download

    model_info = LITEVAE_MODELS[model_name]

    # Download checkpoint (cached automatically by huggingface_hub)
    checkpoint_path = hf_hub_download(
        repo_id=model_info["repo_id"],
        filename=model_info["filename"],
    )
    print(f"  LiteVAE checkpoint: {checkpoint_path}")

    config = OmegaConf.create({"model": model_info["config"]})
    sd = load_file(checkpoint_path)
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    for param in model.parameters():
        param.requires_grad = False

    return model


PRETRAINED_IMAGE_VAES = set(LITEVAE_MODELS.keys())


class ImageVAEBatchProcessor(BatchProcessor):
    """Batched GPU processing for encoding images through VAE."""

    def __init__(
        self,
        encode_fn,
        image_size: tuple[int, int] = (256, 256),
        normalize_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        device: str = "cuda",
    ):
        self.encode_fn = encode_fn
        self.image_size = image_size
        self.normalize_mean = torch.tensor(normalize_mean).view(3, 1, 1)
        self.normalize_std = torch.tensor(normalize_std).view(3, 1, 1)
        self.device = device

        # Move normalization tensors to device
        self.normalize_mean = self.normalize_mean.to(device)
        self.normalize_std = self.normalize_std.to(device)

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess a single PIL image to tensor.

        Args:
            image: PIL Image (RGB)

        Returns:
            Tensor [3, H, W] normalized to [-1, 1] (or whatever normalization is configured)
        """
        # Resize
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)

        # Convert to tensor [3, H, W] in [0, 1]
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        return tensor

    @torch.no_grad()
    def process_batch(
        self,
        data: list[Image.Image],
    ) -> Dict[str, torch.Tensor]:
        """
        Process batch of PIL images through VAE encoder.

        Args:
            data: list of PIL Images

        Returns:
            Dict with:
                - latents: [B, latent_channels, H', W'] VAE latents (mu / mode)
                - mu: [B, latent_channels, H', W'] latent means
                - logvar: [B, latent_channels, H', W'] latent log variances
        """
        # Preprocess images
        tensors = [self.preprocess_image(img) for img in data]
        images = torch.stack(tensors).to(self.device)  # [B, 3, H, W]

        # Normalize
        images = (images - self.normalize_mean) / self.normalize_std

        # Encode through VAE (encode_fn handles API differences)
        return self.encode_fn(images)


class ImageVAEDatasetPreprocessor(Preprocessor):
    """Preprocess image dataset to extract and save VAE latents with optional text conditioning."""

    def __init__(self, args, dataset, output_dir, shard_fields, batch_accumulators, stats_accumulator, device):
        self.args = args
        self.dataset = dataset
        self.output_dir = output_dir
        self.shard_fields = shard_fields
        self.batch_accumulators = batch_accumulators
        self.stats_accumulator = stats_accumulator
        self.device = device

        # Determine image source column
        self.image_column = args.image_column
        self.url_column = args.url_column

        if self.image_column is None and self.url_column is None:
            # Auto-detect
            columns = dataset.column_names
            if "image" in columns:
                self.image_column = "image"
            elif "url" in columns:
                self.url_column = "url"
            elif "image_url" in columns:
                self.url_column = "image_url"
            else:
                raise ValueError(
                    f"Could not auto-detect image column. Available columns: {columns}. "
                    "Please specify --image_column or --url_column."
                )

        print(f"Image source: {'column=' + self.image_column if self.image_column else 'url=' + self.url_column}")

        # Text conditioning settings
        self.text_column = args.text_column
        self.encode_text = args.encode_text
        self.t5_encoder = None

        # Text tokenization (for world model text input)
        self.tokenize_text = args.tokenize_text
        self.tokenizer = None
        if self.tokenize_text and self.text_column:
            from transformers import AutoTokenizer
            print(f"Loading tokenizer: {args.tokenizer_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
            self.max_seq_len = args.max_seq_len
            print(f"  Vocab size: {len(self.tokenizer)}")
            print(f"  Max seq len: {self.max_seq_len}")

        if self.text_column:
            print(f"Text conditioning: column={self.text_column}")
            if self.encode_text:
                print(f"  Encoding text with T5: {args.t5_model_name}")
                print(f"  Max text length: {args.t5_max_length}")
                self.t5_encoder = T5TextEncoder(
                    model_name=args.t5_model_name,
                    max_length=args.t5_max_length,
                    device=device,
                )
            else:
                print("  Storing raw text (no encoding)")
        else:
            print("Text conditioning: disabled")

        # Download settings
        self.download_workers = args.download_workers
        self.download_timeout = args.download_timeout
        print(f"  Download workers: {self.download_workers}")
        print(f"  Download timeout: {self.download_timeout}s")

        # Load VAE encoder
        self.vae_encoder = None
        self._encode_fn = None

        if args.vae_config in PRETRAINED_IMAGE_VAES:
            # Pretrained external VAE (e.g. LiteVAE)
            print(f"Loading pretrained image VAE: {args.vae_config}...")
            litevae_model = _load_litevae(args.vae_config, device=device)
            self.vae_encoder = litevae_model
            print(f"  Parameters: {sum(p.numel() for p in litevae_model.parameters()):,}")

            def _litevae_encode(images):
                posterior = litevae_model.encode(images)
                mu = posterior.mode()
                return {
                    "latents": mu.cpu(),
                    "mu": mu.cpu(),
                    "logvar": torch.zeros_like(mu).cpu(),
                }
            self._encode_fn = _litevae_encode

        elif args.vae_checkpoint_path:
            # Internal VAE
            print(f"Loading VAE encoder from {args.vae_checkpoint_path}...")
            from utils.model_loading_utils import load_model
            self.vae_model = load_model(
                args.vae_checkpoint_path,
                args.vae_config,
                device=device,
            )
            self.vae_encoder = self.vae_model.encoder

            def _internal_encode(images):
                mu, logvar = self.vae_encoder(images)
                return {
                    "latents": mu.cpu(),
                    "mu": mu.cpu(),
                    "logvar": logvar.cpu(),
                }
            self._encode_fn = _internal_encode

        else:
            print("No VAE checkpoint specified - will save raw images as tensors")

        self.image_size = (args.image_size, args.image_size)
        print(f"  Image size: {self.image_size}")
        print(f"  Total samples in dataset: {len(self.dataset):,}")

        # Initialize batch processor
        if self._encode_fn is not None:
            self.batch_processor = ImageVAEBatchProcessor(
                encode_fn=self._encode_fn,
                image_size=self.image_size,
                normalize_mean=tuple(args.normalize_mean),
                normalize_std=tuple(args.normalize_std),
                device=self.device,
            )
        else:
            self.batch_processor = None

        # Initialize shard fields
        shard_fields.update({
            'shard_images': [],  # Raw image tensors (if no VAE)
            'shard_latents': [],  # VAE latents
            'shard_mu': [],
            'shard_logvar': [],
            # Text fields
            'shard_text_embeddings': [],  # T5 encoded text [B, seq_len, hidden]
            'shard_text_attention_mask': [],  # Attention mask [B, seq_len]
            'shard_raw_text': [],  # Raw text strings
            # Tokenized text fields
            'shard_token_ids': [],  # Tokenized text [N, T]
            'shard_text_lengths': [],  # Token counts [N]
        })

        # Initialize batch accumulators
        batch_accumulators['batch_images'] = []
        batch_accumulators['batch_urls'] = []
        batch_accumulators['batch_texts'] = []

    @classmethod
    def add_cli_args(cls, subparsers):
        sub_parser = subparsers.add_parser("image-vae", help="Preprocess image dataset for VAE training")

        # VAE model — set --vae_config to "litevae" for pretrained LiteVAE
        # (auto-downloads from HuggingFace), or an internal config name with --vae_checkpoint_path.
        sub_parser.add_argument("--vae_checkpoint_path", type=str, default=None,
                                help="Path to internal VAE checkpoint (not needed for pretrained VAEs)")
        sub_parser.add_argument("--vae_config", type=str, default="default",
                                help="VAE config name, or 'litevae' for pretrained LiteVAE (auto-downloaded)")

        # Image settings
        sub_parser.add_argument("--image_size", type=int, default=256,
                                help="Target image size (square)")
        sub_parser.add_argument("--normalize_mean", type=float, nargs=3, default=[0.5, 0.5, 0.5],
                                help="Normalization mean (R, G, B)")
        sub_parser.add_argument("--normalize_std", type=float, nargs=3, default=[0.5, 0.5, 0.5],
                                help="Normalization std (R, G, B)")

        # Data source
        sub_parser.add_argument("--image_column", type=str, default=None,
                                help="Column containing PIL images or image arrays")
        sub_parser.add_argument("--url_column", type=str, default=None,
                                help="Column containing image URLs to download")

        # Download settings
        sub_parser.add_argument("--download_workers", type=int, default=8,
                                help="Number of parallel download workers for URL-based datasets")
        sub_parser.add_argument("--download_timeout", type=float, default=10.0,
                                help="Timeout in seconds for image downloads")

        # Filtering
        sub_parser.add_argument("--min_image_size", type=int, default=64,
                                help="Minimum image dimension (skip smaller images)")
        sub_parser.add_argument("--skip_grayscale", action="store_true", default=False,
                                help="Skip grayscale images")

        # Text conditioning
        sub_parser.add_argument("--text_column", type=str, default=None,
                                help="Column containing text captions/prompts/descriptions")
        sub_parser.add_argument("--encode_text", action="store_true", default=False,
                                help="Encode text through T5 (if False, stores raw text)")
        sub_parser.add_argument("--t5_model_name", type=str, default="google/t5-v1_1-small",
                                help="T5 model name for text encoding")
        sub_parser.add_argument("--t5_max_length", type=int, default=128,
                                help="Maximum token length for T5 encoding")

        # Text tokenization (for world model training — saves token IDs alongside latents)
        sub_parser.add_argument("--tokenize_text", action="store_true", default=False,
                                help="Tokenize text into token IDs (for world model text input)")
        sub_parser.add_argument("--tokenizer_name", type=str, default="mistralai/Mistral-7B-v0.1",
                                help="HuggingFace tokenizer for --tokenize_text")
        sub_parser.add_argument("--max_seq_len", type=int, default=128,
                                help="Maximum token sequence length for --tokenize_text")
        
        return sub_parser

    def _load_image_from_example(self, example) -> Optional[Image.Image]:
        """Load image from dataset example (either from image column or by downloading URL)."""
        try:
            if self.image_column:
                img_data = example[self.image_column]
                if isinstance(img_data, Image.Image):
                    return img_data.convert("RGB")
                elif hasattr(img_data, "convert"):
                    return img_data.convert("RGB")
                elif isinstance(img_data, dict) and "bytes" in img_data:
                    return Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
                elif isinstance(img_data, dict) and "path" in img_data:
                    return Image.open(img_data["path"]).convert("RGB")
                else:
                    # Try treating as numpy array
                    import numpy as np
                    if isinstance(img_data, np.ndarray):
                        return Image.fromarray(img_data).convert("RGB")
                    return None
            elif self.url_column:
                url = example[self.url_column]
                return download_image(url, timeout=self.download_timeout)
        except Exception:
            return None
        return None

    def flush_shard(self):
        if not self.shard_fields['shard_latents'] and not self.shard_fields['shard_images']:
            return

        if self.vae_encoder is not None:
            # Save VAE latents
            shard_data = {
                "latents": torch.cat(self.shard_fields['shard_latents'], dim=0),
                "mu": torch.cat(self.shard_fields['shard_mu'], dim=0),
                "logvar": torch.cat(self.shard_fields['shard_logvar'], dim=0),
                "num_samples": sum(l.shape[0] for l in self.shard_fields['shard_latents']),
            }
        else:
            # Save raw image tensors
            shard_data = {
                "images": torch.cat(self.shard_fields['shard_images'], dim=0),
                "num_samples": sum(i.shape[0] for i in self.shard_fields['shard_images']),
            }

        # Add text data if present
        if self.text_column:
            if self.encode_text and self.shard_fields['shard_text_embeddings']:
                shard_data["text_embeddings"] = torch.cat(self.shard_fields['shard_text_embeddings'], dim=0)
                shard_data["text_attention_mask"] = torch.cat(self.shard_fields['shard_text_attention_mask'], dim=0)
            elif self.shard_fields['shard_raw_text']:
                shard_data["text"] = self.shard_fields['shard_raw_text'].copy()

            # Add tokenized text if enabled
            if self.tokenize_text and self.shard_fields['shard_token_ids']:
                max_len = max(t.shape[-1] for t in self.shard_fields['shard_token_ids'])
                pad_id = self.tokenizer.pad_token_id or 0
                padded = []
                for tokens in self.shard_fields['shard_token_ids']:
                    if tokens.shape[-1] < max_len:
                        tokens = F.pad(tokens, (0, max_len - tokens.shape[-1]), value=pad_id)
                    padded.append(tokens)
                shard_data["token_ids"] = torch.stack(padded, dim=0)
                shard_data["text_lengths"] = torch.stack(self.shard_fields['shard_text_lengths'], dim=0)

        shard_path = os.path.join(self.output_dir, f"shard_{self.shard_fields['shard_idx']:06d}.pt")
        torch.save(shard_data, shard_path)

        print(f"  Saved shard {self.shard_fields['shard_idx']} ({shard_data['num_samples']} samples)")

        # Reset shard fields
        self.shard_fields['shard_images'] = []
        self.shard_fields['shard_latents'] = []
        self.shard_fields['shard_mu'] = []
        self.shard_fields['shard_logvar'] = []
        self.shard_fields['shard_text_embeddings'] = []
        self.shard_fields['shard_text_attention_mask'] = []
        self.shard_fields['shard_raw_text'] = []
        self.shard_fields['shard_token_ids'] = []
        self.shard_fields['shard_text_lengths'] = []
        self.shard_fields['shard_idx'] += 1

    def process_and_accumulate(self):
        if not self.batch_accumulators['batch_images']:
            return

        try:
            if self.batch_processor is not None:
                # Process through VAE
                result = self.batch_processor.process_batch(self.batch_accumulators['batch_images'])

                self.shard_fields['shard_latents'].append(result["latents"])
                self.shard_fields['shard_mu'].append(result["mu"])
                self.shard_fields['shard_logvar'].append(result["logvar"])
            else:
                # Save raw tensors
                tensors = []
                for img in self.batch_accumulators['batch_images']:
                    img = img.resize(self.image_size, Image.Resampling.LANCZOS)
                    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                    tensors.append(tensor)
                self.shard_fields['shard_images'].append(torch.stack(tensors))

            # Process text if enabled
            if self.text_column and self.batch_accumulators['batch_texts']:
                if self.encode_text and self.t5_encoder is not None:
                    # Encode through T5
                    text_result = self.t5_encoder.encode_batch(self.batch_accumulators['batch_texts'])
                    self.shard_fields['shard_text_embeddings'].append(text_result["text_embeddings"])
                    self.shard_fields['shard_text_attention_mask'].append(text_result["text_attention_mask"])
                else:
                    # Store raw text
                    self.shard_fields['shard_raw_text'].extend(self.batch_accumulators['batch_texts'])

                # Tokenize text if enabled
                if self.tokenize_text and self.tokenizer is not None:
                    encoded = self.tokenizer(
                        self.batch_accumulators['batch_texts'],
                        truncation=True,
                        max_length=self.max_seq_len,
                        padding=False,
                        return_attention_mask=False,
                    )
                    for input_ids in encoded["input_ids"]:
                        self.shard_fields['shard_token_ids'].append(torch.tensor(input_ids, dtype=torch.long))
                        self.shard_fields['shard_text_lengths'].append(torch.tensor(len(input_ids), dtype=torch.long))

            self.stats_accumulator["saved"] += len(self.batch_accumulators['batch_images'])

            # Flush shard if full
            if self.vae_encoder is not None:
                current_size = sum(l.shape[0] for l in self.shard_fields['shard_latents'])
            else:
                current_size = sum(i.shape[0] for i in self.shard_fields['shard_images'])

            if current_size >= self.args.shard_size:
                self.flush_shard()

        except Exception as e:
            print(f"Batch processing error: {e}")
            traceback.print_exc()
            self.stats_accumulator["skipped"]["error"] += len(self.batch_accumulators['batch_images'])

        self.batch_accumulators['batch_images'] = []
        self.batch_accumulators['batch_texts'] = []

    def _extract_texts_from_example(self, example) -> list[str]:
        """Extract text(s) from dataset example. Returns a list of captions."""
        if not self.text_column:
            return []

        try:
            text = example.get(self.text_column)
            if text is None:
                return []

            # Handle various text formats
            if isinstance(text, str):
                stripped = text.strip()
                return [stripped] if stripped else []
            elif isinstance(text, list):
                # Multiple captions (e.g. Flickr30k) — return all non-empty
                return [str(t).strip() for t in text if str(t).strip()]
            else:
                stripped = str(text).strip()
                return [stripped] if stripped else []
        except Exception:
            return []

    def preprocess_example(self, example) -> bool:
        """Process a single example. Returns True if example was processed."""
        # Load image
        image = self._load_image_from_example(example)

        if image is None:
            self.stats_accumulator["skipped"]["download_failed"] += 1
            return False

        # Check minimum size
        if min(image.size) < self.args.min_image_size:
            self.stats_accumulator["skipped"]["too_small"] += 1
            return False

        # Check grayscale
        if self.args.skip_grayscale and image.mode == "L":
            self.stats_accumulator["skipped"]["grayscale"] += 1
            return False

        # Extract text(s) if configured
        if self.text_column:
            texts = self._extract_texts_from_example(example)
            if not texts:
                self.stats_accumulator["skipped"]["missing_text"] += 1
                return False

            # One example per caption-image pair
            for caption in texts:
                self.batch_accumulators['batch_images'].append(image)
                self.batch_accumulators['batch_texts'].append(caption)

                if len(self.batch_accumulators['batch_images']) >= self.args.gpu_batch_size:
                    self.process_and_accumulate()
        else:
            self.batch_accumulators['batch_images'].append(image)

            if len(self.batch_accumulators['batch_images']) >= self.args.gpu_batch_size:
                self.process_and_accumulate()

        return True

    def parse_config(self) -> dict:
        config = {
            "vae_checkpoint": self.args.vae_checkpoint_path,
            "vae_config": self.args.vae_config,
            "dataset_name": self.args.dataset_name,
            "dataset_config": self.args.dataset_config,
            "split": self.args.split,
            "image_size": self.args.image_size,
            "normalize_mean": self.args.normalize_mean,
            "normalize_std": self.args.normalize_std,
            "image_column": self.image_column,
            "url_column": self.url_column,
            "download_workers": self.args.download_workers,
            "download_timeout": self.args.download_timeout,
            "min_image_size": self.args.min_image_size,
            "skip_grayscale": self.args.skip_grayscale,
            "shard_size": self.args.shard_size,
            "has_vae_latents": self.vae_encoder is not None,
            # Text conditioning
            "text_column": self.text_column,
            "encode_text": self.encode_text,
            "has_text_embeddings": self.text_column is not None and self.encode_text,
            "has_raw_text": self.text_column is not None and not self.encode_text,
            "stats": self.stats_accumulator,
        }

        if self.encode_text:
            config["t5_model_name"] = self.args.t5_model_name
            config["t5_max_length"] = self.args.t5_max_length
            if self.t5_encoder is not None:
                config["t5_hidden_size"] = self.t5_encoder.encoder.config.d_model

        if self.tokenize_text:
            config["tokenizer_name"] = self.args.tokenizer_name
            config["max_seq_len"] = self.args.max_seq_len
            config["vocab_size"] = len(self.tokenizer) if self.tokenizer else None

        return config
