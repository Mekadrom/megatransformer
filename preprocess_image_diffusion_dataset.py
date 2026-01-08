#!/usr/bin/env python3
"""
Single-GPU preprocessing worker.
Processes every Nth sample where N is total_gpus.

This is designed to be run in parallel with other instances on different GPUs.
"""

import os
import io
import json
import argparse
import time
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from torch.amp import autocast
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import requests
from datasets import load_dataset

from transformers import T5Tokenizer, T5EncoderModel


def download_image(url: str, timeout: int = 5, min_size: int = 64) -> Optional[Image.Image]:
    """Download and validate image."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, timeout=timeout, headers=headers, stream=True)
        response.raise_for_status()
        
        if 'image' not in response.headers.get('content-type', '').lower():
            return None
        
        img = Image.open(io.BytesIO(response.content))
        
        if img.width < min_size or img.height < min_size:
            return None
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img
    except Exception:
        return None


def load_vae(checkpoint_path: str, vae_config: str, latent_channels: int, device: str):
    """Load VAE model."""
    from model.image.vae import model_config_lookup
    
    model = model_config_lookup[vae_config](
        latent_channels=latent_channels,
        perceptual_loss_type="none",
    )
    
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
    else:
        state_dict = torch.load(pytorch_path, map_location=device)
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    
    return model


class BatchProcessor:
    """Batched GPU processing for VAE and T5."""
    
    def __init__(
        self,
        text_model_name: str,
        vae_checkpoint: Optional[str],
        vae_config: str,
        latent_channels: int,
        image_size: int,
        max_text_length: int,
        device: str = "cuda",
    ):
        self.device = device
        self.max_text_length = max_text_length
        
        # Text encoder
        print(f"Loading T5 ({text_model_name})...")
        self.tokenizer = T5Tokenizer.from_pretrained(text_model_name)
        self.text_model = T5EncoderModel.from_pretrained(text_model_name)
        self.text_model.eval().to(device)
        
        # VAE
        self.vae = None
        if vae_checkpoint:
            print(f"Loading VAE ({vae_config})...")
            self.vae = load_vae(vae_checkpoint, vae_config, latent_channels, device)
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    
    @torch.no_grad()
    def process_batch(
        self, 
        images: List[Image.Image], 
        captions: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Process batch of images and captions."""
        # Transform images
        image_tensors = torch.stack([self.transform(img) for img in images])
        
        # Encode text (batched)
        text_inputs = self.tokenizer(
            captions,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        
        with autocast(device_type="cuda", dtype=torch.float16):
            text_out = self.text_model(**text_inputs).last_hidden_state
        
        result = {
            "images": image_tensors,
            "text_embeddings": text_out.cpu().float(),
            "text_attention_mask": text_inputs['attention_mask'].cpu(),
        }
        
        # VAE encode
        if self.vae is not None:
            with autocast(device_type="cuda", dtype=torch.float16):
                latent_mu, _ = self.vae.encoder(image_tensors.to(self.device))
            result["latent_mu"] = latent_mu.cpu().float()
        
        return result


def save_shard(shard_data: Dict[str, torch.Tensor], shard_path: str):
    """Save a shard to disk."""
    torch.save(shard_data, shard_path)


def main():
    parser = argparse.ArgumentParser()
    
    # Dataset
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="laion/relaion400m")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--url_column", type=str, default="URL")
    parser.add_argument("--caption_column", type=str, default="TEXT")
    
    # Multi-GPU
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--total_gpus", type=int, default=1)
    
    # Model
    parser.add_argument("--text_model", type=str, default="t5-small")
    parser.add_argument("--max_text_length", type=int, default=512)
    parser.add_argument("--vae_checkpoint", type=str, default=None)
    parser.add_argument("--vae_config", type=str, default="mini")
    parser.add_argument("--latent_channels", type=int, default=4)
    
    # Processing
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--min_image_size", type=int, default=64)
    parser.add_argument("--gpu_batch_size", type=int, default=64)
    parser.add_argument("--num_download_workers", type=int, default=32)
    parser.add_argument("--download_timeout", type=int, default=5)
    parser.add_argument("--shard_size", type=int, default=10000)
    
    # Limits
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_expected_examples", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"
    
    print(f"GPU {args.gpu_id}/{args.total_gpus}")
    print(f"Processing every {args.total_gpus}th sample starting at offset {args.gpu_id}")
    
    # Load dataset
    print(f"Loading dataset {args.dataset_name}...")
    if args.dataset_config:
        dataset = load_dataset(
            args.dataset_name, args.dataset_config,
            split=args.split, streaming=True, trust_remote_code=True
        )
    else:
        dataset = load_dataset(
            args.dataset_name, split=args.split, 
            streaming=True, trust_remote_code=True
        )
    
    # Initialize processor
    processor = BatchProcessor(
        text_model_name=args.text_model,
        vae_checkpoint=args.vae_checkpoint,
        vae_config=args.vae_config,
        latent_channels=args.latent_channels,
        image_size=args.image_size,
        max_text_length=args.max_text_length,
        device=device,
    )
    
    # Stats
    stats = {
        "processed": 0,
        "saved": 0,
        "skipped_no_caption": 0,
        "skipped_download": 0,
        "skipped_error": 0,
    }
    
    # Shard accumulator
    shard_images = []
    shard_text_emb = []
    shard_text_mask = []
    shard_captions = []
    shard_latents = []
    shard_idx = 0
    
    def flush_shard():
        nonlocal shard_images, shard_text_emb, shard_text_mask, shard_captions, shard_latents, shard_idx
        
        if not shard_images:
            return
        
        shard_data = {
            "images": torch.stack(shard_images),
            "text_embeddings": torch.stack(shard_text_emb),
            "text_attention_mask": torch.stack(shard_text_mask),
            "captions": shard_captions,
            "num_samples": len(shard_images),
        }
        
        if shard_latents:
            shard_data["latent_mu"] = torch.stack(shard_latents)
        
        shard_path = os.path.join(args.output_dir, f"shard_{shard_idx:06d}.pt")
        torch.save(shard_data, shard_path)
        
        print(f"  Saved shard {shard_idx} ({len(shard_images)} samples)")
        
        shard_images = []
        shard_text_emb = []
        shard_text_mask = []
        shard_captions = []
        shard_latents = []
        shard_idx += 1
    
    # Batch accumulator
    batch_images = []
    batch_captions = []
    
    def process_and_accumulate():
        nonlocal batch_images, batch_captions
        nonlocal shard_images, shard_text_emb, shard_text_mask, shard_latents
        
        if not batch_images:
            return
        
        try:
            result = processor.process_batch(batch_images, batch_captions)
            
            # Add to shard
            for i in range(len(batch_images)):
                shard_images.append(result["images"][i])
                shard_text_emb.append(result["text_embeddings"][i])
                shard_text_mask.append(result["text_attention_mask"][i])
                shard_captions.append(batch_captions[i])
                if "latent_mu" in result:
                    shard_latents.append(result["latent_mu"][i])
                stats["saved"] += 1
            
            # Flush shard if full
            if len(shard_images) >= args.shard_size:
                flush_shard()
                
        except Exception as e:
            print(f"Batch processing error: {e}")
            stats["skipped_error"] += len(batch_images)
        
        batch_images = []
        batch_captions = []
    
    # Download function for parallel execution
    def try_download(item):
        idx, example = item
        
        caption = example.get(args.caption_column)
        if isinstance(caption, list):
            caption = caption[0] if caption else None
        if not caption or not str(caption).strip():
            return ("no_caption", None)
        
        url = example.get(args.url_column)
        if not url:
            return ("no_url", None)
        
        img = download_image(url, args.download_timeout, args.min_image_size)
        if img is None:
            return ("download_failed", None)
        
        return ("success", (img, caption))
    
    # Main loop
    pbar = tqdm(total=args.num_expected_examples, desc=f"GPU {args.gpu_id}")
    start_time = time.time()
    
    # Buffer for parallel downloads
    download_buffer = []
    buffer_size = args.num_download_workers * 4
    
    dataset_iter = iter(dataset)
    global_idx = 0
    
    try:
        while True:
            # Check limits
            if args.max_samples and stats["saved"] >= args.max_samples:
                break
            
            # Fill download buffer with samples for this GPU
            while len(download_buffer) < buffer_size:
                try:
                    example = next(dataset_iter)
                except StopIteration:
                    break
                
                # Skip if not our sample
                if global_idx >= args.start_idx and global_idx % args.total_gpus == args.gpu_id:
                    download_buffer.append((global_idx, example))
                
                global_idx += 1
            
            if not download_buffer:
                break
            
            # Parallel download
            with ThreadPoolExecutor(max_workers=args.num_download_workers) as executor:
                futures = {executor.submit(try_download, item): item for item in download_buffer}
                
                for future in as_completed(futures):
                    status, data = future.result()
                    stats["processed"] += 1
                    pbar.update(1)
                    
                    if status == "no_caption":
                        stats["skipped_no_caption"] += 1
                    elif status in ("no_url", "download_failed"):
                        stats["skipped_download"] += 1
                    elif status == "success":
                        img, caption = data
                        batch_images.append(img)
                        batch_captions.append(caption)
                        
                        # Process batch when full
                        if len(batch_images) >= args.gpu_batch_size:
                            process_and_accumulate()
            
            download_buffer = []
    
    except KeyboardInterrupt:
        print("\nInterrupted!")
    
    # Final flush
    process_and_accumulate()
    flush_shard()
    
    pbar.close()
    elapsed = time.time() - start_time
    
    # Save stats
    stats["elapsed_seconds"] = elapsed
    stats["samples_per_second"] = stats["saved"] / elapsed if elapsed > 0 else 0
    stats["gpu_id"] = args.gpu_id
    stats["total_gpus"] = args.total_gpus
    
    with open(os.path.join(args.output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"GPU {args.gpu_id} complete!")
    print(f"  Processed: {stats['processed']:,}")
    print(f"  Saved: {stats['saved']:,}")
    print(f"  Skipped (no caption): {stats['skipped_no_caption']:,}")
    print(f"  Skipped (download): {stats['skipped_download']:,}")
    print(f"  Time: {elapsed/3600:.2f} hours")
    print(f"  Speed: {stats['saved']/elapsed:.1f} samples/sec")
    print(f"  Shards: {shard_idx}")


if __name__ == "__main__":
    main()
