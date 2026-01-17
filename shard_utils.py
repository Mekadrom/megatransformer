#!/usr/bin/env python3
"""
Utilities for working with preprocessed shards:
1. Merge shards from multiple GPU outputs
2. ShardedDataset for training
"""

import os
import json
import argparse
import random
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


# ============================================================================
# Merge shards from parallel preprocessing
# ============================================================================

def merge_shards(
    input_dir: str,
    output_dir: str,
    shuffle: bool = True,
    target_shard_size: int = 10000,
):
    """
    Merge shards from multiple GPU subdirectories into a single output.
    
    Args:
        input_dir: Directory containing gpu_0/, gpu_1/, etc. subdirs
        output_dir: Output directory for merged shards
        shuffle: Whether to shuffle samples across shards
        target_shard_size: Target samples per output shard
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all shard files
    all_shards = []
    gpu_dirs = sorted([
        d for d in os.listdir(input_dir) 
        if d.startswith("gpu_") and os.path.isdir(os.path.join(input_dir, d))
    ])
    
    if not gpu_dirs:
        # Maybe shards are directly in input_dir
        gpu_dirs = ["."]
    
    print(f"Found GPU directories: {gpu_dirs}")
    
    for gpu_dir in gpu_dirs:
        shard_dir = os.path.join(input_dir, gpu_dir)
        shards = sorted([
            os.path.join(shard_dir, f) 
            for f in os.listdir(shard_dir) 
            if f.startswith("shard_") and f.endswith(".pt")
        ])
        all_shards.extend(shards)
        print(f"  {gpu_dir}: {len(shards)} shards")
    
    print(f"Total shards to merge: {len(all_shards)}")
    
    if shuffle:
        random.shuffle(all_shards)
    
    # Accumulator for new shards
    acc_images = []
    acc_text_emb = []
    acc_text_mask = []
    acc_latents = []
    has_latents = None
    
    output_shard_idx = 0
    total_samples = 0
    
    def save_merged_shard():
        nonlocal acc_images, acc_text_emb, acc_text_mask, acc_latents, output_shard_idx
        
        if not acc_images:
            return
        
        shard_data = {
            "images": torch.cat(acc_images, dim=0),
            "text_embeddings": torch.cat(acc_text_emb, dim=0),
            "text_attention_mask": torch.cat(acc_text_mask, dim=0),
            "num_samples": sum(x.shape[0] for x in acc_images),
        }
        
        if acc_latents:
            shard_data["latent_mu"] = torch.cat(acc_latents, dim=0)
        
        output_path = os.path.join(output_dir, f"shard_{output_shard_idx:06d}.pt")
        torch.save(shard_data, output_path)
        
        output_shard_idx += 1
        acc_images = []
        acc_text_emb = []
        acc_text_mask = []
        acc_latents = []
    
    # Process all input shards
    for shard_path in tqdm(all_shards, desc="Merging"):
        try:
            shard = torch.load(shard_path)
            
            # Check for latents
            if has_latents is None:
                has_latents = "latent_mu" in shard
            
            acc_images.append(shard["images"])
            acc_text_emb.append(shard["text_embeddings"])
            acc_text_mask.append(shard["text_attention_mask"])
            
            if has_latents and "latent_mu" in shard:
                acc_latents.append(shard["latent_mu"])
            
            total_samples += shard["num_samples"]
            
            # Save if accumulated enough
            current_size = sum(x.shape[0] for x in acc_images)
            if current_size >= target_shard_size:
                save_merged_shard()
                
        except Exception as e:
            print(f"Error loading {shard_path}: {e}")
            continue
    
    # Final save
    save_merged_shard()
    
    # Save metadata
    meta = {
        "total_samples": total_samples,
        "num_shards": output_shard_idx,
        "shard_size": target_shard_size,
        "has_latents": has_latents,
        "source_dirs": gpu_dirs,
    }
    
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nMerge complete!")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Output shards: {output_shard_idx}")
    print(f"  Output dir: {output_dir}")


# ============================================================================
# ShardedDataset for training
# ============================================================================

class ShardedDataset(Dataset):
    """
    Efficient dataset that loads preprocessed shards.
    
    Features:
    - Lazy loading of shards
    - LRU cache for recently accessed shards
    - Optional shuffling within and across shards
    """
    
    def __init__(
        self,
        shard_dir: str,
        use_latents: bool = True,
        cache_size: int = 3,
        return_image: bool = False,
    ):
        """
        Args:
            shard_dir: Directory containing shard_*.pt files
            use_latents: Whether to return latent_mu (for latent diffusion)
            cache_size: Number of shards to keep in memory
            return_image: Whether to also return the image tensor
        """
        self.shard_dir = shard_dir
        self.use_latents = use_latents
        self.cache_size = cache_size
        self.return_image = return_image
        
        # Find all shards
        self.shard_files = sorted([
            f for f in os.listdir(shard_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        ])
        
        if not self.shard_files:
            raise ValueError(f"No shard files found in {shard_dir}")
        
        # Build index
        self.shard_offsets = []
        self.total_samples = 0
        
        print(f"Indexing {len(self.shard_files)} shards...")
        for shard_file in tqdm(self.shard_files):
            self.shard_offsets.append(self.total_samples)
            # Quick peek at shard size
            shard_path = os.path.join(shard_dir, shard_file)
            shard = torch.load(shard_path, map_location="cpu")
            self.total_samples += shard["num_samples"]
        
        print(f"Total samples: {self.total_samples:,}")
        
        # LRU cache
        self._cache = {}
        self._cache_order = []
    
    def _load_shard(self, shard_idx: int) -> Dict[str, torch.Tensor]:
        """Load shard with caching."""
        if shard_idx in self._cache:
            # Move to end (most recent)
            self._cache_order.remove(shard_idx)
            self._cache_order.append(shard_idx)
            return self._cache[shard_idx]
        
        # Load from disk
        shard_path = os.path.join(self.shard_dir, self.shard_files[shard_idx])
        shard = torch.load(shard_path, map_location="cpu")
        
        # Add to cache
        self._cache[shard_idx] = shard
        self._cache_order.append(shard_idx)
        
        # Evict if too large
        while len(self._cache) > self.cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
        
        return shard
    
    def _find_shard(self, idx: int) -> tuple:
        """Find which shard contains the given index."""
        # Binary search
        lo, hi = 0, len(self.shard_offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.shard_offsets[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        return lo, idx - self.shard_offsets[lo]
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shard_idx, local_idx = self._find_shard(idx)
        shard = self._load_shard(shard_idx)
        
        sample = {
            "text_embeddings": shard["text_embeddings"][local_idx],
            "text_attention_mask": shard["text_attention_mask"][local_idx],
        }
        
        if self.use_latents and "latent_mu" in shard:
            sample["latent_mu"] = shard["latent_mu"][local_idx]
            sample["latent_shape"] = list(shard["latent_mu"][local_idx].shape)
        
        if self.return_image:
            sample["image"] = shard["images"][local_idx]
        
        return sample


def _shard_debug_enabled():
    """Check if shard debugging is enabled via environment variable."""
    return os.environ.get("SHARD_DEBUG", "0") == "1"


class ShardAwareSampler(torch.utils.data.Sampler):
    """
    Sampler that groups indices by shard to minimize shard loading.

    Instead of random access across all shards (which causes constant disk I/O),
    this sampler:
    1. Assigns shards to ranks ONCE at initialization (for cache efficiency)
    2. Each epoch, shuffles the order of each rank's assigned shards
    3. Within each shard, shuffles the sample indices
    4. Iterates through all samples in one shard before moving to the next

    This ensures each shard is loaded only once per epoch, and shard-to-rank
    assignment is stable across epochs (critical for caching).

    Supports distributed training by splitting shards across ranks.

    Set SHARD_DEBUG=1 environment variable to enable verbose logging.
    """

    def __init__(
        self,
        shard_offsets: List[int],
        total_samples: int,
        shuffle: bool = True,
        seed: int = 42,
        num_replicas: int = None,
        rank: int = None,
    ):
        """
        Args:
            shard_offsets: List of starting indices for each shard
            total_samples: Total number of samples across all shards
            shuffle: Whether to shuffle shards and samples within shards
            seed: Random seed for reproducibility
            num_replicas: Number of distributed processes (auto-detected if None)
            rank: Rank of current process (auto-detected if None)
        """
        self.shard_offsets = shard_offsets
        self.total_samples = total_samples
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._iter_count = 0  # Track how many times __iter__ is called

        # Auto-detect distributed settings
        if num_replicas is None:
            if torch.distributed.is_initialized():
                num_replicas = torch.distributed.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0

        self.num_replicas = num_replicas
        self.rank = rank

        # Compute shard sizes
        self.shard_sizes = []
        for i in range(len(shard_offsets)):
            if i + 1 < len(shard_offsets):
                size = shard_offsets[i + 1] - shard_offsets[i]
            else:
                size = total_samples - shard_offsets[i]
            self.shard_sizes.append(size)

        # Assign shards to ranks ONCE at initialization (stable across epochs)
        # This is critical for caching - each rank always gets the same shards
        num_shards = len(shard_offsets)
        all_shards = list(range(num_shards))
        if self.num_replicas > 1:
            # Round-robin assignment: rank 0 gets [0, 2, 4, ...], rank 1 gets [1, 3, 5, ...]
            self.my_shards = [all_shards[i] for i in range(self.rank, num_shards, self.num_replicas)]
        else:
            self.my_shards = all_shards

        # Compute number of samples for this rank
        self._num_samples = sum(self.shard_sizes[s] for s in self.my_shards)

        # CRITICAL: Log sample counts to detect rank divergence
        print(f"[ShardAwareSampler] Rank {self.rank}/{self.num_replicas}: "
              f"{len(self.my_shards)} shards, {self._num_samples} samples. "
              f"First 5 shards: {self.my_shards[:5]}")

        # Warn if ranks have different sample counts (causes NCCL timeouts!)
        if self.num_replicas > 1 and torch.distributed.is_initialized():
            # Gather sample counts from all ranks
            sample_counts = [torch.zeros(1, dtype=torch.long, device='cuda') for _ in range(self.num_replicas)]
            my_count = torch.tensor([self._num_samples], dtype=torch.long, device='cuda')
            torch.distributed.all_gather(sample_counts, my_count)
            sample_counts = [c.item() for c in sample_counts]

            if len(set(sample_counts)) > 1:
                print(f"[ShardAwareSampler] WARNING: Ranks have DIFFERENT sample counts: {sample_counts}")
                print(f"[ShardAwareSampler] This WILL cause NCCL timeouts! Padding to max count.")
                max_samples = max(sample_counts)
                self._num_samples = max_samples

    def __iter__(self):
        import time
        iter_start = time.time()
        self._iter_count += 1

        if _shard_debug_enabled():
            print(f"[ShardAwareSampler] Rank {self.rank}: __iter__ called (#{self._iter_count}), epoch={self.epoch}")

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Shuffle the ORDER of this rank's shards (but not which shards it gets)
        if self.shuffle:
            perm = torch.randperm(len(self.my_shards), generator=g).tolist()
            shard_order = [self.my_shards[i] for i in perm]
        else:
            shard_order = self.my_shards

        if _shard_debug_enabled():
            print(f"[ShardAwareSampler] Rank {self.rank}: shard_order first 5: {shard_order[:5]}")

        # Generate indices shard by shard
        indices = []
        for shard_idx in shard_order:
            start = self.shard_offsets[shard_idx]
            size = self.shard_sizes[shard_idx]

            # Indices within this shard
            shard_indices = list(range(start, start + size))

            if self.shuffle:
                # Shuffle within shard (use shard-specific seed for reproducibility)
                shard_g = torch.Generator()
                shard_g.manual_seed(self.seed + self.epoch + shard_idx)
                perm = torch.randperm(len(shard_indices), generator=shard_g).tolist()
                shard_indices = [shard_indices[i] for i in perm]

            indices.extend(shard_indices)

        # Pad indices to match _num_samples if needed (for distributed sync)
        # This handles the case where ranks have different actual sample counts
        actual_samples = len(indices)
        if actual_samples < self._num_samples:
            # Repeat indices from the beginning to pad
            padding_needed = self._num_samples - actual_samples
            indices.extend(indices[:padding_needed])
            if _shard_debug_enabled():
                print(f"[ShardAwareSampler] Rank {self.rank}: Padded {padding_needed} samples "
                      f"({actual_samples} -> {len(indices)}) for distributed sync")

        if _shard_debug_enabled():
            elapsed = time.time() - iter_start
            print(f"[ShardAwareSampler] Rank {self.rank}: __iter__ generated {len(indices)} indices in {elapsed:.2f}s")

        # Wrap iterator to detect exhaustion
        return self._tracked_iter(indices)

    def _tracked_iter(self, indices):
        """Yield indices and log when exhausted."""
        count = 0
        for idx in indices:
            count += 1
            yield idx
        if _shard_debug_enabled():
            print(f"[ShardAwareSampler] Rank {self.rank}: iterator EXHAUSTED after {count} samples (epoch {self.epoch})")

    def __len__(self):
        return self._num_samples

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling reproducibility (important for distributed training)."""
        old_epoch = self.epoch
        self.epoch = epoch
        if _shard_debug_enabled():
            print(f"[ShardAwareSampler] Rank {self.rank}: set_epoch({epoch}) called (was {old_epoch})")


class AudioVAEShardedDataset(Dataset):
    """
    Efficient dataset for loading preprocessed audio VAE shards.

    Loads shards containing:
    - mel_specs: [B, 1, n_mels, T] mel spectrograms
    - mel_lengths: [B] original lengths before padding
    - speaker_embeddings: [B, embedding_dim] speaker embeddings
    - texts: List[str] text transcriptions (optional, for ASR evaluation)

    Features:
    - Lazy loading of shards
    - LRU cache for recently accessed shards
    - Cached shard index for fast startup (saves indexing time on large datasets)

    IMPORTANT: Use get_sampler() to get a ShardAwareSampler for efficient training.
    Random access without shard-aware sampling will be extremely slow.
    """

    SHARD_INDEX_FILE = "shard_index.json"

    def __init__(
        self,
        shard_dir: str,
        cache_size: int = 3,
        audio_max_frames: int = 1875,
    ):
        """
        Args:
            shard_dir: Directory containing shard_*.pt files
            cache_size: Number of shards to keep in memory
            audio_max_frames: Maximum mel spectrogram frames for padding
        """
        self.shard_dir = shard_dir
        self.cache_size = cache_size
        self.audio_max_frames = audio_max_frames

        # Try to load cached index first
        index_path = os.path.join(shard_dir, self.SHARD_INDEX_FILE)
        if os.path.exists(index_path):
            self._load_cached_index(index_path)
        else:
            self._build_and_cache_index(index_path)

        # LRU cache
        self._cache = {}
        self._cache_order = []

    def _load_cached_index(self, index_path: str):
        """Load pre-computed shard index from JSON file."""
        print(f"Loading cached shard index from {index_path}...")
        with open(index_path, "r") as f:
            index_data = json.load(f)

        self.shard_files = index_data["shard_files"]
        self.shard_offsets = index_data["shard_offsets"]
        self.total_samples = index_data["total_samples"]

        # Validate that shard files still exist (spot check first and last)
        files_to_check = [self.shard_files[0], self.shard_files[-1]] if len(self.shard_files) > 1 else self.shard_files
        missing = [f for f in files_to_check if not os.path.exists(os.path.join(self.shard_dir, f))]
        if missing:
            print(f"Warning: shard files missing, rebuilding index...")
            self._build_and_cache_index(index_path)
            return

        print(f"Loaded index: {len(self.shard_files)} shards, {self.total_samples:,} samples")

        # Load speaker info for GRL training
        self._include_speaker_ids = index_data.get("include_speaker_ids", False)
        self._num_speakers = index_data.get("num_speakers", 0)
        self._unique_speaker_ids = index_data.get("unique_speaker_ids", [])

        if self._include_speaker_ids:
            print(f"  Speaker info: {self._num_speakers} unique speakers")

    def _build_and_cache_index(self, index_path: str):
        """Build shard index by scanning all shards, then cache to JSON."""
        # Find all shards
        self.shard_files = sorted([
            f for f in os.listdir(self.shard_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        ])

        if not self.shard_files:
            raise ValueError(f"No shard files found in {self.shard_dir}")

        # Build index
        self.shard_offsets = []
        self.total_samples = 0

        # Initialize speaker info (will be populated from shard_index.json if available)
        self._include_speaker_ids = False
        self._num_speakers = 0
        self._unique_speaker_ids = []

        print(f"Indexing {len(self.shard_files)} audio VAE shards (first time only, will be cached)...")
        for shard_file in tqdm(self.shard_files):
            self.shard_offsets.append(self.total_samples)
            shard_path = os.path.join(self.shard_dir, shard_file)
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
            self.total_samples += shard["num_samples"]

        print(f"Total samples: {self.total_samples:,}")

        # Cache the index for next time
        index_data = {
            "shard_files": self.shard_files,
            "shard_offsets": self.shard_offsets,
            "total_samples": self.total_samples,
            "dataset_type": "audio_vae",
        }
        try:
            with open(index_path, "w") as f:
                json.dump(index_data, f, indent=2)
            print(f"Cached shard index to {index_path}")
        except Exception as e:
            print(f"Warning: could not cache shard index: {e}")

    def _load_shard(self, shard_idx: int) -> Dict[str, torch.Tensor]:
        """Load shard with caching."""
        if shard_idx in self._cache:
            self._cache_order.remove(shard_idx)
            self._cache_order.append(shard_idx)
            return self._cache[shard_idx]

        shard_path = os.path.join(self.shard_dir, self.shard_files[shard_idx])
        shard = torch.load(shard_path, map_location="cpu", weights_only=True)

        self._cache[shard_idx] = shard
        self._cache_order.append(shard_idx)

        while len(self._cache) > self.cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        return shard

    def _find_shard(self, idx: int) -> tuple:
        """Find which shard contains the given index."""
        lo, hi = 0, len(self.shard_offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.shard_offsets[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        return lo, idx - self.shard_offsets[lo]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shard_idx, local_idx = self._find_shard(idx)
        shard = self._load_shard(shard_idx)

        # Return sample dict compatible with AudioVAEDataCollator
        sample = {
            "mel_spec": shard["mel_specs"][local_idx],  # [1, n_mels, T]
            "mel_spec_length": shard["mel_lengths"][local_idx].item(),
            "speaker_embedding": shard["speaker_embeddings"][local_idx],  # [embedding_dim]
        }

        # Include speaker_id if available in shard
        if "speaker_ids" in shard:
            sample["speaker_id"] = shard["speaker_ids"][local_idx].item()

        # Include text if available (for ASR evaluation)
        if "texts" in shard and shard["texts"]:
            sample["text"] = shard["texts"][local_idx]

        return sample

    @property
    def num_speakers(self) -> int:
        """Number of unique speakers in the dataset (for GRL training)."""
        return getattr(self, '_num_speakers', 0)

    @property
    def include_speaker_ids(self) -> bool:
        """Whether the dataset includes speaker IDs."""
        return getattr(self, '_include_speaker_ids', False)

    @property
    def unique_speaker_ids(self) -> list:
        """List of unique speaker IDs in the dataset."""
        return getattr(self, '_unique_speaker_ids', [])

    def get_sampler(self, shuffle: bool = True, seed: int = 42) -> ShardAwareSampler:
        """
        Get a shard-aware sampler for efficient training.

        This sampler groups indices by shard, ensuring each shard is loaded only
        once per epoch instead of thrashing between shards on every random access.

        Args:
            shuffle: Whether to shuffle shards and samples within shards
            seed: Random seed for reproducibility

        Returns:
            ShardAwareSampler configured for this dataset
        """
        return ShardAwareSampler(
            shard_offsets=self.shard_offsets,
            total_samples=self.total_samples,
            shuffle=shuffle,
            seed=seed,
        )


class AudioVAEDataCollator:
    """
    Data collator for audio VAE training.

    Pads to batch max length instead of global max for efficiency.
    Creates masks for both loss computation and attention masking.
    """
    def __init__(
        self,
        audio_max_frames: int = 1875,
        n_mels: int = 80,
        speaker_embedding_dim: int = 192,
    ):
        self.audio_max_frames = audio_max_frames
        self.n_mels = n_mels
        self.speaker_embedding_dim = speaker_embedding_dim

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        import torch.nn.functional as F

        # Filter None examples and collect lengths first
        valid_examples = [ex for ex in examples if ex is not None]
        if not valid_examples:
            raise ValueError("All examples in batch are None")

        # Collect mel specs and lengths
        raw_mel_specs = []
        mel_lengths = []
        speaker_embeddings = []

        for ex in valid_examples:
            mel = ex["mel_spec"]
            mel_length = ex.get("mel_spec_length", mel.shape[-1])

            # Ensure correct shape [1, n_mels, T] for single-channel input
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)  # [n_mels, T] -> [1, n_mels, T]
            elif mel.dim() == 3 and mel.shape[0] != 1:
                # If shape is [n_mels, T, 1] or similar, fix it
                if mel.shape[-1] == 1:
                    mel = mel.squeeze(-1).unsqueeze(0)

            # Clamp length to actual mel length and global max
            mel_length = min(mel_length, mel.shape[-1], self.audio_max_frames)

            raw_mel_specs.append(mel)
            mel_lengths.append(mel_length)

            # Get speaker embedding if available
            speaker_emb = ex.get("speaker_embedding", None)
            if speaker_emb is not None:
                # Ensure shape is [1, speaker_embedding_dim]
                if speaker_emb.dim() == 1:
                    speaker_emb = speaker_emb.unsqueeze(0)
                speaker_embeddings.append(speaker_emb)
            else:
                # Use zeros if no speaker embedding
                speaker_embeddings.append(torch.zeros(1, self.speaker_embedding_dim))

        # Compute batch max length (dynamic padding)
        batch_max_length = max(mel_lengths)

        # Pad/truncate to batch max length and create masks
        mel_specs = []
        mel_spec_masks = []

        for mel, mel_length in zip(raw_mel_specs, mel_lengths):
            # Create mel spec padding mask (1 = valid, 0 = padding)
            mel_mask = torch.zeros(batch_max_length, dtype=torch.float32)
            mel_mask[:mel_length] = 1.0

            # Truncate to batch max length (since data is pre-padded to global max)
            mel = mel[..., :batch_max_length]

            # Pad if needed (shouldn't be needed if data is pre-padded, but just in case)
            if mel.shape[-1] < batch_max_length:
                mel = F.pad(mel, (0, batch_max_length - mel.shape[-1]), value=0)

            mel_specs.append(mel)
            mel_spec_masks.append(mel_mask)

        # Convert lengths to tensor for attention masking
        mel_lengths_tensor = torch.tensor(mel_lengths, dtype=torch.long)

        batch = {
            "mel_spec": torch.stack(mel_specs),
            "mel_spec_mask": torch.stack(mel_spec_masks),  # [B, T] mask for loss (1=valid, 0=padding)
            "mel_spec_lengths": mel_lengths_tensor,  # [B] original lengths for attention masking
            "speaker_embedding": torch.stack(speaker_embeddings),  # [B, 1, speaker_embedding_dim]
        }

        # Include speaker_ids if available in examples
        if any("speaker_id" in ex for ex in valid_examples):
            speaker_ids = [ex.get("speaker_id", -1) for ex in valid_examples]
            batch["speaker_ids"] = speaker_ids  # List of ints (not tensor, for flexibility)

        return batch


class GuBERTFeatureShardedDataset(Dataset):
    """
    Efficient dataset for loading preprocessed GuBERT feature shards.

    Loads shards containing:
    - features: GuBERT encoder features
        - Single layer: [B, encoder_dim, T']
        - Multi-layer: [B, num_layers, encoder_dim, T']
    - feature_lengths: [B] feature lengths before padding
    - mel_lengths: [B] original mel spectrogram lengths (before GuBERT subsampling)
    - speaker_embeddings: [B, embedding_dim] speaker embeddings for conditioning

    For multi-layer features, the VAE can apply learned per-layer normalization
    (LayerNorm, RMSNorm, etc.) before processing.

    Use this dataset to train a VAE on GuBERT feature space.
    """

    SHARD_INDEX_FILE = "shard_index.json"

    def __init__(
        self,
        shard_dir: str,
        cache_size: int = 3,
        max_feature_frames: int = 500,
    ):
        """
        Args:
            shard_dir: Directory containing shard_*.pt files
            cache_size: Number of shards to keep in memory
            max_feature_frames: Maximum feature sequence length for padding
        """
        self.shard_dir = shard_dir
        self.cache_size = cache_size
        self.max_feature_frames = max_feature_frames

        # Try to load cached index first
        index_path = os.path.join(shard_dir, self.SHARD_INDEX_FILE)
        if os.path.exists(index_path):
            self._load_cached_index(index_path)
        else:
            self._build_and_cache_index(index_path)

        # Load config for metadata
        config_path = os.path.join(shard_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)
        else:
            self.config = {}

        self.encoder_dim = self.config.get("encoder_dim", 288)
        self.speaker_embedding_dim = self.config.get("speaker_embedding_dim", 192)
        self.num_layers = self.config.get("num_layers", 1)  # 1 for single layer, >1 for multi-layer
        self.layers = self.config.get("layers", None)  # List of layer indices if multi-layer

        # LRU cache
        self._cache = {}
        self._cache_order = []

    def _load_cached_index(self, index_path: str):
        """Load pre-computed shard index from JSON file."""
        print(f"Loading cached shard index from {index_path}...")
        with open(index_path, "r") as f:
            index_data = json.load(f)

        self.shard_files = index_data["shard_files"]
        self.shard_offsets = index_data["shard_offsets"]
        self.total_samples = index_data["total_samples"]

        print(f"Loaded index: {len(self.shard_files)} shards, {self.total_samples:,} samples")

    def _build_and_cache_index(self, index_path: str):
        """Build shard index by scanning all shards, then cache to JSON."""
        self.shard_files = sorted([
            f for f in os.listdir(self.shard_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        ])

        if not self.shard_files:
            raise ValueError(f"No shard files found in {self.shard_dir}")

        self.shard_offsets = []
        self.total_samples = 0

        print(f"Indexing {len(self.shard_files)} GuBERT feature shards (first time only, will be cached)...")
        for shard_file in tqdm(self.shard_files):
            self.shard_offsets.append(self.total_samples)
            shard_path = os.path.join(self.shard_dir, shard_file)
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
            num_samples = shard["features"].shape[0]
            self.total_samples += num_samples

        # Cache the index
        index_data = {
            "shard_files": self.shard_files,
            "shard_offsets": self.shard_offsets,
            "total_samples": self.total_samples,
        }
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)
        print(f"Cached shard index to {index_path}")

    def _load_shard(self, shard_idx: int):
        """Load a shard with LRU caching."""
        if shard_idx in self._cache:
            self._cache_order.remove(shard_idx)
            self._cache_order.append(shard_idx)
            return self._cache[shard_idx]

        shard_path = os.path.join(self.shard_dir, self.shard_files[shard_idx])
        shard = torch.load(shard_path, map_location="cpu", weights_only=True)

        if len(self._cache) >= self.cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        self._cache[shard_idx] = shard
        self._cache_order.append(shard_idx)
        return shard

    def _find_shard_for_idx(self, idx: int):
        """Binary search to find which shard contains the given global index."""
        import bisect
        shard_idx = bisect.bisect_right(self.shard_offsets, idx) - 1
        local_idx = idx - self.shard_offsets[shard_idx]
        return shard_idx, local_idx

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int):
        shard_idx, local_idx = self._find_shard_for_idx(idx)
        shard = self._load_shard(shard_idx)

        # features shape depends on multi-layer mode:
        # - Single layer: [encoder_dim, T']
        # - Multi-layer:  [num_layers, encoder_dim, T']
        features = shard["features"][local_idx]
        feature_length = shard["feature_lengths"][local_idx].item()
        mel_length = shard["mel_lengths"][local_idx].item()

        sample = {
            "features": features,
            "feature_length": feature_length,
            "mel_length": mel_length,
            "num_layers": self.num_layers,  # For collator to know the format
        }

        if "speaker_embeddings" in shard:
            sample["speaker_embedding"] = shard["speaker_embeddings"][local_idx]

        return sample

    def get_sampler(self, shuffle: bool = True, seed: int = 42) -> ShardAwareSampler:
        """Get a shard-aware sampler for efficient training."""
        return ShardAwareSampler(
            shard_offsets=self.shard_offsets,
            total_samples=self.total_samples,
            shuffle=shuffle,
            seed=seed,
        )


class GuBERTFeatureDataCollator:
    """
    Data collator for GuBERT feature VAE training.

    Handles both single-layer and multi-layer feature formats:
    - Single layer: [encoder_dim, T'] -> batch: [B, encoder_dim, T']
    - Multi-layer: [num_layers, encoder_dim, T'] -> batch: [B, num_layers, encoder_dim, T']

    Pads features to same length within batch and creates masks.
    """

    def __init__(
        self,
        max_feature_frames: int = 500,
        speaker_embedding_dim: int = 192,
    ):
        self.max_feature_frames = max_feature_frames
        self.speaker_embedding_dim = speaker_embedding_dim

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        # Filter out any None examples
        valid_examples = [ex for ex in examples if ex is not None]
        if not valid_examples:
            return {}

        raw_features = []
        feature_lengths = []
        speaker_embeddings = []

        # Detect multi-layer mode from first example
        first_features = valid_examples[0]["features"]
        num_layers = valid_examples[0].get("num_layers", 1)
        is_multi_layer = num_layers > 1 or first_features.dim() == 3

        for ex in valid_examples:
            # Shape depends on mode:
            # - Single layer: [encoder_dim, T']
            # - Multi-layer:  [num_layers, encoder_dim, T']
            features = ex["features"]
            feature_length = ex.get("feature_length", features.shape[-1])

            # Clamp length
            feature_length = min(feature_length, features.shape[-1], self.max_feature_frames)

            raw_features.append(features)
            feature_lengths.append(feature_length)

            speaker_emb = ex.get("speaker_embedding", None)
            if speaker_emb is not None:
                if speaker_emb.dim() == 1:
                    speaker_emb = speaker_emb.unsqueeze(0)
                speaker_embeddings.append(speaker_emb)
            else:
                speaker_embeddings.append(torch.zeros(1, self.speaker_embedding_dim))

        # Compute batch max length (dynamic padding)
        batch_max_length = max(feature_lengths)

        # Pad/truncate to batch max length
        padded_features = []
        feature_masks = []

        for feat, feat_length in zip(raw_features, feature_lengths):
            # Create mask (1 = valid, 0 = padding)
            feat_mask = torch.zeros(batch_max_length, dtype=torch.float32)
            feat_mask[:feat_length] = 1.0

            # Truncate to batch max length (along last dimension for both formats)
            feat = feat[..., :batch_max_length]

            # Pad if needed (along last dimension)
            if feat.shape[-1] < batch_max_length:
                feat = F.pad(feat, (0, batch_max_length - feat.shape[-1]), value=0)

            padded_features.append(feat)
            feature_masks.append(feat_mask)

        # Stack features:
        # - Single layer: [B, encoder_dim, T']
        # - Multi-layer:  [B, num_layers, encoder_dim, T']
        batch = {
            "features": torch.stack(padded_features),
            "feature_mask": torch.stack(feature_masks),  # [B, T'] mask (1=valid, 0=padding)
            "feature_lengths": torch.tensor(feature_lengths, dtype=torch.long),  # [B]
            "speaker_embedding": torch.stack(speaker_embeddings),  # [B, 1, speaker_embedding_dim]
            "num_layers": num_layers,  # For VAE to know the format
        }

        return batch


class AudioDiffusionShardedDataset(Dataset):
    """
    Efficient dataset for loading preprocessed audio diffusion shards.

    Loads shards containing:
    - text_embeddings: [B, T_text, context_dim] T5 text embeddings
    - text_attention_masks: [B, T_text] attention masks
    - mel_lengths: [B] original mel spectrogram lengths
    - speaker_embeddings: [B, embedding_dim] speaker embeddings
    - latent_mus: [B, C, H, T'] VAE-encoded latents (if available)
    - mel_specs: [B, n_mels, T] mel spectrograms (optional, if include_mel_specs was enabled)

    Features:
    - Lazy loading of shards
    - LRU cache for recently accessed shards

    IMPORTANT: Use get_sampler() to get a ShardAwareSampler for efficient training.
    Random access without shard-aware sampling will be extremely slow.
    """

    def __init__(
        self,
        shard_dir: str,
        cache_size: int = 3,
        audio_max_frames: int = 1875,
        latent_max_frames: int = 75,  # audio_max_frames / time_compression
    ):
        """
        Args:
            shard_dir: Directory containing shard_*.pt files
            cache_size: Number of shards to keep in memory
            audio_max_frames: Maximum mel spectrogram frames for padding
            latent_max_frames: Maximum latent frames for padding
        """
        self.shard_dir = shard_dir
        self.cache_size = cache_size
        self.audio_max_frames = audio_max_frames
        self.latent_max_frames = latent_max_frames

        # Find all shards
        self.shard_files = sorted([
            f for f in os.listdir(shard_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        ])

        if not self.shard_files:
            raise ValueError(f"No shard files found in {shard_dir}")

        # Build index
        self.shard_offsets = []
        self.total_samples = 0
        self.has_latents = None
        self.has_mel_specs = None

        print(f"Indexing {len(self.shard_files)} audio diffusion shards...")
        for shard_file in tqdm(self.shard_files):
            self.shard_offsets.append(self.total_samples)
            shard_path = os.path.join(shard_dir, shard_file)
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
            self.total_samples += shard["num_samples"]

            # Check for optional fields
            if self.has_latents is None:
                self.has_latents = "latent_mus" in shard
            if self.has_mel_specs is None:
                self.has_mel_specs = "mel_specs" in shard

        print(f"Total samples: {self.total_samples:,}")
        print(f"Has latents: {self.has_latents}")
        print(f"Has mel specs: {self.has_mel_specs}")

        # LRU cache
        self._cache = {}
        self._cache_order = []

    def _load_shard(self, shard_idx: int) -> Dict[str, torch.Tensor]:
        """Load shard with caching."""
        if shard_idx in self._cache:
            self._cache_order.remove(shard_idx)
            self._cache_order.append(shard_idx)
            return self._cache[shard_idx]

        shard_path = os.path.join(self.shard_dir, self.shard_files[shard_idx])
        shard = torch.load(shard_path, map_location="cpu", weights_only=True)

        self._cache[shard_idx] = shard
        self._cache_order.append(shard_idx)

        while len(self._cache) > self.cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        return shard

    def _find_shard(self, idx: int) -> tuple:
        """Find which shard contains the given index."""
        lo, hi = 0, len(self.shard_offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.shard_offsets[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        return lo, idx - self.shard_offsets[lo]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shard_idx, local_idx = self._find_shard(idx)
        shard = self._load_shard(shard_idx)

        # Return sample dict compatible with AudioDiffusionDataCollator
        sample = {
            "text_embeddings": shard["text_embeddings"][local_idx],  # [T_text, context_dim]
            "text_attention_mask": shard["text_attention_masks"][local_idx],  # [T_text]
            "mel_spec_length": shard["mel_lengths"][local_idx].item(),
            "speaker_embedding": shard["speaker_embeddings"][local_idx],  # [embedding_dim]
        }

        # Add latents if available
        if self.has_latents and "latent_mus" in shard:
            sample["latent_mu"] = shard["latent_mus"][local_idx]  # [C, H, T']
            sample["latent_shape"] = list(shard["latent_mus"][local_idx].shape)

        # Add mel specs if available (optional, mainly for debugging)
        if self.has_mel_specs and "mel_specs" in shard:
            sample["mel_spec"] = shard["mel_specs"][local_idx]  # [n_mels, T]

        return sample

    def get_sampler(self, shuffle: bool = True, seed: int = 42) -> ShardAwareSampler:
        """
        Get a shard-aware sampler for efficient training.

        This sampler groups indices by shard, ensuring each shard is loaded only
        once per epoch instead of thrashing between shards on every random access.

        Args:
            shuffle: Whether to shuffle shards and samples within shards
            seed: Random seed for reproducibility

        Returns:
            ShardAwareSampler configured for this dataset
        """
        return ShardAwareSampler(
            shard_offsets=self.shard_offsets,
            total_samples=self.total_samples,
            shuffle=shuffle,
            seed=seed,
        )


def merge_audio_diffusion_shards(
    input_dir: str,
    output_dir: str,
    shuffle: bool = True,
    target_shard_size: int = 2000,
):
    """
    Merge audio diffusion shards from multiple GPU subdirectories.

    Args:
        input_dir: Directory containing gpu_0/, gpu_1/, etc. subdirs
        output_dir: Output directory for merged shards
        shuffle: Whether to shuffle samples across shards
        target_shard_size: Target samples per output shard
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all shard files
    all_shards = []
    gpu_dirs = sorted([
        d for d in os.listdir(input_dir)
        if d.startswith("gpu_") and os.path.isdir(os.path.join(input_dir, d))
    ])

    if not gpu_dirs:
        gpu_dirs = ["."]

    print(f"Found GPU directories: {gpu_dirs}")

    for gpu_dir in gpu_dirs:
        shard_dir = os.path.join(input_dir, gpu_dir)
        shards = sorted([
            os.path.join(shard_dir, f)
            for f in os.listdir(shard_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        ])
        all_shards.extend(shards)
        print(f"  {gpu_dir}: {len(shards)} shards")

    print(f"Total shards to merge: {len(all_shards)}")

    if shuffle:
        random.shuffle(all_shards)

    # Accumulators
    acc_text_emb = []
    acc_text_mask = []
    acc_mel_lengths = []
    acc_speaker_emb = []
    acc_latent_mus = []
    acc_mel_specs = []
    has_latents = None
    has_mel_specs = None

    output_shard_idx = 0
    total_samples = 0

    def save_merged_shard():
        nonlocal acc_text_emb, acc_text_mask, acc_mel_lengths, acc_speaker_emb
        nonlocal acc_latent_mus, acc_mel_specs, output_shard_idx

        if not acc_text_emb:
            return

        shard_data = {
            "text_embeddings": torch.cat(acc_text_emb, dim=0),
            "text_attention_masks": torch.cat(acc_text_mask, dim=0),
            "mel_lengths": torch.cat(acc_mel_lengths, dim=0),
            "speaker_embeddings": torch.cat(acc_speaker_emb, dim=0),
            "num_samples": sum(x.shape[0] for x in acc_text_emb),
        }

        if acc_latent_mus:
            shard_data["latent_mus"] = torch.cat(acc_latent_mus, dim=0)

        if acc_mel_specs:
            shard_data["mel_specs"] = torch.cat(acc_mel_specs, dim=0)

        output_path = os.path.join(output_dir, f"shard_{output_shard_idx:06d}.pt")
        torch.save(shard_data, output_path)

        output_shard_idx += 1
        acc_text_emb = []
        acc_text_mask = []
        acc_mel_lengths = []
        acc_speaker_emb = []
        acc_latent_mus = []
        acc_mel_specs = []

    for shard_path in tqdm(all_shards, desc="Merging audio diffusion shards"):
        try:
            shard = torch.load(shard_path, weights_only=True)

            # Check for optional fields
            if has_latents is None:
                has_latents = "latent_mus" in shard
            if has_mel_specs is None:
                has_mel_specs = "mel_specs" in shard

            acc_text_emb.append(shard["text_embeddings"])
            acc_text_mask.append(shard["text_attention_masks"])
            acc_mel_lengths.append(shard["mel_lengths"])
            acc_speaker_emb.append(shard["speaker_embeddings"])

            if has_latents and "latent_mus" in shard:
                acc_latent_mus.append(shard["latent_mus"])

            if has_mel_specs and "mel_specs" in shard:
                acc_mel_specs.append(shard["mel_specs"])

            total_samples += shard["num_samples"]

            current_size = sum(x.shape[0] for x in acc_text_emb)
            if current_size >= target_shard_size:
                save_merged_shard()

        except Exception as e:
            print(f"Error loading {shard_path}: {e}")
            continue

    save_merged_shard()

    # Save metadata
    meta = {
        "total_samples": total_samples,
        "num_shards": output_shard_idx,
        "shard_size": target_shard_size,
        "has_latents": has_latents,
        "has_mel_specs": has_mel_specs,
        "source_dirs": gpu_dirs,
        "dataset_type": "audio_diffusion",
    }

    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nMerge complete!")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Output shards: {output_shard_idx}")
    print(f"  Has latents: {has_latents}")
    print(f"  Has mel specs: {has_mel_specs}")
    print(f"  Output dir: {output_dir}")


class VocoderShardedDataset(Dataset):
    """
    Efficient dataset for loading preprocessed vocoder shards.

    Loads shards containing:
    - mel_specs: [B, n_mels, T] mel spectrograms (input)
    - waveform_labels: [B, T_audio] waveform targets
    - target_complex_stfts: [B, n_fft//2+1, T_stft] complex STFT targets
    - speaker_ids: [B] speaker IDs
    - mel_lengths: [B] original mel lengths
    - texts: List[str] text transcriptions (optional, for ASR evaluation)

    Features:
    - Lazy loading of shards
    - LRU cache for recently accessed shards
    - Cached shard index for fast startup (saves ~14 min on large datasets)

    IMPORTANT: Use get_sampler() to get a ShardAwareSampler for efficient training.
    Random access without shard-aware sampling will be extremely slow.
    """

    SHARD_INDEX_FILE = "shard_index.json"

    def __init__(
        self,
        shard_dir: str,
        cache_size: int = 3,
        audio_max_frames: int = 626,
        preload_shards: int = 0,
    ):
        """
        Args:
            shard_dir: Directory containing shard_*.pt files
            cache_size: Number of shards to keep in memory
            audio_max_frames: Maximum mel spectrogram frames for reference
            preload_shards: Number of shards to preload at initialization.
                           Use -1 to preload all shards. Default 0 (no preloading).
        """
        self.shard_dir = shard_dir
        self.cache_size = cache_size
        self.audio_max_frames = audio_max_frames

        # Try to load cached index first (saves ~14 min on large datasets)
        index_path = os.path.join(shard_dir, self.SHARD_INDEX_FILE)
        if os.path.exists(index_path):
            self._load_cached_index(index_path)
        else:
            self._build_and_cache_index(index_path)

        # LRU cache
        self._cache = {}
        self._cache_order = []

        # Preload shards if requested
        if preload_shards != 0:
            n_to_preload = len(self.shard_files) if preload_shards == -1 else min(preload_shards, len(self.shard_files))
            # Increase cache size to hold all preloaded shards
            self.cache_size = max(self.cache_size, n_to_preload)

            print(f"Preloading {n_to_preload} shards into memory...")
            for shard_idx in tqdm(range(n_to_preload)):
                self._load_shard(shard_idx)

            # Estimate memory usage
            if self._cache:
                sample_shard = next(iter(self._cache.values()))
                shard_size_mb = sum(
                    t.element_size() * t.numel() / (1024 * 1024)
                    for t in sample_shard.values() if isinstance(t, torch.Tensor)
                )
                total_mb = shard_size_mb * len(self._cache)
                print(f"Preloaded {len(self._cache)} shards (~{total_mb:.1f}MB total, ~{shard_size_mb:.1f}MB per shard)")

    def _load_cached_index(self, index_path: str):
        """Load pre-computed shard index from JSON file."""
        print(f"Loading cached shard index from {index_path}...")
        with open(index_path, "r") as f:
            index_data = json.load(f)

        self.shard_files = index_data["shard_files"]
        self.shard_offsets = index_data["shard_offsets"]
        self.total_samples = index_data["total_samples"]

        # Validate that shard files still exist (spot check first and last)
        files_to_check = [self.shard_files[0], self.shard_files[-1]] if len(self.shard_files) > 1 else self.shard_files
        missing = [f for f in files_to_check if not os.path.exists(os.path.join(self.shard_dir, f))]
        if missing:
            print(f"Warning: shard files missing, rebuilding index...")
            self._build_and_cache_index(index_path)
            return

        print(f"Loaded index: {len(self.shard_files)} shards, {self.total_samples:,} samples")

    def _build_and_cache_index(self, index_path: str):
        """Build shard index by scanning all shards, then cache to JSON."""
        # Find all shards
        self.shard_files = sorted([
            f for f in os.listdir(self.shard_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        ])

        if not self.shard_files:
            raise ValueError(f"No shard files found in {self.shard_dir}")

        # Build index
        self.shard_offsets = []
        self.total_samples = 0

        print(f"Indexing {len(self.shard_files)} vocoder shards (first time only, will be cached)...")
        for shard_file in tqdm(self.shard_files):
            self.shard_offsets.append(self.total_samples)
            shard_path = os.path.join(self.shard_dir, shard_file)
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
            self.total_samples += shard["num_samples"]

        print(f"Total samples: {self.total_samples:,}")

        # Cache the index for next time
        index_data = {
            "shard_files": self.shard_files,
            "shard_offsets": self.shard_offsets,
            "total_samples": self.total_samples,
            "dataset_type": "vocoder",
        }
        try:
            with open(index_path, "w") as f:
                json.dump(index_data, f, indent=2)
            print(f"Cached shard index to {index_path}")
        except Exception as e:
            print(f"Warning: could not cache shard index: {e}")

    def _load_shard(self, shard_idx: int) -> Dict[str, torch.Tensor]:
        """Load shard with caching."""
        if shard_idx in self._cache:
            self._cache_order.remove(shard_idx)
            self._cache_order.append(shard_idx)
            if _shard_debug_enabled():
                print(f"[VocoderShardedDataset] Cache HIT shard {shard_idx}")
            return self._cache[shard_idx]

        shard_path = os.path.join(self.shard_dir, self.shard_files[shard_idx])

        import time
        start = time.time()
        shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        elapsed = time.time() - start

        # Always log slow loads, optionally log all loads
        file_size_mb = os.path.getsize(shard_path) / (1024 * 1024)
        if elapsed > 1.0 and _shard_debug_enabled():
            rank_str = ""
            if torch.distributed.is_initialized():
                rank_str = f"Rank {torch.distributed.get_rank()}: "
            print(f"[VocoderShardedDataset] {rank_str}Cache MISS shard {shard_idx} "
                  f"({file_size_mb:.1f}MB) loaded in {elapsed:.2f}s. "
                  f"Cache: {len(self._cache)}/{self.cache_size}")

        self._cache[shard_idx] = shard
        self._cache_order.append(shard_idx)

        evicted = []
        while len(self._cache) > self.cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
            evicted.append(oldest)

        if evicted and _shard_debug_enabled():
            print(f"[VocoderShardedDataset] Evicted shards: {evicted}")

        return shard

    def _find_shard(self, idx: int) -> tuple:
        """Find which shard contains the given index."""
        lo, hi = 0, len(self.shard_offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.shard_offsets[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        return lo, idx - self.shard_offsets[lo]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shard_idx, local_idx = self._find_shard(idx)
        shard = self._load_shard(shard_idx)

        # Return sample dict compatible with VocoderDataCollator
        sample = {
            "mel_spec": shard["mel_specs"][local_idx],  # [n_mels, T]
            "waveform_labels": shard["waveform_labels"][local_idx],  # [T_audio]
            "target_complex_stfts": shard["target_complex_stfts"][local_idx],  # [n_fft//2+1, T_stft]
            "speaker_id": shard["speaker_ids"][local_idx].item(),
            "mel_spec_length": shard["mel_lengths"][local_idx].item(),
        }

        # Include text if available (for ASR evaluation)
        if "texts" in shard and shard["texts"]:
            sample["text"] = shard["texts"][local_idx]

        return sample

    def get_sampler(self, shuffle: bool = True, seed: int = 42) -> ShardAwareSampler:
        """
        Get a shard-aware sampler for efficient training.

        This sampler groups indices by shard, ensuring each shard is loaded only
        once per epoch instead of thrashing between shards on every random access.

        Args:
            shuffle: Whether to shuffle shards and samples within shards
            seed: Random seed for reproducibility

        Returns:
            ShardAwareSampler configured for this dataset
        """
        return ShardAwareSampler(
            shard_offsets=self.shard_offsets,
            total_samples=self.total_samples,
            shuffle=shuffle,
            seed=seed,
        )


def merge_vocoder_shards(
    input_dir: str,
    output_dir: str,
    shuffle: bool = True,
    target_shard_size: int = 2000,
):
    """
    Merge vocoder shards from multiple GPU subdirectories.

    Args:
        input_dir: Directory containing gpu_0/, gpu_1/, etc. subdirs
        output_dir: Output directory for merged shards
        shuffle: Whether to shuffle samples across shards
        target_shard_size: Target samples per output shard
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all shard files
    all_shards = []
    gpu_dirs = sorted([
        d for d in os.listdir(input_dir)
        if d.startswith("gpu_") and os.path.isdir(os.path.join(input_dir, d))
    ])

    if not gpu_dirs:
        gpu_dirs = ["."]

    print(f"Found GPU directories: {gpu_dirs}")

    for gpu_dir in gpu_dirs:
        shard_dir = os.path.join(input_dir, gpu_dir)
        shards = sorted([
            os.path.join(shard_dir, f)
            for f in os.listdir(shard_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        ])
        all_shards.extend(shards)
        print(f"  {gpu_dir}: {len(shards)} shards")

    print(f"Total shards to merge: {len(all_shards)}")

    if shuffle:
        random.shuffle(all_shards)

    # Accumulators
    acc_mel_specs = []
    acc_waveform_labels = []
    acc_stfts = []
    acc_speaker_ids = []
    acc_mel_lengths = []
    acc_texts = []

    output_shard_idx = 0
    total_samples = 0

    # Track shard info for index
    shard_files = []
    shard_offsets = []

    def save_merged_shard():
        nonlocal acc_mel_specs, acc_waveform_labels, acc_stfts
        nonlocal acc_speaker_ids, acc_mel_lengths, acc_texts, output_shard_idx, total_samples

        if not acc_mel_specs:
            return

        num_samples_in_shard = sum(x.shape[0] for x in acc_mel_specs)

        shard_data = {
            "mel_specs": torch.cat(acc_mel_specs, dim=0),
            "waveform_labels": torch.cat(acc_waveform_labels, dim=0),
            "target_complex_stfts": torch.cat(acc_stfts, dim=0),
            "speaker_ids": torch.cat(acc_speaker_ids, dim=0),
            "mel_lengths": torch.cat(acc_mel_lengths, dim=0),
            "texts": acc_texts,  # List of strings
            "num_samples": num_samples_in_shard,
        }

        shard_filename = f"shard_{output_shard_idx:06d}.pt"
        output_path = os.path.join(output_dir, shard_filename)
        torch.save(shard_data, output_path)

        # Track for index
        shard_files.append(shard_filename)
        shard_offsets.append(total_samples)
        total_samples += num_samples_in_shard

        output_shard_idx += 1
        acc_mel_specs = []
        acc_waveform_labels = []
        acc_stfts = []
        acc_speaker_ids = []
        acc_mel_lengths = []
        acc_texts = []

    for shard_path in tqdm(all_shards, desc="Merging vocoder shards"):
        try:
            shard = torch.load(shard_path, weights_only=True)

            acc_mel_specs.append(shard["mel_specs"])
            acc_waveform_labels.append(shard["waveform_labels"])
            acc_stfts.append(shard["target_complex_stfts"])
            acc_speaker_ids.append(shard["speaker_ids"])
            acc_mel_lengths.append(shard["mel_lengths"])

            # Handle texts (may not be present in older shards)
            if "texts" in shard and shard["texts"]:
                acc_texts.extend(shard["texts"])
            else:
                # Fill with empty strings if no texts
                acc_texts.extend([""] * shard["num_samples"])

            current_size = sum(x.shape[0] for x in acc_mel_specs)
            if current_size >= target_shard_size:
                save_merged_shard()

        except Exception as e:
            print(f"Error loading {shard_path}: {e}")
            continue

    save_merged_shard()

    # Save metadata
    meta = {
        "total_samples": total_samples,
        "num_shards": output_shard_idx,
        "shard_size": target_shard_size,
        "source_dirs": gpu_dirs,
        "dataset_type": "vocoder",
    }

    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save shard index for fast dataset loading
    shard_index = {
        "shard_files": shard_files,
        "shard_offsets": shard_offsets,
        "total_samples": total_samples,
        "dataset_type": "vocoder",
    }

    with open(os.path.join(output_dir, "shard_index.json"), "w") as f:
        json.dump(shard_index, f, indent=2)

    print(f"\nMerge complete!")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Output shards: {output_shard_idx}")
    print(f"  Output dir: {output_dir}")
    print(f"  Shard index saved (skips 14min indexing on load)")


def merge_audio_vae_shards(
    input_dir: str,
    output_dir: str,
    shuffle: bool = True,
    target_shard_size: int = 5000,
):
    """
    Merge audio VAE shards from multiple GPU subdirectories.

    Args:
        input_dir: Directory containing gpu_0/, gpu_1/, etc. subdirs
        output_dir: Output directory for merged shards
        shuffle: Whether to shuffle samples across shards
        target_shard_size: Target samples per output shard
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all shard files
    all_shards = []
    gpu_dirs = sorted([
        d for d in os.listdir(input_dir)
        if d.startswith("gpu_") and os.path.isdir(os.path.join(input_dir, d))
    ])

    if not gpu_dirs:
        gpu_dirs = ["."]

    print(f"Found GPU directories: {gpu_dirs}")

    # Collect unique speaker IDs from all GPU config.json files
    all_unique_speaker_ids = set()
    include_speaker_ids = False

    for gpu_dir in gpu_dirs:
        shard_dir = os.path.join(input_dir, gpu_dir)
        shards = sorted([
            os.path.join(shard_dir, f)
            for f in os.listdir(shard_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        ])
        all_shards.extend(shards)
        print(f"  {gpu_dir}: {len(shards)} shards")

        # Load config.json to get unique speaker IDs from this GPU
        config_path = os.path.join(shard_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                if config.get("include_speaker_id", False):
                    include_speaker_ids = True
                    gpu_speaker_ids = config.get("unique_speaker_ids", [])
                    all_unique_speaker_ids.update(gpu_speaker_ids)
                    print(f"    Unique speakers from this GPU: {len(gpu_speaker_ids)}")

    print(f"Total shards to merge: {len(all_shards)}")
    if include_speaker_ids:
        print(f"Total unique speakers across all GPUs: {len(all_unique_speaker_ids)}")

    # Create mapping from original speaker_id -> sequential class label
    # Reserve 0 for unknown/missing speaker, so labels are 1, 2, 3, ...
    sorted_speaker_ids = sorted(all_unique_speaker_ids)
    speaker_id_to_label = {sid: idx + 1 for idx, sid in enumerate(sorted_speaker_ids)}
    # Add 0 for unknown (maps to itself)
    speaker_id_to_label[0] = 0
    speaker_id_to_label[-1] = 0  # Also map -1 to unknown
    if include_speaker_ids:
        print(f"Created speaker_id -> class_label mapping (0=unknown, 1-{len(sorted_speaker_ids)} for known speakers)")

    if shuffle:
        random.shuffle(all_shards)

    # Accumulators
    acc_mel_specs = []
    acc_mel_lengths = []
    acc_speaker_emb = []
    acc_speaker_ids = []
    acc_texts = []

    output_shard_idx = 0
    total_samples = 0

    # Track shard info for index
    shard_files = []
    shard_offsets = []

    def save_merged_shard():
        nonlocal acc_mel_specs, acc_mel_lengths, acc_speaker_emb, acc_speaker_ids, acc_texts, output_shard_idx, total_samples

        if not acc_mel_specs:
            return

        num_samples_in_shard = sum(x.shape[0] for x in acc_mel_specs)

        shard_data = {
            "mel_specs": torch.cat(acc_mel_specs, dim=0),
            "mel_lengths": torch.cat(acc_mel_lengths, dim=0),
            "speaker_embeddings": torch.cat(acc_speaker_emb, dim=0),
            "texts": acc_texts,  # List of strings
            "num_samples": num_samples_in_shard,
        }

        # Include speaker_ids if available
        if acc_speaker_ids and all(s is not None for s in acc_speaker_ids):
            shard_data["speaker_ids"] = torch.cat(acc_speaker_ids, dim=0)

        shard_filename = f"shard_{output_shard_idx:06d}.pt"
        output_path = os.path.join(output_dir, shard_filename)
        torch.save(shard_data, output_path)

        # Track for index
        shard_files.append(shard_filename)
        shard_offsets.append(total_samples)
        total_samples += num_samples_in_shard

        output_shard_idx += 1
        acc_mel_specs = []
        acc_mel_lengths = []
        acc_speaker_emb = []
        acc_speaker_ids = []
        acc_texts = []

    for shard_path in tqdm(all_shards, desc="Merging audio VAE shards"):
        try:
            shard = torch.load(shard_path, weights_only=True)

            acc_mel_specs.append(shard["mel_specs"])
            acc_mel_lengths.append(shard["mel_lengths"])
            acc_speaker_emb.append(shard["speaker_embeddings"])

            # Include speaker_ids if available in shard - convert to sequential class labels
            if include_speaker_ids and "speaker_ids" in shard and shard["speaker_ids"] is not None:
                original_ids = shard["speaker_ids"]
                # Convert original speaker IDs to sequential class labels using the mapping
                class_labels = torch.tensor(
                    [speaker_id_to_label.get(sid.item(), 0) for sid in original_ids],
                    dtype=torch.long
                )
                acc_speaker_ids.append(class_labels)

            # Handle texts (may not be present in older shards)
            if "texts" in shard and shard["texts"]:
                acc_texts.extend(shard["texts"])
            else:
                # Fill with empty strings if no texts
                acc_texts.extend([""] * shard["num_samples"])

            current_size = sum(x.shape[0] for x in acc_mel_specs)
            if current_size >= target_shard_size:
                save_merged_shard()

        except Exception as e:
            print(f"Error loading {shard_path}: {e}")
            continue

    save_merged_shard()

    # Save shard index for fast dataset loading
    # num_speakers is len + 1 to account for class 0 (unknown)
    num_speaker_classes = len(all_unique_speaker_ids) + 1 if include_speaker_ids else 0
    shard_index = {
        "shard_files": shard_files,
        "shard_offsets": shard_offsets,
        "total_samples": total_samples,
        "dataset_type": "audio_vae",
        # Include speaker info for GRL training
        "include_speaker_ids": include_speaker_ids,
        # num_speakers = number of classes for cross_entropy (0=unknown, 1..N=known speakers)
        "num_speakers": num_speaker_classes,
        # Original unique speaker IDs (for reference only, not used in training)
        "unique_speaker_ids": sorted(list(all_unique_speaker_ids)) if include_speaker_ids else [],
        # Mapping from original speaker ID to class label (for debugging/reference)
        "speaker_id_to_label": speaker_id_to_label if include_speaker_ids else {},
    }

    with open(os.path.join(output_dir, "shard_index.json"), "w") as f:
        json.dump(shard_index, f, indent=2)

    print(f"\nMerge complete!")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Output shards: {output_shard_idx}")
    print(f"  Output dir: {output_dir}")
    print(f"  Shard index saved (skips indexing on load)")
    if include_speaker_ids:
        print(f"  Total unique speakers: {len(all_unique_speaker_ids)}")
        print(f"  Speaker classes for GRL: {num_speaker_classes} (0=unknown, 1-{len(all_unique_speaker_ids)}=known)")


class ShuffledShardedDataset(Dataset):
    """
    Wraps ShardedDataset with better shuffling.
    
    Instead of random access (which causes cache thrashing),
    this loads shards sequentially but shuffles within each shard.
    """
    
    def __init__(
        self,
        shard_dir: str,
        use_latents: bool = True,
        shuffle_shards: bool = True,
        seed: int = 42,
    ):
        self.shard_dir = shard_dir
        self.use_latents = use_latents
        
        # Find shards
        self.shard_files = sorted([
            f for f in os.listdir(shard_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        ])
        
        if shuffle_shards:
            rng = random.Random(seed)
            rng.shuffle(self.shard_files)
        
        # Count total
        self.total_samples = 0
        for shard_file in self.shard_files:
            shard = torch.load(os.path.join(shard_dir, shard_file))
            self.total_samples += shard["num_samples"]
        
        # Current state
        self._current_shard_idx = 0
        self._current_shard = None
        self._current_indices = None
        self._position_in_shard = 0
        self._global_position = 0
        
        self._load_next_shard()
    
    def _load_next_shard(self):
        """Load the next shard and shuffle its indices."""
        if self._current_shard_idx >= len(self.shard_files):
            return False
        
        shard_path = os.path.join(self.shard_dir, self.shard_files[self._current_shard_idx])
        self._current_shard = torch.load(shard_path)
        
        # Shuffle indices within shard
        n = self._current_shard["num_samples"]
        self._current_indices = list(range(n))
        random.shuffle(self._current_indices)
        self._position_in_shard = 0
        
        self._current_shard_idx += 1
        return True
    
    def __len__(self):
        return self.total_samples
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Dict[str, torch.Tensor]:
        # Check if need next shard
        if self._position_in_shard >= len(self._current_indices):
            if not self._load_next_shard():
                raise StopIteration
        
        # Get sample
        local_idx = self._current_indices[self._position_in_shard]
        
        sample = {
            "text_embeddings": self._current_shard["text_embeddings"][local_idx],
            "text_attention_mask": self._current_shard["text_attention_mask"][local_idx],
        }
        
        if self.use_latents and "latent_mu" in self._current_shard:
            sample["latent"] = self._current_shard["latent_mu"][local_idx]
        
        self._position_in_shard += 1
        self._global_position += 1
        
        return sample


# ============================================================================
# World Model Dataset
# ============================================================================

class WorldModelShardedDataset(Dataset):
    """
    Efficient dataset for loading preprocessed world model shards.

    World model shards contain samples with variable modalities:
    - text_input_ids: Tokenized text with special tokens (always present)
    - audio_mel_spec_latents: Audio VAE latents (optional)
    - voice_mel_spec_latents: Voice VAE latents (optional)
    - image_latents: Image VAE latents (optional)
    - task_type: String describing the task

    Features:
    - Lazy loading of shards
    - LRU cache for recently accessed shards
    - Cached shard index for fast startup

    IMPORTANT: Use get_sampler() to get a ShardAwareSampler for efficient training.
    """

    SHARD_INDEX_FILE = "shard_index.json"

    def __init__(
        self,
        shard_dir: str,
        cache_size: int = 3,
    ):
        """
        Args:
            shard_dir: Directory containing shard_*.pt files
            cache_size: Number of shards to keep in memory
        """
        self.shard_dir = shard_dir
        self.cache_size = cache_size

        # Try to load cached index first
        index_path = os.path.join(shard_dir, self.SHARD_INDEX_FILE)
        if os.path.exists(index_path):
            self._load_cached_index(index_path)
        else:
            self._build_and_cache_index(index_path)

        # LRU cache
        self._cache = {}
        self._cache_order = []

    def _load_cached_index(self, index_path: str):
        """Load pre-computed shard index from JSON file."""
        print(f"Loading cached shard index from {index_path}...")
        with open(index_path, "r") as f:
            index_data = json.load(f)

        self.shard_files = index_data["shard_files"]
        self.shard_offsets = index_data["shard_offsets"]
        self.total_samples = index_data["total_samples"]

        # Validate shard files exist
        files_to_check = [self.shard_files[0], self.shard_files[-1]] if len(self.shard_files) > 1 else self.shard_files
        missing = [f for f in files_to_check if not os.path.exists(os.path.join(self.shard_dir, f))]
        if missing:
            print(f"Warning: shard files missing, rebuilding index...")
            self._build_and_cache_index(index_path)
            return

        print(f"Loaded index: {len(self.shard_files)} shards, {self.total_samples:,} samples")

    def _build_and_cache_index(self, index_path: str):
        """Build shard index by scanning all shards, then cache to JSON."""
        self.shard_files = sorted([
            f for f in os.listdir(self.shard_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        ])

        if not self.shard_files:
            raise ValueError(f"No shard files found in {self.shard_dir}")

        self.shard_offsets = []
        self.total_samples = 0

        print(f"Indexing {len(self.shard_files)} world model shards...")
        for shard_file in tqdm(self.shard_files):
            self.shard_offsets.append(self.total_samples)
            shard_path = os.path.join(self.shard_dir, shard_file)
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            self.total_samples += shard["num_samples"]

        print(f"Total samples: {self.total_samples:,}")

        # Cache the index
        index_data = {
            "shard_files": self.shard_files,
            "shard_offsets": self.shard_offsets,
            "total_samples": self.total_samples,
            "dataset_type": "world_model",
        }
        try:
            with open(index_path, "w") as f:
                json.dump(index_data, f, indent=2)
            print(f"Cached shard index to {index_path}")
        except Exception as e:
            print(f"Warning: could not cache shard index: {e}")

    def _load_shard(self, shard_idx: int) -> Dict:
        """Load shard with caching."""
        if shard_idx in self._cache:
            self._cache_order.remove(shard_idx)
            self._cache_order.append(shard_idx)
            return self._cache[shard_idx]

        shard_path = os.path.join(self.shard_dir, self.shard_files[shard_idx])
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)

        self._cache[shard_idx] = shard
        self._cache_order.append(shard_idx)

        while len(self._cache) > self.cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        return shard

    def _find_shard(self, idx: int) -> tuple:
        """Find which shard contains the given index."""
        lo, hi = 0, len(self.shard_offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.shard_offsets[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        return lo, idx - self.shard_offsets[lo]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shard_idx, local_idx = self._find_shard(idx)
        shard = self._load_shard(shard_idx)

        # World model shards store samples as list of dicts
        sample = shard["samples"][local_idx]

        # Return the sample dict directly
        return sample

    def get_sampler(self, shuffle: bool = True, seed: int = 42) -> ShardAwareSampler:
        """Get a shard-aware sampler for efficient training."""
        return ShardAwareSampler(
            shard_offsets=self.shard_offsets,
            total_samples=self.total_samples,
            shuffle=shuffle,
            seed=seed,
        )


class WorldModelDataCollator:
    """
    Data collator for world model training.

    Handles variable-length multimodal samples:
    - Pads text_input_ids to batch max length
    - Pads audio/voice latents to batch max time dimension
    - Pads image latents (all same spatial size, no padding needed)
    - Creates attention masks and modality presence indicators
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        max_text_length: int = 2048,
        # Audio/voice latent dimensions
        audio_latent_channels: int = 32,
        audio_latent_mel_bins: int = 10,  # n_mels / compression_factor
        voice_latent_channels: int = 32,
        voice_latent_mel_bins: int = 10,
        # Image latent dimensions
        image_latent_channels: int = 4,
        image_latent_size: int = 32,  # image_size / compression_factor
    ):
        self.pad_token_id = pad_token_id
        self.max_text_length = max_text_length
        self.audio_latent_channels = audio_latent_channels
        self.audio_latent_mel_bins = audio_latent_mel_bins
        self.voice_latent_channels = voice_latent_channels
        self.voice_latent_mel_bins = voice_latent_mel_bins
        self.image_latent_channels = image_latent_channels
        self.image_latent_size = image_latent_size

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        import torch.nn.functional as F

        # Filter None examples
        valid_examples = [ex for ex in examples if ex is not None]
        if not valid_examples:
            raise ValueError("All examples in batch are None")

        batch_size = len(valid_examples)

        # Collect text input IDs and compute max length
        text_ids_list = [ex["text_input_ids"] for ex in valid_examples]
        text_lengths = [len(ids) for ids in text_ids_list]
        max_text_len = min(max(text_lengths), self.max_text_length)

        # Pad text input IDs
        padded_text_ids = []
        text_attention_masks = []
        for ids in text_ids_list:
            length = min(len(ids), max_text_len)
            ids = ids[:length]  # Truncate if needed

            # Create attention mask (1 = valid, 0 = padding)
            mask = torch.ones(max_text_len, dtype=torch.long)
            mask[length:] = 0

            # Pad input IDs
            if len(ids) < max_text_len:
                ids = F.pad(ids, (0, max_text_len - len(ids)), value=self.pad_token_id)

            padded_text_ids.append(ids)
            text_attention_masks.append(mask)

        batch = {
            "text_input_ids": torch.stack(padded_text_ids),  # [B, T_text]
            "text_attention_mask": torch.stack(text_attention_masks),  # [B, T_text]
            "text_lengths": torch.tensor(text_lengths, dtype=torch.long),  # [B]
        }

        # Collect task types
        batch["task_types"] = [ex.get("task_type", "unknown") for ex in valid_examples]

        # Handle audio latents
        audio_latents = [ex.get("audio_mel_spec_latents") for ex in valid_examples]
        if any(l is not None for l in audio_latents):
            # Find max time dimension
            audio_time_lengths = [l.shape[-1] if l is not None else 0 for l in audio_latents]
            max_audio_time = max(audio_time_lengths)

            padded_audio = []
            audio_masks = []
            for latent, length in zip(audio_latents, audio_time_lengths):
                if latent is not None:
                    # latent shape: [C, H, T]
                    if latent.shape[-1] < max_audio_time:
                        latent = F.pad(latent, (0, max_audio_time - latent.shape[-1]), value=0)
                    mask = torch.ones(max_audio_time, dtype=torch.float32)
                    mask[length:] = 0
                else:
                    # Create zero tensor for missing modality
                    latent = torch.zeros(
                        self.audio_latent_channels,
                        self.audio_latent_mel_bins,
                        max_audio_time
                    )
                    mask = torch.zeros(max_audio_time, dtype=torch.float32)

                padded_audio.append(latent)
                audio_masks.append(mask)

            batch["audio_mel_spec_latents"] = torch.stack(padded_audio)  # [B, C, H, T]
            batch["audio_latent_mask"] = torch.stack(audio_masks)  # [B, T]
            batch["audio_latent_lengths"] = torch.tensor(audio_time_lengths, dtype=torch.long)

        # Handle voice latents (same structure as audio)
        voice_latents = [ex.get("voice_mel_spec_latents") for ex in valid_examples]
        if any(l is not None for l in voice_latents):
            voice_time_lengths = [l.shape[-1] if l is not None else 0 for l in voice_latents]
            max_voice_time = max(voice_time_lengths)

            padded_voice = []
            voice_masks = []
            for latent, length in zip(voice_latents, voice_time_lengths):
                if latent is not None:
                    if latent.shape[-1] < max_voice_time:
                        latent = F.pad(latent, (0, max_voice_time - latent.shape[-1]), value=0)
                    mask = torch.ones(max_voice_time, dtype=torch.float32)
                    mask[length:] = 0
                else:
                    latent = torch.zeros(
                        self.voice_latent_channels,
                        self.voice_latent_mel_bins,
                        max_voice_time
                    )
                    mask = torch.zeros(max_voice_time, dtype=torch.float32)

                padded_voice.append(latent)
                voice_masks.append(mask)

            batch["voice_mel_spec_latents"] = torch.stack(padded_voice)
            batch["voice_latent_mask"] = torch.stack(voice_masks)
            batch["voice_latent_lengths"] = torch.tensor(voice_time_lengths, dtype=torch.long)

        # Handle image latents (all same spatial size, just stack)
        image_latents = [ex.get("image_latents") for ex in valid_examples]
        if any(l is not None for l in image_latents):
            padded_images = []
            image_present = []
            for latent in image_latents:
                if latent is not None:
                    padded_images.append(latent)
                    image_present.append(1.0)
                else:
                    # Zero tensor for missing
                    padded_images.append(torch.zeros(
                        self.image_latent_channels,
                        self.image_latent_size,
                        self.image_latent_size
                    ))
                    image_present.append(0.0)

            batch["image_latents"] = torch.stack(padded_images)  # [B, C, H, W]
            batch["image_present"] = torch.tensor(image_present, dtype=torch.float32)  # [B]

        return batch


class MultiModalWorldModelDataset(Dataset):
    """
    Dataset for loading multiple world model datasets with modality-aware sampling.

    Supports:
    - Multiple datasets per modality (e.g., LibriSpeech + CommonVoice for text_audio)
    - Weighted sampling across modalities
    - Weighted sampling within modalities (by dataset size or custom weights)
    - Epoch tracking per dataset

    Each dataset directory should have a config.json with a 'modality' field.
    """

    def __init__(
        self,
        dataset_dirs: List[str],
        modality_weights: Optional[Dict[str, float]] = None,
        dataset_weights: Optional[Dict[str, float]] = None,
        cache_size: int = 3,
    ):
        """
        Args:
            dataset_dirs: List of paths to dataset directories (each with shard_*.pt files)
            modality_weights: Weights per modality (e.g., {"text_only": 0.2, "text_audio": 0.5, "text_image": 0.3})
                              If None, weights are proportional to dataset sizes
            dataset_weights: Weights per dataset path (overrides modality-based weighting within modality)
                             If None, datasets within a modality are weighted by size
            cache_size: Number of shards to keep in memory per dataset
        """
        self.dataset_dirs = dataset_dirs
        self.cache_size = cache_size

        # Load each dataset and group by modality
        self.datasets: Dict[str, List[WorldModelShardedDataset]] = {}  # modality -> [datasets]
        self.dataset_info: Dict[str, Dict] = {}  # path -> {modality, config, dataset}

        print(f"Loading {len(dataset_dirs)} world model datasets...")

        for dataset_dir in dataset_dirs:
            # Load config to get modality
            config_path = os.path.join(dataset_dir, "config.json")
            if not os.path.exists(config_path):
                # Try shard_index.json
                config_path = os.path.join(dataset_dir, "shard_index.json")

            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                modality = config.get("modality", "unknown")
            else:
                print(f"Warning: No config found for {dataset_dir}, skipping")
                continue

            # Load the dataset
            try:
                dataset = WorldModelShardedDataset(dataset_dir, cache_size=cache_size)
            except Exception as e:
                print(f"Warning: Failed to load {dataset_dir}: {e}")
                continue

            # Group by modality
            if modality not in self.datasets:
                self.datasets[modality] = []
            self.datasets[modality].append(dataset)

            # Store info
            self.dataset_info[dataset_dir] = {
                "modality": modality,
                "config": config,
                "dataset": dataset,
                "size": len(dataset),
            }

            print(f"  {dataset_dir}: modality={modality}, samples={len(dataset):,}")

        if not self.datasets:
            raise ValueError("No valid datasets loaded")

        # Compute modality weights
        self.modalities = list(self.datasets.keys())
        total_samples_per_modality = {
            mod: sum(len(d) for d in datasets)
            for mod, datasets in self.datasets.items()
        }
        self.total_samples = sum(total_samples_per_modality.values())

        if modality_weights is not None:
            # Normalize user-provided weights
            total_weight = sum(modality_weights.get(m, 0) for m in self.modalities)
            self.modality_weights = {
                m: modality_weights.get(m, 0) / total_weight if total_weight > 0 else 0
                for m in self.modalities
            }
        else:
            # Weight by total samples per modality (even distribution across modalities)
            self.modality_weights = {
                m: 1.0 / len(self.modalities) for m in self.modalities
            }

        # Compute per-dataset weights within each modality
        self.dataset_weights_per_modality: Dict[str, List[float]] = {}
        for modality, datasets in self.datasets.items():
            if dataset_weights is not None:
                # Find weights for datasets in this modality
                weights = []
                for ds_info in self.dataset_info.values():
                    if ds_info["modality"] == modality:
                        ds_path = next(
                            path for path, info in self.dataset_info.items()
                            if info["dataset"] is ds_info["dataset"]
                        )
                        weights.append(dataset_weights.get(ds_path, len(ds_info["dataset"])))
            else:
                # Weight by dataset size within modality
                weights = [len(d) for d in datasets]

            # Normalize
            total_weight = sum(weights)
            self.dataset_weights_per_modality[modality] = [
                w / total_weight if total_weight > 0 else 1.0 / len(weights)
                for w in weights
            ]

        # Build flat index: list of (modality, dataset_idx, sample_idx) for each sample
        self._build_index()

        # Epoch tracking per dataset
        self.epochs_per_dataset: Dict[str, int] = {
            path: 0 for path in self.dataset_info.keys()
        }

        print(f"\nMultiModalWorldModelDataset initialized:")
        print(f"  Modalities: {self.modalities}")
        print(f"  Modality weights: {self.modality_weights}")
        print(f"  Total samples: {self.total_samples:,}")

    def _build_index(self):
        """Build flat index mapping global idx to (modality, dataset_idx, local_idx)."""
        self.index = []
        self.modality_indices: Dict[str, List[int]] = {m: [] for m in self.modalities}

        global_idx = 0
        for modality in self.modalities:
            for dataset_idx, dataset in enumerate(self.datasets[modality]):
                for local_idx in range(len(dataset)):
                    self.index.append((modality, dataset_idx, local_idx))
                    self.modality_indices[modality].append(global_idx)
                    global_idx += 1

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict:
        modality, dataset_idx, local_idx = self.index[idx]
        dataset = self.datasets[modality][dataset_idx]
        sample = dataset[local_idx]
        # Add modality info to sample for collator
        sample["_modality"] = modality
        return sample

    def get_modality_indices(self, modality: str) -> List[int]:
        """Get all global indices for a given modality."""
        return self.modality_indices.get(modality, [])

    def get_sample_by_modality(self, modality: str, idx: int) -> Dict:
        """Get a sample by modality-local index."""
        indices = self.modality_indices[modality]
        global_idx = indices[idx % len(indices)]
        return self[global_idx]


class ModalitySyncedSampler(torch.utils.data.Sampler):
    """
    Sampler for multi-modal world model training with distributed synchronization.

    Features:
    - All ranks sample from the same modality each batch (required for DDP)
    - Weighted sampling across modalities
    - Shuffling within modalities
    - Epoch tracking per modality/dataset

    Usage:
        sampler = ModalitySyncedSampler(dataset, batch_size=32)
        dataloader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(
        self,
        dataset: MultiModalWorldModelDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
        world_size: int = 1,
        rank: int = 0,
    ):
        """
        Args:
            dataset: MultiModalWorldModelDataset instance
            batch_size: Batch size per GPU
            shuffle: Whether to shuffle within modalities
            seed: Random seed for reproducibility
            drop_last: Drop last incomplete batch per modality
            world_size: Number of distributed processes
            rank: This process's rank
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.world_size = world_size
        self.rank = rank
        self.epoch = 0

        # Compute number of batches per modality
        self.batches_per_modality: Dict[str, int] = {}
        for modality in dataset.modalities:
            n_samples = len(dataset.modality_indices[modality])
            # Account for distributed training - divide by world_size
            n_samples_per_rank = n_samples // world_size
            n_batches = n_samples_per_rank // batch_size
            if not drop_last and n_samples_per_rank % batch_size != 0:
                n_batches += 1
            self.batches_per_modality[modality] = n_batches

        # Total batches weighted by modality weights
        self.total_batches = sum(
            int(n_batches * dataset.modality_weights[mod])
            for mod, n_batches in self.batches_per_modality.items()
        )
        # Ensure at least 1 batch per modality with non-zero weight
        if self.total_batches == 0:
            self.total_batches = sum(self.batches_per_modality.values())

        print(f"ModalitySyncedSampler: {self.total_batches} total batches")
        for mod, n in self.batches_per_modality.items():
            print(f"  {mod}: {n} batches (weight={dataset.modality_weights[mod]:.3f})")

    def __iter__(self):
        # Create RNG with epoch-based seed (same across all ranks)
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        rng = random.Random(self.seed + self.epoch)

        # Shuffle indices within each modality
        modality_indices: Dict[str, List[int]] = {}
        for modality in self.dataset.modalities:
            indices = self.dataset.modality_indices[modality].copy()
            if self.shuffle:
                rng.shuffle(indices)
            modality_indices[modality] = indices

        # Create weighted batch order
        # Each entry is (modality, batch_idx_within_modality)
        batch_order = []
        for modality in self.dataset.modalities:
            weight = self.dataset.modality_weights[modality]
            n_batches = self.batches_per_modality[modality]
            n_weighted_batches = max(1, int(n_batches * weight * len(self.dataset.modalities)))
            for batch_idx in range(min(n_weighted_batches, n_batches)):
                batch_order.append((modality, batch_idx))

        # Shuffle batch order (same across all ranks due to shared seed)
        if self.shuffle:
            rng.shuffle(batch_order)

        # Yield batches
        for modality, batch_idx in batch_order:
            indices = modality_indices[modality]

            # Compute this rank's portion
            total_samples = len(indices)
            samples_per_rank = total_samples // self.world_size
            rank_start = self.rank * samples_per_rank

            # Get batch indices for this rank
            batch_start = rank_start + batch_idx * self.batch_size
            batch_end = batch_start + self.batch_size

            if batch_end <= rank_start + samples_per_rank:
                batch_indices = indices[batch_start:batch_end]
                if len(batch_indices) == self.batch_size or not self.drop_last:
                    yield batch_indices

    def __len__(self):
        return self.total_batches

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling across distributed workers."""
        self.epoch = epoch


def merge_gubert_shards(
    input_dir: str,
    output_dir: str,
    shuffle: bool = True,
    target_shard_size: int = 5000,
):
    """
    Merge GuBERT shards from multiple GPU subdirectories.

    Supports both CTC mode (with text_tokens) and masked mode (without text_tokens).

    Args:
        input_dir: Directory containing gpu_0/, gpu_1/, etc. subdirs
        output_dir: Output directory for merged shards
        shuffle: Whether to shuffle samples across shards
        target_shard_size: Target samples per output shard
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all shard files
    all_shards = []
    gpu_dirs = sorted([
        d for d in os.listdir(input_dir)
        if d.startswith("gpu_") and os.path.isdir(os.path.join(input_dir, d))
    ])

    if not gpu_dirs:
        gpu_dirs = ["."]

    print(f"Found GPU directories: {gpu_dirs}")

    # Collect speaker ID mappings from all GPU config.json files
    all_speaker_id_to_idx = {}
    mode = None  # Will be detected from first shard

    for gpu_dir in gpu_dirs:
        shard_dir = os.path.join(input_dir, gpu_dir)
        shards = sorted([
            os.path.join(shard_dir, f)
            for f in os.listdir(shard_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        ])
        all_shards.extend(shards)
        print(f"  {gpu_dir}: {len(shards)} shards")

        # Load config.json to get speaker_id_to_idx mapping and mode
        config_path = os.path.join(shard_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                gpu_mapping = config.get("speaker_id_to_idx", {})
                # Convert string keys back to original types if needed
                for k, v in gpu_mapping.items():
                    all_speaker_id_to_idx[k] = v
                print(f"    Speakers from this GPU: {len(gpu_mapping)}")
                # Detect mode from config
                if mode is None and "mode" in config:
                    mode = config["mode"]

    # If mode not in config, detect from first shard
    if mode is None and all_shards:
        first_shard = torch.load(all_shards[0], weights_only=False)
        mode = "ctc" if "text_tokens" in first_shard else "masked"

    print(f"Total shards to merge: {len(all_shards)}")
    print(f"Total unique speakers: {len(all_speaker_id_to_idx)}")
    print(f"Mode: {mode}")

    has_text = (mode == "ctc")

    # Create new contiguous mapping for all speakers
    sorted_speakers = sorted(all_speaker_id_to_idx.keys())
    new_speaker_id_to_idx = {sid: idx for idx, sid in enumerate(sorted_speakers)}
    # Also create reverse mapping from old idx to new idx
    old_idx_to_new_idx = {}
    for sid, old_idx in all_speaker_id_to_idx.items():
        new_idx = new_speaker_id_to_idx[sid]
        old_idx_to_new_idx[old_idx] = new_idx

    if shuffle:
        random.shuffle(all_shards)

    # Accumulators
    acc_mel_specs = []
    acc_mel_lengths = []
    acc_text_tokens = []  # Only used in CTC mode
    acc_text_lengths = []  # Only used in CTC mode
    acc_speaker_ids = []
    acc_raw_texts = []  # Only used in CTC mode

    output_shard_idx = 0
    total_samples = 0

    # Track shard info for index
    shard_files = []
    shard_offsets = []

    def save_merged_shard():
        nonlocal acc_mel_specs, acc_mel_lengths, acc_text_tokens, acc_text_lengths
        nonlocal acc_speaker_ids, acc_raw_texts, output_shard_idx, total_samples

        if not acc_mel_specs:
            return

        num_samples_in_shard = sum(x.shape[0] for x in acc_mel_specs)

        shard_data = {
            "mel_specs": torch.cat(acc_mel_specs, dim=0),
            "mel_lengths": torch.cat(acc_mel_lengths, dim=0),
            "speaker_ids": torch.cat(acc_speaker_ids, dim=0),
            "num_samples": num_samples_in_shard,
            "mode": mode,
        }

        # Add text data only in CTC mode
        if has_text and acc_text_tokens:
            # Find max text length for padding
            max_text_len = max(t.shape[1] for t in acc_text_tokens)
            padded_text_tokens = []
            for tokens in acc_text_tokens:
                if tokens.shape[1] < max_text_len:
                    tokens = torch.nn.functional.pad(tokens, (0, max_text_len - tokens.shape[1]), value=0)
                padded_text_tokens.append(tokens)

            shard_data["text_tokens"] = torch.cat(padded_text_tokens, dim=0)
            shard_data["text_lengths"] = torch.cat(acc_text_lengths, dim=0)
            shard_data["raw_texts"] = acc_raw_texts

        shard_filename = f"shard_{output_shard_idx:06d}.pt"
        output_path = os.path.join(output_dir, shard_filename)
        torch.save(shard_data, output_path)

        # Track for index
        shard_files.append(shard_filename)
        shard_offsets.append(total_samples)
        total_samples += num_samples_in_shard

        output_shard_idx += 1
        acc_mel_specs = []
        acc_mel_lengths = []
        acc_text_tokens = []
        acc_text_lengths = []
        acc_speaker_ids = []
        acc_raw_texts = []

    for shard_path in tqdm(all_shards, desc="Merging GuBERT shards"):
        try:
            shard = torch.load(shard_path, weights_only=False)

            acc_mel_specs.append(shard["mel_specs"])
            acc_mel_lengths.append(shard["mel_lengths"])

            # Only accumulate text data in CTC mode
            if has_text:
                if "text_tokens" in shard:
                    acc_text_tokens.append(shard["text_tokens"])
                    acc_text_lengths.append(shard["text_lengths"])
                    # Handle raw_texts
                    if "raw_texts" in shard and shard["raw_texts"]:
                        acc_raw_texts.extend(shard["raw_texts"])
                    else:
                        acc_raw_texts.extend([""] * shard["num_samples"])

            # Remap speaker IDs to new contiguous indices
            old_speaker_ids = shard["speaker_ids"]
            new_speaker_ids = torch.tensor(
                [old_idx_to_new_idx.get(sid.item(), sid.item()) for sid in old_speaker_ids],
                dtype=torch.long
            )
            acc_speaker_ids.append(new_speaker_ids)

            # Save when accumulated enough
            current_count = sum(x.shape[0] for x in acc_mel_specs)
            if current_count >= target_shard_size:
                save_merged_shard()

        except Exception as e:
            print(f"Error loading {shard_path}: {e}")
            continue

    # Final save
    save_merged_shard()

    # Save merged config
    merged_config = {
        "num_speakers": len(new_speaker_id_to_idx),
        "speaker_id_to_idx": new_speaker_id_to_idx,
        "total_samples": total_samples,
        "num_shards": output_shard_idx,
        "mode": mode,
    }

    # Try to get audio settings from first GPU config
    first_config_path = os.path.join(input_dir, gpu_dirs[0], "config.json")
    if os.path.exists(first_config_path):
        with open(first_config_path, "r") as f:
            first_config = json.load(f)
            for key in ["sample_rate", "n_mels", "n_fft", "hop_length", "vocab_size"]:
                if key in first_config:
                    merged_config[key] = first_config[key]

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(merged_config, f, indent=2)

    # Save shard index for fast loading
    shard_index = {
        "shard_files": shard_files,
        "shard_offsets": shard_offsets,
        "total_samples": total_samples,
    }
    with open(os.path.join(output_dir, "shard_index.json"), "w") as f:
        json.dump(shard_index, f, indent=2)

    print(f"\nMerge complete!")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Output shards: {output_shard_idx}")
    print(f"  Unique speakers: {len(new_speaker_id_to_idx)}")
    print(f"  Output dir: {output_dir}")


def merge_gubert_feature_shards(
    input_dir: str,
    output_dir: str,
    shuffle: bool = True,
    target_shard_size: int = 2000,
):
    """
    Merge GuBERT feature shards from multiple GPU subdirectories.

    These are features extracted from a trained GuBERT model for VAE training.

    Args:
        input_dir: Directory containing gpu_0/, gpu_1/, etc. subdirs
        output_dir: Output directory for merged shards
        shuffle: Whether to shuffle samples across shards
        target_shard_size: Target samples per output shard
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all shard files
    all_shards = []
    gpu_dirs = sorted([
        d for d in os.listdir(input_dir)
        if d.startswith("gpu_") and os.path.isdir(os.path.join(input_dir, d))
    ])

    if not gpu_dirs:
        gpu_dirs = ["."]

    print(f"Found GPU directories: {gpu_dirs}")

    for gpu_dir in gpu_dirs:
        shard_dir = os.path.join(input_dir, gpu_dir)
        shards = sorted([
            os.path.join(shard_dir, f)
            for f in os.listdir(shard_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        ])
        all_shards.extend(shards)
        print(f"  {gpu_dir}: {len(shards)} shards")

    print(f"Total shards to merge: {len(all_shards)}")

    if shuffle:
        random.shuffle(all_shards)

    # Accumulators
    acc_features = []
    acc_feature_lengths = []
    acc_mel_lengths = []
    acc_speaker_emb = []

    output_shard_idx = 0
    total_samples = 0
    shard_files = []
    shard_offsets = []

    # Get encoder_dim and num_layers from first shard
    first_shard = torch.load(all_shards[0], weights_only=False)
    feat_shape = first_shard["features"].shape
    # Detect multi-layer format:
    # - Single layer: [N, encoder_dim, T']
    # - Multi-layer:  [N, num_layers, encoder_dim, T']
    if len(feat_shape) == 4:
        # Multi-layer format
        num_layers = feat_shape[1]
        encoder_dim = feat_shape[2]
        print(f"Multi-layer format detected: {num_layers} layers, encoder_dim={encoder_dim}")
    else:
        # Single layer format
        num_layers = 1
        encoder_dim = feat_shape[1]
        print(f"Single-layer format: encoder_dim={encoder_dim}")

    def save_merged_shard():
        nonlocal acc_features, acc_feature_lengths, acc_mel_lengths, acc_speaker_emb
        nonlocal output_shard_idx, total_samples, shard_files, shard_offsets

        if not acc_features:
            return

        # Find max feature length in this shard for padding
        max_feature_len = max(f.shape[-1] for f in acc_features)

        # Pad features to same length
        padded_features = []
        for feat in acc_features:
            if feat.shape[-1] < max_feature_len:
                feat = F.pad(feat, (0, max_feature_len - feat.shape[-1]), value=0)
            padded_features.append(feat)

        num_samples_in_shard = sum(f.shape[0] for f in padded_features)

        # Shape depends on multi-layer mode:
        # - Single layer: [N, encoder_dim, T']
        # - Multi-layer:  [N, num_layers, encoder_dim, T']
        shard_data = {
            "features": torch.cat(padded_features, dim=0),
            "feature_lengths": torch.cat(acc_feature_lengths, dim=0),
            "mel_lengths": torch.cat(acc_mel_lengths, dim=0),
            "speaker_embeddings": torch.cat(acc_speaker_emb, dim=0),
            "num_samples": num_samples_in_shard,
        }

        shard_filename = f"shard_{output_shard_idx:06d}.pt"
        output_path = os.path.join(output_dir, shard_filename)
        torch.save(shard_data, output_path)

        # Track for index
        shard_files.append(shard_filename)
        shard_offsets.append(total_samples)
        total_samples += num_samples_in_shard

        output_shard_idx += 1
        acc_features = []
        acc_feature_lengths = []
        acc_mel_lengths = []
        acc_speaker_emb = []

    for shard_path in tqdm(all_shards, desc="Merging GuBERT feature shards"):
        try:
            shard = torch.load(shard_path, weights_only=False)

            acc_features.append(shard["features"])
            acc_feature_lengths.append(shard["feature_lengths"])
            acc_mel_lengths.append(shard["mel_lengths"])
            acc_speaker_emb.append(shard["speaker_embeddings"])

            # Save when accumulated enough
            current_count = sum(f.shape[0] for f in acc_features)
            if current_count >= target_shard_size:
                save_merged_shard()

        except Exception as e:
            print(f"Error loading {shard_path}: {e}")
            continue

    # Final save
    save_merged_shard()

    # Save merged config
    merged_config = {
        "encoder_dim": encoder_dim,
        "num_layers": num_layers,
        "total_samples": total_samples,
        "num_shards": output_shard_idx,
    }

    # Try to get settings from first GPU config
    first_config_path = os.path.join(input_dir, gpu_dirs[0], "config.json")
    if os.path.exists(first_config_path):
        with open(first_config_path, "r") as f:
            first_config = json.load(f)
            for key in ["gubert_checkpoint", "gubert_config", "sample_rate", "n_mels",
                        "n_fft", "hop_length", "max_audio_seconds", "speaker_encoder_type",
                        "speaker_embedding_dim", "total_stride", "normalize", "layers"]:
                if key in first_config:
                    merged_config[key] = first_config[key]

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(merged_config, f, indent=2)

    # Save shard index for fast loading
    shard_index = {
        "shard_files": shard_files,
        "shard_offsets": shard_offsets,
        "total_samples": total_samples,
    }
    with open(os.path.join(output_dir, "shard_index.json"), "w") as f:
        json.dump(shard_index, f, indent=2)

    print(f"\nMerge complete!")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Output shards: {output_shard_idx}")
    print(f"  Encoder dimension: {encoder_dim}")
    print(f"  Num layers: {num_layers}")
    print(f"  Output dir: {output_dir}")


def merge_world_model_shards(
    input_dirs: List[str],
    output_dir: str,
    shuffle: bool = True,
    target_shard_size: int = 2000,
    seed: int = 42,
):
    """
    Merge world model shards from multiple input directories (different datasets).

    Unlike other merge functions that take a single input_dir with GPU subdirs,
    this takes multiple input_dirs (each potentially having GPU subdirs) and
    merges all samples together with shuffling for cross-dataset mixing.

    Args:
        input_dirs: List of directories to merge (each may have gpu_*/ subdirs)
        output_dir: Output directory for merged shards
        shuffle: Whether to shuffle samples across shards
        target_shard_size: Target samples per output shard
        seed: Random seed for reproducible shuffling
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect all shard files and modality info from all input directories
    all_shard_paths = []
    source_modalities = {}  # input_dir -> modality

    for input_dir in input_dirs:
        # Try to read modality from config.json
        config_path = os.path.join(input_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            source_modalities[input_dir] = config.get("modality", "unknown")
        else:
            source_modalities[input_dir] = "unknown"
        # Check for GPU subdirs
        gpu_dirs = sorted([
            d for d in os.listdir(input_dir)
            if d.startswith("gpu_") and os.path.isdir(os.path.join(input_dir, d))
        ])

        if not gpu_dirs:
            gpu_dirs = ["."]

        print(f"Input: {input_dir}")
        for gpu_dir in gpu_dirs:
            shard_dir = os.path.join(input_dir, gpu_dir)
            shards = sorted([
                os.path.join(shard_dir, f)
                for f in os.listdir(shard_dir)
                if f.startswith("shard_") and f.endswith(".pt")
            ])
            all_shard_paths.extend(shards)
            if gpu_dir != ".":
                print(f"  {gpu_dir}: {len(shards)} shards")
            else:
                print(f"  {len(shards)} shards")

    print(f"\nTotal shards to merge: {len(all_shard_paths)}")

    # Build flat list of (shard_path, local_idx) for all samples
    print("Building sample index...")
    sample_refs = []  # List of (shard_path, local_idx)

    for shard_path in tqdm(all_shard_paths, desc="Scanning shards"):
        try:
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            num_samples = shard["num_samples"]
            for local_idx in range(num_samples):
                sample_refs.append((shard_path, local_idx))
        except Exception as e:
            print(f"Error loading {shard_path}: {e}")
            continue

    print(f"Total samples: {len(sample_refs):,}")

    # Shuffle with deterministic seed
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(sample_refs)
        print(f"Shuffled with seed {seed}")

    # Re-shard the samples
    output_shard_idx = 0
    shard_samples = []
    shard_files = []
    shard_offsets = []
    total_written = 0

    # Cache for loaded shards (to avoid reloading same shard repeatedly)
    shard_cache = {}
    cache_order = []
    max_cache_size = 5

    def load_shard_cached(path):
        if path in shard_cache:
            cache_order.remove(path)
            cache_order.append(path)
            return shard_cache[path]

        shard = torch.load(path, map_location="cpu", weights_only=False)
        shard_cache[path] = shard
        cache_order.append(path)

        while len(shard_cache) > max_cache_size:
            oldest = cache_order.pop(0)
            del shard_cache[oldest]

        return shard

    def save_output_shard():
        nonlocal shard_samples, output_shard_idx, total_written

        if not shard_samples:
            return

        shard_filename = f"shard_{output_shard_idx:06d}.pt"
        output_path = os.path.join(output_dir, shard_filename)

        shard_data = {
            "samples": shard_samples,
            "num_samples": len(shard_samples),
        }
        torch.save(shard_data, output_path)

        shard_files.append(shard_filename)
        shard_offsets.append(total_written)
        total_written += len(shard_samples)

        output_shard_idx += 1
        shard_samples = []

    # Process all samples
    for shard_path, local_idx in tqdm(sample_refs, desc="Writing shards"):
        try:
            shard = load_shard_cached(shard_path)
            sample = shard["samples"][local_idx]
            shard_samples.append(sample)

            if len(shard_samples) >= target_shard_size:
                save_output_shard()

        except Exception as e:
            print(f"Error processing {shard_path}[{local_idx}]: {e}")
            continue

    # Final save
    save_output_shard()

    # Save shard index
    index_data = {
        "shard_files": shard_files,
        "shard_offsets": shard_offsets,
        "total_samples": total_written,
        "dataset_type": "world_model",
        "source_dirs": input_dirs,
        "seed": seed,
    }

    with open(os.path.join(output_dir, "shard_index.json"), "w") as f:
        json.dump(index_data, f, indent=2)

    # Save metadata
    meta = {
        "total_samples": total_written,
        "num_shards": output_shard_idx,
        "shard_size": target_shard_size,
        "source_dirs": input_dirs,
        "source_modalities": source_modalities,
        "modalities": list(set(source_modalities.values())),
        "seed": seed,
        "shuffled": shuffle,
        "dataset_type": "world_model",
    }

    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Also save a config.json with modality info for compatibility
    config = {
        "modality": "mixed" if len(set(source_modalities.values())) > 1 else list(source_modalities.values())[0],
        "source_modalities": source_modalities,
        "dataset_type": "world_model",
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nMerge complete!")
    print(f"  Total samples: {total_written:,}")
    print(f"  Output shards: {output_shard_idx}")
    print(f"  Output dir: {output_dir}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Merge command (image diffusion)
    merge_parser = subparsers.add_parser("merge", help="Merge image diffusion shards from multiple GPUs")
    merge_parser.add_argument("--input_dir", type=str, required=True)
    merge_parser.add_argument("--output_dir", type=str, required=True)
    merge_parser.add_argument("--shard_size", type=int, default=10000)
    merge_parser.add_argument("--no_shuffle", action="store_true")

    # Merge audio VAE command
    merge_audio_parser = subparsers.add_parser("merge-audio-vae", help="Merge audio VAE shards from multiple GPUs")
    merge_audio_parser.add_argument("--input_dir", type=str, required=True)
    merge_audio_parser.add_argument("--output_dir", type=str, required=True)
    merge_audio_parser.add_argument("--shard_size", type=int, default=5000)
    merge_audio_parser.add_argument("--no_shuffle", action="store_true")

    # Merge audio diffusion command
    merge_audio_diff_parser = subparsers.add_parser("merge-audio-diffusion", help="Merge audio diffusion shards from multiple GPUs")
    merge_audio_diff_parser.add_argument("--input_dir", type=str, required=True)
    merge_audio_diff_parser.add_argument("--output_dir", type=str, required=True)
    merge_audio_diff_parser.add_argument("--shard_size", type=int, default=2000)
    merge_audio_diff_parser.add_argument("--no_shuffle", action="store_true")

    # Merge vocoder command
    merge_vocoder_parser = subparsers.add_parser("merge-vocoder", help="Merge vocoder shards from multiple GPUs")
    merge_vocoder_parser.add_argument("--input_dir", type=str, required=True)
    merge_vocoder_parser.add_argument("--output_dir", type=str, required=True)
    merge_vocoder_parser.add_argument("--shard_size", type=int, default=2000)
    merge_vocoder_parser.add_argument("--no_shuffle", action="store_true")

    # Merge GuBERT command
    merge_gubert_parser = subparsers.add_parser("merge-gubert", help="Merge GuBERT shards from multiple GPUs")
    merge_gubert_parser.add_argument("--input_dir", type=str, required=True)
    merge_gubert_parser.add_argument("--output_dir", type=str, required=True)
    merge_gubert_parser.add_argument("--shard_size", type=int, default=5000)
    merge_gubert_parser.add_argument("--no_shuffle", action="store_true")

    # Merge GuBERT features command
    merge_gubert_feat_parser = subparsers.add_parser("merge-gubert-features", help="Merge GuBERT feature shards from multiple GPUs")
    merge_gubert_feat_parser.add_argument("--input_dir", type=str, required=True)
    merge_gubert_feat_parser.add_argument("--output_dir", type=str, required=True)
    merge_gubert_feat_parser.add_argument("--shard_size", type=int, default=2000)
    merge_gubert_feat_parser.add_argument("--no_shuffle", action="store_true")

    # Merge world model command
    merge_wm_parser = subparsers.add_parser("merge-world-model", help="Merge world model shards from multiple datasets")
    merge_wm_parser.add_argument("--input_dirs", type=str, nargs="+", required=True,
                                  help="Input directories to merge (each may have gpu_*/ subdirs)")
    merge_wm_parser.add_argument("--output_dir", type=str, required=True)
    merge_wm_parser.add_argument("--shard_size", type=int, default=2000)
    merge_wm_parser.add_argument("--seed", type=int, default=42,
                                  help="Random seed for shuffling (for reproducibility across GPUs)")
    merge_wm_parser.add_argument("--no_shuffle", action="store_true")

    # Info command
    info_parser = subparsers.add_parser("info", help="Print info about shards")
    info_parser.add_argument("--shard_dir", type=str, required=True)
    info_parser.add_argument("--type", type=str, default="auto", choices=["auto", "image", "audio_vae"],
                             help="Dataset type (auto-detects from shard contents)")

    # Build index command (for existing shards without re-merging)
    index_parser = subparsers.add_parser("build-index", help="Build shard_index.json for existing shards (skips 14min indexing on load)")
    index_parser.add_argument("--shard_dir", type=str, required=True, help="Directory containing shard_*.pt files")

    args = parser.parse_args()

    if args.command == "merge":
        merge_shards(
            args.input_dir,
            args.output_dir,
            shuffle=not args.no_shuffle,
            target_shard_size=args.shard_size,
        )

    elif args.command == "merge-audio-vae":
        merge_audio_vae_shards(
            args.input_dir,
            args.output_dir,
            shuffle=not args.no_shuffle,
            target_shard_size=args.shard_size,
        )

    elif args.command == "merge-audio-diffusion":
        merge_audio_diffusion_shards(
            args.input_dir,
            args.output_dir,
            shuffle=not args.no_shuffle,
            target_shard_size=args.shard_size,
        )

    elif args.command == "merge-vocoder":
        merge_vocoder_shards(
            args.input_dir,
            args.output_dir,
            shuffle=not args.no_shuffle,
            target_shard_size=args.shard_size,
        )

    elif args.command == "merge-gubert":
        merge_gubert_shards(
            args.input_dir,
            args.output_dir,
            shuffle=not args.no_shuffle,
            target_shard_size=args.shard_size,
        )

    elif args.command == "merge-gubert-features":
        merge_gubert_feature_shards(
            args.input_dir,
            args.output_dir,
            shuffle=not args.no_shuffle,
            target_shard_size=args.shard_size,
        )

    elif args.command == "merge-world-model":
        merge_world_model_shards(
            args.input_dirs,
            args.output_dir,
            shuffle=not args.no_shuffle,
            target_shard_size=args.shard_size,
            seed=args.seed,
        )

    elif args.command == "info":
        # Auto-detect dataset type
        shard_files = [f for f in os.listdir(args.shard_dir) if f.startswith("shard_") and f.endswith(".pt")]
        if not shard_files:
            print(f"No shard files found in {args.shard_dir}")
            exit(1)

        # Peek at first shard to detect type
        first_shard = torch.load(os.path.join(args.shard_dir, shard_files[0]), map_location="cpu")

        if args.type == "auto":
            if "mel_specs" in first_shard:
                dataset_type = "audio_vae"
            else:
                dataset_type = "image"
        else:
            dataset_type = args.type

        print(f"Shard directory: {args.shard_dir}")
        print(f"Dataset type: {dataset_type}")

        if dataset_type == "audio_vae":
            dataset = AudioVAEShardedDataset(args.shard_dir)
        else:
            dataset = ShardedDataset(args.shard_dir)

        print(f"Total samples: {dataset.total_samples:,}")
        print(f"Number of shards: {len(dataset.shard_files)}")

        # Sample one
        sample = dataset[0]
        print(f"\nSample keys: {list(sample.keys())}")
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape} ({v.dtype})")
            else:
                print(f"  {k}: {v}")

    elif args.command == "build-index":
        # Build shard index for existing shards
        shard_dir = args.shard_dir
        shard_files = sorted([
            f for f in os.listdir(shard_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        ])

        if not shard_files:
            print(f"No shard files found in {shard_dir}")
            exit(1)

        print(f"Building index for {len(shard_files)} shards in {shard_dir}...")

        shard_offsets = []
        total_samples = 0

        for shard_file in tqdm(shard_files):
            shard_offsets.append(total_samples)
            shard_path = os.path.join(shard_dir, shard_file)
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
            total_samples += shard["num_samples"]

        # Auto-detect dataset type from first shard
        first_shard = torch.load(os.path.join(shard_dir, shard_files[0]), map_location="cpu", weights_only=True)
        if "waveform_labels" in first_shard:
            dataset_type = "vocoder"
        elif "speaker_embeddings" in first_shard:
            dataset_type = "audio_vae"
        elif "latent_mus" in first_shard or "text_embeddings" in first_shard:
            dataset_type = "audio_diffusion"
        else:
            dataset_type = "unknown"

        index_data = {
            "shard_files": shard_files,
            "shard_offsets": shard_offsets,
            "total_samples": total_samples,
            "dataset_type": dataset_type,
        }

        index_path = os.path.join(shard_dir, "shard_index.json")
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)

        print(f"\nIndex built successfully!")
        print(f"  Shards: {len(shard_files)}")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Dataset type: {dataset_type}")
        print(f"  Saved to: {index_path}")

    else:
        parser.print_help()
