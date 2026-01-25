import json
import os
import torch


from torch.utils.data import Dataset
from tqdm import tqdm

from scripts.data.dataset import ShardAwareSampler


class ImageVAEShardedDataset(Dataset):
    """
    Efficient dataset for loading preprocessed image VAE shards.

    Loads shards containing:
    - images: list[torch.Tensor] of images [3, H, W]

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
        image_size: int = 256,
    ):
        """
        Args:
            shard_dir: Directory containing shard_*.pt files
            cache_size: Number of shards to keep in memory
            image_size: Size of the images (assumed square)
        """
        self.shard_dir = shard_dir
        self.cache_size = cache_size
        self.image_size = image_size

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

        print(f"Indexing {len(self.shard_files)} image VAE shards (first time only, will be cached)...")
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
            "dataset_type": "image_vae",
        }
        try:
            with open(index_path, "w") as f:
                json.dump(index_data, f, indent=2)
            print(f"Cached shard index to {index_path}")
        except Exception as e:
            print(f"Warning: could not cache shard index: {e}")

    def _load_shard(self, shard_idx: int) -> dict[str, torch.Tensor]:
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        shard_idx, local_idx = self._find_shard(idx)
        shard = self._load_shard(shard_idx)

        image = shard["images"][local_idx]  # [3, H, W]

        # Return sample dict compatible with ImageVAEDataCollator
        sample = {
            "image": image,  # [3, H, W]
        }

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
