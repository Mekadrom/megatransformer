import json
import os
import torch


from torch.utils.data import Dataset
from tqdm import tqdm

from scripts.data.dataset import ShardAwareSampler


class AudioShardedDataset(Dataset):
    """
    Efficient dataset for loading preprocessed audio shards.

    Loads shards containing:
    - features: optional - SIVE encoder features
        - Single layer: [B, encoder_dim, T']
        - Multi-layer: [B, num_layers, encoder_dim, T']
    - feature_lengths: optional - [B] feature lengths before padding
    - mel_specs: [B, n_mel_channels, T] original mel spectrograms
    - mel_lengths: optional - [B] original mel spectrogram lengths (before SIVE subsampling)
    - speaker_embeddings: optional - [B, embedding_dim] speaker embeddings for conditioning
    - f0: optional - [B, T'] fundamental frequency aligned with features
    - voiced: optional - [B, T'] voiced/unvoiced confidence aligned with features
    - waveforms: optional - [B, T] original waveforms

    One of waveforms, mel_specs, or features WILL be present in each shard.
    """

    SHARD_INDEX_FILE = "shard_index.json"

    def __init__(
        self,
        shard_dir: str,
        cache_size: int = 3,
        columns: list[str] = []
    ):
        """
        Args:
            shard_dir: Directory containing shard_*.pt files
            cache_size: Number of shards to keep in memory
            columns: List of columns to load from each shard (e.g. ["features", "mel_specs", "speaker_embeddings"])
        """
        self.shard_dir = shard_dir
        self.cache_size = cache_size
        self.columns = columns

        if not self.columns or len(self.columns) == 0:
            self.columns = [
                "conditions",  # text conditions as tensors
                "features",  # features extracted by a pretrained SIVE model as tensors (could single-layer extractions or multi-layer with an extra layer dimension at dim=1)
                "mel_specs",  # pre-extracted mel spectrograms as tensors
                "speaker_embeddings",  # unnormalized speaker embeddings
                "speaker_ids",  # speaker IDs normalized to the total number of unique speakers in the entire dataset
                "waveforms",  # original waveforms
                "f0",  # f0 baselines as extracted by a pretrained model
                "vuv",  # vuv booleans as extracted by a pretrained model
                "ctc_tokens",  # ctc tokens as tensors
                "text",  # text as strings
            ]

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

        self.encoder_dim = self.config.get("encoder_dim", 128)
        self.speaker_embedding_dim = self.config.get("speaker_embedding_dim", 192)

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

        print(f"Indexing {len(self.shard_files)} SIVE feature shards (first time only, will be cached)...")
        for shard_file in tqdm(self.shard_files):
            self.shard_offsets.append(self.total_samples)
            shard_path = os.path.join(self.shard_dir, shard_file)
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
            self.total_samples += shard["num_samples"]

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

        sample = {}

        if 'features' in shard and 'features' in self.columns:
            # features shape depends on multi-layer mode:
            # - Single layer: [encoder_dim, T']
            # - Multi-layer:  [num_layers, encoder_dim, T']
            sample['features'] = shard["features"][local_idx]
            sample["feature_length"] = shard["feature_lengths"][local_idx]

        if 'waveforms' in shard and 'waveforms' in self.columns:
            sample['waveform'] = shard["waveforms"][local_idx]
            sample["waveform_length"] = shard["waveform_lengths"][local_idx]

        if 'mel_specs' in shard and 'mel_specs' in self.columns:
            sample['mel_spec'] = shard["mel_specs"][local_idx]
            sample["mel_length"] = shard["mel_lengths"][local_idx]

        if "speaker_embeddings" in shard and "speaker_embeddings" in self.columns:
            sample["speaker_embedding"] = shard["speaker_embeddings"][local_idx]

        if "speaker_ids" in shard and "speaker_ids" in self.columns:
            sample["speaker_id"] = shard["speaker_ids"][local_idx]

        # Add F0 data if available
        if "f0" in shard and "f0" in self.columns:
            sample["f0"] = shard["f0"][local_idx]

        if "vuv" in shard and "vuv" in self.columns:
            sample["vuv"] = shard["vuv"][local_idx]

        if "ctc_tokens" in shard and "ctc_tokens" in self.columns:
            sample["ctc_tokens"] = shard["ctc_tokens"][local_idx]
            sample["ctc_length"] = shard["ctc_lengths"][local_idx]

        if "text" in shard and "text" in self.columns:
            sample["text"] = shard["text"][local_idx]

        return sample

    def get_sampler(self, shuffle: bool = True, seed: int = 42) -> ShardAwareSampler:
        """Get a shard-aware sampler for efficient training."""
        return ShardAwareSampler(
            shard_offsets=self.shard_offsets,
            total_samples=self.total_samples,
            shuffle=shuffle,
            seed=seed,
        )
