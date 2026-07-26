import os
import torch



def _shard_debug_enabled():
    """Check if shard debugging is enabled via environment variable."""
    return os.environ.get("SHARD_DEBUG", "0") == "1"


def scan_shard_lengths(shard_dir: str, shard_files: list[str], length_key: str) -> list[int]:
    """Read ONLY the per-sample ``length_key`` array from every shard, concatenated into
    one Python list indexed by global sample index.

    Uses ``torch.load(..., mmap=True)`` so only that small array is paged in, NOT the
    multi-GB feature/waveform/mel tensors beside it — measured ~400x cheaper than a full
    shard load. This is what lets length-bucketing get every sample's length at sampler
    init without either re-loading all shards or duplicating lengths into the index.
    """
    lengths: list[int] = []
    for fname in shard_files:
        path = os.path.join(shard_dir, fname)
        shard = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if length_key not in shard:
            raise KeyError(
                f"shard {fname} has no '{length_key}' (present: "
                f"{[k for k in shard.keys() if k.endswith('_lengths')]}); "
                f"cannot bucket by length on this key."
            )
        lengths.extend(int(x) for x in shard[length_key].tolist())
    return lengths


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
        shard_offsets: list[int],
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


class LengthGroupedShardSampler(ShardAwareSampler):
    """Shard-aware sampler that also groups similar-LENGTH samples into the same batch,
    cutting the padding FLOPs wasted when a batch mixes short and long sequences.

    Bucketing happens WITHIN each shard, so shard locality (one load per shard per epoch)
    is fully preserved — only the intra-shard emission order changes. Per shard, per epoch:
        shuffle indices -> megabatches of batch_size*mega_factor -> sort each by length
        -> chunk into batch_size batches -> shuffle batch order.
    Consecutive ``batch_size`` indices are then length-homogeneous, while the per-epoch
    shuffle keeps batch membership varying across epochs (so it is not the same fixed
    batches every time, which would hurt SGD).

    Emits FLAT indices; the DataLoader chunks them by ``batch_size``. When a shard's size
    is not a multiple of ``batch_size`` its remainder merges with the next shard's head
    into one mixed-length batch — the same shard-boundary straddle the base sampler already
    has, ~1 batch per shard, negligible.
    """

    def __init__(
        self,
        shard_offsets: list[int],
        total_samples: int,
        lengths: list[int],
        batch_size: int,
        mega_factor: int = 25,
        shuffle: bool = True,
        seed: int = 42,
        num_replicas: int = None,
        rank: int = None,
    ):
        super().__init__(shard_offsets, total_samples, shuffle=shuffle, seed=seed,
                         num_replicas=num_replicas, rank=rank)
        if len(lengths) != total_samples:
            raise ValueError(
                f"lengths ({len(lengths)}) must match total_samples ({total_samples})")
        self.lengths = lengths
        self.batch_size = max(1, int(batch_size))
        self.mega_factor = max(1, int(mega_factor))

    def _bucket_shard(self, shard_global: list[int], g: torch.Generator) -> list[int]:
        """Length-bucketed emission order for one shard's global indices."""
        perm = torch.randperm(len(shard_global), generator=g).tolist()
        shuffled = [shard_global[i] for i in perm]

        mb = self.batch_size * self.mega_factor
        batches: list[list[int]] = []
        for s in range(0, len(shuffled), mb):
            mega = shuffled[s:s + mb]
            mega.sort(key=lambda idx: self.lengths[idx])
            for b in range(0, len(mega), self.batch_size):
                batches.append(mega[b:b + self.batch_size])

        # Shuffle the ORDER of the length-homogeneous batches (keeps batches intact).
        border = torch.randperm(len(batches), generator=g).tolist()
        out: list[int] = []
        for bi in border:
            out.extend(batches[bi])
        return out

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            perm = torch.randperm(len(self.my_shards), generator=g).tolist()
            shard_order = [self.my_shards[i] for i in perm]
        else:
            shard_order = self.my_shards

        indices: list[int] = []
        for shard_idx in shard_order:
            start = self.shard_offsets[shard_idx]
            size = self.shard_sizes[shard_idx]
            shard_global = list(range(start, start + size))
            if self.shuffle:
                shard_g = torch.Generator()
                shard_g.manual_seed(self.seed + self.epoch + shard_idx)
                indices.extend(self._bucket_shard(shard_global, shard_g))
            else:
                # Deterministic eval: strict length sort, no shuffle.
                indices.extend(sorted(shard_global, key=lambda idx: self.lengths[idx]))

        # Distributed padding, mirroring the base sampler.
        if len(indices) < self._num_samples:
            indices.extend(indices[:self._num_samples - len(indices)])

        return self._tracked_iter(indices)
