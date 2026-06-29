import json
import os
import bisect
from typing import Optional

import torch

from torch.utils.data import Dataset


class MultimodalShardedDataset(Dataset):
    """
    Multimodal dataset that composes separate per-modality shard directories.

    Wraps independently preprocessed text, audio, and image shard caches.
    Each modality is optional. Samples are drawn by index from each modality
    independently — shorter datasets wrap around via modulo so that every
    modality contributes to every batch element.

    Shard loading uses per-modality LRU caches so that at most one shard per
    modality is active at a time (configurable via cache_size).
    """

    SHARD_INDEX_FILE = "shard_index.json"

    def __init__(
        self,
        text_shard_dir: str = None,
        audio_shard_dir: str = None,
        voice_shard_dir: str = None,
        image_shard_dir: str = None,
        voice_synthesis_shard_dir: str = None,
        text_columns: list[str] = None,
        audio_columns: list[str] = None,
        voice_columns: list[str] = None,
        cache_size: int = 3,
        max_samples: int = None,
        include_tasks: list[str] = None,
    ):
        """
        Args:
            text_shard_dir: Path to preprocessed text shards (token_ids, text_lengths, text)
            audio_shard_dir: Path to preprocessed audio shards (features, mel_specs, etc.)
            voice_shard_dir: Path to preprocessed voice shards (same format as audio).
                This is the canonical voice corpus: it anchors the dataset length
                and is the default source for BOTH voice directions.
            image_shard_dir: Path to preprocessed image shards (images)
            voice_synthesis_shard_dir: Optional override corpus used ONLY by the voice
                synthesis (text→voice) direction. When set, synthesis draws from here while
                transcription (voice→text) keeps drawing from voice_shard_dir. Intended use:
                a clean subset for synthesis vs. the clean+noisy superset (voice_shard_dir)
                for transcription. Requires voice_shard_dir to also be set. When None,
                synthesis and transcription share voice_shard_dir (legacy behavior).
            text_columns: Columns to load from text shards. If None, loads all.
            audio_columns: Columns to load from audio shards. If None, loads all.
            voice_columns: Columns to load from voice shards. If None, mirrors audio_columns.
            cache_size: Number of shards to keep cached per modality
            max_samples: If set, cap the effective dataset length (for overfitting experiments)
            include_tasks: Optional allowlist of task names to keep (e.g.
                ["text_continuation", "voice_synthesis"]). None = all tasks for the
                configured modalities. Filters self.task_types; keyed on task name,
                so the directionless "text_continuation" needs no special handling.
        """
        self.cache_size = cache_size
        self.modalities = {}

        if text_shard_dir is not None:
            self.modalities["text"] = self._init_modality(text_shard_dir, "text")
        if audio_shard_dir is not None:
            self.modalities["audio"] = self._init_modality(audio_shard_dir, "audio")
        if voice_shard_dir is not None:
            self.modalities["voice"] = self._init_modality(voice_shard_dir, "voice")
        if image_shard_dir is not None:
            self.modalities["image"] = self._init_modality(image_shard_dir, "image")

        if not self.modalities:
            raise ValueError("At least one shard directory must be provided")

        # Optional per-(modality, direction) source overrides. By default a task
        # draws from its modality's single corpus (self.modalities[mod]); an entry
        # here gives one direction its own independently-indexed corpus. A "source"
        # dict is structurally identical to a modality dict, so _load_shard /
        # _find_shard_for_idx operate on it unchanged.
        self._direction_sources = {}
        if voice_synthesis_shard_dir is not None:
            if "voice" not in self.modalities:
                raise ValueError(
                    "voice_synthesis_shard_dir was provided but voice_shard_dir was not. "
                    "The base voice corpus is required — it anchors dataset length and is "
                    "the transcription source."
                )
            self._direction_sources[("voice", "synthesis")] = self._init_modality(
                voice_synthesis_shard_dir, "voice:synthesis"
            )

        self.text_columns = text_columns
        self.audio_columns = audio_columns
        self.voice_columns = voice_columns

        # Default audio columns (mirrors AudioShardedDataset defaults)
        _default_audio_columns = [
            "conditions", "features", "mel_specs", "speaker_embeddings",
            "speaker_ids", "waveforms", "f0", "vuv", "ctc_tokens", "text",
        ]
        if self.audio_columns is None and "audio" in self.modalities:
            self.audio_columns = _default_audio_columns
        if self.voice_columns is None and "voice" in self.modalities:
            self.voice_columns = _default_audio_columns

        # Optional task allowlist (None = all tasks for the configured modalities).
        # Filters self.task_types — the single source of truth for the round-robin,
        # length, sampler, and eval grouping — keyed on task NAME, so the
        # directionless "text_continuation" needs no special handling.
        self._include_tasks = set(include_tasks) if include_tasks else None
        if self._include_tasks is not None:
            available = [t[0] for t in self._all_task_types()]
            unknown = self._include_tasks - set(available)
            if unknown:
                raise ValueError(
                    f"include_tasks contains unknown/unavailable task(s) {sorted(unknown)}. "
                    f"Available for the configured modalities ({list(self.modalities)}): {available}."
                )

        # Total length = max across modalities × number of task types, so one
        # "epoch" covers every sample in the largest modality across all tasks
        # (synthesis + transcription). Shorter modalities wrap via modulo.
        # Without the ×n_tasks factor, __getitem__'s `within_task_idx = idx // n_tasks`
        # caps at total_samples // n_tasks, leaving most samples unreachable.
        n_tasks = len(self.task_types)
        all_sources = list(self.modalities.values()) + list(self._direction_sources.values())
        self.total_samples = max(s["total_samples"] for s in all_sources) * n_tasks

        # Cap dataset length for overfitting experiments
        if max_samples is not None and max_samples > 0:
            self.total_samples = min(self.total_samples, max_samples)

        modality_summary = ", ".join(
            f"{name}: {m['total_samples']:,} samples / {len(m['shard_files'])} shards"
            for name, m in self.modalities.items()
        )
        print(f"[MultimodalShardedDataset] {modality_summary}")
        cap_note = f" (capped from max_samples={max_samples})" if max_samples is not None and self.total_samples <= max_samples else ""
        print(f"[MultimodalShardedDataset] Effective length: {self.total_samples:,}{cap_note}")

    def _init_modality(self, shard_dir: str, name: str) -> dict:
        """Load or build shard index for a single modality."""
        index_path = os.path.join(shard_dir, self.SHARD_INDEX_FILE)

        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index_data = json.load(f)
            shard_files = index_data["shard_files"]
            shard_offsets = index_data["shard_offsets"]
            total_samples = index_data["total_samples"]
        else:
            shard_files = sorted([
                f for f in os.listdir(shard_dir)
                if f.startswith("shard_") and f.endswith(".pt")
            ])
            if not shard_files:
                raise ValueError(f"No shard files found in {shard_dir}")

            from tqdm import tqdm
            shard_offsets = []
            total_samples = 0
            print(f"Indexing {len(shard_files)} {name} shards...")
            for shard_file in tqdm(shard_files):
                shard_offsets.append(total_samples)
                shard_path = os.path.join(shard_dir, shard_file)
                shard = torch.load(shard_path, map_location="cpu", weights_only=True, mmap=True)
                total_samples += shard["num_samples"]

            index_data = {
                "shard_files": shard_files,
                "shard_offsets": shard_offsets,
                "total_samples": total_samples,
            }
            with open(index_path, "w") as f:
                json.dump(index_data, f, indent=2)

        # Load config if available
        config_path = os.path.join(shard_dir, "config.json")
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)

        return {
            "shard_dir": shard_dir,
            "shard_files": shard_files,
            "shard_offsets": shard_offsets,
            "total_samples": total_samples,
            "config": config,
            "_cache": {},
            "_cache_order": [],
        }

    def _load_shard(self, modality: dict, shard_idx: int):
        """Load a shard with per-modality LRU caching."""
        cache = modality["_cache"]
        cache_order = modality["_cache_order"]

        if shard_idx in cache:
            cache_order.remove(shard_idx)
            cache_order.append(shard_idx)
            return cache[shard_idx]

        shard_path = os.path.join(modality["shard_dir"], modality["shard_files"][shard_idx])
        shard = torch.load(shard_path, map_location="cpu", weights_only=True, mmap=True)

        if len(cache) >= self.cache_size:
            oldest = cache_order.pop(0)
            del cache[oldest]

        cache[shard_idx] = shard
        cache_order.append(shard_idx)
        return shard

    def _find_shard_for_idx(self, modality: dict, idx: int):
        """Binary search to find which shard contains the given index."""
        shard_idx = bisect.bisect_right(modality["shard_offsets"], idx) - 1
        local_idx = idx - modality["shard_offsets"][shard_idx]
        return shard_idx, local_idx

    def _source_for_task(self, modality_name: str, direction: Optional[str]) -> dict:
        """Resolve the shard source a (modality, direction) task draws from.

        Defaults to the modality's canonical corpus; returns a per-direction
        override source when one was registered (e.g. a clean voice subset for
        the synthesis direction).
        """
        return self._direction_sources.get((modality_name, direction), self.modalities[modality_name])

    def _get_text_sample(self, idx: int, source: dict = None) -> dict:
        mod = source if source is not None else self.modalities["text"]
        wrapped_idx = idx % mod["total_samples"]
        shard_idx, local_idx = self._find_shard_for_idx(mod, wrapped_idx)
        shard = self._load_shard(mod, shard_idx)

        sample = {
            "token_ids": shard["token_ids"][local_idx],
            "text_length": shard["text_lengths"][local_idx],
        }
        if "text" in shard:
            sample["text"] = shard["text"][local_idx]
        elif "raw_text" in shard:
            sample["text"] = shard["raw_text"][local_idx]

        return sample

    def _get_audio_sample(self, idx: int, source: dict = None) -> dict:
        mod = source if source is not None else self.modalities["audio"]
        wrapped_idx = idx % mod["total_samples"]
        shard_idx, local_idx = self._find_shard_for_idx(mod, wrapped_idx)
        shard = self._load_shard(mod, shard_idx)
        columns = self.audio_columns

        sample = {}

        if "features" in shard and "features" in columns:
            sample["features"] = shard["features"][local_idx]
            sample["feature_length"] = shard["feature_lengths"][local_idx]

        if "waveforms" in shard and "waveforms" in columns:
            sample["waveform"] = shard["waveforms"][local_idx]
            sample["waveform_length"] = shard["waveform_lengths"][local_idx]

        if "mel_specs" in shard and "mel_specs" in columns:
            sample["mel_spec"] = shard["mel_specs"][local_idx]
            sample["mel_length"] = shard["mel_lengths"][local_idx]

        if "speaker_embeddings" in shard and "speaker_embeddings" in columns:
            sample["speaker_embedding"] = shard["speaker_embeddings"][local_idx]

        if "speaker_ids" in shard and "speaker_ids" in columns:
            sample["speaker_id"] = shard["speaker_ids"][local_idx]

        if "f0" in shard and "f0" in columns:
            sample["f0"] = shard["f0"][local_idx]

        if "vuv" in shard and "vuv" in columns:
            sample["vuv"] = shard["vuv"][local_idx]

        if "ctc_tokens" in shard and "ctc_tokens" in columns:
            sample["ctc_tokens"] = shard["ctc_tokens"][local_idx]
            sample["ctc_length"] = shard["ctc_lengths"][local_idx]

        if "text" in shard and "text" in columns:
            sample["audio_text"] = shard["text"][local_idx]

        return sample

    def _get_voice_sample(self, idx: int, source: dict = None) -> dict:
        mod = source if source is not None else self.modalities["voice"]
        wrapped_idx = idx % mod["total_samples"]
        shard_idx, local_idx = self._find_shard_for_idx(mod, wrapped_idx)
        shard = self._load_shard(mod, shard_idx)
        columns = self.voice_columns

        sample = {}

        if "features" in shard and "features" in columns:
            sample["features"] = shard["features"][local_idx]
            sample["feature_length"] = shard["feature_lengths"][local_idx]

        if "waveforms" in shard and "waveforms" in columns:
            sample["waveform"] = shard["waveforms"][local_idx]
            sample["waveform_length"] = shard["waveform_lengths"][local_idx]

        if "mel_specs" in shard and "mel_specs" in columns:
            sample["mel_spec"] = shard["mel_specs"][local_idx]
            sample["mel_length"] = shard["mel_lengths"][local_idx]

        if "speaker_embeddings" in shard and "speaker_embeddings" in columns:
            sample["speaker_embedding"] = shard["speaker_embeddings"][local_idx]

        if "speaker_ids" in shard and "speaker_ids" in columns:
            sample["speaker_id"] = shard["speaker_ids"][local_idx]

        if "f0" in shard and "f0" in columns:
            sample["f0"] = shard["f0"][local_idx]

        if "vuv" in shard and "vuv" in columns:
            sample["vuv"] = shard["vuv"][local_idx]

        if "ctc_tokens" in shard and "ctc_tokens" in columns:
            sample["ctc_tokens"] = shard["ctc_tokens"][local_idx]
            sample["ctc_length"] = shard["ctc_lengths"][local_idx]

        if "text" in shard and "text" in columns:
            sample["voice_text"] = shard["text"][local_idx]

        # Extract transcript token IDs if available
        if "token_ids" in shard:
            sample["token_ids"] = shard["token_ids"][local_idx]
            sample["text_length"] = shard["text_lengths"][local_idx]

        return sample

    def _get_image_sample(self, idx: int, source: dict = None) -> dict:
        mod = source if source is not None else self.modalities["image"]
        wrapped_idx = idx % mod["total_samples"]
        shard_idx, local_idx = self._find_shard_for_idx(mod, wrapped_idx)
        shard = self._load_shard(mod, shard_idx)

        sample = {}
        # Support both raw image shards ("images") and precomputed latent shards ("latents")
        if "latents" in shard:
            sample["image"] = shard["latents"][local_idx]
        elif "images" in shard:
            sample["image"] = shard["images"][local_idx]

        # Extract caption text and token IDs if available
        if "token_ids" in shard:
            sample["token_ids"] = shard["token_ids"][local_idx]
            sample["text_length"] = shard["text_lengths"][local_idx]
        if "text" in shard:
            sample["text"] = shard["text"][local_idx]

        return sample

    def __len__(self):
        return self.total_samples

    def _all_task_types(self) -> list:
        """Unfiltered (task_name, modality, direction) list for configured modalities.

        Each media modality produces two tasks (synthesis + transcription); text-only
        (text_continuation) is always present.
        """
        tasks = [("text_continuation", "text", None)]
        for mod in self.modalities:
            if mod == "text":
                continue
            tasks.append((f"{mod}_synthesis", mod, "synthesis"))
            tasks.append((f"{mod}_transcription", mod, "transcription"))
        return tasks

    @property
    def task_types(self) -> list:
        """Active task types, filtered by include_tasks (None = all).

        Single source of truth for the round-robin (__getitem__), dataset length,
        sampler grouping, and eval task grouping — filtering here propagates to all.
        Returns list of (task_name, modality, direction).
        """
        tasks = self._all_task_types()
        if self._include_tasks is not None:
            tasks = [t for t in tasks if t[0] in self._include_tasks]
        return tasks

    def __getitem__(self, idx: int) -> dict:
        """Return a task-specific sample with paired text.

        Indices are distributed round-robin across task types:
        text_continuation, voice_synthesis, voice_transcription,
        image_synthesis, image_transcription, etc.

        Each sample includes a _task field indicating the task type and
        a _direction field for the collator to use.
        """
        tasks = self.task_types
        n_tasks = len(tasks)
        task_idx = idx % n_tasks
        task_name, modality_name, direction = tasks[task_idx]
        within_task_idx = idx // n_tasks

        sample = {"_modality": modality_name, "_task": task_name, "_direction": direction}
        source = self._source_for_task(modality_name, direction)

        if modality_name == "text":
            text_sample = self._get_text_sample(within_task_idx, source)
            sample.update({f"text_{k}": v for k, v in text_sample.items()})

        elif modality_name == "audio":
            audio_sample = self._get_audio_sample(within_task_idx, source)
            sample.update({f"audio_{k}": v for k, v in audio_sample.items()})
            if "token_ids" in audio_sample:
                sample["text_token_ids"] = audio_sample["token_ids"]
                sample["text_text_length"] = audio_sample["text_length"]

        elif modality_name == "voice":
            voice_sample = self._get_voice_sample(within_task_idx, source)
            sample.update({f"voice_{k}": v for k, v in voice_sample.items()})
            if "token_ids" in voice_sample:
                sample["text_token_ids"] = voice_sample["token_ids"]
                sample["text_text_length"] = voice_sample["text_length"]
            if "voice_text" in voice_sample:
                sample["text_text"] = voice_sample["voice_text"]

        elif modality_name == "image":
            image_sample = self._get_image_sample(within_task_idx, source)
            sample.update({f"image_{k}": v for k, v in image_sample.items()})
            if "token_ids" in image_sample:
                sample["text_token_ids"] = image_sample["token_ids"]
                sample["text_text_length"] = image_sample["text_length"]
            if "text" in image_sample:
                sample["text_text"] = image_sample["text"]

        return sample

    def get_sampler(self, shuffle: bool = True, seed: int = 42,
                    batch_size: int = 1, world_size: int = 1,
                    shard_aware: bool = True):
        """
        Get a task-grouped sampler that yields indices such that each
        batch contains only one task type (e.g., all voice_synthesis or
        all image_transcription).

        With round-robin task assignment (idx % n_tasks), indices for
        each task are strided. This sampler groups them into chunks
        aligned to (world_size × batch_size) so all DDP/DeepSpeed ranks
        receive the same task type at each step.

        When shard_aware=True, within-modality shuffling groups samples
        by shard before permuting, so consecutive samples in an epoch
        hit the same shard-cache entry — dramatically reducing LRU
        cache misses vs. the prior uniform-random shuffle. Changes the
        sampling order, so pass False to preserve legacy ordering when
        resuming a checkpoint from a pre-shard-aware run.
        """
        shard_info = None
        if shard_aware:
            shard_info = []
            for task_name, modality_name, direction in self.task_types:
                src = self._source_for_task(modality_name, direction)
                shard_info.append((src["shard_offsets"], src["total_samples"]))
        return ModalityGroupedSampler(
            total_samples=self.total_samples,
            n_modalities=len(self.task_types),
            shuffle=shuffle,
            seed=seed,
            batch_size=batch_size,
            world_size=world_size,
            shard_info=shard_info,
        )


class ModalityGroupedSampler(torch.utils.data.Sampler):
    """Yields indices grouped by modality for homogeneous batches.

    Given round-robin modality assignment (idx % n_modalities), this sampler
    groups all indices of each modality together. Groups are interleaved in
    chunks that align with (world_size × batch_size) so that all DDP/DeepSpeed
    ranks receive the same modality type at each training step.

    Args:
        total_samples: Total dataset length.
        n_modalities: Number of modality types (from dataset).
        shuffle: Whether to shuffle within modality groups.
        seed: Random seed for reproducibility across ranks.
        batch_size: Per-device batch size (for alignment).
        world_size: Number of distributed processes (for alignment).
        shard_info: Optional list parallel to tasks (length n_modalities).
            Each entry is (shard_offsets, modality_total_samples) for the
            modality that task belongs to. When provided together with
            shuffle=True, within-modality permutation groups samples by
            shard and shuffles shard order, minimizing LRU cache misses.
            When None (default), falls back to uniform random shuffle.
    """

    def __init__(self, total_samples: int, n_modalities: int, shuffle: bool = True,
                 seed: int = 42, batch_size: int = 1, world_size: int = 1,
                 shard_info: Optional[list] = None):
        self.total_samples = total_samples
        self.n_modalities = n_modalities
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size
        self.world_size = world_size
        self.shard_info = shard_info
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Build per-modality index lists
        modality_indices = [[] for _ in range(self.n_modalities)]
        for idx in range(self.total_samples):
            mod = idx % self.n_modalities
            modality_indices[mod].append(idx)

        # Order within each modality group
        if self.shard_info is not None and self.shuffle:
            # Shard-aware shuffle: group indices by their owning shard,
            # shuffle within each shard, then shuffle shard order. Samples
            # from the same shard arrive in a contiguous run, so the per-
            # modality LRU cache only misses at shard boundaries rather
            # than probabilistically within every batch.
            for i in range(len(modality_indices)):
                shard_offsets, mod_total = self.shard_info[i]
                shard_groups: dict[int, list[int]] = {}
                for global_idx in modality_indices[i]:
                    within_task_idx = global_idx // self.n_modalities
                    wrapped = within_task_idx % mod_total
                    shard_idx = bisect.bisect_right(shard_offsets, wrapped) - 1
                    shard_groups.setdefault(shard_idx, []).append(global_idx)
                shard_keys = list(shard_groups.keys())
                for sk in shard_keys:
                    group = shard_groups[sk]
                    perm = torch.randperm(len(group), generator=g).tolist()
                    shard_groups[sk] = [group[p] for p in perm]
                sperm = torch.randperm(len(shard_keys), generator=g).tolist()
                shard_keys = [shard_keys[p] for p in sperm]
                modality_indices[i] = [idx for sk in shard_keys for idx in shard_groups[sk]]
        elif self.shuffle:
            # Legacy path: uniform random shuffle within each modality.
            # Preserved byte-identically for resume compatibility.
            for i in range(len(modality_indices)):
                perm = torch.randperm(len(modality_indices[i]), generator=g).tolist()
                modality_indices[i] = [modality_indices[i][p] for p in perm]

        # Chunk size = world_size * batch_size ensures all ranks get the
        # same modality at each step. We interleave modality chunks so
        # training alternates modalities within each epoch.
        chunk_size = max(self.world_size * self.batch_size, 1)

        # Split each modality into chunks
        chunked = []
        for m in range(self.n_modalities):
            indices = modality_indices[m]
            for start in range(0, len(indices), chunk_size):
                chunk = indices[start:start + chunk_size]
                # Pad last chunk to full size so ranks stay aligned
                while len(chunk) < chunk_size and indices:
                    chunk.append(indices[len(chunk) % len(indices)])
                chunked.append(chunk)

        # Shuffle chunk order (all ranks use same seed, same order)
        if self.shuffle:
            perm = torch.randperm(len(chunked), generator=g).tolist()
            chunked = [chunked[p] for p in perm]

        # Flatten
        all_indices = []
        for chunk in chunked:
            all_indices.extend(chunk)

        # Trim to total_samples
        all_indices = all_indices[:self.total_samples]

        return iter(all_indices)
