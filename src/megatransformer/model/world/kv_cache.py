"""
KV Cache implementations for efficient autoregressive generation.

Two strategies:
1. HuginnKVCache: Shared cache across iterations with circular buffer (default, efficient)
2. PerIterationKVCache: Separate cache per iteration (for research, more memory)

The Huginn approach (Remark 6.1, Section 6.2 of the paper) exploits the fact that
all recurrent iterations use the same K,V projection matrices. This means KV entries
from different iterations "match" well enough that the model can zero-shot adapt
to attending to mixed-depth KVs.
"""

from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass, field

import torch


@dataclass
class KVCache:
    """
    Standard KV cache for a single attention layer.

    Stores key and value tensors that can be incrementally updated
    during autoregressive generation.

    Shape conventions:
    - key_cache: (batch_size, n_kv_heads, seq_len, d_queries)
    - value_cache: (batch_size, n_kv_heads, seq_len, d_values)
    """
    key_cache: Optional[torch.Tensor] = None
    value_cache: Optional[torch.Tensor] = None

    @property
    def seq_len(self) -> int:
        """Current cached sequence length."""
        if self.key_cache is None:
            return 0
        return self.key_cache.shape[2]

    def update(
        self,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new keys and values to the cache.

        Args:
            new_keys: New keys, shape (batch, n_kv_heads, new_seq, d_queries)
            new_values: New values, shape (batch, n_kv_heads, new_seq, d_values)

        Returns:
            Full key and value tensors including the new tokens.
        """
        if self.key_cache is None:
            self.key_cache = new_keys
            self.value_cache = new_values
        else:
            self.key_cache = torch.cat([self.key_cache, new_keys], dim=2)
            self.value_cache = torch.cat([self.value_cache, new_values], dim=2)

        return self.key_cache, self.value_cache

    def clear(self):
        """Clear the cache."""
        self.key_cache = None
        self.value_cache = None


@dataclass
class LayerKVCache:
    """KV cache for all layers in a transformer stack."""
    layer_caches: Dict[int, KVCache] = field(default_factory=dict)

    def get_layer(self, layer_idx: int) -> KVCache:
        """Get cache for a specific layer, creating if needed."""
        if layer_idx not in self.layer_caches:
            self.layer_caches[layer_idx] = KVCache()
        return self.layer_caches[layer_idx]

    @property
    def seq_len(self) -> int:
        """Current cached sequence length."""
        if not self.layer_caches:
            return 0
        return next(iter(self.layer_caches.values())).seq_len

    def clear(self):
        """Clear all layer caches."""
        for cache in self.layer_caches.values():
            cache.clear()
        self.layer_caches.clear()


# =============================================================================
# Huginn-style Shared KV Cache (Default, Efficient)
# =============================================================================

@dataclass
class HuginnKVCache:
    """
    Huginn-style shared KV cache with circular buffer for recurrent blocks.

    From the Huginn paper (Section 6.2):
    - Uses a fixed KV-cache budget (e.g., 16 slots)
    - At iteration i, reads/writes to slot (i % budget)
    - Tokens at different positions may have KVs from different iteration depths
    - The model zero-shot adapts to this mixed-depth attention

    This dramatically reduces memory while maintaining performance.
    With budget=4, MTBench score remains at 5.86.

    Structure:
    - slot_caches[slot_idx] = LayerKVCache
    - At iteration i, use slot_caches[i % cache_budget]
    """
    cache_budget: int = 16  # Number of circular buffer slots
    slot_caches: Dict[int, LayerKVCache] = field(default_factory=dict)

    def get_slot(self, iteration: int) -> LayerKVCache:
        """Get the cache slot for a given iteration (circular buffer)."""
        slot_idx = iteration % self.cache_budget
        if slot_idx not in self.slot_caches:
            self.slot_caches[slot_idx] = LayerKVCache()
        return self.slot_caches[slot_idx]

    def get_layer_at_iteration(self, iteration: int, layer_idx: int) -> KVCache:
        """Get cache for a specific layer at a specific iteration."""
        return self.get_slot(iteration).get_layer(layer_idx)

    @property
    def seq_len(self) -> int:
        """Current cached sequence length (from any slot, all should match)."""
        if not self.slot_caches:
            return 0
        return next(iter(self.slot_caches.values())).seq_len

    def clear(self):
        """Clear all slot caches."""
        for cache in self.slot_caches.values():
            cache.clear()
        self.slot_caches.clear()


# =============================================================================
# Per-Iteration KV Cache (For Research, More Memory)
# =============================================================================

@dataclass
class PerIterationKVCache:
    """
    Per-iteration KV cache for recurrent transformer blocks.

    More accurate than Huginn-style but uses more memory.
    Each iteration has its own complete KV cache.

    Structure:
    - iteration_caches[iteration_idx] = LayerKVCache
    """
    max_iterations: int = 64
    iteration_caches: Dict[int, LayerKVCache] = field(default_factory=dict)

    def get_iteration(self, iteration_idx: int) -> LayerKVCache:
        """Get cache for a specific iteration, creating if needed."""
        if iteration_idx not in self.iteration_caches:
            self.iteration_caches[iteration_idx] = LayerKVCache()
        return self.iteration_caches[iteration_idx]

    def get_layer_at_iteration(self, iteration_idx: int, layer_idx: int) -> KVCache:
        """Get cache for a specific layer at a specific iteration."""
        return self.get_iteration(iteration_idx).get_layer(layer_idx)

    @property
    def seq_len(self) -> int:
        """Current cached sequence length."""
        if not self.iteration_caches:
            return 0
        return next(iter(self.iteration_caches.values())).seq_len

    def clear(self):
        """Clear all iteration caches."""
        for cache in self.iteration_caches.values():
            cache.clear()
        self.iteration_caches.clear()


# =============================================================================
# Unified Recurrent KV Cache (Supports Both Strategies)
# =============================================================================

@dataclass
class RecurrentKVCache:
    """
    Unified KV cache for recurrent blocks supporting both strategies.

    Args:
        strategy: "huginn" (default, shared cache) or "per_iteration" (separate caches)
        cache_budget: For huginn strategy, number of circular buffer slots (default 16)
        max_iterations: For per_iteration strategy, maximum iterations to cache
    """
    strategy: str = "huginn"
    cache_budget: int = 16
    max_iterations: int = 64

    _huginn_cache: Optional[HuginnKVCache] = field(default=None, init=False)
    _per_iter_cache: Optional[PerIterationKVCache] = field(default=None, init=False)

    def __post_init__(self):
        if self.strategy == "huginn":
            self._huginn_cache = HuginnKVCache(cache_budget=self.cache_budget)
        elif self.strategy == "per_iteration":
            self._per_iter_cache = PerIterationKVCache(max_iterations=self.max_iterations)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}. Use 'huginn' or 'per_iteration'.")

    def get_layer_at_iteration(self, iteration: int, layer_idx: int) -> KVCache:
        """Get cache for a specific layer at a specific iteration."""
        if self.strategy == "huginn":
            return self._huginn_cache.get_layer_at_iteration(iteration, layer_idx)
        else:
            return self._per_iter_cache.get_layer_at_iteration(iteration, layer_idx)

    @property
    def seq_len(self) -> int:
        """Current cached sequence length."""
        if self.strategy == "huginn":
            return self._huginn_cache.seq_len
        else:
            return self._per_iter_cache.seq_len

    def clear(self):
        """Clear the cache."""
        if self.strategy == "huginn":
            self._huginn_cache.clear()
        else:
            self._per_iter_cache.clear()


# =============================================================================
# Complete World Model KV Cache
# =============================================================================

@dataclass
class WorldModelKVCache:
    """
    Complete KV cache for the world model during generation.

    Contains:
    - recurrent_cache: For the recurrent block
    - position_offset: Current position for RoPE
    - generation_state: Track modality boundaries
    """
    strategy: str = "huginn"
    cache_budget: int = 16

    recurrent_cache: RecurrentKVCache = field(default=None)
    position_offset: int = 0

    # Generation state tracking
    generated_tokens: List[int] = field(default_factory=list)
    in_audio_block: bool = False
    in_voice_block: bool = False
    in_image_block: bool = False
    pending_audio_hidden: Optional[torch.Tensor] = None
    pending_voice_hidden: Optional[torch.Tensor] = None
    pending_image_hidden: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.recurrent_cache is None:
            self.recurrent_cache = RecurrentKVCache(
                strategy=self.strategy,
                cache_budget=self.cache_budget,
            )

    def clear(self):
        """Clear all caches and reset state."""
        self.recurrent_cache.clear()
        self.position_offset = 0
        self.generated_tokens.clear()
        self.in_audio_block = False
        self.in_voice_block = False
        self.in_image_block = False
        self.pending_audio_hidden = None
        self.pending_voice_hidden = None
        self.pending_image_hidden = None

    def update_position(self, num_new_tokens: int):
        """Update position offset after generating new tokens."""
        self.position_offset += num_new_tokens
