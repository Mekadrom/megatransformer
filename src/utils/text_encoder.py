"""
Centralized text encoder utility for extracting text embeddings.

This module provides a singleton pattern for loading and caching text encoder models,
ensuring only one instance is loaded at a time to save GPU memory.

Supports T5 family of models:
- t5_small: 512-dim embeddings, fastest
- t5_base: 768-dim embeddings, balanced
- t5_large: 1024-dim embeddings, higher quality
- t5_3b: 1024-dim embeddings, highest quality (memory intensive)

Usage:
    from utils.text_encoder import get_text_encoder, extract_text_embedding

    # Get cached encoder instance
    encoder = get_text_encoder(encoder_type="t5_small", device="cuda")

    # Extract embedding from text
    embedding = extract_text_embedding(text="Hello world", encoder=encoder)

    # Extract embeddings from batch of texts
    embeddings = extract_text_embedding(text=["Hello", "World"], encoder=encoder)
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Literal, List
from threading import Lock


# Global cache for text encoder singleton
_text_encoder_cache = {
    "model": None,
    "model_type": None,
    "device": None,
}
_cache_lock = Lock()


# Supported text encoder types
TextEncoderType = Literal["t5_small", "t5_base", "t5_large", "t5_3b"]

# Model names on HuggingFace
TEXT_ENCODER_MODEL_NAMES = {
    "t5_small": "google-t5/t5-small",
    "t5_base": "google-t5/t5-base",
    "t5_large": "google-t5/t5-large",
    "t5_3b": "google-t5/t5-3b",
}

# Embedding dimensions for each encoder type
TEXT_EMBEDDING_DIMS = {
    "t5_small": 512,
    "t5_base": 768,
    "t5_large": 1024,
    "t5_3b": 1024,
}

# Default encoder type
DEFAULT_ENCODER_TYPE = "t5_small"


class TextEncoderWrapper(nn.Module):
    """
    Unified wrapper for T5 text encoder models.

    Provides a consistent API for extracting text embeddings:
    - Input: text string or list of strings
    - Output: text embedding [B, embedding_dim] (mean pooled) or [B, T, embedding_dim] (full sequence)
    """

    def __init__(
        self,
        encoder_type: TextEncoderType = DEFAULT_ENCODER_TYPE,
        device: Union[str, torch.device] = "cpu",
        max_length: int = 512,
        pooling: Literal["mean", "last", "first", "none"] = "none",
    ):
        """
        Initialize the text encoder wrapper.

        Args:
            encoder_type: Type of T5 model to use
            device: Device to load the model on
            max_length: Maximum sequence length for tokenization
            pooling: Pooling strategy for sequence embeddings
                - "mean": Mean pool over all tokens (default)
                - "last": Use last token embedding
                - "first": Use first token embedding (CLS-like)
                - "none": Return full sequence [B, T, D]
        """
        super().__init__()
        self.encoder_type = encoder_type
        self.device = torch.device(device)
        self.embedding_dim = TEXT_EMBEDDING_DIMS[encoder_type]
        self.max_length = max_length
        self.pooling = pooling

        self._load_t5()

    def _load_t5(self):
        """Load T5 encoder from HuggingFace."""
        try:
            from transformers import T5EncoderModel, T5Tokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for T5 text encoder. "
                "Install with: pip install transformers"
            )

        model_name = TEXT_ENCODER_MODEL_NAMES[self.encoder_type]
        print(f"Loading {self.encoder_type} text encoder on {self.device}...")

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.encoder.to(self.device)
        self.encoder.eval()

        # Freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        print(f"{self.encoder_type} text encoder loaded (embedding_dim={self.embedding_dim})")

    @torch.no_grad()
    def forward(
        self,
        text: Union[str, List[str]],
        return_attention_mask: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        Extract text embeddings from input text.

        Args:
            text: Input text string or list of strings
            return_attention_mask: If True, also return the attention mask

        Returns:
            If pooling != "none":
                Text embeddings [B, embedding_dim]
            If pooling == "none":
                Text embeddings [B, T, embedding_dim]
            If return_attention_mask:
                Tuple of (embeddings, attention_mask)
        """
        # Handle single string input
        if isinstance(text, str):
            text = [text]
            was_single = True
        else:
            was_single = False

        # Tokenize
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Forward through encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Get hidden states [B, T, D]
        hidden_states = outputs.last_hidden_state

        # Apply pooling
        if self.pooling == "mean":
            # Masked mean pooling
            mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
            embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        elif self.pooling == "last":
            # Get last non-padding token for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
            batch_indices = torch.arange(hidden_states.size(0), device=self.device)
            embeddings = hidden_states[batch_indices, seq_lengths]
        elif self.pooling == "first":
            # Use first token (index 0)
            embeddings = hidden_states[:, 0, :]
        elif self.pooling == "none":
            embeddings = hidden_states
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")

        # Squeeze if single input
        if was_single and self.pooling != "none":
            embeddings = embeddings.squeeze(0)

        if return_attention_mask:
            return embeddings, attention_mask
        return embeddings

    def encode(
        self,
        text: Union[str, List[str]],
        return_attention_mask: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """Alias for forward() for API consistency."""
        return self.forward(text, return_attention_mask=return_attention_mask)

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension for this encoder."""
        return self.embedding_dim

    def get_vocab_size(self) -> int:
        """Return the vocabulary size of the tokenizer."""
        return self.tokenizer.vocab_size


def get_text_encoder(
    encoder_type: TextEncoderType = DEFAULT_ENCODER_TYPE,
    device: Union[str, torch.device] = "cpu",
    force_reload: bool = False,
    **kwargs,
) -> TextEncoderWrapper:
    """
    Get a cached text encoder instance (singleton pattern).

    This ensures only one text encoder is loaded at a time, saving GPU memory.
    If the requested encoder type or device differs from the cached one, the cache
    is invalidated and a new encoder is loaded.

    Args:
        encoder_type: Type of text encoder to use
        device: Device to load the encoder on
        force_reload: If True, reload the encoder even if cached
        **kwargs: Additional arguments passed to TextEncoderWrapper (e.g., max_length, pooling)

    Returns:
        Cached TextEncoderWrapper instance
    """
    global _text_encoder_cache

    device = torch.device(device)

    with _cache_lock:
        # Check if we need to load/reload
        needs_reload = (
            force_reload
            or _text_encoder_cache["model"] is None
            or _text_encoder_cache["model_type"] != encoder_type
            or _text_encoder_cache["device"] != device
        )

        if needs_reload:
            # Clear old model to free memory
            if _text_encoder_cache["model"] is not None:
                del _text_encoder_cache["model"]
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Load new encoder
            _text_encoder_cache["model"] = TextEncoderWrapper(
                encoder_type=encoder_type,
                device=device,
                **kwargs,
            )
            _text_encoder_cache["model_type"] = encoder_type
            _text_encoder_cache["device"] = device

        return _text_encoder_cache["model"]


def clear_text_encoder_cache():
    """
    Clear the text encoder cache and free GPU memory.

    Call this when you're done with text encoding and want to reclaim memory.
    """
    global _text_encoder_cache

    with _cache_lock:
        if _text_encoder_cache["model"] is not None:
            del _text_encoder_cache["model"]
            _text_encoder_cache["model"] = None
            _text_encoder_cache["model_type"] = None
            _text_encoder_cache["device"] = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print("Text encoder cache cleared")


def extract_text_embedding(
    text: Union[str, List[str]],
    encoder: Optional[TextEncoderWrapper] = None,
    device: Union[str, torch.device] = "cpu",
    encoder_type: TextEncoderType = DEFAULT_ENCODER_TYPE,
    return_attention_mask: bool = False,
    **kwargs,
) -> Union[torch.Tensor, tuple]:
    """
    Extract text embedding from input text.

    Convenience function that handles encoder loading automatically.

    Args:
        text: Input text string or list of strings
        encoder: Optional pre-loaded encoder (if None, uses cached singleton)
        device: Device for encoder (only used if encoder is None)
        encoder_type: Type of encoder (only used if encoder is None)
        return_attention_mask: If True, also return attention mask
        **kwargs: Additional arguments passed to encoder (e.g., max_length, pooling)

    Returns:
        Text embedding [B, embedding_dim] or [embedding_dim] if single string input
        If return_attention_mask: tuple of (embeddings, attention_mask)
    """
    if encoder is None:
        encoder = get_text_encoder(encoder_type=encoder_type, device=device, **kwargs)

    return encoder(text, return_attention_mask=return_attention_mask)


def get_text_embedding_dim(encoder_type: TextEncoderType = DEFAULT_ENCODER_TYPE) -> int:
    """Get the embedding dimension for a text encoder type without loading the model."""
    return TEXT_EMBEDDING_DIMS[encoder_type]
