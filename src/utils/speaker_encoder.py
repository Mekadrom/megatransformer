"""
Centralized speaker encoder utility for extracting speaker embeddings.

This module provides a singleton pattern for loading and caching speaker encoder models,
ensuring only one instance is loaded at a time to save GPU memory.

Supports multiple speaker encoder backends:
- ECAPA-TDNN (speechbrain): 192-dim embeddings, trained for speaker verification (uses mel spectrograms)
- WavLM-base: 768-dim embeddings, richer acoustic features (uses waveforms)
- (Future) CAM++: 512-dim embeddings, improved verification model

Usage:
    from utils.speaker_encoder import get_speaker_encoder, extract_speaker_embedding

    # Get cached encoder instance
    encoder = get_speaker_encoder(encoder_type="wavlm", device="cuda")

    # Extract embedding from waveform (WavLM) or mel spectrogram (ECAPA-TDNN)
    embedding = extract_speaker_embedding(waveform=waveform, encoder=encoder)
    embedding = extract_speaker_embedding(mel_spec=mel_spec, encoder=encoder)
"""

import torch
import torch.nn as nn


from typing import Optional, Union, Literal
from threading import Lock


# Global cache for speaker encoder singleton
_speaker_encoder_cache = {
    "model": None,
    "model_type": None,
    "device": None,
}
_cache_lock = Lock()


# Supported speaker encoder types
SpeakerEncoderType = Literal["ecapa_tdnn", "wavlm", "cam++"]

# Embedding dimensions for each encoder type
SPEAKER_EMBEDDING_DIMS = {
    "ecapa_tdnn": 192,
    "wavlm": 768,
    "cam++": 512,  # Future
}

# Input types for each encoder
SPEAKER_ENCODER_INPUT_TYPES = {
    "ecapa_tdnn": "mel",  # Expects mel spectrograms
    "wavlm": "waveform",  # Expects raw waveforms
    "cam++": "waveform",  # Future - likely waveform
}

# Default encoder type
DEFAULT_ENCODER_TYPE = "ecapa_tdnn"


class SpeakerEncoderWrapper(nn.Module):
    """
    Unified wrapper for different speaker encoder backends.

    Provides a consistent API regardless of the underlying model:
    - For ECAPA-TDNN: Input mel spectrogram [B, n_mels, T] or [B, 1, n_mels, T]
    - For WavLM: Input waveform [B, T] or [T]
    - Output: speaker embedding [B, embedding_dim]
    """

    def __init__(
        self,
        encoder_type: SpeakerEncoderType = DEFAULT_ENCODER_TYPE,
        device: Union[str, torch.device] = "cpu",
        wavlm_layer: int = -1,  # Which layer to use for WavLM (-1 = last)
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.device = torch.device(device)
        self.embedding_dim = SPEAKER_EMBEDDING_DIMS[encoder_type]
        self.input_type = SPEAKER_ENCODER_INPUT_TYPES[encoder_type]
        self.wavlm_layer = wavlm_layer

        if encoder_type == "ecapa_tdnn":
            self._load_ecapa_tdnn()
        elif encoder_type == "wavlm":
            self._load_wavlm()
        elif encoder_type == "cam++":
            raise NotImplementedError("CAM++ speaker encoder not yet implemented")
        else:
            raise ValueError(f"Unknown speaker encoder type: {encoder_type}")

    def _load_ecapa_tdnn(self):
        """Load ECAPA-TDNN from SpeechBrain."""
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            raise ImportError(
                "speechbrain is required for ECAPA-TDNN speaker encoder. "
                "Install with: pip install speechbrain"
            )

        print(f"Loading ECAPA-TDNN speaker encoder on {self.device}...")
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb-mel-spec",
            savedir="pretrained_models/spkrec-ecapa-voxceleb-mel-spec",
            run_opts={"device": str(self.device)},
        )
        print(f"ECAPA-TDNN speaker encoder loaded (embedding_dim={self.embedding_dim})")

    def _load_wavlm(self):
        """Load WavLM-base from HuggingFace."""
        try:
            from transformers import WavLMModel, Wav2Vec2FeatureExtractor
        except ImportError:
            raise ImportError(
                "transformers is required for WavLM speaker encoder. "
                "Install with: pip install transformers"
            )

        print(f"Loading WavLM-base speaker encoder on {self.device}...")
        self.encoder = WavLMModel.from_pretrained("microsoft/wavlm-base")
        self.encoder.to(self.device)
        self.encoder.eval()

        # Feature extractor for preprocessing (normalization)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")

        # WavLM-base has 12 transformer layers + input embedding
        # Layer -1 = last layer output, layer 0 = after feature projection
        print(f"WavLM-base speaker encoder loaded (embedding_dim={self.embedding_dim}, layer={self.wavlm_layer})")

    @torch.no_grad()
    def forward(
        self,
        mel_spec: Optional[torch.Tensor] = None,
        waveform: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """
        Extract speaker embeddings from audio input.

        Args:
            mel_spec: Mel spectrogram tensor (for ECAPA-TDNN)
                - Shape [B, n_mels, T] or [B, 1, n_mels, T] or [n_mels, T]
            waveform: Waveform tensor (for WavLM)
                - Shape [B, T] or [T]
            lengths: Optional tensor of actual lengths (before padding)
                - Shape [B] with values in range [1, T]
                - If None, assumes full length for all samples
            sample_rate: Sample rate of waveform (for WavLM, must be 16000)

        Returns:
            Speaker embeddings [B, embedding_dim]
        """
        if self.encoder_type == "ecapa_tdnn":
            if mel_spec is None:
                raise ValueError("ECAPA-TDNN requires mel_spec input")
            return self._forward_ecapa_tdnn(mel_spec, lengths)
        elif self.encoder_type == "wavlm":
            if waveform is None:
                raise ValueError("WavLM requires waveform input")
            return self._forward_wavlm(waveform, lengths, sample_rate)
        else:
            raise NotImplementedError(f"Forward not implemented for {self.encoder_type}")

    def _forward_ecapa_tdnn(
        self,
        mel_spec: torch.Tensor,
        lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """ECAPA-TDNN specific forward pass."""
        # Normalize input shape to [B, n_mels, T]
        if mel_spec.dim() == 2:
            mel_spec = mel_spec.unsqueeze(0)  # [n_mels, T] -> [1, n_mels, T]
        if mel_spec.dim() == 4:
            mel_spec = mel_spec.squeeze(1)  # [B, 1, n_mels, T] -> [B, n_mels, T]

        batch_size, n_mels, time_steps = mel_spec.shape

        # Move to encoder device if needed
        mel_spec = mel_spec.to(self.device)

        # ECAPA-TDNN expects [B, T, n_mels]
        mel_for_ecapa = mel_spec.transpose(1, 2)  # [B, n_mels, T] -> [B, T, n_mels]

        # Compute relative lengths (required by SpeechBrain)
        if lengths is not None:
            rel_lengths = lengths.float().to(self.device) / time_steps
        else:
            rel_lengths = torch.ones(batch_size, device=self.device)

        # Normalize and extract embeddings
        normalized = self.encoder.mods.normalizer(mel_for_ecapa, rel_lengths, epoch=1)
        embeddings = self.encoder.mods.embedding_model(normalized, rel_lengths)

        # Output shape is [B, 1, 192], squeeze to [B, 192]
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(1)

        return embeddings

    def _forward_wavlm(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor],
        sample_rate: int,
    ) -> torch.Tensor:
        """WavLM specific forward pass."""
        # Normalize input shape to [B, T]
        was_unbatched = waveform.dim() == 1
        if was_unbatched:
            waveform = waveform.unsqueeze(0)  # [T] -> [1, T]

        batch_size, num_samples = waveform.shape

        # WavLM expects 16kHz audio
        if sample_rate != 16000:
            raise ValueError(f"WavLM requires 16kHz audio, got {sample_rate}Hz")

        # Move to encoder device
        waveform = waveform.to(self.device)

        # Create attention mask based on lengths
        if lengths is not None:
            # lengths is in samples
            attention_mask = torch.zeros(batch_size, num_samples, device=self.device)
            for i, length in enumerate(lengths):
                attention_mask[i, :int(length)] = 1
        else:
            attention_mask = torch.ones(batch_size, num_samples, device=self.device)

        # Forward through WavLM
        outputs = self.encoder(
            waveform,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Get hidden states from specified layer
        if self.wavlm_layer == -1:
            hidden_states = outputs.last_hidden_state  # [B, T', 768]
        else:
            hidden_states = outputs.hidden_states[self.wavlm_layer]  # [B, T', 768]

        # Mean pooling over time (with attention mask)
        # hidden_states is [B, T', 768] where T' is downsampled time
        # We need to create a mask for the downsampled sequence
        if lengths is not None:
            # WavLM downsamples by factor of ~320 (20ms frames at 16kHz)
            downsample_factor = num_samples / hidden_states.shape[1]
            downsampled_lengths = (lengths.float() / downsample_factor).long().clamp(min=1)

            mask = torch.zeros(batch_size, hidden_states.shape[1], device=self.device)
            for i, length in enumerate(downsampled_lengths):
                mask[i, :length] = 1

            # Masked mean pooling
            mask = mask.unsqueeze(-1)  # [B, T', 1]
            embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            # Simple mean pooling
            embeddings = hidden_states.mean(dim=1)  # [B, 768]

        return embeddings

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension for this encoder."""
        return self.embedding_dim

    def get_input_type(self) -> str:
        """Return the expected input type ('mel' or 'waveform')."""
        return self.input_type


def get_speaker_encoder(
    encoder_type: SpeakerEncoderType = DEFAULT_ENCODER_TYPE,
    device: Union[str, torch.device] = "cpu",
    force_reload: bool = False,
    **kwargs,
) -> SpeakerEncoderWrapper:
    """
    Get a cached speaker encoder instance (singleton pattern).

    This ensures only one speaker encoder is loaded at a time, saving GPU memory.
    If the requested encoder type or device differs from the cached one, the cache
    is invalidated and a new encoder is loaded.

    Args:
        encoder_type: Type of speaker encoder to use
        device: Device to load the encoder on
        force_reload: If True, reload the encoder even if cached
        **kwargs: Additional arguments passed to SpeakerEncoderWrapper (e.g., wavlm_layer)

    Returns:
        Cached SpeakerEncoderWrapper instance
    """
    global _speaker_encoder_cache

    device = torch.device(device)

    with _cache_lock:
        # Check if we need to load/reload
        needs_reload = (
            force_reload
            or _speaker_encoder_cache["model"] is None
            or _speaker_encoder_cache["model_type"] != encoder_type
            or _speaker_encoder_cache["device"] != device
        )

        if needs_reload:
            # Clear old model to free memory
            if _speaker_encoder_cache["model"] is not None:
                del _speaker_encoder_cache["model"]
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Load new encoder
            _speaker_encoder_cache["model"] = SpeakerEncoderWrapper(
                encoder_type=encoder_type,
                device=device,
                **kwargs,
            )
            _speaker_encoder_cache["model_type"] = encoder_type
            _speaker_encoder_cache["device"] = device

        return _speaker_encoder_cache["model"]


def clear_speaker_encoder_cache():
    """
    Clear the speaker encoder cache and free GPU memory.

    Call this when you're done with speaker encoding and want to reclaim memory.
    """
    global _speaker_encoder_cache

    with _cache_lock:
        if _speaker_encoder_cache["model"] is not None:
            del _speaker_encoder_cache["model"]
            _speaker_encoder_cache["model"] = None
            _speaker_encoder_cache["model_type"] = None
            _speaker_encoder_cache["device"] = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print("Speaker encoder cache cleared")


def extract_speaker_embedding(
    mel_spec: Optional[torch.Tensor] = None,
    waveform: Optional[torch.Tensor] = None,
    encoder: Optional[SpeakerEncoderWrapper] = None,
    lengths: Optional[torch.Tensor] = None,
    sample_rate: int = 16000,
    device: Union[str, torch.device] = "cpu",
    encoder_type: SpeakerEncoderType = DEFAULT_ENCODER_TYPE,
) -> torch.Tensor:
    """
    Extract speaker embedding from audio input.

    Convenience function that handles encoder loading automatically.

    Args:
        mel_spec: Mel spectrogram tensor [B, n_mels, T] or [B, 1, n_mels, T] or [n_mels, T]
            (for ECAPA-TDNN)
        waveform: Waveform tensor [B, T] or [T] (for WavLM)
        encoder: Optional pre-loaded encoder (if None, uses cached singleton)
        lengths: Optional actual lengths [B] for variable-length inputs
        sample_rate: Sample rate of waveform (for WavLM)
        device: Device for encoder (only used if encoder is None)
        encoder_type: Type of encoder (only used if encoder is None)

    Returns:
        Speaker embedding [B, embedding_dim] or [embedding_dim] if input was unbatched
    """
    # Determine if input was unbatched
    if mel_spec is not None:
        was_unbatched = mel_spec.dim() == 2
    elif waveform is not None:
        was_unbatched = waveform.dim() == 1
    else:
        raise ValueError("Either mel_spec or waveform must be provided")

    if encoder is None:
        encoder = get_speaker_encoder(encoder_type=encoder_type, device=device)

    embedding = encoder(
        mel_spec=mel_spec,
        waveform=waveform,
        lengths=lengths,
        sample_rate=sample_rate,
    )

    # Return same batch structure as input
    if was_unbatched:
        embedding = embedding.squeeze(0)

    return embedding


def get_speaker_embedding_dim(encoder_type: SpeakerEncoderType = DEFAULT_ENCODER_TYPE) -> int:
    """Get the embedding dimension for a speaker encoder type without loading the model."""
    return SPEAKER_EMBEDDING_DIMS[encoder_type]


def get_speaker_encoder_input_type(encoder_type: SpeakerEncoderType = DEFAULT_ENCODER_TYPE) -> str:
    """Get the expected input type for a speaker encoder ('mel' or 'waveform')."""
    return SPEAKER_ENCODER_INPUT_TYPES[encoder_type]
