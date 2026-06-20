import torch


from torchaudio import transforms
import torchaudio


## Copied from speechbrain with modifications ##

spec_transform_cache = {}
mel_transform_cache = {}


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """Dynamic range compression for audio signals"""
    return torch.log(torch.clamp(x, min=clip_val) * C)


def configurable_mel_spectrogram(
    sample_rate,
    hop_length,
    win_length,
    n_fft,
    n_mels,
    f_min,
    f_max,
    power,
    normalized,
    min_max_energy_norm,
    norm,
    mel_scale,
    compression,
    audio,
    window_provider=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = audio.dtype

    audio = audio.to(torch.float32)

    audio_cache_key = dict(
        sample_rate=sample_rate,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=power,
        normalized=normalized,
        norm=norm,
        mel_scale=mel_scale,
    )

    if str(audio_cache_key) in spec_transform_cache:
        audio_to_mel, mel_scale = spec_transform_cache[str(audio_cache_key)]
    else:
        audio_to_mel = transforms.Spectrogram(
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
            power=power,
            window_fn=window_provider if window_provider is not None else torch.hann_window,
            normalized=normalized,
        ).to(torch.float32)
        spec_transform_cache[str(audio_cache_key)] = (audio_to_mel, mel_scale)

    mel_cache_key = dict(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        norm=norm,
        mel_scale=mel_scale,
    )

    if str(mel_cache_key) in mel_transform_cache:
        mel_scale = mel_transform_cache[str(mel_cache_key)]
    else:
        mel_scale = transforms.MelScale(
            sample_rate=sample_rate,
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            norm=norm,
            mel_scale=mel_scale,
        ).to(torch.float32)
        mel_transform_cache[str(mel_cache_key)] = mel_scale

    audio_to_mel = audio_to_mel.to(audio.device)
    mel_scale = mel_scale.to(audio.device)

    spec = audio_to_mel(audio)
    mel = mel_scale(spec)
    
    # Modified to support both 2D and 3D inputs (unbatched, batched)
    assert mel.shape[-2] == n_mels
    rmse = torch.norm(mel, dim=-2)

    if min_max_energy_norm:
        rmse = (rmse - torch.min(rmse)) / (torch.max(rmse) - torch.min(rmse))

    if compression:
        mel = dynamic_range_compression(mel)

    mel = mel.to(orig_dtype)
    rmse = rmse.to(orig_dtype)
    return mel, rmse

## End of copied code ##


class SharedWindowBuffer:
    def __init__(self):
        self.cache = {}

    def get_window(self, window_size: int, device: torch.device) -> torch.Tensor:
        if window_size not in self.cache:
            window = torch.hann_window(window_size, device=device)
            self.cache[window_size] = window
        return self.cache[window_size].to(device)


def extract_waveforms(audio, sr=16000):
    # Check if audio is already processed
    if isinstance(audio, dict) and 'array' in audio:
        orig_sr = audio['sampling_rate']
        
        waveforms = torch.tensor(audio['array'], dtype=torch.float32).unsqueeze(0)
    elif isinstance(audio, torch.Tensor):
        # waveform as tensor
        waveforms = audio if audio.dim() == 2 else audio.unsqueeze(0)
        # assume default sample rate
        orig_sr = sr
    else:
        # Fallback for direct file paths
        waveforms, orig_sr = torchaudio.load(audio)

    # Resample if needed
    if orig_sr != sr:
        waveforms = torchaudio.transforms.Resample(orig_sr, sr)(waveforms)
    waveforms = waveforms.squeeze(0)
    y = waveforms.numpy()

    return waveforms, y, orig_sr


def extract_mels(shared_window_buffer: SharedWindowBuffer, y, sr=16000, n_mels=80, n_fft=1024, hop_length=256):
    """
    Extract audio features from loaded audio data.

    Args:
        shared_window_buffer: SharedWindowBuffer for STFT windows
        y: Audio data as waveform. Supports:
           - 1D tensor [T] - single sample
           - 2D tensor [B, T] - batch of samples (same length)
           - 2D tensor [1, T] - single sample with channel dim (will be squeezed)
        sr: Target sampling rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length for feature extraction

    Returns:
        log_mel_spec: Log mel spectrogram features
           - [n_mels, T] for single sample input
           - [B, n_mels, T] for batched input
    """
    # Handle input dimensions:
    # - [T] -> [T] (single sample, 1D)
    # - [1, T] -> [T] (single sample with channel dim, squeeze it)
    # - [B, T] where B > 1 -> [B, T] (batch, keep as is)
    if y.dim() == 2 and y.shape[0] == 1:
        # Single sample with channel dim [1, T] -> [T]
        y = y.squeeze(0)
    # For [B, T] batches or [T] single samples, pass through as-is

    # Extract mel spectrogram
    mel_spec, _ = configurable_mel_spectrogram(
        audio=y,
        sample_rate=sr,
        hop_length=hop_length,
        win_length=n_fft,
        n_mels=n_mels,
        n_fft=n_fft,
        f_min=0.0,
        f_max=8000.0,
        power=1,
        normalized=False,
        min_max_energy_norm=False,
        norm="slaney",
        mel_scale="slaney",
        compression=False,
        window_provider=lambda win_size: shared_window_buffer.get_window(win_size, device=y.device)
    )

    # converts to log scale, with clipping to avoid log(0)
    log_mel_spec = torch.log(mel_spec.clamp(min=1e-5))

    return log_mel_spec


def remove_mains_hum(waveform, sample_rate, frequencies=[60, 120, 180, 240]):
    """Remove mains hum and harmonics."""
    for freq in frequencies:
        waveform = torchaudio.functional.bandreject_biquad(
            waveform, sample_rate, central_freq=freq, Q=30.0
        )
    return waveform
