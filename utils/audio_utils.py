import torch

from typing import Tuple

from torchaudio import transforms


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
) -> Tuple[torch.Tensor, torch.Tensor]:
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
