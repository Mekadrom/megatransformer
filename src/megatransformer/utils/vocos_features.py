"""Vocos (charactr/vocos-mel-24khz) mel features + vocoder.

Vocos is a fast (ISTFT-head) 24 kHz neural vocoder, ~13.5M params. Its mel spec is:
  sample_rate=24000, n_fft=1024, win_length=1024, hop_length=256 (-> 93.75 Hz mel),
  n_mels=100, f_min=0, f_max=Nyquist(12000), power=1 (MAGNITUDE), mel_scale=htk,
  then log via safe_log = log(clip(mel, 1e-7)).

To guarantee the SMG's mel TARGETS are byte-for-byte what Vocos expects (any mismatch
in power/clip/norm = garbage synthesis), compute them with Vocos's OWN feature extractor
via `vocos_mel()` rather than replicating the spec by hand. `vocos.decode(mel)` then
reconstructs the waveform losslessly from that mel.
"""
import torch

_CACHE = {}


def load_vocos(device="cpu", model_id="charactr/vocos-mel-24khz"):
    """Load + cache a frozen Vocos model."""
    key = (model_id, str(device))
    if key in _CACHE:
        return _CACHE[key]
    from vocos import Vocos
    v = Vocos.from_pretrained(model_id).to(device).eval()
    for p in v.parameters():
        p.requires_grad = False
    _CACHE[key] = v
    return v


@torch.no_grad()
def vocos_mel(vocos, wav_24k) -> torch.Tensor:
    """wav_24k: [T] or [1, T] float @24 kHz -> [100, T'] log-mel, exactly as Vocos
    expects (its feature_extractor). Use for SMG mel targets."""
    if wav_24k.dim() == 1:
        wav_24k = wav_24k.unsqueeze(0)
    dev = next(vocos.parameters()).device
    feats = vocos.feature_extractor(wav_24k.float().to(dev))  # [1, 100, T']
    return feats[0]


def vocos_frame_rate(model_id="charactr/vocos-mel-24khz") -> float:
    return 24000 / 256  # 93.75 Hz
