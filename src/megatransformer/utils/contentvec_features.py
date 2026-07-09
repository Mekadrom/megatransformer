"""ContentVec (off-the-shelf HuBERT) feature extraction for the SIVE-diagnostics
probes, so speaker-leakage / synthesis-usability can compare an off-the-shelf,
invariance-disentangled content encoder against SIVE on the SAME data + protocol.

ContentVec takes the raw 16 kHz waveform (not mel) and runs at 50 Hz (vs SIVE's
~20.8 Hz). last_hidden_state is 768-dim (`vec768l12`); a hidden layer can be tapped
via `layer`. The model is frozen; features are cached by the caller.
"""
import torch

_CACHE = {}


def load_contentvec(model_id, device):
    """Load + cache a frozen HuBERT ContentVec (transformers.HubertModel)."""
    from transformers import HubertModel
    key = (model_id, str(device))
    if key not in _CACHE:
        m = HubertModel.from_pretrained(model_id).eval().to(device)
        for p in m.parameters():
            p.requires_grad = False
        _CACHE[key] = m
    return _CACHE[key]


@torch.no_grad()
def contentvec_hidden(model, wav_16k, layer=-1):
    """wav_16k: [T] or [1, T] float waveform @16 kHz -> features [T', D].

    layer < 0 = last_hidden_state (768-dim, vec768l12); layer >= 0 = hidden_states[layer]
    (0 = conv-frontend output, 1..12 = transformer layers)."""
    if wav_16k.dim() == 1:
        wav_16k = wav_16k.unsqueeze(0)
    out = model(wav_16k, output_hidden_states=(layer >= 0))
    h = out.last_hidden_state if layer < 0 else out.hidden_states[layer]
    return h[0]  # [T', D]
