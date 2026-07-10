"""ContentVec (off-the-shelf HuBERT) feature extraction for the SIVE-diagnostics
probes, so speaker-leakage / synthesis-usability can compare an off-the-shelf,
invariance-disentangled content encoder against SIVE on the SAME data + protocol.

ContentVec takes the raw 16 kHz waveform (not mel) and runs at 50 Hz (vs SIVE's
~20.8 Hz). Two feature widths from the SAME 95M model:
  - dim=768 (`vec768l12`): last_hidden_state, the transformer's native width.
  - dim=256 (`vec256`): final_proj(last_hidden_state) — the trained projection into
    ContentVec's disentangled/contrastive space; matches SIVE's 256-dim width, so
    it's both the efficient choice and the dimensionality-fair comparison.
The model is frozen; features are cached by the caller.
"""
import torch

_CACHE = {}


def load_contentvec(model_id, device, dim=768):
    """Load + cache a frozen HuBERT ContentVec (transformers.HubertModel).

    dim=256 keeps the checkpoint's `final_proj` head (dropped by plain HubertModel)
    so callers can project last_hidden_state 768 -> 256."""
    from transformers import HubertModel
    key = (model_id, str(device), dim)
    if key in _CACHE:
        return _CACHE[key]
    if dim == 256:
        import torch.nn as nn

        class HubertModelWithFinalProj(HubertModel):
            def __init__(self, config):
                super().__init__(config)
                self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

        m = HubertModelWithFinalProj.from_pretrained(model_id)
    else:
        m = HubertModel.from_pretrained(model_id)
    m = m.eval().to(device)
    for p in m.parameters():
        p.requires_grad = False
    _CACHE[key] = m
    return m


@torch.no_grad()
def contentvec_hidden(model, wav_16k, layer=-1, final_proj=False):
    """wav_16k: [T] or [1, T] float waveform @16 kHz -> features [T', D].

    layer < 0 = last_hidden_state; layer >= 0 = hidden_states[layer].
    final_proj=True applies the 768->256 head (needs a model loaded with dim=256)."""
    if wav_16k.dim() == 1:
        wav_16k = wav_16k.unsqueeze(0)
    out = model(wav_16k, output_hidden_states=(layer >= 0))
    h = out.last_hidden_state if layer < 0 else out.hidden_states[layer]
    if final_proj:
        h = model.final_proj(h)
    return h[0]  # [T', D]
