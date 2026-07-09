"""Off-the-shelf zero-shot voice-conversion demo (FreeVC via Coqui TTS).

Upload a CONTENT clip (what is said) + a TARGET SPEAKER clip (whose voice) and
get the content spoken in the target voice. FreeVC is the architecture you're
after: a disentangled content encoder (WavLM + bottleneck) + a speaker embedding
(from the target reference) + a decoder -> waveform, zero-shot. This is the
sanity check on whether an off-the-shelf "content + speaker-embedding -> audio"
stack beats the custom SIVE/SMG on your own audio.

Setup (torch already installed):
    pip install coqui-tts gradio      # 'coqui-tts' is the maintained fork; imports as TTS
    # first run auto-downloads the FreeVC-24k checkpoint + WavLM

Run:
    python vc_offshelf_demo.py             # opens a local URL
    python vc_offshelf_demo.py --share     # public gradio link
    python vc_offshelf_demo.py --cpu       # force CPU (slow; GPU auto-used otherwise)

Notes:
- FreeVC conditions on a speaker embedding from the target reference, so a SHORT
  target clip (a few seconds) is enough — no big matching pool needed.
- Output is 24 kHz.
- This tests the mechanism (content + speaker embedding -> audio). It does NOT
  by itself give you discrete content TOKENS for the world model to autoregress:
  FreeVC's content encoder is welded to its decoder and its content is continuous.
  For the world-model integration you'd either reuse a system's content latent as
  the generation target, or pair ContentVec/HuBERT units with a small unit vocoder
  conditioned on ECAPA (the decoupled path). Ask and I'll scaffold that.
"""
import argparse

import numpy as np
import torch
import gradio as gr

_TTS = None
_SR = 24000


def get_model(device):
    """Lazy-load FreeVC-24k once (first call downloads weights)."""
    global _TTS
    if _TTS is None:
        from TTS.api import TTS
        print("Loading FreeVC-24k via Coqui TTS — first run downloads weights ...")
        _TTS = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24",
                   progress_bar=False).to(device)
        print("FreeVC loaded.")
    return _TTS


def convert(content_path, target_path, device):
    if not content_path or not target_path:
        raise gr.Error("Please provide BOTH a content clip and a target-speaker clip.")
    tts = get_model(device)
    # content -> WavLM/bottleneck content; target -> speaker embedding; decoder -> wave
    wav = tts.voice_conversion(source_wav=content_path, target_wav=target_path)
    return (_SR, np.asarray(wav, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="Create a public gradio link")
    ap.add_argument("--cpu", action="store_true", help="Force CPU (default: GPU if available)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    device = "cpu" if (args.cpu or not torch.cuda.is_available()) else "cuda"
    print(f"Device: {device}")

    with gr.Blocks(title="Off-the-shelf VC (FreeVC)") as demo:
        gr.Markdown(
            "# Zero-shot Voice Conversion — FreeVC (off-the-shelf)\n"
            "Upload a **content** clip (what is said) and a **target speaker** clip (whose voice). "
            "Output = the content spoken in the target's voice. No training, zero-shot from the reference.\n\n"
            "FreeVC = disentangled content encoder + speaker embedding (from the target) + decoder — "
            "the exact 'content + speaker-embedding → audio' shape you're evaluating."
        )
        with gr.Row():
            content = gr.Audio(label="Content  (source — WHAT is said)", type="filepath")
            target = gr.Audio(label="Target speaker  (WHOSE voice — a few seconds is enough)", type="filepath")
        btn = gr.Button("Convert", variant="primary")
        out = gr.Audio(label="Converted output (24 kHz)", type="numpy")
        btn.click(fn=lambda c, t: convert(c, t, device), inputs=[content, target], outputs=out)

    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
