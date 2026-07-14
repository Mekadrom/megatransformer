# MegaTransformer

Multimodal autoregressive world model combining text, audio, voice, and image
modalities — a recurrent transformer with modality-specific VAE encoders/decoders
and token interleaving for unified sequence processing.

Subpackages: **SIVE** (speaker-invariant voice encoder), **SMG** (SIVE-Mel
generator), **vocoder** (HiFi-GAN mel→wave), the **world model**, and image VAE.

## Setup (uv)

The project is managed with [uv](https://docs.astral.sh/uv/). It provisions its
own Python 3.10 and a `.venv` — no system/conda Python required.

```bash
uv sync                 # core + training deps (torch cu124), into .venv
uv sync --extra demo    # + gradio eval demos
uv sync --extra image   # + image-VAE (LiteVAE) deps (needs ../open-litevae)
```

torch/torchaudio come from the CUDA 12.4 wheel index (`pytorch-cu124`) via the
`training` dependency-group. Run commands with `uv run`:

```bash
uv run python -m megatransformer.scripts.train.train smg --run_name my_run --config small ...
```

## Using megatransformer as a dependency (e.g. ComfyUI)

The torch family is deliberately excluded from the published dependencies, so a
plain pip install never clobbers a host env's CUDA torch:

```bash
pip install "megatransformer @ git+https://github.com/Mekadrom/megatransformer.git"
```

This installs the core (non-torch) deps and relies on the host's existing torch.

## More

See `CLAUDE.md` for architecture, commands, and conventions.
