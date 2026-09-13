# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MegaTransformer is a multimodal autoregressive world model combining text, audio, voice, and image modalities. The project implements a recurrent transformer architecture with modality-specific VAE encoders/decoders and token interleaving for unified sequence processing.

## Commands

### Training

Training uses subcommands for different model types. All training scripts share common arguments.

```bash
# SMG (SIVE-Mel Generator: speaker-conditioned deterministic decoder with FiLM)
python -m megatransformer.scripts.train.train smg --run_name my_run --config small --cache_dir ../cached_datasets/sive_smg_f0

# Vocoder (mel-to-waveform)
python -m megatransformer.scripts.train.train vocoder --run_name my_vocoder --config tiny --cache_dir ../cached_datasets/audio

# SIVE (Speaker-Invariant Voice Encoder with CTC + GRL)
python -m megatransformer.scripts.train.train audio-sive --run_name my_sive --config small --cache_dir ../cached_datasets/audio_sive

# Image VAE
python -m megatransformer.scripts.train.train image-vae --run_name my_image_vae --config small --cache_dir ../cached_datasets/image

# World Model (multimodal)
python -m megatransformer.scripts.train.train world --run_name my_world --config small --include_modes text,audio,image
```

Common training arguments:
- `--resume_from_checkpoint <path>`: Resume from checkpoint
- `--use_deepspeed --deepspeed_config ds_config.json`: Enable DeepSpeed
- `--bf16` / `--fp16`: Mixed precision training
- `--use_gradient_checkpointing`: Memory optimization
- `--use_gan`: Enable GAN training (for VAE models)
- `--use_muon`: Use Muon+AdamW optimizer (with `--lr_muon`, `--lr_adamw`)
- `--use_ema --ema_decay 0.9999`: Exponential moving average
- `--compile_model`: torch.compile the model
- `--metrics_backend tensorboard|wandb`: Logging backend (default: tensorboard)

### Data Preprocessing

```bash
# Audio preprocessing (extracts SIVE features, speaker embeddings, F0, mel specs)
python -m megatransformer.scripts.data.preprocess_dataset audio \
    --dataset_name mozilla-foundation/common_voice_17_0 \
    --dataset_config en --split train \
    --output_dir ../cached_datasets/audio_train \
    --sive_checkpoint_path ./checkpoints/sive \
    --compute_speaker_embeddings --extract_f0 --save_mel_specs

# Voice preprocessing for the world model (CosyVoice 2 units + campplus + SmolLM2 text).
# ONE command -- produces the world-voice cache directly. Needs onnxruntime-gpu>=1.20,<1.24.
# For LOCAL parquet use --dataset_name parquet --data_files '<glob>' and DROP --dataset_config.
python -m megatransformer.scripts.data.preprocess_dataset voice --dataset_name mythicinfinity/libriheavy --dataset_config large --split train --streaming --output_dir ../cached_datasets/libriheavy_cosyvoice2_smollm2/train --content_encoder cosyvoice2 --cosyvoice2_model_dir <CosyVoice2-0.5B snapshot> --compute_speaker_embeddings --tokenize_text --tokenizer_name HuggingFaceTB/SmolLM2-135M --sample_rate 16000 --voice_max_seconds 10.0 --audio_column audio --text_conditions_column text_original --speaker_id_column speaker_id --duration_column audio_duration --max_hours 2000

# Build shard index after preprocessing
python -m megatransformer.scripts.data.preprocess_dataset stat-shards --output_dir ../cached_datasets/audio_train

# Multi-GPU preprocessing (run on each GPU)
python -m megatransformer.scripts.data.preprocess_dataset audio --gpu_id 0 --total_gpus 4 ...
```

### Testing

```bash
pytest tests/                          # Run all tests
python -m pytest tests/ -v             # Verbose output
python -m pytest tests/test_collator_token_placement.py  # Single test file
```

Tests cover data collation logic (BO*/PH/EO* token placement, text target alignment). No linting or formatting tools are configured.

### Inference / Evaluation

```bash
# Voice cloning demo (Gradio UI) — combines SIVE + SMG + vocoder pipeline
python -m megatransformer.scripts.eval.smg.voice_clone --sive_checkpoint_path ./checkpoints/sive --smg_checkpoint_path ./checkpoints/smg --vocoder_checkpoint_path ./checkpoints/vocoder
```

Eval scripts live in `src/megatransformer/scripts/eval/` with subdirectories per modality.

### TensorBoard

```bash
./tensorboard.sh  # or: tensorboard --logdir runs/
```

## Architecture

### Directory Structure

- `src/megatransformer/model/`: Neural network modules
  - `world/`: Core world model (`MegaTransformerWorldModel`, recurrent transformer, KV cache, token interleaving)
  - `voice/`: Voice/speech models — prelude feature extractor, coda generator, plus `sive/` (Speaker-Invariant Voice Encoder) and `vocoder/` (HiFiGAN-based mel-to-wave) subpackages
  - `audio/`: Non-speech audio prelude/coda (`feature_extractor.py`, `generator.py`)
  - `smg/`: SIVE-Mel Generator (speaker-conditioned deterministic decoder with FiLM) — `smg.py`, `discriminator.py`, `criteria.py`, `residual_block.py`
  - `image/`: Image models (VAE, prelude feature extractor, `decoder.py` direct decoder, `diffusion_decoder.py` flow-matching DiT)
  - `text/`: Text feature extractor (prelude with causal transformer) and generator (coda classifier)
  - `transformer.py`: `MegaTransformerBlock` with GQA, rotary embeddings, ALiBi

- `src/megatransformer/config/`: Dataclass configs with predefined configurations (small, medium, large)
  - Each model has `*_CONFIGS` dicts mapping config names to dataclass instances
  - `common.py`: `MegaTransformerBlockConfig` shared across models
  - `image/decoder.py`: `ImageDecoderConfig` (direct) and `DiffusionBridgeImageDecoderConfig` (flow-matching DiT)

- `src/megatransformer/scripts/train/`: Training scripts
  - `train.py`: Main entry point with subcommand routing
  - `trainer.py`: `CommonTrainer` base class extending HuggingFace Trainer
  - `optimizers.py`: `MuonAdamW` custom optimizer
  - `smg/`, `audio/vocoder/`, `audio/sive/`, `image/vae/`, `world/`: Model-specific trainers

- `src/megatransformer/scripts/data/`: Dataset preprocessing and loading
  - `preprocess_dataset.py`: Main preprocessing entry point with modality-specific `Preprocessor` subclasses
  - `audio/`, `image/`, `text/`, `world/`: Per-modality dataset, collator, and preprocessor implementations

- `src/megatransformer/utils/`: Shared utilities
  - `metrics.py`: Central metrics logging module (backend-agnostic singleton)
  - `metrics_backend.py`: `MetricsBackend` protocol, `TensorBoardBackend`, `NoOpBackend`
  - `wandb_backend.py`: `WandBBackend` with context-aware media grouping
  - `visualization.py`: Pure rendering functions (mel specs, attention weights, vocoder audio)
  - `model_loading_utils.py`: `load_model()` function for loading from config + checkpoint
  - `audio_utils.py`: `SharedWindowBuffer` for efficient STFT/mel computation
  - `speaker_encoder.py`: ECAPA-TDNN and WavLM speaker encoders
  - `voice_silence_mask.py`: Inference-only silence detection/masking for SMG-decoded mel spectrograms
  - `megatransformer_utils.py`: Weight init helpers (`linear_weight_init`, `apply_depth_scaled_residual_init`, `conv2d_weight_init`)

### Key Design Patterns

**Config-based model instantiation**: Models use dataclass configs and `from_config()` class methods:
```python
model = load_model(SMG, "small", checkpoint_path=path, overrides={"latent_channels": 32})
```

**Sharded datasets**: Training data is preprocessed into `.pt` shards with a `shard_index.json` manifest. Dataset classes (e.g. `AudioShardedDataset`, `MultimodalShardedDataset`) handle lazy loading with LRU caching. `ShardAwareSampler` groups indices by shard to minimize disk I/O.

**Custom trainers**: Each training script has a trainer class extending `CommonTrainer` (which extends HuggingFace `Trainer`). Each module exports `add_cli_args(subparsers)` and `load_model(args)` functions. Trainers implement `compute_loss()` with model-specific loss computation.

**GAN training**: VAE trainers support optional discriminator training with configurable start conditions (`--gan_start_condition_key step/loss`), adaptive weighting, R1 penalty, and instance noise.

**Training module convention**: Each training submodule in `src/megatransformer/scripts/train/` (e.g. `smg/training.py`) must export:
- `add_cli_args(subparsers)`: Registers the subcommand and its args
- `load_model(args)`: Creates/loads the model from config and optional checkpoint

### Metrics Logging

All metrics logging goes through the centralized `src/megatransformer/utils/metrics.py` module — trainers and visualization callbacks never interact with TensorBoard/W&B directly.

**Architecture** (3 layers):
- `metrics.py`: `MetricsLogger` class + module-level convenience functions (`log_scalar`, `log_image`, `log_audio`, `log_figure`, `log_text`, `log_histogram`, `flush`). Initialized once via `metrics.init_metrics(backend)` in `train.py`.
- `metrics_backend.py`: `MetricsBackend` protocol + `TensorBoardBackend` + `NoOpBackend`. Backend is selected by `--metrics_backend tensorboard|wandb`.
- `visualization.py`: Pure rendering functions that return matplotlib `Figure` objects or numpy arrays — never log anything. Includes `render_mel_spectrogram()`, `render_mel_comparison()`, `render_attention_weights()`, `render_vocoder_audio()`.

**Usage in trainers** — call module-level functions directly:
```python
from utils import metrics
metrics.log_scalar("train/loss", loss, global_step)
metrics.log_text("training/command_line", cmdline, global_step)
```

**Usage in visualization callbacks** — check logger exists, use `metrics.*` and `visualization.*`:
```python
from utils import metrics, visualization
logger = metrics.get_logger()
if logger is None:
    return
fig = visualization.render_mel_comparison(pred_mel, target_mel)
metrics.log_figure("eval/mel_comparison", fig, global_step)
plt.close(fig)
```

**Context grouping** — every `log_*` method accepts an optional `context` dict to group related media:
```python
metrics.log_audio("eval/voice/0", waveform, step, sr, context={
    "mel": mel_figure,           # Figure → logged as figure
    "transcription": "hello",    # str → logged as text
})
```
For TensorBoard, context items are logged as sibling tags (`eval/voice/0/mel`, `eval/voice/0/transcription`). The `WandBBackend` (`supports_context=True`) renders them in unified panels with captions.

**Adding a new backend**: Implement the `MetricsBackend` protocol (8 methods: `add_scalar`, `add_image`, `add_audio`, `add_figure`, `add_text`, `add_histogram`, `flush`, `close`). All methods accept `**kw` to receive the optional `context` kwarg. Set `supports_context = True` on the class to handle context natively (otherwise `MetricsLogger` falls back to sibling-tag dispatch).

### World Model Architecture

`MegaTransformerWorldModel` processes multimodal sequences:
1. Modality-specific feature extractors / preludes (text embedding + causal transformer, audio/voice/image VAE encoders)
2. `TokenInterleaver`: Interleaves modality tokens based on placeholder positions in text
3. `MegatransformerRecurrentBlock`: Recurrent transformer with thought vector mechanism (Huginn-style, additive injection)
4. `TokenUninterleaver`: Separates tokens back to modality-specific sequences
5. Modality-specific generators/codas (text classifier, audio/voice SIVE predictors, image DiT decoder)

**Generation approaches by modality:**
- **Text**: Autoregressive token-by-token. Text prelude (causal) + recurrent block + text coda (causal). All three use KV caching at inference.
- **Voice/Audio**: Autoregressive frame-by-frame. Uses shifted teacher forcing during training (position 0 = zero vector, position t = prelude(frame t-1), target = frame t). Voice prelude is causal with KV caching. Voice coda is causal with KV caching. Includes a stop prediction head (`nn.Linear(d_model, 1)`) to end generation early. No generation queries.
- **Image**: Single-shot via generation queries. Positional-only gen queries → recurrent block → `DiffusionBridgeImageDecoder` (Q-Former bridge + flow-matching DiT with AdaLN-Zero). Not autoregressive.

**Critical inference rules:**
- Every module with causal self-attention must have KV caching during generation, or self-attention becomes a no-op (seq_len=1).
- The text coda must NOT run during voice/audio/image generation steps. During training, the uninterleaver only gives it text positions — feeding media hidden states pollutes its KV cache.
- The `generate()` method in `world_model.py` threads KV caches through: text prelude, recurrent block, text coda, voice/audio prelude, and voice/audio coda.

### Audio Pipeline

1. **SIVE** (Speaker-Invariant Voice Encoder): Conformer encoder with CTC loss + GRL for speaker disentanglement
2. **Audio SMG**: SIVE-Mel Generator — deterministic decoder with FiLM-based speaker conditioning, outputs mel spectrograms
3. **Vocoder**: Mel spectrogram to waveform synthesis (HiFiGAN-based)

## Findings (READ AND WRITE THESE)

`docs/findings/` is the durable, versioned record of what has been measured. **Read the file
for the direction you are working on before proposing experiments, and write results back to
it before the session ends.** `eval_output/` is gitignored and gets cleared, `runs/` holds
scalars without interpretation, and commit messages are searchable but not browsable — so a
result that lives only in those places is effectively lost.

One file per training direction, named `<source>-<target>` with `world` = the recurrent trunk:
`world-voice.md` (text->voice, formerly "world-tts"), `world-image.md` (text->image),
`world-text.md`, `voice-world.md` (transcription), `image-world.md` (captioning), plus
`cross-modal.md` for principles replicated in more than one direction.

Rules that matter (full conventions in `docs/findings/README.md`):
- Every entry is **ESTABLISHED**, **OPEN**, or **RETRACTED**.
- **Never delete a retracted finding.** Strike it, say why it was wrong, keep the date. A
  wrong conclusion that quietly vanishes is worse than one marked wrong, because the old
  version is what people remember and nothing contradicts it. This project has already lost
  time to acting on conclusions that were later invalidated — an eval path that silently
  disabled M-RoPE, eval scripts sampling at a temperature nobody listens at, and decode
  comparisons drawn inside their own noise floor.
- Quote the measurement inline. "Text conditioning is weak" is not a finding; "text-attributed
  fraction 0.029 vs AR's 0.305 at matched step 23000" is.
- Note the protocol when it is load-bearing: teacher-forced vs free-running, matched step,
  seed count, n. Several retractions exist because a protocol detail was wrong and undocumented.
- A finding belongs to the direction it was measured in. Promote to `cross-modal.md` only
  after replication in a second direction.

Active implementation plans live in `docs/plans/`. Read the relevant one before starting work
it covers — they carry the traps and the decisions already made.

## Configuration

DeepSpeed configs are in root: `ds_config.json`, `ds_config_zero-*.json`, `ds_config_int8.json`

Runs are logged to `runs/<run_name>/` (metrics + checkpoints). Backend is selected via `--metrics_backend` (tensorboard or wandb).

## Environment

- Python 3.10, CUDA 12.4. Managed with **[uv](https://docs.astral.sh/uv/)** — it
  provisions its own Python 3.10 and a `.venv`; no system/conda Python needed.
- Setup: **`uv sync --extra cu124`** (core + `training` group + the CUDA 12.4 torch build).
  Add `--extra demo` (gradio) or `--extra image` (LiteVAE, needs `../open-litevae`).
- **The accelerator is an EXTRA — one of `cu124` / `rocm` / `cpu` must be selected.** A bare
  `uv sync` no longer pins the CUDA wheels: torch has no unconditional source, so it would fall
  back to PyPI. uv 0.11.7 has no `default-extras`, so either pass `--extra cu124` every time or
  export `UV_EXTRA=cu124` on the training nodes.
  - CUDA training node: `uv sync --extra cu124 --extra image`
  - ROCm inference node: `uv sync --no-default-groups --inexact --extra rocm --extra demo --extra image`
  - CPU-only: `uv sync --no-default-groups --inexact --extra cpu`
  ⚠️ **Core deps are NOT torch-free**, despite the comment in `pyproject.toml` implying it:
  `speechbrain`, `torchcrepe` and `rotary-embedding-torch` all require torch UNCONDITIONALLY, so
  `--no-default-groups` alone never avoided pulling the `nvidia-*` cu12 wheels. That is why the
  accelerator had to become an extra rather than being left to the `training` group.
  Each branch also needs a version FLOOR (unconstrained, the ROCm fork resolved torch 2.0.1 /
  torchvision 0.15.2, a pair with no wheel for the platform), and `pytorch-triton-rocm` needs an
  explicit `[tool.uv.sources]` mapping because the index is `explicit = true`.
- Run commands with `uv run`, e.g. `uv run python -m megatransformer.scripts.train.train ...`
  (or `source .venv/bin/activate` once, then plain `python -m ...`).
- `uv.lock` is the source of truth (committed) and now carries ALL THREE accelerator branches,
  so one lock serves CUDA, ROCm and CPU nodes. `requirements.txt` is a generated pip fallback
  (`uv export`) with the cu124 index URL — CUDA-only, regenerate after dep changes.
- ⚠️ **Claude should not run `uv` commands** (`uv sync`, `uv lock`, `uv add`, ...). Hand the exact
  command over instead; the user runs it. Dependency state is theirs to change.
- **Dependency layout** (`pyproject.toml`): core `[project.dependencies]` are the
  unpinned, torch-free libs a downstream consumer needs (the ComfyUI SMG-inference
  nodes `pip install megatransformer` and use their own torch — torch/torchaudio
  live in the uv `training` dependency-group, which is invisible to pip consumers).
  Torch cu124 comes from the `pytorch-cu124` index via `[tool.uv.sources]`.

## CosyVoice 2 runtime (world-voice decode)

The voice path decodes through CosyVoice 2's frozen flow + HiFT, which needs a **manual
sys.path checkout** — it is not pip-installable and is not in `uv.lock`:

```bash
./scripts_local/setup_cosyvoice_runtime.sh [target_dir]   # default ~/dev/projects/cosyvoice-runtime
```

Pass it as `--voice_cosyvoice2_runtime_dir` (training, eval scripts and the chat UI all take
this) or `$COSYVOICE_RUNTIME`. Model weights are NOT in it — they stay in the HF cache and go
in via `--voice_cosyvoice2_model_dir`.

`cv_extra/` inside it holds `--no-deps` installs kept deliberately OUT of `.venv`, because
`uv sync` would fight them. They only satisfy module-level imports in `cosyvoice/flow/*.py`
that decode never calls. Two pins matter: `antlr4-python3-runtime==4.9.3` (4.13 raises
"Could not deserialize ATN with version 3") and `setuptools<81` (81+ removed `pkg_resources`,
which `pyworld` imports).

Decode needs neither the ONNX speech tokenizer (that builds caches), the text frontend (we
feed unit ids), nor the Qwen LLM (the world model replaces it) — `_load_configs_without_llm`
keeps only `flow:` and `hift:` from the yaml. A half-built runtime is reported by
`_ensure_importable` with the missing directory named; the raw symptom would otherwise be a
`ModuleNotFoundError` from inside `pydoc.locate`.

## Import Convention

The codebase is a `src`-layout package named `megatransformer` (see `pyproject.toml`).
All imports are absolute under that namespace, e.g. `from megatransformer.model.smg.smg import SMG`.
After `uv sync` (or `pip install -e .`) the package is importable from anywhere — no `PYTHONPATH=src` needed.
Run modules via the package path: `uv run python -m megatransformer.scripts.train.train ...`
