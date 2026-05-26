# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MegaTransformer is a multimodal autoregressive world model combining text, audio, voice, and image modalities. The project implements a recurrent transformer architecture with modality-specific VAE encoders/decoders and token interleaving for unified sequence processing.

## Commands

### Training

Training uses subcommands for different model types. All training scripts share common arguments.

```bash
# SMG (SIVE-Mel Generator: speaker-conditioned deterministic decoder with FiLM)
python -m src.scripts.train.train smg --run_name my_run --config small --cache_dir ../cached_datasets/sive_smg_f0

# Vocoder (mel-to-waveform)
python -m src.scripts.train.train vocoder --run_name my_vocoder --config tiny --cache_dir ../cached_datasets/audio

# SIVE (Speaker-Invariant Voice Encoder with CTC + GRL)
python -m src.scripts.train.train audio-sive --run_name my_sive --config small --cache_dir ../cached_datasets/audio_sive

# Image VAE
python -m src.scripts.train.train image-vae --run_name my_image_vae --config small --cache_dir ../cached_datasets/image

# World Model (multimodal)
python -m src.scripts.train.train world --run_name my_world --config small --include_modes text,audio,image
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
python -m src.scripts.data.preprocess_dataset audio \
    --dataset_name mozilla-foundation/common_voice_17_0 \
    --dataset_config en --split train \
    --output_dir ../cached_datasets/audio_train \
    --sive_checkpoint_path ./checkpoints/sive \
    --compute_speaker_embeddings --extract_f0 --save_mel_specs

# Build shard index after preprocessing
python -m src.scripts.data.preprocess_dataset stat-shards --output_dir ../cached_datasets/audio_train

# Multi-GPU preprocessing (run on each GPU)
python -m src.scripts.data.preprocess_dataset audio --gpu_id 0 --total_gpus 4 ...
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
python -m src.scripts.eval.smg.voice_clone --sive_checkpoint_path ./checkpoints/sive --smg_checkpoint_path ./checkpoints/smg --vocoder_checkpoint_path ./checkpoints/vocoder
```

Eval scripts live in `src/scripts/eval/` with subdirectories per modality.

### TensorBoard

```bash
./tensorboard.sh  # or: tensorboard --logdir runs/
```

## Architecture

### Directory Structure

- `src/model/`: Neural network modules
  - `world/`: Core world model (`MegaTransformerWorldModel`, recurrent transformer, KV cache, token interleaving)
  - `voice/`: Voice/speech models — prelude feature extractor, coda generator, plus `sive/` (Speaker-Invariant Voice Encoder) and `vocoder/` (HiFiGAN-based mel-to-wave) subpackages
  - `audio/`: Non-speech audio prelude/coda (`feature_extractor.py`, `generator.py`)
  - `smg/`: SIVE-Mel Generator (speaker-conditioned deterministic decoder with FiLM) — `smg.py`, `discriminator.py`, `criteria.py`, `residual_block.py`
  - `image/`: Image models (VAE, prelude feature extractor, `decoder.py` direct decoder, `diffusion_decoder.py` flow-matching DiT)
  - `text/`: Text feature extractor (prelude with causal transformer) and generator (coda classifier)
  - `transformer.py`: `MegaTransformerBlock` with GQA, rotary embeddings, ALiBi

- `src/config/`: Dataclass configs with predefined configurations (small, medium, large)
  - Each model has `*_CONFIGS` dicts mapping config names to dataclass instances
  - `common.py`: `MegaTransformerBlockConfig` shared across models
  - `image/decoder.py`: `ImageDecoderConfig` (direct) and `DiffusionBridgeImageDecoderConfig` (flow-matching DiT)

- `src/scripts/train/`: Training scripts
  - `train.py`: Main entry point with subcommand routing
  - `trainer.py`: `CommonTrainer` base class extending HuggingFace Trainer
  - `optimizers.py`: `MuonAdamW` custom optimizer
  - `smg/`, `audio/vocoder/`, `audio/sive/`, `image/vae/`, `world/`: Model-specific trainers

- `src/scripts/data/`: Dataset preprocessing and loading
  - `preprocess_dataset.py`: Main preprocessing entry point with modality-specific `Preprocessor` subclasses
  - `audio/`, `image/`, `text/`, `world/`: Per-modality dataset, collator, and preprocessor implementations

- `src/utils/`: Shared utilities
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

**Training module convention**: Each training submodule in `src/scripts/train/` (e.g. `smg/training.py`) must export:
- `add_cli_args(subparsers)`: Registers the subcommand and its args
- `load_model(args)`: Creates/loads the model from config and optional checkpoint

### Metrics Logging

All metrics logging goes through the centralized `src/utils/metrics.py` module — trainers and visualization callbacks never interact with TensorBoard/W&B directly.

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

## Configuration

DeepSpeed configs are in root: `ds_config.json`, `ds_config_zero-*.json`, `ds_config_int8.json`

Runs are logged to `runs/<run_name>/` (metrics + checkpoints). Backend is selected via `--metrics_backend` (tensorboard or wandb).

## Environment

- Python 3.10, CUDA 12.4
- Dependencies: `pip install -r requirements.txt`
- venv: `source venv/bin/activate`

## Import Convention

All imports use relative-to-`src/` paths (e.g. `from model.smg.smg import SMG`, not `from src.model...`). The project is run as a module from the repo root: `python -m src.scripts.train.train ...`
