# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MegaTransformer is a multimodal autoregressive world model combining text, audio, voice, and image modalities. The project implements a recurrent transformer architecture with modality-specific VAE encoders/decoders and token interleaving for unified sequence processing.

## Commands

### Training

Training uses subcommands for different model types. All training scripts share common arguments.

```bash
# Audio CVAE (speaker-conditioned VAE with FiLM)
python -m src.scripts.train.train audio-cvae --run_name my_run --config small --cache_dir ../cached_datasets/sive_cvae_f0

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

### TensorBoard

```bash
./tensorboard.sh  # or: tensorboard --logdir runs/
```

## Architecture

### Directory Structure

- `src/model/`: Neural network modules
  - `world/`: Core world model (`MegaTransformerWorldModel`, recurrent transformer, KV cache)
  - `audio/`: Audio models (VAE, SIVE conformer, vocoder)
  - `image/`: Image VAE models
  - `text/`: Text feature extractor and generator
  - `transformer.py`: `MegaTransformerBlock` with GQA, rotary embeddings, ALiBi

- `src/config/`: Dataclass configs with predefined configurations (small, medium, large)
  - Each model has `*_CONFIGS` dicts mapping config names to dataclass instances
  - `common.py`: `MegaTransformerBlockConfig` shared across models

- `src/scripts/train/`: Training scripts
  - `train.py`: Main entry point with subcommand routing
  - `trainer.py`: `CommonTrainer` base class extending HuggingFace Trainer
  - `optimizers.py`: `MuonAdamW` custom optimizer
  - `audio/vae/`, `audio/vocoder/`, `audio/sive/`, `image/vae/`, `world/`: Model-specific trainers

- `src/scripts/data/`: Dataset preprocessing and loading
  - `preprocess_dataset.py`: Main preprocessing entry point with modality-specific `Preprocessor` subclasses
  - `audio/`, `image/`, `text/`, `world/`: Per-modality dataset, collator, and preprocessor implementations

- `src/utils/`: Shared utilities
  - `model_loading_utils.py`: `load_model()` function for loading from config + checkpoint
  - `audio_utils.py`: `SharedWindowBuffer` for efficient STFT/mel computation
  - `speaker_encoder.py`: ECAPA-TDNN and WavLM speaker encoders

### Key Design Patterns

**Config-based model instantiation**: Models use dataclass configs and `from_config()` class methods:
```python
model = load_model(AudioVAE, "small", checkpoint_path=path, overrides={"latent_channels": 32})
```

**Sharded datasets**: Training data is preprocessed into `.pt` shards with a `shard_index.json` manifest. Dataset classes (e.g. `AudioShardedDataset`, `MultimodalShardedDataset`) handle lazy loading with LRU caching. `ShardAwareSampler` groups indices by shard to minimize disk I/O.

**Custom trainers**: Each training script has a trainer class extending `CommonTrainer` (which extends HuggingFace `Trainer`). Each module exports `add_cli_args(subparsers)` and `load_model(args)` functions. Trainers implement `compute_loss()` with model-specific loss computation and TensorBoard logging.

**GAN training**: VAE trainers support optional discriminator training with configurable start conditions (`--gan_start_condition_key step/loss`), adaptive weighting, R1 penalty, and instance noise.

**Training module convention**: Each training submodule in `src/scripts/train/` (e.g. `audio/vae/training.py`) must export:
- `add_cli_args(subparsers)`: Registers the subcommand and its args
- `load_model(args)`: Creates/loads the model from config and optional checkpoint

### World Model Architecture

`MegaTransformerWorldModel` processes multimodal sequences:
1. Modality-specific feature extractors (text embedding, audio/image VAE encoders)
2. `TokenInterleaver`: Interleaves modality tokens based on placeholder positions in text
3. `MegatransformerRecurrentBlock`: Recurrent transformer with thought vector mechanism
4. `TokenUninterleaver`: Separates tokens back to modality-specific sequences
5. Modality-specific generators/codas (text classifier, audio/image VAE decoders)

### Audio Pipeline

1. **SIVE** (Speaker-Invariant Voice Encoder): Conformer encoder with CTC loss + GRL for speaker disentanglement
2. **Audio CVAE**: VAE with FiLM-based speaker conditioning, outputs mel spectrograms
3. **Vocoder**: Mel spectrogram to waveform synthesis (HiFiGAN-based)

## Configuration

DeepSpeed configs are in root: `ds_config.json`, `ds_config_zero-*.json`, `ds_config_int8.json`

Runs are logged to `runs/<run_name>/` (TensorBoard + checkpoints).

## Import Convention

All imports use relative-to-`src/` paths (e.g. `from model.audio.vae.vae import AudioVAE`, not `from src.model...`). The project is run as a module from the repo root: `python -m src.scripts.train.train ...`
