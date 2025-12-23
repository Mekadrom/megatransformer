# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MegaTransformer is a highly configurable transformer framework for multimodal deep generative models. It supports text, audio, and image modalities with modern transformer techniques including rotary embeddings, RMSNorm, SwiGLU activations, and recurrent/iterative deepening blocks.

## Common Commands

### Training with DeepSpeed (recommended)
```bash
deepspeed --num_gpus=2 <pretrain_script> \
    --use_deepspeed \
    --bf16 \
    --run_name my_run_name \
    --config tiny \
    --num_train_epoch 100 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 8 \
    --use_gradient_checkpointing \
    --deepspeed_config ds_config_zero-2.json
```

### Preprocessing Scripts

All preprocessing scripts save individual `.pt` files per sample to a specified output folder. VAE datasets support data augmentation during preprocessing.

**Audio Pipeline:**
- `preprocess_audio_vocoder_dataset.py` - Encodes audio into spectrograms and waveform labels
- `preprocess_audio_vae_dataset.py` - Encodes audio into spectrograms
- `preprocess_audio_diffusion_dataset.py` - Uses VAE checkpoint to encode audio into latent space with conditions

**Image Pipeline:**
- `preprocess_image_vae_dataset.py` - Crops/scales images and saves as labels
- `preprocess_image_diffusion_dataset.py` - Uses VAE checkpoint to encode images into latent space with conditions

### Training Scripts

Training order: Vocoder → VAE → Diffusion (each stage requires the previous)

**Audio:**
- `pretrain_vocoder.py` - Train first for audio (converts spectrograms to waveforms)
- `pretrain_audio_vae.py` - Optionally uses vocoder for waveform listening during training
- `pretrain_audio_diffusion.py` - Requires VAE: `--vae_checkpoint runs/audio_vae/<run>/checkpoint-STEP/ --vae_config <config>`. Optionally uses vocoder for waveform listening.

**Image:**
- `pretrain_image_vae.py` - Train first for images
- `pretrain_image_diffusion.py` - Requires VAE: `--vae_checkpoint runs/image_vae/<run>/checkpoint-STEP/ --vae_config <config>`

**Other:**
- `pretrain_recurrent_vae.py` - Iterative refinement VAE
- `pretrain_multimodal.py` - Multimodal world model (WIP)

See `utils/megatransformer_utils.py` for common script arguments used across all pretraining scripts.

## Architecture Overview

### Core Module Structure
```
model/
├── multimodal.py      # Main multimodal transformer (encoder + decoders)
├── diffusion.py       # UNet-based diffusion with time embeddings
├── recurrent.py       # Iterative deepening blocks with early exit
├── causal.py          # Transformer blocks (SimpleBlock, MegaTransformerBlock)
├── attention.py       # Multi-query attention with rotary/ALiBi support
├── vae.py             # Base VAE class with perceptual losses
├── audio/             # Audio VAE, diffusion, discriminators, vocoders
├── image/             # Image VAE, diffusion, discriminators, recurrent VAE
└── text/              # Text feature extractors
```

### Key Class Hierarchy

**Transformer Stack:**
- `SimpleBlock` wraps a `ModuleList` of `MegaTransformerBlock`
- `MegaTransformerBlock` contains `MegaTransformerSelfAttention` + `SimpleFFN`
- Attention supports rotary embeddings, ALiBi bias, multi-query heads, KV caching

**Multimodal Model:**
- `MegaTransformerMultimodal` extends HuggingFace's `PreTrainedModel`
- `MegaTransformerMultimodalEncoder` handles modality-specific feature extraction
- Each modality has prelude blocks (input processing) before the main transformer

**Layer Structure:**
- `n_prelude_layers` - Initial processing layers
- `n_layers` - Main transformer stack
- `n_recurrent_layers` - Iterative deepening with early exit (KL divergence threshold)
- `n_coda_layers` - Final processing layers

### Configuration System

`MegaTransformerConfig` (in `utils/configuration.py`) extends HuggingFace's `PretrainedConfig` with 100+ parameters:

- **Core**: `hidden_size`, `n_layers`, `n_heads`, `intermediate_size`
- **Attention**: `use_rotary_embedding`, `use_alibi_bias`, `n_query_groups`
- **Norms**: `norm_type` ("layernorm"/"rmsnorm"), pre/post norms for attn and FFN
- **Recurrent**: `recurrent_mean_thinking_steps`, `recurrent_exit_criteria_threshold`, `recurrent_adapter_method`
- **Audio**: `audio_n_mels`, `audio_sample_rate`, encoder/decoder configs
- **Image**: `image_size`, `image_encoder_patch_size`, decoder configs

### Training Infrastructure

Custom trainers in `utils/training_utils.py`:
- `ImageVAEGANTrainer` / `AudioVAEGANTrainer` - Handle discriminator loss alternation
- `GrokFastMATrainer` / `GrokfastEMATrainer` - Gradient filtering for stable training

Callbacks for logging reconstructions to TensorBoard.

### Data Pipeline

Dataset loaders in `dataset_loading/`:
- VAE datasets load cached preprocessed data
- Diffusion datasets load latent codes from trained VAE
- Collators handle batching with padding

## DeepSpeed Configurations

- `ds_config.json` - Stage 0 (no ZeRO)
- `ds_config_zero-1.json` - Optimizer state sharding
- `ds_config_zero-2.json` - Gradient + optimizer sharding (recommended)
- `ds_config_zero-3.json` - Full parameter sharding
- `ds_config_int8.json` - INT8 quantization

## Key Dependencies

- PyTorch 2.6.0 with CUDA 12.4
- HuggingFace transformers 4.49.0
- DeepSpeed 0.16.4
- librosa for audio processing
- rotary-embedding-torch for RoPE

## Outputs

- Checkpoints saved to `runs/<model_type>/<run_name>/checkpoint-STEP/`
- TensorBoard logs in `logs/`
- Cached datasets in `cached_datasets/`
- Generated samples in `inference/generated/`