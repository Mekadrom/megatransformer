# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Communications with user

The user will provide larger text context, images, and other files for analysis in the `logs/` folder, with most text snippets going in `logs/claude.txt`. Do not modify these files under any circumstances, but if the user says "I have provided..." or some variant thereof, you may assume the user has provided the relevant files under `logs/` or explicity in `logs/claude.txt`.

## Project Overview

MegaTransformer is a highly configurable transformer module that constructs models from YAML configurations. It supports multimodal training (text, audio, image) and implements modern transformer architectures.

## Development Commands

### Training

**Main transformer training (DeepSpeed ZeRO-2):**
```bash
deepspeed --num_gpus=2 pretrain_wm.py \
    --use_deepspeed \
    --bf16 \
    --run_name my_run_name \
    --config gpt2_small \
    --max_steps 300000 \
    --gradient_accumulation_steps 8 \
    --use_gradient_checkpointing \
    --deepspeed_config ds_config_zero-2.json
```

**Vocoder training:**
```bash
python pretrain_vocoder.py \
    --run_name vocoder_run \
    --vocoder_type tiny_lightheaded_freq_domain_vocoder \
    --train_data_dir cached_datasets/librispeech_train_vocoder_cached \
    --val_data_dir cached_datasets/librispeech_val_vocoder_cached
```

**Audio diffusion training:**
```bash
python pretrain_audio_diffusion.py \
    --run_name diffusion_run \
    --diffusion_type small_dit \
    --train_data_dir cached_datasets/librispeech_train_diffusion_cached \
    --val_data_dir cached_datasets/librispeech_val_diffusion_cached
```

**Audio VAE training (with speaker embedding conditioning):**
```bash
python pretrain_audio_vae.py \
    --run_name audio_vae_run \
    --config small \
    --latent_channels 8 \
    --speaker_embedding_dim 192 \
    --train_cache_dir cached_datasets/librispeech_train_diffusion_cached \
    --val_cache_dir cached_datasets/librispeech_val_diffusion_cached
```

**Image VAE training:**
```bash
python pretrain_image_vae.py \
    --run_name image_vae_run \
    --config gan_4_2 \
    --train_data_dir cached_datasets/image_train_cached \
    --val_data_dir cached_datasets/image_val_cached
```

**Image diffusion training (latent space):**
```bash
python pretrain_image_diffusion.py \
    --run_name image_diffusion_run \
    --diffusion_type small_dit \
    --train_data_dir cached_datasets/image_train_diffusion_latents \
    --val_data_dir cached_datasets/image_val_diffusion_latents \
    --vae_checkpoint runs/image_vae_run/checkpoint-XXX
```

### Dataset Preprocessing

**Vocoder dataset:**
```bash
python preprocess_vocoder_dataset.py \
    --output_dir cached_datasets/librispeech_train_vocoder_cached \
    --split train.360
```

**Audio diffusion dataset:**
```bash
python preprocess_audio_diffusion_dataset.py \
    --output_dir cached_datasets/librispeech_train_diffusion_cached \
    --split train.360 \
    --max_conditions 512
```

**Image VAE dataset:**
```bash
python preprocess_image_vae_dataset.py \
    --output_dir cached_datasets/image_train_cached \
    --dataset_name your_dataset
```

**Image diffusion latent dataset (requires trained VAE):**
```bash
python preprocess_image_diffusion_dataset.py \
    --output_dir cached_datasets/image_train_diffusion_latents \
    --vae_checkpoint runs/image_vae_run/checkpoint-XXX \
    --vae_config small
```

### Monitoring

```bash
./tensorboard.sh  # Starts TensorBoard on port 6006
```

## Core Architecture

### Model Registry Pattern

Each model type has a `model_config_lookup` dictionary that maps config names to factory functions:

- `model/megatransformer_causal.py`: `lookup` dict with configs like `gpt2_small`, `modern_medium`
- `model/megatransformer_multimodal.py`: Multimodal model configs
- `model/megatransformer_recurrent.py`: Recurrent transformer configs
- `model/audio/vocoders/vocoders.py`: Vocoder configs like `tiny_lightheaded_freq_domain_vocoder`, `tiny_splitband_freq_domain_vocoder`
- `model/audio/diffusion.py`: Audio diffusion configs like `small_dit`, `medium_dit`
- `model/audio/vae.py`: Audio VAE configs like `small`, `small_wide`
- `model/image/vae.py`: Image VAE configs like `gan_4_2`, `small_4_2`
- `model/image/diffusion.py`: Image diffusion configs

### Training Script Selection (`pretrain_wm.py`)

The main training script auto-selects model type based on config:
- Multiple modalities or non-text → `megatransformer_multimodal`
- Config contains 'recurrent' → `megatransformer_recurrent`
- Default → `megatransformer_causal`

### Audio Pipeline

**Vocoder architecture** (`model/audio/vocoders/`):
- `FrequencyDomainVocoderBase`: Predicts STFT magnitude + phase, uses iSTFT for waveform
- `LightHeadedFrequencyDomainVocoder`: Lightweight variant with reduced head dimensions
- `SplitBandFrequencyDomainVocoder`: Separate heads for low/high frequency bands
- `VocoderWithLoss`: Wrapper that combines vocoder with loss computation

**Diffusion** (`model/audio/diffusion.py`):
- DiT-style architecture for mel spectrogram generation
- Text conditioning via T5 embeddings with cross-attention

**Shared utilities:**
- `SharedWindowBuffer`: Caches STFT windows across components
- `model/audio/criteria.py`: Loss functions (mel reconstruction, phase-intransitive, discriminator losses)

### VAE Architecture

**Base VAE** (`model/vae.py`): Generic VAE wrapper combining encoder, decoder, and loss computation.

**Audio VAE** (`model/audio/vae.py`):
- Compresses mel spectrograms [1, 80, T] → latent [C, 10, T/75]
- Speaker conditioning via FiLM (Feature-wise Linear Modulation) in decoder
- Speaker embeddings from ECAPA-TDNN (192-dim) are NOT encoded in latent space

**Image VAE** (`model/image/vae.py`):
- Standard image VAE with optional GAN discriminator training

### Image Pipeline

**Image diffusion** (`model/image/diffusion.py`):
- DiT-style architecture operating on VAE latents
- Cross-attention for text conditioning
- Uses `ImageCrossAttentionBlockSimple` with 1x1 convolutions for any spatial size

### Configuration System

`MegaTransformerConfig` in `megatransformer_utils.py` defines all model hyperparameters. Key groups:
- Attention: `n_heads`, `n_query_groups`, `d_queries`, `d_values`
- Position encoding: `use_rotary_embedding`, `use_alibi_bias`, `use_positional_embedding`
- FFN: `ffn_type` (mlp, moe), `intermediate_activation` (gelu, swiglu, etc.)
- Normalization: `norm_type` (layernorm, rmsnorm), `pre_attn_norm`, `post_attn_norm`
- Recurrent: `recurrent_mean_thinking_steps`, `recurrent_exit_criteria`

## File Structure

- `runs/`: Training outputs organized by run name
- `cached_datasets/`: Preprocessed datasets (.pt files)
- `inference/examples/`: Test audio files for vocoder evaluation
- `ds_config_zero-*.json`: DeepSpeed configurations (ZeRO-2 recommended)

## Key Implementation Notes

- Vocoders output `(waveform, stft)` tuple; the STFT is used for loss computation
- Audio uses 16kHz sample rate, 80 mel bins, n_fft=1024, hop_length=256 by default
- Dataset classes expect cached .pt files with tensors; collators handle padding/batching
