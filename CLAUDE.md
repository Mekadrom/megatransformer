# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
MegaTransformer is a highly configurable transformer module that constructs models from YAML configurations. It supports multimodal training (text, audio, image) and implements modern transformer architectures with cutting-edge features.

## Core Architecture

### Main Model Types
- **megatransformer_causal**: Standard causal language models (GPT-style)
- **megatransformer_multimodal**: Multimodal models supporting text/audio/image inputs
- **megatransformer_recurrent**: Recurrent transformer with dynamic iteration depth
- **megatransformer_diffusion**: Diffusion model implementations

### Key Components
- **model/**: Contains all transformer implementations and modules
  - `megatransformer_multimodal.py`: Main multimodal architecture
  - `megatransformer_blocks.py`: Core transformer blocks
  - `megatransformer_attn.py`: Attention mechanisms with RoPE support
  - `megatransformer_modules.py`: Shared utility modules
- **dataset_loading/**: Multimodal dataset loaders
  - `multimodal_dataset.py`: Main dataset class supporting mixed modalities
  - `image_loading.py`, `audio_loading.py`: Modality-specific loaders
- **megatransformer_utils.py**: Core utilities for model loading, argument parsing, and configuration

### Model Selection Logic
The training script (`pretrain_wm.py`) automatically selects the appropriate model type:
- Multiple modalities or non-text modes → `megatransformer_multimodal`
- Config contains 'recurrent' → `megatransformer_recurrent`
- Default → `megatransformer_causal`

## Development Commands

### Training
Primary training command using DeepSpeed ZeRO-2:
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

### Monitoring
Launch TensorBoard to monitor training:
```bash
./tensorboard.sh
```
This starts TensorBoard on port 6006 with logs from the `runs/` directory.

### DeepSpeed Configurations
Available DeepSpeed configs:
- `ds_config_zero-2.json`: Recommended for best compatibility/performance
- `ds_config_zero-1.json`, `ds_config_zero-3.json`: Alternative configurations
- `ds_config.json`: Basic configuration

## Key Implementation Features

### Modern Transformer Features
- **RoFormer**: Rotary Position Embedding for better length generalization
- **ReZero**: Weight initialization for stable deep training
- **Post-LayerNorm**: Better than pre-LN for training stability
- **Admin**: Training stabilization for very large models
- **GLU variants**: SwiGLU and other gated activations in FFN
- **Mixture of Experts**: Million-expert FFN implementation
- **Grokfast**: Slow gradient amplification for stable training

### Multimodal Support
- Text tokenization with configurable tokenizers
- Audio waveform processing and encoding
- Image patch encoding and processing
- Mixed-modality training with shared transformer backbone

### Training Infrastructure
- DeepSpeed integration for distributed training
- Gradient checkpointing for memory efficiency
- Custom trainers and callbacks in `custom_trainers.py` and `custom_callbacks.py`
- Model saving/loading with automatic resumption

## File Structure Notes
- Training outputs go to `runs/` directory organized by run name
- Cached datasets stored in `cached_datasets/`
- Generated outputs in `generated_images/` for image tasks
- Model checkpoints include DeepSpeed state for proper resumption