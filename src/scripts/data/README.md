# Data Preprocessing

Main entry point for all preprocessing is `preprocess_dataset.py`. Each modality has a subcommand with its own arguments.

## Text Preprocessing

Tokenizes text documents into shards of token IDs using the Mistral tokenizer.

```bash
PYTHONPATH=src python3 -m scripts.data.preprocess_dataset text --dataset_name monology/pile-uncopyrighted --split train --output_dir ../cached_datasets/text_pile_train --max_seq_len 2048 --max_samples 500000 --streaming
```

Validation split (offset to avoid overlap with training data):

```bash
PYTHONPATH=src python3 -m scripts.data.preprocess_dataset text --dataset_name monology/pile-uncopyrighted --split train --output_dir ../cached_datasets/text_pile_val --max_seq_len 2048 --max_samples 5000 --start_idx 2000000 --streaming
```

## Audio/Voice Preprocessing

Extracts SIVE features, speaker embeddings, mel spectrograms, F0, and tokenized transcripts from audio datasets. Used for both the `audio` and `voice` modalities (same shard format, separate directories).

```bash
PYTHONPATH=src python3 -m scripts.data.preprocess_dataset audio --dataset_name mozilla-foundation/common_voice_17_0 --dataset_config en --split train --output_dir ../cached_datasets/voice_sive_train --sive_checkpoint_path ./checkpoints/sive --sive_config tiny_deep_3xdownsample_conv2d_batchnorm_0_0_layer10 --compute_speaker_embeddings --extract_f0 --save_mel_specs --tokenize_text --max_seq_len 128 --streaming
```

Multi-GPU (run one per GPU, shards are interleaved):

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 -m scripts.data.preprocess_dataset audio --gpu_id 0 --total_gpus 2 ... &
PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 python3 -m scripts.data.preprocess_dataset audio --gpu_id 1 --total_gpus 2 ... &
```

## Image Preprocessing

Encodes images through a VAE (e.g. LiteVAE) into latent tensors, with tokenized captions.

```bash
PYTHONPATH=src python3 -m scripts.data.preprocess_dataset image-vae --dataset_name ... --split train --output_dir ../cached_datasets/image_vae_train --image_size 256 --tokenize_text --max_seq_len 128
```

## stat-shards

Builds the `shard_index.json` manifest after preprocessing. Also globally remaps speaker IDs to dense sequential integers across train and val splits.

```bash
PYTHONPATH=src python3 -m scripts.data.preprocess_dataset stat-shards --output_dir ../cached_datasets/voice_sive_train --speaker_id_column speaker_ids --additional_shard_dirs ../cached_datasets/voice_sive_val
```

Run this after all GPUs finish preprocessing and after both train/val splits are complete. The `--additional_shard_dirs` flag ensures speaker IDs are consistent across splits.

## offset_speaker_ids

Offsets speaker IDs in one shard directory to avoid collisions when merging datasets from different sources. Run before copying shards into an existing directory.

```bash
# Check current max speaker ID in existing shards
PYTHONPATH=src python3 -m scripts.data.offset_speaker_ids --shard_dir ../cached_datasets/voice_existing_train --dry_run

# Offset new shards above existing max
PYTHONPATH=src python3 -m scripts.data.offset_speaker_ids --shard_dir ../cached_datasets/voice_new_train --offset_above ../cached_datasets/voice_existing_train

# Then merge and re-index
cp ../cached_datasets/voice_new_train/shard_*.pt ../cached_datasets/voice_existing_train/
PYTHONPATH=src python3 -m scripts.data.preprocess_dataset stat-shards --output_dir ../cached_datasets/voice_existing_train --speaker_id_column speaker_ids --additional_shard_dirs ../cached_datasets/voice_existing_val
```

## World Model Data

The world model training uses `MultimodalShardedDataset` which loads from separate per-modality shard directories. The `MultimodalDataCollator` handles:

- Injecting boundary tokens (BOV/EOV, BOI/EOI) and placeholder tokens
- Randomly choosing synthesis vs transcription direction per sample
- Appending EOS tokens (skipped for truncated text-only samples)
- Padding within batches

No separate preprocessing step is needed for the world model — it consumes the text, voice, and image shards directly.
