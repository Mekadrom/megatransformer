# World Model Evaluation Scripts

Standalone eval scripts that load a checkpoint and compute quantitative metrics. Each script logs results to stdout and optionally to TensorBoard. The step number is auto-inferred from the checkpoint path (e.g. `checkpoint-3000` → step 3000) unless `--step` is provided.

## Text Perplexity

Cross-entropy and perplexity on held-out text (teacher-forcing, no generation).

```bash
python -m megatransformer.scripts.eval.world.eval_text_perplexity --checkpoint_path runs/world/my_run/checkpoint-3000 --config small_sum_dit --text_cache_dir ./cached_datasets/text_pile_2048 --split val --bf16 --tie_word_embeddings --log_dir runs/world/my_run
```

## Voice Transcription (WER)

Generates text transcripts from voice inputs and computes Word Error Rate.

Requires: `pip install jiwer`

```bash
python -m megatransformer.scripts.eval.world.eval_voice_transcription --checkpoint_path runs/world/my_run/checkpoint-3000 --config small_sum_dit --include_modes text,voice --cache_dir ./cached_datasets/text_pile_2048 --voice_cache_dir ./cached_datasets/voice_sive_train --split val --bf16 --tie_word_embeddings --log_dir runs/world/my_run
```

## Image Transcription (CLIPScore)

Generates text captions from images and scores them against the original image using CLIP.

Requires: `open_clip_torch`

```bash
python -m megatransformer.scripts.eval.world.eval_image_transcription --checkpoint_path runs/world/my_run/checkpoint-3000 --config small_sum_dit --include_modes text,image --cache_dir ./cached_datasets/text_pile_2048 --image_cache_dir ./cached_datasets/image_vae --split val --bf16 --tie_word_embeddings --log_dir runs/world/my_run --image_vae_decoder_config litevae
```

## Voice Synthesis (MCD + SIVE Cosine Similarity)

Generates voice from text prompts and computes Mel Cepstral Distortion against ground-truth mel spectrograms. Optionally saves generated .wav files.

```bash
python -m megatransformer.scripts.eval.world.eval_voice_synthesis --checkpoint_path runs/world/my_run/checkpoint-3000 --config small_sum_dit --include_modes text,voice --cache_dir ./cached_datasets/text_pile_2048 --voice_cache_dir ./cached_datasets/voice_sive_train --split val --bf16 --tie_word_embeddings --log_dir runs/world/my_run --voice_smg_checkpoint_path ./runs/smg/my_smg/checkpoint-300000 --voice_smg_config medium_decoder_only_1d_3x --voice_smg_latent_channels 128 --static_speaker_embedding_path ./logs/speaker_embedding_1.pt --save_audio ./eval_audio
```

## Image Synthesis (CLIPScore + FID)

Generates images from text prompts and computes CLIPScore (text-image alignment) and FID (distribution quality). Optionally saves generated images.

Requires: `open_clip_torch`, `torchmetrics` (for FID)

```bash
python -m megatransformer.scripts.eval.world.eval_image_synthesis --checkpoint_path runs/world/my_run/checkpoint-3000 --config small_sum_dit --include_modes text,image --cache_dir ./cached_datasets/text_pile_2048 --image_cache_dir ./cached_datasets/image_vae --split val --bf16 --tie_word_embeddings --log_dir runs/world/my_run --image_vae_decoder_config litevae --save_images ./eval_images
```

## Standalone Visualization

Runs the full training visualization callback (same scenarios as during training) against a checkpoint, logging to a separate TensorBoard directory.

```bash
python -m megatransformer.scripts.eval.world.visualize --checkpoint_path runs/world/my_run/checkpoint-3000 --config small_sum_dit --cache_dir ./cached_datasets/text_pile_2048 --voice_cache_dir ./cached_datasets/voice_sive_train --image_cache_dir ./cached_datasets/image_vae --include_modes text,voice,image --log_dir runs/eval_viz --step 3000 --bf16 --use_memorization_dataset --max_samples 50 --vocoder_config hifigan --voice_smg_checkpoint_path ./runs/smg/my_smg/checkpoint-300000 --voice_smg_config medium_decoder_only_1d_3x --voice_smg_latent_channels 128 --static_speaker_embedding_path ./logs/speaker_embedding_1.pt --image_vae_decoder_config litevae
```

## TensorBoard Logging

All scripts support `--log_dir` to write metrics to TensorBoard. Point `--log_dir` at your training run directory to overlay eval metrics on training curves:

```bash
--log_dir runs/world/my_run
```

Metrics appear under the `eval/` prefix:
- `eval/text_perplexity`, `eval/text_cross_entropy`, `eval/text_bits_per_token`
- `eval/voice_transcription_wer_corpus`, `eval/voice_transcription_wer_mean`
- `eval/image_transcription_clipscore_mean`
- `eval/voice_synthesis_mcd_mean`, `eval/voice_synthesis_sive_cosine_mean`
- `eval/image_synthesis_clipscore_mean`, `eval/image_synthesis_fid`

## Memorization Testing

To evaluate on training data (memorization test), use `--split train`:

```bash
--split train --use_memorization_dataset --max_samples 50
```
