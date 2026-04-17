# Research TODO & Ablation Plan

## Immediate (Pre-Training)

- [x] Fix text coda KV cache pollution in prompt initialization (transcription tasks)
- [x] Fix TemporalRefine train/inference mismatch → FramewiseRefine for voice coda
- [x] Add EOS token to training data (collator)
- [x] Fix speaker ID remapping (global via stat-shards, not per-split)
- [x] Re-preprocess voice data (correct capitalization, dense speaker IDs)
- [x] Re-preprocess text data (2048 sequence length) — text_pile_2048_train, 500k samples ~1B tokens
- [x] Run stat-shards on new voice train + val shards — 1316 speakers, consistent across splits
- [x] Validate all preprocessing with shard inspection before starting training

## 240M Base Model Training

- [ ] Train 240M model on full data (1M text, 132k voice, 800k image) for 1-3 epochs
- [ ] Monitor convergence per modality — voice will wrap ~7.5x per text epoch
- [ ] Run eval scripts (perplexity, WER, CLIPScore, MCD) at checkpoints
- [x] Confirm EOS prediction works (model stops generating when appropriate)
- [ ] Observe cross-modal consecutive generation (image→image, voice→voice) — curiosity, meaningful only on real data
  - **Early memorization finding**: at step 2500 with 50 examples, image→image reproduces the input image in 2/4 cases and cross-fades between memorized examples in 2/4. Suggests the recurrent block encodes image content that conditions DiT output, not random generation. Worth revisiting on real data for thematic/categorical coherence.

## Architecture Ablations (240M scale)

### Recurrent Block
- [ ] **Recurrent vs non-recurrent**: matched params/FLOPs comparison. Train a standard transformer with equivalent total compute budget (account for ~32 iterations). This is the key contribution for a paper.
- [ ] **Iteration depth**: ablate mean_thinking_steps (8, 16, 32, 64). Does more thinking help cross-modal tasks more than same-modal?
- [ ] **Block count vs iteration count**: e.g., 6 blocks × 32 iters vs 12 blocks × 16 iters at matched compute

### Coda Design
- [x] **Positional encoding cleanup**: text prelude stacked sinusoidal PE + RoPE (redundant). Ablation showed RoPE-only improves all text metrics with no effect on voice/image. Applied to default `small_sum_dit` config.
- [ ] **Coda RoPE**: try disabling RoPE in codas (`use_rotary_embedding=False`) — codas apply sequential positions on uninterleaved tokens which may encode misleading positions.
- [ ] **Voice start embedding**: learned `nn.Parameter` for voice position 0 instead of zero vector. Cheap experiment, may improve first-frame prediction.
- [ ] **Scheduled sampling for voice**: gradually replace teacher forcing with model predictions during training to reduce exposure bias. May close the train/inference gap for voice synthesis.

### Modality Balance
- [ ] **Loss weighting sweep**: the whitened losses should be roughly comparable, but verify by sweeping emphasis multipliers
- [ ] **Voice prelude output norm**: image prelude has LayerNorm to prevent scale dominance. Voice interleaved activations have std=2.73 vs text std=1.07. Try adding output norm to voice prelude.

## Scaling to 1B

- [ ] Design 1B config: scale recurrent block (larger d_model or more blocks), keep codas relatively thin
- [ ] Expand voice data: add GigaSpeech or LibriLight (requires SIVE + CVAE retraining on new distribution)
- [ ] Expand text data: more Pile, or switch to FineWeb/RedPajama
- [ ] Image data is sufficient at 800k for 1B
- [ ] Target hardware: 4×4090 (2-4 weeks) or 8×H100-NVLink rental (~1 week, ~$2600)

## Post-Hoc Modality Addition (Ablation Study)

Test whether new modalities can be added without full retraining:

- [ ] Pretrain base model (text + voice + image)
- [ ] Freeze recurrent block (or apply LoRA)
- [ ] Add new audio effects prelude + coda + boundary tokens
- [ ] Expand token embedding and LM head for new special tokens
- [ ] Finetune on audio effects data
- [ ] **Ablations**:
  - Full finetune vs frozen backbone + new modules only
  - LoRA on recurrent block vs full freeze
  - Convergence speed vs training from scratch
  - Catastrophic forgetting on existing modalities
- [ ] Target finding: "adding a new modality requires training only ~15% of parameters while preserving existing capabilities"

## Instruction Finetuning (Post Base Model)

- [ ] Text-only SFT first (OpenAssistant or SlimOrca) — teach instruction following
- [ ] Image SFT: adapt LLaVA-Instruct (preprocess images through LiteVAE, wrap in BOI/EOI format)
- [ ] Voice SFT: construct from LibriSpeech + templated instructions
- [ ] Add instruction template special tokens to vocabulary
- [ ] Loss masking: only compute loss on assistant response tokens
- [ ] Consider DPO with UltraFeedback for text alignment
- [ ] May need increased max_seq_len (2048-4096) for multi-turn conversations

## Interpretability / Mechanistic Analysis

### Thought State Analysis
- [ ] Record full thought trajectories (all ~32 iterations) for sample inputs
- [ ] Visualize convergence dynamics: smooth vs jumping, per-modality convergence rates
- [ ] Analyze early vs late iteration representations — what does the model "decide" at each stage?

### Sparse Autoencoders on Thought Space
- [ ] Train SAE on thought states pooled across iterations and positions
- [ ] Identify features that are modality-specific vs cross-modal
- [ ] Identify features that activate early (parsing) vs late (semantic integration)
- [ ] The shared representation space across all iterations is uniquely clean for this — one SAE covers all iteration depths

### Thought Injection / Steering
- [ ] Identify concept directions via SAE features
- [ ] Inject at different iterations (early = fundamental alteration, late = subtle nudge)
- [ ] Test cross-modal steering: inject a "dog" feature from an image context, observe effect on text/voice generation

### Cross-Modal Grounding
- [ ] Compare thought trajectories for semantically related inputs across modalities
- [ ] Do "dog" (text), a dog image, and a spoken "dog" converge to similar thought states?
- [ ] Quantify alignment (cosine similarity, CKA) across modalities in the shared thought space

## Future Modalities

### Audio (Non-Speech)
- [ ] Ambient sounds, sound effects, music
- [ ] Would need a different encoder than SIVE (AudioMAE or BEATs pretrained on AudioSet)
- [ ] Good candidate for the post-hoc modality addition experiment

### Video
- [ ] Architecturally straightforward: sequence of image latents + temporal position encoding
- [ ] Main challenge is compute and sequence length (5s @ 8fps × 64 patches = 2560 tokens)
- [ ] Would likely need 4096+ max_seq_len
- [ ] Datasets: WebVid, HowTo100M

## Paper Angles

1. **Primary**: Huginn-style recurrent reasoning applied to multimodal generation. Show that iterative thinking in a shared representation space enables cross-modal reasoning with fewer parameters than standard transformers.
2. **Modularity**: demonstrate post-hoc modality addition with frozen/LoRA'd backbone.
3. **Interpretability**: thought trajectory analysis and SAE features in the recurrent multimodal space — uniquely enabled by the shared-weight iterative architecture.
