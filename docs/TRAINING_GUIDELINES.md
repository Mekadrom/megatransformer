# Training Guidelines for Generative Models

Best practices and hyperparameter recommendations for training vocoders, VAEs, and diffusion models.

---

## Table of Contents

1. [General Principles](#general-principles)
2. [Vocoder (~5M params, MRF/HiFi-GAN style)](#vocoder-5m-params)
3. [Audio VAE (~5M param decoder)](#audio-vae-5m-param-decoder)
4. [Image VAE (~5M param decoder)](#image-vae-5m-param-decoder)
5. [Audio Diffusion (~45M params)](#audio-diffusion-45m-params)
6. [Image Diffusion (~45M params)](#image-diffusion-45m-params)
7. [GAN Training Dynamics](#gan-training-dynamics)
8. [Common Pitfalls](#common-pitfalls)

---

## General Principles

### Optimizer: AdamW

| Model Type | Beta1 | Beta2 | Epsilon | Weight Decay |
|------------|-------|-------|---------|--------------|
| VAE / Diffusion | 0.9 | 0.999 | 1e-8 | 0.01 |
| GAN Generator | 0.8 | 0.99 | 1e-8 | 0.0 - 0.01 |
| GAN Discriminator | 0.8 | 0.99 | 1e-8 | 0.0 |

**Notes**:
- **(0.8, 0.99)** for GANs comes from HiFi-GAN and is widely adopted for audio
- **(0.9, 0.999)** is standard for non-adversarial training
- **(0.5, 0.9)** or **(0.0, 0.9)** are sometimes used for discriminators to increase responsiveness, but can be unstable
- Weight decay on discriminators is typically **zero** - regularizing D can hurt G's learning signal
- Weight decay on generators is optional; use 0.01 if you see overfitting

### LR Schedule Types

| Schedule | Best For | Description |
|----------|----------|-------------|
| Cosine with warmup | Diffusion, VAE | Smooth decay, good final convergence |
| Constant with warmup | GANs, short runs | Stable throughout, no decay tuning |
| Linear decay | Long GAN training | Gradual reduction helps late-stage |
| Exponential decay | HiFi-GAN style | 0.999 per epoch is common |

### Warmup Guidelines

| Training Length | Warmup Steps | Warmup % |
|-----------------|--------------|----------|
| < 50k steps | 500-1000 | 2-5% |
| 50k-200k steps | 1000-2000 | 1-2% |
| > 200k steps | 2000-5000 | ~1% |

---

## Vocoder (~5M params)

HiFi-GAN, BigVGAN, and similar MRF (Multi-Receptive Field) vocoders.

### Hyperparameters

| Parameter | Recommended | Range | Notes |
|-----------|-------------|-------|-------|
| Learning Rate (G) | **2e-4** | 1e-4 - 4e-4 | HiFi-GAN default |
| Learning Rate (D) | **2e-4** | 1e-4 - 4e-4 | Same as G typically |
| Weight Decay | 0.0 | 0 - 0.01 | Often zero for vocoders |
| Batch Size | 16-32 | 8-64 | Limited by audio length |
| AdamW Betas | (0.8, 0.99) | - | Standard for audio GANs |
| LR Decay | 0.999/epoch | 0.995 - 0.9999 | Exponential decay |
| Total Steps | 500k-1M | - | Until quality saturates |

### GAN Configuration

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| GAN Loss Type | **Hinge loss** | Most stable for audio |
| G:D Parameter Ratio | **1:2 to 1:3** | D should be larger |
| Update Ratio | 1:1 | Update both every step |
| Discriminators | MSD + MPD | Multi-scale + Multi-period |

### Loss Weights (HiFi-GAN style)

| Loss Component | Weight | Notes |
|----------------|--------|-------|
| Adversarial (G) | 1.0 | Sum of all discriminator losses |
| Feature Matching | **2.0** | L1 on intermediate D features |
| Mel Spectrogram L1 | **45.0** | Critical - drives spectral accuracy |

**Total G loss** = `1.0 * adv_loss + 2.0 * fm_loss + 45.0 * mel_loss`

The high mel weight (45.0) ensures the generator prioritizes spectral fidelity while GAN losses add fine detail and naturalness.

### Alternative: Frequency-Domain Vocoders

If using complex STFT prediction instead of waveform:

| Loss Component | Weight |
|----------------|--------|
| Magnitude STFT | 1.0 |
| Phase loss (IF/GD) | 0.1 - 1.0 |
| Adversarial | 1.0 |
| Feature Matching | 2.0 - 10.0 |

### Training Tips

- **Audio segment length**: 1-2 seconds (8192-32768 samples at 16kHz)
- **Discriminator warmup**: Not usually needed; train jointly from start
- Use **gradient clipping** (max_norm=1.0) if you see instability
- MSD processes at multiple resolutions (1x, 0.5x, 0.25x)
- MPD uses periods [2, 3, 5, 7, 11] typically
- Train until mel-cepstral distortion (MCD) and PESQ/UTMOS plateau

---

## Audio VAE (~5M param decoder)

Variational autoencoder for mel spectrograms or raw audio.

### Hyperparameters

| Parameter | Recommended | Range |
|-----------|-------------|-------|
| Learning Rate | **1e-4** | 5e-5 - 3e-4 |
| Weight Decay | 0.01 | 0.001 - 0.1 |
| Batch Size | 32-64 | 16-128 |
| AdamW Betas | (0.9, 0.999) | - |
| Warmup Steps | 1000-2000 | - |
| LR Schedule | Cosine | - |

### GAN Configuration (if using adversarial training)

| Parameter | Recommended |
|-----------|-------------|
| GAN Loss Type | **Hinge loss** |
| G:D Parameter Ratio | 1:1 to 1:2 |
| D Learning Rate | 2e-4 |
| GAN Start Step | **10k-50k** (delayed) |

### Loss Weights

| Loss Component | Weight | Notes |
|----------------|--------|-------|
| Reconstruction (L1) | **1.0** | Base loss on mel spectrogram |
| KL Divergence | **1e-6 to 1e-4** | Use annealing |
| Multi-scale Mel | **1.0** | Multiple FFT sizes |
| Adversarial (G) | **0.1 - 0.5** | After GAN warmup |
| Feature Matching | **1.0 - 2.0** | Stabilizes GAN |

### KL Annealing Schedule

KL annealing prevents posterior collapse by letting reconstruction dominate early:

```
Steps 0-5k:       kl_weight = 0
Steps 5k-50k:     kl_weight = linear(0 -> target)
Steps 50k+:       kl_weight = target (1e-6 to 1e-4)
```

**Typical target**: `1e-6` for high-fidelity reconstruction, `1e-4` for better latent structure.

### Latent Space Design

| Latent Channels | Use Case |
|-----------------|----------|
| 4 | Minimal, good for small diffusion models |
| 8 | Balanced (common choice) |
| 16 | Rich representation, larger diffusion needed |

### Training Tips

- **Pre-train without GAN** for 10k-50k steps to stabilize reconstruction
- Monitor **latent KL** - if it collapses to 0, increase KL weight
- If reconstructions are blurry, add GAN or increase perceptual loss
- Use multi-scale spectral loss (multiple FFT sizes: 512, 1024, 2048)

---

## Image VAE (~5M param decoder)

Variational autoencoder for images (Stable Diffusion style).

### Hyperparameters

| Parameter | Recommended | Range |
|-----------|-------------|-------|
| Learning Rate | **1e-4** | 4.5e-6 - 2e-4 |
| Weight Decay | 0.0 - 0.01 | - |
| Batch Size | 8-32 | Memory dependent |
| AdamW Betas | (0.9, 0.999) | - |
| Warmup Steps | 1000-5000 | - |
| LR Schedule | Cosine | Or constant |

### GAN Configuration

| Parameter | Recommended |
|-----------|-------------|
| GAN Loss Type | **Hinge loss** |
| G:D Parameter Ratio | 1:1 |
| D Learning Rate | 1e-4 to 2e-4 |
| Discriminator | **PatchGAN** (NLayerDiscriminator) |
| GAN Start Step | 50k+ (after VAE stabilizes) |

### Loss Weights (SD-VAE style)

| Loss Component | Weight | Notes |
|----------------|--------|-------|
| Reconstruction (L1) | **1.0** | Pixel-space L1 |
| KL Divergence | **1e-6** | Very low for images |
| Perceptual (LPIPS) | **1.0** | VGG-based perceptual |
| Adversarial (G) | **0.5** | PatchGAN loss |
| Feature Matching | **0.0** | See note below |

**Notes**:
- Stable Diffusion's VAE uses extremely low KL weight (1e-6) prioritizing reconstruction quality
- The LPIPS perceptual loss is crucial for visual quality
- **Feature matching is typically NOT used** for image VAEs (unlike audio). LPIPS already provides perceptual feature supervision. If you do use feature matching, keep it low (0.1-0.5) to avoid over-smoothing.

### Discriminator Selection

| Discriminator | Params | Best For | Notes |
|---------------|--------|----------|-------|
| **PatchGAN** | ~700K-2.8M | Default choice | NLayerDiscriminator (n=3), works at any resolution |
| **StyleGAN** | ~20-25M | High-quality 256+ | Progressive, needs image_size, use with R1 |
| **Multi-scale** | ~2-4M | Varied textures | Multiple PatchGANs at different scales |

**Recommendation**:
- For **standard VAE training**: Use **PatchGAN** (NLayerDiscriminator with 3 layers). It's efficient, proven, and resolution-agnostic.
- For **high-quality 256x256+**: Consider **StyleGAN discriminator** with R1 penalty - but only if you have sufficient compute and training time.
- StyleGAN discriminators are ~10-20x larger and slower but can produce slightly sharper details at high resolutions.
- If using StyleGAN discriminator, you **must use R1 gradient penalty** (no spectral norm by design).

### Latent Space Design

For 256x256 images:
- **Downscale factor**: 8x (256 → 32)
- **Latent channels**: 4
- **Latent shape**: 32 x 32 x 4

For 512x512 images:
- **Latent shape**: 64 x 64 x 4

### Training Tips

- LPIPS perceptual loss >> pixel L1 for visual quality
- Use **EMA** of weights (decay 0.999-0.9999) for final model
- **R1 gradient penalty** on discriminator improves stability:
  - Weight: 10.0
  - Apply every 16 steps
- Data augmentation: random crop, horizontal flip (no color augment)

---

## Audio Diffusion (~45M params)

Latent diffusion model using frozen Audio VAE.

### Hyperparameters

| Parameter | Recommended | Range |
|-----------|-------------|-------|
| Learning Rate | **1e-4** | 5e-5 - 3e-4 |
| Weight Decay | 0.01 | 0.0 - 0.1 |
| Batch Size (per GPU) | 32-64 | - |
| **Effective Batch Size** | **256-512** | Via gradient accumulation |
| AdamW Betas | (0.9, 0.999) | - |
| Warmup Steps | 2000-5000 | 1-2% of total |
| LR Schedule | Cosine | - |
| **EMA Decay** | **0.9999** | Critical for inference |

### Diffusion Settings

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Noise Schedule | **Cosine** | Or scaled linear |
| Prediction Type | **v-prediction** | More stable than epsilon |
| Training Timesteps | 1000 | Standard |
| Inference Steps | 50-100 | With DDIM/DPM++ |
| **Min-SNR Gamma** | **5.0** | Balances timestep losses |

### Loss Configuration

| Component | Notes |
|-----------|-------|
| Loss | MSE on v-prediction (or epsilon) |
| Weighting | Min-SNR-gamma (5.0) recommended |

Min-SNR weighting prevents early timesteps from dominating the loss.

### Classifier-Free Guidance (CFG)

| Parameter | Recommended |
|-----------|-------------|
| Conditioning dropout | 10-20% |
| Inference CFG scale | **3.0 - 7.0** |

Audio typically uses **lower CFG** than images (3-7 vs 7-12).

### Training Tips

- **Effective batch size matters**: 256+ significantly improves quality
- **Always use EMA model** for inference (not training weights)
- Freeze VAE completely (no gradients)
- Cross-attention for conditioning (text, class embeddings)
- Train 200k-500k steps depending on dataset size

---

## Image Diffusion (~45M params)

Latent diffusion model using frozen Image VAE.

### Hyperparameters

| Parameter | Recommended | Range |
|-----------|-------------|-------|
| Learning Rate | **1e-4** | 1e-5 - 2e-4 |
| Weight Decay | 0.01 | 0.0 - 0.1 |
| Batch Size (per GPU) | 32-128 | - |
| **Effective Batch Size** | **256-2048** | Larger = better |
| AdamW Betas | (0.9, 0.999) | - |
| Warmup Steps | 5000-10000 | - |
| LR Schedule | Cosine | - |
| **EMA Decay** | **0.9999** | - |

### Diffusion Settings

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Noise Schedule | **Scaled linear** | SD-style, or cosine |
| Prediction Type | v-prediction or epsilon | Both work |
| Training Timesteps | 1000 | - |
| Inference Steps | 20-50 | DPM++ 2M recommended |
| Min-SNR Gamma | 5.0 | - |

### Classifier-Free Guidance

| Parameter | Recommended |
|-----------|-------------|
| Conditioning dropout | 10% |
| Inference CFG scale | **7.0 - 12.0** |

### Training Tips

- **Larger effective batch size** is more important than longer training
- Use **DPM++ 2M** or **DDIM** samplers for fast inference
- Center crop + random flip for augmentation
- Consider **offset noise** (0.1) for better dark/bright image generation
- Train 100k-500k+ steps depending on dataset

---

## GAN Training Dynamics

### Healthy Training Indicators

| Metric | Healthy Range | Problem If |
|--------|---------------|------------|
| D(real) | 0.7 - 0.9 | > 0.95 (D too strong) |
| D(fake) | 0.3 - 0.6 | < 0.1 (mode collapse) |
| G loss | Slowly decreasing | Diverging or flat |
| D loss | Stable, ~0.5-1.5 | Near 0 (D won) |

### Hinge Loss vs Non-Saturating (Logistic) Loss

**Hinge Loss** (recommended):
```
D_loss = mean(relu(1 - D(real))) + mean(relu(1 + D(fake)))
G_loss = -mean(D(fake))
```

**Non-Saturating Loss**:
```
D_loss = -mean(log(D(real))) - mean(log(1 - D(fake)))
G_loss = -mean(log(D(fake)))
```

| Aspect | Hinge | Non-Saturating |
|--------|-------|----------------|
| Stability | More stable | Can saturate |
| Gradients | Bounded, linear | Can vanish |
| Use when | Default choice | Simpler setups |

**Recommendation**: Use **hinge loss** for audio and image GANs.

### Balancing G and D

**If D is too strong** (D_real > 0.95, D_fake < 0.1):
1. Reduce D learning rate by 2x
2. Add label smoothing (real labels = 0.9)
3. Add instance noise to D inputs
4. Skip D updates (update G 2x per D update)
5. Add R1 gradient penalty

**If D is too weak** (D_real < 0.7, D_fake > 0.5):
1. Increase D capacity
2. Reduce G learning rate
3. Remove regularization from D
4. Check for bugs in loss computation

### G:D Parameter Ratio Guidelines

| Model Type | Ratio | Notes |
|------------|-------|-------|
| Vocoder | **1:2 to 1:3** | MSD+MPD are large |
| Audio VAE | 1:1 to 1:2 | Moderate D size |
| Image VAE | 1:1 | PatchGAN is lightweight |

---

## Common Pitfalls

### 1. Posterior Collapse in VAEs

**Symptom**: KL loss → 0, latents unused, blurry outputs

**Fixes**:
- KL annealing (start at 0)
- Free bits / KL thresholding
- Reduce decoder capacity
- Use discrete bottleneck (VQ-VAE)

### 2. GAN Mode Collapse

**Symptom**: Generator produces identical/similar outputs

**Fixes**:
- Feature matching loss
- Minibatch discrimination
- Spectral normalization
- Lower G learning rate
- R1 gradient penalty

### 3. Discriminator Dominates

**Symptom**: D loss → 0, G loss increases, no improvement

**Fixes**:
- TTUR (different LRs)
- Label smoothing
- Instance noise
- Skip D updates
- R1 penalty

### 4. Training Instability / NaN

**Symptom**: Loss spikes, NaN values, divergence

**Fixes**:
- Gradient clipping (max_norm=1.0)
- Lower learning rate
- bf16 instead of fp16
- Check data for NaN/Inf
- Reduce batch size

### 5. Diffusion Generates Noise

**Symptom**: Outputs are noisy or have artifacts

**Fixes**:
- Use EMA model (not training weights)
- Lower CFG scale
- More inference steps
- Check VAE is frozen
- Train longer

---

## Quick Reference

### Vocoder (5M, HiFi-GAN style)
```
LR: 2e-4 | Betas: (0.8, 0.99) | Decay: 0.999/epoch
Mel L1: 45.0 | FM: 2.0 | Adv: 1.0
GAN: Hinge | G:D ratio: 1:2-1:3
```

### Audio VAE (5M decoder)
```
LR: 1e-4 | WD: 0.01 | Betas: (0.9, 0.999)
KL: 1e-6→1e-4 (annealed) | Recon: 1.0 | GAN: 0.1-0.5
Delay GAN 10-50k steps
```

### Image VAE (5M decoder)
```
LR: 1e-4 | WD: 0.0 | Betas: (0.9, 0.999)
KL: 1e-6 | LPIPS: 1.0 | Recon: 1.0 | GAN: 0.5
PatchGAN with R1 penalty
```

### Audio Diffusion (45M)
```
LR: 1e-4 | WD: 0.01 | Betas: (0.9, 0.999)
Effective BS: 256+ | EMA: 0.9999 | Min-SNR: 5
CFG: 3-7 | Schedule: cosine | Pred: v-prediction
```

### Image Diffusion (45M)
```
LR: 1e-4 | WD: 0.01 | Betas: (0.9, 0.999)
Effective BS: 256-2048 | EMA: 0.9999 | Min-SNR: 5
CFG: 7-12 | Schedule: scaled-linear | Pred: v/epsilon
```

---

## References

- HiFi-GAN: https://arxiv.org/abs/2010.05646
- BigVGAN: https://arxiv.org/abs/2206.04658
- Stable Diffusion VAE: https://arxiv.org/abs/2112.10752
- Min-SNR Weighting: https://arxiv.org/abs/2303.09556
- Progressive Distillation: https://arxiv.org/abs/2202.00512