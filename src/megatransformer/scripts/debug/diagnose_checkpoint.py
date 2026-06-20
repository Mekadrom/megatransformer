"""
Diagnose a world model checkpoint by tracing activations and gradients
across all modalities (text, voice, image).

Usage:
    python -m megatransformer.scripts.debug.diagnose_checkpoint --checkpoint ./runs/world/<run>/checkpoint-<step> --config small_sum_dit --text_cache_dir ./cached_datasets/text_pile --voice_cache_dir ./cached_datasets/audio_sive_tiny_deep_3xdownsample_conv2d_batchnorm_0_0_layer10 --image_cache_dir ./cached_datasets/image_vae
"""

import argparse
import copy
import math
import os
import torch

from megatransformer.config.world.world_model import WORLD_MODEL_CONFIGS
from megatransformer.model.world.world_model import MegaTransformerWorldModel
from megatransformer.scripts.data.world.dataset import MultimodalShardedDataset
from megatransformer.scripts.data.world.data_collator import MultimodalDataCollator


def load_model(config_name, checkpoint_path, tie_word_embeddings=False,
               gen_query_mode=None, n_image_gen_positions=None,
               include_modes=None):
    config = copy.deepcopy(WORLD_MODEL_CONFIGS[config_name])
    if include_modes is not None:
        config.include_modes = include_modes
    if tie_word_embeddings:
        config.tie_word_embeddings = True
    if gen_query_mode is not None:
        config.gen_query_mode = gen_query_mode
    if n_image_gen_positions is not None:
        config.n_image_gen_positions = n_image_gen_positions
    model = MegaTransformerWorldModel(config)

    ckpt_file = os.path.join(checkpoint_path, 'pytorch_model.bin')
    if not os.path.exists(ckpt_file):
        safetensors_file = os.path.join(checkpoint_path, 'model.safetensors')
        if os.path.exists(safetensors_file):
            from safetensors.torch import load_file
            sd = load_file(safetensors_file, device='cpu')
        else:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")
    else:
        sd = torch.load(ckpt_file, map_location='cpu', weights_only=True)

    model_sd = model.state_dict()
    filtered_sd = {}
    skipped = []
    for k, v in sd.items():
        if k in model_sd and v.shape != model_sd[k].shape:
            skipped.append(f"  {k}: ckpt {list(v.shape)} vs model {list(model_sd[k].shape)}")
        else:
            filtered_sd[k] = v
    if skipped:
        print(f"  WARNING: skipped {len(skipped)} shape-mismatched params:")
        for s in skipped:
            print(s)
    model.load_state_dict(filtered_sd, strict=False)
    return model


def r(name, tensor, indent=2):
    if tensor is None:
        print(f"{' ' * indent}{name}: None")
        return
    t = tensor.float()
    print(f"{' ' * indent}{name}: shape={list(t.shape)}, std={t.std():.6f}, mean={t.mean():.6f}, range=[{t.min():.4f}, {t.max():.4f}]")


def save_latent_grid(latents: torch.Tensor, save_path: str, title: str):
    """Save a (B, C, H, W) latent tensor as a grid of per-sample, per-channel images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if latents.dim() != 4:
        return
    t = latents.detach().float().cpu()
    B, C, H, W = t.shape
    fig, axes = plt.subplots(B, C, figsize=(C * 1.5, B * 1.5), squeeze=False)
    for b in range(B):
        for c in range(C):
            ax = axes[b][c]
            ax.imshow(t[b, c].numpy(), cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            if b == 0:
                ax.set_title(f"c{c}", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"s{b}", fontsize=8)
    fig.suptitle(f"{title} | std={t.std():.4f} mean={t.mean():.4f}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


# ──────────────────────────────────────────────────────────────────────
# Learned parameter summary
# ──────────────────────────────────────────────────────────────────────

def report_learned_params(model):
    print("\n--- Learned Parameters ---")

    # Recurrent block
    rb = model.recurrent_block
    if rb.projection is not None:
        r("recurrent/projection.weight", rb.projection.weight)
    for i, block in enumerate(rb.recurrent_blocks):
        total_params = sum(p.numel() for p in block.parameters())
        weight_std = torch.cat([p.flatten() for p in block.parameters()]).std()
        print(f"  recurrent/block{i}: {total_params:,} params, weight std={weight_std:.6f}")

    # Text prelude
    if model.text_feature_extractor is not None:
        fe = model.text_feature_extractor
        print(f"  text_prelude: {sum(p.numel() for p in fe.parameters()):,} params")
        if hasattr(fe, 'embedding'):
            r("text_prelude/embedding.weight", fe.embedding.weight)

    # Text coda
    if model.text_generator is not None:
        tg = model.text_generator
        print(f"  text_coda: {sum(p.numel() for p in tg.parameters()):,} params")
        if hasattr(tg, 'lm_head'):
            r("text_coda/lm_head.weight", tg.lm_head.weight)

    # Voice prelude
    if model.voice_feature_extractor is not None:
        vfe = model.voice_feature_extractor
        print(f"  voice_prelude: {sum(p.numel() for p in vfe.parameters()):,} params")
        r("voice_prelude/projection.weight", vfe.projection.weight)

    # Voice coda
    if model.voice_generator is not None:
        vg = model.voice_generator
        print(f"  voice_coda: {sum(p.numel() for p in vg.parameters()):,} params")
        r("voice_coda/feature_projection.weight", vg.feature_projection.weight)
        if hasattr(vg, 'stop_head'):
            r("voice_coda/stop_head.weight", vg.stop_head.weight)
            print(f"  voice_coda/stop_head.bias: {vg.stop_head.bias.data.item():.4f}")

    # Audio prelude/coda (if present)
    if model.audio_feature_extractor is not None:
        print(f"  audio_prelude: {sum(p.numel() for p in model.audio_feature_extractor.parameters()):,} params")
    if model.audio_generator is not None:
        print(f"  audio_coda: {sum(p.numel() for p in model.audio_generator.parameters()):,} params")

    # Image prelude
    if model.image_feature_extractor is not None:
        ife = model.image_feature_extractor
        print(f"  image_prelude: {sum(p.numel() for p in ife.parameters()):,} params")

    # Image coda
    if model.image_generator is not None:
        cd = model.image_generator
        print(f"  image_coda: {sum(p.numel() for p in cd.parameters()):,} params, class={type(cd).__name__}")
        if hasattr(cd, 'bridge') and hasattr(cd.bridge, 'queries'):
            r("image_coda/bridge.queries", cd.bridge.queries)
        if hasattr(cd, 'output_scale'):
            r("image_coda/output_scale", cd.output_scale.data)
            r("image_coda/output_bias", cd.output_bias.data)

    # Image gen queries
    if hasattr(model, 'image_gen_queries'):
        r("image_gen_queries", model.image_gen_queries)
    else:
        print(f"  image_gen_queries: not present (positional_only mode)")
    if hasattr(model, 'image_gen_pos_embedding'):
        print(f"  image_gen_pos_embedding (frozen): std={model.image_gen_pos_embedding.pe.std():.4f}")


# ──────────────────────────────────────────────────────────────────────
# Forward pass with activation probes
# ──────────────────────────────────────────────────────────────────────

def report_activations(model, batch, save_dir):
    print("\n--- Forward Pass ---")
    probes = {}

    def hook_token_interleaver(module, args, output):
        interleaved, attn_mask, modality_map = output
        probes['interleaved_pre_scale'] = interleaved.detach().clone()
        probes['modality_map'] = modality_map
        return output

    def hook_recurrent_block(module, args, output):
        probes['recurrent_output'] = output[0].detach().clone()
        return output

    def hook_bridge(module, args, output):
        # output: (B, n_bridge_queries, d_model) — bridge K/V tokens for the DiT
        probes['bridge_output'] = output.detach().clone()
        return output

    handles = [
        model.token_interleaver.register_forward_hook(hook_token_interleaver),
        model.recurrent_block.register_forward_hook(hook_recurrent_block),
    ]
    if model.image_generator is not None and hasattr(model.image_generator, 'bridge'):
        handles.append(
            model.image_generator.bridge.register_forward_hook(hook_bridge)
        )

    model.eval()
    with torch.no_grad():
        outputs = model(
            text_input_ids=batch['text_token_ids'][:, :-1],
            image_inputs=batch.get('image_images', torch.zeros(1)).unsqueeze(1) if 'image_images' in batch else None,
            image_latent_labels=batch.get('image_images'),
            voice_inputs=batch.get('voice_features', torch.zeros(1)).unsqueeze(1) if 'voice_features' in batch else None,
            voice_lengths=batch.get('voice_feature_lengths', torch.zeros(1)).unsqueeze(1) if 'voice_feature_lengths' in batch else None,
            precomputed_latents=True,
            is_synthesis=batch.get('is_synthesis'),
        )

    for h in handles:
        h.remove()

    # Latent visualizations
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n  -- Saving Latent Visualizations to {save_dir} --")
    if 'image_images' in batch:
        save_latent_grid(batch['image_images'], os.path.join(save_dir, "input_image_latents.png"), "INPUT image latents")
        torch.save(batch['image_images'].detach().cpu(), os.path.join(save_dir, "input_image_latents.pt"))
    pred_image = outputs.get('image_latent_preds')
    if pred_image is not None:
        save_latent_grid(pred_image, os.path.join(save_dir, "pred_image_latents.png"), "PREDICTED image latents")
        torch.save(pred_image.detach().cpu(), os.path.join(save_dir, "pred_image_latents.pt"))

    if 'voice_features' in batch:
        torch.save(batch['voice_features'].detach().cpu(), os.path.join(save_dir, "input_voice_features.pt"))
    pred_voice = outputs.get('voice_latent_preds')
    if pred_voice is not None:
        torch.save(pred_voice.detach().cpu(), os.path.join(save_dir, "pred_voice_features.pt"))

    # Interleaved activations per modality
    print("\n  -- Intermediate Activations --")
    if 'interleaved_pre_scale' in probes:
        r("interleaved_pre_scale", probes['interleaved_pre_scale'])
        if 'modality_map' in probes:
            mmap = probes['modality_map']
            interleaved = probes['interleaved_pre_scale']
            mod_ids = {'text': 0, 'audio': 1, 'voice': 2, 'image': 3}
            for mod_name, mod_id in mod_ids.items():
                mask = (mmap == mod_id)
                if mask.any():
                    mod_tokens = interleaved[mask]
                    r(f"interleaved[{mod_name}]", mod_tokens)

    # Recurrent output
    if 'recurrent_output' in probes:
        r("recurrent_output_raw", probes['recurrent_output'])

    # Per-modality recurrent output stats
    print("\n  -- Per-Modality Recurrent Output Stats --")
    for mod in ['text', 'voice', 'audio', 'image']:
        for stat in ['token_var', 'seq_var', 'syn_seq_var', 'trans_seq_var', 'entropy']:
            key = f"recurrent_out/{mod}_{stat}"
            v = outputs.get(key)
            if v is not None:
                print(f"  {key}: {v:.6f}")

    # Per-modality cross-sample recurrent output diff. Answers "do the
    # recurrent outputs at position type X differ when the inputs to other
    # modalities differ?" In transcription direction this is how we tell
    # whether text positions are absorbing voice/image content via the
    # recurrent block's self-attention.
    if 'recurrent_output' in probes and 'modality_map' in probes:
        rec_out = probes['recurrent_output']
        mmap = probes['modality_map']
        mod_ids = {'text': 0, 'audio': 1, 'voice': 2, 'image': 3}
        if rec_out.shape[0] >= 2:
            print("\n  -- Per-Modality Cross-Sample Recurrent Output Diff --")
            for mod_name, mod_id in mod_ids.items():
                per_sample = []
                for b in range(rec_out.shape[0]):
                    mask = (mmap[b] == mod_id)
                    if mask.any():
                        per_sample.append(rec_out[b][mask])
                if len(per_sample) >= 2:
                    # Pairwise across (0,1); match on min length to handle
                    # variable-length modality runs (e.g. text prompts of
                    # different lengths).
                    n = min(s.shape[0] for s in per_sample)
                    a = per_sample[0][:n].float()
                    b2 = per_sample[1][:n].float()
                    diff = (a - b2).abs()
                    ratio = diff.mean() / a.std().clamp_min(1e-6)
                    print(f"  recurrent_out[{mod_name}] cross_sample: mean={diff.mean():.6e}, max={diff.max():.6e}, "
                          f"SNR={ratio.item():.2%} (over {n} matched positions)")

    # Coda predictions
    print("\n  -- Coda Predictions --")
    for key in ['image_latent_preds', 'voice_latent_preds', 'audio_latent_preds']:
        v = outputs.get(key)
        if v is not None:
            r(key, v)
            if v.shape[0] >= 2:
                diff = (v[0] - v[1]).abs()
                print(f"    |sample0 - sample1|: max={diff.max():.6e}, mean={diff.mean():.6e}")

    # Image recurrent tokens and normalized coda input
    rec = outputs.get('image_recurrent_tokens')
    if rec is not None:
        r("image_recurrent_tokens", rec)
        if rec.shape[0] >= 2:
            diff = (rec[0] - rec[1]).abs()
            print(f"    |sample0 - sample1|: max={diff.max():.6e}, mean={diff.mean():.6e}")
        if hasattr(model, 'image_coda_input_norm'):
            normed = model.image_coda_input_norm(rec)
            print(f"  image_coda_input_normed: std={normed.std():.4f}, seq_var={normed.var(dim=1).mean():.6f}")

    # Bridge output — is the bridge amplifying or preserving the prompt signal
    # in image_recurrent_tokens (its input)?
    bridge_out = probes.get('bridge_output')
    if bridge_out is not None:
        r("bridge_output (cond_kv)", bridge_out)
        if bridge_out.shape[0] >= 2:
            diff = (bridge_out[0] - bridge_out[1]).abs()
            print(f"    |sample0 - sample1|: max={diff.max():.6e}, mean={diff.mean():.6e}")
        # Cross-token variance and entropy — analogous to recurrent_out stats
        bo = bridge_out.float()
        print(f"    bridge seq_var: {bo.var(dim=1).mean():.6f}  (cross-token var per d_model dim)")
        print(f"    bridge token_var: {bo.var(dim=-1).mean():.6f}  (cross-d_model var per token)")
        # Mean-pooled global vector that feeds AdaLN modulation
        cond_global = bo.mean(dim=1)
        r("cond_global (AdaLN input)", cond_global)
        if cond_global.shape[0] >= 2:
            gdiff = (cond_global[0] - cond_global[1]).abs()
            print(f"    cond_global |sample0 - sample1|: max={gdiff.max():.6e}, mean={gdiff.mean():.6e}")

    return outputs


# ──────────────────────────────────────────────────────────────────────
# Gradient analysis
# ──────────────────────────────────────────────────────────────────────

def compute_loss(outputs, batch):
    """Replicate the trainer's loss computation so all modalities get gradients.

    This mirrors WorldModelTrainer.compute_loss() but without requiring a
    full trainer instance.
    """
    import torch.nn.functional as F

    device = next(v.device for v in outputs.values() if isinstance(v, torch.Tensor))
    total_loss = torch.tensor(0.0, device=device)
    components = {}
    eps = 1e-7

    # ── Text cross-entropy ──
    logits = outputs.get("logits")
    text_targets = batch.get("text_token_ids")
    if logits is not None and text_targets is not None:
        # Shifted targets: predict token t+1 from position t
        text_targets = text_targets[:, 1:]  # drop first token
        B, T, V = logits.size()
        T_min = min(T, text_targets.shape[1])
        logits_aligned = logits[:, :T_min, :].contiguous()
        targets_aligned = text_targets[:, :T_min].contiguous()
        text_loss_raw = F.cross_entropy(
            logits_aligned.reshape(-1, V),
            targets_aligned.reshape(-1),
            ignore_index=-100,
        )
        log_vocab = math.log(V)
        text_loss_norm = text_loss_raw / log_vocab
        total_loss = total_loss + text_loss_norm
        components["text_loss_raw"] = text_loss_raw.detach()
        components["text_loss_norm"] = text_loss_norm.detach()

    # ── Voice/Audio reconstruction (whitened L1+MSE + var loss) ──
    for mod, label_key, length_key in [
        ("voice", "voice_features", "voice_feature_lengths"),
        ("audio", "audio_features", "audio_feature_lengths"),
    ]:
        preds = outputs.get(f"{mod}_latent_preds")
        labels = batch.get(label_key)
        if preds is None or labels is None or preds.numel() == 0:
            continue
        labels = labels.to(preds.device)
        lengths = batch.get(length_key)

        # Masked stats
        if lengths is not None:
            lengths = lengths.to(preds.device).view(-1)
            T_feat = preds.shape[-1]
            mask = torch.arange(T_feat, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            while mask.dim() < preds.dim():
                mask = mask.unsqueeze(1)
            preds_m = preds * mask
            labels_m = labels * mask
            n_real = mask.expand_as(preds).sum().clamp_min(1).float()
            label_std = labels_m.detach().pow(2).sum().div(n_real).sqrt().clamp_min(eps)
            diff = preds_m - labels_m
            l1_raw = diff.abs().sum() / n_real
            mse_raw = diff.pow(2).sum() / n_real
        else:
            label_std = labels.detach().std().clamp_min(eps)
            l1_raw = F.l1_loss(preds, labels)
            mse_raw = F.mse_loss(preds, labels)

        label_var = label_std * label_std
        l1_norm = l1_raw / label_std
        mse_norm = mse_raw / label_var
        recon = l1_norm + mse_norm

        # Variance loss
        pred_std = preds.flatten(1).std(dim=1)
        label_std_per = labels.detach().flatten(1).std(dim=1).clamp_min(eps)
        var_loss = (pred_std / label_std_per - 1.0).abs().mean()

        mod_loss = recon + var_loss
        total_loss = total_loss + mod_loss
        components[f"{mod}_latent_l1_raw"] = l1_raw.detach()
        components[f"{mod}_latent_l1_norm"] = l1_norm.detach()
        components[f"{mod}_latent_mse_raw"] = mse_raw.detach()
        components[f"{mod}_latent_mse_norm"] = mse_norm.detach()
        components[f"{mod}_latent_var_loss"] = var_loss.detach()

        # Stop loss
        stop_logits = outputs.get(f"{mod}_stop_logits")
        if stop_logits is not None and lengths is not None:
            T_stop = stop_logits.shape[-1]
            stop_target = (torch.arange(T_stop, device=device).unsqueeze(0)
                           >= lengths.unsqueeze(1)).float()
            stop_loss_raw = F.binary_cross_entropy_with_logits(stop_logits, stop_target)
            stop_loss_norm = stop_loss_raw / 0.6931471805599453
            total_loss = total_loss + stop_loss_norm
            components[f"{mod}_stop_loss_norm"] = stop_loss_norm.detach()

    # ── Image diffusion loss (already computed inside the model) ──
    img_diff_loss = outputs.get("image_diffusion_loss")
    if img_diff_loss is not None and img_diff_loss.requires_grad:
        total_loss = total_loss + img_diff_loss
        components["image_diffusion_loss"] = img_diff_loss.detach()

    return total_loss, components


def report_gradients(model, batch):
    print("\n--- Gradients ---")
    model.train()
    model.zero_grad()
    outputs = model(
        text_input_ids=batch['text_token_ids'][:, :-1],
        image_inputs=batch.get('image_images', torch.zeros(1)).unsqueeze(1) if 'image_images' in batch else None,
        image_latent_labels=batch.get('image_images'),
        voice_inputs=batch.get('voice_features', torch.zeros(1)).unsqueeze(1) if 'voice_features' in batch else None,
        voice_lengths=batch.get('voice_feature_lengths', torch.zeros(1)).unsqueeze(1) if 'voice_feature_lengths' in batch else None,
        precomputed_latents=True,
        is_synthesis=batch.get('is_synthesis'),
    )

    # Compute full loss (text CE + voice/audio recon + image diffusion)
    loss, components = compute_loss(outputs, batch)

    print("\n  -- Loss Components --")
    for k, v in sorted(components.items()):
        print(f"  {k}: {v.item():.6e}")
    print(f"\n  total_loss: {loss.item():.6e}")

    if loss.grad_fn is None:
        print("  WARNING: no differentiable loss — skipping gradient analysis")
        return

    loss.backward()

    def grad_summary(name, module):
        """Print aggregate gradient stats for a module."""
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if not grads:
            print(f"  {name}: NO GRADS")
            return
        flat = torch.cat([g.flatten() for g in grads])
        print(f"  {name}: grad L2={flat.norm():.6f}, max={flat.abs().max():.6f}, mean_abs={flat.abs().mean():.6f}")

    def per_layer_grad(name, layers):
        """Print per-layer gradient norms for a ModuleList."""
        for i, layer in enumerate(layers):
            grads = [p.grad for p in layer.parameters() if p.grad is not None]
            if grads:
                l2 = torch.cat([g.flatten() for g in grads]).norm()
                print(f"  {name}/layer{i}: grad L2={l2:.6f}")
            else:
                print(f"  {name}/layer{i}: NO GRADS")

    # Recurrent block
    print("\n  -- Recurrent Block Gradients --")
    grad_summary("recurrent_block (all)", model.recurrent_block)
    if model.recurrent_block.projection is not None:
        grad_summary("recurrent/projection", model.recurrent_block.projection)
    per_layer_grad("recurrent/block", model.recurrent_block.recurrent_blocks)

    # Text prelude
    if model.text_feature_extractor is not None:
        print("\n  -- Text Prelude Gradients --")
        grad_summary("text_prelude (all)", model.text_feature_extractor)
        if hasattr(model.text_feature_extractor, 'prelude'):
            per_layer_grad("text_prelude", model.text_feature_extractor.prelude)

    # Text coda
    if model.text_generator is not None:
        print("\n  -- Text Coda Gradients --")
        grad_summary("text_coda (all)", model.text_generator)
        if hasattr(model.text_generator, 'coda'):
            per_layer_grad("text_coda", model.text_generator.coda)

    # Voice prelude
    if model.voice_feature_extractor is not None:
        print("\n  -- Voice Prelude Gradients --")
        grad_summary("voice_prelude (all)", model.voice_feature_extractor)
        if hasattr(model.voice_feature_extractor, 'prelude'):
            per_layer_grad("voice_prelude", model.voice_feature_extractor.prelude)

    # Voice coda
    if model.voice_generator is not None:
        print("\n  -- Voice Coda Gradients --")
        grad_summary("voice_coda (all)", model.voice_generator)
        if hasattr(model.voice_generator, 'coda'):
            per_layer_grad("voice_coda", model.voice_generator.coda)
        if hasattr(model.voice_generator, 'stop_head'):
            grad_summary("voice_coda/stop_head", model.voice_generator.stop_head)

    # Audio prelude/coda
    if model.audio_feature_extractor is not None:
        print("\n  -- Audio Prelude Gradients --")
        grad_summary("audio_prelude (all)", model.audio_feature_extractor)
    if model.audio_generator is not None:
        print("\n  -- Audio Coda Gradients --")
        grad_summary("audio_coda (all)", model.audio_generator)

    # Image prelude
    if model.image_feature_extractor is not None:
        print("\n  -- Image Prelude Gradients --")
        grad_summary("image_prelude (all)", model.image_feature_extractor)
        if hasattr(model.image_feature_extractor, 'prelude'):
            per_layer_grad("image_prelude", model.image_feature_extractor.prelude)

    # Image coda
    if model.image_generator is not None:
        print("\n  -- Image Coda Gradients --")
        gen = model.image_generator
        grad_summary("image_coda (all)", gen)
        if hasattr(gen, 'dit') and hasattr(gen.dit, 'blocks'):
            per_layer_grad("image_dit", gen.dit.blocks)
        if hasattr(gen, 'bridge') and hasattr(gen.bridge, 'layers'):
            per_layer_grad("image_bridge", gen.bridge.layers)
        if hasattr(gen, 'layers'):
            per_layer_grad("image_coda", gen.layers)
        if hasattr(gen, 'encoder_layers') and not hasattr(gen, 'dit'):
            per_layer_grad("image_coda_enc", gen.encoder_layers)

    # Image gen queries
    if hasattr(model, 'image_gen_queries') and model.image_gen_queries.grad is not None:
        r("image_gen_queries grad", model.image_gen_queries.grad)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def repeated_forward_stats(model, batch, n_repeats):
    """Run the forward pass N times and aggregate stochastic metrics.

    The DiT training_forward samples a random timestep and noise per call,
    and the recurrent block samples a variable iteration count, so single-run
    loss/gradient numbers are noisy. This collects N samples of each stochastic
    scalar and reports mean ± stddev so trajectory comparisons aren't fooled
    by per-batch random draws.
    """
    model.eval()
    collected: dict[str, list[float]] = {}

    bridge_probe: dict = {}
    rec_probe: dict = {}

    def _bridge_hook(module, args, output):
        bridge_probe['out'] = output.detach()
        return output

    def _rec_hook(module, args, output):
        rec_probe['out'] = output[0].detach()
        return output

    def _interleaver_hook(module, args, output):
        # output: (interleaved, attn_mask, modality_map)
        rec_probe['modality_map'] = output[2].detach() if hasattr(output[2], 'detach') else output[2]
        return output

    bridge_handle = None
    if model.image_generator is not None and hasattr(model.image_generator, 'bridge'):
        bridge_handle = model.image_generator.bridge.register_forward_hook(_bridge_hook)
    rec_handle = model.recurrent_block.register_forward_hook(_rec_hook)
    interleaver_handle = model.token_interleaver.register_forward_hook(_interleaver_hook)

    def _add(name: str, value):
        if not isinstance(value, (int, float)):
            try:
                value = float(value.item())
            except Exception:
                return
        if not math.isfinite(value):
            return
        collected.setdefault(name, []).append(value)

    for _ in range(n_repeats):
        with torch.no_grad():
            outputs = model(
                text_input_ids=batch['text_token_ids'][:, :-1],
                image_inputs=batch.get('image_images', torch.zeros(1)).unsqueeze(1) if 'image_images' in batch else None,
                image_latent_labels=batch.get('image_images'),
                voice_inputs=batch.get('voice_features', torch.zeros(1)).unsqueeze(1) if 'voice_features' in batch else None,
                voice_lengths=batch.get('voice_feature_lengths', torch.zeros(1)).unsqueeze(1) if 'voice_feature_lengths' in batch else None,
                precomputed_latents=True,
                is_synthesis=batch.get('is_synthesis'),
            )
        # Per-modality recurrent output stats (deterministic-ish; included for completeness)
        for mod in ('text', 'voice', 'audio', 'image'):
            for stat in ('token_var', 'seq_var', 'syn_seq_var', 'trans_seq_var', 'entropy'):
                key = f"recurrent_out/{mod}_{stat}"
                if outputs.get(key) is not None:
                    _add(key, outputs[key])
        # Cross-sample diffs (depend on stochastic forward)
        for key in ('image_latent_preds', 'voice_latent_preds', 'audio_latent_preds', 'image_recurrent_tokens'):
            v = outputs.get(key)
            if v is not None and v.shape[0] >= 2:
                diff = (v[0] - v[1]).abs()
                _add(f"{key}/cross_sample_mean", diff.mean())
                _add(f"{key}/cross_sample_max", diff.max())
        # Per-modality cross-sample diff on recurrent block output, using
        # modality_map to slice positions by type. Answers "does the recurrent
        # block mix signal from other-modality inputs into this-modality
        # positions?" Low diff = weak mixing.
        rec_out = rec_probe.get('out')
        mmap = rec_probe.get('modality_map')
        if rec_out is not None and mmap is not None and rec_out.shape[0] >= 2:
            mod_ids = {'text': 0, 'audio': 1, 'voice': 2, 'image': 3}
            for mod_name, mod_id in mod_ids.items():
                per_sample = []
                for b in range(rec_out.shape[0]):
                    mask = (mmap[b] == mod_id)
                    if mask.any():
                        per_sample.append(rec_out[b][mask])
                if len(per_sample) >= 2:
                    n = min(s.shape[0] for s in per_sample)
                    a = per_sample[0][:n].float()
                    b_t = per_sample[1][:n].float()
                    diff = (a - b_t).abs()
                    _add(f"rec_out[{mod_name}]/cross_sample_mean", diff.mean())
                    _add(f"rec_out[{mod_name}]/cross_sample_max", diff.max())
                    _add(f"rec_out[{mod_name}]/std", a.std())
        # Bridge output stats (cond_kv) and pooled cond_global. The bridge
        # itself is deterministic given recurrent output, but recurrent output
        # has stochastic iteration count, so re-aggregating per repeat catches
        # any iteration-count-driven variance.
        bo = bridge_probe.get('out')
        if bo is not None:
            bf = bo.float()
            _add("bridge_out/std", bf.std())
            _add("bridge_out/seq_var", bf.var(dim=1).mean())
            _add("bridge_out/token_var", bf.var(dim=-1).mean())
            if bf.shape[0] >= 2:
                diff = (bf[0] - bf[1]).abs()
                _add("bridge_out/cross_sample_mean", diff.mean())
                _add("bridge_out/cross_sample_max", diff.max())
                cond_global = bf.mean(dim=1)
                gdiff = (cond_global[0] - cond_global[1]).abs()
                _add("cond_global/cross_sample_mean", gdiff.mean())
                _add("cond_global/cross_sample_max", gdiff.max())
                _add("cond_global/std", cond_global.std())
        # Loss components (the noisy ones live here)
        _, components = compute_loss(outputs, batch)
        for k, v in components.items():
            _add(k, v)
        # compute_loss gates image_diffusion_loss on requires_grad, which is
        # False under no_grad — pull it directly from outputs so it's tracked.
        for k in ("image_diffusion_loss", "image_diffusion_loss_raw"):
            v = outputs.get(k)
            if v is not None:
                _add(k, v)

    if bridge_handle is not None:
        bridge_handle.remove()
    rec_handle.remove()
    interleaver_handle.remove()
    return collected


def print_repeated_summary(collected, n_repeats):
    print(f"\n--- Aggregated Stats over {n_repeats} Forwards ---")
    print(f"  {'metric':<50}  {'mean':>14}  {'std':>14}  {'rel_std':>8}")
    for name in sorted(collected.keys()):
        values = collected[name]
        if len(values) < 2:
            continue
        t = torch.tensor(values, dtype=torch.float64)
        mean = t.mean().item()
        std = t.std().item()
        rel = (std / abs(mean)) if abs(mean) > 1e-12 else float('nan')
        print(f"  {name:<50}  {mean:>14.6e}  {std:>14.6e}  {rel:>8.2%}")


def diagnose(model, batch, label="", n_repeats=1):
    print(f"\n{'=' * 70}")
    print(f"Checkpoint: {label}")
    print(f"{'=' * 70}")

    report_learned_params(model)

    save_dir = os.path.join(label, "diagnose")
    report_activations(model, batch, save_dir)
    report_gradients(model, batch)

    if n_repeats > 1:
        collected = repeated_forward_stats(model, batch, n_repeats)
        print_repeated_summary(collected, n_repeats)


def main():
    parser = argparse.ArgumentParser(description="Diagnose world model checkpoint across all modalities")
    parser.add_argument('--checkpoint', type=str, required=True, nargs='+')
    parser.add_argument('--config', type=str, default='small_sum_dit')
    parser.add_argument('--include_modes', type=str, default='text,voice,image')
    parser.add_argument('--max_samples', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=2, help='Samples per batch for diagnosis')
    parser.add_argument('--direction', type=str, default='synthesis', choices=['synthesis', 'transcription', 'mixed'],
                        help='Force collator direction for diagnosis')
    parser.add_argument('--tie_word_embeddings', action='store_true')
    parser.add_argument('--gen_query_mode', type=str, default=None, choices=['learned', 'positional_only'])
    parser.add_argument('--n_image_gen_positions', type=int, default=None)
    # Dataset paths
    parser.add_argument('--text_cache_dir', type=str, default=None)
    parser.add_argument('--voice_cache_dir', type=str, default=None)
    parser.add_argument('--audio_cache_dir', type=str, default=None)
    parser.add_argument('--image_cache_dir', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None, help='Base cache dir (uses <dir>_train)')
    parser.add_argument('--use_memorization_dataset', action='store_true')
    parser.add_argument('--n_repeats', type=int, default=1,
                        help='Run the forward pass N times and report mean ± stddev for stochastic metrics (loss, gradients from random timestep/noise draws). Use 5-10 for stable trajectory comparisons.')
    args = parser.parse_args()

    include_modes = [m.strip() for m in args.include_modes.split(',')]

    # Resolve shard dirs
    def resolve(specific, base):
        d = specific or base
        if d is None:
            return None
        candidate = d + "_train"
        if os.path.isdir(candidate):
            return candidate
        if os.path.isdir(d):
            return d
        print(f"Warning: {candidate} not found")
        return None

    text_dir = resolve(args.text_cache_dir, args.cache_dir) if 'text' in include_modes else None
    voice_dir = resolve(args.voice_cache_dir, args.cache_dir) if 'voice' in include_modes else None
    audio_dir = resolve(args.audio_cache_dir, args.cache_dir) if 'audio' in include_modes else None
    image_dir = resolve(args.image_cache_dir, args.cache_dir) if 'image' in include_modes else None

    if args.use_memorization_dataset:
        from megatransformer.scripts.data.world.memorization_dataset import MultimodalMemorizationDataset
        ds = MultimodalMemorizationDataset(
            text_shard_dir=text_dir,
            voice_shard_dir=voice_dir,
            audio_shard_dir=audio_dir,
            image_shard_dir=image_dir,
            max_samples=args.max_samples,
        )
    else:
        ds = MultimodalShardedDataset(
            text_shard_dir=text_dir,
            voice_shard_dir=voice_dir,
            audio_shard_dir=audio_dir,
            image_shard_dir=image_dir,
            cache_size=1,
            max_samples=args.max_samples,
        )

    print(f"Dataset: {len(ds)} samples")
    if hasattr(ds, 'task_types'):
        tasks = ds.task_types
        print(f"Task types ({len(tasks)}):")
        for i, t in enumerate(tasks):
            print(f"  {i}: {t}")

    # Build a batch that covers requested modalities
    collator = MultimodalDataCollator(max_seq_len=1024)
    if args.direction != 'mixed':
        collator.force_direction = args.direction

    # Try to pick samples that have all requested modalities
    tasks = ds.task_types if hasattr(ds, 'task_types') else []
    n_tasks = len(tasks) if tasks else 1
    n = args.batch_size

    # Find indices per modality
    modality_indices = {}
    for mod in include_modes:
        if mod == 'text':
            continue
        indices = []
        target_dir = args.direction if args.direction != 'mixed' else 'synthesis'
        for i in range(n_tasks):
            if tasks and tasks[i][1] == mod and tasks[i][2] == target_dir:
                for k in range(n):
                    idx = i + n_tasks * k
                    if idx < len(ds):
                        indices.append(idx)
                break
        if not indices:
            # Fallback: any task with this modality
            for i in range(n_tasks):
                if tasks and tasks[i][1] == mod:
                    for k in range(n):
                        idx = i + n_tasks * k
                        if idx < len(ds):
                            indices.append(idx)
                    break
        modality_indices[mod] = indices[:n]

    # Combine all indices, deduplicate
    all_indices = set()
    for indices in modality_indices.values():
        all_indices.update(indices)
    # Add text indices if needed
    if 'text' in include_modes and not all_indices:
        for i in range(n_tasks):
            if tasks and tasks[i][1] == 'text':
                for k in range(n):
                    idx = i + n_tasks * k
                    if idx < len(ds):
                        all_indices.add(idx)
                break
    all_indices = sorted(all_indices)[:max(n, len(all_indices))]

    if not all_indices:
        # Fallback: just use first n samples
        all_indices = list(range(min(n, len(ds))))

    print(f"\nUsing sample indices: {all_indices}")
    samples = [ds[i] for i in all_indices]
    batch = collator(samples)

    print("\n--- Batch Contents ---")
    for k, v in sorted(batch.items()):
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")
        elif isinstance(v, list):
            print(f"  {k}: list len={len(v)}")
        else:
            print(f"  {k}: {v}")
    if 'text_token_ids' in batch:
        ids = batch['text_token_ids']
        print(f"  text_token_ids unique: {sorted(ids.unique().tolist())}")
    if 'is_synthesis' in batch:
        print(f"  is_synthesis: {batch['is_synthesis']}")

    for ckpt in args.checkpoint:
        model = load_model(
            args.config, ckpt,
            tie_word_embeddings=args.tie_word_embeddings,
            gen_query_mode=args.gen_query_mode,
            n_image_gen_positions=args.n_image_gen_positions,
            include_modes=include_modes,
        )
        diagnose(model, batch, label=ckpt, n_repeats=args.n_repeats)


if __name__ == '__main__':
    main()
