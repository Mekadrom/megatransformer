"""
Deep activation and gradient analysis of 200m_safe models with pre/post projection iteration norm.
Loads trained checkpoints and traces forward+backward to find where image signal dies.

Usage:
    python -m megatransformer.scripts.debug.analyze_iteration_norm
"""

import sys
import torch
import torch.nn.functional as F

from megatransformer.config.world.world_model import WORLD_MODEL_CONFIGS
from megatransformer.model.world.world_model import MegaTransformerWorldModel
from megatransformer.scripts.data.world.dataset import MultimodalShardedDataset
from megatransformer.scripts.data.world.data_collator import MultimodalDataCollator


def load_model(config_name, checkpoint_path, iteration_norm, device="cpu"):
    import copy
    config = copy.deepcopy(WORLD_MODEL_CONFIGS[config_name])
    config.include_modes = ["text", "voice", "image"]
    config.recurrent_block_config.iteration_norm = iteration_norm
    model = MegaTransformerWorldModel(config)
    state_dict = torch.load(
        f"{checkpoint_path}/pytorch_model.bin",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=False)
    return model


def report(name, tensor, indent=2):
    prefix = " " * indent
    if tensor is None:
        print(f"{prefix}{name}: None")
        return
    t = tensor.float()
    std = t.std().item()
    mean = t.mean().item()
    mn = t.min().item()
    mx = t.max().item()
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    flag = ""
    if has_nan:
        flag = " *** NaN ***"
    elif has_inf:
        flag = " *** Inf ***"
    elif std < 1e-6:
        flag = " *** NEAR ZERO ***"
    elif std > 50:
        flag = " *** VERY LARGE ***"
    print(f"{prefix}{name}: std={std:.4f}, mean={mean:.4f}, range=[{mn:.4f}, {mx:.4f}]{flag}")


def analyze_model(model, batch, label):
    print(f"\n{'=' * 80}")
    print(f"MODEL: {label}")
    print(f"{'=' * 80}")

    device = next(model.parameters()).device
    text_ids = batch["text_token_ids"].to(device)
    voice_feats = batch.get("voice_features")
    voice_lengths = batch.get("voice_feature_lengths")
    image_data = batch.get("image_images")

    if voice_feats is not None:
        voice_feats = voice_feats.to(device)
    if voice_lengths is not None:
        voice_lengths = voice_lengths.to(device)
    if image_data is not None:
        image_data = image_data.to(device)

    # ══════════════════════════════════════════════════════════════════════
    # FORWARD ANALYSIS (no grad)
    # ══════════════════════════════════════════════════════════════════════
    model.eval()
    with torch.no_grad():
        # Feature extractors
        print("\n--- Feature Extractors ---")
        text_hidden = model.text_feature_extractor(text_ids[:, :-1])
        report("text_hidden", text_hidden)

        voice_inputs = voice_feats.unsqueeze(1) if voice_feats is not None else None
        voice_lens = voice_lengths.unsqueeze(1) if voice_lengths is not None else None
        image_inputs = image_data.unsqueeze(1) if image_data is not None else None

        # Voice prelude
        if voice_inputs is not None and model.voice_feature_extractor is not None:
            B = voice_inputs.shape[0]
            v_flat = voice_inputs.view(B, *voice_inputs.shape[2:])
            voice_hidden = model.voice_feature_extractor(v_flat).unsqueeze(1)
            report("voice_hidden", voice_hidden)

        # Image prelude
        if image_inputs is not None and model.image_feature_extractor is not None:
            B = image_inputs.shape[0]
            i_flat = image_inputs.view(B, *image_inputs.shape[2:])
            image_hidden = model.image_feature_extractor(i_flat, precomputed_latents=True).unsqueeze(1)
            report("image_hidden", image_hidden)

        # Interleave
        interleaved, attn_mask, modality_map = model.token_interleaver(
            text_hidden_states=text_hidden,
            text_token_ids=text_ids[:, :-1],
            voice_hidden_states=voice_hidden if voice_inputs is not None else None,
            voice_lengths=voice_lens,
            image_hidden_states=image_hidden if image_inputs is not None else None,
        )
        scaled = interleaved * model.embed_scale
        report("interleaved_scaled", scaled)

        # Per-modality interleaved stats
        for mod_id, mod_name in [(0, "text"), (2, "voice"), (3, "image")]:
            mask = modality_map[0] == mod_id
            if mask.any():
                report(f"  {mod_name}_tokens", scaled[0][mask])

        # ── Recurrent block detailed trace ──
        print("\n--- Recurrent Block (iteration-by-iteration) ---")
        rb = model.recurrent_block
        x_0 = scaled
        thought = rb.initialize_thinking_state(x_0)

        n_iters = min(8, rb.mean_thinking_steps)
        for it in range(n_iters):
            h = rb._combine(x_0, thought)
            if it == 0:
                report(f"iter{it}_combined", h)

            for bi, block in enumerate(rb.recurrent_blocks):
                h_in = h.clone()
                h, _ = block(h)
                residual = (h - h_in).std().item()
                if it < 3:
                    report(f"iter{it}_block{bi}", h)
                    print(f"      residual_contribution={residual:.6f}")

            new_thought = rb._extract_thought(h)
            report(f"iter{it}_thought", new_thought)

            # Per-modality thought state
            for mod_id, mod_name in [(0, "text"), (2, "voice"), (3, "image")]:
                mask = modality_map[0] == mod_id
                if mask.any():
                    report(f"  iter{it}_{mod_name}_thought", new_thought[0][mask])

            thought = new_thought

        # Full forward
        rec_out, _, iters, _ = rb(scaled, attention_mask=attn_mask)
        report("recurrent_output_final", rec_out)
        print(f"  total_iterations: {iters}")

        # Uninterleave
        uninterleaved = model.token_uninterleaver(rec_out, modality_map)
        for key in ["text", "voice", "image"]:
            report(f"uninterleaved_{key}", uninterleaved[key])

        # Codas
        print("\n--- Codas ---")
        if uninterleaved["text"] is not None:
            tc = model.text_generator
            h = uninterleaved["text"]
            for i, block in enumerate(tc.coda):
                hidden, _ = block(h)
                h = h + hidden
            logits = tc.lm_head(h)
            report("text_logits", logits)

        if uninterleaved["voice"] is not None:
            vc = model.voice_generator
            h = uninterleaved["voice"]
            for i, block in enumerate(vc.coda):
                hidden, _ = block(h)
                h = h + hidden
            feat = vc.feature_projection(h)
            feat = feat * vc.output_scale + vc.output_bias
            feat = feat.permute(0, 2, 1)
            if vc.temporal_refine is not None:
                feat = vc.temporal_refine(feat)
            report("voice_latent_preds", feat)

        if uninterleaved["image"] is not None:
            ic = model.image_generator
            h = uninterleaved["image"]
            # Encoder: self-attention over content tokens
            for i, block in enumerate(ic.encoder_layers):
                hidden, _ = block(h)
                h = h + hidden
                report(f"image_encoder_layer{i}", h)
            h = ic.encoder_output_norm(h)
            report("image_encoder_normed", h)
            # Decoder: spatial queries cross-attend to encoded content
            B = h.shape[0]
            queries = ic.spatial_queries.expand(B, -1, -1) + ic.pos_embedding
            for i, block in enumerate(ic.layers):
                queries, _ = block(queries, encoder_hidden_states=h)
                report(f"image_decoder_layer{i}", queries)
            latent_preds = ic.unpatchify(queries)
            report("image_before_denorm", latent_preds)
            if ic.use_output_denorm:
                latent_preds = latent_preds * ic.output_scale[None, :, None, None] + ic.output_bias[None, :, None, None]
            report("image_latent_preds", latent_preds)
            if image_data is not None:
                report("image_target", image_data)
                print(f"    L1={F.l1_loss(latent_preds, image_data).item():.4f}, MSE={F.mse_loss(latent_preds, image_data).item():.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # GRADIENT ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    print("\n--- Gradient Analysis ---")
    model.train()
    model.zero_grad()

    outputs = model(
        text_input_ids=text_ids[:, :-1],
        voice_inputs=voice_feats.unsqueeze(1) if voice_feats is not None else None,
        voice_lengths=voice_lengths.unsqueeze(1) if voice_lengths is not None else None,
        image_inputs=image_data.unsqueeze(1) if image_data is not None else None,
        precomputed_latents=True,
    )

    # Build loss from all modalities
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    if "logits" in outputs:
        loss = loss + outputs["logits"].sum() * 1e-6
    if "voice_latent_preds" in outputs:
        loss = loss + outputs["voice_latent_preds"].sum() * 1e-6
    if "image_latent_preds" in outputs:
        loss = loss + outputs["image_latent_preds"].sum() * 1e-6
    loss.backward()

    # Per-module gradient norms
    groups = {
        "text_embedding": model.text_feature_extractor,
        "voice_prelude": model.voice_feature_extractor,
        "image_prelude": model.image_feature_extractor,
        "text_coda": model.text_generator,
        "voice_coda": model.voice_generator,
        "image_generator": model.image_generator,
    }

    # Per recurrent block
    for i, block in enumerate(model.recurrent_block.recurrent_blocks):
        groups[f"recurrent/block{i}"] = block
        groups[f"recurrent/block{i}/attn"] = block.self_attn
        groups[f"recurrent/block{i}/ffn"] = block.ffn

    if model.recurrent_block.projection is not None:
        groups["recurrent/projection"] = model.recurrent_block.projection
    if model.recurrent_block.pre_projection_norm is not None:
        groups["recurrent/pre_proj_norm"] = model.recurrent_block.pre_projection_norm
    if model.recurrent_block.post_projection_norm is not None:
        groups["recurrent/post_proj_norm"] = model.recurrent_block.post_projection_norm

    print(f"  {'Module':<35} {'L2':>10} {'RMS':>12} {'Max':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*12} {'-'*10}")
    for name, module in groups.items():
        total_sq = 0.0
        n_params = 0
        max_grad = 0.0
        for p in module.parameters():
            if p.grad is not None:
                g = p.grad.float()
                total_sq += g.norm(2).item() ** 2
                n_params += p.numel()
                max_grad = max(max_grad, g.abs().max().item())
        if n_params > 0:
            l2 = total_sq ** 0.5
            rms = (total_sq / n_params) ** 0.5
            print(f"  {name:<35} {l2:>10.6f} {rms:>12.8f} {max_grad:>10.6f}")
        else:
            print(f"  {name:<35} {'NO GRAD':>10}")

    # Weight magnitude analysis
    print("\n--- Learned Weight Magnitudes ---")
    print(f"  {'Module':<35} {'Weight std':>12} {'Bias std':>12}")
    print(f"  {'-'*35} {'-'*12} {'-'*12}")
    for name, module in groups.items():
        w_stds = []
        b_stds = []
        for pname, p in module.named_parameters():
            if 'weight' in pname and p.dim() >= 2:
                w_stds.append(p.std().item())
            elif 'bias' in pname:
                b_stds.append(p.std().item())
        if w_stds:
            w_avg = sum(w_stds) / len(w_stds)
            b_avg = sum(b_stds) / len(b_stds) if b_stds else 0.0
            print(f"  {name:<35} {w_avg:>12.6f} {b_avg:>12.6f}")

    # Output scale/bias learned values
    if model.image_generator is not None and hasattr(model.image_generator, 'output_scale'):
        print(f"\n  image output_scale: mean={model.image_generator.output_scale.data.mean().item():.4f}, std={model.image_generator.output_scale.data.std().item():.4f}")
        print(f"  image output_bias: mean={model.image_generator.output_bias.data.mean().item():.4f}, std={model.image_generator.output_bias.data.std().item():.4f}")
    if model.voice_generator is not None and hasattr(model.voice_generator, 'output_scale'):
        print(f"  voice output_scale: mean={model.voice_generator.output_scale.data.mean().item():.4f}, std={model.voice_generator.output_scale.data.std().item():.4f}")
        print(f"  voice output_bias: mean={model.voice_generator.output_bias.data.mean().item():.4f}, std={model.voice_generator.output_bias.data.std().item():.4f}")


def main():
    print("Loading training data...")
    dataset = MultimodalShardedDataset(
        text_shard_dir="./cached_datasets/text_pile_train",
        voice_shard_dir="./cached_datasets/audio_sive_tiny_deep_3xdownsample_conv2d_batchnorm_0_0_layer10_train",
        image_shard_dir="./cached_datasets/image_vae_train",
        cache_size=1,
        max_samples=32,
    )
    collator = MultimodalDataCollator(max_seq_len=1024)
    batch = collator([dataset[0], dataset[1]])

    checkpoints = [
        ("200m_safe", "./runs/world/memorize32_test_200m_safe_pre_projection_iteration_norm_0_0/checkpoint-2000", "pre_projection", "PRE-PROJECTION NORM"),
        ("200m_safe", "./runs/world/memorize32_test_200m_safe_post_projection_iteration_norm_0_0/checkpoint-2000", "post_projection", "POST-PROJECTION NORM"),
    ]

    for config_name, ckpt_path, iter_norm, label in checkpoints:
        model = load_model(config_name, ckpt_path, iter_norm)
        analyze_model(model, batch, label)


if __name__ == "__main__":
    main()
