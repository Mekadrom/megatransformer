"""
Compare hidden states between a working model (200m_safe_1blk) and a failing model (200m_safe)
at every point in the forward pass, using real training data.

Usage:
    python -m scripts.debug.compare_hidden_states
"""

import sys
import torch
import torch.nn.functional as F

from config.world.world_model import WORLD_MODEL_CONFIGS
from model.world.world_model import MegaTransformerWorldModel
from scripts.data.world.dataset import MultimodalShardedDataset
from scripts.data.world.data_collator import MultimodalDataCollator
from utils.constants import (
    VOICE_PLACEHOLDER_TOKEN_ID,
    IMAGE_PLACEHOLDER_TOKEN_ID,
)


def load_model(config_name, checkpoint_path, device="cpu"):
    config = WORLD_MODEL_CONFIGS[config_name]
    config.include_modes = ["text", "voice", "image"]
    model = MegaTransformerWorldModel(config)
    state_dict = torch.load(
        f"{checkpoint_path}/pytorch_model.bin",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
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
    print(f"{prefix}{name}: shape={list(tensor.shape)}, std={std:.6f}, mean={mean:.6f}, range=[{mn:.4f}, {mx:.4f}]{flag}")


def trace_forward(model, batch, label):
    """Run forward pass with hooks to capture every intermediate state."""
    print(f"\n{'=' * 70}")
    print(f"MODEL: {label}")
    print(f"{'=' * 70}")

    device = next(model.parameters()).device

    # Move batch to device
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

    with torch.no_grad():
        # ── 1. Feature Extractors ──
        print("\n--- Feature Extractors ---")
        text_hidden = model.text_feature_extractor(text_ids[:, :-1])
        report("text_hidden", text_hidden)

        voice_inputs = None
        voice_lens = None
        if voice_feats is not None:
            voice_inputs = voice_feats.unsqueeze(1)
            if voice_lengths is not None:
                voice_lens = voice_lengths.unsqueeze(1)

        image_inputs = None
        if image_data is not None:
            image_inputs = image_data.unsqueeze(1)

        # Voice prelude
        if voice_inputs is not None and model.voice_feature_extractor is not None:
            B, n_voice = voice_inputs.shape[:2]
            voice_flat = voice_inputs.view(B * n_voice, *voice_inputs.shape[2:])

            # Trace through voice prelude internals
            vfe = model.voice_feature_extractor
            v = voice_flat.permute(0, 2, 1).contiguous()
            if vfe.input_norm is not None:
                v = vfe.input_norm(v)
            v_proj = vfe.projection(v)
            v_proj = vfe.pos_encoding(v_proj)
            report("voice_after_projection", v_proj)

            x = v_proj
            for i, block in enumerate(vfe.prelude):
                hidden, _ = block(x)
                x = x + hidden
                report(f"voice_prelude_layer{i}", x)

            voice_hidden_flat = x
            voice_hidden = voice_hidden_flat.view(B, n_voice, *voice_hidden_flat.shape[1:])
            report("voice_hidden_final", voice_hidden)

        # Image prelude
        if image_inputs is not None and model.image_feature_extractor is not None:
            B, n_img = image_inputs.shape[:2]
            img_flat = image_inputs.view(B * n_img, *image_inputs.shape[2:])

            ife = model.image_feature_extractor
            latent_images = img_flat
            if ife.input_norm is not None:
                if isinstance(ife.input_norm, torch.nn.InstanceNorm2d):
                    latent_images = ife.input_norm(latent_images)
                else:
                    latent_images = ife.input_norm(latent_images.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            report("image_after_norm", latent_images)

            patch_emb = ife.patch_embedding(latent_images)
            patch_emb = patch_emb + ife.pos_embedding
            report("image_after_patch_embed", patch_emb)

            x = patch_emb
            for i, block in enumerate(ife.prelude):
                hidden, _ = block(x)
                x = x + hidden
                report(f"image_prelude_layer{i}", x)

            image_hidden_flat = x
            image_hidden = image_hidden_flat.view(B, n_img, *image_hidden_flat.shape[1:])
            report("image_hidden_final", image_hidden)

        # ── 2. Interleaving ──
        print("\n--- Interleaving ---")
        interleaved, attn_mask, modality_map = model.token_interleaver(
            text_hidden_states=text_hidden,
            text_token_ids=text_ids[:, :-1],
            voice_hidden_states=voice_hidden if voice_inputs is not None else None,
            voice_lengths=voice_lens,
            image_hidden_states=image_hidden if image_inputs is not None else None,
        )
        report("interleaved_tokens", interleaved)

        # Embed scaling
        scaled = interleaved * model.embed_scale
        report("after_embed_scale", scaled)

        # Modality breakdown
        for mod_id, mod_name in [(0, "text"), (2, "voice"), (3, "image")]:
            mask = modality_map[0] == mod_id
            if mask.any():
                mod_tokens = scaled[0][mask]
                report(f"  scaled_{mod_name}_tokens", mod_tokens)

        # ── 3. Recurrent Block ──
        print("\n--- Recurrent Block ---")
        rb = model.recurrent_block
        x_0 = scaled
        thought = rb.initialize_thinking_state(x_0)
        report("thought_init", thought)

        combined = rb._combine(x_0, thought)
        report("combined_input", combined)

        # Trace first 3 iterations
        last_thought = thought
        for it in range(min(3, rb.mean_thinking_steps)):
            h = rb._combine(x_0, last_thought)
            for bi, block in enumerate(rb.recurrent_blocks):
                h_before = h.clone()
                h, _ = block(h)
                residual = (h - h_before).std().item()
                if it == 0:
                    report(f"iter{it}_block{bi}_output", h)
                    print(f"    residual contribution: {residual:.6f}")

            new_thought = rb._extract_thought(h)
            if it < 3:
                report(f"iter{it}_thought", new_thought)
            last_thought = new_thought

        # Full forward through recurrent
        rec_out, _, iters, _ = rb(scaled, attention_mask=attn_mask)
        report("recurrent_output", rec_out)
        print(f"  iterations: {iters}")

        # ── 4. Uninterleaving ──
        print("\n--- Uninterleaving ---")
        uninterleaved = model.token_uninterleaver(rec_out, modality_map)
        for key in ["text", "voice", "image"]:
            report(f"uninterleaved_{key}", uninterleaved[key])

        # ── 5. Codas ──
        print("\n--- Text Coda ---")
        if uninterleaved["text"] is not None:
            tc = model.text_generator
            h = uninterleaved["text"]
            for i, block in enumerate(tc.coda):
                hidden, _ = block(h)
                h = h + hidden
                report(f"text_coda_layer{i}", h)
            logits = tc.lm_head(h)
            report("text_logits", logits)
            # Top predictions
            top_ids = logits[0, :5].argmax(dim=-1)
            print(f"  top-1 predictions (first 5 positions): {top_ids.tolist()}")

        print("\n--- Voice Coda ---")
        if uninterleaved["voice"] is not None:
            vc = model.voice_generator
            h = uninterleaved["voice"]
            for i, block in enumerate(vc.coda):
                hidden, _ = block(h)
                h = h + hidden
                report(f"voice_coda_layer{i}", h)
            feat = vc.feature_projection(h)
            feat = feat * vc.output_scale + vc.output_bias
            feat = feat.permute(0, 2, 1)
            if vc.temporal_refine is not None:
                feat = vc.temporal_refine(feat)
            report("voice_latent_preds", feat)

        print("\n--- Image Cross-Decoder ---")
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
            report("image_spatial_queries", queries)
            for i, block in enumerate(ic.layers):
                queries, _ = block(queries, encoder_hidden_states=h)
                report(f"image_decoder_layer{i}", queries)
            latent_preds = ic.unpatchify(queries)
            report("image_before_denorm", latent_preds)
            if ic.use_output_denorm:
                latent_preds = latent_preds * ic.output_scale[None, :, None, None] + ic.output_bias[None, :, None, None]
            report("image_latent_preds", latent_preds)

            # Compare to target
            if image_data is not None:
                report("image_target", image_data)
                l1 = F.l1_loss(latent_preds, image_data[:latent_preds.shape[0]]).item()
                mse = F.mse_loss(latent_preds, image_data[:latent_preds.shape[0]]).item()
                print(f"  vs target: L1={l1:.4f}, MSE={mse:.4f}")


def main():
    # Load one batch of training data
    print("Loading training data...")
    dataset = MultimodalShardedDataset(
        text_shard_dir="./cached_datasets/text_pile_train",
        voice_shard_dir="./cached_datasets/audio_sive_tiny_deep_3xdownsample_conv2d_batchnorm_0_0_layer10_train",
        image_shard_dir="./cached_datasets/image_vae_train",
        cache_size=1,
        max_samples=32,
    )

    collator = MultimodalDataCollator(max_seq_len=1024)
    # Get first 2 samples
    samples = [dataset[0], dataset[1]]
    batch = collator(samples)

    print(f"Batch keys: {list(batch.keys())}")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")

    # Load both models
    print("\nLoading models...")
    model_good = load_model(
        "200m_safe_1blk",
        "./runs/world/memorize32_test_200m_safe_1blk_instancenorm_1_0/checkpoint-1000",
    )
    model_bad = load_model(
        "200m_safe",
        "./runs/world/memorize32_test_200m_safe_1_0/checkpoint-1000",
    )

    # Trace both
    trace_forward(model_good, batch, "200m_safe_1blk (WORKS)")
    trace_forward(model_bad, batch, "200m_safe (FAILS - black images)")


if __name__ == "__main__":
    main()
