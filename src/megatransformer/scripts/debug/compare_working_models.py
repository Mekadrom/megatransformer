"""
Compare two working models to understand what makes them succeed.
- researcher (d=512, 1 block, layernorm on images)
- 200m_safe_1blk (d=640, 1 block, instancenorm on images)

Usage:
    python -m megatransformer.scripts.debug.compare_working_models
"""

import torch
import torch.nn.functional as F
import copy

from megatransformer.config.world.world_model import WORLD_MODEL_CONFIGS
from megatransformer.model.world.world_model import MegaTransformerWorldModel
from megatransformer.scripts.data.world.dataset import MultimodalShardedDataset
from megatransformer.scripts.data.world.data_collator import MultimodalDataCollator


def load_model(config_name, checkpoint_path, device="cpu"):
    config = copy.deepcopy(WORLD_MODEL_CONFIGS[config_name])
    config.include_modes = ["text", "voice", "image"]
    model = MegaTransformerWorldModel(config)
    state_dict = torch.load(
        f"{checkpoint_path}/pytorch_model.bin",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=False)
    return model


def r(name, tensor, indent=2):
    if tensor is None:
        print(f"{' '*indent}{name}: None")
        return
    t = tensor.float()
    print(f"{' '*indent}{name}: std={t.std():.4f}, mean={t.mean():.4f}, range=[{t.min():.4f}, {t.max():.4f}]")


def analyze(model, batch, label):
    print(f"\n{'='*80}")
    print(f"MODEL: {label}")
    print(f"{'='*80}")

    text_ids = batch["text_token_ids"]
    voice_feats = batch.get("voice_features")
    voice_lengths = batch.get("voice_feature_lengths")
    image_data = batch.get("image_images")

    model.eval()
    with torch.no_grad():
        # Feature extractors
        print("\n--- Preludes ---")
        text_hidden = model.text_feature_extractor(text_ids[:, :-1])
        r("text_hidden", text_hidden)

        voice_in = voice_feats.unsqueeze(1)
        voice_lens = voice_lengths.unsqueeze(1)
        image_in = image_data.unsqueeze(1)

        # Voice prelude internals
        vfe = model.voice_feature_extractor
        v = voice_feats.permute(0, 2, 1).contiguous()
        if vfe.input_norm is not None:
            v = vfe.input_norm(v)
        v_proj = vfe.projection(v)
        v_proj = vfe.pos_encoding(v_proj)
        r("voice_after_proj", v_proj)
        x = v_proj
        for i, block in enumerate(vfe.prelude):
            h, _ = block(x)
            x = x + h
        r("voice_after_blocks", x)
        if hasattr(vfe, 'output_norm'):
            x = vfe.output_norm(x)
            r("voice_after_output_norm", x)
        voice_hidden = x.unsqueeze(1)

        # Image prelude internals
        ife = model.image_feature_extractor
        lat = image_data
        if ife.input_norm is not None:
            if isinstance(ife.input_norm, torch.nn.InstanceNorm2d):
                lat = ife.input_norm(lat)
            else:
                lat = ife.input_norm(lat.permute(0,2,3,1)).permute(0,3,1,2)
        r("image_after_input_norm", lat)
        patch_emb = ife.patch_embedding(lat) + ife.pos_embedding
        r("image_after_patch_embed", patch_emb)
        x = patch_emb
        for i, block in enumerate(ife.prelude):
            h, _ = block(x)
            x = x + h
        r("image_after_blocks", x)
        if hasattr(ife, 'output_norm'):
            x = ife.output_norm(x)
            r("image_after_output_norm", x)
        image_hidden = x.unsqueeze(1)

        # Interleave
        print("\n--- Interleaving ---")
        interleaved, attn_mask, modality_map = model.token_interleaver(
            text_hidden_states=text_hidden, text_token_ids=text_ids[:, :-1],
            voice_hidden_states=voice_hidden, voice_lengths=voice_lens,
            image_hidden_states=image_hidden,
        )
        scaled = interleaved * model.embed_scale
        r("interleaved", scaled)
        for mid, mn in [(0,"text"), (2,"voice"), (3,"image")]:
            mask = modality_map[0] == mid
            if mask.any():
                r(f"  {mn}_tokens", scaled[0][mask])

        # Recurrent
        print("\n--- Recurrent ---")
        rb = model.recurrent_block
        thought = rb.initialize_thinking_state(scaled)
        r("thought_init", thought)

        for it in range(min(5, rb.mean_thinking_steps)):
            h = rb._combine(scaled, thought)
            if it == 0:
                r(f"iter0_combined", h)
                for bi, block in enumerate(rb.recurrent_blocks):
                    h_in = h.clone()
                    h, _ = block(h)
                    res = (h - h_in).std().item()
                    r(f"iter0_block{bi}", h)
                    print(f"      residual={res:.6f}")
                thought = rb._extract_thought(h)
            else:
                thought = rb._run_iteration(scaled, thought, it, attn_mask, None, 0, False, False, None)
            r(f"iter{it}_thought", thought)
            # Per-modality
            for mid, mn in [(0,"text"), (2,"voice"), (3,"image")]:
                mask = modality_map[0] == mid
                if mask.any():
                    r(f"  {mn}", thought[0][mask])

        # Full forward
        rec_out, _, iters, _ = rb(scaled, attention_mask=attn_mask)
        r("recurrent_final", rec_out)
        print(f"  iterations: {iters}")

        uninterleaved = model.token_uninterleaver(rec_out, modality_map)
        for k in ["text", "voice", "image"]:
            r(f"uninterleaved_{k}", uninterleaved[k])

        # Codas
        print("\n--- Codas ---")

        # Image cross-decoder detail
        if uninterleaved["image"] is not None:
            ic = model.image_generator
            h = uninterleaved["image"]
            r("image_generator_input", h)
            # Encoder: self-attention over content tokens
            for i, block in enumerate(ic.encoder_layers):
                hidden, _ = block(h)
                h = h + hidden
                r(f"image_encoder_layer{i}", h)
            h = ic.encoder_output_norm(h)
            r("image_encoder_normed", h)
            # Decoder: spatial queries cross-attend to encoded content
            B = h.shape[0]
            queries = ic.spatial_queries.expand(B, -1, -1) + ic.pos_embedding
            r("image_spatial_queries", queries)
            for i, block in enumerate(ic.layers):
                queries, _ = block(queries, encoder_hidden_states=h)
                r(f"image_decoder_layer{i}", queries)
            preds = ic.unpatchify(queries)
            r("image_before_denorm", preds)
            if ic.use_output_denorm:
                preds = preds * ic.output_scale[None,:,None,None] + ic.output_bias[None,:,None,None]
            r("image_preds", preds)
            r("image_target", image_data)
            print(f"    L1={F.l1_loss(preds, image_data).item():.4f}, MSE={F.mse_loss(preds, image_data).item():.4f}")

            # Check output_scale/bias
            if hasattr(ic, 'output_scale'):
                print(f"    output_scale: mean={ic.output_scale.data.mean():.4f}, std={ic.output_scale.data.std():.4f}")
                print(f"    output_bias: mean={ic.output_bias.data.mean():.4f}, std={ic.output_bias.data.std():.4f}")

        # Voice coda
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
            r("voice_preds", feat)

    # Gradient analysis
    print("\n--- Gradients ---")
    model.train()
    model.zero_grad()
    outputs = model(
        text_input_ids=text_ids[:, :-1],
        voice_inputs=voice_feats.unsqueeze(1),
        voice_lengths=voice_lengths.unsqueeze(1),
        image_inputs=image_data.unsqueeze(1),
        precomputed_latents=True,
    )
    loss = torch.tensor(0.0)
    if "logits" in outputs:
        loss = loss + outputs["logits"].sum() * 1e-6
    if "voice_latent_preds" in outputs:
        loss = loss + outputs["voice_latent_preds"].sum() * 1e-6
    if "image_latent_preds" in outputs:
        loss = loss + outputs["image_latent_preds"].sum() * 1e-6
    loss.backward()

    modules = {
        "text_embed": model.text_feature_extractor,
        "voice_prelude": model.voice_feature_extractor,
        "image_prelude": model.image_feature_extractor,
        "recurrent": model.recurrent_block,
        "recurrent/proj": model.recurrent_block.projection,
        "text_coda": model.text_generator,
        "voice_coda": model.voice_generator,
        "image_generator": model.image_generator,
    }
    for i, block in enumerate(model.recurrent_block.recurrent_blocks):
        modules[f"rec/block{i}"] = block

    print(f"  {'Module':<25} {'L2':>10} {'RMS':>12}")
    for name, mod in modules.items():
        if mod is None:
            continue
        sq = sum(p.grad.float().norm(2).item()**2 for p in mod.parameters() if p.grad is not None)
        n = sum(p.numel() for p in mod.parameters() if p.grad is not None)
        l2 = sq**0.5
        rms = (sq/n)**0.5 if n > 0 else 0
        print(f"  {name:<25} {l2:>10.4f} {rms:>12.8f}")

    # Weight magnitudes
    print("\n--- Weight Magnitudes ---")
    for name, mod in modules.items():
        if mod is None:
            continue
        stds = [p.std().item() for p in mod.parameters() if p.dim() >= 2]
        if stds:
            print(f"  {name:<25} avg_std={sum(stds)/len(stds):.6f}")


def main():
    print("Loading data...")
    dataset = MultimodalShardedDataset(
        text_shard_dir="./cached_datasets/text_pile_train",
        voice_shard_dir="./cached_datasets/audio_sive_tiny_deep_3xdownsample_conv2d_batchnorm_0_0_layer10_train",
        image_shard_dir="./cached_datasets/image_vae_train",
        cache_size=1, max_samples=32,
    )
    batch = MultimodalDataCollator(max_seq_len=1024)([dataset[0], dataset[1]])

    models = [
        ("researcher", "./runs/world/memorize32_test_researcher_0_0/checkpoint-1000", "RESEARCHER (d=512, 1blk, works)"),
        ("200m_safe_1blk", "./runs/world/memorize32_test_200m_safe_1blk_instancenorm_1_0/checkpoint-2000", "200m_safe_1blk (d=640, 1blk, works)"),
    ]

    for config_name, ckpt, label in models:
        model = load_model(config_name, ckpt)
        analyze(model, batch, label)


if __name__ == "__main__":
    main()
