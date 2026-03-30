"""
Diagnose a world model checkpoint by tracing activations and gradients.

Usage:
    PYTHONPATH=src python3 -m scripts.debug.diagnose_checkpoint \
        --checkpoint ./runs/world/<run>/checkpoint-<step> \
        --config huginn_small_causal_cross_attn
"""

import argparse
import copy
import torch
import torch.nn.functional as F

from config.world.world_model import WORLD_MODEL_CONFIGS
from model.world.world_model import MegaTransformerWorldModel
from scripts.data.world.dataset import MultimodalShardedDataset
from scripts.data.world.data_collator import MultimodalDataCollator


def load_model(config_name, checkpoint_path):
    config = copy.deepcopy(WORLD_MODEL_CONFIGS[config_name])
    config.include_modes = ['text', 'voice', 'image']
    model = MegaTransformerWorldModel(config)
    sd = torch.load(f'{checkpoint_path}/pytorch_model.bin', map_location='cpu', weights_only=True)
    model.load_state_dict(sd, strict=False)
    return model


def r(name, tensor, indent=2):
    if tensor is None:
        print(f"{' ' * indent}{name}: None")
        return
    t = tensor.float()
    print(f"{' ' * indent}{name}: std={t.std():.6f}, mean={t.mean():.6f}, range=[{t.min():.4f}, {t.max():.4f}]")


def diagnose(model, batch, label=""):
    print(f"\n{'=' * 70}")
    print(f"Checkpoint: {label}")
    print(f"{'=' * 70}")

    # Learned parameters
    print("\n--- Learned Parameters ---")
    print(f"  image_gen_queries: std={model.image_gen_queries.std():.4f}")
    if hasattr(model, 'image_gen_pos_embedding'):
        print(f"  image_gen_pos_embedding: std={model.image_gen_pos_embedding.std():.4f}")
    if hasattr(model, 'voice_gen_queries'):
        print(f"  voice_gen_queries: std={model.voice_gen_queries.std():.4f}")
    print(f"  projection weight: std={model.recurrent_block.projection.weight.std():.6f}")
    if model.image_cross_decoder is not None:
        cd = model.image_cross_decoder
        print(f"  cross_decoder spatial_queries: std={cd.spatial_queries.std():.4f}")
        if hasattr(cd, 'output_scale'):
            print(f"  cross_decoder output_scale: mean={cd.output_scale.data.mean():.4f}, std={cd.output_scale.data.std():.4f}")
            print(f"  cross_decoder output_bias: mean={cd.output_bias.data.mean():.4f}, std={cd.output_bias.data.std():.4f}")
    if hasattr(model, 'image_cross_input_norm'):
        print(f"  image_cross_input_norm weight: mean={model.image_cross_input_norm.weight.mean():.4f}")
        print(f"  image_cross_input_norm bias: mean={model.image_cross_input_norm.bias.mean():.4f}")

    # Forward pass
    print("\n--- Forward Pass ---")
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

        for key in ['image_recurrent_tokens', 'image_prelude_tokens',
                     'image_cross_latent_preds', 'image_latent_preds',
                     'voice_latent_preds']:
            v = outputs.get(key)
            if v is not None:
                r(key, v)

        for key in ['recurrent_out/text_token_var', 'recurrent_out/text_seq_var',
                     'recurrent_out/voice_token_var', 'recurrent_out/voice_seq_var',
                     'recurrent_out/image_token_var', 'recurrent_out/image_seq_var']:
            v = outputs.get(key)
            if v is not None:
                print(f"  {key}: {v:.6f}")

        # Check normalized cross-decoder input
        rec = outputs.get('image_recurrent_tokens')
        if rec is not None and hasattr(model, 'image_cross_input_norm'):
            normed = model.image_cross_input_norm(rec)
            print(f"  cross_input_normed: std={normed.std():.4f}, seq_var={normed.var(dim=1).mean():.6f}")

        # Per-sample cross output
        cross = outputs.get('image_cross_latent_preds')
        if cross is not None:
            for b in range(cross.shape[0]):
                print(f"  cross_preds sample {b}: std={cross[b].std():.4f}, mean={cross[b].mean():.4f}")

    # Gradient analysis
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

    loss = torch.tensor(0.0)
    for k, v in outputs.items():
        if 'loss' in k and isinstance(v, torch.Tensor) and v.requires_grad:
            loss = loss + v
    if loss.item() == 0:
        for k in ['image_cross_latent_preds', 'image_latent_preds', 'voice_latent_preds']:
            v = outputs.get(k)
            if v is not None and v.requires_grad:
                loss = loss + v.sum() * 1e-6
    loss.backward()

    groups = {
        'image_gen_queries': model.image_gen_queries,
        'recurrent/proj': model.recurrent_block.projection.weight,
    }
    if hasattr(model, 'image_gen_pos_embedding'):
        groups['image_gen_pos_embedding'] = model.image_gen_pos_embedding
    if model.image_cross_decoder is not None:
        for li, layer in enumerate(model.image_cross_decoder.layers):
            sq = sum(p.grad.norm().item() ** 2 for p in layer.parameters() if p.grad is not None)
            groups[f'cross_decoder/layer{li}'] = None
            print(f"  cross_decoder/layer{li}: grad L2={sq ** 0.5:.6f}")

    for name, param in groups.items():
        if param is None:
            continue
        if param.grad is not None:
            print(f"  {name}: grad L2={param.grad.norm():.6f}, max={param.grad.abs().max():.6f}")
        else:
            print(f"  {name}: NO GRAD")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, nargs='+')
    parser.add_argument('--config', type=str, default='huginn_small_causal_cross_attn')
    parser.add_argument('--max_samples', type=int, default=32)
    args = parser.parse_args()

    ds = MultimodalShardedDataset(
        text_shard_dir='./cached_datasets/text_pile_train',
        voice_shard_dir='./cached_datasets/audio_sive_tiny_deep_3xdownsample_conv2d_batchnorm_0_0_layer10_train',
        image_shard_dir='./cached_datasets/image_vae_train',
        cache_size=1, max_samples=args.max_samples,
    )
    modality_names = list(ds.modalities.keys())
    image_indices = [i for i in range(len(ds)) if modality_names[i % len(modality_names)] == 'image'][:2]
    collator = MultimodalDataCollator(max_seq_len=1024)
    collator.force_direction = 'synthesis'
    samples = [ds[i] for i in image_indices]
    batch = collator(samples)

    for ckpt in args.checkpoint:
        model = load_model(args.config, ckpt)
        diagnose(model, batch, label=ckpt)


if __name__ == '__main__':
    main()
