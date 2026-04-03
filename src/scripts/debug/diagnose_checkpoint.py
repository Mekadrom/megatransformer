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
    if model.image_generator is not None:
        cd = model.image_generator
        print(f"  image_coda spatial_queries: std={cd.spatial_queries.std():.4f}")
        if hasattr(cd, 'output_scale'):
            print(f"  image_coda output_scale: mean={cd.output_scale.data.mean():.4f}, std={cd.output_scale.data.std():.4f}")
            print(f"  image_coda output_bias: mean={cd.output_bias.data.mean():.4f}, std={cd.output_bias.data.std():.4f}")
    if hasattr(model, 'image_coda_input_norm'):
        print(f"  image_coda_input_norm weight: mean={model.image_coda_input_norm.weight.mean():.4f}")
        print(f"  image_coda_input_norm bias: mean={model.image_coda_input_norm.bias.mean():.4f}")

    # Forward pass with intermediate probes
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

    def hook_image_text_conditioning(module, args, output):
        conditioned, _ = output
        probes['conditioned_queries'] = conditioned.detach().clone()
        return output

    handles = []
    handles.append(model.token_interleaver.register_forward_hook(hook_token_interleaver))
    handles.append(model.recurrent_block.register_forward_hook(hook_recurrent_block))
    if hasattr(model, 'image_text_conditioning'):
        handles.append(model.image_text_conditioning.register_forward_hook(hook_image_text_conditioning))

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

    # Intermediate probes
    print("\n  -- Intermediate Activations --")
    if 'conditioned_queries' in probes:
        r("after_text_conditioning", probes['conditioned_queries'])
    if 'interleaved_pre_scale' in probes:
        r("interleaved_pre_scale", probes['interleaved_pre_scale'])
        if 'modality_map' in probes:
            mmap = probes['modality_map']
            interleaved = probes['interleaved_pre_scale']
            for mod_name in ['text', 'image', 'voice']:
                mask = (mmap == {'text': 0, 'image': 3, 'voice': 2, 'audio': 1}.get(mod_name, -1))
                if mask.any():
                    mod_tokens = interleaved[mask]
                    r(f"interleaved[{mod_name}]", mod_tokens)
    if 'recurrent_output' in probes:
        r("recurrent_output_raw", probes['recurrent_output'])

        for key in ['image_recurrent_tokens', 'image_prelude_tokens',
                     'image_latent_preds',
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
        if rec is not None and hasattr(model, 'image_coda_input_norm'):
            normed = model.image_coda_input_norm(rec)
            print(f"  coda_input_normed: std={normed.std():.4f}, seq_var={normed.var(dim=1).mean():.6f}")

        # Per-sample cross output
        cross = outputs.get('image_latent_preds')
        if cross is not None:
            for b in range(cross.shape[0]):
                print(f"  coda_preds sample {b}: std={cross[b].std():.4f}, mean={cross[b].mean():.4f}")

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

    loss = torch.tensor(0.0, requires_grad=True)
    for k, v in outputs.items():
        if 'loss' in k and isinstance(v, torch.Tensor) and v.requires_grad:
            loss = loss + v
    if loss.grad_fn is None:
        for k in ['image_latent_preds', 'voice_latent_preds']:
            v = outputs.get(k)
            if v is not None and v.requires_grad:
                loss = loss + v.sum() * 1e-6
    if loss.grad_fn is not None:
        loss.backward()
    else:
        print("  WARNING: no differentiable outputs — skipping gradient analysis")

    groups = {
        'image_gen_queries': model.image_gen_queries,
        'recurrent/proj': model.recurrent_block.projection.weight,
    }
    if hasattr(model, 'image_gen_pos_embedding'):
        groups['image_gen_pos_embedding'] = model.image_gen_pos_embedding
    if model.image_generator is not None:
        for li, layer in enumerate(model.image_generator.layers):
            sq = sum(p.grad.norm().item() ** 2 for p in layer.parameters() if p.grad is not None)
            groups[f'image_coda/layer{li}'] = None
            print(f"  image_coda/layer{li}: grad L2={sq ** 0.5:.6f}")

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
    tasks = ds.task_types
    image_indices = [i for i in range(len(ds)) if tasks[i % len(tasks)][1] == 'image'][:2]
    collator = MultimodalDataCollator(max_seq_len=1024)
    collator.force_direction = 'synthesis'
    samples = [ds[i] for i in image_indices]
    batch = collator(samples)

    print("\n--- Batch Contents ---")
    for k, v in batch.items():
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
        model = load_model(args.config, ckpt)
        diagnose(model, batch, label=ckpt)


if __name__ == '__main__':
    main()
