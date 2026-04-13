"""
Diagnose a world model checkpoint by tracing activations and gradients.

Usage:
    PYTHONPATH=src python3 -m scripts.debug.diagnose_checkpoint \
        --checkpoint ./runs/world/<run>/checkpoint-<step> \
        --config huginn_small_causal_cross_attn
"""

import argparse
import copy
import os
import torch

from config.world.world_model import WORLD_MODEL_CONFIGS
from model.world.world_model import MegaTransformerWorldModel
from scripts.data.world.dataset import MultimodalShardedDataset
from scripts.data.world.data_collator import MultimodalDataCollator


def load_model(config_name, checkpoint_path, tie_word_embeddings=False, gen_query_mode=None, n_image_gen_positions=None):
    config = copy.deepcopy(WORLD_MODEL_CONFIGS[config_name])
    config.include_modes = ['text', 'voice', 'image']
    if tie_word_embeddings:
        config.tie_word_embeddings = True
    if gen_query_mode is not None:
        config.gen_query_mode = gen_query_mode
    if n_image_gen_positions is not None:
        config.n_image_gen_positions = n_image_gen_positions
    model = MegaTransformerWorldModel(config)
    sd = torch.load(f'{checkpoint_path}/pytorch_model.bin', map_location='cpu', weights_only=True)
    # Filter out parameters whose shapes don't match the current model (e.g.
    # checkpoints trained before a config change like patch_size or max_audio_duration).
    # This lets us load what we can and skip the rest with a warning.
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
    print(f"{' ' * indent}{name}: std={t.std():.6f}, mean={t.mean():.6f}, range=[{t.min():.4f}, {t.max():.4f}]")


def save_latent_grid(latents: torch.Tensor, save_path: str, title: str):
    """Save a (B, C, H, W) latent tensor as a grid of per-sample, per-channel images.

    Each sample becomes a row, each channel a column. Useful for visually
    verifying that latents are not all zeros / not collapsed.
    """
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


def diagnose(model, batch, label=""):
    print(f"\n{'=' * 70}")
    print(f"Checkpoint: {label}")
    print(f"{'=' * 70}")

    # Learned parameters
    print("\n--- Learned Parameters ---")
    if hasattr(model, 'image_gen_queries'):
        print(f"  image_gen_queries: std={model.image_gen_queries.std():.4f}")
    else:
        print(f"  image_gen_queries: not present (positional_only mode)")
    if hasattr(model, 'image_gen_pos_embedding'):
        print(f"  image_gen_pos_embedding (frozen): std={model.image_gen_pos_embedding.pe.std():.4f}")
    # Voice/audio gen queries removed — voice uses autoregressive generation
    if hasattr(model.recurrent_block, 'projection') and model.recurrent_block.projection is not None:
        print(f"  projection weight: std={model.recurrent_block.projection.weight.std():.6f}")
    if model.image_generator is not None:
        cd = model.image_generator
        print(f"  image_coda class: {type(cd).__name__}")
        # ImageDecoder in cross_attention mode has spatial_queries; in direct
        # mode it does not. DiffusionBridgeImageDecoder has bridge.queries.
        if hasattr(cd, 'spatial_queries'):
            print(f"  image_coda spatial_queries: std={cd.spatial_queries.std():.4f}")
        elif hasattr(cd, 'bridge') and hasattr(cd.bridge, 'queries'):
            print(f"  image_coda bridge.queries: std={cd.bridge.queries.std():.4f}")
        else:
            print(f"  image_coda has no learned spatial/bridge queries (direct mode)")
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

    # Save input + predicted latents as visualization grids so the user can
    # eyeball that the input isn't all zeros and the predictions aren't collapsed.
    save_dir = os.path.join(label, "diagnose")
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n  -- Saving Latent Visualizations to {save_dir} --")
    if 'image_images' in batch:
        save_latent_grid(
            batch['image_images'],
            os.path.join(save_dir, "input_latents.png"),
            "INPUT image_latent_labels",
        )
        # Also save raw tensor for re-loading later
        torch.save(batch['image_images'].detach().cpu(), os.path.join(save_dir, "input_latents.pt"))
    pred_latents = outputs.get('image_latent_preds')
    if pred_latents is not None:
        save_latent_grid(
            pred_latents,
            os.path.join(save_dir, "pred_latents.png"),
            "PREDICTED image_latent_preds",
        )
        torch.save(pred_latents.detach().cpu(), os.path.join(save_dir, "pred_latents.pt"))

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

        for key in ['image_recurrent_tokens', 'image_latent_preds', 'voice_latent_preds']:
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
            if cross.shape[0] >= 2:
                diff = (cross[0] - cross[1]).abs()
                print(f"  coda_preds |sample0 - sample1|: max={diff.max():.6e}, mean={diff.mean():.6e}")

        # Per-sample recurrent output at image positions: confirms whether the
        # recurrent block routes any sample-specific info to image positions.
        rec = outputs.get('image_recurrent_tokens')
        if rec is not None and rec.shape[0] >= 2:
            for b in range(rec.shape[0]):
                print(f"  image_recurrent_tokens sample {b}: std={rec[b].std():.6f}, mean={rec[b].mean():.6f}")
            diff = (rec[0] - rec[1]).abs()
            print(f"  image_recurrent_tokens |sample0 - sample1|: max={diff.max():.6e}, mean={diff.mean():.6e}")

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

    # Print loss components so we can tell whether the loss is actually small
    # or whether the gradient chain is broken.
    print("\n  -- Loss Components --")
    for k, v in outputs.items():
        if 'loss' in k and isinstance(v, torch.Tensor):
            print(f"  {k}: value={v.item():.6e}, requires_grad={v.requires_grad}, grad_fn={v.grad_fn is not None}")

    loss = torch.tensor(0.0, requires_grad=True)
    used_fallback = False
    for k, v in outputs.items():
        if 'loss' in k and isinstance(v, torch.Tensor) and v.requires_grad:
            loss = loss + v
    if loss.grad_fn is None:
        used_fallback = True
        for k in ['image_latent_preds', 'voice_latent_preds']:
            v = outputs.get(k)
            if v is not None and v.requires_grad:
                loss = loss + v.sum() * 1e-6
    print(f"  total_loss: value={loss.item():.6e}, used_fallback={used_fallback}")
    if loss.grad_fn is not None:
        loss.backward()
    else:
        print("  WARNING: no differentiable outputs — skipping gradient analysis")

    groups = {
        'recurrent/proj': model.recurrent_block.projection.weight if model.recurrent_block.projection is not None else None,
    }
    if hasattr(model, 'image_gen_queries'):
        groups['image_gen_queries'] = model.image_gen_queries
    # image_gen_pos_embedding is a frozen buffer, no gradients to track

    if model.image_generator is not None:
        gen = model.image_generator
        # Pick the layer stack to inspect based on which decoder is in use:
        # - ImageDecoder direct mode → encoder_layers (the only stack)
        # - ImageDecoder cross_attention mode → decoder_layers ("layers")
        # - DiffusionBridgeImageDecoder → dit.blocks (and bridge.layers)
        if hasattr(gen, 'layers'):
            stack_label = "image_coda"
            stack = gen.layers
        elif hasattr(gen, 'encoder_layers') and not hasattr(gen, 'dit'):
            stack_label = "image_coda_enc"
            stack = gen.encoder_layers
        elif hasattr(gen, 'dit') and hasattr(gen.dit, 'blocks'):
            stack_label = "image_dit"
            stack = gen.dit.blocks
        else:
            stack_label = None
            stack = []
        if stack is not None:
            for li, layer in enumerate(stack):
                sq = sum(p.grad.norm().item() ** 2 for p in layer.parameters() if p.grad is not None)
                print(f"  {stack_label}/layer{li}: grad L2={sq ** 0.5:.6f}")
        # For the diffusion bridge also report the bridge stack separately.
        if hasattr(gen, 'bridge') and hasattr(gen.bridge, 'layers'):
            for li, layer in enumerate(gen.bridge.layers):
                sq = sum(p.grad.norm().item() ** 2 for p in layer.parameters() if p.grad is not None)
                print(f"  image_bridge/layer{li}: grad L2={sq ** 0.5:.6f}")

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
    parser.add_argument('--tie_word_embeddings', action='store_true', help='Tie text embedding and LM head weights')
    parser.add_argument('--gen_query_mode', type=str, default=None, choices=['learned', 'positional_only'],
                        help='Generation query mode (default: use config)')
    parser.add_argument('--n_image_gen_positions', type=int, default=None,
                        help='Number of image gen query positions (default: use config)')
    args = parser.parse_args()

    ds = MultimodalShardedDataset(
        text_shard_dir='./cached_datasets/text_pile_train',
        voice_shard_dir='./cached_datasets/audio_sive_tiny_deep_3xdownsample_conv2d_batchnorm_0_0_layer10_train',
        image_shard_dir='./cached_datasets/image_vae_train',
        cache_size=1, max_samples=args.max_samples,
    )
    tasks = ds.task_types
    n_tasks = len(tasks)
    # Find the first index whose task is image_synthesis. Striding by n_tasks
    # then walks through distinct within_task_idx values for the SAME task,
    # so we get genuinely different image samples (not the same image picked
    # via image_synthesis vs image_transcription, which both map to within=0).
    image_synth_base = next(
        (i for i in range(n_tasks) if tasks[i][1] == 'image' and tasks[i][2] == 'synthesis'),
        None,
    )
    if image_synth_base is None:
        raise RuntimeError("No image_synthesis task found in dataset task_types")
    image_indices = [image_synth_base + n_tasks * k for k in range(2)]
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
        model = load_model(args.config, ckpt, tie_word_embeddings=args.tie_word_embeddings, gen_query_mode=args.gen_query_mode, n_image_gen_positions=args.n_image_gen_positions)
        diagnose(model, batch, label=ckpt)


if __name__ == '__main__':
    main()
