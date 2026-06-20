"""
Plot per-iteration recurrent block activation stats for a world model checkpoint.

Loads a checkpoint, runs a forward pass on a small batch, and saves the
iteration stats plot as an image. Supports image, voice, and text modalities.

Usage:
    # Image synthesis (default):
    python -m megatransformer.scripts.debug.plot_iteration_stats --checkpoint ./runs/world/<run>/checkpoint-<step> --config huginn_small_causal_cross_attn --output ./stats.png

    # Voice synthesis:
    python -m megatransformer.scripts.debug.plot_iteration_stats --checkpoint ./runs/world/<run>/checkpoint-<step> --modality voice --output ./stats_voice.png

    # Text continuation:
    python -m megatransformer.scripts.debug.plot_iteration_stats --checkpoint ./runs/world/<run>/checkpoint-<step> --modality text --output ./stats_text.png

    # Voice transcription:
    python -m megatransformer.scripts.debug.plot_iteration_stats --checkpoint ./runs/world/<run>/checkpoint-<step> --modality voice --direction transcription --output ./stats_voice_transcription.png

    # Multiple checkpoints (overlaid comparison):
    python -m megatransformer.scripts.debug.plot_iteration_stats --checkpoint ./runs/world/<run>/checkpoint-1000 ./runs/world/<run>/checkpoint-5000 --output ./comparison.png

    # Per-token individual plots:
    python -m megatransformer.scripts.debug.plot_iteration_stats --checkpoint ./runs/world/<run>/checkpoint-<step> --output ./stats.png --per_token
"""

import argparse
import copy
import os

import matplotlib.pyplot as plt
import torch

from megatransformer.config.world.world_model import WORLD_MODEL_CONFIGS
from megatransformer.model.world.world_model import MegaTransformerWorldModel
from megatransformer.scripts.data.world.dataset import MultimodalShardedDataset
from megatransformer.scripts.data.world.data_collator import MultimodalDataCollator


def load_model(config_name, checkpoint_path, tie_word_embeddings=False):
    config = copy.deepcopy(WORLD_MODEL_CONFIGS[config_name])
    config.include_modes = ['text', 'voice', 'image']
    if tie_word_embeddings:
        config.tie_word_embeddings = True
    model = MegaTransformerWorldModel(config)
    sd = torch.load(f'{checkpoint_path}/pytorch_model.bin', map_location='cpu', weights_only=True)
    model.load_state_dict(sd, strict=False)
    return model


def collect_iteration_stats(model, batch):
    """Run a forward pass with iteration stat tracking and return stats + modality map."""
    model.eval()
    model.recurrent_block.track_iteration_stats = True

    # Hook the token interleaver to capture the modality map
    modality_map = [None]

    def hook_interleaver(module, args, output):
        _, _, mmap = output
        modality_map[0] = mmap.cpu()
        return output

    handle = model.token_interleaver.register_forward_hook(hook_interleaver)

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

    handle.remove()
    model.recurrent_block.track_iteration_stats = False
    return outputs.get("iteration_stats", []), modality_map[0]


def plot_single(iteration_stats, title, output_path):
    """Plot averaged stats from a single checkpoint."""
    from megatransformer.utils.visualization import render_iteration_stats
    fig = render_iteration_stats(iteration_stats)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved averaged plot to {output_path}")


def plot_per_token(iteration_stats, output_dir, batch_idx=0, modality_map=None):
    """Save one plot per token position."""
    from megatransformer.utils.visualization import render_token_iteration_stats

    os.makedirs(output_dir, exist_ok=True)

    # Determine seq_len from the first stat tensor
    first_stat = iteration_stats[0]
    sample_tensor = first_stat["std"]
    seq_len = sample_tensor.shape[1]

    for tok in range(seq_len):
        modality_label = ""
        if modality_map is not None:
            mod_id = modality_map[batch_idx, tok].item()
            mod_names = {0: "text", 1: "audio", 2: "voice", 3: "image"}
            modality_label = f" ({mod_names.get(mod_id, f'mod{mod_id}')})"

        fig = render_token_iteration_stats(
            iteration_stats, tok, batch_idx=batch_idx,
            title=f"Token {tok}{modality_label}",
        )
        fig.savefig(os.path.join(output_dir, f"token_{tok:04d}.png"), dpi=100, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved {seq_len} per-token plots to {output_dir}/")


def plot_comparison(all_stats, labels, output_path):
    """Overlay stats from multiple checkpoints on the same plot."""
    stat_names = ["std", "mean", "max", "min", "norm"]
    has_kl = any("kl" in s for stats in all_stats for s in stats)
    if has_kl:
        stat_names.append("kl")

    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(len(stat_names), 1, figsize=(10, 2.5 * len(stat_names)), sharex=True)
    if len(stat_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, stat_names):
        for i, (stats, label) in enumerate(zip(all_stats, labels)):
            iterations = list(range(len(stats)))
            values = [s.get(name, 0.0) for s in stats]
            ax.plot(iterations, values, marker='.', markersize=3, linewidth=1, color=colors[i % len(colors)], label=label)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        if name == "kl":
            ax.set_yscale('symlog', linthresh=1e-6)
            ax.set_ylabel('KL divergence')

    axes[-1].set_xlabel('Iteration')
    fig.suptitle('Thought state activation stats per recurrent iteration', fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved to {output_path}")


def build_batch(ds, tasks, modality, direction):
    """Build a batch for a given modality and direction."""
    if modality == 'text':
        task_indices = [i for i in range(len(ds)) if tasks[i % len(tasks)][1] == 'text'][:2]
    else:
        task_indices = [i for i in range(len(ds)) if tasks[i % len(tasks)][1] == modality and tasks[i % len(tasks)][2] == direction][:2]

    if not task_indices:
        return None

    collator = MultimodalDataCollator(max_seq_len=1024)
    if modality != 'text':
        collator.force_direction = direction
    samples = [ds[i] for i in task_indices]
    return collator(samples)


def run_single_task(model, ds, tasks, modality, direction, output_base, per_token=False):
    """Run stats collection and plotting for one modality/direction combo."""
    from megatransformer.utils.visualization import render_all_tokens_iteration_stats

    label = f"{modality}_{direction}" if modality != 'text' else "text"
    batch = build_batch(ds, tasks, modality, direction)
    if batch is None:
        print(f"  Skipping {label}: no samples found")
        return

    print(f"  Running {label}...")
    stats, modality_map = collect_iteration_stats(model, batch)
    if not stats:
        print(f"  Skipping {label}: no iteration stats collected")
        return

    avg_path = f"{output_base}_{label}.png"
    plot_single(stats, f"{label}", avg_path)

    overlay_path = f"{output_base}_{label}_all_tokens.png"
    fig = render_all_tokens_iteration_stats(stats, modality_map=modality_map)
    fig.suptitle(label, fontsize=12)
    fig.tight_layout()
    fig.savefig(overlay_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved all-tokens overlay to {overlay_path}")

    if per_token:
        output_dir = f"{output_base}_{label}_per_token"
        plot_per_token(stats, output_dir, modality_map=modality_map)


ALL_TASKS = [
    ("text", "continuation"),
    ("voice", "synthesis"),
    ("voice", "transcription"),
    ("image", "synthesis"),
    ("image", "transcription"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, nargs='+')
    parser.add_argument('--config', type=str, default='huginn_small_causal_cross_attn')
    parser.add_argument('--output', type=str, default='./iteration_stats.png')
    parser.add_argument('--max_samples', type=int, default=32)
    parser.add_argument('--per_token', action='store_true', help='Save individual plots per token position')
    parser.add_argument('--modality', type=str, default='image', choices=['text', 'voice', 'image'], help='Which modality task to generate samples for')
    parser.add_argument('--direction', type=str, default='synthesis', choices=['synthesis', 'transcription'], help='Task direction')
    parser.add_argument('--all', action='store_true', dest='all_tasks', help='Run all modalities and directions, saving separate images')
    parser.add_argument('--tie_word_embeddings', action='store_true', help='Tie text embedding and LM head weights')
    args = parser.parse_args()

    ds = MultimodalShardedDataset(
        text_shard_dir='./cached_datasets/text_pile_train',
        voice_shard_dir='./cached_datasets/audio_sive_tiny_deep_3xdownsample_conv2d_batchnorm_0_0_layer10_train',
        image_shard_dir='./cached_datasets/image_vae_train',
        cache_size=1, max_samples=args.max_samples,
    )
    tasks = ds.task_types

    if len(args.checkpoint) > 1 and not args.all_tasks:
        # Multi-checkpoint comparison (uses single modality/direction)
        batch = build_batch(ds, tasks, args.modality, args.direction)
        if batch is None:
            print(f"No {args.modality} {args.direction} samples found in dataset")
            return

        all_stats = []
        labels = []
        for ckpt in args.checkpoint:
            model = load_model(args.config, ckpt, tie_word_embeddings=args.tie_word_embeddings)
            stats, _ = collect_iteration_stats(model, batch)
            if stats:
                all_stats.append(stats)
                labels.append(os.path.basename(os.path.dirname(ckpt)) + "/" + os.path.basename(ckpt))
            else:
                print(f"Warning: no stats from {ckpt}")
            del model

        if not all_stats:
            print("No iteration stats collected from any checkpoint")
            return
        plot_comparison(all_stats, labels, args.output)
        return

    # Single or multi checkpoint with --all
    output_base = os.path.splitext(args.output)[0]

    for ckpt in args.checkpoint:
        model = load_model(args.config, ckpt, tie_word_embeddings=args.tie_word_embeddings)
        ckpt_name = os.path.basename(ckpt)

        if len(args.checkpoint) > 1:
            ckpt_output_base = f"{output_base}_{ckpt_name}"
            os.makedirs(os.path.dirname(ckpt_output_base) or '.', exist_ok=True)
        else:
            ckpt_output_base = output_base
            os.makedirs(os.path.dirname(ckpt_output_base) or '.', exist_ok=True)

        print(f"Checkpoint: {ckpt}")

        if args.all_tasks:
            for modality, direction in ALL_TASKS:
                run_single_task(model, ds, tasks, modality, direction, ckpt_output_base, args.per_token)
        else:
            run_single_task(model, ds, tasks, args.modality, args.direction, ckpt_output_base, args.per_token)

        del model


if __name__ == '__main__':
    main()
