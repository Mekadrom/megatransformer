"""
Timelapse visualization of SIVE encoder features across training checkpoints.

Loads each checkpoint sequentially, runs inference on a single audio sample,
and compiles per-layer hidden features into MP4 animations showing how
feature extraction evolves during training.

Usage:
    python -m src.scripts.eval.audio.sive.timelapse_sive_encoder_features \
        --run_dir runs/sive/tiny_deep_0_0 \
        --config tiny_deep \
        --cache_dir ../cached_datasets/audio_sive_val \
        --sample_index 42 \
        --output_dir ./timelapse_output \
        --fps 10 \
        --device cuda \
        --num_speakers 921 \
        --checkpoint_step_filter 5000
"""

import argparse
import glob
import os
import re

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm

from model.audio.sive.sive import SpeakerInvariantVoiceEncoder
from scripts.data.audio.dataset import AudioShardedDataset
from utils.model_loading_utils import load_model


def discover_checkpoints(run_dir: str, step_filter: int | None = None) -> list[tuple[int, str]]:
    """
    Find and sort checkpoint directories by step number.

    Returns:
        List of (step_number, checkpoint_path) tuples, sorted ascending.
    """
    pattern = os.path.join(run_dir, "checkpoint-*")
    dirs = glob.glob(pattern)

    checkpoints = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        basename = os.path.basename(d)
        match = re.match(r"checkpoint-(\d+)", basename)
        if not match:
            continue
        step = int(match.group(1))
        # Verify model file exists
        has_safetensors = os.path.exists(os.path.join(d, "model.safetensors"))
        has_pytorch = os.path.exists(os.path.join(d, "pytorch_model.bin"))
        if not has_safetensors and not has_pytorch:
            continue
        checkpoints.append((step, d))

    checkpoints.sort(key=lambda x: x[0])

    if step_filter is not None and step_filter > 0:
        checkpoints = [(s, p) for s, p in checkpoints if s % step_filter == 0]

    return checkpoints


def load_sample(cache_dir: str, sample_index: int) -> tuple[torch.Tensor, int]:
    """
    Load a single mel spectrogram sample from the sharded dataset.

    Returns:
        mel_spec: [n_mels, T] tensor
        mel_length: int
    """
    dataset = AudioShardedDataset(cache_dir, columns=["mel_specs"])
    sample = dataset[sample_index]
    return sample["mel_spec"], sample["mel_length"]


@torch.no_grad()
def extract_features_for_checkpoint(
    config: str,
    checkpoint_path: str,
    mel_spec: torch.Tensor,
    mel_length: int,
    device: str,
    num_speakers: int,
    speaker_pooling: str,
) -> list[np.ndarray]:
    """
    Load a checkpoint and extract all hidden layer features for one sample.

    Returns:
        all_hiddens: list of [D, T_valid] numpy arrays (one per layer including conv_subsample),
                     sliced to valid (non-padded) length using feature_lengths.
    """
    model = load_model(
        SpeakerInvariantVoiceEncoder,
        config,
        checkpoint_path=checkpoint_path,
        device=device,
        strict=False,
        overrides={"num_speakers": num_speakers, 'speaker_pooling': speaker_pooling},
    )
    model.eval()

    mel_batch = mel_spec.unsqueeze(0).to(device)  # [1, n_mels, T]
    lengths = torch.tensor([mel_length], device=device, dtype=torch.long)

    result = model(mel_batch, lengths=lengths, grl_alpha=0.0, return_all_hiddens=True)

    all_hiddens_tensors = result["all_hiddens"]  # list of [1, T', D]
    valid_len = result["feature_lengths"][0].item()  # scalar int

    all_hiddens = []
    for h in all_hiddens_tensors:
        # h: [1, T', D] -> slice to valid length -> [D, T_valid]
        feat = h[0, :valid_len, :].transpose(0, 1).cpu().numpy()
        all_hiddens.append(feat)

    del model
    torch.cuda.empty_cache()

    return all_hiddens


def normalize_features(feat: np.ndarray) -> np.ndarray:
    """Normalize features to [0, 1] range per-frame."""
    fmin = feat.min()
    fmax = feat.max()
    return (feat - fmin) / (fmax - fmin + 1e-8)


def build_combined_frame(
    all_layer_feats: list[np.ndarray],
    separator_width: int = 2,
) -> np.ndarray:
    """
    Horizontally concatenate normalized feature maps from all layers with white separators.

    Args:
        all_layer_feats: list of [D, T'] arrays (already normalized)
        separator_width: width of white separator columns

    Returns:
        combined: [max_D, total_width] array
    """
    max_d = max(f.shape[0] for f in all_layer_feats)
    parts = []
    for i, feat in enumerate(all_layer_feats):
        # Pad height if needed
        if feat.shape[0] < max_d:
            pad = np.zeros((max_d - feat.shape[0], feat.shape[1]))
            feat = np.vstack([feat, pad])
        parts.append(feat)
        if i < len(all_layer_feats) - 1:
            sep = np.ones((max_d, separator_width))
            parts.append(sep)
    return np.hstack(parts)


def get_layer_names(num_encoder_blocks: int) -> list[str]:
    """Generate layer names for conv_subsample + encoder blocks."""
    names = ["conv_subsample"]
    for i in range(num_encoder_blocks):
        names.append(f"encoder_block_{i + 1}")
    return names


def generate_single_layer_video(
    frames: list[np.ndarray],
    steps: list[int],
    layer_name: str,
    output_path: str,
    fps: int,
    upscale_factor: int = 4,
):
    """
    Generate an MP4 video for a single layer's features across checkpoints.

    Args:
        frames: list of [D, T'] numpy arrays (one per checkpoint)
        steps: list of step numbers
        layer_name: name for the title
        output_path: path to save MP4
        fps: frames per second
        upscale_factor: nearest-neighbor upscale factor
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05)

    # Initialize with first frame
    first_frame = normalize_features(frames[0])
    first_frame = np.flip(first_frame, axis=0)  # Flip vertically
    first_frame = np.repeat(np.repeat(first_frame, upscale_factor, axis=0), upscale_factor, axis=1)

    im = ax.imshow(first_frame, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(layer_name, fontsize=12)
    ax.axis("off")

    step_text = ax.text(
        0.02, 0.95, f"step {steps[0]}",
        transform=ax.transAxes,
        fontsize=10,
        color="lime",
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
    )

    def update(frame_idx):
        feat = normalize_features(frames[frame_idx])
        feat = np.flip(feat, axis=0)
        feat = np.repeat(np.repeat(feat, upscale_factor, axis=0), upscale_factor, axis=1)
        im.set_data(feat)
        step_text.set_text(f"step {steps[frame_idx]}")
        return [im, step_text]

    anim = FuncAnimation(fig, update, frames=len(frames), blit=True, interval=1000 // fps)
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=[
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-crf", "18", "-pix_fmt", "yuv420p",
    ])
    anim.save(output_path, writer=writer)
    plt.close(fig)


def generate_combined_video(
    all_layer_frames: list[list[np.ndarray]],
    steps: list[int],
    layer_names: list[str],
    output_path: str,
    fps: int,
):
    """
    Generate a combined MP4 with all layers side-by-side at native resolution.

    Args:
        all_layer_frames: list (per layer) of lists (per checkpoint) of [D, T'] arrays
        steps: list of step numbers
        layer_names: names for each layer
        output_path: path to save MP4
        fps: frames per second
    """
    # Build first combined frame to determine dimensions
    first_combined = build_combined_frame(
        [normalize_features(layer_frames[0]) for layer_frames in all_layer_frames]
    )
    h, w = first_combined.shape

    # Scale figure size proportionally
    fig_width = max(12, w / 40)
    fig_height = max(3, h / 40 + 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02)

    combined_flipped = np.flip(first_combined, axis=0)
    im = ax.imshow(combined_flipped, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax.axis("off")

    # Add layer labels along the top
    # Calculate x positions for each layer label
    x_positions = []
    x_offset = 0
    separator_width = 2
    for i, layer_frames in enumerate(all_layer_frames):
        layer_width = layer_frames[0].shape[1]
        x_center = x_offset + layer_width / 2
        x_positions.append(x_center)
        x_offset += layer_width + (separator_width if i < len(all_layer_frames) - 1 else 0)

    for name, x_pos in zip(layer_names, x_positions):
        # Short label: just the index
        short_name = name.replace("encoder_block_", "enc_").replace("conv_subsample", "conv")
        ax.text(
            x_pos, -2, short_name,
            ha="center", va="bottom",
            fontsize=6, color="white",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.5),
        )

    step_text = ax.text(
        0.01, 0.97, f"step {steps[0]}",
        transform=ax.transAxes,
        fontsize=10,
        color="lime",
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
    )

    def update(frame_idx):
        combined = build_combined_frame(
            [normalize_features(layer_frames[frame_idx]) for layer_frames in all_layer_frames]
        )
        combined = np.flip(combined, axis=0)
        im.set_data(combined)
        step_text.set_text(f"step {steps[frame_idx]}")
        return [im, step_text]

    anim = FuncAnimation(fig, update, frames=len(steps), blit=True, interval=1000 // fps)
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=[
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-crf", "18", "-pix_fmt", "yuv420p",
    ])
    anim.save(output_path, writer=writer)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Timelapse of SIVE encoder features across checkpoints")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to training run directory")
    parser.add_argument("--config", type=str, required=True, help="SIVE config name (e.g. tiny_deep)")
    parser.add_argument("--cache_dir", type=str, required=True, help="Path to cached audio dataset")
    parser.add_argument("--sample_index", type=int, default=0, help="Index of sample in dataset")
    parser.add_argument("--output_dir", type=str, default="./timelapse_output", help="Output directory for MP4s")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for videos")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference")
    parser.add_argument("--num_speakers", type=int, default=921, help="Number of speakers (must match training)")
    parser.add_argument("--checkpoint_step_filter", type=int, default=None, help="Only use every Nth step checkpoint")
    parser.add_argument("--speaker_pooling", type=str, default="attentive_statistics", help="Pooling method for speaker classifier (must match training)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Discover checkpoints
    print(f"Discovering checkpoints in {args.run_dir}...")
    checkpoints = discover_checkpoints(args.run_dir, args.checkpoint_step_filter)
    if not checkpoints:
        print("No valid checkpoints found!")
        return
    print(f"Found {len(checkpoints)} checkpoints: steps {checkpoints[0][0]} to {checkpoints[-1][0]}")

    # 2. Load sample
    print(f"Loading sample {args.sample_index} from {args.cache_dir}...")
    mel_spec, mel_length = load_sample(args.cache_dir, args.sample_index)
    print(f"Mel spec shape: {mel_spec.shape}, length: {mel_length}")

    # 3. Extract features for each checkpoint
    steps = []
    # all_features[layer_idx][checkpoint_idx] = [D, T'] numpy array
    all_features: list[list[np.ndarray]] = []

    for step, ckpt_path in tqdm(checkpoints, desc="Extracting features"):
        hiddens = extract_features_for_checkpoint(
            config=args.config,
            checkpoint_path=ckpt_path,
            mel_spec=mel_spec,
            mel_length=mel_length,
            device=args.device,
            num_speakers=args.num_speakers,
            speaker_pooling=args.speaker_pooling,
        )
        steps.append(step)

        # Initialize layer lists on first checkpoint
        if not all_features:
            all_features = [[] for _ in range(len(hiddens))]

        for layer_idx, h in enumerate(hiddens):
            all_features[layer_idx].append(h)

    num_layers = len(all_features)
    layer_names = get_layer_names(num_layers - 1)  # -1 because first is conv_subsample
    print(f"\nExtracted {num_layers} layers across {len(steps)} checkpoints")

    # 4. Generate per-layer MP4s (upscaled)
    print("\nGenerating per-layer videos...")
    for layer_idx in tqdm(range(num_layers), desc="Per-layer videos"):
        layer_name = layer_names[layer_idx]
        filename = f"layer_{layer_idx:02d}_{layer_name}.mp4"
        output_path = os.path.join(args.output_dir, filename)
        generate_single_layer_video(
            frames=all_features[layer_idx],
            steps=steps,
            layer_name=layer_name,
            output_path=output_path,
            fps=args.fps,
        )

    # 5. Generate combined MP4 (native resolution)
    print("Generating combined video...")
    combined_path = os.path.join(args.output_dir, "combined_all_layers.mp4")
    generate_combined_video(
        all_layer_frames=all_features,
        steps=steps,
        layer_names=layer_names,
        output_path=combined_path,
        fps=args.fps,
    )

    # 6. Summary
    print(f"\nDone! Generated {num_layers + 1} videos in {args.output_dir}/")
    print(f"  Per-layer videos: layer_00_conv_subsample.mp4 ... layer_{num_layers - 1:02d}_{layer_names[-1]}.mp4")
    print(f"  Combined video:   combined_all_layers.mp4")
    print(f"  Steps: {steps[0]} to {steps[-1]} ({len(steps)} frames at {args.fps} fps)")


if __name__ == "__main__":
    main()
