import os
from typing import Optional

import torch

from model.audio.vocoder.vocoder import Vocoder


def load_model(
    model_cls,
    config_name: str,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
    overrides: dict = {},
):
    """
    Load a model from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory (containing model.safetensors or pytorch_model.bin)
        config_name: Config name from the model class (e.g., "small", "medium")
        device: Device to load the model on

    Returns:
        model in eval mode
    """
    # Create model with same config
    model = model_cls.from_config(config_name, **overrides)
    model = model.to(device)

    if checkpoint_path is None:
        return model

    # Try to load from safetensors first, then pytorch_model.bin
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model from {safetensors_path}")
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model from {pytorch_path}")
    else:
        raise FileNotFoundError(
            f"No model checkpoint found at {checkpoint_path}. "
            f"Expected model.safetensors or pytorch_model.bin"
        )

    return model


def load_vocoder(vocoder_checkpoint_path, vocoder_config, shared_window_buffer):
    """Lazily load vocoder on first use."""
    if vocoder_checkpoint_path is None:
        return

    if not os.path.exists(vocoder_checkpoint_path):
        print(f"Vocoder checkpoint not found at {vocoder_checkpoint_path}")
        return

    try:
        vocoder = load_model(Vocoder, vocoder_config, checkpoint_path=vocoder_checkpoint_path, overrides={"shared_window_buffer": shared_window_buffer})

        # Remove weight normalization for inference optimization
        if hasattr(vocoder.vocoder, 'remove_weight_norm'):
            vocoder.vocoder.remove_weight_norm()

        vocoder.eval()
        print(f"Loaded vocoder from {vocoder_checkpoint_path}")
        print(f"Vocoder parameters: {sum(p.numel() for p in vocoder.parameters()):,}")
    except Exception as e:
        print(f"Failed to load vocoder: {e}")
        vocoder = None
    return vocoder
