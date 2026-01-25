import os
from typing import Optional

import torch

from model.audio.vocoder.vocoder import Vocoder
from model.ema import EMAModel
from utils.audio_utils import SharedWindowBuffer


def load_model(
    model_cls,
    config_name: str,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
    overrides: dict = {},
    strict: bool = False
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
        model.load_state_dict(state_dict, strict=strict)
        print(f"Loaded model from {safetensors_path}")
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=strict)
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

    class VocoderWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vocoder: Optional[Vocoder] = None

        @classmethod
        def from_config(cls, config_name: str, shared_window_buffer: Optional[SharedWindowBuffer], **overrides) -> "VocoderWrapper":
            wrapper = cls()
            wrapper.vocoder = Vocoder.from_config(config_name, shared_window_buffer=shared_window_buffer, **overrides)
            return wrapper

    try:
        vocoder = load_model(VocoderWrapper, vocoder_config, checkpoint_path=vocoder_checkpoint_path, overrides={"shared_window_buffer": shared_window_buffer}, strict=True).vocoder
        vocoder.eval()
        print(f"Loaded vocoder from {vocoder_checkpoint_path}")
        print(f"Vocoder parameters: {sum(p.numel() for p in vocoder.parameters()):,}")
    except Exception as e:
        print(f"Failed to load vocoder: {e}")
        raise e
    return vocoder


def load_discriminator(
    resume_from_checkpoint: str,
    discriminator: torch.nn.Module,
    discriminator_optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.nn.Module, Optional[torch.optim.Optimizer], bool]:
    """
    Load discriminator from checkpoint if it exists.

    Handles errors gracefully - if loading fails, returns the fresh discriminator
    and continues training from scratch.
    """
    if resume_from_checkpoint is None:
        print("No checkpoint path provided, training discriminator from scratch")
        return discriminator, discriminator_optimizer, False

    discriminator_path = os.path.join(resume_from_checkpoint, "discriminator.pt")
    if os.path.exists(discriminator_path):
        print(f"Loading discriminator from {discriminator_path}")
        try:
            checkpoint = torch.load(discriminator_path, map_location=device, weights_only=True)
            discriminator.load_state_dict(checkpoint["discriminator_state_dict"], strict=False)

            if discriminator_optimizer is not None and checkpoint.get("discriminator_optimizer_state_dict"):
                try:
                    discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer_state_dict"])
                except Exception as e:
                    print(f"Warning: Failed to load discriminator optimizer state: {e}")
                    print("Continuing with fresh optimizer state...")

            return discriminator, discriminator_optimizer, True
        except Exception as e:
            print(f"Warning: Failed to load discriminator checkpoint: {e}")
            print("Continuing with fresh discriminator...")
            return discriminator, discriminator_optimizer, False

    print("No existing discriminator checkpoint found, training from scratch")
    return discriminator, discriminator_optimizer, False


def load_learned_speaker_classifier(
    resume_from_checkpoint: str,
    learned_speaker_classifier: torch.nn.Module,
    learned_speaker_classifier_optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.nn.Module, Optional[torch.optim.Optimizer], bool]:
    """
    Load learned speaker classifier (speaker ID on embeddings) from checkpoint if it exists.

    Handles errors gracefully - if loading fails, returns the fresh classifier
    and continues training from scratch.
    """
    if resume_from_checkpoint is None:
        print("No checkpoint path provided, training learned speaker classifier from scratch")
        return learned_speaker_classifier, learned_speaker_classifier_optimizer, False

    learned_speaker_classifier_path = os.path.join(resume_from_checkpoint, "learned_speaker_classifier.pt")
    if os.path.exists(learned_speaker_classifier_path):
        print(f"Loading learned speaker classifier from {learned_speaker_classifier_path}")
        try:
            checkpoint = torch.load(learned_speaker_classifier_path, map_location=device, weights_only=True)
            learned_speaker_classifier.load_state_dict(checkpoint["learned_speaker_classifier_state_dict"])

            if learned_speaker_classifier_optimizer is not None and checkpoint.get("learned_speaker_classifier_optimizer_state_dict"):
                try:
                    learned_speaker_classifier_optimizer.load_state_dict(checkpoint["learned_speaker_classifier_optimizer_state_dict"])
                except Exception as e:
                    print(f"Warning: Failed to load learned speaker classifier optimizer state: {e}")
                    print("Continuing with fresh optimizer state...")

            return learned_speaker_classifier, learned_speaker_classifier_optimizer, True
        except Exception as e:
            print(f"Warning: Failed to load learned speaker classifier checkpoint: {e}")
            print("Continuing with fresh learned speaker classifier...")
            return learned_speaker_classifier, learned_speaker_classifier_optimizer, False

    print("No existing learned speaker classifier checkpoint found, training from scratch")
    return learned_speaker_classifier, learned_speaker_classifier_optimizer, False



def load_ema_state(ema: EMAModel, checkpoint_path: str) -> bool:
    """Load EMA state from checkpoint if available.

    Args:
        ema: The EMA model to load state into
        checkpoint_path: Path to checkpoint directory

    Returns:
        True if EMA state was loaded, False otherwise
    """
    ema_path = os.path.join(checkpoint_path, "ema_state.pt")
    if os.path.exists(ema_path):
        state_dict = torch.load(ema_path, map_location="cpu")
        ema.load_state_dict(state_dict)
        print(f"Loaded EMA state from {ema_path} (step {ema.step})")
        return True
    return False
