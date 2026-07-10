import os
from types import SimpleNamespace
from typing import Optional

import torch

from megatransformer.model.voice.vocoder.vocoder import Vocoder
from megatransformer.model.ema import EMAModel
from megatransformer.utils.audio_utils import SharedWindowBuffer


def load_model(
    model_cls,
    config_name: str,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
    overrides: dict = {},
    strict: bool = False,
    allow_size_mismatch: bool = False,
):
    """
    Load a model from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory (containing model.safetensors or pytorch_model.bin)
        config_name: Config name from the model class (e.g., "small", "medium")
        device: Device to load the model on
        allow_size_mismatch: If True, drop checkpoint keys whose tensor shape
            doesn't match the current model. Use when an unused head's output
            dim differs (e.g. SIVE speaker_classifier across train/eval splits).

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
        if allow_size_mismatch:
            state_dict = _filter_size_mismatched(state_dict, model)
        model.load_state_dict(state_dict, strict=strict)
        print(f"Loaded model from {safetensors_path}")
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location=device, weights_only=True)
        if allow_size_mismatch:
            state_dict = _filter_size_mismatched(state_dict, model)
        model.load_state_dict(state_dict, strict=strict)
        print(f"Loaded model from {pytorch_path}")
    else:
        raise FileNotFoundError(
            f"No model checkpoint found at {checkpoint_path}. "
            f"Expected model.safetensors or pytorch_model.bin"
        )

    return model


def _filter_size_mismatched(state_dict: dict, model: torch.nn.Module) -> dict:
    """Drop checkpoint entries whose tensor shape disagrees with the model's
    parameter at the same key. Prints a one-line warning per dropped key."""
    model_shapes = {n: p.shape for n, p in model.state_dict().items()}
    filtered = {}
    for k, v in state_dict.items():
        target = model_shapes.get(k)
        if target is not None and tuple(target) != tuple(v.shape):
            print(f"  Skipping size-mismatched key {k}: ckpt {tuple(v.shape)} vs model {tuple(target)}")
            continue
        filtered[k] = v
    return filtered


class PretrainedVocoderWrapper(torch.nn.Module):
    """Wraps a pretrained vocoder to match the expected interface.

    Expected interface: vocoder(mel_spec) -> {"pred_waveform": waveform}
    where mel_spec is [B, n_mels, T] and waveform is [B, T] or [T].
    """
    def __init__(self, vocoder: torch.nn.Module, name: str = "pretrained",
                 hop_length: int = 256, sample_rate: int = 16000):
        super().__init__()
        self.vocoder = vocoder
        self.name = name
        # Advertise the frame rate this vocoder was TRAINED at so callers can
        # resample a mismatched-hop mel before synthesis (e.g. a 50 Hz ContentVec
        # SMG mel @hop320 driving this 62.5 Hz @hop256 HiFi-GAN). Every vocoding
        # call site reads `getattr(vocoder, "config", None).hop_length`; without
        # this the pretrained wrapper had no `.config`, callers fell back to the
        # MEL hop, saw "no mismatch", skipped the resample, and the 50 Hz mel
        # played 62.5/50 = 1.25x too fast (correct pitch, faster speech).
        self.config = SimpleNamespace(hop_length=hop_length, sample_rate=sample_rate)

    def forward(self, mel_spec: torch.Tensor) -> dict[str, torch.Tensor]:
        waveform = self.vocoder(mel_spec)
        # Handle various output formats
        if isinstance(waveform, dict):
            return waveform
        if isinstance(waveform, (tuple, list)):
            waveform = waveform[0]
        # Squeeze channel dim if present: [B, 1, T] -> [B, T]
        if waveform.dim() == 3 and waveform.shape[1] == 1:
            waveform = waveform.squeeze(1)
        return {"pred_waveform": waveform}


def _load_pretrained_hifigan():
    """Load SpeechBrain HiFi-GAN trained on LibriTTS at 16kHz.

    Mel parameters: 80 bins, 1024 n_fft, 256 hop_length, slaney norm, f_max=8000.
    """
    try:
        from speechbrain.inference.vocoders import HIFIGAN
    except ImportError:
        raise ImportError(
            "speechbrain is required for pretrained HiFi-GAN vocoder. "
            "Install it with: pip install speechbrain"
        )

    # Patch huggingface_hub to handle deprecated use_auth_token kwarg
    import huggingface_hub
    _original_hf_download = huggingface_hub.hf_hub_download
    def _patched_hf_download(*args, **kwargs):
        kwargs.pop("use_auth_token", None)
        return _original_hf_download(*args, **kwargs)
    huggingface_hub.hf_hub_download = _patched_hf_download

    try:
        # Pass an explicit, parseable device string. SpeechBrain otherwise
        # auto-detects the bare string "cuda", fails to split it into
        # (type, index), and prints "Could not parse CUDA device string 'cuda'
        # ... Falling back to device 0". The fallback lands on the right GPU
        # anyway (CUDA_VISIBLE_DEVICES remaps it to index 0), so this is purely
        # cosmetic — it just silences the warning.
        hifi_gan = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-libritts-16kHz",
            savedir="pretrained_models/tts-hifigan-libritts-16kHz",
            run_opts={"device": "cuda:0" if torch.cuda.is_available() else "cpu"},
        )
    finally:
        huggingface_hub.hf_hub_download = _original_hf_download

    # Extract the generator directly from SpeechBrain's ModuleDict and cast to float32.
    # We bypass decode_batch/infer because those only cast device (not dtype),
    # causing bf16 mismatches when training with --bf16.
    generator = hifi_gan.mods.generator.float()

    class HiFiGANInner(torch.nn.Module):
        def __init__(self, gen):
            super().__init__()
            self.generator = gen

        def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
            mel_spec = mel_spec.float()
            return self.generator.inference(mel_spec).squeeze(1)  # [B, 1, T] -> [B, T]

    inner = HiFiGANInner(generator)
    # SpeechBrain LibriTTS HiFi-GAN: 16 kHz, 256 hop_length (62.5 Hz mel frames).
    return PretrainedVocoderWrapper(inner, name="hifigan-libritts-16khz",
                                    hop_length=256, sample_rate=16000)


PRETRAINED_VOCODERS = {
    "hifigan": _load_pretrained_hifigan,
}


def load_vocoder(vocoder_checkpoint_path, vocoder_config, shared_window_buffer, is_wrapped: bool = False):
    """Load vocoder for mel-to-waveform conversion.

    For pretrained vocoders, set vocoder_config to a pretrained name (e.g. "hifigan")
    and vocoder_checkpoint_path can be None.

    Available pretrained vocoders:
        - "hifigan": SpeechBrain HiFi-GAN trained on LibriTTS at 16kHz
                     (80 mel bins, 256 hop_length, 1024 n_fft, slaney norm)
    """
    # Check for pretrained vocoders first
    if vocoder_config in PRETRAINED_VOCODERS:
        try:
            vocoder = PRETRAINED_VOCODERS[vocoder_config]()
            vocoder.eval()
            print(f"Loaded pretrained vocoder: {vocoder.name}")
            print(f"Vocoder parameters: {sum(p.numel() for p in vocoder.parameters()):,}")
            return vocoder
        except Exception as e:
            print(f"Failed to load pretrained vocoder '{vocoder_config}': {e}")
            raise e

    # Fall back to custom vocoder loading
    if vocoder_checkpoint_path is None:
        return

    if not os.path.exists(vocoder_checkpoint_path):
        print(f"Vocoder checkpoint not found at {vocoder_checkpoint_path}")
        return

    try:
        if is_wrapped:
            class VocoderWrapper(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.vocoder: Optional[Vocoder] = None

                @classmethod
                def from_config(cls, config_name: str, shared_window_buffer: Optional[SharedWindowBuffer], **overrides) -> "VocoderWrapper":
                    wrapper = cls()
                    wrapper.vocoder = Vocoder.from_config(config_name, shared_window_buffer=shared_window_buffer, **overrides)
                    return wrapper

            vocoder = load_model(VocoderWrapper, vocoder_config, checkpoint_path=vocoder_checkpoint_path, overrides={"shared_window_buffer": shared_window_buffer}, strict=False).vocoder
            vocoder.eval()
        else:
            vocoder = load_model(Vocoder, vocoder_config, checkpoint_path=vocoder_checkpoint_path, overrides={"shared_window_buffer": shared_window_buffer}, strict=False)
            vocoder.eval()
    except Exception as e:
        print(f"Failed to load vocoder: {e}")
        raise e

    print(f"Loaded vocoder from {vocoder_checkpoint_path}")
    print(f"Vocoder parameters: {sum(p.numel() for p in vocoder.parameters()):,}")
    print(f"Vocoder structure: {vocoder}")
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
