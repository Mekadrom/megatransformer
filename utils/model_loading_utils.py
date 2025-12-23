import os
import glob

import torch
import torch.nn as nn


from . import configuration
from model.audio.shared_window_buffer import SharedWindowBuffer
from model.audio.vocoders.freq_domain_vocoder import LightHeadedFrequencyDomainVocoder, SplitBandLowFreqMeanFreqDomainVocoder, SplitBandFrequencyDomainVocoder
from model.audio.vocoders.vocoders import VocoderWithLoss


def load_model(finetune, model, run_dir):
    # load if model file exists
    if os.path.exists(run_dir):
        globbed_checkpoint_folders = glob.glob(os.path.join(run_dir, "checkpoint-*", "pytorch_model.bin"))
        # sort by step number (format checkpoint-<step>/pytorch_model.bin)
        if globbed_checkpoint_folders:
            sorted_checkpoints = sorted(globbed_checkpoint_folders, key=lambda x: int(x.split("-")[-1].split(os.path.sep)[0]))
            latest_checkpoint = sorted_checkpoints[-1]
            print(f"Loading model from {latest_checkpoint}")
            try:
                model.load_state_dict(torch.load(latest_checkpoint), strict=False)
                model_loaded = True
            except RuntimeError as e:
                print(f"Error loading model: {e}. This is most likely due to a mismatch in model architecture.")
                model_loaded = False
        else:
            print(f"No checkpoints found in {run_dir}.")
            model_loaded = False
    else:
        print(f"Model directory {run_dir} does not exist.")
        model_loaded = False

    if not model_loaded:
        print("Model not loaded from checkpoint.")
        if finetune:
            raise ValueError("Fine-tuning is enabled but no checkpoint found. Please check the run directory, or your configuration.")

    return model, model_loaded


def create_vocoder_from_config(
    vocoder_config: dict,
    shared_window_buffer: SharedWindowBuffer | None = None,
) -> nn.Module:
    """
    Create a vocoder instance from a config dict.
    This is used to load pruned checkpoints.
    """
    if shared_window_buffer is None:
        shared_window_buffer = SharedWindowBuffer()

    vocoder_class_name = vocoder_config['vocoder_class']

    common_args = {
        'shared_window_buffer': shared_window_buffer,
        'n_mels': vocoder_config['n_mels'],
        'n_fft': vocoder_config['n_fft'],
        'hop_length': vocoder_config['hop_length'],
        'hidden_dim': vocoder_config['hidden_dim'],
        'num_layers': vocoder_config['num_layers'],
        'convnext_mult': vocoder_config['convnext_mult'],
    }

    if vocoder_class_name == 'LightHeadedFrequencyDomainVocoder':
        return LightHeadedFrequencyDomainVocoder(**common_args)

    elif vocoder_class_name == 'SplitBandFrequencyDomainVocoder':
        return SplitBandFrequencyDomainVocoder(
            **common_args,
            cutoff_bin=vocoder_config['cutoff_bin'],
        )

    elif vocoder_class_name == 'SplitBandLowFreqMeanFreqDomainVocoder':
        return SplitBandLowFreqMeanFreqDomainVocoder(
            **common_args,
            cutoff_bin=vocoder_config['cutoff_bin'],
            low_freq_kernel=vocoder_config['low_freq_kernel'],
            high_freq_kernel=vocoder_config['high_freq_kernel'],
        )

    else:
        raise ValueError(f"Unknown vocoder class: {vocoder_class_name}")


def load_pruned_vocoder(
    checkpoint_path: str,
    device: str | torch.device = 'cpu',
    shared_window_buffer: SharedWindowBuffer | None = None,
) -> VocoderWithLoss:
    """
    Load a pruned vocoder checkpoint.

    This function automatically reconstructs the model architecture from the
    saved config, so you don't need to know the exact dimensions.

    Usage:
        model = load_pruned_vocoder('runs/my_vocoder_pruned/pruned_checkpoint.pt')
        model.eval()
        with torch.no_grad():
            waveform, stft = model.vocoder(mel_spec)

    Args:
        checkpoint_path: Path to the pruned checkpoint
        device: Device to load the model to
        shared_window_buffer: Optional shared buffer (created if not provided)

    Returns:
        VocoderWithLoss model with loaded weights
    """
    if shared_window_buffer is None:
        shared_window_buffer = SharedWindowBuffer()

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Check if this is a pruned checkpoint with embedded config
    if 'vocoder_config' not in checkpoint:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not contain 'vocoder_config'. "
            "This checkpoint was likely created before config embedding was added. "
            "You'll need to manually create the model with the correct hidden_dim."
        )

    vocoder_config = checkpoint['vocoder_config']
    loss_config = checkpoint.get('loss_config', {})

    # Create vocoder from config
    vocoder = create_vocoder_from_config(vocoder_config, shared_window_buffer)

    # Create MegaTransformerConfig for VocoderWithLoss
    mt_config = configuration.MegaTransformerConfig(
        audio_n_mels=vocoder_config['n_mels'],
        audio_n_fft=vocoder_config['n_fft'],
        audio_hop_length=vocoder_config['hop_length'],
    )

    # Create VocoderWithLoss wrapper
    model_with_loss = VocoderWithLoss(
        vocoder=vocoder,
        shared_window_buffer=shared_window_buffer,
        config=mt_config,
        **loss_config,
    )

    # Load state dict
    model_with_loss.load_state_dict(checkpoint['model_state_dict'])

    return model_with_loss.to(device)
