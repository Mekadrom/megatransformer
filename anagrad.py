import argparse
import os
import torch
from pretrain_vocoder import VocoderDataCollator, VocoderDataset, create_small_vocoder_model


argparser = argparse.ArgumentParser()

# ex "runs/vocoder/hyperparam_trial_1_0/checkpoint-6500/pytorch_model.bin"

argparser.add_argument("--run_dir", type=str, required=True, help="Directory containing the model checkpoint.")
argparser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory name.")

args = argparser.parse_args()


def analyze_gradient_distribution(model, top_k=10):
    """Find which parameters have the largest gradients."""
    grad_info = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_max = param.grad.abs().max().item()
            grad_info.append({
                'name': name,
                'norm': grad_norm,
                'max': grad_max,
                'shape': tuple(param.shape),
                'numel': param.numel(),
            })
    
    # Sort by norm
    grad_info.sort(key=lambda x: x['norm'], reverse=True)
    
    print(f"\n{'='*60}")
    print(f"Top {top_k} parameters by gradient norm:")
    print(f"{'='*60}")
    for i, info in enumerate(grad_info[:top_k]):
        print(f"{i+1}. {info['name']}")
        print(f"   norm: {info['norm']:.2f}, max: {info['max']:.4f}, shape: {info['shape']}")
    
    total_norm = sum(g['norm']**2 for g in grad_info) ** 0.5
    top_contribution = sum(g['norm']**2 for g in grad_info[:top_k]) ** 0.5
    print(f"\nTotal grad norm: {total_norm:.2f}")
    print(f"Top {top_k} contribute: {top_contribution:.2f} ({100*top_contribution/total_norm:.1f}%)")
    
    return grad_info

model = create_small_vocoder_model(1, 1, 1, 0, 1, False)
config = model.config

path = os.path.join(args.run_dir, args.checkpoint, "pytorch_model.bin")

print(f"Loading model state from {path}")

state_dict = torch.load(path)
model.load_state_dict(state_dict)
model = model.to("cpu")
model.train()

default_path = os.path.join('inference', 'examples', 'test_alm.mp3')

dataset = VocoderDataset(
    config=model.config,
    t5_tokenizer=None,
    approximated_length=1_100_000,
    sample_rate=model.config.audio_sample_rate,
    n_mels=model.config.audio_n_mels,
    n_fft=model.config.audio_n_fft,
    hop_length=model.config.audio_hop_length,
    audio_max_frames=model.config.audio_max_frames,
    cache_dir="./dataset_cache",
    split="train",
)

data_collator = VocoderDataCollator(
    audio_max_frames=model.config.audio_max_frames,
    audio_max_waveform_length=model.config.audio_max_waveform_length,
    n_mels=model.config.audio_n_mels,
)

dataset_iter = iter(dataset)

batch = data_collator([next(dataset_iter) for _ in range(16)])

outputs = model(
    mel_spec=batch["mel_spec"],
    text_input_ids=None,
    text_attention_mask=None,
    waveform_labels=batch["waveform_labels"],
)

recon_loss = outputs["loss"]
recon_loss.backward()

# Call this in your training loop after loss.backward():
analyze_gradient_distribution(model.vocoder)
