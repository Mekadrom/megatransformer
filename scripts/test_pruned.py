import torch
import torchaudio

from utils.audio_utils import SharedWindowBuffer
from utils.model_loading_utils import load_pruned_vocoder


# Load the pruned model
model = load_pruned_vocoder('runs/vocoder/splitband_lowfreq_mean_0_1/pruned_checkpoint.pt', device='cuda')
model.eval()

# Test with a mel spectrogram
# Option 1: From an audio file
waveform, sr = torchaudio.load('inference/examples/test_alm_1.mp3')
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

# Compute mel spectrogram (you'll need your mel transform)
shared_buffer = SharedWindowBuffer()
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=256,
    n_mels=80,
)
mel = mel_transform(waveform)
mel = torch.log(mel.clamp(min=1e-5))  # Log mel

# Run inference
with torch.no_grad():
    mel = mel.to('cuda')
    waveform_out, stft = model.vocoder(mel)

# Save output
torchaudio.save('output.wav', waveform_out.cpu(), 16000)
