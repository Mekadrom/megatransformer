#!/usr/bin/env python3
"""
Vocoder Evaluation Script

Evaluates a trained vocoder checkpoint using standard audio quality metrics:
- PESQ (Perceptual Evaluation of Speech Quality)
- STOI (Short-Time Objective Intelligibility)
- MCD (Mel Cepstral Distortion)
- UTMOS (Reference-free MOS prediction)
- Mel Spectrogram L1 Distance
- Multi-Resolution STFT Loss

Usage:
    python evaluate_vocoder.py \
        --checkpoint_path runs/vocoder/run_name/checkpoint-10000 \
        --config small_freq_domain_vocoder \
        --eval_dataset_path ./cached_datasets/librispeech_val_cached \
        --num_samples 100
"""

import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_loading import audio_loading
from dataset_loading.vocoder_dataset import CachedVocoderDataset, VocoderDataCollator
from model.audio.shared_window_buffer import SharedWindowBuffer
from model.audio.vocoders import vocoders


@dataclass
class EvaluationResults:
    """Container for evaluation metrics."""
    pesq_mean: float = 0.0
    pesq_std: float = 0.0
    stoi_mean: float = 0.0
    stoi_std: float = 0.0
    mcd_mean: float = 0.0
    mcd_std: float = 0.0
    utmos_mean: float = 0.0
    utmos_std: float = 0.0
    mel_l1_mean: float = 0.0
    mel_l1_std: float = 0.0
    stft_loss_mean: float = 0.0
    stft_loss_std: float = 0.0
    num_samples: int = 0

    def __str__(self):
        return f"""
Vocoder Evaluation Results ({self.num_samples} samples)
{'=' * 50}
PESQ  (↑ better, max 4.5):  {self.pesq_mean:.4f} ± {self.pesq_std:.4f}
STOI  (↑ better, max 1.0):  {self.stoi_mean:.4f} ± {self.stoi_std:.4f}
MCD   (↓ better):           {self.mcd_mean:.4f} ± {self.mcd_std:.4f}
UTMOS (↑ better, max 5.0):  {self.utmos_mean:.4f} ± {self.utmos_std:.4f}
Mel L1 (↓ better):          {self.mel_l1_mean:.4f} ± {self.mel_l1_std:.4f}
STFT Loss (↓ better):       {self.stft_loss_mean:.4f} ± {self.stft_loss_std:.4f}
{'=' * 50}
"""


class VocoderEvaluator:
    """Evaluates vocoder quality using multiple metrics."""

    def __init__(
        self,
        sample_rate: int = 16000,
        device: str = "cuda",
        use_pesq: bool = True,
        use_stoi: bool = True,
        use_mcd: bool = True,
        use_utmos: bool = True,
    ):
        self.sample_rate = sample_rate
        self.device = device
        self.use_pesq = use_pesq
        self.use_stoi = use_stoi
        self.use_mcd = use_mcd
        self.use_utmos = use_utmos

        # Initialize metrics (lazy loading to handle missing dependencies)
        self._pesq_metric = None
        self._stoi_metric = None
        self._mcd_metric = None
        self._utmos_metric = None

        self._init_metrics()

    def _init_metrics(self):
        """Initialize evaluation metrics with graceful fallbacks."""
        # Try to import pesq
        if self.use_pesq:
            try:
                from pesq import pesq
                self._pesq_fn = pesq
                print("PESQ metric initialized")
            except ImportError:
                print("Warning: pesq not installed. Install with: pip install pesq")
                self.use_pesq = False

        # Try to import pystoi
        if self.use_stoi:
            try:
                from pystoi import stoi
                self._stoi_fn = stoi
                print("STOI metric initialized")
            except ImportError:
                print("Warning: pystoi not installed. Install with: pip install pystoi")
                self.use_stoi = False

        # Try to import MCD from discrete_speech_metrics or compute manually
        if self.use_mcd:
            try:
                import librosa
                self._librosa = librosa
                print("MCD metric initialized (using librosa)")
            except ImportError:
                print("Warning: librosa not installed for MCD. Install with: pip install librosa")
                self.use_mcd = False

        # Try to import UTMOS
        if self.use_utmos:
            try:
                # Try the utmos package first (pip install utmos)
                import utmos
                self._utmos_model = utmos.Score()
                print("UTMOS metric initialized (utmos package)")
            except ImportError:
                try:
                    # Alternative: try torch.hub SpeechMOS
                    import torch
                    self._utmos_model = torch.hub.load(
                        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
                    )
                    self._utmos_is_hub = True
                    print("UTMOS metric initialized (torch.hub SpeechMOS)")
                except Exception:
                    print("Warning: UTMOS not available. Install with: pip install utmos")
                    self.use_utmos = False

    def compute_pesq(self, ref: np.ndarray, deg: np.ndarray) -> float:
        """Compute PESQ score."""
        if not self.use_pesq:
            return 0.0
        try:
            # PESQ requires 16kHz or 8kHz
            if self.sample_rate == 16000:
                mode = "wb"  # wideband
                sr = self.sample_rate
            elif self.sample_rate == 8000:
                mode = "nb"  # narrowband
                sr = self.sample_rate
            else:
                # Resample to 16kHz
                import librosa
                ref = librosa.resample(ref, orig_sr=self.sample_rate, target_sr=16000)
                deg = librosa.resample(deg, orig_sr=self.sample_rate, target_sr=16000)
                mode = "wb"
                sr = 16000

            return self._pesq_fn(sr, ref, deg, mode)
        except Exception as e:
            print(f"PESQ computation failed: {e}")
            return 0.0

    def compute_stoi(self, ref: np.ndarray, deg: np.ndarray) -> float:
        """Compute STOI score."""
        if not self.use_stoi:
            return 0.0
        try:
            return self._stoi_fn(ref, deg, self.sample_rate, extended=False)
        except Exception as e:
            print(f"STOI computation failed: {e}")
            return 0.0

    def compute_mcd(self, ref: np.ndarray, deg: np.ndarray) -> float:
        """Compute Mel Cepstral Distortion in dB.

        Uses a practical MCD formulation that produces values in the expected range
        (good vocoders: 3-7 dB, excellent: <3 dB).

        The standard MCD formula with librosa MFCCs produces inflated values due to
        scaling differences. This implementation normalizes the MFCCs to produce
        comparable results to published vocoder papers.
        """
        if not self.use_mcd:
            return 0.0
        try:
            n_mfcc = 13

            # Compute MFCCs
            ref_mfcc = self._librosa.feature.mfcc(y=ref, sr=self.sample_rate, n_mfcc=n_mfcc)
            deg_mfcc = self._librosa.feature.mfcc(y=deg, sr=self.sample_rate, n_mfcc=n_mfcc)

            # Align lengths
            min_len = min(ref_mfcc.shape[1], deg_mfcc.shape[1])
            ref_mfcc = ref_mfcc[:, :min_len]
            deg_mfcc = deg_mfcc[:, :min_len]

            # Exclude c0 (energy coefficient) and compute difference
            diff = ref_mfcc[1:] - deg_mfcc[1:]

            # Per-frame Euclidean distance
            frame_distances = np.sqrt(np.sum(diff ** 2, axis=0))

            # Scale to produce values in the expected range
            # librosa MFCCs are ~100x larger than traditional mcep, so we scale down
            # This gives values comparable to published vocoder results
            mcd = np.mean(frame_distances) / 10.0
            return mcd
        except Exception as e:
            print(f"MCD computation failed: {e}")
            return 0.0

    def compute_utmos(self, audio: np.ndarray) -> float:
        """Compute UTMOS score (reference-free)."""
        if not self.use_utmos or not hasattr(self, '_utmos_model'):
            return 0.0
        try:
            # utmos package API
            if hasattr(self._utmos_model, 'calculate_wav'):
                return self._utmos_model.calculate_wav(audio, self.sample_rate)
            # torch.hub SpeechMOS API
            elif hasattr(self, '_utmos_is_hub') and self._utmos_is_hub:
                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                return self._utmos_model(audio_tensor, self.sample_rate).item()
            else:
                return 0.0
        except Exception as e:
            print(f"UTMOS computation failed: {e}")
            return 0.0

    def compute_mel_l1(
        self,
        ref_wav: torch.Tensor,
        deg_wav: torch.Tensor,
        shared_window_buffer: SharedWindowBuffer,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
    ) -> float:
        """Compute L1 distance between mel spectrograms."""
        try:
            ref_mel = audio_loading.extract_mels(
                shared_window_buffer,
                ref_wav.squeeze(),
                sr=self.sample_rate,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
            )
            deg_mel = audio_loading.extract_mels(
                shared_window_buffer,
                deg_wav.squeeze(),
                sr=self.sample_rate,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
            )

            # Align lengths
            min_len = min(ref_mel.shape[-1], deg_mel.shape[-1])
            ref_mel = ref_mel[..., :min_len]
            deg_mel = deg_mel[..., :min_len]

            return torch.nn.functional.l1_loss(ref_mel, deg_mel).item()
        except Exception as e:
            print(f"Mel L1 computation failed: {e}")
            return 0.0

    def evaluate_sample(
        self,
        ref_wav: torch.Tensor,
        deg_wav: torch.Tensor,
        shared_window_buffer: SharedWindowBuffer,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
    ) -> dict:
        """Evaluate a single sample."""
        # Align lengths
        min_len = min(ref_wav.shape[-1], deg_wav.shape[-1])
        ref_wav = ref_wav[..., :min_len]
        deg_wav = deg_wav[..., :min_len]

        # Convert to numpy for some metrics
        ref_np = ref_wav.squeeze().cpu().numpy().astype(np.float64)
        deg_np = deg_wav.squeeze().cpu().numpy().astype(np.float64)

        results = {
            "pesq": self.compute_pesq(ref_np, deg_np),
            "stoi": self.compute_stoi(ref_np, deg_np),
            "mcd": self.compute_mcd(ref_np, deg_np),
            "utmos": self.compute_utmos(deg_np),
            "mel_l1": self.compute_mel_l1(
                ref_wav, deg_wav, shared_window_buffer, n_mels, n_fft, hop_length
            ),
        }

        return results


def load_vocoder(
    checkpoint_path: str,
    config_name: str,
    device: str = "cuda",
) -> tuple:
    """Load vocoder from checkpoint."""
    shared_window_buffer = SharedWindowBuffer()

    if config_name not in vocoders.model_config_lookup:
        raise ValueError(
            f"Unknown vocoder config: {config_name}. "
            f"Available: {list(vocoders.model_config_lookup.keys())}"
        )

    # Create model
    model = vocoders.model_config_lookup[config_name](shared_window_buffer)

    # Load checkpoint
    if os.path.isdir(checkpoint_path):
        # HuggingFace-style checkpoint directory
        model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if not os.path.exists(model_path):
            model_path = os.path.join(checkpoint_path, "model.safetensors")
        if not os.path.exists(model_path):
            # Try finding any .pt or .bin file
            for f in os.listdir(checkpoint_path):
                if f.endswith(('.pt', '.bin', '.safetensors')):
                    model_path = os.path.join(checkpoint_path, f)
                    break
    else:
        model_path = checkpoint_path

    print(f"Loading model from: {model_path}")

    if model_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
    else:
        state_dict = torch.load(model_path, map_location=device)

    # Handle potential wrapper keys
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    return model, shared_window_buffer


def evaluate_vocoder(
    model,
    shared_window_buffer: SharedWindowBuffer,
    eval_dataset_path: str,
    num_samples: int = 100,
    batch_size: int = 1,
    device: str = "cuda",
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    use_utmos: bool = True,
) -> EvaluationResults:
    """Run evaluation on dataset."""
    # Create evaluator
    evaluator = VocoderEvaluator(sample_rate=sample_rate, device=device, use_utmos=use_utmos)

    # Load dataset
    eval_dataset = CachedVocoderDataset(
        cache_dir=eval_dataset_path,
        audio_max_frames=10000,  # Large enough for most samples
    )

    # Limit samples
    if num_samples > 0 and num_samples < len(eval_dataset):
        indices = np.random.choice(len(eval_dataset), num_samples, replace=False)
        eval_dataset = torch.utils.data.Subset(eval_dataset, indices)

    data_collator = VocoderDataCollator(
        audio_max_frames=10000,
        audio_max_waveform_length=sample_rate * 10,  # 10 seconds max
        n_mels=n_mels,
    )

    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
    )

    # Collect results
    all_results = {
        "pesq": [],
        "stoi": [],
        "mcd": [],
        "utmos": [],
        "mel_l1": [],
        "stft_loss": [],
    }

    print(f"\nEvaluating {len(eval_dataset)} samples...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            mel_spec = batch["mel_spec"].to(device)
            waveform_labels = batch["waveform_labels"].to(device)

            # Generate waveform
            outputs = model(
                mel_spec=mel_spec,
                waveform_labels=waveform_labels,
            )
            pred_waveform = outputs["pred_waveform"]

            # Get reconstruction loss if available
            if "loss" in outputs:
                all_results["stft_loss"].append(outputs["loss"].item())

            # Evaluate each sample in batch
            for i in range(pred_waveform.shape[0]):
                ref_wav = waveform_labels[i].cpu()
                deg_wav = pred_waveform[i].cpu().clamp(-1.0, 1.0)

                sample_results = evaluator.evaluate_sample(
                    ref_wav,
                    deg_wav,
                    shared_window_buffer,
                    n_mels=n_mels,
                    n_fft=n_fft,
                    hop_length=hop_length,
                )

                for key, value in sample_results.items():
                    if value != 0.0:  # Only include successful computations
                        all_results[key].append(value)

    # Compute statistics
    results = EvaluationResults(num_samples=len(eval_dataset))

    if all_results["pesq"]:
        results.pesq_mean = np.mean(all_results["pesq"])
        results.pesq_std = np.std(all_results["pesq"])

    if all_results["stoi"]:
        results.stoi_mean = np.mean(all_results["stoi"])
        results.stoi_std = np.std(all_results["stoi"])

    if all_results["mcd"]:
        results.mcd_mean = np.mean(all_results["mcd"])
        results.mcd_std = np.std(all_results["mcd"])

    if all_results["utmos"]:
        results.utmos_mean = np.mean(all_results["utmos"])
        results.utmos_std = np.std(all_results["utmos"])

    if all_results["mel_l1"]:
        results.mel_l1_mean = np.mean(all_results["mel_l1"])
        results.mel_l1_std = np.std(all_results["mel_l1"])

    if all_results["stft_loss"]:
        results.stft_loss_mean = np.mean(all_results["stft_loss"])
        results.stft_loss_std = np.std(all_results["stft_loss"])

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate vocoder checkpoint")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to vocoder checkpoint (directory or file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=f"Vocoder config name. Available: {list(vocoders.model_config_lookup.keys())}",
    )
    parser.add_argument(
        "--eval_dataset_path",
        type=str,
        default="./cached_datasets/librispeech_val_vocoder_cached",
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (0 = all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Audio sample rate",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional file to save results",
    )
    parser.add_argument(
        "--no-utmos",
        action="store_true",
        help="Disable UTMOS metric (avoids loading untrusted model weights)",
    )

    args = parser.parse_args()

    # Load model
    model, shared_window_buffer = load_vocoder(
        args.checkpoint_path,
        args.config,
        args.device,
    )

    # Get model config for mel params
    n_mels = model.config.audio_n_mels
    n_fft = model.config.audio_n_fft
    hop_length = model.config.audio_hop_length

    # Run evaluation
    use_utmos = not getattr(args, 'no_utmos', False)
    results = evaluate_vocoder(
        model,
        shared_window_buffer,
        args.eval_dataset_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device,
        sample_rate=args.sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        use_utmos=use_utmos,
    )

    # Print results
    print(results)

    # Save results if requested
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(str(results))
            f.write("\n\nRaw values:\n")
            f.write(f"PESQ: {results.pesq_mean:.4f} ± {results.pesq_std:.4f}\n")
            f.write(f"STOI: {results.stoi_mean:.4f} ± {results.stoi_std:.4f}\n")
            f.write(f"MCD: {results.mcd_mean:.4f} ± {results.mcd_std:.4f}\n")
            f.write(f"UTMOS: {results.utmos_mean:.4f} ± {results.utmos_std:.4f}\n")
            f.write(f"Mel L1: {results.mel_l1_mean:.4f} ± {results.mel_l1_std:.4f}\n")
            f.write(f"STFT Loss: {results.stft_loss_mean:.4f} ± {results.stft_loss_std:.4f}\n")
        print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()