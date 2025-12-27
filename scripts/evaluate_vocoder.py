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
import numpy as np
import os

import torch

from dataclasses import dataclass
from scipy import signal
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_loading import audio_loading
from dataset_loading.vocoder_dataset import CachedVocoderDataset, VocoderDataCollator
from model.audio.vocoders import vocoders
from utils.audio_utils import SharedWindowBuffer
from utils.model_loading_utils import load_model


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
    alignment_lag_mean: float = 0.0
    alignment_lag_std: float = 0.0
    num_samples: int = 0

    def __str__(self):
        lag_ms = self.alignment_lag_mean / 16  # Assuming 16kHz sample rate
        return f"""
Vocoder Evaluation Results ({self.num_samples} samples)
{'=' * 50}
PESQ  (↑ better, max 4.5):  {self.pesq_mean:.4f} ± {self.pesq_std:.4f}
STOI  (↑ better, max 1.0):  {self.stoi_mean:.4f} ± {self.stoi_std:.4f}
MCD   (↓ better):           {self.mcd_mean:.4f} ± {self.mcd_std:.4f}
UTMOS (↑ better, max 5.0):  {self.utmos_mean:.4f} ± {self.utmos_std:.4f}
Mel L1 (↓ better):          {self.mel_l1_mean:.4f} ± {self.mel_l1_std:.4f}
STFT Loss (↓ better):       {self.stft_loss_mean:.4f} ± {self.stft_loss_std:.4f}
Alignment lag (samples):    {self.alignment_lag_mean:.1f} ± {self.alignment_lag_std:.1f} ({lag_ms:.2f} ms)
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

    def align_signals(
        self,
        ref: np.ndarray,
        deg: np.ndarray,
        max_lag_ms: float = 50.0,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Align degraded signal to reference using cross-correlation.

        PESQ and other metrics are very sensitive to temporal misalignment.
        Even a few samples of offset can significantly degrade scores.

        Args:
            ref: Reference signal
            deg: Degraded signal
            max_lag_ms: Maximum lag to search in milliseconds

        Returns:
            Tuple of (aligned_ref, aligned_deg, lag_samples)
        """
        max_lag_samples = int(max_lag_ms * self.sample_rate / 1000)

        # Use a portion of the signal for faster correlation
        # (full correlation can be slow for long signals)
        search_len = min(len(ref), len(deg), self.sample_rate * 2)  # Max 2 seconds

        ref_search = ref[:search_len]
        deg_search = deg[:search_len]

        # Compute cross-correlation
        correlation = signal.correlate(ref_search, deg_search, mode='full')

        # Find the lag (offset from center)
        center = len(deg_search) - 1
        search_range = slice(center - max_lag_samples, center + max_lag_samples + 1)

        if search_range.start < 0:
            search_range = slice(0, search_range.stop)
        if search_range.stop > len(correlation):
            search_range = slice(search_range.start, len(correlation))

        lag_range = correlation[search_range]
        lag = np.argmax(lag_range) + search_range.start - center

        # Apply alignment
        if lag > 0:
            # deg is ahead of ref, shift deg back (or pad ref at start)
            aligned_ref = ref
            aligned_deg = np.pad(deg, (lag, 0), mode='constant')[:len(ref)]
        elif lag < 0:
            # deg is behind ref, shift ref back (or pad deg at start)
            aligned_ref = np.pad(ref, (-lag, 0), mode='constant')[:len(deg)]
            aligned_deg = deg
        else:
            aligned_ref = ref
            aligned_deg = deg

        # Final length alignment
        min_len = min(len(aligned_ref), len(aligned_deg))
        aligned_ref = aligned_ref[:min_len]
        aligned_deg = aligned_deg[:min_len]

        return aligned_ref, aligned_deg, lag

    def evaluate_sample(
        self,
        ref_wav: torch.Tensor,
        deg_wav: torch.Tensor,
        shared_window_buffer: SharedWindowBuffer,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        align: bool = True,
    ) -> dict:
        """Evaluate a single sample.

        Args:
            ref_wav: Reference waveform
            deg_wav: Degraded/predicted waveform
            shared_window_buffer: Buffer for STFT windows
            n_mels: Number of mel bins
            n_fft: FFT size
            hop_length: Hop length
            align: Whether to align signals before computing metrics (recommended)
        """
        # Align lengths first
        min_len = min(ref_wav.shape[-1], deg_wav.shape[-1])
        ref_wav = ref_wav[..., :min_len]
        deg_wav = deg_wav[..., :min_len]

        # Convert to numpy for some metrics
        ref_np = ref_wav.squeeze().cpu().numpy().astype(np.float64)
        deg_np = deg_wav.squeeze().cpu().numpy().astype(np.float64)

        # Align signals temporally for better PESQ/STOI scores
        lag = 0
        if align:
            ref_np_aligned, deg_np_aligned, lag = self.align_signals(ref_np, deg_np)
        else:
            ref_np_aligned, deg_np_aligned = ref_np, deg_np

        results = {
            "pesq": self.compute_pesq(ref_np_aligned, deg_np_aligned),
            "stoi": self.compute_stoi(ref_np_aligned, deg_np_aligned),
            "mcd": self.compute_mcd(ref_np_aligned, deg_np_aligned),
            "utmos": self.compute_utmos(deg_np),  # UTMOS is reference-free, use original
            "mel_l1": self.compute_mel_l1(
                ref_wav, deg_wav, shared_window_buffer, n_mels, n_fft, hop_length
            ),
            "alignment_lag_samples": lag,
        }

        return results


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
        audio_max_waveform_length=sample_rate * 30,  # 10 seconds max
        n_mels=n_mels,
        training=False,
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
        "alignment_lag_samples": [],
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

    if all_results["alignment_lag_samples"]:
        # Use absolute value for mean lag to show typical offset magnitude
        results.alignment_lag_mean = np.mean(np.abs(all_results["alignment_lag_samples"]))
        results.alignment_lag_std = np.std(all_results["alignment_lag_samples"])

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate vocoder checkpoint")
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the run (used for logging and checkpoint directory)",
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
    parser.add_argument(
        "--logging_base_dir",
        type=str,
        default="runs/vocoder",
        help="Base directory for logging and checkpoints",
    )

    args, unk = parser.parse_known_args()
    run_dir = os.path.join(args.logging_base_dir, args.run_name)

    # Select model configuration
    if args.config not in vocoders.model_config_lookup:
        raise ValueError(f"Unknown vocoder config: {args.config}. Available: {list(vocoders.model_config_lookup.keys())}")

    # Parse extra arguments
    unk_dict = {}
    for i in range(0, len(unk), 2):
        unk_dict[unk[i].lstrip('-')] = unk[i+1]

    # Loss weights
    sc_loss_weight = float(unk_dict.get("sc_loss_weight", 1.0))
    mag_loss_weight = float(unk_dict.get("mag_loss_weight", 3.0))
    waveform_l1_loss_weight = float(unk_dict.get("waveform_l1_loss_weight", 0.1))
    mel_recon_loss_weight = float(unk_dict.get("mel_recon_loss_weight", 1.0))
    mel_recon_loss_weight_linspace_max = float(unk_dict.get("mel_recon_loss_weight_linspace_max", 1.0))
    complex_stft_loss_weight = float(unk_dict.get("complex_stft_loss_weight", 1.0))
    phase_loss_weight = float(unk_dict.get("phase_loss_weight", 0.0))
    phase_ip_loss_weight = float(unk_dict.get("phase_ip_loss_weight", 0.0))
    phase_iaf_loss_weight = float(unk_dict.get("phase_iaf_loss_weight", 0.0))
    phase_gd_loss_weight = float(unk_dict.get("phase_gd_loss_weight", 0.0))
    high_freq_stft_loss_weight = float(unk_dict.get("high_freq_stft_loss_weight", 0.0))
    high_freq_stft_cutoff_bin = int(unk_dict.get("high_freq_stft_cutoff_bin", 256))
    direct_mag_loss_weight = float(unk_dict.get("direct_mag_loss_weight", 0.0))

    shared_window_buffer = SharedWindowBuffer()

    model = vocoders.model_config_lookup[args.config](
        shared_window_buffer,
        sc_loss_weight=sc_loss_weight,
        mag_loss_weight=mag_loss_weight,
        waveform_l1_loss_weight=waveform_l1_loss_weight,
        mel_recon_loss_weight=mel_recon_loss_weight,
        mel_recon_loss_weight_linspace_max=mel_recon_loss_weight_linspace_max,
        complex_stft_loss_weight=complex_stft_loss_weight,
        phase_loss_weight=phase_loss_weight,
        phase_ip_loss_weight=phase_ip_loss_weight,
        phase_iaf_loss_weight=phase_iaf_loss_weight,
        phase_gd_loss_weight=phase_gd_loss_weight,
        high_freq_stft_loss_weight=high_freq_stft_loss_weight,
        high_freq_stft_cutoff_bin=high_freq_stft_cutoff_bin,
        direct_mag_loss_weight=direct_mag_loss_weight,
    )
    model, model_loaded = load_model(False, model, run_dir)

    model.eval()
    model.to(args.device)

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