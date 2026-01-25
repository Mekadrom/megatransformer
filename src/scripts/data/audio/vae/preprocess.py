
import argparse
import os
import torch
import torch.nn.functional as F
import traceback

import torchcrepe


from platform import processor
from typing import Dict, Optional

from datasets import load_dataset, Audio

from model.audio.sive.sive import SpeakerInvariantVoiceEncoder
from scripts.data.preprocessor import BatchProcessor
from utils import audio_utils
from utils.audio_utils import SharedWindowBuffer, extract_mels
from utils.model_loading_utils import load_model
from utils.speaker_encoder import SpeakerEncoderType, get_speaker_embedding_dim, get_speaker_encoder, get_speaker_encoder_input_type


def extract_f0_batch_gpu(
    waveforms: torch.Tensor,
    sample_rate: int = 16000,
    hop_length: int = 256,
    fmin: float = 50.0,
    fmax: float = 550.0,
    device: str = "cuda",
    f0_batch_size: int = 512,
) -> tuple:
    """
    Extract F0 (fundamental frequency) using torchcrepe on GPU (batched).

    Args:
        waveforms: [B, T] batch of audio waveforms as torch tensor
        sample_rate: Audio sample rate
        hop_length: Hop length for analysis (should match mel spectrogram)
        fmin: Minimum F0 frequency (Hz)
        fmax: Maximum F0 frequency (Hz)
        device: Device for computation

    Returns:
        log_f0: [B, T'] Log F0 contour (always computed, even for unvoiced)
        voiced: [B, T'] Soft voicing probability (0-1, from periodicity)
    """
    # torchcrepe expects [batch, samples]
    if waveforms.dim() == 1:
        waveforms = waveforms.unsqueeze(0)

    waveforms = waveforms.to(device)

    # Extract pitch and periodicity (voicing confidence)
    # Using viterbi decoder for smoother pitch contours
    # batch_size controls frames per CNN forward pass, not audio files
    # Higher = better GPU utilization but more memory
    pitch, periodicity = torchcrepe.predict(
        waveforms,
        sample_rate,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        model='full',
        decoder=torchcrepe.decode.viterbi,
        return_periodicity=True,
        batch_size=f0_batch_size,
        device=device,
        pad=True,
    )

    # pitch: [B, T'] in Hz
    # periodicity: [B, T'] confidence 0-1 (soft voicing label)

    # Convert pitch to log scale (always compute, even for low-confidence frames)
    # The model will learn to weight predictions by voicing confidence
    log_f0 = torch.log(pitch.clamp(min=1.0))

    # Return soft voicing (periodicity) instead of hard threshold
    voiced = periodicity

    return log_f0.cpu(), voiced.cpu()


class SIVEFeatureBatchProcessor(BatchProcessor):
    """Batched GPU processing for extracting SIVE features."""

    def __init__(
        self,
        sive_model,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        max_audio_seconds: int = 30,
        compute_speaker_embeddings: bool = True,
        speaker_encoder_type: SpeakerEncoderType = "ecapa_tdnn",
        normalize: bool = False,
        layers: Optional[list[int]] = None,
        extract_f0: bool = True,
        f0_fmin: float = 50.0,
        f0_fmax: float = 550.0,
        f0_batch_size: int = 512,
        device: str = "cuda",
    ):
        self.sive_model = sive_model
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_audio_seconds = max_audio_seconds
        self.device = device
        self.speaker_encoder_type = speaker_encoder_type
        self.normalize = normalize
        self.layers = layers
        self.extract_f0_enabled = extract_f0
        self.f0_fmin = f0_fmin
        self.f0_fmax = f0_fmax
        self.f0_batch_size = f0_batch_size

        self.audio_max_frames = (max_audio_seconds * sample_rate) // hop_length
        self.shared_window_buffer = SharedWindowBuffer()

        # Get SIVE encoder dimension
        # Note: we keep layers separate (not concatenated), so encoder_dim is always base_dim
        self.encoder_dim = sive_model.config.encoder_dim
        self.num_layers = len(layers) if layers is not None else 1

        # Speaker encoder (uses centralized cached singleton)
        self.speaker_encoder = None
        self.speaker_embedding_dim = get_speaker_embedding_dim(speaker_encoder_type)
        self.speaker_encoder_input_type = get_speaker_encoder_input_type(speaker_encoder_type)
        if compute_speaker_embeddings:
            print(f"Loading {speaker_encoder_type} speaker encoder (cached singleton)...")
            self.speaker_encoder = get_speaker_encoder(
                encoder_type=speaker_encoder_type,
                device=device,
            )
            print(f"  Speaker encoder loaded (embedding_dim={self.speaker_embedding_dim}, input={self.speaker_encoder_input_type})")

    @torch.no_grad()
    def process_batch(
        self,
        data: list[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Process batch of waveforms to SIVE features and optional speaker embeddings.

        Args:
            waveforms: list of [T] waveform tensors

        Returns:
            Dict with:
                - features: SIVE features (transposed)
                    - Single layer: [B, encoder_dim, T']
                    - Multi-layer: [B, num_layers, encoder_dim, T']
                - feature_lengths: [B] original lengths before padding
                - mel_specs: [B, n_mels, T] mel spectrograms
                - mel_lengths: [B] mel spectrogram lengths (before SIVE subsampling)
                - speaker_embeddings: [B, embedding_dim] speaker embeddings
                - f0: [B, T] log F0 contour (if extract_f0 enabled)
                - voiced: [B, T] voicing mask (if extract_f0 enabled)
        """
        # alias for clarity
        waveforms = data  # List of [T] tensors

        batch_size = len(waveforms)

        # Process mel spectrograms
        mel_specs = []
        mel_lengths = []
        waveform_lengths = []

        for waveform in waveforms:
            waveform_lengths.append(len(waveform))

            # Extract mel spectrogram
            mel = extract_mels(
                self.shared_window_buffer,
                waveform,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )

            mel_length = min(mel.shape[-1], self.audio_max_frames)
            mel_lengths.append(mel_length)

            # Pad or truncate to max frames
            if mel.shape[-1] < self.audio_max_frames:
                mel = F.pad(mel, (0, self.audio_max_frames - mel.shape[-1]), value=0)
            elif mel.shape[-1] > self.audio_max_frames:
                mel = mel[..., :self.audio_max_frames]

            mel_specs.append(mel)

        # Stack mel specs: [B, n_mels, T]
        mel_specs = torch.stack(mel_specs)
        mel_lengths_tensor = torch.tensor(mel_lengths, dtype=torch.long)

        # Extract F0 on GPU (batched) if enabled
        f0_batch = None
        voiced_batch = None
        if self.extract_f0_enabled:
            # Pad waveforms to same length for batched processing
            max_waveform_len = max(waveform_lengths)
            padded_waveforms = []
            for waveform in waveforms:
                if len(waveform) < max_waveform_len:
                    waveform = F.pad(waveform, (0, max_waveform_len - len(waveform)), value=0)
                padded_waveforms.append(waveform)
            waveform_batch = torch.stack(padded_waveforms)  # [B, T_audio]

            # GPU batched F0 extraction using torchcrepe
            log_f0_batch, voiced_batch = extract_f0_batch_gpu(
                waveform_batch,
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
                fmin=self.f0_fmin,
                fmax=self.f0_fmax,
                device=self.device,
                f0_batch_size=self.f0_batch_size,
            )
            # log_f0_batch: [B, T'], voiced_batch: [B, T']

            # Pad or truncate to max_audio_frames (same as mel)
            if log_f0_batch.shape[-1] < self.audio_max_frames:
                log_f0_batch = F.pad(log_f0_batch, (0, self.audio_max_frames - log_f0_batch.shape[-1]), value=0)
                voiced_batch = F.pad(voiced_batch, (0, self.audio_max_frames - voiced_batch.shape[-1]), value=0)
            elif log_f0_batch.shape[-1] > self.audio_max_frames:
                log_f0_batch = log_f0_batch[..., :self.audio_max_frames]
                voiced_batch = voiced_batch[..., :self.audio_max_frames]

            f0_batch = log_f0_batch
            voiced_batch = voiced_batch

        # Extract SIVE features
        mel_specs_gpu = mel_specs.to(self.device)

        if self.layers is not None and len(self.layers) > 1:
            # Multi-layer extraction: get each layer separately
            layer_features_list = []
            for layer_idx in self.layers:
                layer_features, feature_lengths = self.sive_model.extract_features(
                    mel_specs_gpu,
                    lengths=mel_lengths_tensor.to(self.device),
                    layer=layer_idx,  # Single layer at a time
                )
                # layer_features: [B, T', encoder_dim]
                layer_features_list.append(layer_features)

            # Stack to [B, num_layers, T', encoder_dim]
            features = torch.stack(layer_features_list, dim=1)
            # Transpose to [B, num_layers, encoder_dim, T']
            features = features.permute(0, 1, 3, 2).cpu()
        else:
            # Single layer extraction
            features, feature_lengths = self.sive_model.extract_features(
                mel_specs_gpu,
                lengths=mel_lengths_tensor.to(self.device),
                layer=self.layers if isinstance(self.layers, int) else self.layers[0],
            )
            # features: [B, T', encoder_dim]
            # Transpose to [B, encoder_dim, T'] for consistency with mel spec format
            features = features.permute(0, 2, 1).cpu()  # [B, encoder_dim, T']

        # Compute speaker embeddings
        if self.speaker_encoder is not None:
            if self.speaker_encoder_input_type == "waveform":
                # WavLM: needs waveforms padded to same length
                max_waveform_len = max(waveform_lengths)
                padded_waveforms = []
                for waveform in waveforms:
                    if len(waveform) < max_waveform_len:
                        waveform = F.pad(waveform, (0, max_waveform_len - len(waveform)), value=0)
                    padded_waveforms.append(waveform)

                waveform_batch = torch.stack(padded_waveforms).to(self.device)
                waveform_lengths_tensor = torch.tensor(waveform_lengths, dtype=torch.long)

                speaker_embeddings = self.speaker_encoder(
                    waveform=waveform_batch,
                    lengths=waveform_lengths_tensor,
                    sample_rate=self.sample_rate,
                ).cpu()
            else:
                # ECAPA-TDNN: needs mel specs
                speaker_embeddings = self.speaker_encoder(
                    mel_spec=mel_specs,
                    lengths=mel_lengths_tensor,
                ).cpu()
        else:
            speaker_embeddings = torch.zeros(batch_size, self.speaker_embedding_dim)

        result = {
            "features": features,  # [B, encoder_dim, T']
            "feature_lengths": feature_lengths.cpu(),  # [B]
            "mel_specs": mel_specs,  # [B, n_mels, T]
            "mel_lengths": mel_lengths_tensor,  # [B]
            "speaker_embeddings": speaker_embeddings,  # [B, embedding_dim]
        }

        # Add F0 data if extracted
        if self.extract_f0_enabled and f0_batch is not None:
            result["f0"] = f0_batch  # [B, T]
            result["voiced"] = voiced_batch  # [B, T]

        return result

class SIVEFeatureDatasetPreprocessor:
    """Preprocess dataset to extract and save SIVE features."""

    def __init__(self, args, dataset, output_dir, shard_fields, batch_accumulators, stats_accumulator, device):
        self.args = args
        self.dataset = dataset
        self.output_dir = output_dir
        self.shard_fields = shard_fields
        self.batch_accumulators = batch_accumulators
        self.stats_accumulator = stats_accumulator
        self.device = device

        model_config_overrides = {}
        if args.speaker_classifier_hidden_dim:
            model_config_overrides['speaker_classifier_hidden_dim'] = args.speaker_classifier_hidden_dim
        if args.speaker_pooling:
            model_config_overrides['speaker_pooling'] = args.speaker_pooling

        # Load SIVE model
        print(f"Loading SIVE model from {args.sive_checkpoint_path}...")
        self.sive_model = load_model(
            args.sive_checkpoint_path,
            args.sive_config,
            device=device,
            overrides=model_config_overrides,
        )

        print(f"  SIVE config: {args.sive_config}")
        print(f"  Encoder dimension: {self.sive_model.config.encoder_dim}")
        print(f"  Total stride (subsampling): {self.sive_model.conv_subsample.total_stride}x")

        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=args.sample_rate))
        print(f"  Total samples in dataset: {len(self.dataset):,}")

        # Feature extraction settings
        print(f"  Normalize features: {args.normalize}")
        if args.layers:
            print(f"  Multi-scale layers: {args.layers}")

        self.batch_processor = SIVEFeatureBatchProcessor(
            sive_model=self.sive_model,
            sample_rate=self.args.sample_rate,
            n_mels=self.args.n_mels,
            n_fft=self.args.n_fft,
            hop_length=self.args.hop_length,
            max_audio_seconds=self.args.max_audio_seconds,
            compute_speaker_embeddings=self.args.compute_speaker_embeddings,
            speaker_encoder_type=self.args.speaker_encoder_type,
            normalize=self.args.normalize,
            layers=self.args.layers,
            extract_f0=self.args.extract_f0,
            f0_fmin=self.args.f0_fmin,
            f0_fmax=self.args.f0_fmax,
            f0_batch_size=self.args.f0_batch_size,
            device=self.device,
        )

        print(f"  Output feature dimension: {self.batch_processor.encoder_dim}")
        print(f"  Extract F0: {self.args.extract_f0}")
        if self.args.extract_f0:
            print(f"  F0 range: {self.args.f0_fmin}-{self.args.f0_fmax} Hz")

        shard_fields.update({
            'shard_features': [],
            'shard_feature_lengths': [],
            'shard_mel_specs': [],
            'shard_mel_lengths': [],
            'shard_speaker_embeddings': [],
            'shard_f0': [],
            'shard_voiced': []
        })

    @classmethod
    def add_cli_args(cls, subparsers):
        sub_parser = subparsers.add_parser("audio-cvae", help="Preprocess audio dataset through SIVE for VAE training")
    
        # SIVE model
        sub_parser.add_argument("--sive_checkpoint_path", type=str, required=True,
                            help="Path to SIVE checkpoint directory")
        sub_parser.add_argument("--sive_config", type=str, default="small",
                            help="SIVE config name (tiny, small, medium, large)")

        # Audio settings
        sub_parser.add_argument("--sample_rate", type=int, default=16000)
        sub_parser.add_argument("--n_mels", type=int, default=80)
        sub_parser.add_argument("--n_fft", type=int, default=1024)
        sub_parser.add_argument("--hop_length", type=int, default=256)
        sub_parser.add_argument("--max_audio_seconds", type=int, default=30,
                            help="Maximum audio length in seconds")

        # Speaker embeddings
        sub_parser.add_argument("--compute_speaker_embeddings", action="store_true", default=True)
        sub_parser.add_argument("--no_speaker_embeddings", action="store_false",
                            dest="compute_speaker_embeddings")
        sub_parser.add_argument("--speaker_encoder_type", type=str, default="ecapa_tdnn",
                            choices=["ecapa_tdnn", "wavlm"],
                            help="Speaker encoder type")

        # Feature extraction options
        sub_parser.add_argument("--normalize", action="store_true", default=False,
                            help="Apply LayerNorm to features (default: False for VAE-friendly features)")
        sub_parser.add_argument("--layers", type=int, nargs="*", default=None,
                            help="Layer indices for multi-scale features (e.g., --layers -1 -3). "
                                "If not specified, uses final layer only.")

        # F0 extraction options
        sub_parser.add_argument("--extract_f0", action="store_true", default=True,
                            help="Extract F0 contour for conditioning (default: True)")
        sub_parser.add_argument("--f0_fmin", type=float, default=50.0,
                            help="Minimum F0 frequency in Hz (default: 50)")
        sub_parser.add_argument("--f0_fmax", type=float, default=550.0,
                            help="Maximum F0 frequency in Hz (default: 550, torchcrepe max)")
        sub_parser.add_argument("--f0_batch_size", type=int, default=128,
                            help="Batch size for torchcrepe F0 extraction (frames per CNN forward pass). "
                                "Higher = faster but more VRAM. Default 128, try 64 if OOM.")

        # Filtering
        sub_parser.add_argument("--min_audio_energy", type=float, default=0.05,
                            help="Minimum audio energy (skip silent samples)")
        sub_parser.add_argument("--min_audio_std", type=float, default=0.02,
                            help="Minimum audio std (skip near-silent samples)")
        sub_parser.add_argument("--remove_mains_hum", action="store_true", default=True,
                            help="Remove 50/60Hz mains hum from audio")
        
        sub_parser.add_argument("--speaker_classifier_hidden_dim", type=int, default=None,
                            help="Hidden dimension for SIVE speaker classifier (overrides encoder_dim)")
        sub_parser.add_argument("--speaker_pooling", type=str, default="attentive_statistics",
                            choices=["statistics", "attentive_statistics", "mean", "max"],
                            help="Pooling type for SIVE speaker classifier")
        
        return sub_parser

    def flush_shard(self):
        if not self.shard_fields['shard_features']:
            return

        # Find max feature length in this shard for padding
        max_feature_len = max(f.shape[-1] for f in self.shard_fields['shard_features'])

        # Pad features to same length
        padded_features = []
        for feat in self.shard_fields['shard_features']:
            if feat.shape[-1] < max_feature_len:
                feat = F.pad(feat, (0, max_feature_len - feat.shape[-1]), value=0)
            padded_features.append(feat)

        # Shape depends on multi-layer mode:
        # - Single layer: [N, encoder_dim, T']
        # - Multi-layer:  [N, num_layers, encoder_dim, T']
        shard_data = {
            "features": torch.cat(padded_features, dim=0),
            "feature_lengths": torch.cat(self.shard_fields['shard_feature_lengths'], dim=0),
            "mel_specs": torch.cat(self.shard_fields['shard_mel_specs'], dim=0),
            "mel_lengths": torch.cat(self.shard_fields['shard_mel_lengths'], dim=0),
            "speaker_embeddings": torch.cat(self.shard_fields['shard_speaker_embeddings'], dim=0),
            "num_samples": sum(f.shape[0] for f in self.shard_fields['shard_features']),
            "f0": torch.cat(self.shard_fields['shard_f0'], dim=0),
            "voiced": torch.cat(self.shard_fields['shard_voiced'], dim=0),
        }

        shard_path = os.path.join(self.output_dir, f"shard_{self.shard_fields['shard_idx']:06d}.pt")
        torch.save(shard_data, shard_path)

        print(f"  Saved shard {self.shard_fields['shard_idx']} ({shard_data['num_samples']} samples)")

        self.shard_fields['shard_features'] = []
        self.shard_fields['shard_feature_lengths'] = []
        self.shard_fields['shard_mel_specs'] = []
        self.shard_fields['shard_mel_lengths'] = []
        self.shard_fields['shard_speaker_embeddings'] = []
        self.shard_fields['shard_f0'] = []
        self.shard_fields['shard_voiced'] = []
        self.shard_fields['shard_idx'] += 1
    
    def process_and_accumulate(self):
        if not self.batch_accumulators['batch_waveforms']:
            return

        try:
            result = self.batch_processor.process_batch(self.batch_accumulators['batch_waveforms'])

            # Add to shard
            self.shard_fields['shard_features'].append(result["features"])
            self.shard_fields['shard_feature_lengths'].append(result["feature_lengths"])
            self.shard_fields['shard_mel_specs'].append(result["mel_specs"])
            self.shard_fields['shard_mel_lengths'].append(result["mel_lengths"])
            self.shard_fields['shard_speaker_embeddings'].append(result["speaker_embeddings"])
            self.shard_fields['shard_f0'].append(result["f0"])
            self.shard_fields['shard_voiced'].append(result["voiced"])
            self.stats_accumulator["saved"] += len(self.batch_accumulators['batch_waveforms'])

            # Flush shard if full
            current_size = sum(f.shape[0] for f in self.shard_fields['shard_features'])
            if current_size >= self.args.shard_size:
                self.flush_shard()
        except Exception as e:
            print(f"Batch processing error: {e}")
            traceback.print_exc()
            self.stats_accumulator["skipped"]["error"] += len(self.batch_accumulators['batch_waveforms'])

        self.batch_accumulators['batch_waveforms'] = []
    
    def preprocess_example(self, example):
        # Extract waveform
        audio = example["audio"]
        waveform = torch.tensor(audio["array"], dtype=torch.float32)

        # Skip silent/near-silent audio
        if waveform.abs().max() < self.args.min_audio_energy or waveform.std() < self.args.min_audio_std:
            self.stats_accumulator["skipped"]["silent"] += 1
            # won't accumulate into batch accumulators, still pbars
            return

        # Remove mains hum if enabled
        if self.args.remove_mains_hum:
            waveform = audio_utils.remove_mains_hum(waveform.unsqueeze(0), self.args.sample_rate).squeeze(0)

        # Truncate if too long
        max_samples = self.args.max_audio_seconds * self.args.sample_rate
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]

        # Add to batch
        self.batch_accumulators['batch_waveforms'].append(waveform)

        # Process batch when full
        if len(self.batch_accumulators['batch_waveforms']) >= self.args.gpu_batch_size:
            self.process_and_accumulate()

    def parse_config(self) -> dict:
        return {
            "sive_checkpoint": self.args.sive_checkpoint,
            "sive_config": self.args.sive_config,
            "encoder_dim": self.sive_model.config.encoder_dim,
            "total_stride": self.sive_model.conv_subsample.total_stride,
            "dataset_name": self.args.dataset_name,
            "dataset_config": self.args.dataset_config,
            "split": self.args.split,
            "sample_rate": self.args.sample_rate,
            "n_mels": self.args.n_mels,
            "n_fft": self.args.n_fft,
            "hop_length": self.args.hop_length,
            "max_audio_seconds": self.args.max_audio_seconds,
            "compute_speaker_embeddings": self.args.compute_speaker_embeddings,
            "speaker_encoder_type": self.args.speaker_encoder_type,
            "speaker_embedding_dim": self.processor.speaker_embedding_dim,
            "remove_mains_hum": self.args.remove_mains_hum,
            "shard_size": self.args.shard_size,
            # Feature extraction settings
            "normalize": self.args.normalize,
            "layers": self.args.layers,  # None for single layer (default), list for multi-layer
            "num_layers": self.processor.num_layers,  # 1 for single layer, >1 for multi-layer
            # F0 extraction settings
            "extract_f0": self.args.extract_f0,
            "f0_fmin": self.args.f0_fmin,
            "f0_fmax": self.args.f0_fmax,
            "stats": self.stats_accumulator,
        }
