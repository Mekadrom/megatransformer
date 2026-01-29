
import argparse
import os
import torch
import torch.nn.functional as F
import traceback

from scripts.data.text_batch_processor import TextConditionsBatchProcessor
import torchcrepe


from platform import processor
from typing import Any, Dict, Optional

from datasets import load_dataset, Audio

from model.audio.sive.sive import SpeakerInvariantVoiceEncoder
from scripts.data.preprocessor import BatchProcessor, Preprocessor
from utils import audio_utils
from utils.audio_utils import SharedWindowBuffer, extract_mels
from utils.model_loading_utils import load_model
from utils.speaker_encoder import SpeakerEncoderType, get_speaker_embedding_dim, get_speaker_encoder, get_speaker_encoder_input_type
from utils.text_encoder import TextEncoderType


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
        shared_window_buffer: SharedWindowBuffer,
        normalize: bool = False,
        layers: Optional[list[int]] = None,
        device: str = "cuda",
    ):
        self.sive_model = sive_model
        self.shared_window_buffer = shared_window_buffer
        self.device = device
        self.normalize = normalize
        self.layers = layers


        # Get SIVE encoder dimension
        # Note: we keep layers separate (not concatenated), so encoder_dim is always base_dim
        self.encoder_dim = sive_model.config.encoder_dim
        self.num_layers = len(layers) if layers is not None else 1

    @torch.no_grad()
    def process_batch(
        self,
        mel_specs: torch.Tensor,
        mel_spec_lengths: torch.Tensor,
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
        # Extract SIVE features
        if self.layers is not None and len(self.layers) > 1:
            # Multi-layer extraction: get each layer separately
            layer_features_list = []
            for layer_idx in self.layers:
                layer_features, feature_lengths = self.sive_model.extract_features(
                    mel_specs,
                    lengths=mel_spec_lengths.to(self.device),
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
                mel_specs,
                lengths=mel_spec_lengths.to(self.device),
                layer=self.layers if isinstance(self.layers, int) else self.layers[0],
            )
            # features: [B, T', encoder_dim]
            # Transpose to [B, encoder_dim, T'] for consistency with mel spec format
            features = features.permute(0, 2, 1).cpu()  # [B, encoder_dim, T']

        return {
            "features": features,  # [B, encoder_dim, T']
            "feature_lengths": feature_lengths.cpu(),  # [B]
        }


class SpeakerEmbeddingBatchProcessor(BatchProcessor):
    """Batched GPU processing for extracting speaker embeddings."""

    def __init__(
        self,
        shared_window_buffer: SharedWindowBuffer,
        sample_rate: int = 16000,
        hop_length: int = 256,
        audio_max_seconds: int = 30,
        speaker_encoder_type: SpeakerEncoderType = "ecapa_tdnn",
        device: str = "cuda",
    ):
        self.shared_window_buffer = shared_window_buffer
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.device = device

        self.audio_max_frames = (audio_max_seconds * sample_rate) // hop_length

        # Speaker encoder (uses centralized cached singleton)
        self.speaker_encoder = None
        self.speaker_embedding_dim = get_speaker_embedding_dim(speaker_encoder_type)
        self.speaker_encoder_input_type = get_speaker_encoder_input_type(speaker_encoder_type)
        print(f"Loading {speaker_encoder_type} speaker encoder (cached singleton)...")
        self.speaker_encoder = get_speaker_encoder(
            encoder_type=speaker_encoder_type,
            device=device,
        )
        print(f"  Speaker encoder loaded (embedding_dim={self.speaker_embedding_dim}, input={self.speaker_encoder_input_type})")

    @torch.no_grad()
    def process_batch(
        self,
        waveforms: list[torch.Tensor],
        waveform_lengths: torch.Tensor,
        mel_specs: torch.Tensor,
        mel_lengths_tensor: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Process batch of waveforms to extract speaker embeddings.

        Args:
            waveforms: list of [T] waveform tensors

        Returns:
            Dict with:
                - speaker_embeddings: [B, embedding_dim] speaker embeddings
        """
        # Compute speaker embeddings
        speaker_embeddings = None
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

        return {
            "speaker_embeddings": speaker_embeddings,  # [B, embedding_dim]
        }

class F0VUVBatchProcessor(BatchProcessor):
    """Batched GPU processing for extracting F0 and VUV labels."""

    def __init__(
        self,
        shared_window_buffer: SharedWindowBuffer,
        sample_rate: int = 16000,
        hop_length: int = 256,
        audio_max_seconds: int = 30,
        f0_fmin: float = 50.0,
        f0_fmax: float = 550.0,
        f0_batch_size: int = 512,
        device: str = "cuda",
    ):
        self.shared_window_buffer = shared_window_buffer
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f0_fmin = f0_fmin
        self.f0_fmax = f0_fmax
        self.f0_batch_size = f0_batch_size
        self.device = device

        self.audio_max_frames = (audio_max_seconds * sample_rate) // hop_length

    @torch.no_grad()
    def process_batch(
        self,
        waveforms: list[torch.Tensor],
        waveform_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Process batch of waveforms to extract speaker embeddings, F0 and VUV labels.

        Args:
            waveforms: list of [T] waveform tensors

        Returns:
            Dict with:
                - f0: [B, T] log F0 contour (if extract_f0 enabled)
                - vuv: [B, T] voicing mask (if extract_f0 enabled)
        """
        # Extract F0 on GPU (batched) if enabled
        # Pad waveforms to same length for batched processing
        max_waveform_len = max(waveform_lengths)
        padded_waveforms = []
        for waveform in waveforms:
            if len(waveform) < max_waveform_len:
                waveform = F.pad(waveform, (0, max_waveform_len - len(waveform)), value=0)
            padded_waveforms.append(waveform)
        waveform_batch = torch.stack(padded_waveforms)  # [B, T_audio]

        # GPU batched F0 extraction using torchcrepe
        log_f0_batch, vuv_batch = extract_f0_batch_gpu(
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
            vuv_batch = F.pad(vuv_batch, (0, self.audio_max_frames - vuv_batch.shape[-1]), value=0)
        elif log_f0_batch.shape[-1] > self.audio_max_frames:
            log_f0_batch = log_f0_batch[..., :self.audio_max_frames]
            vuv_batch = vuv_batch[..., :self.audio_max_frames]

        return {
            "f0": log_f0_batch,  # [B, T]
            "vuv": vuv_batch,  # [B, T]
        }


class AudioDatasetPreprocessor(Preprocessor):
    """Preprocess dataset to extract and save SIVE features."""

    def __init__(self, args, dataset, output_dir, shard_fields, batch_accumulators, stats_accumulator, device):
        self.args = args
        self.dataset = dataset
        self.output_dir = output_dir
        self.shard_fields = shard_fields
        self.batch_accumulators = batch_accumulators
        self.stats_accumulator = stats_accumulator
        self.device = device

        assert args.save_waveforms or args.save_mel_specs or args.sive_checkpoint_path is not None, \
            "At least one of --save_waveforms, --save_mel_specs, or --sive_checkpoint_path must be specified."

        self.audio_max_frames = (args.audio_max_seconds * args.sample_rate) // args.hop_length

        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=args.sample_rate))
        print(f"  Total samples in dataset: {len(self.dataset):,}")

        self.num_unique_speaker_ids = 0
        if args.compute_speaker_embeddings:
            self._convert_speaker_ids()

        self.shared_window_buffer = SharedWindowBuffer()

        self.sive_batch_processor = None
        if args.sive_checkpoint_path is not None:
            model_config_overrides = {}
            if args.speaker_classifier_hidden_dim:
                model_config_overrides['speaker_classifier_hidden_dim'] = args.speaker_classifier_hidden_dim
            if args.speaker_pooling:
                model_config_overrides['speaker_pooling'] = args.speaker_pooling

            # Load SIVE model
            print(f"Loading SIVE model from {args.sive_checkpoint_path}...")
            sive_model = load_model(
                SpeakerInvariantVoiceEncoder,
                args.sive_config,
                args.sive_checkpoint_path,
                device=device,
                overrides=model_config_overrides,
            )

            print(f"  SIVE config: {args.sive_config}")
            print(f"  Encoder dimension: {sive_model.config.encoder_dim}")
            print(f"  Total stride (subsampling): {sive_model.conv_subsample.total_stride}x")

            # Feature extraction settings
            print(f"  Normalize features: {args.normalize}")
            if args.layers:
                print(f"  Multi-scale layers: {args.layers}")

            self.sive_batch_processor = SIVEFeatureBatchProcessor(
                sive_model=sive_model,
                shared_window_buffer=self.shared_window_buffer,
                normalize=self.args.normalize,
                layers=self.args.layers,
                device=self.device,
            )
            print(f"  Output feature dimension: {self.sive_batch_processor.encoder_dim}")
            shard_fields.update({
                'shard_features': [],
                'shard_feature_lengths': [],
            })

        self.speaker_feature_batch_processor = None
        if args.compute_speaker_embeddings:
            self.speaker_feature_batch_processor = SpeakerEmbeddingBatchProcessor(
                shared_window_buffer=self.shared_window_buffer,
                sample_rate=self.args.sample_rate,
                hop_length=self.args.hop_length,
                audio_max_seconds=self.args.audio_max_seconds,
                speaker_encoder_type=self.args.speaker_encoder_type,
                device=self.device,
            )

        self.f0_vuv_batch_processor = None
        if args.extract_f0:
            self.f0_vuv_batch_processor = F0VUVBatchProcessor(
                shared_window_buffer=self.shared_window_buffer,
                sample_rate=self.args.sample_rate,
                hop_length=self.args.hop_length,
                audio_max_seconds=self.args.audio_max_seconds,
                f0_fmin=self.args.f0_fmin,
                f0_fmax=self.args.f0_fmax,
                f0_batch_size=self.args.f0_batch_size,
                device=self.device,
            )

        self.conditions_processor = None
        if args.extract_conditions:
            self.conditions_processor = TextConditionsBatchProcessor(
                text_embedding_model=args.text_condition_embedding_model,
                device=self.device,
            )

        if args.compute_speaker_embeddings:
            shard_fields.update({
                'shard_speaker_embeddings': [],
                'shard_speaker_ids': [],
            })

        print(f"  Extract F0: {self.args.extract_f0}")
        if args.extract_f0:
            print(f"  F0 range: {self.args.f0_fmin}-{self.args.f0_fmax} Hz")
            shard_fields.update({
                'shard_f0': [],
                'shard_vuv': []
            })

        if args.save_mel_specs:
            shard_fields.update({
                'shard_mel_specs': [],
                'shard_mel_lengths': [],
            })

        if args.extract_conditions:
            shard_fields.update({
                'shard_conditions': [],
                'shard_conditions_lengths': [],
            })

        if args.save_text:
            shard_fields.update({
                'shard_text': [],
            })

        shard_fields.update({
            'shard_waveforms': [],
            'shard_waveform_lengths': [],
        })

    def _convert_speaker_ids(self):
        # iterates dataset to collects unique speaker ids by speaker id column
        # then iterates over again to replace with sorted index within unique ids
        if not self.args.compute_speaker_embeddings:
            return
        
        speaker_id_column = self.args.speaker_id_column
        print(f"Converting speaker IDs from column '{speaker_id_column}' to integer indices...")

        unique_speaker_ids = set()
        for example in self.dataset:
            speaker_id = example.get(speaker_id_column, "unknown")
            unique_speaker_ids.add(speaker_id)
        unique_speaker_ids = sorted(list(unique_speaker_ids))

        speaker_id_to_index = {speaker_id: idx for idx, speaker_id in enumerate(unique_speaker_ids)}
        print(f"  Found {len(unique_speaker_ids)} unique speaker IDs.")

        def map_speaker_id(example):
            speaker_id = example.get(speaker_id_column, "unknown")
            example[speaker_id_column] = speaker_id_to_index.get(speaker_id, -1)
            return example
        self.dataset = self.dataset.map(map_speaker_id)

        print(f"  Converted speaker IDs to integer indices.")

        self.num_unique_speaker_ids = len(unique_speaker_ids)

    @classmethod
    def add_cli_args(cls, subparsers):
        sub_parser = subparsers.add_parser("audio", help="Preprocess audio dataset through SIVE for VAE training")
    
        # SIVE model
        sub_parser.add_argument("--sive_checkpoint_path", type=str,
                            help="Path to SIVE checkpoint directory. If not specified, features are not saved.")
        sub_parser.add_argument("--sive_config", type=str, default="small",
                            help="SIVE config name (tiny, small, medium, large)")
        
        # Feature extraction options
        sub_parser.add_argument("--normalize", action="store_true", default=False,
                            help="Apply LayerNorm to features (default: False for VAE-friendly features)")
        sub_parser.add_argument("--layers", type=int, nargs="*", default=None,
                            help="Layer indices for multi-scale features (e.g., --layers -1 -3). "
                                "If not specified, uses final layer only.")

        # Audio settings
        sub_parser.add_argument("--sample_rate", type=int, default=16000)
        sub_parser.add_argument("--n_mels", type=int, default=80)
        sub_parser.add_argument("--n_fft", type=int, default=1024)
        sub_parser.add_argument("--hop_length", type=int, default=256)
        sub_parser.add_argument("--audio_max_seconds", type=int, default=20,
                            help="Maximum audio length in seconds")

        # Speaker embeddings
        sub_parser.add_argument("--compute_speaker_embeddings", action="store_true")
        sub_parser.add_argument("--speaker_encoder_type", type=str, default="ecapa_tdnn",
                            choices=["ecapa_tdnn", "wavlm"],
                            help="Speaker encoder type")

        # F0 extraction options
        sub_parser.add_argument("--extract_f0", action="store_true",
                            help="Extract F0 contour for conditioning.")
        sub_parser.add_argument("--f0_fmin", type=float, default=50.0,
                            help="Minimum F0 frequency in Hz (default: 50)")
        sub_parser.add_argument("--f0_fmax", type=float, default=550.0,
                            help="Maximum F0 frequency in Hz (default: 550, torchcrepe max)")
        sub_parser.add_argument("--f0_batch_size", type=int, default=64,
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
        sub_parser.add_argument("--speaker_id_column", type=str, default="speaker_id",
                            help="Name of the speaker ID column in the dataset")
        
        sub_parser.add_argument("--save_waveforms", action="store_true", default=False,
                                help="Enable saving waveforms.")
        sub_parser.add_argument("--save_mel_specs", action="store_true", default=False,
                                help="Enable saving mel spectrograms.")
        
        sub_parser.add_argument("--audio_column", type=str, default="audio",
                            help="Name of the audio column in the dataset")
        
        # conditions
        sub_parser.add_argument("--extract_conditions", action="store_true",
                            help="Whether to extract conditions from the dataset")
        sub_parser.add_argument("--text_conditions_column", type=str, default="text",
                            help="Name of the audio column in the dataset")
        sub_parser.add_argument("--text_condition_embedding_model", type=str, default="t5_small",
                            help="Model to use for condition embeddings")
        sub_parser.add_argument("--save_text", action="store_true", default=False,
                                help="Enable saving raw text strings.")

        return sub_parser

    def flush_shard(self):
        # Find max feature length in this shard for padding
        shard_data = {}
        
        num_samples = 0

        if self.args.sive_checkpoint_path is not None:
            # Pad features to same length
            # Shape depends on multi-layer mode:
            # - Single layer: [N, encoder_dim, T']
            # - Multi-layer:  [N, num_layers, encoder_dim, T']
            max_feature_len = max(f.shape[-1] for f in self.shard_fields['shard_features'])
            padded_features = []
            for feat in self.shard_fields['shard_features']:
                if feat.shape[-1] < max_feature_len:
                    feat = F.pad(feat, (0, max_feature_len - feat.shape[-1]), value=0)
                padded_features.append(feat)
            
            shard_data["features"] = torch.cat(padded_features, dim=0)
            shard_data["feature_lengths"] = torch.cat(self.shard_fields['shard_feature_lengths'], dim=0)
            
            num_samples = shard_data["features"].shape[0]
            
            self.shard_fields['shard_features'] = []
            self.shard_fields['shard_feature_lengths'] = []

        if self.args.save_waveforms:
            # Pad waveforms to same length
            max_waveform_len = max(f.shape[-1] for f in self.shard_fields['shard_waveforms'])
            padded_waveforms = []
            for waveform in self.shard_fields['shard_waveforms']:
                if waveform.shape[-1] < max_waveform_len:
                    waveform = F.pad(waveform, (0, max_waveform_len - waveform.shape[-1]), value=0)
                padded_waveforms.append(waveform)
            
            shard_data["waveforms"] = torch.stack(padded_waveforms, dim=0)
            shard_data["waveform_lengths"] = torch.cat(self.shard_fields['shard_waveform_lengths'], dim=0)
            
            num_samples = shard_data["waveforms"].shape[0]

        # always reset because these drive several extractions
        self.shard_fields['shard_waveforms'] = []
        self.shard_fields['shard_waveform_lengths'] = []

        if self.args.save_mel_specs:
            # Pad mel specs to same length
            max_mel_len = max(f.shape[-1] for f in self.shard_fields['shard_mel_specs'])
            padded_mel_specs = []
            for mel_spec in self.shard_fields['shard_mel_specs']:
                if mel_spec.shape[-1] < max_mel_len:
                    mel_spec = F.pad(mel_spec, (0, max_mel_len - mel_spec.shape[-1]), value=0)
                padded_mel_specs.append(mel_spec)

            shard_data["mel_specs"] = torch.cat(padded_mel_specs, dim=0)
            shard_data["mel_lengths"] = torch.cat(self.shard_fields['shard_mel_lengths'], dim=0)
            
            num_samples = shard_data["mel_specs"].shape[0]
            
            self.shard_fields['shard_mel_specs'] = []
            self.shard_fields['shard_mel_lengths'] = []

        if self.args.compute_speaker_embeddings:
            shard_data["speaker_embeddings"] = torch.cat(self.shard_fields['shard_speaker_embeddings'], dim=0)
            shard_data["speaker_ids"] = torch.tensor(self.shard_fields['shard_speaker_ids'], dtype=torch.long)
            self.shard_fields['shard_speaker_embeddings'] = []
            self.shard_fields['shard_speaker_ids'] = []

        if self.args.extract_f0:
            shard_data["f0"] = torch.cat(self.shard_fields['shard_f0'], dim=0)
            shard_data["vuv"] = torch.cat(self.shard_fields['shard_vuv'], dim=0)
            self.shard_fields['shard_f0'] = []
            self.shard_fields['shard_vuv'] = []

        if self.args.extract_conditions:
            # pad conditions to same length
            max_conditions_len = max(f.shape[0] for f in self.shard_fields['shard_conditions'])
            padded_conditions = []
            for cond in self.shard_fields['shard_conditions']:
                if cond.shape[0] < max_conditions_len:
                    cond = F.pad(cond, (0, 0, 0, max_conditions_len - cond.shape[0]), value=0)
                padded_conditions.append(cond)
            shard_data["conditions"] = torch.stack(padded_conditions, dim=0)
            shard_data["conditions_lengths"] = torch.cat(self.shard_fields['shard_conditions_lengths'], dim=0)
            self.shard_fields['shard_conditions'] = []
            self.shard_fields['shard_conditions_lengths'] = []

        if self.args.save_text:
            # just store raw strings in shard (usually for visualization)
            shard_data["text"] = self.shard_fields['shard_text']
            self.shard_fields['shard_text'] = []

        shard_data["num_samples"] = num_samples

        shard_path = os.path.join(self.output_dir, f"shard_{self.shard_fields['shard_idx']:06d}.pt")
        torch.save(shard_data, shard_path)

        print(f"  Saved shard {self.shard_fields['shard_idx']} ({shard_data['num_samples']} samples)")

        self.shard_fields['shard_idx'] += self.args.total_gpus
    
    def encode_to_mels(self, waveforms: list[torch.Tensor]) -> tuple:
        mel_specs = []
        mel_lengths = []
        waveform_lengths = []

        for waveform in waveforms:
            waveform_lengths.append(len(waveform))

            # Extract mel spectrogram
            mel = extract_mels(
                self.shared_window_buffer,
                waveform,
                sr=self.args.sample_rate,
                n_mels=self.args.n_mels,
                n_fft=self.args.n_fft,
                hop_length=self.args.hop_length,
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
        waveform_lengths_tensor = torch.tensor(waveform_lengths, dtype=torch.long)
        return mel_specs, mel_lengths_tensor, waveform_lengths_tensor

    def process_and_accumulate(self):
        try:
            waveforms = self.batch_accumulators['batch_waveforms']

            mel_specs, mel_spec_lengths, waveform_lengths = self.encode_to_mels(waveforms)

            if self.sive_batch_processor is not None:
                features_result = self.sive_batch_processor.process_batch(mel_specs, mel_spec_lengths)
            
            if self.args.compute_speaker_embeddings:
                speaker_features_result = self.speaker_feature_batch_processor.process_batch(waveforms, waveform_lengths, mel_specs, mel_spec_lengths)

            if self.args.extract_f0:
                f0_vuv_result = self.f0_vuv_batch_processor.process_batch(waveforms, waveform_lengths)

            if self.args.extract_conditions:
                conditions = self.batch_accumulators['batch_conditions']
                conditions_result = self.conditions_processor.process_batch(conditions)

            # Add to shard
            # count_key tracks which field to use for shard size checking (doesn't matter which one, just needs to be one of them)
            count_key = ""
            if self.args.sive_checkpoint_path is not None:
                self.shard_fields['shard_features'].append(features_result["features"])
                self.shard_fields['shard_feature_lengths'].append(features_result["feature_lengths"])
                count_key = "shard_features"
            
            if self.args.save_waveforms:
                self.shard_fields['shard_waveforms'].extend(waveforms)
                self.shard_fields['shard_waveform_lengths'].append(waveform_lengths)
                count_key = "shard_waveforms"
            
            if self.args.save_mel_specs:
                self.shard_fields['shard_mel_specs'].append(mel_specs)
                self.shard_fields['shard_mel_lengths'].append(mel_spec_lengths)
                count_key = "shard_mel_specs"
            
            if self.args.compute_speaker_embeddings:
                self.shard_fields['shard_speaker_embeddings'].append(speaker_features_result["speaker_embeddings"])
                self.shard_fields['shard_speaker_ids'].extend(self.batch_accumulators['batch_speaker_ids'])

            if self.args.extract_f0:
                self.shard_fields['shard_f0'].append(f0_vuv_result["f0"])
                self.shard_fields['shard_vuv'].append(f0_vuv_result["vuv"])

            if self.args.extract_conditions:
                self.shard_fields['shard_conditions'].extend(conditions_result["conditions"])
                self.shard_fields['shard_conditions_lengths'].append(conditions_result["conditions_lengths"])

            if self.args.save_text:
                self.shard_fields['shard_text'].extend(conditions)

            self.stats_accumulator["saved"] += len(self.batch_accumulators['batch_waveforms'])

            # Flush shard if full
            current_size = sum(f.shape[0] for f in self.shard_fields[count_key])
            if current_size >= self.args.shard_size:
                self.flush_shard()
        except Exception as e:
            print(f"Batch processing error: {e}")
            traceback.print_exc()
            self.stats_accumulator["skipped"]["error"] += len(self.batch_accumulators['batch_waveforms'])

        # reset batch accumulators
        self.batch_accumulators['batch_waveforms'] = []
        if self.args.compute_speaker_embeddings:
            self.batch_accumulators['batch_speaker_ids'] = []
        if self.args.extract_conditions:
            self.batch_accumulators['batch_conditions'] = []
    
    def preprocess_example(self, example) -> bool:
        # Extract fields
        audio = example[self.args.audio_column]

        waveform = torch.tensor(audio["array"], dtype=torch.float32)

        # Skip silent/near-silent audio
        if waveform.abs().max() < self.args.min_audio_energy or waveform.std() < self.args.min_audio_std:
            self.stats_accumulator["skipped"]["silent"] += 1
            # won't accumulate into batch accumulators, still pbars
            return False

        # Remove mains hum if enabled
        if self.args.remove_mains_hum:
            waveform = audio_utils.remove_mains_hum(waveform.unsqueeze(0), self.args.sample_rate).squeeze(0)

        # Truncate if too long
        max_samples = self.args.audio_max_seconds * self.args.sample_rate
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]

        if 'batch_waveforms' not in self.batch_accumulators:
            self.batch_accumulators['batch_waveforms'] = []
        if 'batch_speaker_ids' not in self.batch_accumulators and self.args.compute_speaker_embeddings:
            self.batch_accumulators['batch_speaker_ids'] = []
        if 'batch_conditions' not in self.batch_accumulators and self.args.extract_conditions:
            self.batch_accumulators['batch_conditions'] = []

        # Add to batch
        self.batch_accumulators['batch_waveforms'].append(waveform)
        if self.args.compute_speaker_embeddings:
            speaker_id = example.get(self.args.speaker_id_column, "unknown")
            self.batch_accumulators['batch_speaker_ids'].append(speaker_id)
        if self.args.extract_conditions:
            conditions = example[self.args.text_conditions_column]
            self.batch_accumulators['batch_conditions'].append(conditions)

        # Process batch when full
        if len(self.batch_accumulators['batch_waveforms']) >= self.args.gpu_batch_size:
            self.process_and_accumulate()

        return True

    def parse_config(self) -> dict:
        return {
            "sive_checkpoint": self.args.sive_checkpoint_path,
            "sive_config": self.args.sive_config,
            "encoder_dim": self.sive_batch_processor.sive_model.config.encoder_dim if self.sive_batch_processor is not None else 0,
            "total_stride": self.sive_batch_processor.sive_model.conv_subsample.total_stride if self.sive_batch_processor is not None else 0,
            "dataset_name": self.args.dataset_name,
            "dataset_config": self.args.dataset_config,
            "split": self.args.split,
            "sample_rate": self.args.sample_rate,
            "n_mels": self.args.n_mels,
            "n_fft": self.args.n_fft,
            "hop_length": self.args.hop_length,
            "audio_max_seconds": self.args.audio_max_seconds,
            "compute_speaker_embeddings": self.args.compute_speaker_embeddings,
            "speaker_encoder_type": self.args.speaker_encoder_type,
            "speaker_embedding_dim": self.speaker_feature_batch_processor.speaker_embedding_dim if self.speaker_feature_batch_processor is not None else 0,
            "remove_mains_hum": self.args.remove_mains_hum,
            "shard_size": self.args.shard_size,
            # Feature extraction settings
            "normalize": self.args.normalize,
            "layers": self.args.layers,  # None for single layer (default), list for multi-layer
            "num_layers": self.sive_batch_processor.num_layers if self.sive_batch_processor is not None else 0,  # 1 for single layer, >1 for multi-layer
            # F0 extraction settings
            "extract_f0": self.args.extract_f0,
            "f0_fmin": self.args.f0_fmin,
            "f0_fmax": self.args.f0_fmax,
            "stats": self.stats_accumulator,
            "save_mel_specs": self.args.save_mel_specs,
            "save_waveforms": self.args.save_waveforms,
            "save_text": self.args.save_text,
            "num_unique_speakers": self.num_unique_speaker_ids,
        }
