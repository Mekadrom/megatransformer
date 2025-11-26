import os

os.environ["DEEPSPEED_UNIT_TEST"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ['NCCL_TIMEOUT'] = '1200000'

from datasets import load_dataset, Audio
from torch import nn
from torch.amp import autocast
from torch.utils.data import IterableDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments, Trainer, TrainerCallback, T5EncoderModel, T5Tokenizer
from transformers.integrations import TensorBoardCallback
from typing import Iterator, Any, List, Dict, Optional

from dataset_loading import audio_loading
from model.megatransformer_audio_decoder import AudioVocoder, MultiResolutionSTFTLoss

import librosa
import librosa.display
import matplotlib.pyplot as plt
import megatransformer_utils
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


# ============================================================================
# Discriminator using periodic convolutions
# ============================================================================

class PeriodDiscriminator(nn.Module):
    def __init__(self, period):
        self.period = period
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(1024, 1, (3, 1), padding=(1, 0)),
        ])
    
    def forward(self, x):
        features = []
        # Reshape: [B, T] -> [B, 1, T//period, period]
        b, t = x.shape
        pad = (self.period - (t % self.period)) % self.period
        x = F.pad(x, (0, pad))
        x = x.view(b, 1, -1, self.period)
        
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        return x, features

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in [2, 3, 5, 7, 11]
        ])
    
    def forward(self, x):
        outputs, all_features = [], []
        for d in self.discriminators:
            out, feats = d(x)
            outputs.append(out)
            all_features.append(feats)
        return outputs, all_features

# ============================================================================
# Vocoder Model Wrapper with T5 Text Conditioning
# ============================================================================

class VocoderWithT5Conditioning(nn.Module):
    """
    Wrapper model that combines AudioVocoder with frozen T5 text embeddings for conditioning.
    """
    def __init__(self,
                 config: megatransformer_utils.MegaTransformerConfig,
                 sc_loss_weight: float = 1.0,
                 mag_loss_weight: float = 3.0,
                 waveform_l1_loss_weight: float = 0.1,
                 mel_recon_loss_weight: float = 1.0,
                 t5_model_name: str = "google/t5-v1_1-base"):
        super().__init__()
        self.config = config
        self.sc_loss_weight = sc_loss_weight
        self.mag_loss_weight = mag_loss_weight
        self.waveform_l1_loss_weight = waveform_l1_loss_weight
        self.mel_recon_loss_weight = mel_recon_loss_weight

        # Frozen T5 encoder for text conditioning
        self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        self.t5_encoder = T5EncoderModel.from_pretrained(t5_model_name)

        # Freeze T5 parameters
        for param in self.t5_encoder.parameters():
            param.requires_grad = False

        # Get T5 hidden size
        self.t5_hidden_size = self.t5_encoder.config.d_model

        # Project T5 embeddings to vocoder conditioning size
        self.condition_proj = nn.Linear(self.t5_hidden_size, config.hidden_size)

        # AudioVocoder from megatransformer_audio_decoder
        self.vocoder = AudioVocoder(
            hidden_channels=config.audio_vocoder_hidden_channels,
            in_channels=config.audio_n_mels,
            conditioning_channels=config.hidden_size,
            upsample_factors=config.audio_vocoder_upsample_factors,
            n_residual_layers=config.audio_vocoder_n_residual_layers,
            conditioning_enabled=True,
            conditioning_mode='attention',
            attention_n_heads=4,
        )

        # Loss functions
        self.stft_loss = MultiResolutionSTFTLoss()

        # Mel spectrogram extractor for reconstruction loss
        self.mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.audio_sample_rate,
            n_fft=config.audio_n_fft,
            hop_length=config.audio_hop_length,
            n_mels=config.audio_n_mels,
            power=1.0,  # Magnitude spectrogram
        )

    def get_text_embeddings(self, text_input_ids: torch.Tensor, text_attention_mask: torch.Tensor) -> torch.Tensor:
        """Get frozen T5 embeddings for text conditioning."""
        with torch.no_grad():
            t5_outputs = self.t5_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
            )
            text_embeddings = t5_outputs.last_hidden_state

        # Project to vocoder conditioning dimension
        text_embeddings = self.condition_proj(text_embeddings)
        return text_embeddings

    def forward(
        self,
        mel_spec: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        waveform_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the vocoder with T5 text conditioning.

        Args:
            mel_spec: [B, n_mels, T_mel] Mel spectrogram input
            text_input_ids: [B, seq_len] T5 tokenized text
            text_attention_mask: [B, seq_len] Attention mask for T5
            waveform_labels: [B, T_audio] Target waveform for training

        Returns:
            Dictionary with loss and outputs
        """
        # Get text conditioning from frozen T5
        text_condition = self.get_text_embeddings(text_input_ids, text_attention_mask)

        # Generate waveform through vocoder
        pred_waveform = self.vocoder(mel_spec, condition=text_condition)

        outputs = {"pred_waveform": pred_waveform}

        if waveform_labels is not None:
            # Ensure waveform_labels has batch dimension to match pred_waveform
            if waveform_labels.dim() == 1:
                waveform_labels = waveform_labels.unsqueeze(0)

            # Align waveform lengths
            min_len = min(pred_waveform.shape[-1], waveform_labels.shape[-1])
            pred_waveform_aligned = pred_waveform[..., :min_len]
            waveform_labels_aligned = waveform_labels[..., :min_len]

            # Compute losses
            waveform_l1 = F.l1_loss(pred_waveform_aligned, waveform_labels_aligned)
            # STFT loss expects [B, 1, T] shape
            sc_loss, mag_loss = self.stft_loss(
                pred_waveform_aligned.unsqueeze(1) if pred_waveform_aligned.dim() == 2 else pred_waveform_aligned,
                waveform_labels_aligned.unsqueeze(1) if waveform_labels_aligned.dim() == 2 else waveform_labels_aligned
            )

            # Mel reconstruction loss: extract mel from predicted waveform and compare to input mel
            # This ensures round-trip consistency: mel -> waveform -> mel
            # Note: cuFFT doesn't support bfloat16, so we cast to float32 for mel extraction
            # We also need to temporarily move the mel transform to float32 for the filterbank matmul
            orig_dtype = pred_waveform_aligned.dtype
            self.mel_spec_transform = self.mel_spec_transform.float()
            pred_mel_from_waveform = self.mel_spec_transform(pred_waveform_aligned.float()).to(orig_dtype)
            self.mel_spec_transform = self.mel_spec_transform.to(orig_dtype)
            # Align mel lengths (may differ slightly due to padding/edge effects)
            mel_min_len = min(pred_mel_from_waveform.shape[-1], mel_spec.shape[-1])
            pred_mel_aligned = pred_mel_from_waveform[..., :mel_min_len]
            input_mel_aligned = mel_spec[..., :mel_min_len]
            # Log-scale mel comparison (more perceptually meaningful)
            mel_recon_loss = F.l1_loss(
                torch.log(pred_mel_aligned.clamp(min=1e-5)),
                torch.log(input_mel_aligned.clamp(min=1e-5))
            )

            # Total loss
            total_loss = (self.sc_loss_weight * sc_loss) + (self.mag_loss_weight * mag_loss) + (self.waveform_l1_loss_weight * waveform_l1) + (self.mel_recon_loss_weight * mel_recon_loss)

            outputs.update({
                "loss": total_loss,
                "waveform_l1": waveform_l1,
                "sc_loss": sc_loss,
                "mag_loss": mag_loss,
                "mel_recon_loss": mel_recon_loss,
            })

        return outputs


# ============================================================================
# Model Creation Functions
# ============================================================================

def create_small_vocoder_model(sc_loss_weight, mag_loss_weight, waveform_l1_loss_weight, mel_recon_loss_weight) -> VocoderWithT5Conditioning:
    """Create a small vocoder model for testing."""
    config = megatransformer_utils.MegaTransformerConfig(
        hidden_size=256,
        audio_n_mels=128,
        audio_n_fft=1024,
        audio_hop_length=512,
        audio_max_duration=10.0,
        audio_sample_rate=16000,
        audio_vocoder_hidden_channels=512,
        audio_vocoder_upsample_factors=[8, 8, 8],
        audio_vocoder_n_residual_layers=3,
    )
    return VocoderWithT5Conditioning(
        config,
        sc_loss_weight=sc_loss_weight,
        mag_loss_weight=mag_loss_weight,
        waveform_l1_loss_weight=waveform_l1_loss_weight,
        mel_recon_loss_weight=mel_recon_loss_weight,
        t5_model_name="google/t5-v1_1-small"
    )


def create_medium_vocoder_model() -> VocoderWithT5Conditioning:
    """Create a medium vocoder model."""
    config = megatransformer_utils.MegaTransformerConfig(
        hidden_size=512,
        audio_n_mels=128,
        audio_n_fft=1024,
        audio_hop_length=512,
        audio_max_duration=10.0,
        audio_sample_rate=16000,
        audio_vocoder_hidden_channels=1024,
        audio_vocoder_upsample_factors=[8, 8, 8],
        audio_vocoder_n_residual_layers=4,
    )
    return VocoderWithT5Conditioning(config, t5_model_name="google/t5-v1_1-base")


def create_large_vocoder_model() -> VocoderWithT5Conditioning:
    """Create a large vocoder model."""
    config = megatransformer_utils.MegaTransformerConfig(
        hidden_size=768,
        audio_n_mels=128,
        audio_n_fft=1024,
        audio_hop_length=512,
        audio_max_duration=10.0,
        audio_sample_rate=16000,
        audio_vocoder_hidden_channels=2048,
        audio_vocoder_upsample_factors=[8, 8, 8],
        audio_vocoder_n_residual_layers=6,
    )
    return VocoderWithT5Conditioning(config, t5_model_name="google/t5-v1_1-large")


model_config_lookup = {
    "small_vocoder": create_small_vocoder_model,
    "medium_vocoder": create_medium_vocoder_model,
    "large_vocoder": create_large_vocoder_model,
}


# ============================================================================
# Dataset Loading
# ============================================================================

class VocoderDataset(IterableDataset):
    """
    Dataset for vocoder training that loads audio with transcriptions.
    Returns mel spectrograms, waveforms, and text for T5 conditioning.
    """
    def __init__(
        self,
        config: megatransformer_utils.MegaTransformerConfig,
        t5_tokenizer: T5Tokenizer,
        approximated_length: int,
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        audio_max_frames: int,
        cache_dir: str = "cached_datasets",
        split: str = "train",
        dataset_name: str = "fixie-ai/common_voice_17_0",
        dataset_config: str = "en",
        max_text_length: int = 256,
    ):
        self.config = config
        self.t5_tokenizer = t5_tokenizer
        self.approximated_length = approximated_length
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.audio_max_frames = audio_max_frames
        self.cache_dir = cache_dir
        self.split = split
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.max_text_length = max_text_length

        # Load the base audio dataset
        self.dataset = load_dataset(
            dataset_name,
            dataset_config,
            cache_dir=cache_dir,
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=sample_rate))

    def __len__(self) -> int:
        return self.approximated_length

    def __iter__(self) -> Iterator[Any]:
        for example in self.dataset:
            try:
                processed = self._process_example(example)
                if processed is not None:
                    yield processed
            except Exception as e:
                print(f"Error processing example: {e}")
                continue

    def _process_example(self, example: Dict) -> Optional[Dict]:
        """Process a single example into vocoder training format."""
        # Get text - handle different column names
        text = example.get("text") or example.get("sentence") or example.get("caption")
        if text is None:
            return None

        # Get audio
        audio = example["audio"]

        # Extract waveform and mel spectrogram
        waveforms, y, _ = audio_loading.extract_waveforms(audio, sr=self.sample_rate)
        mel_spec = audio_loading.extract_mels(
            y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        # Skip if mel spectrogram is too long
        if mel_spec.shape[-1] > self.audio_max_frames:
            return None

        # Tokenize text for T5
        text_encoding = self.t5_tokenizer(
            text,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "mel_spec": mel_spec,
            "waveform_labels": waveforms,
            "text_input_ids": text_encoding["input_ids"].squeeze(0),
            "text_attention_mask": text_encoding["attention_mask"].squeeze(0),
        }


class VocoderDataCollator:
    """Data collator for vocoder training."""

    def __init__(
        self,
        audio_max_frames: int,
        audio_max_waveform_length: int,
        n_mels: int,
    ):
        self.audio_max_frames = audio_max_frames
        self.audio_max_waveform_length = audio_max_waveform_length
        self.n_mels = n_mels

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        # Pad mel spectrograms
        mel_specs = []
        for ex in examples:
            mel = ex["mel_spec"]
            if mel.shape[-1] < self.audio_max_frames:
                mel = F.pad(mel, (0, self.audio_max_frames - mel.shape[-1]), value=0)
            mel_specs.append(mel)

        # Pad waveforms
        waveforms = []
        for ex in examples:
            wav = ex["waveform_labels"]
            if wav.shape[-1] < self.audio_max_waveform_length:
                wav = F.pad(wav, (0, self.audio_max_waveform_length - wav.shape[-1]), value=0)
            elif wav.shape[-1] > self.audio_max_waveform_length:
                wav = wav[..., :self.audio_max_waveform_length]
            waveforms.append(wav)

        # Stack tensors
        batch = {
            "mel_spec": torch.stack(mel_specs),
            "waveform_labels": torch.stack(waveforms),
            "text_input_ids": torch.stack([ex["text_input_ids"] for ex in examples]),
            "text_attention_mask": torch.stack([ex["text_attention_mask"] for ex in examples]),
        }

        return batch


# ============================================================================
# Vocoder Reconstruction Callback
# ============================================================================

def get_writer(trainer: Trainer) -> Optional[SummaryWriter]:
    """Get TensorBoard writer from trainer callbacks."""
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, TensorBoardCallback):
            if callback.tb_writer is not None:
                return callback.tb_writer
    return None


class VocoderReconstructionCallback(TrainerCallback):
    """
    Callback for logging vocoder audio reconstruction during training.
    Periodically generates audio from test mel spectrograms and logs to TensorBoard.
    """

    def __init__(
        self,
        t5_tokenizer: T5Tokenizer,
        step_offset: int = 0,
        generation_steps: int = 1000,
        audio_sample_rate: int = 16000,
        audio_n_mels: int = 128,
        audio_n_fft: int = 1024,
        audio_hop_length: int = 512,
        test_audio_path: Optional[str] = None,
        test_audio_text: Optional[str] = None,
    ):
        self.trainer: Optional[Trainer] = None
        self.t5_tokenizer = t5_tokenizer
        self.step_offset = step_offset
        self.generation_steps = generation_steps
        self.audio_sample_rate = audio_sample_rate
        self.audio_n_mels = audio_n_mels
        self.audio_n_fft = audio_n_fft
        self.audio_hop_length = audio_hop_length

        # Load test audio
        if test_audio_path is not None and os.path.exists(test_audio_path):
            self.test_audio_waveforms, orig_sr = torchaudio.load(test_audio_path)
            if orig_sr != audio_sample_rate:
                self.test_audio_waveforms = torchaudio.transforms.Resample(
                    orig_freq=orig_sr, new_freq=audio_sample_rate
                )(self.test_audio_waveforms)
        else:
            # Default test audio path
            default_path = os.path.join('inference', 'examples', 'test_alm.mp3')
            if os.path.exists(default_path):
                self.test_audio_waveforms, orig_sr = torchaudio.load(default_path)
                self.test_audio_waveforms = torchaudio.transforms.Resample(
                    orig_freq=orig_sr, new_freq=audio_sample_rate
                )(self.test_audio_waveforms)
            else:
                # Generate synthetic test audio (sine wave)
                print("No test audio found, generating synthetic sine wave...")
                duration = 3.0  # seconds
                freq = 440.0  # Hz (A4 note)
                t = torch.linspace(0, duration, int(audio_sample_rate * duration))
                self.test_audio_waveforms = torch.sin(2 * np.pi * freq * t).unsqueeze(0)

        # Extract mel spectrogram from test audio
        self.test_mel_spec = audio_loading.extract_mels(
            self.test_audio_waveforms[0].numpy(),
            sr=audio_sample_rate,
            n_mels=audio_n_mels,
            n_fft=audio_n_fft,
            hop_length=audio_hop_length,
        )

        # Test text for conditioning
        self.test_text = test_audio_text or "It is from Westport, above the villages of Murrisk and Lecanvey."
        self.test_text_encoding = t5_tokenizer(
            self.test_text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def on_step_end(self, args, state, control, model: VocoderWithT5Conditioning = None, **kwargs):
        global_step = state.global_step + self.step_offset

        if ((global_step == 1) or (global_step % self.generation_steps == 0)) and state.is_world_process_zero:
            writer = get_writer(self.trainer)
            if writer is None:
                print("No TensorBoard writer found, skipping audio reconstruction...")
                return

            print(f"Reconstructing audio at step {global_step}...")

            # Determine device
            if torch.distributed.is_initialized():
                device = torch.device(f"cuda:{torch.distributed.get_rank()}")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # Move test data to device
            test_mel = self.test_mel_spec.unsqueeze(0).to(device)
            test_text_ids = self.test_text_encoding["input_ids"].to(device)
            test_text_mask = self.test_text_encoding["attention_mask"].to(device)
            test_waveform = self.test_audio_waveforms.to(device)

            with torch.no_grad():
                dtype = torch.bfloat16 if bool(args.bf16) else torch.float16 if args.fp16 else torch.float32
                with autocast(device.type, dtype=dtype):
                    # Generate waveform from mel spectrogram
                    outputs = model(
                        mel_spec=test_mel,
                        text_input_ids=test_text_ids,
                        text_attention_mask=test_text_mask,
                        waveform_labels=test_waveform[0] if test_waveform.dim() > 1 else test_waveform,
                    )

                    pred_waveform = outputs["pred_waveform"]

                    # Clip waveform to valid range
                    pred_waveform = torch.clamp(pred_waveform, -1.0, 1.0)
                    pred_waveform_cpu = pred_waveform[0].to(torch.float64).cpu()

                    # Log conditioning text
                    writer.add_text(
                        "vocoder_reconstruction/conditioning_text",
                        self.test_text,
                        global_step,
                    )

                    # Log ground truth audio
                    gt_waveform = self.test_audio_waveforms[0].to(torch.float64)
                    writer.add_audio(
                        "vocoder_reconstruction/ground_truth",
                        gt_waveform,
                        global_step,
                        sample_rate=self.audio_sample_rate,
                    )

                    # Log reconstructed audio
                    writer.add_audio(
                        "vocoder_reconstruction/predicted",
                        pred_waveform_cpu,
                        global_step,
                        sample_rate=self.audio_sample_rate,
                    )

                    # Save reconstructed audio to file
                    if self.trainer is not None and hasattr(self.trainer.args, "output_dir"):
                        audio_filepath = os.path.join(
                            self.trainer.args.output_dir,
                            f"reconstructed_audio_step_{global_step}.wav"
                        )
                        self._save_audio_to_file(
                            pred_waveform_cpu,
                            audio_filepath,
                            sample_rate=self.audio_sample_rate,
                        )

                    # Log mel spectrogram visualizations
                    try:
                        # Ground truth mel spectrogram
                        writer.add_image(
                            "vocoder_reconstruction/mel_spec_input",
                            self._visualize_mel_spec(
                                self.test_mel_spec.numpy(),
                                self.audio_sample_rate
                            ),
                            global_step,
                        )

                        # Reconstructed waveform's mel spectrogram
                        reconstructed_mel = audio_loading.extract_mels(
                            pred_waveform_cpu.numpy(),
                            sr=self.audio_sample_rate,
                            n_mels=self.audio_n_mels,
                            n_fft=self.audio_n_fft,
                            hop_length=self.audio_hop_length,
                        )
                        writer.add_image(
                            "vocoder_reconstruction/mel_spec_output",
                            self._visualize_mel_spec(
                                reconstructed_mel.numpy(),
                                self.audio_sample_rate
                            ),
                            global_step,
                        )
                    except Exception as e:
                        writer.add_text(
                            "vocoder_reconstruction/mel_spec_error",
                            f"Error visualizing mel spec: {e}",
                            global_step,
                        )

                    # Log losses
                    if "loss" in outputs:
                        writer.add_scalar(
                            "vocoder_reconstruction/loss",
                            outputs["loss"].item(),
                            global_step,
                        )
                    if "waveform_l1" in outputs:
                        writer.add_scalar(
                            "vocoder_reconstruction/waveform_l1",
                            outputs["waveform_l1"].item(),
                            global_step,
                        )
                    if "sc_loss" in outputs:
                        writer.add_scalar(
                            "vocoder_reconstruction/sc_loss",
                            outputs["sc_loss"].item(),
                            global_step,
                        )
                    if "mag_loss" in outputs:
                        writer.add_scalar(
                            "vocoder_reconstruction/mag_loss",
                            outputs["mag_loss"].item(),
                            global_step,
                        )

    def _save_audio_to_file(
        self,
        waveform: torch.Tensor,
        filepath: str,
        sample_rate: int,
        normalize: bool = True,
        bits_per_sample: int = 16,
    ):
        """Save waveform to audio file."""
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if normalize:
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        torchaudio.save(
            filepath,
            waveform.cpu(),
            sample_rate,
            bits_per_sample=bits_per_sample,
            format="wav",
        )
        print(f"Audio saved to {filepath}")

    def _visualize_mel_spec(self, mel_spec: np.ndarray, sample_rate: int) -> np.ndarray:
        """Generate mel spectrogram visualization for TensorBoard."""
        # Handle tensor input
        if hasattr(mel_spec, 'numpy'):
            mel_spec = mel_spec.numpy()

        # Ensure 2D
        if mel_spec.ndim == 3:
            mel_spec = mel_spec.squeeze(0)

        # Normalize to [0, 1] range
        mel_spec_norm = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)

        # Create figure with Agg backend to avoid display issues
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(
            mel_spec_norm,
            x_axis='time',
            y_axis='mel',
            sr=sample_rate,
            fmax=8000,
            ax=ax,
        )
        plt.colorbar(img, format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()

        # Convert figure to numpy array using Agg renderer
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape((height, width, 4))[:, :, :3]  # Drop alpha channel

        plt.close(fig)

        # TensorBoard expects (C, H, W) for add_image
        data = data.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

        return data


# ============================================================================
# Custom Trainer for Vocoder
# ============================================================================

class VocoderTrainer(Trainer):
    """Custom trainer for vocoder that handles the specific loss computation."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._ensure_tensorboard_writer()

        outputs = model(
            mel_spec=inputs["mel_spec"],
            text_input_ids=inputs["text_input_ids"],
            text_attention_mask=inputs["text_attention_mask"],
            waveform_labels=inputs["waveform_labels"],
        )

        loss = outputs["loss"]

        # Log individual losses
        if self.state.global_step % self.args.logging_steps == 0 and hasattr(self, "writer") and self.writer is not None:
            prefix = "train/" if model.training else "eval/"
            self._log_scalar(f"{prefix}waveform_l1", outputs.get("waveform_l1", 0))
            self._log_scalar(f"{prefix}sc_loss", outputs.get("sc_loss", 0))
            self._log_scalar(f"{prefix}mag_loss", outputs.get("mag_loss", 0))
            self._log_scalar(f"{prefix}mel_recon_loss", outputs.get("mel_recon_loss", 0))

        return (loss, outputs) if return_outputs else loss

    def _log_scalar(self, tag, value):
        if hasattr(self, "writer") and self.writer is not None:
            if isinstance(value, torch.Tensor):
                value = value.item()
            if value != 0.0:
                self.writer.add_scalar(tag, value, self.state.global_step)

    def _ensure_tensorboard_writer(self):
        if hasattr(self, "writer"):
            return

        for callback in self.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback):
                self.writer = callback.tb_writer
                return

        self.writer = None


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    args, unk = megatransformer_utils.parse_args()
    run_dir = os.path.join(args.logging_base_dir, args.run_name)

    # Select model configuration
    if args.config not in model_config_lookup:
        raise ValueError(f"Unknown vocoder config: {args.config}. Available: {list(model_config_lookup.keys())}")

    unk_dict = {}
    for i in range(0, len(unk), 2):
        unk_dict[unk[i]] = unk[i+1]

    model = model_config_lookup[args.config](unk_dict.get("sc_loss_weight", 1.0),
                                            unk_dict.get("mag_loss_weight", 3.0),
                                            unk_dict.get("waveform_l1_loss_weight", 0.1),
                                            unk_dict.get("mel_recon_loss_weight", 1.0))
    model, model_loaded = megatransformer_utils.load_model(False, model, run_dir)

    if args.local_rank == 0 or not args.use_deepspeed:
        print(f"Model structure: {model}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"  Vocoder parameters: {sum(p.numel() for p in model.vocoder.parameters()):,}")
        print(f"  Condition projection parameters: {sum(p.numel() for p in model.condition_proj.parameters()):,}")
        print(f"  T5 parameters (frozen): {sum(p.numel() for p in model.t5_encoder.parameters()):,}")

    model = megatransformer_utils.setup_int8_training(args, model)

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    # Training arguments
    training_args = TrainingArguments(
        tpu_num_cores=8 if args.use_xla else None,
        output_dir=run_dir,
        overwrite_output_dir=True,
        lr_scheduler_type="cosine",
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs if args.num_train_epochs > 0 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        weight_decay=args.weight_decay,
        report_to="tensorboard",
        logging_dir=run_dir,
        logging_steps=args.logging_steps,
        eval_strategy="no",
        save_safetensors=False,
        save_steps=args.save_steps,
        gradient_checkpointing=args.use_gradient_checkpointing,
        bf16=args.bf16,
        fp16=args.fp16,
        max_grad_norm=args.max_grad_norm,
        torch_compile=args.compile_model and not args.use_deepspeed and not args.use_xla,
        deepspeed=args.deepspeed_config if args.use_deepspeed and not args.use_xla else None,
        use_cpu=args.cpu,
        log_level=args.log_level,
        logging_first_step=True,
        local_rank=args.local_rank,
    )

    # Create datasets
    train_dataset = VocoderDataset(
        config=model.config,
        t5_tokenizer=model.t5_tokenizer,
        approximated_length=1_100_000,
        sample_rate=model.config.audio_sample_rate,
        n_mels=model.config.audio_n_mels,
        n_fft=model.config.audio_n_fft,
        hop_length=model.config.audio_hop_length,
        audio_max_frames=model.config.audio_max_frames,
        cache_dir=args.dataset_cache_dir,
        split="train",
    )

    eval_dataset = VocoderDataset(
        config=model.config,
        t5_tokenizer=model.t5_tokenizer,
        approximated_length=16_400,
        sample_rate=model.config.audio_sample_rate,
        n_mels=model.config.audio_n_mels,
        n_fft=model.config.audio_n_fft,
        hop_length=model.config.audio_hop_length,
        audio_max_frames=model.config.audio_max_frames,
        cache_dir=args.dataset_cache_dir,
        split="validation",
    )

    # Create data collator
    data_collator = VocoderDataCollator(
        audio_max_frames=model.config.audio_max_frames,
        audio_max_waveform_length=model.config.audio_max_waveform_length,
        n_mels=model.config.audio_n_mels,
    )

    # Create trainer
    trainer = VocoderTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Add reconstruction callback for monitoring training progress
    reconstruction_callback = VocoderReconstructionCallback(
        t5_tokenizer=model.t5_tokenizer,
        step_offset=args.start_step,
        generation_steps=args.generation_steps,
        audio_sample_rate=model.config.audio_sample_rate,
        audio_n_mels=model.config.audio_n_mels,
        audio_n_fft=model.config.audio_n_fft,
        audio_hop_length=model.config.audio_hop_length,
    )
    trainer.add_callback(reconstruction_callback)
    reconstruction_callback.trainer = trainer

    print(f"Starting vocoder training with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
