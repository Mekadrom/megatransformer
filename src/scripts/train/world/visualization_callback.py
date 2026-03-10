from typing import Optional

import numpy as np
import torch
from torch.amp import autocast
from transformers import Trainer

from scripts.train.visualization_callback import VisualizationCallback
from utils.constants import (
    BOA_TOKEN_ID, EOA_TOKEN_ID,
    BOV_TOKEN_ID, EOV_TOKEN_ID,
    BOI_TOKEN_ID, EOI_TOKEN_ID,
    AUDIO_PLACEHOLDER_TOKEN_ID,
    VOICE_PLACEHOLDER_TOKEN_ID,
    IMAGE_PLACEHOLDER_TOKEN_ID,
)
from utils.train_utils import get_writer


class WorldModelVisualizationCallback(VisualizationCallback):
    """
    Visualization callback for the multimodal world model.

    Runs 7 cross-modal generation scenarios on each evaluation:
    1. Text-only continuation
    2. Text -> Audio synthesis
    3. Audio -> Text transcription
    4. Text -> Image synthesis
    5. Image -> Text description
    6. Audio -> Image cross-modal
    7. Image -> Audio cross-modal
    """

    def __init__(
        self,
        tokenizer=None,
        vocoder: Optional[torch.nn.Module] = None,
        image_vae_decoder: Optional[torch.nn.Module] = None,
        voice_cvae_decoder: Optional[torch.nn.Module] = None,
        static_speaker_embedding: Optional[torch.Tensor] = None,
        num_eval_samples: int = 4,
        step_offset: int = 0,
        audio_sample_rate: int = 16000,
        audio_n_mels: int = 80,
        audio_n_fft: int = 1024,
        audio_hop_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.vocoder = vocoder
        self.image_vae_decoder = image_vae_decoder
        self.voice_cvae_decoder = voice_cvae_decoder
        self.static_speaker_embedding = static_speaker_embedding
        self.num_eval_samples = num_eval_samples
        self.step_offset = step_offset if step_offset is not None else 0
        self.audio_sample_rate = audio_sample_rate
        self.audio_n_mels = audio_n_mels
        self.audio_n_fft = audio_n_fft
        self.audio_hop_length = audio_hop_length

        self.trainer: Optional[Trainer] = None

    def _ensure_tokenizer(self):
        """Lazy-load a tokenizer if none was provided."""
        if self.tokenizer is not None:
            return
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")

    def _decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text, filtering out special tokens."""
        self._ensure_tokenizer()
        if self.tokenizer is None:
            return f"[token_ids: {token_ids.tolist()[:20]}...]"
        # Filter out media tokens (>= 32000)
        text_ids = [t for t in token_ids.tolist() if t < 32000]
        return self.tokenizer.decode(text_ids, skip_special_tokens=True)

    def _get_eval_samples(self, eval_dataset, collator, n: int,
                          requires_audio=False, requires_voice=False, requires_image=False):
        """Get n random samples from eval dataset that match modality requirements."""
        indices = torch.randperm(len(eval_dataset))
        samples = []
        for idx in indices:
            sample = eval_dataset[idx.item()]
            if requires_audio and not any(k.startswith("audio_") for k in sample):
                continue
            if requires_voice and not any(k.startswith("voice_") for k in sample):
                continue
            if requires_image and "image_images" not in sample and "image_image" not in sample:
                continue
            samples.append(sample)
            if len(samples) >= n:
                break
        return samples

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        global_step = state.global_step + self.step_offset

        if not state.is_world_process_zero:
            return

        writer = get_writer(self.trainer)
        if writer is None:
            print("No TensorBoard writer found, skipping world model visualization...")
            return

        eval_dataset = self.trainer.eval_dataset
        if eval_dataset is None or len(eval_dataset) == 0:
            print("No eval dataset available, skipping visualization...")
            return

        collator = self.trainer.data_collator
        device = self._get_device()
        model.eval()

        dtype = torch.bfloat16 if bool(args.bf16) else torch.float16 if args.fp16 else torch.float32

        print(f"Running world model visualization at step {global_step}...")

        with torch.no_grad():
            with autocast(device.type, dtype=dtype, enabled=args.bf16 or args.fp16):
                # Scenario 1: Text-only generation
                self._scenario_text_continuation(
                    model, eval_dataset, collator, device, writer, global_step
                )

                # Scenario 2: Text -> Voice synthesis
                self._scenario_text_to_voice(
                    model, eval_dataset, collator, device, writer, global_step
                )

                # Scenario 3: Voice -> Text transcription
                self._scenario_voice_to_text(
                    model, eval_dataset, collator, device, writer, global_step
                )

                # Scenario 4: Text -> Image synthesis
                self._scenario_text_to_image(
                    model, eval_dataset, collator, device, writer, global_step
                )

                # Scenario 5: Image -> Text description
                self._scenario_image_to_text(
                    model, eval_dataset, collator, device, writer, global_step
                )

                # Scenario 6: Voice -> Image
                self._scenario_voice_to_image(
                    model, eval_dataset, collator, device, writer, global_step
                )

                # Scenario 7: Image -> Voice
                self._scenario_image_to_voice(
                    model, eval_dataset, collator, device, writer, global_step
                )

        print(f"World model visualization complete at step {global_step}")
        writer.flush()

    def _build_prompt_ids(self, tokens: list[int], device: torch.device) -> torch.Tensor:
        """Build a prompt tensor from a list of token IDs. Shape: (1, seq_len)."""
        return torch.tensor([tokens], dtype=torch.long, device=device)

    def _log_recurrent_iterations(self, outputs: dict, writer, tag: str, sample_idx: int, global_step: int):
        """Log recurrent iteration count statistics from generate() outputs."""
        iter_counts = outputs.get("recurrent_iteration_counts")
        if not iter_counts:
            return
        counts = torch.tensor(iter_counts, dtype=torch.float32)
        writer.add_scalar(f"{tag}/{sample_idx}/recurrent_iters_mean", counts.mean().item(), global_step)
        writer.add_scalar(f"{tag}/{sample_idx}/recurrent_iters_min", counts.min().item(), global_step)
        writer.add_scalar(f"{tag}/{sample_idx}/recurrent_iters_max", counts.max().item(), global_step)
        writer.add_histogram(f"{tag}/{sample_idx}/recurrent_iters", counts, global_step)
        prompt_iters = outputs.get("prompt_recurrent_iterations")
        if prompt_iters is not None:
            writer.add_scalar(f"{tag}/{sample_idx}/recurrent_iters_prompt", prompt_iters, global_step)

    def _scenario_text_continuation(self, model, eval_dataset, collator, device, writer, global_step):
        """Scenario 1: Text-only generation (take text, generate continuation)."""
        tag = "eval_world/1_text_continuation"
        samples = self._get_eval_samples(eval_dataset, collator, self.num_eval_samples)
        if not samples:
            return

        for i, sample in enumerate(samples):
            token_ids = sample.get("text_token_ids")
            if token_ids is None:
                continue

            text_length = sample.get("text_text_length", token_ids.shape[0])
            if isinstance(text_length, torch.Tensor):
                text_length = text_length.item()

            # Use first half as prompt
            prompt_len = max(1, text_length // 2)
            prompt = token_ids[:prompt_len].unsqueeze(0).to(device)

            outputs = model.generate(
                text_input_ids=prompt,
                max_new_tokens=min(256, text_length),
                temperature=0.8,
                top_p=0.9,
            )

            gen_ids = outputs["generated_token_ids"][0]
            input_text = self._decode_tokens(token_ids[:prompt_len])
            gen_text = self._decode_tokens(gen_ids)
            full_target = self._decode_tokens(token_ids[:text_length])

            writer.add_text(f"{tag}/{i}/input", input_text, global_step)
            writer.add_text(f"{tag}/{i}/generated", gen_text, global_step)
            writer.add_text(f"{tag}/{i}/target", full_target, global_step)
            self._log_recurrent_iterations(outputs, writer, tag, i, global_step)

    def _scenario_text_to_voice(self, model, eval_dataset, collator, device, writer, global_step):
        """Scenario 2: Text -> Voice synthesis."""
        tag = "eval_world/2_text_to_voice"
        samples = self._get_eval_samples(
            eval_dataset, collator, self.num_eval_samples, requires_voice=True
        )
        if not samples:
            return

        for i, sample in enumerate(samples):
            token_ids = sample.get("text_token_ids")
            if token_ids is None:
                continue

            text_length = sample.get("text_text_length", token_ids.shape[0])
            if isinstance(text_length, torch.Tensor):
                text_length = text_length.item()

            # Build prompt: [text] [BOV]
            prompt_tokens = token_ids[:text_length].tolist() + [BOV_TOKEN_ID]
            prompt = self._build_prompt_ids(prompt_tokens, device)

            outputs = model.generate(
                text_input_ids=prompt,
                max_new_tokens=512,
                temperature=0.8,
            )

            input_text = self._decode_tokens(token_ids[:text_length])
            writer.add_text(f"{tag}/{i}/input_text", input_text, global_step)

            # Log generated voice if available
            voice_preds = outputs.get("voice_latent_preds")
            if voice_preds is not None and voice_preds.numel() > 0:
                pred_latent = voice_preds[0, 0]  # (C, T)
                writer.add_image(
                    f"{tag}/{i}/generated_latent",
                    self._latent_to_image(pred_latent),
                    global_step,
                )

                self._log_audio_with_cvae(
                    pred_latent, sample, writer, global_step, f"{tag}/{i}"
                )

            self._log_recurrent_iterations(outputs, writer, tag, i, global_step)

            # Log target voice for comparison
            target_features = sample.get("voice_features")
            if target_features is not None:
                writer.add_image(
                    f"{tag}/{i}/target_latent",
                    self._latent_to_image(target_features),
                    global_step,
                )

    def _prepare_audio_for_generate(self, sample, device):
        """Prepare SIVE audio features from a sample for model.generate().

        Returns (audio_inputs, audio_lengths) shaped for generate() or (None, None).
        Audio inputs are shaped (1, 1, C, T) — batch=1, n_audio=1.
        """
        audio_features = sample.get("audio_features")
        if audio_features is None:
            return None, None

        audio_data = audio_features.to(device)
        # Normalize to (1, 1, C, T)
        if audio_data.dim() == 2:
            # (C, T) -> (1, 1, C, T)
            audio_data = audio_data.unsqueeze(0).unsqueeze(0)
        elif audio_data.dim() == 3:
            # (B, C, T) -> (B, 1, C, T)
            audio_data = audio_data.unsqueeze(1)

        feat_length = sample.get("audio_feature_length")
        if feat_length is not None:
            if isinstance(feat_length, torch.Tensor):
                audio_lengths = feat_length.unsqueeze(0).unsqueeze(0).to(device)
            else:
                audio_lengths = torch.tensor([[feat_length]], device=device)
        else:
            audio_lengths = torch.tensor([[audio_data.shape[-1]]], device=device)

        return audio_data, audio_lengths

    def _prepare_voice_for_generate(self, sample, device):
        """Prepare SIVE voice features from a sample for model.generate().

        Returns (voice_inputs, voice_lengths) shaped for generate() or (None, None).
        Voice inputs are shaped (1, 1, C, T) — batch=1, n_voice=1.
        """
        voice_features = sample.get("voice_features")
        if voice_features is None:
            return None, None

        voice_data = voice_features.to(device)
        if voice_data.dim() == 2:
            voice_data = voice_data.unsqueeze(0).unsqueeze(0)
        elif voice_data.dim() == 3:
            voice_data = voice_data.unsqueeze(1)

        feat_length = sample.get("voice_feature_length")
        if feat_length is not None:
            if isinstance(feat_length, torch.Tensor):
                voice_lengths = feat_length.unsqueeze(0).unsqueeze(0).to(device)
            else:
                voice_lengths = torch.tensor([[feat_length]], device=device)
        else:
            voice_lengths = torch.tensor([[voice_data.shape[-1]]], device=device)

        return voice_data, voice_lengths

    def _prepare_image_for_generate(self, sample, device):
        """Prepare image from a sample for model.generate().

        Returns image_inputs shaped (1, 1, C, H, W) or None.
        """
        image = sample.get("image_images")
        if image is None:
            image = sample.get("image_image")
        if image is None:
            return None

        image_data = image.to(device)
        if image_data.dim() == 3:
            image_data = image_data.unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
        elif image_data.dim() == 4:
            image_data = image_data.unsqueeze(0)
        return image_data

    def _scenario_voice_to_text(self, model, eval_dataset, collator, device, writer, global_step):
        """Scenario 3: Voice -> Text transcription."""
        tag = "eval_world/3_voice_to_text"
        samples = self._get_eval_samples(
            eval_dataset, collator, self.num_eval_samples, requires_voice=True
        )
        if not samples:
            return

        for i, sample in enumerate(samples):
            token_ids = sample.get("text_token_ids")
            voice_features = sample.get("voice_features")
            if token_ids is None or voice_features is None:
                continue

            text_length = sample.get("text_text_length", token_ids.shape[0])
            if isinstance(text_length, torch.Tensor):
                text_length = text_length.item()

            # Build prompt: [BOV] [VOICE_PLACEHOLDER] [EOV]
            prompt_tokens = [BOV_TOKEN_ID, VOICE_PLACEHOLDER_TOKEN_ID, EOV_TOKEN_ID]
            prompt = self._build_prompt_ids(prompt_tokens, device)

            voice_inputs, voice_lengths = self._prepare_voice_for_generate(sample, device)

            outputs = model.generate(
                text_input_ids=prompt,
                max_new_tokens=256,
                temperature=0.8,
                top_p=0.9,
                voice_inputs=voice_inputs,
                voice_lengths=voice_lengths,
            )

            gen_ids = outputs["generated_token_ids"][0]
            gen_text = self._decode_tokens(gen_ids)
            target_text = self._decode_tokens(token_ids[:text_length])

            writer.add_text(f"{tag}/{i}/generated", gen_text, global_step)
            writer.add_text(f"{tag}/{i}/target", target_text, global_step)
            self._log_recurrent_iterations(outputs, writer, tag, i, global_step)

    def _scenario_text_to_image(self, model, eval_dataset, collator, device, writer, global_step):
        """Scenario 4: Text -> Image synthesis."""
        tag = "eval_world/4_text_to_image"
        samples = self._get_eval_samples(
            eval_dataset, collator, self.num_eval_samples, requires_image=True
        )
        if not samples:
            return

        for i, sample in enumerate(samples):
            token_ids = sample.get("text_token_ids")
            if token_ids is None:
                continue

            text_length = sample.get("text_text_length", token_ids.shape[0])
            if isinstance(text_length, torch.Tensor):
                text_length = text_length.item()

            # Build prompt: [text] [BOI]
            prompt_tokens = token_ids[:text_length].tolist() + [BOI_TOKEN_ID]
            prompt = self._build_prompt_ids(prompt_tokens, device)

            outputs = model.generate(
                text_input_ids=prompt,
                max_new_tokens=512,
                temperature=0.8,
            )

            input_text = self._decode_tokens(token_ids[:text_length])
            writer.add_text(f"{tag}/{i}/input_text", input_text, global_step)

            image_preds = outputs.get("image_latent_preds")
            if image_preds is not None and image_preds.numel() > 0:
                pred_latent = image_preds[0, 0]  # (C, H, W)
                writer.add_image(
                    f"{tag}/{i}/generated_latent",
                    self._latent_to_image(pred_latent),
                    global_step,
                )

                # Decode through image VAE if available
                if self.image_vae_decoder is not None:
                    self._try_decode_image(
                        pred_latent, writer, global_step,
                        f"{tag}/{i}/generated_image"
                    )

            # Log target image
            target_image = sample.get("image_images")
            if target_image is None:
                target_image = sample.get("image_image")
            if target_image is not None:
                writer.add_image(
                    f"{tag}/{i}/target_latent",
                    self._latent_to_image(target_image),
                    global_step,
                )
                if self.image_vae_decoder is not None:
                    self._try_decode_image(
                        target_image, writer, global_step,
                        f"{tag}/{i}/target_image"
                    )

            self._log_recurrent_iterations(outputs, writer, tag, i, global_step)

    def _scenario_image_to_text(self, model, eval_dataset, collator, device, writer, global_step):
        """Scenario 5: Image -> Text description."""
        tag = "eval_world/5_image_to_text"
        samples = self._get_eval_samples(
            eval_dataset, collator, self.num_eval_samples, requires_image=True
        )
        if not samples:
            return

        for i, sample in enumerate(samples):
            token_ids = sample.get("text_token_ids")
            if token_ids is None:
                continue

            text_length = sample.get("text_text_length", token_ids.shape[0])
            if isinstance(text_length, torch.Tensor):
                text_length = text_length.item()

            # Build prompt: [BOI] [IMAGE_PLACEHOLDER] [EOI]
            prompt_tokens = [BOI_TOKEN_ID, IMAGE_PLACEHOLDER_TOKEN_ID, EOI_TOKEN_ID]
            prompt = self._build_prompt_ids(prompt_tokens, device)

            image_inputs = self._prepare_image_for_generate(sample, device)

            outputs = model.generate(
                text_input_ids=prompt,
                max_new_tokens=256,
                temperature=0.8,
                top_p=0.9,
                image_inputs=image_inputs,
            )

            gen_ids = outputs["generated_token_ids"][0]
            gen_text = self._decode_tokens(gen_ids)
            target_text = self._decode_tokens(token_ids[:text_length])

            writer.add_text(f"{tag}/{i}/generated", gen_text, global_step)
            writer.add_text(f"{tag}/{i}/target", target_text, global_step)
            self._log_recurrent_iterations(outputs, writer, tag, i, global_step)

    def _scenario_voice_to_image(self, model, eval_dataset, collator, device, writer, global_step):
        """Scenario 6: Voice -> Image cross-modal generation."""
        tag = "eval_world/6_voice_to_image"
        samples = self._get_eval_samples(
            eval_dataset, collator, self.num_eval_samples, requires_voice=True
        )
        if not samples:
            return

        for i, sample in enumerate(samples):
            # Build prompt: [BOV] [VOICE_PLACEHOLDER] [EOV] [BOI]
            prompt_tokens = [
                BOV_TOKEN_ID, VOICE_PLACEHOLDER_TOKEN_ID, EOV_TOKEN_ID,
                BOI_TOKEN_ID,
            ]
            prompt = self._build_prompt_ids(prompt_tokens, device)

            voice_inputs, voice_lengths = self._prepare_voice_for_generate(sample, device)

            outputs = model.generate(
                text_input_ids=prompt,
                max_new_tokens=512,
                temperature=0.8,
                voice_inputs=voice_inputs,
                voice_lengths=voice_lengths,
            )

            image_preds = outputs.get("image_latent_preds")
            if image_preds is not None and image_preds.numel() > 0:
                pred_latent = image_preds[0, 0]
                writer.add_image(
                    f"{tag}/{i}/generated_image_latent",
                    self._latent_to_image(pred_latent),
                    global_step,
                )

                if self.image_vae_decoder is not None:
                    self._try_decode_image(
                        pred_latent, writer, global_step,
                        f"{tag}/{i}/generated_image"
                    )

            self._log_recurrent_iterations(outputs, writer, tag, i, global_step)

    def _scenario_image_to_voice(self, model, eval_dataset, collator, device, writer, global_step):
        """Scenario 7: Image -> Voice cross-modal generation."""
        tag = "eval_world/7_image_to_voice"
        samples = self._get_eval_samples(
            eval_dataset, collator, self.num_eval_samples, requires_image=True
        )
        if not samples:
            return

        for i, sample in enumerate(samples):
            # Build prompt: [BOI] [IMAGE_PLACEHOLDER] [EOI] [BOV]
            prompt_tokens = [
                BOI_TOKEN_ID, IMAGE_PLACEHOLDER_TOKEN_ID, EOI_TOKEN_ID,
                BOV_TOKEN_ID,
            ]
            prompt = self._build_prompt_ids(prompt_tokens, device)

            image_inputs = self._prepare_image_for_generate(sample, device)

            outputs = model.generate(
                text_input_ids=prompt,
                max_new_tokens=512,
                temperature=0.8,
                image_inputs=image_inputs,
            )

            voice_preds = outputs.get("voice_latent_preds")
            if voice_preds is not None and voice_preds.numel() > 0:
                pred_latent = voice_preds[0, 0]
                writer.add_image(
                    f"{tag}/{i}/generated_voice_latent",
                    self._latent_to_image(pred_latent),
                    global_step,
                )

                # Image->voice has no ground-truth speaker, use static only
                self._log_audio_with_cvae(
                    pred_latent, sample, writer, global_step, f"{tag}/{i}"
                )

            self._log_recurrent_iterations(outputs, writer, tag, i, global_step)

    # --- Helper methods ---

    def _latent_to_image(self, latent: torch.Tensor) -> np.ndarray:
        """Convert a latent tensor to a grid of per-channel grayscale images.

        Each channel is individually normalized to [0, 1] and arranged in a grid.
        Returns (1, grid_H, grid_W) for TensorBoard add_image (grayscale).
        """
        import math

        latent = latent.float().cpu()

        if latent.dim() == 1:
            side = int(latent.shape[0] ** 0.5) + 1
            padded = torch.zeros(side * side)
            padded[:latent.shape[0]] = latent
            latent = padded.view(1, side, side)
        elif latent.dim() == 2:
            latent = latent.unsqueeze(0)

        C, H, W = latent.shape

        # Normalize each channel independently
        channels = []
        for c in range(C):
            ch = latent[c]
            vmin, vmax = ch.min(), ch.max()
            if vmax - vmin > 1e-8:
                ch = (ch - vmin) / (vmax - vmin)
            else:
                ch = torch.zeros_like(ch)
            channels.append(ch)

        # Arrange in a grid (e.g. 12 channels -> 3x4 or 4x3)
        ncols = math.ceil(math.sqrt(C))
        nrows = math.ceil(C / ncols)

        grid = torch.zeros(nrows * H, ncols * W)
        for idx, ch in enumerate(channels):
            r, c = divmod(idx, ncols)
            grid[r * H:(r + 1) * H, c * W:(c + 1) * W] = ch

        return grid.unsqueeze(0).numpy()  # (1, grid_H, grid_W)

    def _decode_audio_latent_to_mel(self, latent: torch.Tensor, speaker_embedding: torch.Tensor) -> Optional[torch.Tensor]:
        """Decode SIVE feature latent through CVAE decoder to mel spectrogram.

        Args:
            latent: (C, T) predicted SIVE features from the world model's audio coda
            speaker_embedding: (speaker_dim,) speaker embedding

        Returns:
            mel_spec (n_mels, T) or None on failure
        """
        try:
            if self.voice_cvae_decoder is None:
                return None

            device = next(self.voice_cvae_decoder.parameters()).device
            dtype = next(self.voice_cvae_decoder.parameters()).dtype

            z = latent.to(device=device, dtype=dtype).unsqueeze(0)  # (1, C, T)
            spk = speaker_embedding.to(device=device, dtype=dtype).unsqueeze(0)  # (1, speaker_dim)

            with torch.no_grad():
                # Pass features=z so the F0 predictor can run (SIVE features are the latent)
                mel = self.voice_cvae_decoder.decode(z=z, speaker_embedding=spk, features=z)

            # mel: (1, n_mels, T) -> (n_mels, T)
            if isinstance(mel, dict):
                mel = mel.get("reconstructed", mel.get("output", next(iter(mel.values()))))
            if isinstance(mel, torch.Tensor):
                return mel[0].float().cpu()
            return None
        except Exception as e:
            print(f"Warning: CVAE decode failed: {e}")
            return None

    def _decode_and_log_audio(self, pred_latent, speaker_embedding, writer, global_step, tag_prefix, speaker_label):
        """Decode latent -> mel via CVAE, then mel -> waveform via vocoder, logging both."""
        mel = self._decode_audio_latent_to_mel(pred_latent, speaker_embedding)
        if mel is None:
            return

        # Log mel spectrogram as image
        mel_img = self._visualize_mel_spec(mel.numpy())
        writer.add_image(
            f"{tag_prefix}/{speaker_label}_mel",
            mel_img, global_step,
        )

        # Vocode to audio if possible
        if self.vocoder is not None:
            self._log_vocoder_audio(writer, mel, global_step, f"{tag_prefix}/{speaker_label}_audio")

    def _log_audio_with_cvae(self, pred_latent, sample, writer, global_step, tag_prefix):
        """Run dual-speaker CVAE decoding: ground-truth speaker + static speaker."""
        if self.voice_cvae_decoder is None:
            # Fallback to direct vocoder (old behavior)
            if self.vocoder is not None:
                self._try_vocoder_from_latent(
                    None, pred_latent, writer, global_step,
                    f"{tag_prefix}/generated_audio"
                )
            return

        # Decode with ground-truth speaker embedding from the sample
        gt_speaker_emb = None
        for key in ("voice_speaker_embeddings", "voice_speaker_embedding",
                     "audio_speaker_embeddings", "audio_speaker_embedding"):
            v = sample.get(key)
            if v is not None:
                gt_speaker_emb = v
                break
        if gt_speaker_emb is not None:
            self._decode_and_log_audio(
                pred_latent, gt_speaker_emb, writer, global_step,
                tag_prefix, "gt_speaker"
            )

        # Decode with static speaker embedding
        if self.static_speaker_embedding is not None:
            self._decode_and_log_audio(
                pred_latent, self.static_speaker_embedding, writer, global_step,
                tag_prefix, "static_speaker"
            )

    def _try_vocoder_from_latent(self, model, latent, writer, global_step, tag):
        """Try to decode SIVE feature latent directly through vocoder.

        Note: This is a fallback — SIVE features are not mel spectrograms, so
        direct vocoding only makes sense if feature_channels == n_mels.
        Prefer using CVAE decoder → vocoder for proper audio synthesis.
        """
        try:
            if self.vocoder is None:
                return

            mel = latent.float().cpu()
            # SIVE features are (C, T) — use directly as (n_mels, T) if C matches
            self._log_vocoder_audio(writer, mel, global_step, tag)
        except Exception as e:
            print(f"Warning: Vocoder decoding failed for {tag}: {e}")

    def _try_decode_image(self, latent, writer, global_step, tag):
        """Try to decode image latent through image VAE decoder."""
        try:
            if self.image_vae_decoder is None:
                return

            device = next(self.image_vae_decoder.parameters()).device
            dtype = next(self.image_vae_decoder.parameters()).dtype
            latent_input = latent.to(device=device, dtype=dtype).unsqueeze(0)

            with torch.no_grad():
                # LiteVAE uses .decode(), our custom decoder uses forward()
                if hasattr(self.image_vae_decoder, 'decode'):
                    decoded = self.image_vae_decoder.decode(latent_input)
                    if hasattr(decoded, 'sample'):
                        decoded = decoded.sample
                else:
                    decoded = self.image_vae_decoder(latent_input)

            # decoded: (1, 3, H, W) — normalize to [0, 1]
            img = decoded[0].float().cpu()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            writer.add_image(tag, img, global_step)
        except Exception as e:
            print(f"Warning: Image VAE decoding failed for {tag}: {e}")
