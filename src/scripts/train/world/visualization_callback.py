from typing import List, Optional

import math
import numpy as np
import torch
import torch.nn.functional as F
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

        # Static prompts for synthesis scenarios — avoids markup/JS garbage from web-scraped data
        self.VOICE_SYNTHESIS_PROMPTS = [
            "The quick brown fox jumps over the lazy dog near the riverbank.",
            "She sold seashells by the seashore on a warm summer afternoon.",
            "In a hole in the ground there lived a hobbit.",
            "To be or not to be, that is the question.",
            "The rain in Spain falls mainly on the plain.",
            "All human beings are born free and equal in dignity and rights.",
            "It was a bright cold day in April, and the clocks were striking thirteen.",
            "The only thing we have to fear is fear itself.",
        ]
        self.IMAGE_SYNTHESIS_PROMPTS = [
            "A golden retriever sitting in a sunny meadow with wildflowers.",
            "A red sports car parked on a winding mountain road at sunset.",
            "A cozy kitchen with a steaming cup of coffee on a wooden table.",
            "A snow-covered cabin in the woods under a starry night sky.",
            "A colorful hot air balloon floating above rolling green hills.",
            "An old lighthouse standing on a rocky cliff above crashing waves.",
            "A bustling city street at night with neon signs and reflections.",
            "A bowl of fresh fruit on a marble countertop in natural light.",
        ]

    def _encode_static_prompt(self, text: str, suffix_tokens: list[int], max_new_tokens: int, device: torch.device) -> torch.Tensor:
        """Tokenize a static text prompt, append suffix tokens, and cap to MAX_SEQ_LEN."""
        self._ensure_tokenizer()
        if self.tokenizer is None:
            # Fallback: just use suffix tokens
            return self._build_prompt_ids(suffix_tokens, device)
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        prompt_tokens = self._cap_prompt_tokens(token_ids, suffix_tokens, max_new_tokens)
        return self._build_prompt_ids(prompt_tokens, device)

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
                # Training data reconstruction — tracks memorization/overfitting
                self._scenario_train_reconstruction(
                    model, args, device, writer, global_step, dtype
                )

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

    # Maximum total sequence length (prompt + generated) the model supports.
    MAX_SEQ_LEN = 1024

    def _cap_prompt_tokens(
        self, text_tokens: list[int], suffix_tokens: list[int], max_new_tokens: int
    ) -> list[int]:
        """Truncate text_tokens so prompt + max_new_tokens <= MAX_SEQ_LEN.

        Args:
            text_tokens: The variable-length text portion of the prompt.
            suffix_tokens: Fixed tokens appended after text (e.g. [BOV]).
            max_new_tokens: How many tokens will be generated after the prompt.

        Returns:
            Full prompt token list (text_tokens + suffix_tokens), truncated if needed.
        """
        budget = self.MAX_SEQ_LEN - max_new_tokens - len(suffix_tokens)
        if budget < 1:
            budget = 1
        return text_tokens[:budget] + suffix_tokens

    def _build_prompt_ids(self, tokens: list[int], device: torch.device) -> torch.Tensor:
        """Build a prompt tensor from a list of token IDs. Shape: (1, seq_len)."""
        return torch.tensor([tokens], dtype=torch.long, device=device)

    def _log_generation_metrics(
        self, outputs: dict, sample: dict, model, device,
        writer, tag: str, sample_idx: int, global_step: int,
        pred_latent=None, target_latent=None, modality: str = "",
    ):
        """Log all generation quality metrics for a single sample."""
        t = f"{tag}/{sample_idx}"
        self._log_recurrent_iterations(outputs, writer, t, global_step)
        self._log_thought_convergence(outputs, writer, t, global_step)
        self._log_token_entropy(outputs, writer, t, global_step)
        self._log_modality_timing(outputs, writer, t, global_step)
        self._log_token_repetition(outputs, writer, t, global_step)
        self._log_text_perplexity(outputs, model, device, writer, t, global_step)
        if pred_latent is not None and target_latent is not None:
            self._log_latent_statistics(pred_latent, target_latent, writer, t, global_step, modality)
            self._log_latent_similarity(pred_latent, target_latent, writer, t, global_step, modality)
        elif pred_latent is not None:
            self._log_latent_statistics(pred_latent, None, writer, t, global_step, modality)

    # --- Individual metric loggers ---

    def _log_recurrent_iterations(self, outputs: dict, writer, tag: str, global_step: int):
        """Log recurrent iteration count statistics from generate() outputs."""
        iter_counts = outputs.get("recurrent_iteration_counts")
        if not iter_counts:
            return
        counts = torch.tensor(iter_counts, dtype=torch.float32)
        writer.add_scalar(f"{tag}/recurrent_iters_mean", counts.mean().item(), global_step)
        writer.add_scalar(f"{tag}/recurrent_iters_min", counts.min().item(), global_step)
        writer.add_scalar(f"{tag}/recurrent_iters_max", counts.max().item(), global_step)
        writer.add_histogram(f"{tag}/recurrent_iters", counts, global_step)
        prompt_iters = outputs.get("prompt_recurrent_iterations")
        if prompt_iters is not None:
            writer.add_scalar(f"{tag}/recurrent_iters_prompt", prompt_iters, global_step)

    def _log_thought_convergence(self, outputs: dict, writer, tag: str, global_step: int):
        """Log per-token KL divergence between final consecutive thought states.

        Shows how converged the recurrent thinking was for each generated token.
        Lower = more converged. Spikes indicate tokens the model "thought harder" about.
        """
        kl_finals = outputs.get("recurrent_kl_final")
        if not kl_finals:
            return
        kl_tensor = torch.tensor(kl_finals, dtype=torch.float32)
        # Filter out inf/nan from potential numerical issues
        valid = kl_tensor[torch.isfinite(kl_tensor)]
        if valid.numel() == 0:
            return
        writer.add_scalar(f"{tag}/thought_kl_mean", valid.mean().item(), global_step)
        writer.add_scalar(f"{tag}/thought_kl_max", valid.max().item(), global_step)
        writer.add_histogram(f"{tag}/thought_kl", valid, global_step)

        # Also log prompt KL trace if available
        prompt_kls = outputs.get("prompt_recurrent_kl")
        if prompt_kls:
            prompt_kl = torch.tensor(prompt_kls, dtype=torch.float32)
            prompt_valid = prompt_kl[torch.isfinite(prompt_kl)]
            if prompt_valid.numel() > 0:
                writer.add_scalar(f"{tag}/thought_kl_prompt_final", prompt_valid[-1].item(), global_step)

    def _log_token_entropy(self, outputs: dict, writer, tag: str, global_step: int):
        """Log entropy of the softmax distribution at each generated token.

        High entropy = model is uncertain. Low entropy = confident prediction.
        Trending downward during training indicates the model is learning.
        """
        logits = outputs.get("text_logits")
        if logits is None:
            return
        # logits: (batch, seq, vocab) — use batch 0
        logits_b = logits[0].float().cpu()  # (seq, vocab)
        probs = F.softmax(logits_b, dim=-1)
        log_probs = F.log_softmax(logits_b, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (seq,)
        writer.add_scalar(f"{tag}/token_entropy_mean", entropy.mean().item(), global_step)
        writer.add_scalar(f"{tag}/token_entropy_min", entropy.min().item(), global_step)
        writer.add_scalar(f"{tag}/token_entropy_max", entropy.max().item(), global_step)
        writer.add_histogram(f"{tag}/token_entropy", entropy, global_step)

    def _log_modality_timing(self, outputs: dict, writer, tag: str, global_step: int):
        """Log position of first BO*/EO* tokens in generated sequence.

        Tracks when the model decides to begin/end media generation.
        """
        gen_ids = outputs.get("generated_token_ids")
        if gen_ids is None:
            return
        ids = gen_ids[0].tolist()  # batch 0
        modality_tokens = {
            "BOV": BOV_TOKEN_ID, "EOV": EOV_TOKEN_ID,
            "BOI": BOI_TOKEN_ID, "EOI": EOI_TOKEN_ID,
            "BOA": BOA_TOKEN_ID, "EOA": EOA_TOKEN_ID,
        }
        for name, tid in modality_tokens.items():
            if tid in ids:
                writer.add_scalar(f"{tag}/first_{name}_position", ids.index(tid), global_step)

    def _log_token_repetition(self, outputs: dict, writer, tag: str, global_step: int):
        """Log fraction of generated tokens that repeat the previous token.

        High repetition rate indicates degenerate looping behavior.
        """
        gen_ids = outputs.get("generated_token_ids")
        if gen_ids is None:
            return
        ids = gen_ids[0].tolist()  # batch 0
        if len(ids) < 2:
            return
        repeats = sum(1 for j in range(1, len(ids)) if ids[j] == ids[j - 1])
        rate = repeats / (len(ids) - 1)
        writer.add_scalar(f"{tag}/token_repetition_rate", rate, global_step)

        # Also log longest consecutive repeat streak
        max_streak = 0
        current_streak = 0
        for j in range(1, len(ids)):
            if ids[j] == ids[j - 1]:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        writer.add_scalar(f"{tag}/token_max_repeat_streak", max_streak, global_step)

    def _log_text_perplexity(
        self, outputs: dict, model, device, writer, tag: str, global_step: int,
    ):
        """Log perplexity of generated text by feeding it back through the model.

        Lower perplexity = model finds its own output more probable = higher quality.
        """
        gen_ids = outputs.get("generated_token_ids")
        if gen_ids is None or gen_ids.shape[1] < 2:
            return
        try:
            # Use generated tokens as input, compute cross-entropy on shifted targets
            input_ids = gen_ids[:, :-1].to(device)  # (1, seq-1)
            targets = gen_ids[:, 1:].to(device)  # (1, seq-1)

            # Run through text feature extractor + recurrent block + text coda
            text_hidden = model.text_feature_extractor(input_ids)
            recurrent_out, _, _, _ = model.recurrent_block(text_hidden)
            text_out = model.text_generator(recurrent_out)
            logits = text_out["logits"]  # (1, seq-1, vocab)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,  # ignore padding
            )
            perplexity = torch.exp(loss).item()
            if math.isfinite(perplexity):
                writer.add_scalar(f"{tag}/text_perplexity", perplexity, global_step)
        except Exception as e:
            print(f"Warning: perplexity computation failed: {e}")

    def _log_latent_statistics(
        self, pred_latent: torch.Tensor, target_latent: Optional[torch.Tensor],
        writer, tag: str, global_step: int, modality: str,
    ):
        """Log mean/std/min/max of generated (and target) latents.

        Detects mode collapse (low std) or latent explosion (extreme values).
        """
        prefix = f"{tag}/{modality}_latent" if modality else f"{tag}/latent"
        pred = pred_latent.float().cpu()
        writer.add_scalar(f"{prefix}_pred_mean", pred.mean().item(), global_step)
        writer.add_scalar(f"{prefix}_pred_std", pred.std().item(), global_step)
        writer.add_scalar(f"{prefix}_pred_min", pred.min().item(), global_step)
        writer.add_scalar(f"{prefix}_pred_max", pred.max().item(), global_step)

        if target_latent is not None:
            tgt = target_latent.float().cpu()
            writer.add_scalar(f"{prefix}_target_mean", tgt.mean().item(), global_step)
            writer.add_scalar(f"{prefix}_target_std", tgt.std().item(), global_step)
            writer.add_scalar(f"{prefix}_target_min", tgt.min().item(), global_step)
            writer.add_scalar(f"{prefix}_target_max", tgt.max().item(), global_step)

    def _log_latent_similarity(
        self, pred_latent: torch.Tensor, target_latent: torch.Tensor,
        writer, tag: str, global_step: int, modality: str,
    ):
        """Log cosine similarity between generated and target latents.

        Simple proxy for reconstruction quality. 1.0 = perfect match, 0.0 = orthogonal.
        """
        prefix = f"{tag}/{modality}_latent" if modality else f"{tag}/latent"
        pred_flat = pred_latent.float().cpu().flatten()
        tgt_flat = target_latent.float().cpu().flatten()
        # Truncate to same length if needed
        min_len = min(pred_flat.shape[0], tgt_flat.shape[0])
        pred_flat = pred_flat[:min_len]
        tgt_flat = tgt_flat[:min_len]
        cos_sim = F.cosine_similarity(pred_flat.unsqueeze(0), tgt_flat.unsqueeze(0)).item()
        writer.add_scalar(f"{prefix}_cosine_sim", cos_sim, global_step)

    def _scenario_train_reconstruction(self, model, args, device, writer, global_step, dtype):
        """Run forward pass on fixed training samples and compare predictions to ground truth.

        This tracks memorization/overfitting: as training progresses, the model's
        predictions on its own training data should increasingly match the targets.
        Uses the same fixed sample indices every eval for consistent comparison.
        """
        tag = "train_world/reconstruction"
        train_dataset = self.trainer.train_dataset
        if train_dataset is None or len(train_dataset) == 0:
            return

        collator = self.trainer.data_collator
        n = min(self.num_eval_samples, len(train_dataset))

        # Fixed indices — same samples every eval for consistent tracking
        if not hasattr(self, '_train_recon_indices'):
            gen = torch.Generator().manual_seed(42)
            self._train_recon_indices = torch.randperm(len(train_dataset), generator=gen)[:n].tolist()

        try:
            samples = [train_dataset[i] for i in self._train_recon_indices]
        except Exception as e:
            print(f"Warning: Failed to load training samples for reconstruction: {e}")
            return

        # Collate into a batch
        batch = collator(samples)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # --- Forward pass (mirrors compute_loss logic) ---
        from utils.constants import (
            AUDIO_PLACEHOLDER_TOKEN_ID as APH,
            VOICE_PLACEHOLDER_TOKEN_ID as VPH,
            IMAGE_PLACEHOLDER_TOKEN_ID as IPH,
        )
        placeholder_ids = {APH, VPH, IPH}

        text_input_ids = batch.get("text_token_ids")
        text_targets = None
        if text_input_ids is not None:
            full_ids = text_input_ids
            non_ph_mask_full = torch.ones_like(full_ids, dtype=torch.bool)
            for pid in placeholder_ids:
                non_ph_mask_full &= (full_ids != pid)

            model_text_ids = full_ids[:, :-1].contiguous()

            non_ph_input = torch.ones_like(model_text_ids, dtype=torch.bool)
            for pid in placeholder_ids:
                non_ph_input &= (model_text_ids != pid)

            target_list = []
            for b in range(full_ids.shape[0]):
                clean = full_ids[b][non_ph_mask_full[b]]
                shifted = clean[1:]
                K = non_ph_input[b].sum().item()
                if shifted.shape[0] >= K:
                    target_list.append(shifted[:K])
                else:
                    target_list.append(torch.cat([shifted, shifted.new_zeros(K - shifted.shape[0])]))

            max_len = max(t.shape[0] for t in target_list)
            padded = [torch.cat([t, t.new_zeros(max_len - t.shape[0])]) if t.shape[0] < max_len else t for t in target_list]
            text_targets = torch.stack(padded)
            text_input_ids = model_text_ids

        # Media inputs
        audio_inputs, audio_lengths, audio_labels = None, None, None
        audio_data = batch.get("audio_features")
        if audio_data is not None:
            audio_inputs = audio_data.unsqueeze(1)
            audio_labels = audio_data.clone()
            fl = batch.get("audio_feature_lengths")
            if fl is not None:
                audio_lengths = fl.unsqueeze(1)

        voice_inputs, voice_lengths, voice_labels = None, None, None
        voice_data = batch.get("voice_features")
        if voice_data is not None:
            voice_inputs = voice_data.unsqueeze(1)
            voice_labels = voice_data.clone()
            fl = batch.get("voice_feature_lengths")
            if fl is not None:
                voice_lengths = fl.unsqueeze(1)

        image_inputs, image_labels = None, None
        image_data = batch.get("image_images")
        if image_data is not None:
            image_inputs = image_data.unsqueeze(1)
            image_labels = image_data.clone()

        outputs = model(
            text_input_ids=text_input_ids,
            audio_inputs=audio_inputs,
            audio_lengths=audio_lengths,
            voice_inputs=voice_inputs,
            voice_lengths=voice_lengths,
            image_inputs=image_inputs,
            precomputed_latents=True,
            decode_outputs=False,
        )

        # --- Log text reconstruction quality ---
        logits = outputs.get("logits")
        if logits is not None and text_targets is not None:
            B, T, V = logits.size()
            T_min = min(T, text_targets.shape[1])
            logits_aligned = logits[:, :T_min, :].contiguous()
            targets_aligned = text_targets[:, :T_min].contiguous()

            # Per-sample accuracy and loss
            preds = logits_aligned.argmax(dim=-1)  # [B, T]
            for i in range(min(n, B)):
                # Mask out padding (target == 0)
                valid = targets_aligned[i] != 0
                if valid.sum() == 0:
                    continue
                correct = (preds[i][valid] == targets_aligned[i][valid]).float().mean().item()
                writer.add_scalar(f"{tag}/text_accuracy/{i}", correct, global_step)

                # Log predicted vs target text
                pred_text = self._decode_tokens(preds[i][valid])
                target_text = self._decode_tokens(targets_aligned[i][valid])
                writer.add_text(f"{tag}/text/{i}/predicted", pred_text, global_step)
                writer.add_text(f"{tag}/text/{i}/target", target_text, global_step)

            # Batch-level CE loss
            ce_loss = F.cross_entropy(
                logits_aligned.reshape(-1, V), targets_aligned.reshape(-1), ignore_index=0,
            )
            writer.add_scalar(f"{tag}/text_ce_loss", ce_loss.item(), global_step)
            if ce_loss.item() < 20:
                writer.add_scalar(f"{tag}/text_perplexity", torch.exp(ce_loss).item(), global_step)

        # --- Log voice reconstruction quality ---
        voice_preds = outputs.get("voice_latent_preds")
        if voice_preds is not None and voice_labels is not None:
            v_l1 = F.l1_loss(voice_preds, voice_labels).item()
            v_mse = F.mse_loss(voice_preds, voice_labels).item()
            writer.add_scalar(f"{tag}/voice_latent_l1", v_l1, global_step)
            writer.add_scalar(f"{tag}/voice_latent_mse", v_mse, global_step)

            for i in range(min(n, voice_preds.shape[0])):
                pred_lat = voice_preds[i]  # (C, T)
                tgt_lat = voice_labels[i]  # (C, T)
                writer.add_image(f"{tag}/voice/{i}/predicted", self._latent_to_image(pred_lat), global_step)
                writer.add_image(f"{tag}/voice/{i}/target", self._latent_to_image(tgt_lat), global_step)

                cos = F.cosine_similarity(pred_lat.flatten().unsqueeze(0), tgt_lat.flatten().unsqueeze(0)).item()
                writer.add_scalar(f"{tag}/voice_cosine_sim/{i}", cos, global_step)

                # Decode predicted voice to audio if CVAE available
                sample = samples[i] if i < len(samples) else {}
                self._log_audio_with_cvae(pred_lat, sample, writer, global_step, f"{tag}/voice/{i}/pred")

            # Also decode target voice for comparison (first sample only)
            if len(samples) > 0:
                self._log_audio_with_cvae(voice_labels[0], samples[0], writer, global_step, f"{tag}/voice/0/target")

        # --- Log audio reconstruction quality ---
        audio_preds = outputs.get("audio_latent_preds")
        if audio_preds is not None and audio_labels is not None:
            a_l1 = F.l1_loss(audio_preds, audio_labels).item()
            a_mse = F.mse_loss(audio_preds, audio_labels).item()
            writer.add_scalar(f"{tag}/audio_latent_l1", a_l1, global_step)
            writer.add_scalar(f"{tag}/audio_latent_mse", a_mse, global_step)

            for i in range(min(n, audio_preds.shape[0])):
                writer.add_image(f"{tag}/audio/{i}/predicted", self._latent_to_image(audio_preds[i]), global_step)
                writer.add_image(f"{tag}/audio/{i}/target", self._latent_to_image(audio_labels[i]), global_step)

        # --- Log image reconstruction quality ---
        image_preds = outputs.get("image_latent_preds")
        if image_preds is not None and image_labels is not None:
            i_l1 = F.l1_loss(image_preds, image_labels).item()
            i_mse = F.mse_loss(image_preds, image_labels).item()
            writer.add_scalar(f"{tag}/image_latent_l1", i_l1, global_step)
            writer.add_scalar(f"{tag}/image_latent_mse", i_mse, global_step)

            for i in range(min(n, image_preds.shape[0])):
                writer.add_image(f"{tag}/image/{i}/predicted", self._latent_to_image(image_preds[i]), global_step)
                writer.add_image(f"{tag}/image/{i}/target", self._latent_to_image(image_labels[i]), global_step)

                cos = F.cosine_similarity(image_preds[i].flatten().unsqueeze(0), image_labels[i].flatten().unsqueeze(0)).item()
                writer.add_scalar(f"{tag}/image_cosine_sim/{i}", cos, global_step)

                # Decode through image VAE if available
                if self.image_vae_decoder is not None:
                    self._try_decode_image(image_preds[i], writer, global_step, f"{tag}/image/{i}/pred_decoded")
                    self._try_decode_image(image_labels[i], writer, global_step, f"{tag}/image/{i}/target_decoded")

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

            # Use first half as prompt, cap so total doesn't exceed MAX_SEQ_LEN
            prompt_len = max(1, text_length // 2)
            max_new = min(256, text_length)
            # Ensure prompt + generation fits
            prompt_len = min(prompt_len, self.MAX_SEQ_LEN - max_new)
            prompt_len = max(1, prompt_len)
            prompt = token_ids[:prompt_len].unsqueeze(0).to(device)

            outputs = model.generate(
                text_input_ids=prompt,
                max_new_tokens=max_new,
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
            self._log_generation_metrics(
                outputs, sample, model, device, writer, tag, i, global_step,
            )

    def _scenario_text_to_voice(self, model, eval_dataset, collator, device, writer, global_step):
        """Scenario 2: Text -> Voice synthesis using static prompts."""
        tag = "eval_world/2_text_to_voice"

        # Use static prompts instead of dataset text (avoids markup/JS garbage)
        n = min(self.num_eval_samples, len(self.VOICE_SYNTHESIS_PROMPTS))
        # Still need samples for target voice comparison and speaker embeddings
        samples = self._get_eval_samples(
            eval_dataset, collator, n, requires_voice=True
        )

        for i in range(n):
            prompt_text = self.VOICE_SYNTHESIS_PROMPTS[i]
            max_new = 512
            prompt = self._encode_static_prompt(prompt_text, [BOV_TOKEN_ID], max_new, device)

            outputs = model.generate(
                text_input_ids=prompt,
                max_new_tokens=max_new,
                temperature=0.8,
            )

            writer.add_text(f"{tag}/{i}/input_text", prompt_text, global_step)
            sample = samples[i] if i < len(samples) else {}

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

            # Log target voice for comparison
            target_features = sample.get("voice_features")
            if target_features is not None:
                writer.add_image(
                    f"{tag}/{i}/target_latent",
                    self._latent_to_image(target_features),
                    global_step,
                )

            voice_preds = outputs.get("voice_latent_preds")
            gen_latent = voice_preds[0, 0] if voice_preds is not None and voice_preds.numel() > 0 else None
            self._log_generation_metrics(
                outputs, sample, model, device, writer, tag, i, global_step,
                pred_latent=gen_latent, target_latent=target_features, modality="voice",
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
            self._log_generation_metrics(
                outputs, sample, model, device, writer, tag, i, global_step,
            )

    def _scenario_text_to_image(self, model, eval_dataset, collator, device, writer, global_step):
        """Scenario 4: Text -> Image synthesis using static prompts."""
        tag = "eval_world/4_text_to_image"

        # Use static prompts instead of dataset text (avoids markup/JS garbage)
        n = min(self.num_eval_samples, len(self.IMAGE_SYNTHESIS_PROMPTS))
        # Still need samples for target image comparison
        samples = self._get_eval_samples(
            eval_dataset, collator, n, requires_image=True
        )

        for i in range(n):
            prompt_text = self.IMAGE_SYNTHESIS_PROMPTS[i]
            max_new = 512
            prompt = self._encode_static_prompt(prompt_text, [BOI_TOKEN_ID], max_new, device)

            outputs = model.generate(
                text_input_ids=prompt,
                max_new_tokens=max_new,
                temperature=0.8,
            )

            writer.add_text(f"{tag}/{i}/input_text", prompt_text, global_step)
            sample = samples[i] if i < len(samples) else {}

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

            image_preds_all = outputs.get("image_latent_preds")
            gen_latent = image_preds_all[0, 0] if image_preds_all is not None and image_preds_all.numel() > 0 else None
            target_image_latent = sample.get("image_images")
            if target_image_latent is None:
                target_image_latent = sample.get("image_image")
            self._log_generation_metrics(
                outputs, sample, model, device, writer, tag, i, global_step,
                pred_latent=gen_latent, target_latent=target_image_latent, modality="image",
            )

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
            self._log_generation_metrics(
                outputs, sample, model, device, writer, tag, i, global_step,
            )

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

            img_preds = outputs.get("image_latent_preds")
            gen_img = img_preds[0, 0] if img_preds is not None and img_preds.numel() > 0 else None
            self._log_generation_metrics(
                outputs, sample, model, device, writer, tag, i, global_step,
                pred_latent=gen_img, modality="image",
            )

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

            voice_preds = outputs.get("voice_latent_preds")
            gen_voice = voice_preds[0, 0] if voice_preds is not None and voice_preds.numel() > 0 else None
            self._log_generation_metrics(
                outputs, sample, model, device, writer, tag, i, global_step,
                pred_latent=gen_voice, modality="voice",
            )

    # --- Helper methods ---

    def _latent_to_image(self, latent: torch.Tensor) -> np.ndarray:
        """Convert a latent tensor to a grid of per-channel grayscale images.

        Each channel is individually normalized to [0, 1] and arranged in a grid.
        Returns (1, grid_H, grid_W) for TensorBoard add_image (grayscale).
        """

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
