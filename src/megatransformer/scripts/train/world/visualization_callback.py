from typing import List, Optional

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from transformers import Trainer

from matplotlib import pyplot as plt

from megatransformer.scripts.train.visualization_callback import VisualizationCallback
from megatransformer.utils import metrics
from megatransformer.utils import visualization
from megatransformer.utils import constants


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
        voice_smg_decoder: Optional[torch.nn.Module] = None,
        voice_cosyvoice2_decoder: Optional[torch.nn.Module] = None,
        static_speaker_embedding: Optional[torch.Tensor] = None,
        num_eval_samples: int = 4,
        step_offset: int = 0,
        voice_sample_rate: int = 16000,
        voice_n_mels: int = 80,
        voice_n_fft: int = 1024,
        voice_hop_length: int = 256,
        voice_temperature: float = 0.6,
        voice_variance_floor: float = 0.0,
        voice_ras_win: int = 0,
        voice_ras_tau: float = 0.1,
        include_modes: Optional[list[str]] = None,
        include_tasks: Optional[list[str]] = None,
        voice_token_budget: Optional[int] = None,
        audio_token_budget: Optional[int] = None,
        special_token_base: int = constants.SPECIAL_TOKEN_BASE,
        tokenizer_name: str = "mistralai/Mistral-7B-v0.1",
    ):
        # How many content frames generate() may emit before force-closing with EO*.
        # None => keep generate()'s own default, which is a hardcoded 209 (SIVE
        # @hop256/stride3) and truncates a 50 Hz ContentVec render to ~4.2s.
        self.voice_token_budget = voice_token_budget
        self.audio_token_budget = audio_token_budget
        # What this run actually trains. Scenarios outside it are skipped rather than
        # emitting empty/meaningless panels — a text->voice run has no business rendering
        # transcription or cross-modal examples. None = no restriction (all tasks).
        self.include_modes = set(include_modes) if include_modes else {"text", "audio", "voice", "image"}
        self.include_tasks = set(include_tasks) if include_tasks else None
        # Control-token ids + text vocab boundary for THIS model (base 32000 Mistral default, or the
        # pretrained LLM's native vocab). Used to inject BO*/EO*/placeholder tokens into generation
        # prompts and to strip control tokens when decoding -- must match the model + the data.
        self._sp = constants.special_token_ids(special_token_base)
        self._special_base = int(special_token_base)
        self._tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.vocoder = vocoder
        self.image_vae_decoder = image_vae_decoder
        self.voice_smg_decoder = voice_smg_decoder
        # Frozen CosyVoice 2 flow+HiFT. When present it TAKES PRECEDENCE over the SMG path:
        # a CosyVoice 2 run's voice coda emits unit ids over CosyVoice's codebook, which the
        # SMG (trained on Mimi cb0 latents) cannot decode.
        self.voice_cosyvoice2_decoder = voice_cosyvoice2_decoder
        self.static_speaker_embedding = static_speaker_embedding
        # Lazily pinned in _resolve_static_speaker when no explicit static embedding was
        # given, so the TTS renders still have a voice that is constant across evals.
        self._pinned_speaker = None
        self._warned_no_smg = False
        self._warned_no_units = False
        self.num_eval_samples = num_eval_samples
        self.step_offset = step_offset if step_offset is not None else 0
        # Voice sampling for TB eval renders. Only bites when the model was trained with
        # --voice_stochastic_output (heteroscedastic coda) — else logvar is None and generate()
        # ignores it (deterministic mu). 0 = deterministic; ~0.5-0.7 = moderate stochasticity.
        self.voice_temperature = voice_temperature
        self.voice_ras_win = int(voice_ras_win or 0)
        self.voice_ras_tau = float(voice_ras_tau)
        self.voice_variance_floor = voice_variance_floor
        self.voice_sample_rate = voice_sample_rate
        self.voice_n_mels = voice_n_mels
        self.voice_n_fft = voice_n_fft
        self.voice_hop_length = voice_hop_length

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

    def _vocode(self, mel):
        """Vocode a mel, resampling its time axis to the vocoder's frame rate when the
        SMG mel hop (voice_hop_length) differs from the vocoder's — e.g. a 50 Hz
        ContentVec mel @hop320 driving a 62.5 Hz @hop256 vocoder. No-op when equal."""
        return visualization.render_vocoder_audio(
            self.vocoder, mel, mel_hop_length=self.voice_hop_length,
        )

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
            self.tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")

    def _decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text, filtering out special tokens."""
        self._ensure_tokenizer()
        if self.tokenizer is None:
            return f"[token_ids: {token_ids.tolist()[:20]}...]"
        # Filter out control tokens (>= the text-vocab boundary for this model)
        text_ids = [t for t in token_ids.tolist() if t < self._special_base]
        return self.tokenizer.decode(text_ids, skip_special_tokens=True)

    def _get_eval_samples(self, eval_dataset, collator, n: int,
                          requires_audio=False, requires_voice=False, requires_image=False,
                          requires_text_only=False):
        """Get n random samples from eval dataset that match modality requirements."""
        # Early-out: if a required modality isn't present in the dataset at all,
        # no index can satisfy the filter and the loop below would walk the
        # entire eval set — every eval, for every absent-modality scenario — for
        # nothing. (e.g. running --include_modes text,voice still calls the
        # text_to_image / image_to_text scenarios.)
        mods = getattr(eval_dataset, "modalities", None)
        if mods is not None:
            if (requires_audio and "audio" not in mods) or \
               (requires_voice and "voice" not in mods) or \
               (requires_image and "image" not in mods):
                return []
        # Bound the scan so a sparse-but-present modality can't degrade into a
        # full-eval sweep. Visit at most `budget` indices, in shard (sorted)
        # order so the dataset's LRU shard cache loads each shard at most once.
        budget = min(len(eval_dataset), max(n * 50, 1000))
        indices = torch.randperm(len(eval_dataset))[:budget].sort().values
        samples = []
        for idx in indices:
            sample = eval_dataset[idx.item()]
            if requires_text_only and sample.get("_modality") != "text":
                continue
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

    def _generate(self, model, **kwargs):
        """self._generate(model, ) with THIS run's media budgets applied.

        Wrapping rather than passing the budgets at each of the ~17 call sites: a site
        that forgot them would silently truncate its renders, which is exactly the failure
        being fixed. Explicit kwargs still win, so a caller can override per-scenario.
        """
        if self.voice_token_budget is not None:
            kwargs.setdefault("voice_token_budget", self.voice_token_budget)
        if self.audio_token_budget is not None:
            kwargs.setdefault("audio_token_budget", self.audio_token_budget)
        # Repetition-aware sampling for the RENDERS only — this changes nothing about
        # training, just what the eval audio sounds like. Measured at distill@44k:
        # adj_repeat 0.1034 -> 0.0071 (teacher 0.0070) and longest_run 56 -> 7, i.e. it
        # removes the repeated-frame artifacts that otherwise mask everything else by ear.
        if self.voice_ras_win:
            kwargs.setdefault("voice_ras_win", self.voice_ras_win)
            kwargs.setdefault("voice_ras_tau", self.voice_ras_tau)
        return model.generate(**kwargs)

    def _scenario_enabled(self, required_modes: set, satisfying_tasks: set) -> bool:
        """Should this scenario run, given --include_modes / --include_tasks?

        Every required mode must be enabled. If the scenario names tasks, at least one
        must be enabled — include_tasks=None means "all tasks", so nothing is gated out.
        Cross-modal scenarios name no tasks and are gated by their modes alone.
        """
        if not required_modes.issubset(self.include_modes):
            return False
        if self.include_tasks is not None:
            # A named --include_tasks gates task-agnostic (cross-modal) scenarios OUT too: if
            # the user asked for specific tasks, cross-modal round-trips aren't among them and
            # are just noise (e.g. voice_to_voice under a voice_synthesis-only run).
            if not satisfying_tasks:
                return False
            return bool(satisfying_tasks & self.include_tasks)
        return True

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        global_step = state.global_step + self.step_offset

        if not state.is_world_process_zero:
            return

        logger = metrics.get_logger()
        if logger is None:
            print("No metrics logger found, skipping world model visualization...")
            return

        eval_dataset = self.trainer.eval_dataset
        if eval_dataset is None or len(eval_dataset) == 0:
            print("No eval dataset available, skipping visualization...")
            return

        collator = self.trainer.data_collator
        device = self._get_device()
        model.eval()

        dtype = torch.bfloat16 if bool(args.bf16) else torch.float16 if args.fp16 else torch.float32

        # (method, required modes, satisfying tasks). A scenario runs only if every
        # required mode is enabled AND at least one satisfying task is enabled. Empty
        # task set = the scenario is task-agnostic (mode gating alone decides).
        train_data_scenarios = [
            (self._scenario_train_text_continuation, {"text"}, {"text_continuation"}),
            (self._scenario_train_generation, set(), {"voice_synthesis", "audio_synthesis", "image_synthesis"}),
            (self._scenario_train_transcription, set(), {"voice_transcription", "audio_transcription", "image_transcription"}),
        ]
        # Cross-modal round-trips media through the model, so it needs at least two media
        # modalities to mean anything — with only voice it just renders voice_to_voice.
        # Not expressible as a required-modes subset, hence the explicit count.
        if len(self.include_modes & {"audio", "voice", "image"}) >= 2:
            train_data_scenarios.append((self._scenario_train_cross_modal, set(), set()))
        eval_scenarios = [
            (self._scenario_text_continuation, {"text"}, {"text_continuation"}),
            (self._scenario_text_to_voice, {"voice"}, {"voice_synthesis"}),
            (self._scenario_voice_to_text, {"voice"}, {"voice_transcription"}),
            (self._scenario_text_to_image, {"image"}, {"image_synthesis"}),
            (self._scenario_image_to_text, {"image"}, {"image_transcription"}),
            (self._scenario_voice_to_image, {"voice", "image"}, set()),
            (self._scenario_image_to_voice, {"voice", "image"}, set()),
        ]

        selected = [fn.__name__ for fn, m, t in train_data_scenarios + eval_scenarios
                    if self._scenario_enabled(m, t)]
        print(f"Running world model visualization at step {global_step}... "
              f"({len(selected)} applicable scenarios: {', '.join(s.replace('_scenario_', '') for s in selected)})")

        with torch.no_grad():
            with autocast(device.type, dtype=dtype, enabled=args.bf16 or args.fp16):
                for scenario_fn, modes, tasks in train_data_scenarios:
                    if not self._scenario_enabled(modes, tasks):
                        continue
                    try:
                        scenario_fn(model, args, device, global_step, dtype)
                    except Exception as e:
                        import traceback
                        print(f"Warning: Train-data scenario {scenario_fn.__name__} failed: {e}")
                        traceback.print_exc()

                # Eval scenarios — generation and transcription with eval dataset
                for scenario_fn, modes, tasks in eval_scenarios:
                    if not self._scenario_enabled(modes, tasks):
                        continue
                    try:
                        scenario_fn(model, eval_dataset, collator, device, global_step)
                    except Exception as e:
                        import traceback
                        print(f"Warning: Eval scenario {scenario_fn.__name__} failed: {e}")
                        traceback.print_exc()

        print(f"World model visualization complete at step {global_step}")
        metrics.flush()

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
        tag: str, sample_idx: int, global_step: int,
        pred_latent=None, target_latent=None, modality: str = "",
    ):
        """Log all generation quality metrics for a single sample."""
        t = f"{tag}/{sample_idx}"
        self._log_recurrent_iterations(outputs, t, global_step)
        self._log_thought_convergence(outputs, t, global_step)
        self._log_token_entropy(outputs, t, global_step)
        self._log_modality_timing(outputs, t, global_step)
        self._log_token_repetition(outputs, t, global_step)
        self._log_text_perplexity(outputs, model, device, t, global_step)
        if pred_latent is not None and target_latent is not None:
            self._log_latent_statistics(pred_latent, target_latent, t, global_step, modality)
            self._log_latent_similarity(pred_latent, target_latent, t, global_step, modality)
        elif pred_latent is not None:
            self._log_latent_statistics(pred_latent, None, t, global_step, modality)

    # --- Individual metric loggers ---

    def _log_recurrent_iterations(self, outputs: dict, tag: str, global_step: int):
        """Log recurrent iteration count statistics from generate() outputs."""
        iter_counts = outputs.get("recurrent_iteration_counts")
        if not iter_counts:
            return
        counts = torch.tensor(iter_counts, dtype=torch.float32)
        metrics.log_scalar(f"{tag}/recurrent_iters_mean", counts.mean().item(), global_step)
        metrics.log_scalar(f"{tag}/recurrent_iters_min", counts.min().item(), global_step)
        metrics.log_scalar(f"{tag}/recurrent_iters_max", counts.max().item(), global_step)
        try:
            if counts.numel() > 1:
                metrics.log_histogram(f"{tag}/recurrent_iters", counts, global_step)
        except ValueError:
            pass
        prompt_iters = outputs.get("prompt_recurrent_iterations")
        if prompt_iters is not None:
            metrics.log_scalar(f"{tag}/recurrent_iters_prompt", prompt_iters, global_step)

    def _log_thought_convergence(self, outputs: dict, tag: str, global_step: int):
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
        metrics.log_scalar(f"{tag}/thought_kl_mean", valid.mean().item(), global_step)
        metrics.log_scalar(f"{tag}/thought_kl_max", valid.max().item(), global_step)
        try:
            if valid.numel() > 1:
                metrics.log_histogram(f"{tag}/thought_kl", valid, global_step)
        except ValueError:
            pass

        # Also log prompt KL trace if available
        prompt_kls = outputs.get("prompt_recurrent_kl")
        if prompt_kls:
            prompt_kl = torch.tensor(prompt_kls, dtype=torch.float32)
            prompt_valid = prompt_kl[torch.isfinite(prompt_kl)]
            if prompt_valid.numel() > 0:
                metrics.log_scalar(f"{tag}/thought_kl_prompt_final", prompt_valid[-1].item(), global_step)

    def _log_token_entropy(self, outputs: dict, tag: str, global_step: int):
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
        metrics.log_scalar(f"{tag}/token_entropy_mean", entropy.mean().item(), global_step)
        metrics.log_scalar(f"{tag}/token_entropy_min", entropy.min().item(), global_step)
        metrics.log_scalar(f"{tag}/token_entropy_max", entropy.max().item(), global_step)
        try:
            if entropy.numel() > 1:
                metrics.log_histogram(f"{tag}/token_entropy", entropy, global_step)
        except ValueError:
            pass

    def _log_modality_timing(self, outputs: dict, tag: str, global_step: int):
        """Log position of first BO*/EO* tokens in generated sequence.

        Tracks when the model decides to begin/end media generation.
        """
        gen_ids = outputs.get("generated_token_ids")
        if gen_ids is None:
            return
        ids = gen_ids[0].tolist()  # batch 0
        modality_tokens = {
            "BOV": self._sp.BOV, "EOV": self._sp.EOV,
            "BOI": self._sp.BOI, "EOI": self._sp.EOI,
            "BOA": self._sp.BOA, "EOA": self._sp.EOA,
        }
        for name, tid in modality_tokens.items():
            if tid in ids:
                metrics.log_scalar(f"{tag}/first_{name}_position", ids.index(tid), global_step)

    def _log_token_repetition(self, outputs: dict, tag: str, global_step: int):
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
        metrics.log_scalar(f"{tag}/token_repetition_rate", rate, global_step)

        # Also log longest consecutive repeat streak
        max_streak = 0
        current_streak = 0
        for j in range(1, len(ids)):
            if ids[j] == ids[j - 1]:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        metrics.log_scalar(f"{tag}/token_max_repeat_streak", max_streak, global_step)

    def _log_text_perplexity(
        self, outputs: dict, model, device, tag: str, global_step: int,
    ):
        """Log SELF-perplexity: score the model's own generation back through the model.

        Lower = the model finds its own output more probable. This is a fluency proxy and
        is NOT comparable to eval/{task}/text_loss, which is cross-entropy against ground
        truth — hence the distinct `text_self_perplexity` tag. Note it also reaches into
        submodules directly rather than calling forward(), so it bypasses the interleaver.
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
            recurrent_out, _, _, _, _ = model.recurrent_block(text_hidden * model.embed_scale)
            text_out = model.text_generator(recurrent_out)
            logits = text_out["logits"]  # (1, seq-1, vocab)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,  # ignore padding
            )
            perplexity = torch.exp(loss).item()
            if math.isfinite(perplexity):
                metrics.log_scalar(f"{tag}/text_self_perplexity", perplexity, global_step)
        except Exception as e:
            print(f"Warning: perplexity computation failed: {e}")

    def _log_latent_statistics(
        self, pred_latent: torch.Tensor, target_latent: Optional[torch.Tensor],
        tag: str, global_step: int, modality: str,
    ):
        """Log mean/std/min/max of generated (and target) latents.

        Detects mode collapse (low std) or latent explosion (extreme values).
        """
        prefix = f"{tag}/{modality}_latent" if modality else f"{tag}/latent"
        pred = pred_latent.float().cpu()
        metrics.log_scalar(f"{prefix}_pred_mean", pred.mean().item(), global_step)
        metrics.log_scalar(f"{prefix}_pred_std", pred.std().item(), global_step)
        metrics.log_scalar(f"{prefix}_pred_min", pred.min().item(), global_step)
        metrics.log_scalar(f"{prefix}_pred_max", pred.max().item(), global_step)

        if target_latent is not None:
            tgt = target_latent.float().cpu()
            metrics.log_scalar(f"{prefix}_target_mean", tgt.mean().item(), global_step)
            metrics.log_scalar(f"{prefix}_target_std", tgt.std().item(), global_step)
            metrics.log_scalar(f"{prefix}_target_min", tgt.min().item(), global_step)
            metrics.log_scalar(f"{prefix}_target_max", tgt.max().item(), global_step)

    def _log_latent_similarity(
        self, pred_latent: torch.Tensor, target_latent: torch.Tensor,
        tag: str, global_step: int, modality: str,
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
        metrics.log_scalar(f"{prefix}_cosine_sim", cos_sim, global_step)

    def _scenario_train_reconstruction(self, model, args, device, global_step, dtype):
        """Run forward pass on fixed training samples and compare predictions to ground truth.

        This tracks memorization/overfitting: as training progresses, the model's
        predictions on its own training data should increasingly match the targets.
        Uses the same fixed sample indices every eval for consistent comparison.
        """
        tag = "train_data/reconstruction"
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
        placeholder_ids = {self._sp.AUDIO_PLACEHOLDER, self._sp.VOICE_PLACEHOLDER, self._sp.IMAGE_PLACEHOLDER}

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
                metrics.log_scalar(f"{tag}/{i}/text_accuracy", correct, global_step)

                # Log predicted vs target text
                pred_text = self._decode_tokens(preds[i][valid])
                target_text = self._decode_tokens(targets_aligned[i][valid])
                metrics.log_text(f"{tag}/text/{i}/predicted", pred_text, global_step)
                metrics.log_text(f"{tag}/text/{i}/target", target_text, global_step)

            # Batch-level CE loss
            ce_loss = F.cross_entropy(
                logits_aligned.reshape(-1, V), targets_aligned.reshape(-1), ignore_index=0,
            )
            metrics.log_scalar(f"{tag}/text_ce_loss", ce_loss.item(), global_step)
            if ce_loss.item() < 20:
                metrics.log_scalar(f"{tag}/text_perplexity", torch.exp(ce_loss).item(), global_step)

        # --- Log voice reconstruction quality ---
        voice_preds = outputs.get("voice_latent_preds")
        if voice_preds is not None and voice_labels is not None:
            # On the deduped path voice_latent_preds is segment-rate (and untrained -- the
            # unit head owns the loss), while voice_labels stay 50Hz, so the time dims
            # differ. Align to the common prefix so l1/mse/cosine don't crash; the numbers
            # are a rough proxy either way (and noise on the units path).
            if voice_preds.shape[-1] != voice_labels.shape[-1]:
                Tc = min(voice_preds.shape[-1], voice_labels.shape[-1])
                voice_preds = voice_preds[..., :Tc]
                voice_labels = voice_labels[..., :Tc]
            v_l1 = F.l1_loss(voice_preds, voice_labels).item()
            v_mse = F.mse_loss(voice_preds, voice_labels).item()
            metrics.log_scalar(f"{tag}/voice_latent_l1_loss", v_l1, global_step)
            metrics.log_scalar(f"{tag}/voice_latent_mse_loss", v_mse, global_step)

            for i in range(min(n, voice_preds.shape[0])):
                pred_lat = voice_preds[i]  # (C, T)
                tgt_lat = voice_labels[i]  # (C, T)
                metrics.log_image(f"{tag}/voice/{i}/predicted", self._latent_to_image(pred_lat), global_step)
                metrics.log_image(f"{tag}/voice/{i}/target", self._latent_to_image(tgt_lat), global_step)

                cos = F.cosine_similarity(pred_lat.flatten().unsqueeze(0), tgt_lat.flatten().unsqueeze(0)).item()
                metrics.log_scalar(f"{tag}/{i}/voice_cosine_sim", cos, global_step)

                # Decode predicted voice to audio if SMG available
                sample = samples[i] if i < len(samples) else {}
                # Teacher-forced, so the discrete prediction is the unit head's argmax at each
                # position. Trim to the utterance's true length: past it the inputs are padding,
                # so those "predictions" are noise that the decoder would render as babble.
                self._log_audio_with_smg(
                    pred_lat, sample, global_step, f"{tag}/voice/{i}/pred",
                    unit_ids=self._generated_unit_ids(outputs, idx=i),
                    unit_length=self._batch_voice_length(batch, i),
                )

            # Also decode target voice for comparison (first sample only)
            if len(samples) > 0:
                self._log_audio_with_smg(
                    voice_labels[0], samples[0], global_step, f"{tag}/voice/0/target",
                    unit_ids=(batch.get("voice_unit_ids")[0]
                              if batch.get("voice_unit_ids") is not None else None),
                    unit_length=self._batch_voice_length(batch, 0),
                )

        # --- Log audio reconstruction quality ---
        audio_preds = outputs.get("audio_latent_preds")
        if audio_preds is not None and audio_labels is not None:
            a_l1 = F.l1_loss(audio_preds, audio_labels).item()
            a_mse = F.mse_loss(audio_preds, audio_labels).item()
            metrics.log_scalar(f"{tag}/audio_latent_l1_loss", a_l1, global_step)
            metrics.log_scalar(f"{tag}/audio_latent_mse_loss", a_mse, global_step)

            for i in range(min(n, audio_preds.shape[0])):
                metrics.log_image(f"{tag}/audio/{i}/predicted", self._latent_to_image(audio_preds[i]), global_step)
                metrics.log_image(f"{tag}/audio/{i}/target", self._latent_to_image(audio_labels[i]), global_step)

        # --- Log image reconstruction quality ---
        image_preds = outputs.get("image_latent_preds")
        if image_preds is not None and image_labels is not None:
            i_l1 = F.l1_loss(image_preds, image_labels).item()
            i_mse = F.mse_loss(image_preds, image_labels).item()
            metrics.log_scalar(f"{tag}/image_latent_l1_loss", i_l1, global_step)
            metrics.log_scalar(f"{tag}/image_latent_mse_loss", i_mse, global_step)

            for i in range(min(n, image_preds.shape[0])):
                metrics.log_image(f"{tag}/image/{i}/predicted", self._latent_to_image(image_preds[i]), global_step)
                metrics.log_image(f"{tag}/image/{i}/target", self._latent_to_image(image_labels[i]), global_step)

                cos = F.cosine_similarity(image_preds[i].flatten().unsqueeze(0), image_labels[i].flatten().unsqueeze(0)).item()
                metrics.log_scalar(f"{tag}/{i}/image_cosine_sim", cos, global_step)

                # Decode through image VAE if available
                if self.image_vae_decoder is not None:
                    self._try_decode_image(image_preds[i], global_step, f"{tag}/image/{i}/pred_decoded")
                    self._try_decode_image(image_labels[i], global_step, f"{tag}/image/{i}/target_decoded")

    def _scenario_train_text_continuation(self, model, args, device, global_step, dtype):
        """Complete memorized text from training samples.

        Finds text-only samples (text_continuation tasks) from the training
        dataset, slices each partway through, and has the model generate the
        rest. Logs the generated continuation alongside the target text so
        memorization quality can be assessed at a glance.
        """
        tag = "train_data/text_continuation"
        train_dataset = self.trainer.train_dataset
        if train_dataset is None or len(train_dataset) == 0:
            return

        n = min(self.num_eval_samples, len(train_dataset))

        # Find text-only indices (text_continuation tasks), cached
        if not hasattr(self, '_train_text_indices'):
            tasks = train_dataset.task_types if hasattr(train_dataset, 'task_types') else []
            n_tasks = len(tasks) if tasks else 1
            text_indices = []
            for idx in range(len(train_dataset)):
                task_idx = idx % n_tasks
                if tasks and tasks[task_idx][1] == "text":
                    text_indices.append(idx)
                    if len(text_indices) >= n:
                        break
            self._train_text_indices = text_indices

        for i, idx in enumerate(self._train_text_indices):
            try:
                sample = train_dataset[idx]
                token_ids = sample.get("text_token_ids")
                if token_ids is None:
                    continue

                text_length = sample.get("text_text_length", token_ids.shape[0])
                if isinstance(text_length, torch.Tensor):
                    text_length = text_length.item()

                # Use first half as prompt
                prompt_len = max(1, text_length // 2)
                max_new = min(256, text_length)
                prompt_len = min(prompt_len, 1024 - max_new)
                prompt_len = max(1, prompt_len)
                prompt = token_ids[:prompt_len].unsqueeze(0).to(device)

                outputs = self._generate(model, 
                    text_input_ids=prompt,
                    max_new_tokens=max_new,
                    temperature=0.8,
                voice_temperature=self.voice_temperature,
                voice_variance_floor=self.voice_variance_floor,
                    top_p=0.9,
                )

                gen_ids = outputs.get("generated_token_ids")
                if gen_ids is not None:
                    input_text = self._decode_tokens(token_ids[:prompt_len])
                    gen_text = self._decode_tokens(gen_ids[0])
                    full_target = self._decode_tokens(token_ids[:text_length])

                    metrics.log_text(f"{tag}/{i}/generated", gen_text[:500], global_step, context={
                        "prompt": input_text[:500],
                        "target": full_target[:500],
                    })
            except Exception as e:
                print(f"Warning: Train text continuation failed for sample {i}: {e}")

    def _scenario_train_generation(self, model, args, device, global_step, dtype):
        """Generate media autoregressively from training sample text prompts.

        Selects fixed voice and image samples (by modality) from the training
        dataset. For each, provides only the text prompt and generates the
        media from scratch. This is the honest measure of generation quality.
        """
        tag = "train_data/generation"
        train_dataset = self.trainer.train_dataset
        if train_dataset is None or len(train_dataset) == 0:
            return

        collator = self.trainer.data_collator
        n = min(self.num_eval_samples, len(train_dataset))

        # Select fixed per-modality indices (computed once, cached)
        if not hasattr(self, '_train_gen_voice_indices') or not hasattr(self, '_train_gen_image_indices'):
            # Find indices for each modality type by inspecting actual samples.
            # Only hunt for a modality that's actually present, and bound the
            # scan — otherwise an absent/sparse modality turns this into a full
            # sweep of the (large) train set looking for samples that can't or
            # rarely exist. Quotas for absent modalities are 0 so the break
            # below fires without scanning further.
            mods = getattr(train_dataset, "modalities", None)
            want_voice = n if (mods is None or "voice" in mods) else 0
            want_image = n if (mods is None or "image" in mods) else 0
            budget = min(len(train_dataset), max(n * 50, 2000))
            voice_indices = []
            image_indices = []
            seen_image_hashes = set()
            seen_voice_hashes = set()
            for idx in range(budget):
                try:
                    sample = train_dataset[idx]
                except Exception:
                    continue
                mod = sample.get("_modality", None)
                has_voice = mod == "voice" or any(k.startswith("voice_") for k in sample)
                has_image = mod == "image" or "image_image" in sample
                # Skip duplicates by hashing the media content
                if has_voice and len(voice_indices) < n:
                    feat = sample.get("voice_features")
                    h = hash(feat.data_ptr()) if feat is not None else idx
                    if feat is not None:
                        h = feat.flatten()[:16].sum().item()  # rough content hash
                    if h not in seen_voice_hashes:
                        seen_voice_hashes.add(h)
                        voice_indices.append(idx)
                elif has_image and len(image_indices) < n:
                    img = sample.get("image_image")
                    h = img.flatten()[:16].sum().item() if img is not None else idx
                    if h not in seen_image_hashes:
                        seen_image_hashes.add(h)
                        image_indices.append(idx)
                if len(voice_indices) >= want_voice and len(image_indices) >= want_image:
                    break
            self._train_gen_voice_indices = voice_indices
            self._train_gen_image_indices = image_indices
            print(f"[gen_debug] Found {len(voice_indices)} unique voice, {len(image_indices)} unique image")


        # Force synthesis direction
        prev_direction = getattr(collator, 'force_direction', None)
        collator.force_direction = "synthesis"

        # Voice generation from voice samples
        for i, idx in enumerate(self._train_gen_voice_indices):
            try:
                sample = train_dataset[idx]
                voice_batch = collator([sample])
                text_ids = voice_batch["text_token_ids"][0]
                bov_positions = (text_ids == self._sp.BOV).nonzero(as_tuple=True)[0]
                if len(bov_positions) == 0:
                    continue

                bov_pos = bov_positions[0].item()
                prompt = text_ids[:bov_pos + 1].unsqueeze(0).to(device)

                # Log prompt text (decode from token_ids if raw text not available)
                prompt_text = sample.get("text_text", sample.get("voice_voice_text", ""))
                if not prompt_text and "text_token_ids" in sample:
                    prompt_text = self._decode_tokens(sample["text_token_ids"])
                decoded = self._decode_tokens(text_ids[:bov_pos + 1])

                outputs = self._generate(model, 
                    text_input_ids=prompt, max_new_tokens=512, temperature=0.8,
                    voice_temperature=self.voice_temperature, voice_variance_floor=self.voice_variance_floor,
                )

                # Log the generated text alongside the media so memorization
                # tests can verify what text the model produced for the prompt.
                gen_ids = outputs.get("generated_token_ids")
                if gen_ids is not None:
                    gen_text = self._decode_tokens(gen_ids[0])
                    target_text_full = sample.get("text_text", sample.get("voice_voice_text", ""))
                    if isinstance(target_text_full, list):
                        target_text_full = target_text_full[0] if target_text_full else ""
                    if not target_text_full and "text_token_ids" in sample:
                        target_text_full = self._decode_tokens(sample["text_token_ids"])
                    ctx = {"prompt": str(prompt_text)[:500] if prompt_text else decoded[:500]}
                    if target_text_full:
                        ctx["target"] = str(target_text_full)[:500]
                    # Only log the generated TEXT when text generation is an enabled task. Under
                    # a voice_synthesis-only run text loss is masked, so any emitted text is OOD
                    # noise (the prompt/target context above is still useful and rides on the
                    # audio/mel tags).
                    if self.include_tasks is None or "text_continuation" in self.include_tasks:
                        metrics.log_text(f"{tag}/voice/{i}/generated_text", gen_text[:500], global_step, context=ctx)

                voice_preds = outputs.get("voice_latent_preds")
                if voice_preds is not None and voice_preds.numel() > 0:
                    pred_lat = voice_preds[0, 0]
                    metrics.log_image(f"{tag}/voice/{i}/generated", self._latent_to_image(pred_lat), global_step, context={
                        "prompt": str(prompt_text)[:500] if prompt_text else decoded[:500],
                    })

                    tgt_lat = voice_batch.get("voice_features")
                    if tgt_lat is not None:
                        tgt_lat = tgt_lat[0]
                        metrics.log_image(f"{tag}/voice/{i}/target", self._latent_to_image(tgt_lat), global_step)
                        # Free-running generation almost never matches the target frame count
                        # (the stop head decides length; deduped durations vary it further),
                        # so flatten() over the full tensors mismatches. Compare over the
                        # common prefix — a rough similarity proxy anyway, since generation is
                        # not time-aligned to the target.
                        Tc = min(pred_lat.shape[-1], tgt_lat.shape[-1])
                        cos = F.cosine_similarity(
                            pred_lat[..., :Tc].flatten().unsqueeze(0),
                            tgt_lat[..., :Tc].flatten().to(pred_lat.device).unsqueeze(0),
                        ).item()
                        metrics.log_scalar(f"{tag}/{i}/voice_cosine_sim", cos, global_step)
                        # Cached target units are 0-PADDED to the batch's frame count and 0 is a
                        # legal unit, so the length must come from voice_feature_lengths or the
                        # decoder renders the padding as breathy non-speech.
                        tgt_units = voice_batch.get("voice_unit_ids")
                        self._log_audio_with_smg(
                            tgt_lat, sample, global_step, f"{tag}/voice/{i}/target",
                            unit_ids=(tgt_units[0] if tgt_units is not None else None),
                            unit_length=self._batch_voice_length(voice_batch, 0),
                        )

                    # Generated audio must render the world model's PREDICTED F0 contour, not
                    # the sample's GT contour — that's the prosody the AR test is judging.
                    gen_f0 = outputs.get("voice_f0_preds")
                    # Free-running trace: already exact (EOV stripped by generate()), so NO
                    # length trim — its length IS the signal this scenario exists to show.
                    self._log_audio_with_smg(
                        pred_lat, sample, global_step, f"{tag}/voice/{i}/generated",
                        f0_contour=(gen_f0[0, 0] if gen_f0 is not None and gen_f0.numel() > 0 else None),
                        unit_ids=self._generated_unit_ids(outputs),
                    )
            except Exception as e:
                print(f"Warning: Train generation (voice) failed for sample {i}: {e}")

        # Image generation from image samples
        print(f"[gen_debug] image indices: {self._train_gen_image_indices}")
        for i, idx in enumerate(self._train_gen_image_indices):
            try:
                sample = train_dataset[idx]
                print(f"[gen_debug] image sample {i} keys: {sorted(sample.keys())}")
                image_batch = collator([sample])
                print(f"[gen_debug] image batch keys: {sorted(image_batch.keys())}")
                text_ids = image_batch["text_token_ids"][0]
                boi_positions = (text_ids == self._sp.BOI).nonzero(as_tuple=True)[0]
                print(f"[gen_debug] BOI positions: {boi_positions.tolist()}, text_ids shape: {text_ids.shape}")
                if len(boi_positions) == 0:
                    print(f"[gen_debug] No BOI found, skipping")
                    continue

                boi_pos = boi_positions[0].item()
                prompt = text_ids[:boi_pos + 1].unsqueeze(0).to(device)

                # Log prompt text
                prompt_text = sample.get("text_text", sample.get("image_text", ""))
                if not prompt_text and "text_token_ids" in sample:
                    prompt_text = self._decode_tokens(sample["text_token_ids"])
                decoded = self._decode_tokens(text_ids[:boi_pos + 1])

                outputs = self._generate(model, 
                    text_input_ids=prompt, max_new_tokens=512, temperature=0.8,
                    voice_temperature=self.voice_temperature, voice_variance_floor=self.voice_variance_floor,
                )

                # Log the generated text alongside the media.
                gen_ids = outputs.get("generated_token_ids")
                if gen_ids is not None:
                    gen_text = self._decode_tokens(gen_ids[0])
                    target_text_full = sample.get("text_text", sample.get("image_text", ""))
                    if isinstance(target_text_full, list):
                        target_text_full = target_text_full[0] if target_text_full else ""
                    if not target_text_full and "text_token_ids" in sample:
                        target_text_full = self._decode_tokens(sample["text_token_ids"])
                    ctx = {"prompt": str(prompt_text)[:500] if prompt_text else decoded[:500]}
                    if target_text_full:
                        ctx["target"] = str(target_text_full)[:500]
                    metrics.log_text(f"{tag}/image/{i}/generated_text", gen_text[:500], global_step, context=ctx)

                image_preds = outputs.get("image_latent_preds")
                if image_preds is not None and image_preds.numel() > 0:
                    pred_lat = image_preds[0, 0]
                    metrics.log_image(f"{tag}/image/{i}/generated", self._latent_to_image(pred_lat), global_step, context={
                        "prompt": str(prompt_text)[:500] if prompt_text else decoded[:500],
                    })

                    tgt_lat = image_batch.get("image_images")
                    if tgt_lat is not None:
                        tgt_lat = tgt_lat[0]
                        metrics.log_image(f"{tag}/image/{i}/target", self._latent_to_image(tgt_lat), global_step)
                        cos = F.cosine_similarity(pred_lat.flatten().unsqueeze(0), tgt_lat.flatten().to(pred_lat.device).unsqueeze(0)).item()
                        metrics.log_scalar(f"{tag}/{i}/image_cosine_sim", cos, global_step)

                    if self.image_vae_decoder is not None:
                        self._try_decode_image(pred_lat, global_step, f"{tag}/image/{i}/generated_decoded")
                        if tgt_lat is not None:
                            self._try_decode_image(tgt_lat, global_step, f"{tag}/image/{i}/target_decoded")
            except Exception as e:
                print(f"Warning: Train generation (image) failed for sample {i}: {e}")

        collator.force_direction = prev_direction

    def _scenario_train_transcription(self, model, args, device, global_step, dtype):
        """Transcribe/describe training media: provide voice/image, generate text.

        Uses the same per-modality indices as _scenario_train_generation.
        For voice: provides voice features as input, generates text transcription.
        For image: provides image as input, generates text description.
        """
        tag = "train_data/transcription"
        train_dataset = self.trainer.train_dataset
        if train_dataset is None or len(train_dataset) == 0:
            return

        collator = self.trainer.data_collator
        n = min(self.num_eval_samples, len(train_dataset))

        # Reuse per-modality indices from generation (ensures we get actual voice/image samples)
        if not hasattr(self, '_train_gen_voice_indices') or not hasattr(self, '_train_gen_image_indices'):
            print(f"[transcription_debug] Indices not populated, skipping")
            return
        print(f"[transcription_debug] voice_indices={self._train_gen_voice_indices}, image_indices={self._train_gen_image_indices}")


        # Voice transcription
        for i, idx in enumerate(self._train_gen_voice_indices):
            try:
                sample = train_dataset[idx]
                voice_features = sample.get("voice_features")
                if voice_features is None:
                    continue

                # Build transcription prompt: [BOV] [VOICE_PH] [EOV]
                # The prelude processes the voice, it gets interleaved, then the
                # model generates text after EOV.
                prompt = torch.tensor(
                    [[self._sp.BOV, self._sp.VOICE_PLACEHOLDER, self._sp.EOV]],
                    dtype=torch.long, device=device,
                )
                voice_input = voice_features.unsqueeze(0).unsqueeze(0).to(device)
                voice_len = torch.tensor([[voice_features.shape[-1]]], device=device)

                outputs = self._generate(model, 
                    text_input_ids=prompt,
                    max_new_tokens=256,
                    temperature=0.8,
                voice_temperature=self.voice_temperature,
                voice_variance_floor=self.voice_variance_floor,
                    voice_inputs=voice_input,
                    voice_lengths=voice_len,
                    precomputed_latents=True,
                )

                gen_ids = outputs.get("generated_token_ids")

                # Log target text (decode from token_ids if raw text not available)
                target_text = sample.get("text_text", sample.get("voice_voice_text", ""))
                if isinstance(target_text, list):
                    target_text = target_text[0] if target_text else ""
                if not target_text and "text_token_ids" in sample:
                    target_text = self._decode_tokens(sample["text_token_ids"])

                if gen_ids is not None:
                    gen_text = self._decode_tokens(gen_ids[0])
                    target_ctx = {}
                    if target_text:
                        target_ctx["target"] = str(target_text)[:500]
                    metrics.log_text(f"{tag}/voice/{i}/transcription", gen_text[:500], global_step, context=target_ctx)

                # Log input audio
                self._log_audio_with_smg(
                    voice_features, sample, global_step,
                    f"{tag}/voice/{i}/input"
                )
            except Exception as e:
                print(f"Warning: Train transcription (voice) failed for sample {i}: {e}")

        # Image description
        for i, idx in enumerate(self._train_gen_image_indices):
            try:
                sample = train_dataset[idx]
                image_data = sample.get("image_image")
                if image_data is None:
                    continue

                # Build description prompt: [BOI] [IMAGE_PH] [EOI]
                prompt = torch.tensor(
                    [[self._sp.BOI, self._sp.IMAGE_PLACEHOLDER, self._sp.EOI]],
                    dtype=torch.long, device=device,
                )
                image_input = image_data.unsqueeze(0).unsqueeze(0).to(device)

                outputs = self._generate(model, 
                    text_input_ids=prompt,
                    max_new_tokens=256,
                    temperature=0.8,
                voice_temperature=self.voice_temperature,
                voice_variance_floor=self.voice_variance_floor,
                    image_inputs=image_input,
                    precomputed_latents=True,
                )

                gen_ids = outputs.get("generated_token_ids")

                # Log target text (decode from token_ids if raw text not available)
                target_text = sample.get("text_text", sample.get("image_text", ""))
                if isinstance(target_text, list):
                    target_text = target_text[0] if target_text else ""
                if not target_text and "text_token_ids" in sample:
                    target_text = self._decode_tokens(sample["text_token_ids"])

                if gen_ids is not None:
                    gen_text = self._decode_tokens(gen_ids[0])
                    target_ctx = {}
                    if target_text:
                        target_ctx["target"] = str(target_text)[:500]
                    metrics.log_text(f"{tag}/image/{i}/description", gen_text[:500], global_step, context=target_ctx)

                # Log input image
                if self.image_vae_decoder is not None:
                    self._try_decode_image(
                        image_data, global_step,
                        f"{tag}/image/{i}/input_image"
                    )
                metrics.log_image(
                    f"{tag}/image/{i}/input_latent",
                    self._latent_to_image(image_data),
                    global_step,
                )
            except Exception as e:
                print(f"Warning: Train transcription (image) failed for sample {i}: {e}")

    def _scenario_train_cross_modal(self, model, args, device, global_step, dtype):
        """Consecutive same-modality generation: input one example, generate another.

        Tests whether the model produces coherent output for a second consecutive
        media example of the same modality:
        - Voice→Voice: [BOV] [VOICE_PH] [EOV] [BOV] → generate voice
        - Image→Image: [BOI] [IMAGE_PH] [EOI] [BOI] → generate image
        """
        tag = "train_data/cross_modal"
        train_dataset = self.trainer.train_dataset
        if train_dataset is None or len(train_dataset) == 0:
            return

        if not hasattr(self, '_train_gen_voice_indices') or not hasattr(self, '_train_gen_image_indices'):
            return


        # Voice→Voice: provide voice input, generate a second voice clip
        for i, idx in enumerate(self._train_gen_voice_indices):
            try:
                sample = train_dataset[idx]
                voice_features = sample.get("voice_features")
                if voice_features is None:
                    continue

                # Prompt: [BOV] [VOICE_PH] [EOV] [BOV]
                prompt = torch.tensor(
                    [[self._sp.BOV, self._sp.VOICE_PLACEHOLDER, self._sp.EOV, self._sp.BOV]],
                    dtype=torch.long, device=device,
                )
                voice_input = voice_features.unsqueeze(0).unsqueeze(0).to(device)
                voice_len = torch.tensor([[voice_features.shape[-1]]], device=device)

                outputs = self._generate(model, 
                    text_input_ids=prompt,
                    max_new_tokens=512,
                    temperature=0.8,
                voice_temperature=self.voice_temperature,
                voice_variance_floor=self.voice_variance_floor,
                    voice_inputs=voice_input,
                    voice_lengths=voice_len,
                    precomputed_latents=True,
                )

                # Log input voice
                metrics.log_image(
                    f"{tag}/voice_to_voice/{i}/input_latent",
                    self._latent_to_image(voice_features),
                    global_step,
                )
                self._log_audio_with_smg(
                    voice_features, sample, global_step,
                    f"{tag}/voice_to_voice/{i}/input"
                )

                # Log generated voice
                voice_preds = outputs.get("voice_latent_preds")
                if voice_preds is not None and voice_preds.numel() > 0:
                    pred_latent = voice_preds[0, 0]
                    metrics.log_image(
                        f"{tag}/voice_to_voice/{i}/generated_latent",
                        self._latent_to_image(pred_latent),
                        global_step,
                    )
                    self._log_audio_with_smg(
                        pred_latent, sample, global_step,
                        f"{tag}/voice_to_voice/{i}/generated"
                    )

                # Log any generated text (should be minimal/empty if model learned EOS)
                gen_ids = outputs.get("generated_token_ids")
                if gen_ids is not None:
                    gen_text = self._decode_tokens(gen_ids[0])
                    if gen_text.strip():
                        metrics.log_text(f"{tag}/voice_to_voice/{i}/extra_text", gen_text[:500], global_step)
            except Exception as e:
                print(f"Warning: Train cross-modal (voice→voice) failed for sample {i}: {e}")

        # Image→Image: provide image input, generate a second image
        for i, idx in enumerate(self._train_gen_image_indices):
            try:
                sample = train_dataset[idx]
                image_data = sample.get("image_image")
                if image_data is None:
                    continue

                # Prompt: [BOI] [IMAGE_PH] [EOI] [BOI]
                prompt = torch.tensor(
                    [[self._sp.BOI, self._sp.IMAGE_PLACEHOLDER, self._sp.EOI, self._sp.BOI]],
                    dtype=torch.long, device=device,
                )
                image_input = image_data.unsqueeze(0).unsqueeze(0).to(device)

                outputs = self._generate(model, 
                    text_input_ids=prompt,
                    max_new_tokens=512,
                    temperature=0.8,
                voice_temperature=self.voice_temperature,
                voice_variance_floor=self.voice_variance_floor,
                    image_inputs=image_input,
                    precomputed_latents=True,
                )

                # Log input image
                metrics.log_image(
                    f"{tag}/image_to_image/{i}/input_latent",
                    self._latent_to_image(image_data),
                    global_step,
                )
                if self.image_vae_decoder is not None:
                    self._try_decode_image(
                        image_data, global_step,
                        f"{tag}/image_to_image/{i}/input_image"
                    )

                # Log generated image
                image_preds = outputs.get("image_latent_preds")
                if image_preds is not None and image_preds.numel() > 0:
                    pred_latent = image_preds[0, 0]
                    metrics.log_image(
                        f"{tag}/image_to_image/{i}/generated_latent",
                        self._latent_to_image(pred_latent),
                        global_step,
                    )
                    if self.image_vae_decoder is not None:
                        self._try_decode_image(
                            pred_latent, global_step,
                            f"{tag}/image_to_image/{i}/generated_image"
                        )

                # Log any generated text
                gen_ids = outputs.get("generated_token_ids")
                if gen_ids is not None:
                    gen_text = self._decode_tokens(gen_ids[0])
                    if gen_text.strip():
                        metrics.log_text(f"{tag}/image_to_image/{i}/extra_text", gen_text[:500], global_step)
            except Exception as e:
                print(f"Warning: Train cross-modal (image→image) failed for sample {i}: {e}")

    def _scenario_text_continuation(self, model, eval_dataset, collator, device, global_step):
        """Scenario 1: Text-only generation (take text, generate continuation)."""
        tag = "text_continuation"
        samples = self._get_eval_samples(eval_dataset, collator, self.num_eval_samples, requires_text_only=True)
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

            outputs = self._generate(model, 
                text_input_ids=prompt,
                max_new_tokens=max_new,
                temperature=0.8,
                voice_temperature=self.voice_temperature,
                voice_variance_floor=self.voice_variance_floor,
                top_p=0.9,
            )

            gen_ids = outputs["generated_token_ids"][0]
            input_text = self._decode_tokens(token_ids[:prompt_len])
            gen_text = self._decode_tokens(gen_ids)
            full_target = self._decode_tokens(token_ids[:text_length])

            metrics.log_text(f"{tag}/{i}/generated", gen_text, global_step, context={
                "input": input_text,
                "target": full_target,
            })
            self._log_generation_metrics(
                outputs, sample, model, device, tag, i, global_step,
            )

    def _scenario_text_to_voice(self, model, eval_dataset, collator, device, global_step):
        """Scenario 2: Text -> Voice synthesis using dataset captions.

        Uses the actual transcript from each eval sample as the prompt, so the
        target voice matches the text the model is conditioned on.
        """
        tag = "text_to_voice"

        samples = self._get_eval_samples(
            eval_dataset, collator, self.num_eval_samples, requires_voice=True
        )
        if not samples:
            return

        for i, sample in enumerate(samples):
            # Get transcript from dataset sample
            prompt_text = sample.get("text_text", "")
            if not prompt_text and "text_token_ids" in sample:
                prompt_text = self._decode_tokens(sample["text_token_ids"])
            if not prompt_text:
                continue

            max_new = 512
            prompt = self._encode_static_prompt(str(prompt_text)[:500], [self._sp.BOV], max_new, device)

            outputs = self._generate(model, 
                text_input_ids=prompt,
                max_new_tokens=max_new,
                temperature=0.8,
                voice_temperature=self.voice_temperature,
                voice_variance_floor=self.voice_variance_floor,
            )

            # Log generated voice if available
            voice_preds = outputs.get("voice_latent_preds")
            if voice_preds is not None and voice_preds.numel() > 0:
                pred_latent = voice_preds[0, 0]  # (C, T)
                metrics.log_image(
                    f"{tag}/{i}/generated_latent",
                    self._latent_to_image(pred_latent),
                    global_step,
                    context={"prompt": prompt_text},
                )

                # The coda's own speaker-normalized contour when it has an F0 head; None
                # falls back to the SMG predicting F0 from the units.
                f0_p = outputs.get("voice_f0_preds")
                self._log_audio_with_smg(
                    pred_latent, sample, global_step, f"{tag}/{i}",
                    f0_contour=(f0_p[0, 0] if f0_p is not None and f0_p.numel() > 0 else None),
                    unit_ids=self._generated_unit_ids(outputs),
                )

            # Log target voice for comparison
            target_features = sample.get("voice_features")
            if target_features is not None:
                metrics.log_image(
                    f"{tag}/{i}/target_latent",
                    self._latent_to_image(target_features),
                    global_step,
                )
                # Ground-truth audio: on the discrete path the cache stores the units directly,
                # so the target render is the frozen decoder's own reconstruction = the ceiling
                # this run's generated audio is being listened to against. CosyVoice-only —
                # the SMG path never logged a target here and stays byte-identical.
                if self.voice_cosyvoice2_decoder is not None:
                    self._log_audio_with_cosyvoice2(
                        sample.get("voice_unit_ids"), sample, global_step, f"{tag}/{i}/target",
                        length=sample.get("voice_feature_length"),   # cached units are 0-padded
                    )

            voice_preds = outputs.get("voice_latent_preds")
            gen_latent = voice_preds[0, 0] if voice_preds is not None and voice_preds.numel() > 0 else None
            self._log_generation_metrics(
                outputs, sample, model, device, tag, i, global_step,
                pred_latent=gen_latent, target_latent=target_features, modality="voice",
            )

    def _prepare_audio_for_generate(self, sample, device):
        """Prepare SIVE audio features from a sample for self._generate(model, ).

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
        """Prepare SIVE voice features from a sample for self._generate(model, ).

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
        """Prepare image from a sample for self._generate(model, ).

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

    def _scenario_voice_to_text(self, model, eval_dataset, collator, device, global_step):
        """Scenario 3: Voice -> Text transcription."""
        tag = "voice_to_text"
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
            prompt_tokens = [self._sp.BOV, self._sp.VOICE_PLACEHOLDER, self._sp.EOV]
            prompt = self._build_prompt_ids(prompt_tokens, device)

            voice_inputs, voice_lengths = self._prepare_voice_for_generate(sample, device)

            outputs = self._generate(model, 
                text_input_ids=prompt,
                max_new_tokens=256,
                temperature=0.8,
                voice_temperature=self.voice_temperature,
                voice_variance_floor=self.voice_variance_floor,
                top_p=0.9,
                voice_inputs=voice_inputs,
                voice_lengths=voice_lengths,
            )

            gen_ids = outputs["generated_token_ids"][0]
            gen_text = self._decode_tokens(gen_ids)
            target_text = self._decode_tokens(token_ids[:text_length])

            # Log input voice audio alongside generated/target text
            self._log_audio_with_smg(voice_features, sample, global_step, f"{tag}/{i}/input")

            metrics.log_text(f"{tag}/{i}/generated", gen_text, global_step, context={
                "target": target_text,
            })
            self._log_generation_metrics(
                outputs, sample, model, device, tag, i, global_step,
            )

    def _image_gen_is_adapter(self, model):
        """True if the image generator is a conditioning ADAPTER (SDXL or Z-Image): it
        predicts frozen-decoder conditioning, not a latent, so the in-loop latent viz can't
        render it (and its 'target' is a dummy placeholder latent -> solid black). Rendering
        is done by the sidecars scripts/eval/world/eval_{sdxl,zimage}_adapter.py."""
        m = model.module if hasattr(model, "module") else model
        gen = getattr(m, "image_generator", None)
        try:
            from megatransformer.model.image.sdxl_adapter import SDXLConditioningAdapter
            from megatransformer.model.image.zimage_adapter import ZImageConditioningAdapter
            return isinstance(gen, (SDXLConditioningAdapter, ZImageConditioningAdapter))
        except Exception:
            return False

    def _render_sdxl_adapter(self, model, eval_dataset, collator, device, global_step, tag):
        """Opt-in in-loop image viz for the SDXL adapter (IMAGE_EVAL_RENDER_SDXL=1).

        Loads the FULL SDXL pipeline (~5GB UNet + VAE + gen activations) *alongside* the
        resident training state (optimizer/grads/master weights — HF doesn't offload them
        at eval). May OOM; that's the point of the flag. The pipe is loaded and FREED each
        eval so it never permanently starves training memory, and OOM is caught so the run
        survives and falls back to the sidecar (eval_sdxl_adapter.py).
        """
        import numpy as np
        pipe = None
        try:
            torch.cuda.empty_cache()
            from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DPMSolverMultistepScheduler
            print("  [viz] IMAGE_EVAL_RENDER_SDXL=1: loading SDXL for in-loop render...")
            _vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", vae=_vae, torch_dtype=torch.float16,
                use_safetensors=True).to(device)
            # DPM++ 2M Karras: project-default SDXL sampler (crisper than stock Euler).
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
            pipe.set_progress_bar_config(disable=True)
            neg_pe, _, neg_pp, _ = pipe.encode_prompt(prompt="", device=device,
                                                      num_images_per_prompt=1, do_classifier_free_guidance=False)
            n = min(self.num_eval_samples, 4)
            samples = self._get_eval_samples(eval_dataset, collator, n, requires_image=True)
            unwrapped = model.module if hasattr(model, "module") else model

            def _render(seq, pool, seed):
                g = torch.Generator(device=device).manual_seed(seed)
                return pipe(prompt_embeds=seq.half(), pooled_prompt_embeds=pool.half(),
                            negative_prompt_embeds=neg_pe.half(), negative_pooled_prompt_embeds=neg_pp.half(),
                            num_inference_steps=20, guidance_scale=7.0, height=1024, width=1024,
                            generator=g).images[0]

            def _chw(pil):
                return np.asarray(pil).astype(np.float32).transpose(2, 0, 1) / 255.0

            for i, sample in enumerate(samples):
                batch = collator([sample])
                caption = (batch.get("text_texts") or [""])[0] or ""
                with torch.no_grad():
                    out = unwrapped(text_input_ids=batch["text_token_ids"].to(device),
                                    image_inputs=batch["image_images"].unsqueeze(1).to(device),
                                    precomputed_latents=True,
                                    is_synthesis=batch["is_synthesis"].to(device),
                                    decode_outputs=False)
                    seq, pool = out.get("image_clip_seq_pred"), out.get("image_clip_pooled_pred")
                    if seq is None:
                        continue
                    seq_t, _, pool_t, _ = pipe.encode_prompt(prompt=caption, device=device,
                                                             num_images_per_prompt=1, do_classifier_free_guidance=False)
                    gen = _render(seq[:1], pool[:1], 1000 + i)
                    tgt = _render(seq_t, pool_t, 1000 + i)
                metrics.log_image(f"{tag}/image/{i}/generated", _chw(gen), global_step, context={"prompt": caption[:500]})
                metrics.log_image(f"{tag}/image/{i}/target", _chw(tgt), global_step, context={"prompt": caption[:500]})
            metrics.flush()
            print(f"  [viz] in-loop SDXL render OK ({n} samples).")
        except torch.cuda.OutOfMemoryError as e:
            print(f"  [viz] in-loop SDXL render OOM'd (training state + SDXL exceed VRAM): "
                  f"{str(e)[:100]}. Use the sidecar (eval_sdxl_adapter.py) instead.")
        except Exception as e:
            print(f"  [viz] in-loop SDXL render failed: {type(e).__name__}: {str(e)[:150]}")
        finally:
            del pipe
            torch.cuda.empty_cache()

    def _scenario_text_to_image(self, model, eval_dataset, collator, device, global_step):
        """Scenario 4: Text -> Image synthesis using dataset captions.

        Uses the actual caption from each eval sample as the prompt, so the
        target image matches the text the model is conditioned on.
        """
        tag = "text_to_image"

        if self._image_gen_is_adapter(model):
            import os
            m = model.module if hasattr(model, "module") else model
            from megatransformer.model.image.sdxl_adapter import SDXLConditioningAdapter
            is_sdxl = isinstance(getattr(m, "image_generator", None), SDXLConditioningAdapter)
            # In-loop render is SDXL-only + opt-in (loading the pipe alongside training may
            # OOM). Z-Image has no in-loop path (20GB) -> always skip; use the sidecar.
            if is_sdxl and os.environ.get("IMAGE_EVAL_RENDER_SDXL"):
                self._render_sdxl_adapter(model, eval_dataset, collator, device, global_step, tag)
            else:
                print("  [viz] image gen = conditioning adapter (SDXL/Z-Image); skipping in-loop "
                      "render (its latent target is a dummy placeholder -> black). Use the sidecar "
                      "eval_sdxl_adapter.py / eval_zimage_adapter.py; SDXL also supports "
                      "IMAGE_EVAL_RENDER_SDXL=1.")
            return

        samples = self._get_eval_samples(
            eval_dataset, collator, self.num_eval_samples, requires_image=True
        )
        if not samples:
            return

        for i, sample in enumerate(samples):
            # Get caption from dataset sample
            prompt_text = sample.get("text_text", "")
            if not prompt_text and "text_token_ids" in sample:
                prompt_text = self._decode_tokens(sample["text_token_ids"])
            if not prompt_text:
                continue

            max_new = 512
            prompt = self._encode_static_prompt(str(prompt_text)[:500], [self._sp.BOI], max_new, device)

            outputs = self._generate(model, 
                text_input_ids=prompt,
                max_new_tokens=max_new,
                temperature=0.8,
                voice_temperature=self.voice_temperature,
                voice_variance_floor=self.voice_variance_floor,
            )

            image_preds = outputs.get("image_latent_preds")
            if image_preds is not None and image_preds.numel() > 0:
                pred_latent = image_preds[0, 0]  # (C, H, W)
                metrics.log_image(
                    f"{tag}/{i}/generated_latent",
                    self._latent_to_image(pred_latent),
                    global_step,
                    context={"prompt": str(prompt_text)[:500]},
                )

                if self.image_vae_decoder is not None:
                    self._try_decode_image(
                        pred_latent, global_step,
                        f"{tag}/{i}/generated_image"
                    )

            # Log target image (matches the caption used as prompt)
            target_image = sample.get("image_images")
            if target_image is None:
                target_image = sample.get("image_image")
            if target_image is not None:
                metrics.log_image(
                    f"{tag}/{i}/target_latent",
                    self._latent_to_image(target_image),
                    global_step,
                )
                if self.image_vae_decoder is not None:
                    self._try_decode_image(
                        target_image, global_step,
                        f"{tag}/{i}/target_image"
                    )

            image_preds_all = outputs.get("image_latent_preds")
            gen_latent = image_preds_all[0, 0] if image_preds_all is not None and image_preds_all.numel() > 0 else None
            target_image_latent = sample.get("image_images")
            if target_image_latent is None:
                target_image_latent = sample.get("image_image")
            self._log_generation_metrics(
                outputs, sample, model, device, tag, i, global_step,
                pred_latent=gen_latent, target_latent=target_image_latent, modality="image",
            )

    def _scenario_image_to_text(self, model, eval_dataset, collator, device, global_step):
        """Scenario 5: Image -> Text description."""
        tag = "image_to_text"
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
            prompt_tokens = [self._sp.BOI, self._sp.IMAGE_PLACEHOLDER, self._sp.EOI]
            prompt = self._build_prompt_ids(prompt_tokens, device)

            image_inputs = self._prepare_image_for_generate(sample, device)

            outputs = self._generate(model, 
                text_input_ids=prompt,
                max_new_tokens=256,
                temperature=0.8,
                voice_temperature=self.voice_temperature,
                voice_variance_floor=self.voice_variance_floor,
                top_p=0.9,
                image_inputs=image_inputs,
            )

            gen_ids = outputs["generated_token_ids"][0]
            gen_text = self._decode_tokens(gen_ids)
            target_text = self._decode_tokens(token_ids[:text_length])

            # Log input image alongside generated/target text
            context = {"target": target_text}

            image_data = sample.get("image_images")
            if image_data is None:
                image_data = sample.get("image_image")
            if image_data is not None:
                metrics.log_image(f"{tag}/{i}/input_latent", self._latent_to_image(image_data), global_step)
                if self.image_vae_decoder is not None:
                    self._try_decode_image(image_data, global_step, f"{tag}/{i}/input_image")

            metrics.log_text(f"{tag}/{i}/generated", gen_text, global_step, context=context)
            self._log_generation_metrics(
                outputs, sample, model, device, tag, i, global_step,
            )

    def _scenario_voice_to_image(self, model, eval_dataset, collator, device, global_step):
        """Scenario 6: Voice -> Image cross-modal generation."""
        tag = "voice_to_image"
        samples = self._get_eval_samples(
            eval_dataset, collator, self.num_eval_samples, requires_voice=True
        )
        if not samples:
            return

        for i, sample in enumerate(samples):
            # Build prompt: [BOV] [VOICE_PLACEHOLDER] [EOV] [BOI]
            prompt_tokens = [
                self._sp.BOV, self._sp.VOICE_PLACEHOLDER, self._sp.EOV,
                self._sp.BOI,
            ]
            prompt = self._build_prompt_ids(prompt_tokens, device)

            voice_inputs, voice_lengths = self._prepare_voice_for_generate(sample, device)

            outputs = self._generate(model, 
                text_input_ids=prompt,
                max_new_tokens=512,
                temperature=0.8,
                voice_temperature=self.voice_temperature,
                voice_variance_floor=self.voice_variance_floor,
                voice_inputs=voice_inputs,
                voice_lengths=voice_lengths,
            )

            image_preds = outputs.get("image_latent_preds")
            if image_preds is not None and image_preds.numel() > 0:
                pred_latent = image_preds[0, 0]
                metrics.log_image(
                    f"{tag}/{i}/generated_image_latent",
                    self._latent_to_image(pred_latent),
                    global_step,
                )

                if self.image_vae_decoder is not None:
                    self._try_decode_image(
                        pred_latent, global_step,
                        f"{tag}/{i}/generated_image"
                    )

            img_preds = outputs.get("image_latent_preds")
            gen_img = img_preds[0, 0] if img_preds is not None and img_preds.numel() > 0 else None
            self._log_generation_metrics(
                outputs, sample, model, device, tag, i, global_step,
                pred_latent=gen_img, modality="image",
            )

    def _scenario_image_to_voice(self, model, eval_dataset, collator, device, global_step):
        """Scenario 7: Image -> Voice cross-modal generation."""
        tag = "image_to_voice"
        samples = self._get_eval_samples(
            eval_dataset, collator, self.num_eval_samples, requires_image=True
        )
        if not samples:
            return

        for i, sample in enumerate(samples):
            # Build prompt: [BOI] [IMAGE_PLACEHOLDER] [EOI] [BOV]
            prompt_tokens = [
                self._sp.BOI, self._sp.IMAGE_PLACEHOLDER, self._sp.EOI,
                self._sp.BOV,
            ]
            prompt = self._build_prompt_ids(prompt_tokens, device)

            image_inputs = self._prepare_image_for_generate(sample, device)

            outputs = self._generate(model, 
                text_input_ids=prompt,
                max_new_tokens=512,
                temperature=0.8,
                voice_temperature=self.voice_temperature,
                voice_variance_floor=self.voice_variance_floor,
                image_inputs=image_inputs,
            )

            voice_preds = outputs.get("voice_latent_preds")
            if voice_preds is not None and voice_preds.numel() > 0:
                pred_latent = voice_preds[0, 0]
                metrics.log_image(
                    f"{tag}/{i}/generated_voice_latent",
                    self._latent_to_image(pred_latent),
                    global_step,
                )

                # Image->voice has no ground-truth speaker, use static only. Generated =>
                # render the model's PREDICTED F0 contour, not any GT fallback.
                gen_f0 = outputs.get("voice_f0_preds")
                self._log_audio_with_smg(
                    pred_latent, sample, global_step, f"{tag}/{i}",
                    f0_contour=(gen_f0[0, 0] if gen_f0 is not None and gen_f0.numel() > 0 else None),
                )

            voice_preds = outputs.get("voice_latent_preds")
            gen_voice = voice_preds[0, 0] if voice_preds is not None and voice_preds.numel() > 0 else None
            self._log_generation_metrics(
                outputs, sample, model, device, tag, i, global_step,
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

    def _decode_audio_latent_to_mel(
        self, latent: torch.Tensor, speaker_embedding: torch.Tensor,
        f0_contour: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Decode SIVE feature latent through SMG decoder to mel spectrogram.

        Args:
            latent: (C, T) predicted SIVE features from the world model's audio coda
            speaker_embedding: (speaker_dim,) speaker embedding
            f0_contour: (T,) the coda's SPEAKER-NORMALIZED F0 contour, if it has an F0
                head. In sigma units, NOT log Hz -- it goes to the SMG's F0 predictor,
                which denormalizes it with ECAPA. Handing it to f0_embedding() directly
                would read ~1.1 sigma as ~1.1 log Hz, i.e. 3 Hz.

        Returns:
            mel_spec (n_mels, T) or None on failure
        """
        try:
            if self.voice_smg_decoder is None:
                return None

            device = next(self.voice_smg_decoder.parameters()).device
            dtype = next(self.voice_smg_decoder.parameters()).dtype

            z = latent.to(device=device, dtype=dtype).unsqueeze(0)  # (1, C, T)
            spk = speaker_embedding.to(device=device, dtype=dtype).unsqueeze(0)  # (1, speaker_dim)

            # Always hand the SMG `features=z`: its voicing branch reads content, and on
            # the contour path its F0 trunk reads the contour. Passing the world's contour
            # lets the coda drive prosody -- with quantized units prosody is gone by
            # construction, and the SMG's predictor alone sees only (units, speaker).
            kwargs = {}
            if f0_contour is not None:
                c = f0_contour.to(device=device, dtype=dtype).reshape(1, -1)
                n = min(c.shape[-1], z.shape[-1])
                c, z = c[..., :n], z[..., :n]
                kwargs["f0_contour"] = c

            with torch.no_grad():
                mel = self.voice_smg_decoder.decode(
                    z=z, speaker_embedding=spk, features=z, **kwargs,
                )

            # mel: (1, n_mels, T) -> (n_mels, T)
            if isinstance(mel, tuple):
                mel = mel[0]  # decode can return (mel, film_stats) tuple
            if isinstance(mel, dict):
                mel = mel.get("reconstructed", mel.get("output", next(iter(mel.values()))))
            if isinstance(mel, torch.Tensor):
                if mel.numel() == 0:
                    print(f"Warning: SMG decode returned empty tensor with shape {mel.shape}")
                    return None
                return mel[0].float().cpu()
            print(f"Warning: SMG decode returned unexpected type: {type(mel)}")
            return None
        except Exception as e:
            import traceback
            print(f"Warning: SMG decode failed: {e}")
            traceback.print_exc()
            return None

    def _decode_and_log_audio(self, pred_latent, speaker_embedding, global_step, tag_prefix, speaker_label, f0_contour=None):
        """Decode latent -> mel via SMG, then mel -> waveform via vocoder, logging both."""
        mel = self._decode_audio_latent_to_mel(pred_latent, speaker_embedding, f0_contour=f0_contour)
        if mel is None:
            print(f"[audio_debug] _decode_audio_latent_to_mel returned None for {tag_prefix}/{speaker_label}")
            return
        if mel.numel() == 0:
            print(f"[audio_debug] _decode_audio_latent_to_mel returned empty tensor for {tag_prefix}/{speaker_label}")
            return
        # Log mel spectrogram as figure
        mel_np = mel.numpy()
        if mel_np.size == 0 or mel_np.shape[0] == 0 or mel_np.shape[-1] == 0:
            return
        fig = visualization.render_mel_spectrogram(mel_np, hop_length=self.voice_hop_length, sample_rate=self.voice_sample_rate, n_fft=self.voice_n_fft)
        if self.vocoder is not None:
            try:
                waveform = self._vocode(mel)
                metrics.log_audio(f"{tag_prefix}/{speaker_label}_audio", waveform, global_step, self.voice_sample_rate, context={
                    "mel": fig,
                })
            except Exception as e:
                print(f"Warning: Vocoder audio rendering failed for {tag_prefix}/{speaker_label}_audio: {e}")
                metrics.log_figure(f"{tag_prefix}/{speaker_label}_mel", fig, global_step)
        else:
            metrics.log_figure(f"{tag_prefix}/{speaker_label}_mel", fig, global_step)
        plt.close(fig)

    def _resolve_static_speaker(self, gt_speaker_emb):
        """The fixed reference voice for TTS renders.

        Prefers --static_speaker_embedding_path. Without it, pin a RANDOM eval speaker
        embedding (collected from the first pool of eval samples) and reuse it forever. A
        static path was ECAPA-SMG-specific (the best-recon ECAPA vector) and doesn't apply to
        a WavLM SMG, so a random WavLM eval speaker is the analog. Still PINNED once for
        consistency: you are listening for the model improving, not the voice drifting.
        """
        if self.static_speaker_embedding is not None:
            return self.static_speaker_embedding
        if self._pinned_speaker is None and gt_speaker_emb is not None:
            if not hasattr(self, "_speaker_pool"):
                self._speaker_pool = []
            self._speaker_pool.append(gt_speaker_emb.detach().clone().cpu())
            if len(self._speaker_pool) >= 8:  # seen enough eval speakers -> pin a random one
                import random as _rnd
                self._pinned_speaker = self._speaker_pool[_rnd.randrange(len(self._speaker_pool))]
                print(f"[viz] No --static_speaker_embedding_path; pinned a RANDOM eval speaker "
                      f"(1 of {len(self._speaker_pool)}) as the fixed reference voice for TTS renders.",
                      flush=True)
        if self._pinned_speaker is not None:
            return self._pinned_speaker
        # provisional until the pool fills (first eval only): the latest seen
        return self._speaker_pool[-1] if getattr(self, "_speaker_pool", None) else None

    @staticmethod
    def _generated_unit_ids(outputs, idx: int = 0):
        """Content unit ids for sample `idx` of a generate() OR teacher-forced call, or None.

        Prefers `voice_unit_id_trace` (what generation actually emitted, EOV already stripped
        by generate()). Falls back to argmax over `voice_unit_logits` for teacher-forced
        outputs. Never derives ids from voice_latent_preds: that is the regression head, a
        sibling of the unit classifier, not the model's discrete prediction.
        """
        trace = outputs.get("voice_unit_id_trace")
        if trace:
            seq = trace[idx] if isinstance(trace[0], (list, tuple)) else trace
            if len(seq) > 0:
                return torch.tensor([int(x) for x in seq], dtype=torch.long)
        logits = outputs.get("voice_unit_logits")
        if logits is not None and logits.numel() > 0 and idx < logits.shape[0]:
            return logits[idx].argmax(-1).reshape(-1).detach().cpu()
        return None

    def _log_audio_with_cosyvoice2(self, unit_ids, sample, global_step, tag_prefix, length=None):
        """Decode CosyVoice 2 content unit ids -> 24 kHz audio with the frozen flow+HiFT.

        Unlike the SMG path this consumes the voice coda's CLASSIFIER output (unit ids), not
        the regression head's latent -- on a discrete run those are sibling heads and the
        latent is not what the model is judged on. Callers must therefore pass unit_ids
        explicitly; we never fall back to the sample's GROUND-TRUTH units, which would
        silently log target audio under a `pred` tag.
        """
        if unit_ids is None:
            if not self._warned_no_units:
                self._warned_no_units = True
                print(f"[viz] WARNING: CosyVoice 2 decoder loaded but no unit ids were passed to "
                      f"{tag_prefix} — skipping audio for this render (NOT falling back to the "
                      f"latent or to ground-truth units).", flush=True)
            return
        ids = unit_ids.reshape(-1)
        # TRIM TO THE VALID LENGTH FIRST. Cached unit_ids are padded to the shard's max frame
        # count with ZERO, and 0 is a legal CosyVoice unit -- so value-based filtering alone
        # cannot see the padding, and the decoder renders it as breathy repeated non-speech
        # tacked onto the end. The true length lives in the scalar voice_feature_length.
        # NOT defaulted from the sample: generated ids (voice_unit_id_trace) are already exact,
        # and trimming them to the GT length would destroy the free-running-length signal.
        # Callers passing CACHED units must pass the length explicitly.
        if length is not None:
            n = int(length.item() if hasattr(length, "item") else length)
            if 0 < n < ids.numel():
                ids = ids[:n]
        # Then strip EOV / negative padding / out-of-codebook ids. The bound is the CODEBOOK
        # size (input_embedding rows = 6561), NOT flow.input_size — that is the feature width
        # (512) and would silently discard almost every unit.
        emb = getattr(self.voice_cosyvoice2_decoder.flow, "input_embedding", None)
        vocab = int(emb.weight.shape[0]) if emb is not None else None
        ids = ids[ids >= 0]
        if vocab:
            ids = ids[ids < vocab]
        if ids.numel() == 0:
            return

        gt_spk = None
        for key in ("voice_speaker_embeddings", "voice_speaker_embedding"):
            v = sample.get(key)
            if v is not None:
                gt_spk = v.reshape(-1)
                break
        # Dual-speaker render, mirroring the SMG path: the sample's OWN speaker, plus a PINNED
        # static reference voice. The static render is the control for speaker drift — with
        # embedding-only conditioning the frozen decoder falls back toward its own (lower-
        # pitched) prior whenever the units are off-manifold, so a wandering GT render with a
        # STABLE static render means the units are at fault, not the embedding. Both feed the
        # decoder identical units, so any difference between them is speaker conditioning alone.
        static_spk = self._resolve_static_speaker(gt_spk)
        renders = [("", gt_spk)] if gt_spk is not None else []
        if static_spk is not None and (gt_spk is None or static_spk is not gt_spk):
            renders.append(("_static_speaker", static_spk))
        if not renders:
            return

        sr = self.voice_cosyvoice2_decoder.sample_rate
        for suffix, spk in renders:
            try:
                wav = self.voice_cosyvoice2_decoder.decode(ids, spk)
            except Exception as e:
                print(f"Warning: CosyVoice 2 decode failed for {tag_prefix}{suffix}: "
                      f"{type(e).__name__}: {e}")
                continue
            if wav is None or wav.numel() == 0:
                continue
            metrics.log_audio(f"{tag_prefix}{suffix}_audio", wav, global_step, sr, context={
                "units": f"{ids.numel()} units -> {wav.numel()/sr:.2f}s @ {sr}Hz",
            })

    @staticmethod
    def _batch_voice_length(batch, i):
        """Valid voice frame count for row i of a collated batch, or None."""
        lens = batch.get("voice_feature_lengths") if isinstance(batch, dict) else None
        if lens is None or i >= len(lens):
            return None
        v = lens[i]
        # Collated lengths can be per-span (shape (n_spans,)); the voice path here is single-span.
        return int(v.reshape(-1)[0].item()) if hasattr(v, "reshape") else int(v)

    def _log_audio_with_smg(self, pred_latent, sample, global_step, tag_prefix, f0_contour=None,
                            unit_ids=None, unit_length=None):
        """Run dual-speaker SMG decoding: ground-truth speaker + static speaker.

        On a CosyVoice 2 run the frozen flow+HiFT replaces the SMG entirely and consumes
        `unit_ids` instead of the latent (see _log_audio_with_cosyvoice2). `unit_length` trims
        PADDED unit sources (cached targets, teacher-forced argmax over a padded batch); leave
        it None for a free-running trace, which is already exact.
        """
        if self.voice_cosyvoice2_decoder is not None:
            self._log_audio_with_cosyvoice2(unit_ids, sample, global_step, tag_prefix,
                                            length=unit_length)
            return
        if self.voice_smg_decoder is None:
            if not self._warned_no_smg:
                self._warned_no_smg = True
                print("[viz] WARNING: no SMG decoder loaded — NO TTS AUDIO WILL BE LOGGED. "
                      "Pass --voice_smg_checkpoint_path (and check the path exists; a bad "
                      "path is caught and downgraded to a warning at load time).", flush=True)
            # Fallback to direct vocoder (old behavior)
            if self.vocoder is not None:
                self._try_vocoder_from_latent(
                    None, pred_latent, global_step,
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

        # A contour-mode SMG (f0_predictor_input="contour") has no internal F0 predictor and
        # REQUIRES a speaker-normalized contour, or decode() raises. Generated decodes pass the
        # world model's PREDICTED contour explicitly; for target/input decodes of GROUND-TRUTH
        # features the caller passes nothing, so fall back to the sample's own GT contour (the
        # same normalize_f0 output the SMG trained on). This never overrides an explicit contour.
        if f0_contour is None:
            for key in ("voice_f0_contour", "f0_contour", "audio_f0_contour"):
                v = sample.get(key)
                if v is not None:
                    f0_contour = v[0] if (hasattr(v, "dim") and v.dim() > 1) else v
                    break

        if gt_speaker_emb is not None:
            self._decode_and_log_audio(
                pred_latent, gt_speaker_emb, global_step,
                tag_prefix, "gt_speaker", f0_contour=f0_contour,
            )

        # Decode with the fixed reference speaker, so renders are comparable across evals
        static_emb = self._resolve_static_speaker(gt_speaker_emb)
        if static_emb is not None:
            self._decode_and_log_audio(
                pred_latent, static_emb, global_step,
                tag_prefix, "static_speaker", f0_contour=f0_contour,
            )

    def _try_vocoder_from_latent(self, model, latent, global_step, tag):
        """Try to decode SIVE feature latent directly through vocoder.

        Note: This is a fallback — SIVE features are not mel spectrograms, so
        direct vocoding only makes sense if feature_channels == n_mels.
        Prefer using SMG decoder → vocoder for proper audio synthesis.
        """
        try:
            if self.vocoder is None:
                return

            mel = latent.float().cpu()
            # SIVE features are (C, T) — use directly as (n_mels, T) if C matches
            waveform = self._vocode(mel)
            metrics.log_audio(tag, waveform, global_step, self.voice_sample_rate)
        except Exception as e:
            print(f"Warning: Vocoder decoding failed for {tag}: {e}")

    def _try_decode_image(self, latent, global_step, tag):
        """Try to decode image latent through image VAE decoder.

        Uses canonical de-normalization `(x + 1) / 2` for LiteVAE outputs,
        which were trained on inputs in [-1, 1] (preprocessor mean=std=0.5).
        Any overshoot clips to pure black/white and any undershoot shows as
        muddy gray — artifacts are visible rather than papered over.

        Previously used per-image `(img - min) / (max - min)` min/max
        stretching which always produces a "viewable" image regardless of
        the underlying latent being in-distribution or not, masking
        scale/training issues.
        """
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

            # decoded: (1, 3, H, W) in [-1, 1] → de-normalize to [0, 1]
            raw = decoded[0].float().cpu()
            img = ((raw + 1.0) / 2.0).clamp(0, 1)
            metrics.log_image(tag, img, global_step)
        except Exception as e:
            print(f"Warning: Image VAE decoding failed for {tag}: {e}")
