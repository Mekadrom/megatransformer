
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
from itertools import product
from typing import Optional, Literal

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from PIL import Image
from torch.amp import autocast
from transformers import Trainer, TrainerCallback
from transformers.integrations import TensorBoardCallback

from dataset_loading import image_loading
from utils.megatransformer_utils import sanitize_model



def get_writer(trainer: Trainer):
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, TensorBoardCallback):
            if callback.tb_writer is not None:
                return callback.tb_writer
    return None


class CLIPScoreEvaluationCallback(TrainerCallback):
    """
    Callback for evaluating diffusion models using CLIP scores.

    Features:
    - Computes CLIP score (text-image alignment) on generated samples
    - OOD generalization probing via combinatorial prompt templates
    - Output diversity measurement (detects mode collapse)
    - All metrics logged to TensorBoard

    Usage:
        callback = CLIPScoreEvaluationCallback(
            eval_steps=1000,
            vae=vae_model,
            text_encoder=t5_model,
            text_tokenizer=t5_tokenizer,
            train_prompts=["a red car", "a blue house"],  # Optional: for ID vs OOD comparison
        )
        trainer.add_callback(callback)
    """

    # Template components for OOD probing
    # These are combined combinatorially to create novel prompts
    COLORS = ["red", "blue", "green", "yellow", "purple", "orange"]
    OBJECTS = ["car", "house", "dog", "cat", "tree", "flower", "bird", "boat"]
    STYLES = ["photo of", "painting of", "sketch of", "drawing of"]
    SCENES = ["in a field", "on a beach", "in a city", "in a forest", "on a mountain"]

    def __init__(
        self,
        image_size: int = 32,
        eval_steps: int = 1000,
        n_eval_samples: int = 16,
        sampling_timesteps: int = 25,
        vae: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        text_tokenizer=None,
        train_prompts: Optional[list[str]] = None,
        n_ood_probes: int = 32,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        guidance_scale: float = 3.0,
        ema=None,
        step_offset: int = 0,
    ):
        """
        Args:
            eval_steps: Run evaluation every N steps
            n_eval_samples: Number of samples to generate per prompt batch
            sampling_timesteps: Diffusion sampling steps (fewer = faster)
            vae: VAE decoder for latent->image conversion
            text_encoder: Text encoder (e.g., T5) for prompt encoding
            text_tokenizer: Tokenizer for text encoder
            train_prompts: List of training prompts (for ID vs OOD comparison)
            n_ood_probes: Number of combinatorial OOD prompts to generate
            clip_model_name: HuggingFace CLIP model for scoring
            guidance_scale: Classifier-free guidance scale
            ema: Optional EMA model wrapper
            step_offset: Offset for global step (for resumed training)
        """
        self.trainer: Optional[Trainer] = None
        self.image_size = image_size
        self.eval_steps = eval_steps
        self.n_eval_samples = n_eval_samples
        self.sampling_timesteps = sampling_timesteps
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer
        self.train_prompts = train_prompts or []
        self.n_ood_probes = n_ood_probes
        self.guidance_scale = guidance_scale
        self.ema = ema
        self.step_offset = step_offset

        # Will be loaded lazily
        self._clip_model = None
        self._clip_processor = None
        self.clip_model_name = clip_model_name

        # Generate OOD probe prompts (combinatorial)
        self.ood_prompts = self._generate_ood_prompts()

    def _generate_ood_prompts(self) -> list[str]:
        """Generate diverse OOD prompts by combining template components."""
        prompts = []

        # Combinatorial: style + color + object + scene
        for style, color, obj, scene in product(
            self.STYLES[:2], self.COLORS[:3], self.OBJECTS[:4], self.SCENES[:2]
        ):
            prompts.append(f"a {style} a {color} {obj} {scene}")

        # Shuffle and limit
        random.shuffle(prompts)
        return prompts[:self.n_ood_probes]

    def _load_clip(self, device):
        """Lazily load CLIP model."""
        if self._clip_model is None:
            from transformers import CLIPModel, CLIPProcessor
            print(f"Loading CLIP model: {self.clip_model_name}")
            self._clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(device)
            self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self._clip_model.eval()
        return self._clip_model, self._clip_processor

    @torch.no_grad()
    def compute_clip_score(
        self,
        images: torch.Tensor,
        prompts: list[str],
        device: torch.device,
    ) -> tuple[float, list[float]]:
        """
        Compute CLIP scores between images and prompts.

        Args:
            images: Tensor of images [B, C, H, W] in range [0, 1] or [-1, 1]
            prompts: List of text prompts (same length as batch)
            device: Device to run on

        Returns:
            mean_score: Average CLIP score across batch
            scores: Individual scores per sample
        """
        clip_model, clip_processor = self._load_clip(device)

        # Normalize images to [0, 1] if needed
        if images.min() < 0:
            images = (images + 1) / 2
        images = images.clamp(0, 1)

        # Convert to PIL for CLIP processor
        pil_images = []
        for img in images:
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))

        # Process inputs
        inputs = clip_processor(
            text=prompts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
        ).to(device)

        # Get embeddings
        outputs = clip_model(**inputs)

        # Compute cosine similarity (CLIP score)
        image_embeds = outputs.image_embeds  # [B, D]
        text_embeds = outputs.text_embeds    # [B, D]

        # Normalize
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Per-sample cosine similarity
        scores = (image_embeds * text_embeds).sum(dim=-1).tolist()
        mean_score = sum(scores) / len(scores)

        return mean_score, scores

    @torch.no_grad()
    def compute_output_diversity(self, images: torch.Tensor) -> float:
        """
        Measure diversity of generated images using pairwise LPIPS-like distance.
        High diversity = model is not mode-collapsed.

        Uses simple feature-space variance as a cheap proxy.
        """
        if images.shape[0] < 2:
            return 0.0

        # Flatten images and compute pairwise distances
        flat = images.view(images.shape[0], -1)  # [B, C*H*W]

        # Compute mean pairwise L2 distance (normalized)
        flat_norm = F.normalize(flat, dim=-1)
        pairwise_sim = flat_norm @ flat_norm.T  # [B, B]

        # Exclude diagonal (self-similarity)
        mask = ~torch.eye(pairwise_sim.shape[0], dtype=torch.bool, device=pairwise_sim.device)
        mean_similarity = pairwise_sim[mask].mean().item()

        # Diversity = 1 - similarity (higher = more diverse)
        diversity = 1.0 - mean_similarity
        return diversity

    @torch.no_grad()
    def generate_samples(
        self,
        model,
        prompts: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        """Generate images from prompts using the diffusion model."""
        # Encode prompts
        text_inputs = self.text_tokenizer(
            prompts,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        self.text_encoder = self.text_encoder.to(device)

        with torch.no_grad():
            text_embeds = self.text_encoder(**text_inputs).last_hidden_state

        self.text_encoder = self.text_encoder.to("cpu")

        with autocast(device_type=device.type):
            # Sample from diffusion model
            latents = model.sample(
                device=device,
                batch_size=len(prompts),
                condition=text_embeds,
                override_sampling_steps=self.sampling_timesteps,
                image_size=self.image_size,
                generator=torch.Generator(device).manual_seed(42),
                guidance_scale=self.guidance_scale,
                sampler="dpm_solver_pp",
                dpm_solver_order=2,
            )

        # Handle list output format
        if isinstance(latents, list):
            latents = latents[0]

        # Decode with VAE if available
        if self.vae is not None:
            self.vae.eval()
            self.vae.to(device)
            images = self.vae.decode(latents)
        else:
            images = latents

        return images

    def _log_images_to_tensorboard(
        self,
        writer,
        global_step: int,
        ood_images: list[tuple[torch.Tensor, str, float]],
        id_images: list[tuple[torch.Tensor, str, float]],
        max_images: int = 8,
    ):
        """
        Log generated images to TensorBoard with prompts and CLIP scores.

        Args:
            writer: TensorBoard SummaryWriter
            global_step: Current training step
            ood_images: List of (image, prompt, score) for OOD samples
            id_images: List of (image, prompt, score) for ID samples
            max_images: Maximum number of images to log per category
        """
        if writer is None:
            return

        def prepare_image_grid(images_data: list, tag_prefix: str):
            """Create an image grid with annotations."""
            if not images_data:
                return

            # Sort by score (best first) and limit
            images_data = sorted(images_data, key=lambda x: x[2], reverse=True)[:max_images]

            # Stack images into grid
            images = []
            for img, prompt, score in images_data:
                # Normalize to [0, 1] if needed
                if img.min() < 0:
                    img = (img + 1) / 2
                img = img.clamp(0, 1)
                images.append(img)

            if not images:
                return

            # Create grid
            grid = torch.stack(images, dim=0)  # [N, C, H, W]

            # Log as image grid
            writer.add_images(
                f"{tag_prefix}/samples",
                grid,
                global_step,
                dataformats='NCHW'
            )

            # Log individual images with prompts as text
            for i, (img, prompt, score) in enumerate(images_data[:4]):  # Top 4
                # Normalize
                if img.min() < 0:
                    img = (img + 1) / 2
                img = img.clamp(0, 1)

                writer.add_image(
                    f"{tag_prefix}/sample_{i}",
                    img,
                    global_step,
                    dataformats='CHW'
                )
                # Log prompt as text
                writer.add_text(
                    f"{tag_prefix}/prompt_{i}",
                    f"**{prompt}** (CLIP: {score:.3f})",
                    global_step
                )

        # Log OOD samples (sorted by CLIP score - best generalizations first)
        prepare_image_grid(ood_images, "eval_ood")

        # Log ID samples
        prepare_image_grid(id_images, "eval_id")

        # Also log worst OOD samples to see failure modes
        if ood_images:
            worst_ood = sorted(ood_images, key=lambda x: x[2])[:4]  # Lowest scores
            for i, (img, prompt, score) in enumerate(worst_ood):
                if img.min() < 0:
                    img = (img + 1) / 2
                img = img.clamp(0, 1)
                writer.add_image(f"eval_ood_worst/sample_{i}", img, global_step, dataformats='CHW')
                writer.add_text(f"eval_ood_worst/prompt_{i}", f"**{prompt}** (CLIP: {score:.3f})", global_step)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Run evaluation at specified intervals."""
        global_step = state.global_step + self.step_offset

        if global_step == 0 or global_step % self.eval_steps != 0:
            return

        if model is None or self.text_encoder is None or self.text_tokenizer is None:
            return

        device = next(model.parameters()).device
        writer = get_writer(self.trainer)

        # Use EMA model if available
        eval_model = model
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        try:
            model.eval()

            all_scores = []
            ood_scores = []
            id_scores = []
            diversities = []

            # Store images for TensorBoard logging
            ood_images_to_log = []  # List of (image, prompt, score)
            id_images_to_log = []

            # Evaluate on OOD probes (combinatorial prompts)
            print(f"\n[Step {global_step}] Running CLIP evaluation on {len(self.ood_prompts)} OOD prompts...")

            # Process in batches
            batch_size = min(4, len(self.ood_prompts))
            for i in range(0, min(len(self.ood_prompts), self.n_eval_samples), batch_size):
                batch_prompts = self.ood_prompts[i:i + batch_size]

                images = self.generate_samples(eval_model, batch_prompts, device)
                _, scores = self.compute_clip_score(images, batch_prompts, device)
                diversity = self.compute_output_diversity(images)

                ood_scores.extend(scores)
                all_scores.extend(scores)
                diversities.append(diversity)

                # Store for logging (keep images on CPU to save GPU memory)
                for img, prompt, score in zip(images, batch_prompts, scores):
                    ood_images_to_log.append((img.cpu(), prompt, score))

            # Evaluate on training prompts if provided (for ID comparison)
            if self.train_prompts:
                print(f"  Evaluating on {min(len(self.train_prompts), self.n_eval_samples)} ID prompts...")
                for i in range(0, min(len(self.train_prompts), self.n_eval_samples), batch_size):
                    batch_prompts = self.train_prompts[i:i + batch_size]

                    try:
                        images = self.generate_samples(eval_model, batch_prompts, device)
                        _, scores = self.compute_clip_score(images, batch_prompts, device)

                        id_scores.extend(scores)
                        all_scores.extend(scores)

                        # Store for logging
                        for img, prompt, score in zip(images, batch_prompts, scores):
                            id_images_to_log.append((img.cpu(), prompt, score))
                    except Exception as e:
                        print(f"  Warning: Failed to evaluate ID batch: {e}")
                        continue

            # Log metrics
            if writer is not None and all_scores:
                # Overall CLIP score
                mean_clip = sum(all_scores) / len(all_scores)
                writer.add_scalar("eval/clip_score_mean", mean_clip, global_step)

                # Score distribution
                writer.add_scalar("eval/clip_score_min", min(all_scores), global_step)
                writer.add_scalar("eval/clip_score_max", max(all_scores), global_step)
                writer.add_scalar("eval/clip_score_std", torch.tensor(all_scores).std().item(), global_step)

                # OOD vs ID comparison (generalization gap)
                if ood_scores:
                    ood_mean = sum(ood_scores) / len(ood_scores)
                    writer.add_scalar("eval/clip_score_ood", ood_mean, global_step)

                    # Count "successful" OOD generations (CLIP > threshold)
                    ood_success_rate = sum(1 for s in ood_scores if s > 0.2) / len(ood_scores)
                    writer.add_scalar("eval/ood_success_rate", ood_success_rate, global_step)

                if id_scores:
                    id_mean = sum(id_scores) / len(id_scores)
                    writer.add_scalar("eval/clip_score_id", id_mean, global_step)

                    # Generalization gap (smaller = better generalization)
                    if ood_scores:
                        gen_gap = id_mean - ood_mean
                        writer.add_scalar("eval/generalization_gap", gen_gap, global_step)

                # Output diversity (detects mode collapse)
                if diversities:
                    mean_diversity = sum(diversities) / len(diversities)
                    writer.add_scalar("eval/output_diversity", mean_diversity, global_step)

                print(f"  CLIP Score: {mean_clip:.3f} (OOD: {sum(ood_scores)/len(ood_scores) if ood_scores else 0:.3f})")
                if diversities:
                    print(f"  Output Diversity: {mean_diversity:.3f}")

                # Log images to TensorBoard
                self._log_images_to_tensorboard(
                    writer, global_step, ood_images_to_log, id_images_to_log
                )

        finally:
            # Restore original weights if using EMA
            if self.ema is not None:
                self.ema.restore()
            model.train()


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, stop_step: int):
        self.stop_step = stop_step

    def on_step_begin(self, args, state, control, **kwargs):
        if self.stop_step > 0 and state.global_step >= self.stop_step:
            print(f"Early stopping at step {state.global_step} as per stop_step={self.stop_step}.")
            control.should_training_stop = True


class ReduceLROnPlateauCallback(TrainerCallback):
    """
    Reduces learning rate when a monitored metric has stopped improving.

    Similar to torch.optim.lr_scheduler.ReduceLROnPlateau but works as a
    HuggingFace Trainer callback.

    Usage:
        callback = ReduceLROnPlateauCallback(
            monitor="eval_loss",  # or "eval/clip_score_mean", "loss", etc.
            mode="min",           # "min" for loss, "max" for accuracy/CLIP score
            factor=0.5,           # Multiply LR by this factor
            patience=3,           # Number of checks with no improvement
            min_lr=1e-7,          # Minimum LR
            check_every_n_steps=1000,  # How often to check (if using step-based)
        )
        trainer.add_callback(callback)

    The callback checks metrics at evaluation time (on_evaluate) or at
    specified step intervals using logged metrics.
    """

    def __init__(
        self,
        monitor: str = "eval_loss",
        mode: str = "min",
        factor: float = 0.5,
        patience: int = 3,
        min_lr: float = 1e-7,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        check_every_n_steps: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Args:
            monitor: Metric name to monitor. Common options:
                - "eval_loss" or "loss" (evaluation/training loss)
                - "eval/clip_score_mean" (CLIP score from CLIPScoreEvaluationCallback)
                - Any metric logged to trainer.state.log_history
            mode: "min" if lower is better, "max" if higher is better
            factor: Factor to multiply LR by when reducing (0 < factor < 1)
            patience: Number of checks with no improvement before reducing LR
            min_lr: Minimum learning rate
            threshold: Threshold for measuring improvement
            threshold_mode: "rel" for relative, "abs" for absolute threshold
            cooldown: Number of checks to wait after LR reduction before resuming
            check_every_n_steps: If set, check at step intervals instead of only on_evaluate
            verbose: Print LR changes
        """
        assert 0.0 < factor < 1.0, "Factor must be between 0 and 1"
        assert mode in ("min", "max"), "Mode must be 'min' or 'max'"

        self.trainer: Optional[Trainer] = None
        self.monitor = monitor
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.check_every_n_steps = check_every_n_steps
        self.verbose = verbose

        # State
        self.best = None
        self.num_bad_checks = 0
        self.cooldown_counter = 0
        self.last_check_step = 0

        # For mode="min", we want to minimize; for "max", maximize
        self.mode_worse = float('inf') if mode == "min" else float('-inf')

    def _is_improvement(self, current: float, best: float) -> bool:
        """Check if current is an improvement over best."""
        if self.threshold_mode == "rel":
            if self.mode == "min":
                return current < best * (1 - self.threshold)
            else:
                return current > best * (1 + self.threshold)
        else:  # abs
            if self.mode == "min":
                return current < best - self.threshold
            else:
                return current > best + self.threshold

    def _get_metric_value(self, state) -> Optional[float]:
        """Extract the monitored metric from trainer state."""
        # Check log_history for the metric
        if not state.log_history:
            return None

        # Search backwards through log history for the metric
        for log_entry in reversed(state.log_history):
            if self.monitor in log_entry:
                return log_entry[self.monitor]

            # Also check without prefix for common metrics
            if self.monitor == "eval_loss" and "eval_loss" in log_entry:
                return log_entry["eval_loss"]
            if self.monitor == "loss" and "loss" in log_entry:
                return log_entry["loss"]

        if self.trainer is not None and self.trainer.last_logged_loss is not None:
            return self.trainer.last_logged_loss

        return None

    def _reduce_lr(self, optimizer, scheduler=None) -> tuple[list[float], list[float]]:
        """
        Reduce learning rate for all parameter groups.

        This modifies both the optimizer AND the scheduler's base_lrs to ensure
        the change persists (since the scheduler overwrites optimizer LRs each step).
        """
        old_lrs = []
        new_lrs = []

        # Reduce base_lrs in scheduler (this is what the scheduler uses for calculations)
        if scheduler is not None and hasattr(scheduler, 'base_lrs'):
            for i, base_lr in enumerate(scheduler.base_lrs):
                new_base_lr = max(base_lr * self.factor, self.min_lr)
                scheduler.base_lrs[i] = new_base_lr
                old_lrs.append(base_lr)
                new_lrs.append(new_base_lr)

            # Also update initial_lr in optimizer param_groups (some schedulers use this)
            for i, param_group in enumerate(optimizer.param_groups):
                if 'initial_lr' in param_group:
                    param_group['initial_lr'] = max(param_group['initial_lr'] * self.factor, self.min_lr)
        else:
            # Fallback: modify optimizer directly (may be overwritten by scheduler)
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = max(old_lr * self.factor, self.min_lr)
                param_group['lr'] = new_lr
                old_lrs.append(old_lr)
                new_lrs.append(new_lr)

                # Also try to update initial_lr if present
                if 'initial_lr' in param_group:
                    param_group['initial_lr'] = new_lr

        return old_lrs, new_lrs

    def _check_and_reduce(self, state, optimizer, scheduler=None):
        """Check metric and reduce LR if needed."""
        current = self._get_metric_value(state)

        if current is None:
            print(f"[ReduceLROnPlateau] Warning: Monitored metric '{self.monitor}' not found.")
            return

        # Initialize best on first check
        if self.best is None:
            self.best = current
            if self.verbose:
                print(f"[ReduceLROnPlateau] Initial {self.monitor}: {current:.6f}")
            return

        # Handle cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return

        # Check for improvement
        if self._is_improvement(current, self.best):
            self.best = current
            self.num_bad_checks = 0
            if self.verbose:
                print(f"[ReduceLROnPlateau] {self.monitor} improved to {current:.6f}")
        else:
            self.num_bad_checks += 1
            if self.verbose:
                print(f"[ReduceLROnPlateau] {self.monitor}: {current:.6f} "
                      f"(no improvement for {self.num_bad_checks}/{self.patience} checks)")

        # Reduce LR if patience exceeded
        if self.num_bad_checks >= self.patience:
            old_lrs, new_lrs = self._reduce_lr(optimizer, scheduler)
            self.num_bad_checks = 0
            self.cooldown_counter = self.cooldown

            if self.verbose:
                for i, (old_lr, new_lr) in enumerate(zip(old_lrs, new_lrs)):
                    if old_lr != new_lr:
                        print(f"[ReduceLROnPlateau] Reducing LR (base) for group {i}: "
                              f"{old_lr:.2e} -> {new_lr:.2e}")

    def _get_optimizer_and_scheduler(self):
        """Get optimizer and scheduler from trainer."""
        optimizer = None
        scheduler = None

        if self.trainer is not None:
            optimizer = self.trainer.optimizer
            scheduler = self.trainer.lr_scheduler

        return optimizer, scheduler

    def on_evaluate(self, args, state, control, **kwargs):
        """Check metric after evaluation."""
        optimizer, scheduler = self._get_optimizer_and_scheduler()

        if optimizer is not None:
            self._check_and_reduce(state, optimizer, scheduler)

    def on_step_end(self, args, state, control, **kwargs):
        """Optionally check at step intervals."""
        if self.check_every_n_steps is None:
            return

        if state.global_step - self.last_check_step < self.check_every_n_steps:
            return

        self.last_check_step = state.global_step

        optimizer, scheduler = self._get_optimizer_and_scheduler()

        if optimizer is not None:
            self._check_and_reduce(state, optimizer, scheduler)

    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        return {
            "best": self.best,
            "num_bad_checks": self.num_bad_checks,
            "cooldown_counter": self.cooldown_counter,
            "last_check_step": self.last_check_step,
        }

    def load_state_dict(self, state_dict: dict):
        """Load state from checkpoint."""
        self.best = state_dict.get("best", self.best)
        self.num_bad_checks = state_dict.get("num_bad_checks", self.num_bad_checks)
        self.cooldown_counter = state_dict.get("cooldown_counter", self.cooldown_counter)
        self.last_check_step = state_dict.get("last_check_step", self.last_check_step)


def gradfilter_ma(
    m: nn.Module,
    grads: Optional[dict[str, deque]] = None,
    window_size: int = 100,
    lamb: float = 5.0,
    filter_type: Literal['mean', 'sum'] = 'mean',
    warmup: bool = True,
    trigger: bool = False,
) -> dict[str, deque]:
    if grads is None:
        grads = {n: deque(maxlen=window_size) for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n].append(p.grad.data.detach())  # .cpu())

            # Modify the gradients
            if not warmup or len(grads[n]) == window_size and not trigger:
                if filter_type == "mean":
                    avg = sum(grads[n]) / len(grads[n])
                elif filter_type == "sum":
                    avg = sum(grads[n])
                else:
                    raise ValueError(f"Unrecognized filter_type {filter_type}")
                p.grad.data = p.grad.data + avg * lamb

    return grads


class GrokFastMATrainer(Trainer):
    def __init__(self, *args, window_size=100, lamb=5.0, filter_type='mean', warmup=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size
        self.lamb = lamb
        self.filter_type = filter_type
        self.warmup = warmup
        self.grads = None
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs)
        # Apply gradfilter_ma after gradients are computed but before optimizer step
        self.grads = gradfilter_ma(model, self.grads, self.window_size, self.lamb, self.filter_type, self.warmup)
        return loss
    

def gradfilter_ema(
    m: nn.Module,
    grads: Optional[dict[str, torch.Tensor]] = None,
    alpha: float = 0.98,
    lamb: float = 2.0,
) -> dict[str, torch.Tensor]:
    if grads is None:
        grads = {n: p.grad.data.detach() for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}
    
    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            p.grad.data = p.grad.data + grads[n] * lamb
    
    return grads


class GrokfastEMATrainer(Trainer):
    def __init__(self, *args, alpha=0.98, lamb=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.lamb = lamb
        self.grads = None
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs)
        # Apply gradfilter_ema after gradients are computed but before optimizer step
        self.grads = gradfilter_ema(model, self.grads, self.alpha, self.lamb)
        return loss
    

class DebugTrainer(Trainer):
    def get_train_dataloader(self):
        train_dataloader = super().get_train_dataloader()
        return train_dataloader
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        # Log batch data
        if self.state.global_step < 5:  # Only for first few steps
            print(f"Step {self.state.global_step} inputs:")
            for k, v in inputs.items():
                if hasattr(v, "shape"):
                    print(f"  {k}: {v.shape}")
            
            # Print tokens
            if "input_ids" in inputs:
                tokens = inputs["input_ids"][0].tolist()
                print(f"  First example tokens: {tokens}...")
                
                # Decode if tokenizer is available
                if hasattr(self, "tokenizer"):
                    decoded = self.processing_class.decode(tokens)
                    print(f"  Decoded: {decoded[:100]}...")
        
        return super().training_step(model, inputs)


class DefaultTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self.get_tensorboard_writer()
        # inputs["output_hidden_states"] = True
        # inputs["return_dict"] = False

        sanitized_model = sanitize_model(model)

        for name, param in sanitized_model.named_parameters():
            if param.dtype == torch.long or param.dtype == torch.int64:
                print(f"Found long parameter: {name}")
                param.data = param.data.to(torch.float32)

        sanitized_model.config.current_epoch = self.state.epoch
        sanitized_model.config.current_global_step = self.state.global_step

        stacked_losses, outputs = super().compute_loss(model, inputs, return_outputs=True)

        if self.state.global_step % self.args.logging_steps == 0 and self.writer is not None:
            prefix = "train/" if model.training else "eval/"

            self.writer.add_scalar(f"fetch_misses", image_loading.misses, self.state.global_step)

            if not isinstance(outputs, tuple) and hasattr(outputs, "n_steps_no_grad") and hasattr(outputs, "k_steps_grad"):
                self.log_steps(prefix, outputs.n_steps_no_grad, outputs.k_steps_grad)
            elif isinstance(outputs, tuple):
                self.log_steps(prefix, outputs[-2], outputs[-1])
                if stacked_losses is not None:
                    if stacked_losses.ndim == 2 and stacked_losses.shape[1] == 5:
                        self.optional_log_loss(f"{prefix}recon_loss", stacked_losses[:, 0].mean(dim=0).item(), self.state.global_step)
                        self.optional_log_loss(f"{prefix}ssim_loss", stacked_losses[:, 1].mean(dim=0).item(), self.state.global_step)
                        self.optional_log_loss(f"{prefix}kl_divergence", stacked_losses[:, 2].mean(dim=0).item(), self.state.global_step)
                        self.optional_log_loss(f"{prefix}mu_loss", stacked_losses[:, 3].mean(dim=0).item(), self.state.global_step)
                        self.optional_log_loss(f"{prefix}logvar_loss", stacked_losses[:, 4].mean(dim=0).item(), self.state.global_step)
                    if stacked_losses.ndim == 1 and stacked_losses.shape[0] == 6:
                        self.optional_log_loss(f"{prefix}text_loss", stacked_losses[0].mean(dim=0).item(), self.state.global_step)
                        self.optional_log_loss(f"{prefix}text_loss", stacked_losses[0].mean(dim=0).item(), self.state.global_step)
                        self.optional_log_loss(f"{prefix}mel_l1_loss", stacked_losses[1].mean(dim=0).item(), self.state.global_step)
                        self.optional_log_loss(f"{prefix}waveform_l1_loss", stacked_losses[2].mean(dim=0).item(), self.state.global_step)
                        self.optional_log_loss(f"{prefix}sc_loss", stacked_losses[3].mean(dim=0).item(), self.state.global_step)
                        self.optional_log_loss(f"{prefix}mag_loss", stacked_losses[4].mean(dim=0).item(), self.state.global_step)
                        self.optional_log_loss(f"{prefix}image_mse_loss", stacked_losses[5].mean(dim=0).item(), self.state.global_step)
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                for i, hidden_state in enumerate(outputs.hidden_states):
                    token_correlation = get_token_correlation(hidden_state)
                    self.writer.add_scalar(f"{prefix}token_correlation_{i}", token_correlation, self.state.global_step)
        
        actual_loss = (
            (stacked_losses[0].mean() * 1.0) # text loss
                + (stacked_losses[1].mean() * 5.0) # audio mel l1 loss
                + (stacked_losses[2].mean() * 0.5) # audio waveform l1 loss
                + (stacked_losses[3].mean() * 0.1) # audio spectral convergence loss
                + (stacked_losses[4].mean() * 0.15) # audio log magnitude loss
                + (stacked_losses[5].mean() * 5.0) # image mse loss
        ) if stacked_losses is not None else outputs.loss
        return (actual_loss, outputs) if return_outputs else actual_loss
    
    def optional_log_loss(self, tag, value, step):
        if value != 0.0:
            self.writer.add_scalar(tag, value, step)

    def log_steps(self, prefix, n_steps_no_grad, k_steps_grad):
        if n_steps_no_grad is None or k_steps_grad is None:
            return
        self.writer.add_scalar(f"{prefix}n_steps_no_grad", n_steps_no_grad, self.state.global_step)
        self.writer.add_scalar(f"{prefix}k_steps_grad", k_steps_grad, self.state.global_step)

    def get_tensorboard_writer(self):
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback):
                self.writer = callback.tb_writer
                break

        if not hasattr(self, "writer"):
            print("Warning: No TensorBoard writer found. Please check your callback setup.")
            self.writer = None


def trainer_lookup(argparser_args, trainer_name, default=DefaultTrainer):
    if trainer_name == "grokfast_ema":
        return lambda *args, **kwargs: GrokfastEMATrainer(
            *args,
            alpha=argparser_args.grokfast_ema_alpha,
            lamb=argparser_args.grokfast_ema_lambda,
            **kwargs
        )
    elif trainer_name == "grokfast_ma":
        return lambda *args, **kwargs: GrokFastMATrainer(
            *args,
            window_size=argparser_args.grokfast_ma_window_size,
            lamb=argparser_args.grokfast_ma_lambda,
            filter_type=argparser_args.grokfast_ma_filter_type,
            warmup=argparser_args.grokfast_ma_warmup,
            **kwargs
        ),
    elif trainer_name == "debug":
        return DebugTrainer
    return default


def setup_int8_training(args, model):
    # Method 1: Using PEFT with Bits and Bytes quantization
    if args.use_int8_peft:
        print("Setting up INT8 training with PEFT/LoRA")
        
        model = prepare_model_for_kbit_training(model, args.use_gradient_checkpointing)
        
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["query", "key", "value", "dense"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    # Method 2: Using DeepSpeed ZeroQuant (configured in ds_config.json)
    elif args.use_int8_deepspeed:
        print("Using DeepSpeed for INT8 quantization during training")
        return model
    # No INT8 training
    else:
        return model


def get_token_correlation(hidden_states):
    x_c = hidden_states - hidden_states.mean(dim=1, keepdim=True)

    normed_x = x_c / x_c.norm(dim=-1, keepdim=True)

    token_correlation = (normed_x @ normed_x.transpose(1, 2)).mean() - (1 / hidden_states.shape[1])

    return token_correlation


def create_multimodal_optimizer(model, weight_decay):
    audio_decoder_params = set(model.output_transform.audio_decoder.parameters())
    vocoder_params = set(model.output_transform.audio_decoder.vocoder.parameters())

    # Get parameters unique to audio_decoder (excluding vocoder)
    audio_decoder_only_params = [p for p in audio_decoder_params if p not in vocoder_params]

    # Create AdamW optimizer with these groups
    optimizer = torch.optim.AdamW([
        {'params': model.input_transform.parameters(), 'lr': 1e-4},
        {'params': model.world_model.parameters(), 'lr': 5e-5},
        {'params': model.output_transform.text_coda.parameters(), 'lr': 1e-4},
        {'params': model.output_transform.text_decoder.parameters(), 'lr': 2e-4},
        {'params': model.output_transform.audio_coda.parameters(), 'lr': 1e-4},
        {'params': audio_decoder_only_params, 'lr': 2e-5},
        {'params': model.output_transform.audio_decoder.vocoder.parameters(), 'lr': 3e-5},
        {'params': model.output_transform.image_coda.parameters(), 'lr': 1e-4},
        {'params': model.output_transform.image_decoder.parameters(), 'lr': 2e-5},
    ], weight_decay=weight_decay)
    return optimizer