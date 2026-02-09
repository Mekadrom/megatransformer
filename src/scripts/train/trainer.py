import abc
import math

import torch


from typing import Any, Mapping, Optional, Union

from transformers.integrations.integration_utils import TensorBoardCallback
from transformers.trainer import Trainer


class CommonTrainer(abc.ABC, Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        Override to use shard-aware sampler for sharded datasets.

        This ensures samples are grouped by shard, minimizing disk I/O
        by loading each shard only once per epoch.
        """
        if self._shard_sampler is not None:
            # Update epoch for proper shuffling reproducibility
            epoch = 0
            if self.state is not None and self.state.epoch is not None:
                epoch = int(self.state.epoch)
            self._shard_sampler.set_epoch(epoch)
            return self._shard_sampler

        # Fall back to default sampler for non-sharded datasets
        return super()._get_train_sampler()

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            if self.is_deepspeed_enabled and (torch.is_floating_point(data)):
                kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data

    def is_gan_enabled(self, global_step: int, vae_loss: torch.Tensor) -> bool:
        """
        Check if GAN training should be enabled based on the configured conditions.

        Supports two modes:
        - "step": Start GAN training after a specific step
        - "reconstruction_criteria_met": Start when VAE loss drops below threshold

        Once GAN training starts, it stays enabled (via gan_already_started flag).
        """
        if self.discriminator is None:
            return False

        if self.gan_already_started:
            return True

        if self.gan_start_condition_key is None:
            # Legacy mode: always enabled if discriminator exists
            return True

        if self.gan_start_condition_key == "step":
            return global_step >= int(self.gan_start_condition_value)

        if self.gan_start_condition_key == "reconstruction_criteria_met":
            # Start GAN when VAE loss drops below threshold
            threshold = float(self.gan_start_condition_value)
            return vae_loss.item() < threshold

        return False

    def _log_scalar(self, tag, value, global_step, skip_zero=True):
        if self.writer is not None:
            if isinstance(value, torch.Tensor):
                value = value.detach().item()
            # Skip zero values by default (for unused losses), but allow explicit logging of zeros
            if not skip_zero or value != 0.0:
                self.writer.add_scalar(tag, value, global_step)

    def _ensure_tensorboard_writer(self):
        if hasattr(self, "writer") and self.writer is not None:
            return

        for callback in self.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback):
                self.writer = callback.tb_writer
                return

        self.writer = None

    @abc.abstractmethod
    def start_train_print(self, args):
        pass

    def get_loss_weight(
        self,
        base_weight: float,
        step: int,
        start_step: int,
        rampup_steps: int,
        anneal_start_step: int,
        anneal_steps: int,
        min_ratio: float = 0.1
    ) -> float:
        """
        Compute loss weight with ramp-up and ramp-down phases.
        
        Phases:
            1. Before start_step: weight = 0
            2. Ramp-up: start_step -> start_step + rampup_steps (0 -> base_weight)
            3. Plateau: start_step + rampup_steps -> anneal_start_step (base_weight)
            4. Ramp-down: anneal_start_step -> anneal_start_step + anneal_steps (base_weight -> base_weight * min_ratio)
            5. After ramp-down: weight = base_weight * min_ratio
        """
        rampup_end = start_step + rampup_steps
        anneal_end = anneal_start_step + anneal_steps
        
        assert rampup_end <= anneal_start_step, "Ramp-up must end before anneal starts"

        # Phase 1: Before start - weight is 0
        if step < start_step:
            return 0.0
        
        # Phase 2: Ramp-up (cosine from 0 to base_weight)
        if step < rampup_end:
            if rampup_steps == 0:
                return base_weight
            progress = (step - start_step) / rampup_steps
            # Cosine ramp-up: starts at 0, ends at 1
            ramp = 0.5 * (1 - math.cos(math.pi * progress))
            return base_weight * ramp
        
        # Phase 3: Plateau at base_weight
        if step < anneal_start_step:
            return base_weight
        
        # Phase 4: Ramp-down (cosine from base_weight to base_weight * min_ratio)
        if step < anneal_end:
            if anneal_steps == 0:
                return base_weight * min_ratio
            progress = (step - anneal_start_step) / anneal_steps
            ratio = min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
            return base_weight * ratio
        
        # Phase 5: After ramp-down - hold at minimum
        return base_weight * min_ratio
