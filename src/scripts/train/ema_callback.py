import os

import torch


from typing import Optional

from transformers.trainer_callback import TrainerCallback

from model.ema import EMAModel


class EMAUpdateCallback(TrainerCallback):
    """Callback to update EMA weights after each training step and save/load EMA state."""

    def __init__(self, ema: Optional[EMAModel] = None):
        self.ema = ema

    def on_step_end(self, args, state, control, **kwargs):
        if self.ema is not None:
            self.ema.update()

    def on_save(self, args, state, control, **kwargs):
        """Save EMA weights alongside model checkpoint."""
        if self.ema is None:
            return

        # Determine checkpoint directory
        checkpoint_folder = f"checkpoint-{state.global_step}"
        output_dir = os.path.join(args.output_dir, checkpoint_folder)

        # Save EMA state dict
        ema_path = os.path.join(output_dir, "ema_state.pt")
        torch.save(self.ema.state_dict(), ema_path)
        print(f"Saved EMA state to {ema_path}")
