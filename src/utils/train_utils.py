from typing import Optional

from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer, TrainerCallback
from transformers.integrations import TensorBoardCallback


def get_writer(trainer: Trainer) -> Optional[SummaryWriter]:
    """Get TensorBoard writer from trainer callbacks."""
    if hasattr(trainer, 'callback_handler'):
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback):
                if callback.tb_writer is not None:
                    return callback.tb_writer
    return None


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, stop_step: int):
        self.stop_step = stop_step

    def on_step_begin(self, args, state, control, **kwargs):
        if self.stop_step > 0 and state.global_step >= self.stop_step:
            print(f"Early stopping at step {state.global_step} as per stop_step={self.stop_step}.")
            control.should_training_stop = True


def check_tpu_availability():
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        
        device = xm.xla_device()
        print(f"TPU is available! Device: {device}")
        
        tpu_cores = xm.xrt_world_size()
        print(f"Number of TPU cores: {tpu_cores}")
        print(f"TPU type: {torch_xla._XLAC._get_tpu_type()}")
        
        return True
    except (ImportError, EnvironmentError, RuntimeError) as e:
        print(f"TPU is not available: {e}")
        return False
