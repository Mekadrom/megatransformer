from collections import deque
from torch import nn
from transformers import Trainer
from transformers.integrations import TensorBoardCallback
from typing import Optional, Literal

from dataset_loading import image_loading
from model import megatransformer_image_decoder

import megatransformer_utils
import torch


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
        self.grads = self.gradfilter_ma(model, self.grads, self.window_size, self.lamb, self.filter_type, self.warmup)
        return loss
    
    def gradfilter_ma(
        self, 
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


class GrokfastEMATrainer(Trainer):
    def __init__(self, *args, alpha=0.98, lamb=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.lamb = lamb
        self.grads = None
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs)
        # Apply gradfilter_ema after gradients are computed but before optimizer step
        self.grads = self.gradfilter_ema(model, self.grads, self.alpha, self.lamb)
        return loss
    
    def gradfilter_ema(
        self, 
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

        sanitized_model = megatransformer_utils.sanitize_model(model)

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
                        self.optional_log_loss(f"{prefix}kl_loss", stacked_losses[:, 2].mean(dim=0).item(), self.state.global_step)
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
                    token_correlation = megatransformer_utils.get_token_correlation(hidden_state)
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
