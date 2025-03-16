from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer, TrainerCallback
from transformers.integrations import TensorBoardCallback
from typing import Optional

import math
import os
import torch


def get_writer(trainer: Trainer):
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, TensorBoardCallback):
            if callback.tb_writer is not None:
                return callback.tb_writer
            
    return None


class GenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, prompts, generation_steps=2000):
        self.trainer: Optional[Trainer] = None
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.generation_steps = generation_steps
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if ((state.global_step == 1) or (state.global_step % self.generation_steps == 0)) and state.is_world_process_zero:
            writer = get_writer(self.trainer)
            if writer is None:
                print("No TensorBoard writer found, skipping generation...")
                return

            inputs = self.tokenizer(self.prompts, padding=True, return_tensors="pt").to(model.device)

            print(f"Generating text at step {state.global_step}...")
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=False,
                    max_length=100,
                    num_return_sequences=1,
                    do_sample=True,
                    top_p=0.92,
                    temperature=0.7,
                )
        
            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for i, text in enumerate(generated_texts):
                writer.add_text(f"generation/sample_{i}", text, state.global_step)


class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.trainer: Optional[Trainer] = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            print("No logs found, skipping...")
            return

        writer = get_writer(self.trainer)
        if writer is None:
            print("No TensorBoard writer found, skipping...")
            return

        self.add_perplexity(writer, logs, state)

        model = kwargs.get("model", None)
        tokenizer = kwargs.get("processing_class", None)
        if model is not None and tokenizer is not None:
            embedding_weights = model.get_input_embeddings().weight.data.clone().to(torch.float32).cpu().numpy()
            vocab = tokenizer.get_vocab()
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
            tokens = [token for token, _ in sorted_vocab]

            assert len(tokens) == embedding_weights.shape[0], "Mismatch between tokens and embedding weights"
            writer.add_embedding(
                mat=embedding_weights,
                metadata=tokens,
                tag='token_embeddings',
                global_step=state.global_step,
            )
        else:
            print("Model or tokenizer not found, skipping embedding logging...")

    def add_perplexity(self, writer, logs, state):
        is_eval = any(key.startswith("eval_") for key in logs.keys())
        loss_key = "eval_loss" if is_eval else "loss"

        if loss_key in logs:
            perplexity = math.exp(logs[loss_key])
            tag = "eval/perplexity" if is_eval else "train/perplexity"
            writer.add_scalar(tag, perplexity, state.global_step)
