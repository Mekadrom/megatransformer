from transformers import TrainerCallback

import math
import torch


class GenerationCallback(TrainerCallback):
    def __init__(self, writer, tokenizer, prompts, generation_steps=2000):
        self.writer = writer
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.generation_steps = generation_steps
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if (state.global_step % self.generation_steps == 0) and state.is_world_process_zero:
            device = next(model.parameters()).device
            
            inputs = self.tokenizer(self.prompts, padding=True, return_tensors="pt").to(device)
            
            model.eval()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=100,
                    num_return_sequences=1,
                    do_sample=True,
                    top_p=0.92,
                    temperature=0.7,
                )
            
            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for i, text in enumerate(generated_texts):
                self.writer.add_text(f"generation/sample_{i}", text, state.global_step)
            
            model.train()


class PerplexityCallback(TrainerCallback):
    def __init__(self, writer):
        self.writer = writer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        is_eval = any(key.startswith("eval_") for key in logs.keys())
        loss_key = "eval_loss" if is_eval else "loss"

        if loss_key in logs:
            perplexity = math.exp(logs[loss_key])
            tag = "eval/perplexity" if is_eval else "train/perplexity"
            self.writer.add_scalar(tag, perplexity, state.global_step)
