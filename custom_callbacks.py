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
        if ((state.global_step == 1) or (state.global_step % self.generation_steps == 0)) and state.is_world_process_zero:
            inputs = self.tokenizer(self.prompts, padding=True, return_tensors="pt").to(model.device)

            print(f"Generating text at step {state.global_step}...")
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=True,
                    max_length=100,
                    num_return_sequences=1,
                    do_sample=True,
                    top_p=0.92,
                    temperature=0.7,
                )
        
            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for i, text in enumerate(generated_texts):
                self.writer.add_text(f"generation/sample_{i}", text, state.global_step)


class DeepspeedGenerationCallback(TrainerCallback):
    def __init__(self, writer, tokenizer, prompts, generation_steps=2000):
        self.writer = writer
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.generation_steps = generation_steps
        
    def on_step_end(self, args, state, control, **kwargs):
        if ((state.global_step == 1) or (state.global_step % self.generation_steps == 0)) and state.is_world_process_zero:
            # Get the trainer instance
            trainer = kwargs.get('trainer', None)
            if trainer is None:
                print("Trainer not available in callback, skipping generation.")
                return
            
            # Get the unwrapped model for generation
            if hasattr(trainer, 'model_wrapped') and trainer.model_wrapped is not None:
                model = trainer.model_wrapped
            else:
                model = trainer.model
                
            # Handle DeepSpeed specifically
            if hasattr(trainer, 'deepspeed'):
                # If using DeepSpeed, we need to get the model differently
                print(f"Using DeepSpeed for generation at step {state.global_step}...")
                if hasattr(model, 'module'):
                    model = model.module
                
                # For ZeRO-3, we need to consolidate the model to run generation
                if hasattr(trainer.deepspeed, "zero3_enabled") and trainer.deepspeed.zero3_enabled:
                    # Temporarily consolidate for generation
                    with trainer.deepspeed.zero3_consolidated_model() as consolidated_model:
                        self._do_generation(consolidated_model, state)
                    return
            
            # Regular generation for non-DeepSpeed or ZeRO-1/2
            self._do_generation(model, state)
    
    def _do_generation(self, model, state):
        inputs = self.tokenizer(self.prompts, padding=True, return_tensors="pt").to(model.device)
        
        print(f"Generating text at step {state.global_step}...")
        
        with torch.no_grad():
            # Put model in eval mode for generation
            model_training = model.training
            model.eval()
            
            try:
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=True,
                    max_length=100,
                    num_return_sequences=1,
                    do_sample=True,
                    top_p=0.92,
                    temperature=0.7,
                )
                
                generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                for i, text in enumerate(generated_texts):
                    self.writer.add_text(f"generation/sample_{i}", text, state.global_step)
                    print(f"Sample {i}: {text[:100]}...")  # Print first 100 chars
                    
            except Exception as e:
                print(f"Generation failed with error: {e}")
            
            # Restore model's training state
            if model_training:
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
