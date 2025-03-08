from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter

from . import megatransformer_utils
from .model import megatransformer_causal

import argparse
import math
import os
import re
import torch


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


def create_gpt2_model(tokenizer, max_position_embeddings):
    # gpt2-small closest equivalent (~124M params)
    return megatransformer_causal.MegaTransformerCausalLMHead(megatransformer_utils.MegaTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=max_position_embeddings,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    ))

def create_modern_model(tokenizer, max_position_embeddings):
    # uses more modern approaches to causal language modeling (~148M params)
    return megatransformer_causal.MegaTransformerCausalLMHead(megatransformer_utils.MegaTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=max_position_embeddings,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_activation="swiglu",
        norm_type="rmsnorm",
        ffn_type="mlp",
        use_positional_embedding=False,
        use_sinusoidal_embedding=False,
        use_rotary_embedding=False,
        use_alibi_bias=True,
        use_qkv_bias=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    ))

def create_huginn_model(tokenizer, max_position_embeddings):
    # uses a recurrent approach to emulate a deeper model
    return megatransformer_causal.MegaTransformerCausalLMHead(megatransformer_utils.MegaTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=max_position_embeddings,
        hidden_size=768,
        num_hidden_layers=None,
        num_prelude_layers=2,
        num_recurrent_layers=4,
        num_coda_layers=2,
        num_attention_heads=12,
        intermediate_activation="swiglu",
        norm_type="rmsnorm",
        ffn_type="gated",
        use_positional_embedding=False,
        use_sinusoidal_embedding=False,
        use_rotary_embedding=False,
        use_alibi_bias=True,
        use_qkv_bias=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    ))


is_tpu_available = megatransformer_utils.check_tpu_availability()
print(f"TPU available: {is_tpu_available}")

argparser = argparse.ArgumentParser()

argparser.add_argument('--run_name', type=str, help='Name of the run')
argparser.add_argument('--seed', type=int, default=42, help='Random seed')
argparser.add_argument('--compile_model', action='store_true', help='Whether to compile the model')
argparser.add_argument('--tokenizer_name', type=str, default="mistralai/Mistral-7B-v0.1", help='Tokenizer name')
argparser.add_argument('--config', type=str, default="modern", help='Model configuration: gpt2, modern, or huginn')

argparser.add_argument('--dataset_name', type=str, default='wikitext', help='Dataset name')
argparser.add_argument('--dataset_config_name', type=str, default='wikitext-103-v1', help='Dataset config name')
argparser.add_argument('--max_position_embeddings', type=int, default=1024, help='Max position embeddings (maximum sequence length)')

argparser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
argparser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
argparser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs')
argparser.add_argument('--batch_size', type=int, default=4, help='Batch size')
argparser.add_argument('--gradient_accumulation_steps', type=int, default=32, help='Gradient accumulation steps')
argparser.add_argument('--warmup_ratio', type=float, default=0.03, help='Warmup ratio')
argparser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
argparser.add_argument('--fp16', action='store_true', help='Whether to use fp16')
argparser.add_argument('--bf16', action='store_true', help='Whether to use bf16')
argparser.add_argument('--use_xla', action='store_true', default=is_tpu_available, help='Whether to use XLA')

args = argparser.parse_args()

run_dir = os.path.join("runs", "causal", "wikitext", args.run_name)

writer = SummaryWriter(run_dir)

megatransformer_utils.set_seed_everywhere(args.seed)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, add_bos_token=False)

print(f"default tokenizer.padding_side: {tokenizer.padding_side}")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.bos_token = tokenizer.eos_token
tokenizer.padding_side = "right"

if args.config == "gpt2":
    model_maker = create_gpt2_model
elif args.config == "huginn":
    model_maker = create_huginn_model
else:
    model_maker = create_modern_model

model = model_maker(tokenizer, args.max_position_embeddings)

print(f"tokenizer: {tokenizer}")
print(f"model structure: {model}")
print(f"model parameters: {(sum(p.numel() for p in model.parameters())):,}")
print(f"trainable model parameters: {(sum(p.numel() for p in model.parameters() if p.requires_grad)):,}")

datasets = load_dataset(args.dataset_name, args.dataset_config_name)

def clean_dataset(examples):
    texts = examples["text"]
    cleaned_texts = []
    for text in texts:
        # Replace special tokens
        cleaned = text.replace("@-@", "-")
        cleaned = cleaned.replace("@,@", ",")
        
        # Fix spaces around punctuation
        for punct in ",.!?;:)]\"'":
            cleaned = cleaned.replace(f" {punct}", punct)
        
        for punct in "([\"'":
            cleaned = cleaned.replace(f"{punct} ", punct)
        
        # Fix double spaces
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        
        # Normalize multiple newlines (optional)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        if not cleaned.strip():
            continue

        cleaned_texts.append(cleaned)
    return {"text": cleaned_texts}

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=args.max_position_embeddings, padding_side="right")

datasets = datasets.map(clean_dataset, batched=True)
tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(
    # i'm basically only ever going to be able to afford the smallest tpu-v2 or v3
    # tpu_num_cores=8 if args.use_xla else None,

    output_dir=run_dir,
    overwrite_output_dir=True,
    learning_rate=args.learning_rate,
    lr_scheduler_type="cosine",
    warmup_ratio=args.warmup_ratio,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    num_train_epochs=args.num_train_epochs,
    weight_decay=args.weight_decay,
    report_to="tensorboard",
    logging_dir=run_dir,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=1000,
    save_safetensors=False,
    save_steps=500,
    bf16=args.bf16,
    fp16=args.fp16,
    max_grad_norm=args.max_grad_norm,
    torch_compile=args.compile_model,
)

generation_callback = GenerationCallback(
    writer,
    tokenizer=tokenizer,
    prompts=[
        "In this paper, we propose a novel approach to",
        "The Higgs boson, sometimes called the Higgs particle, is",
        "The capital of France is",
        "2 + 2 ="
    ],
    generation_steps=1000,
)

perplexity_callback = PerplexityCallback(writer)

# trainer = DebugTrainer(
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
)

trainer.add_callback(generation_callback)
trainer.add_callback(perplexity_callback)

trainer.train()
eval_results = trainer.evaluate()
