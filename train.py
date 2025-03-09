from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

from model import megatransformer_causal

import argparse
import custom_callbacks
import custom_trainers
import megatransformer_utils
import os
import re


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
argparser.add_argument('--use_gradient_checkpointing', action='store_true', help='Whether to use gradient checkpointing')
argparser.add_argument('--fp16', action='store_true', help='Whether to use fp16')
argparser.add_argument('--bf16', action='store_true', help='Whether to use bf16')
argparser.add_argument('--use_xla', action='store_true', default=is_tpu_available, help='Whether to use XLA')

argparser.add_argument('--trainer', type=str, default="default", help='Trainer type: grokfast_ema, grokfast_ma, debug, or default')
argparser.add_argument('--grokfast_ema_alpha', type=float, default=0.98, help='Alpha for GrokFast EMA trainer')
argparser.add_argument('--grokfast_ema_lambda', type=float, default=2.0, help='Lambda for GrokFast EMA trainer')
argparser.add_argument('--grokfast_ma_window_size', type=int, default=100, help='Window size for GrokFast MA trainer')
argparser.add_argument('--grokfast_ma_lambda', type=float, default=5.0, help='Lambda for GrokFast MA trainer')
argparser.add_argument('--grokfast_ma_filter_type', type=str, default='mean', help='Filter type for GrokFast MA trainer')
argparser.add_argument('--grokfast_ma_warmup', action='store_true', help='Whether to use warmup for GrokFast MA trainer')

# deepspeed
argparser.add_argument('--use_deepspeed', action='store_true', help='Whether to use DeepSpeed')
argparser.add_argument('--deepspeed_config', type=str, default='ds_config.json', help='DeepSpeed configuration file')
argparser.add_argument('--zero_stage', type=int, default=3, help='ZeRO optimization stage (0, 1, 2, or 3)')
argparser.add_argument('--offload_optimizer', action='store_true', help='Offload optimizer states to CPU')
argparser.add_argument('--offload_param', action='store_true', help='Offload parameters to CPU')

# peft lora/int8 training
argparser.add_argument('--use_int8_peft', action='store_true', help='Use INT8 with PEFT/LoRA')
argparser.add_argument('--use_int8_deepspeed', action='store_true', help='Use DeepSpeed INT8 quantization')
argparser.add_argument('--lora_rank', type=int, default=16, help='Rank for LoRA adaptation')
argparser.add_argument('--lora_alpha', type=int, default=32, help='Alpha for LoRA adaptation')
argparser.add_argument('--lora_dropout', type=float, default=0.05, help='Dropout for LoRA adaptation')

args, unk = argparser.parse_known_args()

print(f"unknown args: {unk}")

run_dir = os.path.join("runs", "causal", args.dataset_name, args.run_name)

writer = SummaryWriter(run_dir)

megatransformer_utils.set_seed_everywhere(args.seed)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, add_bos_token=False)

print(f"default tokenizer.padding_side: {tokenizer.padding_side}")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.bos_token = tokenizer.eos_token
tokenizer.padding_side = "right"

if args.config == "gpt2":
    model_maker = megatransformer_causal.create_gpt2_model
elif args.config == "huginn":
    model_maker = megatransformer_causal.create_huginn_model
else:
    model_maker = megatransformer_causal.create_modern_model

model = model_maker(tokenizer, args.max_position_embeddings)

model = megatransformer_utils.setup_int8_training(args, model)

print(f"tokenizer: {tokenizer}")
print(f"model structure: {model}")
print(f"model parameters: {(sum(p.numel() for p in model.parameters())):,}")
print(f"trainable model parameters: {(sum(p.numel() for p in model.parameters() if p.requires_grad)):,}")

datasets = load_dataset(args.dataset_name, args.dataset_config_name)

def clean_dataset(examples):
    texts = examples["text"]
    cleaned_texts = []
    for text in texts:
        # replace special tokens
        cleaned = text.replace("@-@", "-")
        cleaned = cleaned.replace("@,@", ",")
        
        # fix spaces around punctuation
        for punct in ",.!?;:)]\"'":
            cleaned = cleaned.replace(f" {punct}", punct)
        
        for punct in "([\"'":
            cleaned = cleaned.replace(f"{punct} ", punct)
        
        # fix double spaces
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        
        # normalize multiple newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        if not cleaned.strip():
            continue

        cleaned_texts.append(cleaned)
    return {"text": cleaned_texts}

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=args.max_position_embeddings)

datasets = datasets.map(clean_dataset, batched=True)
tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

if args.use_deepspeed:
    if os.path.exists(args.deepspeed_config):
        print(f"Loading DeepSpeed config from {args.deepspeed_config}")
    else:
        raise FileNotFoundError(f"DeepSpeed config file {args.deepspeed_config} not found.")

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
    gradient_checkpointing=args.use_gradient_checkpointing,
    bf16=args.bf16,
    fp16=args.fp16,
    max_grad_norm=args.max_grad_norm,
    torch_compile=args.compile_model and not args.use_deepspeed,
    deepspeed=args.deepspeed_config if args.use_deepspeed else None,
)

argparser_args = args
if args.trainer == "grokfast_ema":
    trainer_maker = lambda *args, **kwargs: custom_trainers.GrokfastEMATrainer(*args, alpha=argparser_args.grokfast_ema_alpha, lamb=argparser_args.grokfast_ema_lambda, **kwargs)
elif args.trainer == "grokfast_ma":
    trainer_maker = lambda *args, **kwargs: custom_trainers.GrokFastMATrainer(*args, window_size=argparser_args.grokfast_ma_window_size, lamb=argparser_args.grokfast_ma_lambda, filter_type=argparser_args.grokfast_ma_filter_type, warmup=argparser_args.grokfast_ma_warmup, **kwargs)
elif args.trainer == "debug":
    trainer_maker = custom_trainers.DebugTrainer
else:
    trainer_maker = Trainer

trainer = trainer_maker(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
)

if not args.use_xla and not args.use_deepspeed:
    trainer.add_callback(custom_callbacks.GenerationCallback(
        writer,
        tokenizer=tokenizer,
        prompts=[
            "In this paper, we propose a novel approach to",
            "The Higgs boson, sometimes called the Higgs particle, is",
            "The capital of France is",
            "2 + 2 ="
        ],
        generation_steps=1000,
    ))
trainer.add_callback(custom_callbacks.PerplexityCallback(writer))

trainer.train()
