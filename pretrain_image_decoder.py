from transformers import AutoTokenizer, TrainingArguments

from dataset_loading import multimodal_dataset
from model import megatransformer_image_decoder

import custom_callbacks
import custom_trainers
import dataset_loading
import megatransformer_utils
import os
import torch


args, unk = megatransformer_utils.parse_args()
run_dir = os.path.join(args.logging_base_dir, args.run_name)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, add_bos_token=False)
print(f"default tokenizer.padding_side: {tokenizer.padding_side}")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.bos_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model_maker = megatransformer_image_decoder

model = model_maker.model_config_lookup(args.config)(tokenizer, args.max_position_embeddings, args.use_gradient_checkpointing)
model, model_loaded = megatransformer_utils.load_model(False, model, run_dir)

if args.local_rank == 0 or not args.use_deepspeed:
    print(f"model structure: {model}")
    print(f"model parameters: {(sum(p.numel() for p in model.parameters())):,}")
    print(f"trainable model parameters: {(sum(p.numel() for p in model.parameters() if p.requires_grad)):,}")

    print(f"model.text_recurrent parameters: {(sum(p.numel() for p in model.text_encoder.parameters())):,}")
    print(f"model.image_recon parameters: {(sum(p.numel() for p in model.image_recon.parameters())):,}")
    if isinstance(model.image_recon, megatransformer_image_decoder.ImageVAE):
        print(f"model.image_recon.encoder.parameters: {(sum(p.numel() for p in model.image_recon.encoder.parameters())):,}")
        print(f"model.image_recon.decoder.parameters: {(sum(p.numel() for p in model.image_recon.decoder.parameters())):,}")

    print(f"modified tokenizer: {tokenizer}")
    print(f"special tokens: {tokenizer.special_tokens_map}")

    print(f"DeepSpeed config path: {args.deepspeed_config}")
    print(f"DeepSpeed enabled: {args.use_deepspeed}")
    print(f"XLA enabled: {args.use_xla}")

model = megatransformer_utils.setup_int8_training(args, model)

if not os.path.exists(run_dir):
    os.makedirs(run_dir)

training_args = TrainingArguments(
    tpu_num_cores=8 if args.use_xla else None,
    output_dir=run_dir,
    overwrite_output_dir=True,
    lr_scheduler_type="cosine",
    warmup_ratio=args.warmup_ratio,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=1 if args.config == 'huginn' else args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    num_train_epochs=args.num_train_epochs if args.num_train_epochs > 0 else 1,
    max_steps=args.max_steps if args.max_steps > 0 else -1,
    weight_decay=args.weight_decay,
    report_to="tensorboard",
    logging_dir=run_dir,
    logging_steps=args.logging_steps,
    eval_strategy="steps",
    eval_steps=args.eval_steps,
    save_safetensors=False,
    save_steps=args.save_steps,
    gradient_checkpointing=args.use_gradient_checkpointing,
    bf16=args.bf16,
    fp16=args.fp16,
    max_grad_norm=args.max_grad_norm,
    torch_compile=args.compile_model and not args.use_deepspeed and not args.use_xla,
    deepspeed=args.deepspeed_config if args.use_deepspeed and not args.use_xla else None,
    use_cpu=args.cpu,
    log_level=args.log_level,
    logging_first_step=True,
    local_rank=args.local_rank,
)

train_dataset = dataset_loading.load_dataset(
    tokenizer,
    args.max_position_embeddings,
    "train",
    "image",
    image_size=model.config.image_size,
    streaming=True,
    cache_dir=args.dataset_cache_dir,
)

validation_dataset = dataset_loading.load_dataset(
    tokenizer,
    args.max_position_embeddings,
    "validation",
    "image",
    image_size=model.config.image_size,
    streaming=True,
    cache_dir=args.dataset_cache_dir,
)

data_collator = multimodal_dataset.DataCollatorForMultimodalLanguageModeling(
    tokenizer=tokenizer,
    max_position_embeddings=args.max_position_embeddings,
    image_size=model.config.image_size,
    modes=["image"],
    mlm=False,
)

optimizer = torch.optim.AdamW([
    {'params': model.text_encoder.parameters(), 'lr': 1e-4},
    {'params': model.image_recon.parameters(), 'lr': 1e-4},
], weight_decay=args.weight_decay)

trainer = custom_trainers.trainer_lookup(args, args.trainer)(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=tokenizer,
    optimizers=(optimizer, None)
)

generation_callback = custom_callbacks.ImageGenerationCallback(
    tokenizer=tokenizer,
    step_offset=args.start_step,
    generation_steps=args.generation_steps,
)
trainer.add_callback(generation_callback)
generation_callback.trainer = trainer

metrics_callback = custom_callbacks.MetricsCallback(step_offset=args.start_step, is_add_perplexity=False)
trainer.add_callback(metrics_callback)
metrics_callback.trainer = trainer

print(f"Starting training with {sum(p.numel() for p in model.parameters()):,} parameters")
trainer.train()
