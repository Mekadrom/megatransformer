from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling

from model import megatransformer_causal

import custom_callbacks
import custom_trainers
import dataset_loading
import megatransformer_utils
import os


args, unk = megatransformer_utils.parse_args()
run_dir = os.path.join(args.logging_base_dir, args.dataset_name, args.run_name)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, add_bos_token=False)
print(f"default tokenizer.padding_side: {tokenizer.padding_side}")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.bos_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print(f"modified tokenizer: {tokenizer}")

model = megatransformer_causal.model_config_lookup(args.config)(tokenizer, args.max_position_embeddings)
model = megatransformer_utils.load_model(False, model, run_dir)
model = megatransformer_utils.setup_int8_training(args, model)

if not os.path.exists(run_dir):
    os.makedirs(run_dir)

writer = SummaryWriter(run_dir)
training_args = TrainingArguments(
    tpu_num_cores=8 if args.use_xla else None,
    output_dir=run_dir,
    overwrite_output_dir=True,
    learning_rate=args.learning_rate,
    lr_scheduler_type="cosine",
    warmup_ratio=args.warmup_ratio,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=1 if args.config == 'huginn' else args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    num_train_epochs=args.num_train_epochs,
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
)
datasets = dataset_loading.load_dataset(args.dataset_name, args.dataset_config_name, tokenizer, args.max_position_embeddings)

generation_callback = custom_callbacks.GenerationCallback(
    tokenizer=tokenizer,
    prompts=[
        "In this paper, we propose a novel approach to",
        "The Higgs boson, sometimes called the Higgs particle, is",
        "The capital of France is",
        "2 + 2 ="
    ],
    generation_steps=args.generation_steps,
)
metrics_callback = custom_callbacks.MetricsCallback()

trainer = custom_trainers.trainer_lookup(args, args.trainer)(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=datasets["train"],
    eval_dataset=datasets["validation"],
    processing_class=tokenizer,
    callbacks=[generation_callback, metrics_callback]
)

generation_callback.trainer = trainer
metrics_callback.trainer = trainer

print(f"Starting training with {sum(p.numel() for p in model.parameters()):,} parameters")
trainer.train()
