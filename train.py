from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments, DataCollatorForLanguageModeling

import custom_callbacks
import custom_trainers
import megatransformer_utils
import os


args, unk = megatransformer_utils.parse_args()
run_dir = os.path.join("runs", "causal", args.dataset_name, args.run_name)

tokenizer, model = megatransformer_utils.load_model_and_tokenizer(args, run_dir)

if not os.path.exists(run_dir):
    os.makedirs(run_dir)

writer = SummaryWriter(run_dir)
training_args = TrainingArguments(
    # tpu_num_cores=8 if args.use_xla else None,
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
    torch_compile=args.compile_model and not args.use_deepspeed,
    deepspeed=args.deepspeed_config if args.use_deepspeed else None,
)
datasets = megatransformer_utils.make_datasets(args.dataset_name, args.dataset_config_name, tokenizer, args.max_position_embeddings)
trainer = custom_trainers.trainer_lookup(args, args.trainer)(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=datasets["train"],
    eval_dataset=datasets["validation"],
    processing_class=tokenizer,
    callbacks=[
        custom_callbacks.GenerationCallback(
            writer,
            tokenizer=tokenizer,
            prompts=[
                "In this paper, we propose a novel approach to",
                "The Higgs boson, sometimes called the Higgs particle, is",
                "The capital of France is",
                "2 + 2 ="
            ],
            generation_steps=args.generation_steps,
        ),
        custom_callbacks.PerplexityCallback(writer)
    ]
)

trainer.train()
