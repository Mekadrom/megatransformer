from transformers import AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling

from dataset_loading import multimodal_dataset
from model import megatransformer_causal, megatransformer_multimodal, megatransformer_recurrent

import custom_callbacks
import custom_trainers
import megatransformer_utils
import os


args, unk = megatransformer_utils.parse_args()
run_dir = os.path.join(args.logging_base_dir, args.run_name)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, add_bos_token=False)
print(f"default tokenizer.padding_side: {tokenizer.padding_side}")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.bos_token = tokenizer.eos_token
tokenizer.padding_side = "right"

if len(args.include_modes) > 1 or "text" not in args.include_modes:
    # multimodal supports mixed mode training, but also single mode for audio transcription/generation and image description/generation
    model_maker = megatransformer_multimodal
elif 'recurrent' in args.config:
    model_maker = megatransformer_recurrent
else:
    model_maker = megatransformer_causal

model = model_maker.model_config_lookup(args.config)(tokenizer, args.max_position_embeddings)
model = megatransformer_utils.load_model(False, model, run_dir)
model = megatransformer_utils.setup_int8_training(args, model)

print(f"modified tokenizer: {tokenizer}")

print(f"special tokens: {tokenizer.special_tokens_map}")

if not os.path.exists(run_dir):
    os.makedirs(run_dir)

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

print(f"DeepSpeed config path: {args.deepspeed_config}")
print(f"DeepSpeed enabled: {args.use_deepspeed}")
print(f"XLA enabled: {args.use_xla}")
print(f"Final DeepSpeed setting: {training_args.deepspeed}")

text_examples = 10_000
audio_examples = 10_000
image_examples = 10_000

text_weight = 1.0 if "text" in args.include_modes else 0.0
audio_weight = 1.0 if "audio" in args.include_modes else 0.0
image_weight = 1.0 if "image" in args.include_modes else 0.0

train_dataset = multimodal_dataset.MultimodalDataset(
    approximated_length=text_examples + audio_examples + image_examples,
    tokenizer=tokenizer,
    n_mels=model.config.audio_n_mels,
    n_fft=model.config.audio_n_fft,
    hop_length=model.config.audio_hop_length,
    audio_max_frames=model.config.audio_max_frames,
    image_size=model.config.image_size,
    cache_dir=args.dataset_cache_dir,
    text_weight=text_weight,
    audio_weight=audio_weight,
    image_weight=image_weight,
    split="train",
    seed=args.seed,
    max_position_embeddings=args.max_position_embeddings,
)

validation_dataset = multimodal_dataset.MultimodalDataset(
    approximated_length=3_760 + 9_150 + 12_400,
    tokenizer=tokenizer,
    n_mels=model.config.audio_n_mels,
    n_fft=model.config.audio_n_fft,
    hop_length=model.config.audio_hop_length,
    audio_max_frames=model.config.audio_max_frames,
    image_size=model.config.image_size,
    cache_dir=args.dataset_cache_dir,
    text_weight=text_weight,
    audio_weight=audio_weight,
    image_weight=image_weight,
    split="validation",
    seed=args.seed,
    max_position_embeddings=args.max_position_embeddings,
)

if 'multimodal' in args.config.lower():
    data_collator = multimodal_dataset.DataCollatorForMultimodalLanguageModeling(
        tokenizer=tokenizer,
        max_position_embeddings=args.max_position_embeddings,
        image_size=model.config.image_size,
        audio_max_frames=model.config.audio_max_frames,
        mlm=False
    )
else:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = custom_trainers.trainer_lookup(args, args.trainer)(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=tokenizer,
)

if 'multimodal' in args.config.lower():
    generation_callback = custom_callbacks.MultimodalGenerationCallback(
        tokenizer=tokenizer,
        text_only_prompts=[
            "In this paper, we propose a novel approach to",
            "The Higgs boson, sometimes called the Higgs particle, is",
            "The capital of France is",
            "2 + 2 ="
        ],
        generation_steps=args.generation_steps,
    )
    trainer.add_callback(generation_callback)
    generation_callback.trainer = trainer
else:
    # todo: implement for multimodal
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
    trainer.add_callback(generation_callback)
    generation_callback.trainer = trainer

metrics_callback = custom_callbacks.MetricsCallback()
trainer.add_callback(metrics_callback)
metrics_callback.trainer = trainer

print(f"Starting training with {sum(p.numel() for p in model.parameters()):,} parameters")
trainer.train()
