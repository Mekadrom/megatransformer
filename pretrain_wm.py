import os

from model.audio.shared_window_buffer import SharedWindowBuffer

os.environ["DEEPSPEED_UNIT_TEST"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ['NCCL_TIMEOUT'] = '1200000'

from transformers import AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback

from dataset_loading import multimodal_dataset
from model import megatransformer_audio_decoder, megatransformer_audio_encoder, megatransformer_causal, megatransformer_image_decoder, megatransformer_image_encoder, megatransformer_multimodal, megatransformer_recurrent, megatransformer_text_encoder

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

shared_window_buffer = SharedWindowBuffer()

model = model_maker.model_config_lookup(args.config)(tokenizer, args.max_position_embeddings)
model, model_loaded = megatransformer_utils.load_model(False, model, run_dir)

if args.local_rank == 0 or not args.use_deepspeed:
    print(f"model structure: {model}")
    print(f"model parameters: {(sum(p.numel() for p in model.parameters())):,}")
    print(f"trainable model parameters: {(sum(p.numel() for p in model.parameters() if p.requires_grad)):,}")

    if len(args.include_modes) > 1 or "text" not in args.include_modes:
        print(f"\tmodel.input_transform parameters: {(sum(p.numel() for p in model.input_transform.parameters())):,}")
        print(f"\t\tmodel.input_transform.text_embedding parameters: {(sum(p.numel() for p in model.input_transform.text_embedding.parameters())):,}")
        print(f"\t\t\tmodel.input_transform.text_embedding.wte parameters: {(sum(p.numel() for p in model.input_transform.text_embedding.wte.parameters())):,}")
        print(f"\t\t\tmodel.input_transform.text_embedding.prelude parameters: {(sum(p.numel() for p in model.input_transform.text_embedding.prelude.parameters())):,}")

        print(f"\t\tmodel.input_transform.audio_embedding parameters: {(sum(p.numel() for p in model.input_transform.audio_embedding.parameters())):,}")
        print(f"\t\t\tmodel.input_transform.audio_embedding.conv_feature_extractor parameters: {(sum(p.numel() for p in model.input_transform.audio_embedding.conv_feature_extractor.parameters())):,}")
        print(f"\t\t\tmodel.input_transform.audio_embedding.conv_projection parameters: {(sum(p.numel() for p in model.input_transform.audio_embedding.conv_projection.parameters())):,}")
        print(f"\t\t\tmodel.input_transform.audio_embedding.prelude parameters: {(sum(p.numel() for p in model.input_transform.audio_embedding.prelude.parameters())):,}")

        print(f"\t\tmodel.input_transform.image_embedding parameters: {(sum(p.numel() for p in model.input_transform.image_embedding.parameters())):,}")
        print(f"\t\t\tmodel.input_transform.image_embedding.patch_embed parameters: {(sum(p.numel() for p in model.input_transform.image_embedding.patch_embed.parameters())):,}")
        print(f"\t\t\tmodel.input_transform.image_embedding.prelude parameters: {(sum(p.numel() for p in model.input_transform.image_embedding.prelude.parameters())):,}")

        print(f"\tmodel.world_model parameters: {(sum(p.numel() for p in model.world_model.parameters())):,}")

        print(f"\tmodel.output_transform parameters: {(sum(p.numel() for p in model.output_transform.parameters())):,}")
        print(f"\t\tmodel.output_transform.text_coda parameters: {(sum(p.numel() for p in model.output_transform.text_coda.parameters())):,}")
        print(f"\t\tmodel.output_transform.text_decoder parameters: {(sum(p.numel() for p in model.output_transform.text_decoder.parameters())):,}")

        print(f"\t\tmodel.output_transform.audio_coda parameters: {(sum(p.numel() for p in model.output_transform.audio_coda.parameters())):,}")
        print(f"\t\tmodel.output_transform.audio_decoder parameters: {(sum(p.numel() for p in model.output_transform.audio_decoder.parameters())):,}")
        if hasattr(model.output_transform.audio_decoder, "vocoder"):
            print(f"\t\t\tmodel.output_transform.audio_decoder.vocoder parameters: {(sum(p.numel() for p in model.output_transform.audio_decoder.vocoder.parameters())):,}")
        print(f"\t\t\tmodel.output_transform.audio_decoder.unet parameters: {(sum(p.numel() for p in model.output_transform.audio_decoder.unet.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.audio_decoder.unet.time_transform parameters: {(sum(p.numel() for p in model.output_transform.audio_decoder.unet.time_transform.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.audio_decoder.unet.init_conv parameters: {(sum(p.numel() for p in model.output_transform.audio_decoder.unet.init_conv.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.audio_decoder.unet.down_blocks parameters: {(sum(p.numel() for p in model.output_transform.audio_decoder.unet.down_blocks.parameters())):,}")
        for d, down_block in enumerate(model.output_transform.audio_decoder.unet.down_blocks):
            print(f"\t\t\t\t\tmodel.output_transform.audio_decoder.unet.down_blocks[{d}] parameters: {(sum(p.numel() for p in down_block.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.audio_decoder.unet.middle_res_block parameters: {(sum(p.numel() for p in model.output_transform.audio_decoder.unet.middle_res_block.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.audio_decoder.unet.middle_attn_block parameters: {(sum(p.numel() for p in model.output_transform.audio_decoder.unet.middle_attn_block.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.audio_decoder.unet.middle_res_block2 parameters: {(sum(p.numel() for p in model.output_transform.audio_decoder.unet.middle_res_block2.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.audio_decoder.unet.up_blocks parameters: {(sum(p.numel() for p in model.output_transform.audio_decoder.unet.up_blocks.parameters())):,}")
        for u, up_block in enumerate(model.output_transform.audio_decoder.unet.up_blocks):
            print(f"\t\t\t\t\tmodel.output_transform.audio_decoder.unet.up_blocks[{u}] parameters: {(sum(p.numel() for p in up_block.parameters())):,}")
            print(f"\t\t\t\t\t\tmodel.output_transform.audio_decoder.unet.up_blocks[{u}].upsample parameters: {(sum(p.numel() for p in up_block.upsample.parameters())):,}")
            print(f"\t\t\t\t\t\tmodel.output_transform.audio_decoder.unet.up_blocks[{u}].res_blocks parameters: {(sum(p.numel() for p in up_block.res_blocks.parameters())):,}")
            print(f"\t\t\t\t\t\tmodel.output_transform.audio_decoder.unet.up_blocks[{u}].attn_blocks parameters: {(sum(p.numel() for p in up_block.attn_blocks.parameters())):,}")
            print(f"\t\t\t\t\t\tmodel.output_transform.audio_decoder.unet.up_blocks[{u}].cross_attn_blocks parameters: {(sum(p.numel() for p in up_block.cross_attn_blocks.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.audio_decoder.unet.final_res_block parameters: {(sum(p.numel() for p in model.output_transform.audio_decoder.unet.final_res_block.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.audio_decoder.unet.final_conv parameters: {(sum(p.numel() for p in model.output_transform.audio_decoder.unet.final_conv.parameters())):,}")

        print(f"\t\tmodel.output_transform.image_coda parameters: {(sum(p.numel() for p in model.output_transform.image_coda.parameters())):,}")
        print(f"\t\tmodel.output_transform.image_decoder parameters: {(sum(p.numel() for p in model.output_transform.image_decoder.parameters())):,}")
        print(f"\t\t\tmodel.output_transform.image_decoder.unet parameters: {(sum(p.numel() for p in model.output_transform.image_decoder.unet.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.image_decoder.unet.time_transform parameters: {(sum(p.numel() for p in model.output_transform.image_decoder.unet.time_transform.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.image_decoder.unet.init_conv parameters: {(sum(p.numel() for p in model.output_transform.image_decoder.unet.init_conv.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.image_decoder.unet.down_blocks parameters: {(sum(p.numel() for p in model.output_transform.image_decoder.unet.down_blocks.parameters())):,}")
        for d, down_block in enumerate(model.output_transform.image_decoder.unet.down_blocks):
            print(f"\t\t\t\t\tmodel.output_transform.image_decoder.unet.down_blocks[{d}] parameters: {(sum(p.numel() for p in down_block.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.image_decoder.unet.middle_res_block parameters: {(sum(p.numel() for p in model.output_transform.image_decoder.unet.middle_res_block.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.image_decoder.unet.middle_attn_block parameters: {(sum(p.numel() for p in model.output_transform.image_decoder.unet.middle_attn_block.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.image_decoder.unet.middle_res_block2 parameters: {(sum(p.numel() for p in model.output_transform.image_decoder.unet.middle_res_block2.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.image_decoder.unet.up_blocks parameters: {(sum(p.numel() for p in model.output_transform.image_decoder.unet.up_blocks.parameters())):,}")
        for u, up_block in enumerate(model.output_transform.image_decoder.unet.up_blocks):
            print(f"\t\t\t\t\tmodel.output_transform.image_decoder.unet.up_blocks[{u}] parameters: {(sum(p.numel() for p in up_block.parameters())):,}")
            print(f"\t\t\t\t\t\tmodel.output_transform.image_decoder.unet.up_blocks[{u}].upsample parameters: {(sum(p.numel() for p in up_block.upsample.parameters())):,}")
            print(f"\t\t\t\t\t\tmodel.output_transform.image_decoder.unet.up_blocks[{u}].res_blocks parameters: {(sum(p.numel() for p in up_block.res_blocks.parameters())):,}")
            print(f"\t\t\t\t\t\tmodel.output_transform.image_decoder.unet.up_blocks[{u}].attn_blocks parameters: {(sum(p.numel() for p in up_block.attn_blocks.parameters())):,}")
            print(f"\t\t\t\t\t\tmodel.output_transform.image_decoder.unet.up_blocks[{u}].cross_attn_blocks parameters: {(sum(p.numel() for p in up_block.cross_attn_blocks.parameters())):,}")
            for c, cross_attn_block in enumerate(up_block.cross_attn_blocks):
                print(f"\t\t\t\t\t\t\tmodel.output_transform.image_decoder.unet.up_blocks[{u}].cross_attn_blocks[{c}] parameters: {(sum(p.numel() for p in cross_attn_block.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.image_decoder.unet.final_res_block parameters: {(sum(p.numel() for p in model.output_transform.image_decoder.unet.final_res_block.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.image_decoder.unet.final_conv parameters: {(sum(p.numel() for p in model.output_transform.image_decoder.unet.final_conv.parameters())):,}")

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
    # eval_strategy="steps",
    eval_strategy="no",
    # eval_steps=args.eval_steps,
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

# print(f"Training arguments: {training_args}")

text_weight = 1.0 if "text" in args.include_modes else 0.0
audio_weight = 1.0 if "audio" in args.include_modes else 0.0
image_weight = 1.0 if "image" in args.include_modes else 0.0

train_dataset = multimodal_dataset.MultimodalDataset(
    model.config,
    approximated_length=300_000,
    tokenizer=tokenizer,
    sample_rate=model.config.audio_sample_rate,
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
    shared_window_buffer=shared_window_buffer
)

validation_dataset = multimodal_dataset.MultimodalDataset(
    model.config,
    approximated_length=200_000,
    tokenizer=tokenizer,
    sample_rate=model.config.audio_sample_rate,
    n_mels=model.config.audio_n_mels,
    n_fft=model.config.audio_n_fft,
    hop_length=model.config.audio_hop_length,
    audio_max_frames=model.config.audio_max_frames,
    image_size=model.config.image_size,
    cache_dir=args.dataset_cache_dir,
    text_weight=0.0, # no text in validation
    audio_weight=audio_weight,
    image_weight=image_weight,
    split="validation",
    seed=args.seed,
    max_position_embeddings=args.max_position_embeddings,
)

data_collator: DataCollatorForLanguageModeling
if 'multimodal' in args.config.lower():
    data_collator = multimodal_dataset.DataCollatorForMultimodalLanguageModeling(
        tokenizer=tokenizer,
        max_position_embeddings=args.max_position_embeddings,
        image_size=model.config.image_size,
        audio_max_frames=model.config.audio_max_frames,
        audio_max_waveform_length=model.config.audio_max_waveform_length,
        modes=args.include_modes,
        mlm=False,
    )
else:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

optimizer = None
if "multimodal" in args.config.lower() and not "frankenstein" in args.config.lower():
    optimizer = megatransformer_utils.create_multimodal_optimizer(model, args.weight_decay)

trainer = custom_trainers.trainer_lookup(args, args.trainer)(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=tokenizer,
    optimizers=(optimizer, None)
)

model.config.include_modes = args.include_modes.split(",") if isinstance(args.include_modes, str) else args.include_modes

prompts = [
    "In this paper, we propose a novel approach to",
    "The Higgs boson, sometimes called the Higgs particle, is",
    "The capital of France is",
    "2 + 2 ="
]

generation_callback: TrainerCallback
if 'multimodal' in args.config.lower():
    generation_callback = custom_callbacks.MultimodalGenerationCallback(
        tokenizer=tokenizer,
        text_only_prompts=prompts,
        step_offset=args.start_step,
        generation_steps=args.generation_steps,
    )
    trainer.add_callback(generation_callback)
    generation_callback.trainer = trainer
else:
    # todo: implement for multimodal
    generation_callback = custom_callbacks.GenerationCallback(
        tokenizer=tokenizer,
        prompts=prompts,
        step_offset=args.start_step,
        generation_steps=args.generation_steps,
    )
    trainer.add_callback(generation_callback)
    generation_callback.trainer = trainer

metrics_callback = custom_callbacks.MetricsCallback(step_offset=args.start_step)
trainer.add_callback(metrics_callback)
metrics_callback.trainer = trainer

print(f"Starting training with {sum(p.numel() for p in model.parameters()):,} parameters")
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
