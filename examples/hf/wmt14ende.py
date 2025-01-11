from datasets import load_dataset
from megatransformer import config, custom_trainers, lr_scheduler, megatransformer, visualization_helper
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, set_seed, Seq2SeqTrainingArguments

import argparse
import evaluate
import numpy as np
import os
import torch

argparser = argparse.ArgumentParser()
argparser.add_argument('--run_name', type=str, required=True)
argparser.add_argument('--config', type=str, default=os.path.join('config', 'aiayn.yaml'))

args, unk = argparser.parse_known_args()

model_config = config.TransformerConfig()
model_config.load_yaml(args.config)

# load command line overrides (unk args since they aren't registered above)
model_config.__dict__.update({k: v for k, v in zip(unk[::2], unk[1::2])})

run_dir = os.path.join('runs', args.run_name)
os.makedirs(run_dir, exist_ok=True)

set_seed(42)
n_epochs = 1
batch_size = 32
moe_diversity_loss_coefficient = 0.0
moe_diversity_inclusion_epoch = 0
warmup_steps = 8000
label_smoothing = 0.1
device = 'cpu'

tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer)
model = megatransformer.MegaTransformer(model_config).to(device)

print(model)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

def preprocess_function(examples):
    sources, targets = zip(*[(example["en"], example["de"]) for example in examples["translation"]])
    model_inputs = tokenizer(
        sources,
        text_target=targets,
        max_length=model_config.maxlen,
        truncation=True,
        padding=True,
        return_tensors=None,
        return_attention_mask=False,
    )
    return model_inputs

dataset = load_dataset('wmt/wmt14', 'de-en')
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    return_tensors="pt",
    label_pad_token_id=tokenizer.pad_token_id
)

training_args = Seq2SeqTrainingArguments(
    output_dir=run_dir,
    logging_dir=run_dir,
    report_to=["tensorboard"],
    save_total_limit=3,

    logging_strategy="steps",
    logging_steps=100,
    
    evaluation_strategy="epoch",
    save_strategy="epoch",
    predict_with_generate=True,
    push_to_hub=False,
    
    learning_rate=5e-5,
    dataloader_pin_memory=False,
    num_train_epochs=n_epochs,

    gradient_accumulation_steps=4,
    per_device_train_batch_size=32,
    save_safetensors=True,
    no_cuda=device == 'cpu'
)

optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
scheduler = lr_scheduler.get_noam_scheduler(optimizer=optimizer, warmup_steps=warmup_steps, d_model=model_config.d_model)
trainer = custom_trainers.CompositeMoESeq2SeqTrainer(
    moe_diversity_loss_coefficient=moe_diversity_loss_coefficient,
    moe_diversity_inclusion_epoch=moe_diversity_inclusion_epoch,
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    optimizers=(optimizer, scheduler)
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # calculate BLEU score
    metric = evaluate.load("sacrebleu")
    bleu_score = metric.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels]
    )

    images = []

    def log_callback(name, image_data, global_step, dataformats):
        images.append((name, image_data))

    src = tokenizer.encode(text="Anyone who retains the ability to recognise beauty will never become old.", return_tensors="pt")['input_ids']
    tgt = tokenizer.encode(text="Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.", return_tensors="pt")['input_ids']

    src_labels = [tokenizer.decode(t, skip_special_tokens=True) for t in src]
    tgt_labels = [tokenizer.decode(t, skip_special_tokens=False) for t in tgt]

    visualization_helper.viz_model(model_config.encoder_config.device, model_config.decoder_config.device, model, log_callback, 0, model_config.maxlen, src, src_labels, tgt, tgt_labels)

    return {
        "bleu": bleu_score["score"],
        "decoded_preds": decoded_preds[:5],
        "decoded_labels": decoded_labels[:5],
        "images": images
    }

trainer.compute_metrics = compute_metrics

if __name__ == "__main__":
    trainer.train()

"""
Example Usage:
`python -m examples.hf.wmt14ende --config configs/seq2seq/aiayn.yaml --run_name my_seq2seq_run`
from the root of the repository.
"""
