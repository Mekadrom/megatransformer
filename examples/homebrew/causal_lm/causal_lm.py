from datasets import load_dataset
from megatransformer import config, criteria, lr_scheduler, megatransformer, transformer_utils
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, set_seed

import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def average_gradients(model: megatransformer.MegaTransformer):
    """Average gradients across all devices"""
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size

def average_model_weights(model: megatransformer.MegaTransformer):
    """Average model weights across all devices"""
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= size

argparser = argparse.ArgumentParser()

argparser.add_argument("--run_name", type=str, required=True)
argparser.add_argument("--data", type=str, default="data")
argparser.add_argument("--config", type=str, default="config.yaml")

args, unk = argparser.parse_known_args()

model_config = config.TransformerConfig()
model_config.load_yaml(args.config)

# load command line overrides (unk args since they aren't registered above)
model_config.__dict__.update({k: v for k, v in zip(unk[::2], unk[1::2])})

run_dir = os.path.join('runs', args.run_name)
os.makedirs(run_dir, exist_ok=True)

set_seed(42)

tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer)
model = megatransformer.MegaTransformer(model_config)

world_size = torch.cuda.device_count()

def preprocess_function(examples):
    targets = zip(*[example['text'] for example in examples["translation"]])
    model_inputs = tokenizer(
        targets,
        max_length=model_config.maxlen,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        return_attention_mask=False,
        return_token_type_ids=False
    )
    return model_inputs

dataset = load_dataset("HuggingFaceFW/fineweb", "default")
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

criterion = criteria.LabelSmoothedCE(eps=args.label_smoothing)
def train(rank, world_size, sync_frequency, sync_method='gradients'):
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://localhost:23456',
        world_size=world_size,
        rank=rank
    )

    local_model = model.to(rank)
    local_model = DistributedDataParallel(local_model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.get_noam_scheduler(optimizer, warmup_steps=args.warmup_steps, d_model=model_config.d_model)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size // 2,
        sampler=sampler,
        shuffle=False
    )
