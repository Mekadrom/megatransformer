from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from tqdm import tqdm

import argparse
import dataset_loading

argparser = argparse.ArgumentParser()
argparser.add_argument("--tokenizer_name", type=str, default="mistralai/Mistral-7B-v0.1", help="Tokenizer name")
argparser.add_argument("--dataset_name", type=str, default="wikitext", help="Path to the dataset") # or gair-prox/FineWeb-pro
argparser.add_argument("--dataset_config_name", type=str, default="wikitext-103-v1", help="Dataset config name")
argparser.add_argument("--max_position_embeddings", type=int, default=8192, help="Max position embeddings (maximum sequence length)")
argparser.add_argument("--dataset_cache_dir", type=str, default="cached_datasets", help="Path to the dataset cache directory")

args = argparser.parse_args()

if args.dataset_config_name == '' or args.dataset_config_name == 'None':
    setattr(args, 'dataset_config_name', None)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, add_bos_token=False)
print(f"default tokenizer.padding_side: {tokenizer.padding_side}")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.bos_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print(f"modified tokenizer: {tokenizer}")

datasets = dataset_loading.load_dataset(args.dataset_name, args.dataset_config_name, tokenizer, args.max_position_embeddings, cache_dir=args.dataset_cache_dir)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

dataloader = DataLoader(
    datasets["train"],
    batch_size=1,
    collate_fn=collator,
)

n_examples = 0
total_tokens = 0
longest_example = 0
shortest_example = float("inf")
for batch in tqdm(dataloader, total=100000000 if "fineweb" in args.dataset_name.lower() else None):
    n_examples += 1
    length = batch["input_ids"].shape[1]
    total_tokens += length
    longest_example = max(longest_example, length)
    shortest_example = min(shortest_example, length)

print(f"Number of examples: {n_examples:,}")
print(f"Total tokens: {total_tokens:,}")
print(f"Average tokens per example: {(total_tokens / n_examples):.3f}")
print(f"Longest example: {longest_example:,}")
