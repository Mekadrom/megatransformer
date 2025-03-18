from datasets import load_dataset, load_from_disk

import os


def load(dataset_name,
         dataset_config_name,
         tokenizer,
         max_position_embeddings,
         streaming=False,
         stride=128,
         save_path=None):
    if save_path is not None and not streaming and os.path.exists(save_path):
        print(f"Loading dataset from {save_path}")
        dataset = load_from_disk(save_path)

    dataset = load_dataset(dataset_name, dataset_config_name, streaming=streaming)

    def tokenize_function(examples):
        texts = examples["text"]
        all_input_ids = []
        attention_masks = []
        for text in texts:
            tokenized = tokenizer(text, add_special_tokens=False)
            input_ids = tokenized.input_ids
            
            for i in range(0, len(input_ids), stride):
                end = min(i + max_position_embeddings, len(input_ids))
                window = input_ids[i:end]
                
                # only include windows of sufficient size
                if len(window) >= stride or end == len(input_ids):
                    all_input_ids.append(window)
                    attention_masks.append([1] * len(window))
        return {
            "input_ids": all_input_ids,
            "attention_mask": attention_masks,
        }
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    if save_path is not None and not streaming:
        print(f"Saving dataset to {save_path}")
        tokenized_dataset.save_to_disk(save_path)
    return tokenized_dataset
