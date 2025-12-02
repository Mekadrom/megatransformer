from datasets import load_dataset

import torch


def load_text_dataset(dataset_name,
                      dataset_config_name,
                      split,
                      tokenizer,
                      max_position_embeddings,
                      streaming=False,
                      stride=128,
                      cache_dir=None):
    print(f"Loading dataset {dataset_name} with config {dataset_config_name}, split {split} for text.")

    tries = 0
    while tries < 10:
        try:
            dataset = load_dataset(dataset_name, dataset_config_name, streaming=streaming, cache_dir=cache_dir, split=split, trust_remote_code=True)
            break
        except Exception as e:
            print(f"Error loading dataset: {e}. Retrying...")
            tries += 1
            continue

    if tries == 10:
        print("Failed to load dataset after 10 attempts.")
        return None

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
                    all_input_ids.append(torch.tensor(window))
                    attention_masks.append([1] * len(window))

        max_length = max([len(ids) for ids in all_input_ids])
        all_input_ids = [torch.nn.functional.pad(ids, (0, max_length - len(ids)), value=0) for ids in all_input_ids]
        attention_masks = [torch.nn.functional.pad(torch.tensor(mask), (0, max_length - len(mask)), value=0) for mask in attention_masks]

        return {
            "input_ids": torch.stack(all_input_ids),
            "attention_mask": torch.stack(attention_masks),
        }
    return dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
