from datasets import load_dataset


def load(dataset_name, dataset_config_name, tokenizer, max_position_embeddings):
    dataset = load_dataset(dataset_name, dataset_config_name)

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=max_position_embeddings)

    return dataset.map(tokenize_function, batched=True, remove_columns=["text"])
