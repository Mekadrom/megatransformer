from dataset_loading import generic, wikitext


backup_validation_dataset_name = "wikitext"
backup_validation_dataset_config_name = "wikitext-2-v1"


def load_dataset(
    dataset_name: str,
    dataset_config_name: str,
    tokenizer,
    max_position_embeddings: int,
    streaming: bool = False,
    save_path: str = None,
):
    print(f"Loading dataset {dataset_name} with config {dataset_config_name}")
    streaming = "fineweb" in dataset_name.lower()
    if "wikitext" in dataset_name.lower():
        dataset = wikitext.load(
            dataset_name,
            dataset_config_name,
            tokenizer,
            max_position_embeddings
        )
    else:
        dataset = generic.load(
            dataset_name,
            dataset_config_name,
            tokenizer,
            max_position_embeddings,
            streaming=streaming,
            save_path=save_path
        )

    if "validation" not in dataset:
        print(f"No validation set found, loading backup validation dataset {backup_validation_dataset_name}/{backup_validation_dataset_config_name}")
        if "wikitext" in backup_validation_dataset_name.lower():
            backup_dataset = wikitext.load(
                backup_validation_dataset_name,
                backup_validation_dataset_config_name,
                tokenizer,
                max_position_embeddings
            )
        else:
            backup_dataset = generic.load(
                backup_validation_dataset_name,
                backup_validation_dataset_config_name,
                tokenizer,
                max_position_embeddings
            )
        dataset["validation"] = backup_dataset["train"]

    if not streaming:
        print(f"Number of training examples: {len(dataset['train']):,}")
    print(f"Number of validation examples: {len(dataset['validation']):,}")

    return dataset
