import generic_unprocessed
import wikitext

def load_dataset(
    dataset_name: str,
    dataset_config_name: str,
    tokenizer,
    max_position_embeddings: int
):
    if "wikitext" in dataset_name.lower():
        return wikitext.load(
            dataset_name,
            dataset_config_name,
            tokenizer,
            max_position_embeddings
        )
    
    print(f"Loading default dataset {dataset_name} with config {dataset_config_name}")
    return generic_unprocessed.load(
        dataset_name,
        dataset_config_name,
        tokenizer,
        max_position_embeddings
    )
