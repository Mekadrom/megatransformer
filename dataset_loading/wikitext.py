import re

from datasets import load_dataset



def clean_wikitext_examples(examples):
    texts: list[str] = examples["text"]
    cleaned_texts = []
    for text in texts:
        # replace special tokens
        cleaned = text.replace("@-@", "-")
        cleaned = cleaned.replace("@,@", ",")
        cleaned = cleaned.replace("@.@", ".")
        
        # fix spaces around punctuation
        for punct in ",.!?;:)]":
            cleaned = cleaned.replace(f" {punct}", punct)
        
        for punct in "([":
            cleaned = cleaned.replace(f"{punct} ", punct)
        
        # fix double spaces
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        
        # normalize multiple newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

        cleaned = cleaned.strip()
        
        if cleaned == "" or cleaned == "\n":
            continue

        cleaned_texts.append(cleaned)
    return {"text": cleaned_texts}

def load_text_dataset(dataset_name, dataset_config_name, split, tokenizer, max_position_embeddings, streaming=False, cache_dir=None):
    dataset = load_dataset(dataset_name, dataset_config_name, split=split, streaming=streaming, cache_dir=cache_dir)

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=max_position_embeddings)

    dataset = dataset.map(clean_wikitext_examples, batched=True)
    return dataset.map(tokenize_function, batched=True, remove_columns=["text"])
