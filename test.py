from model import megatransformer_causal
from transformers import AutoTokenizer, pipeline

import argparse
import os
import torch
import time

argparser = argparse.ArgumentParser()
argparser.add_argument("--logging_base_dir", type=str, default=os.path.join('runs', 'causal'), help="Base directory for logging")
argparser.add_argument("--run_name", type=str, required=True, help="Name of the run")
argparser.add_argument("--tokenizer_name", type=str, default="mistralai/Mistral-7B-v0.1", help="Tokenizer name")
argparser.add_argument("--dataset_name", type=str, default="wikitext", help="Path to the dataset")
argparser.add_argument("--config", type=str, default="modern_small", help="Model configuration: gpt2, modern, or huginn")
argparser.add_argument("--max_position_embeddings", type=int, default=1024, help="Max position embeddings (maximum sequence length)")
argparser.add_argument("--max_test_length", type=int, default=256, help="Maximum length for test generation")
argparser.add_argument("--compile_model", action="store_true", help="Whether to compile the model")
argparser.add_argument("--bf16", action="store_true", help="Whether to use bf16")
argparser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training (cuda or cpu)")

args = argparser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

model = megatransformer_causal.model_config_lookup(args.config)(tokenizer, args.max_position_embeddings)

model_file_path = os.path.join(args.logging_base_dir, args.dataset_name, args.run_name, "pytorch_model.bin")
if os.path.exists(model_file_path):
    print(f"Loading model from {model_file_path}")
    model.load_state_dict(torch.load(model_file_path), strict=False)

if args.compile_model:
    model = torch.compile(model)
    print(f"Model compiled and moved to {args.device}")

if args.bf16:
    model = model.to(torch.bfloat16)
    print("Model converted to bf16")

model = model.to(args.device)
model.eval()

print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

while True:
    user_input = input("Enter a prompt (or 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    inputs = tokenizer(user_input, return_tensors="pt").to(args.device)
    with torch.no_grad():
        start = time.time()
        outputs = model.generate(
            use_cache=False,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=args.max_test_length,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.99,
            temperature=0.9,
        )
        print(f"Generation time: {time.time() - start:.2f} seconds")
    
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(f"Generated text: {generated_texts[0]}")
