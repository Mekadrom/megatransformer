import deepspeed
import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser = deepspeed.add_config_arguments(parser)
    args, unk = parser.parse_known_args()
    
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    print("About to initialize DeepSpeed")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer
    )
    print("DeepSpeed initialized successfully!")
    
    # Try a basic forward pass
    dummy_input = torch.randn(2, 10).to(model_engine.device)
    output = model_engine(dummy_input)
    print("Forward pass successful!")

if __name__ == "__main__":
    main()
