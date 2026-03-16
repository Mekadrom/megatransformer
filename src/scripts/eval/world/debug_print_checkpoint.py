import argparse
import os
import torch

from model.world.world_model import MegaTransformerWorldModel
from scripts.train.world import training
from utils import megatransformer_utils


def debug_print_checkpoint(args, module_filter):
    # load world model from checkpoint
    model = training.load_model(args, device='cpu')
    # print(model)

    optimizer_state = torch.load(os.path.join(args.resume_from_checkpoint, "optimizer.pt"), map_location='cpu')

    model_sd = model.state_dict()

    param_names = list(model_sd.keys())
    param_index_map = {i: name for i, name in enumerate(param_names)}

    head_prefixes = ["image_generator.", "voice_generator.", "text_generator."]

    for i, state in optimizer_state["state"].items():
        name = param_index_map.get(i, f"unknown_{i}")
        if not any(name.startswith(p) for p in head_prefixes):
            continue

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        param = model_sd[name]
        
        effective_step = exp_avg / (exp_avg_sq.sqrt() + 1e-8)

        print(f"{name}:")
        print(f"  param     — mean={param.mean():.6f}, std={param.std():.6f}, norm={param.norm():.6f}")
        print(f"  exp_avg   — mean={exp_avg.mean():.6f}, norm={exp_avg.norm():.6f}")
        print(f"  exp_avg_sq— mean={exp_avg_sq.mean():.6f}, norm={exp_avg_sq.norm():.6f}")
        print(f"  eff. step — mean={effective_step.mean():.6f}, std={effective_step.std():.6f}")
        print()

    # for n, p in model.named_parameters():
    #     if module_filter and module_filter not in n:
    #         continue

    #     # print weight statistics
    #     if hasattr(p, "data") and p.data is not None:
    #         megatransformer_utils.print_debug_tensor(n, p.data)
    #     else:
    #         print(f"{n} has no weight data")

    #     # print grad statistics
    #     if p.grad is not None:
    #         megatransformer_utils.print_debug_tensor(n + ".grad", p.grad)
    #     else:
    #         print(f"{n} has no grad data")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Debug print checkpoint")

    argparser.add_argument("--config", type=str, required=True, help="World model config")
    argparser.add_argument("--resume_from_checkpoint", type=str, required=True, help="Path to the model checkpoint")
    argparser.add_argument('--include_modes', type=str, default='text,audio,image', help='Comma-separated list of modes to include (e.g., text,audio,image or audio,image), order agnostic')
    argparser.add_argument('--module_filter', type=str, default='', help='Only print parameters whose module name contains this string (e.g., "text_encoder")')

    args, unk = argparser.parse_known_args()

    unk_dict = {}
    for i in range(0, len(unk), 2):
        unk_dict[unk[i].lstrip('-')] = unk[i+1]
    print(f"Unknown arguments: {unk_dict}")

    debug_print_checkpoint(args, args.__dict__.pop('module_filter'))
