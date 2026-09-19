"""Score the ENTIRE text eval corpus with a frozen reference LM.

Gives the world-text run a fixed comparison line: 'what does an off-the-shelf model of
comparable size get on exactly this data?'. The corpus is SmolLM2-tokenized, so a SmolLM2
model scores it natively -- no retokenization, no cross-tokenizer correction.
"""
import sys, math, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM
from megatransformer.scripts.data.world.dataset import MultimodalShardedDataset

MODEL = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M"
SHARDS = sys.argv[2] if len(sys.argv) > 2 else "cached_datasets/Mekadrom/text_huginn-mix_smollm2/val"
BS = 8

ds = MultimodalShardedDataset(text_shard_dir=SHARDS)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
tot_nll, tot_tok = 0.0, 0
with torch.no_grad():
    for i in range(0, len(ds), BS):
        ids = torch.stack([ds[j]["text_token_ids"] for j in range(i, min(i+BS, len(ds)))]).cuda()
        lg = m(input_ids=ids).logits.float()
        # sum reduction so the corpus mean is over TOKENS, not over batches of unequal size
        nll = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                              ids[:, 1:].reshape(-1), reduction="sum")
        tot_nll += float(nll); tot_tok += ids[:, 1:].numel()
        if (i // BS) % 200 == 0:
            print(f"  {i:>7}/{len(ds)}  running nats/tok {tot_nll/max(tot_tok,1):.4f}", flush=True)
ce = tot_nll / tot_tok
print(f"\nMODEL {MODEL}")
print(f"  tokens scored   {tot_tok:,}")
print(f"  CE (nats/token) {ce:.4f}")
print(f"  PPL             {math.exp(ce):.2f}")
print(f"  loss_norm       {ce/math.log(49161):.4f}   (vs uniform baseline 1.0)")
