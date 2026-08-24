"""Does a bistream checkpoint generate PAST its first chunk?

The symptom this exists to catch: every render is ~1 second (one 30-frame chunk) with the
right content. That is what a fill_token read as end-of-utterance looks like -- correct
learning, truncated decode.
"""
import sys, torch
sys.path.insert(0, "scripts_local")
from megatransformer.scripts.eval.world.visualize import load_world_model, resolve_shard_dir
from megatransformer.scripts.data.world.data_collator import MultimodalDataCollator
from megatransformer.scripts.data.world.memorization_dataset import MultimodalMemorizationDataset
from megatransformer.utils.codebook import load_codebook
from megatransformer.utils import constants
import world_voice_ar_diagnostics as diag
from argparse import Namespace

CACHE = "./cached_datasets/Mekadrom/libritts_r_cosyvoice2_smollm2"
CB = f"{CACHE}/val/cosyvoice2_codebook.pt"
CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/world/world_voice_memorize32_bistream_0/checkpoint-500"
DEV = sys.argv[2] if len(sys.argv) > 2 else "cuda:3"
K_TEXT, S = 5, 30

cb = load_codebook(CB); K, D = int(cb.shape[0]), int(cb.shape[1])
a = Namespace(config="small_sum", checkpoint_path=CKPT, codebook=CB, cache_dir=CACHE,
              voice_predict_f0=False, text_encoder_model="HuggingFaceTB/SmolLM2-135M",
              n=8, mrope_scale_side="text", mrope_voice_rate=6.0)
args = diag.build_args(a, D)
model = load_world_model(args, DEV); model.set_voice_codebook(cb); model.to(DEV).eval()
sp_base = getattr(model.config, "special_token_base", constants.SPECIAL_TOKEN_BASE)
sp = constants.special_token_ids(sp_base)
print(f"unit head width: {model.voice_generator.unit_head.out_features} (K={K}, fill={K+1})")

ds = MultimodalMemorizationDataset(voice_shard_dir=resolve_shard_dir(CACHE, "train"),
                                   max_samples=32, voice_codebook=CB)
col = MultimodalDataCollator(max_seq_len=1024, max_waveforms=240000, max_mel_spec_frames=625,
                             max_sive_feature_frames=250, voice_eov_id=K,
                             special_token_base=sp_base, eos_token_id=model.config.eos_token_id
                             if hasattr(model.config, "eos_token_id") else 2,
                             bistream_text_chunk=K_TEXT, bistream_voice_chunk=S,
                             bistream_prob=1.0, voice_fill_id=K + 1)
col.force_direction = "synthesis"

print(f"\n{'idx':>4} {'target':>7} {'no-cont':>8} {'with-cont':>10}  verdict")
for i in range(4):
    s = ds[i]
    b = col([s])
    tids = b["text_token_ids"][0]
    bov = (tids == sp.BOV).nonzero(as_tuple=True)[0]
    prompt = tids[:int(bov[0]) + 1].unsqueeze(0).to(DEV)
    full = s["text_token_ids"][:int(s["text_text_length"])]
    tgt = int(s["voice_feature_length"])

    def gen(bistream):
        kw = {}
        if bistream:
            kw = {"voice_bistream_text": full.reshape(1, -1).to(DEV),
                  "voice_bistream_text_chunk": K_TEXT,
                  "voice_bistream_text_offset": int(bov[0])}
        with torch.no_grad():
            o = model.generate(text_input_ids=prompt, max_new_tokens=512, temperature=0.0,
                               voice_temperature=0.0, **kw)
        v = o.get("voice_latent_preds")
        return 0 if v is None or v.numel() == 0 else int(v[0, 0].shape[-1])

    a_len, b_len = gen(False), gen(True)
    ok = "OK" if b_len > a_len + S // 2 else "STILL TRUNCATED"
    print(f"{i:>4} {tgt:>7} {a_len:>8} {b_len:>10}  {ok}")
