"""CFG guidance probe (teacher-forced): does classifier-free guidance amplify text conditioning?

For a sweep of guidance scale w, combine the conditional and null-text (unconditional) unit
logits as guided = uncond + w*(cond - uncond), and measure content-unit top-1 accuracy against
the true units. w=0 is the pure unconditional, w=1 is the plain conditional, w>1 is amplified.
If accuracy RISES past w=1, guidance is extracting real text signal -> CFG is worth committing to.

ONLY meaningful on a checkpoint TRAINED with --voice_cfg_text_dropout_prob>0; otherwise the
null_text_embed is zero-init/untrained and the "unconditional" forward is not a real prior.
"""
import argparse
from argparse import Namespace

import torch
import torch.nn.functional as F

from megatransformer.scripts.eval.world.eval_voice_synthesis import load_world_model, load_dataset
from megatransformer.scripts.data.world.data_collator import MultimodalDataCollator
from megatransformer.utils.codebook import load_codebook


def wargs(a):
    return Namespace(
        config=a.config, checkpoint_path=a.checkpoint_path, include_modes="voice",
        tie_word_embeddings=False, voice_feature_channels=256, voice_predict_f0=True,
        voice_codebook_path=a.codebook, cache_dir=None, text_cache_dir=None,
        voice_cache_dir=a.cache_dir, use_memorization_dataset=False, max_samples=a.n,
        voice_cfg_enabled=True)  # build with null_text_embed so a CFG checkpoint loads + uncond works


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--cache_dir", required=True, help="voice VAL shard dir")
    ap.add_argument("--codebook", required=True)
    ap.add_argument("--config", default="small_sum")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.0, 1.0, 1.5, 2.0, 3.0, 5.0])
    ap.add_argument("--ngram_ceiling", type=float, default=0.2325)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    cb = load_codebook(a.codebook)
    K = int(cb.shape[0])
    model = load_world_model(wargs(a), dev)
    model.set_voice_codebook(cb)
    model.to(dev).eval()
    ds = load_dataset(wargs(a), "val")
    col = MultimodalDataCollator(max_seq_len=1024, max_waveforms=160000, max_mel_spec_frames=625,
                                 max_sive_feature_frames=209, voice_eov_id=K)

    hits = {w: 0 for w in a.scales}
    tot = 0
    idxs = list(range(min(a.n, len(ds))))
    for s in range(0, len(idxs), a.bs):
        samples = [ds[i] for i in idxs[s:s + a.bs]]
        samples = [x for x in samples if any(k.startswith("voice_") for k in x)]
        if not samples:
            continue
        col.force_direction = "synthesis"
        b = col(samples)
        text = b["text_token_ids"].to(dev)
        vin = b["voice_features"].unsqueeze(1).to(dev)
        vlen = b["voice_feature_lengths"].unsqueeze(1).to(dev)
        vlbl = b["voice_features"].to(dev)
        tgt = b["voice_unit_ids"].to(dev)
        syn = b["is_synthesis"].to(dev)

        def fwd(force_null):
            return model(text_input_ids=text, voice_inputs=vin, voice_lengths=vlen,
                         voice_latent_labels=vlbl, is_synthesis=syn, decode_outputs=False,
                         cfg_force_null_text=force_null)["voice_unit_logits"]

        cond = fwd(False).float()
        uncond = fwd(True).float()
        T = cond.shape[1]
        t = tgt[:, :T]
        if t.shape[1] < T:
            t = F.pad(t, (0, T - t.shape[1]), value=-100)
        content = (t >= 0) & (t < K)
        if not content.any():
            continue
        tot += int(content.sum())
        for w in a.scales:
            guided = uncond + w * (cond - uncond)
            hits[w] += int((guided.argmax(-1)[content] == t[content]).sum())

    print(f"\ncontent positions: {tot}   n-gram ceiling ref: {a.ngram_ceiling}")
    print(f"{'w':>6} {'acc':>9}   (w=0 uncond | w=1 plain cond | w>1 guidance-amplified)")
    print("-" * 40)
    base = hits.get(1.0, 0) / max(tot, 1)
    for w in a.scales:
        acc = hits[w] / max(tot, 1)
        flag = ""
        if w > 1.0 and acc > base:
            flag = f"  (+{acc - base:.4f} vs cond -> guidance helps)"
        print(f"{w:>6.1f} {acc:>9.4f}{flag}")
    print("\nRead: acc climbing above the w=1 value as w increases => CFG extracts real text "
          "signal, commit to a co-trained run. Flat/declining => text signal too weak to amplify.")


if __name__ == "__main__":
    main()
