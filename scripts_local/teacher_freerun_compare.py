"""Teacher (CosyVoice2 LLM) FREE-RUNNING token stats vs ground truth.

The 20k probe showed top-1 unit accuracy is a poor quality proxy: the teacher makes
intelligible speech at only 0.1245 top-1, with 4.02 nats of predictive entropy. So what
separates its clean speech from our garbage is not accuracy — it is whether the SAMPLED
sequence stays on-manifold. This measures that manifold directly.

Generates units from text with the teacher's own AR decoding (no prompt, no speaker — the
Qwen2LM lm_input is [sos, text, task_id, prompt_speech] and carries NO speaker term), then
reports the same degeneration statistics `world_voice_ar_diagnostics.seq_degeneration` computes
for our student, so all three (teacher / GT / student) are directly comparable.

  uv run python scripts_local/teacher_freerun_compare.py --model_dir <cv2 snapshot> \
      --cache_dir cached_datasets/Mekadrom/libriheavy_cosyvoice2_smollm2/val --n 32 --device cuda:3
"""
import os, sys, glob, argparse, functools, json

import torch

RT = os.environ.get("COSYVOICE_RUNTIME", "/home/zadar/dev/projects/cosyvoice-runtime")
sys.path.insert(0, os.path.join(RT, "cv_extra"))
sys.path.insert(0, os.path.join(RT, "CosyVoice", "third_party", "Matcha-TTS"))
sys.path.insert(0, os.path.join(RT, "CosyVoice"))

# NOTE: inlined rather than imported from world_voice_ar_diagnostics so this script can run
# under the ISOLATED CosyVoice venv (transformers 4.44) as well as the project venv — the
# incremental decode path is version-sensitive and has to be compared across both. Keep in
# sync with world_voice_ar_diagnostics.seq_degeneration.
import math
from collections import Counter


def seq_degeneration(seqs, K):
    if not seqs:
        return {}
    lens = [len(s) for s in seqs]
    rep = run = tot_pairs = 0
    uni, bi, tri = Counter(), set(), set()
    n_bi = n_tri = 0
    for s in seqs:
        for u in s:
            uni[u] += 1
        for t in range(1, len(s)):
            tot_pairs += 1
            if s[t] == s[t - 1]:
                rep += 1
        cur = 1
        for t in range(1, len(s)):
            cur = cur + 1 if s[t] == s[t - 1] else 1
            run = max(run, cur)
        for t in range(1, len(s)):
            bi.add((s[t - 1], s[t])); n_bi += 1
        for t in range(2, len(s)):
            tri.add((s[t - 2], s[t - 1], s[t])); n_tri += 1
    total = sum(uni.values())
    probs = [c / total for c in uni.values()]
    ent = -sum(p * math.log2(p) for p in probs)
    return {
        "n_seqs": len(seqs),
        "len_mean": sum(lens) / len(lens), "len_min": min(lens), "len_max": max(lens),
        "adj_repeat_rate": rep / max(tot_pairs, 1),
        "longest_run": run,
        "distinct_units": len(uni), "coverage": len(uni) / K,
        "unit_entropy_bits": ent, "max_entropy_bits": math.log2(K),
        "distinct_bigram_ratio": len(bi) / max(n_bi, 1),
        "distinct_trigram_ratio": len(tri) / max(n_tri, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--top_k", type=int, default=25, help="teacher's own default decoding")
    ap.add_argument("--no_prompt", dest="use_prompt", action="store_false", default=True,
                    help="Run WITHOUT a reference prompt (out-of-distribution for the released "
                         "model — it degenerates; kept only to reproduce that finding).")
    ap.add_argument("--out_dir", default="eval_output/world_tts_cosy_ab")
    a = ap.parse_args()

    from cosyvoice.llm.llm import Qwen2LM, Qwen2Encoder
    from cosyvoice.utils.common import ras_sampling
    from transformers import AutoTokenizer

    qwen_path = os.path.join(a.model_dir, "CosyVoice-BlankEN")
    lm = Qwen2LM(llm_input_size=896, llm_output_size=896, speech_token_size=6561,
                 llm=Qwen2Encoder(pretrain_path=qwen_path),
                 sampling=functools.partial(ras_sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1),
                 length_normalized_loss=True, lsm_weight=0, mix_ratio=[5, 15])
    lm.load_state_dict(torch.load(os.path.join(a.model_dir, "llm.pt"), map_location="cpu",
                                  weights_only=False), strict=True)
    # The Qwen2 backbone loads bf16 while the bolted-on speech modules are fp32, and
    # inference() concatenates them internally — so unify on fp32 (0.5B ~ 2GB) rather than
    # trying to cast at the call site.
    lm.to(a.device).float().eval()
    tok = AutoTokenizer.from_pretrained(qwen_path)

    s = torch.load(sorted(glob.glob(os.path.join(a.cache_dir, "shard_*.pt")))[0],
                   map_location="cpu", weights_only=False)
    K = 6561

    # Zero-shot contract: the released LLM path is ALWAYS driven with a reference
    # (prompt_text + its speech tokens). Running it with empty prompts is out-of-distribution
    # and it degenerates (3x length, looping) — measured. So pair each target utterance with a
    # DIFFERENT utterance from the SAME speaker as the prompt, which is exactly what
    # inference_zero_shot supplies.
    spk = s["speaker_ids"].tolist()
    by_spk = {}
    for j, sp in enumerate(spk):
        by_spk.setdefault(sp, []).append(j)

    gen_seqs, gt_seqs, ratios = [], [], []
    empty_l = torch.zeros(1, 0, dtype=torch.long, device=a.device)
    zero = torch.tensor([0], dtype=torch.int32, device=a.device)
    done = 0
    for i in range(len(s["text"])):
        L = int(s["feature_lengths"][i])
        if L < 20:
            continue
        # pick a prompt utterance: same speaker, different index, decent length
        cands = [j for j in by_spk.get(spk[i], []) if j != i and 20 <= int(s["feature_lengths"][j]) <= 200]
        if a.use_prompt and not cands:
            continue
        text_ids = tok(s["text"][i], return_tensors="pt")["input_ids"].to(a.device)
        if a.use_prompt:
            j = cands[0]
            Lj = int(s["feature_lengths"][j])
            p_text = tok(s["text"][j], return_tensors="pt")["input_ids"].to(a.device)
            p_tok = s["unit_ids"][j, :Lj].to(a.device).unsqueeze(0)
            p_text_len = torch.tensor([p_text.shape[1]], dtype=torch.int32, device=a.device)
            p_tok_len = torch.tensor([Lj], dtype=torch.int32, device=a.device)
        else:
            p_text, p_tok = empty_l, empty_l
            p_text_len, p_tok_len = zero.clone(), zero.clone()
        with torch.no_grad():
            out = [int(t) for t in lm.inference(
                text=text_ids, text_len=torch.tensor([text_ids.shape[1]], dtype=torch.int32, device=a.device),
                prompt_text=p_text, prompt_text_len=p_text_len,
                prompt_speech_token=p_tok, prompt_speech_token_len=p_tok_len,
                embedding=torch.zeros(0, device=a.device), sampling=a.top_k)]
        out = [t for t in out if 0 <= t < K]          # strip eos/fill
        if not out:
            continue
        gen_seqs.append(out)
        gt_seqs.append(s["unit_ids"][i, :L].tolist())
        ratios.append(len(out) / L)
        done += 1
        if done % 8 == 0:
            print(f"  {done}/{a.n} ...", flush=True)
        if done >= a.n:
            break

    t_deg, g_deg = seq_degeneration(gen_seqs, K), seq_degeneration(gt_seqs, K)
    keys = ["n_seqs", "len_mean", "adj_repeat_rate", "longest_run", "distinct_units",
            "coverage", "unit_entropy_bits", "distinct_bigram_ratio", "distinct_trigram_ratio"]
    print(f"\n{'metric':<24}{'TEACHER free-run':>18}{'GROUND TRUTH':>16}")
    print("-" * 58)
    for k in keys:
        tv, gv = t_deg.get(k), g_deg.get(k)
        f = (lambda v: f"{v:.4f}" if isinstance(v, float) else str(v))
        print(f"{k:<24}{f(tv):>18}{f(gv):>16}")
    print(f"\nteacher/GT length ratio: mean {sum(ratios)/len(ratios):.3f}")
    print("\nThe teacher's numbers are the ON-MANIFOLD reference: this is what a model that "
          "produces intelligible speech looks like distributionally. Compare the student's "
          "section-3 table from world_voice_ar_diagnostics against the TEACHER column, not GT.")

    os.makedirs(a.out_dir, exist_ok=True)
    p = os.path.join(a.out_dir, "teacher_freerun.json")
    json.dump({"teacher": t_deg, "ground_truth": g_deg,
               "len_ratio_mean": sum(ratios) / len(ratios), "n": done, "top_k": a.top_k},
              open(p, "w"), indent=2)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
