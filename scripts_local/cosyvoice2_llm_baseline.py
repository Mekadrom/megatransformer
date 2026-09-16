"""Load CosyVoice 2's OWN autoregressive speech LM (llm.pt) -- the 641.9M module the world
model replaces -- so the two can be compared on identical prompts.

The packaged `CosyVoice2` class cannot be used: its __init__ builds CosyVoiceFrontEnd, which
imports modelscope and a text-normalisation stack this runtime deliberately does not carry.
Everything that frontend would provide, we already have -- the ONNX speech tokenizer and
campplus live in megatransformer.utils.cosyvoice2_encoders, and the decoder is
CosyVoice2Decoder. Only the LLM module itself is missing, so it is constructed directly.

The yaml filter is the MIRROR of cosyvoice2_decoder._load_configs_without_llm: that one keeps
flow+hift and drops llm; this keeps llm and drops flow+hift, because constructing `flow:`
imports `conformer`, which is not installed (and is not needed to run the LLM).
"""
import os


def load_cosyvoice2_llm(model_dir: str, runtime_dir: str = None, device: str = "cuda:0",
                        dtype=None):
    import sys
    import torch
    runtime_dir = runtime_dir or os.path.expanduser("~/dev/projects/cosyvoice-runtime")
    cv = os.path.join(runtime_dir, "CosyVoice")
    for p in (cv, os.path.join(cv, "third_party/Matcha-TTS")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from hyperpyyaml import load_hyperpyyaml

    raw = open(os.path.join(model_dir, "cosyvoice2.yaml")).read()
    _KEEP_NEW = ("llm:",)
    _DROP_KEYS = ("data_pipeline:", "data_pipeline_gan:", "train_conf:")
    keep, skipping = [], False
    for ln in raw.split("\n"):
        if ln and not ln[0].isspace() and ":" in ln:
            top = ln.split(":", 1)[0] + ":"
            if top in _KEEP_NEW:
                skipping = False
            elif top in _DROP_KEYS or "!name:" in ln or "!new:" in ln:
                skipping = True
            else:
                skipping = False
        if not skipping:
            keep.append(ln)
    cfg = load_hyperpyyaml(
        "\n".join(keep),
        overrides={"qwen_pretrain_path": os.path.join(model_dir, "CosyVoice-BlankEN")},
    )
    llm = cfg["llm"]
    sd = torch.load(os.path.join(model_dir, "llm.pt"), map_location="cpu", weights_only=False)
    missing, unexpected = llm.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [cv2-llm] {len(missing)} missing keys (first: {missing[:2]})")
    # llm.pt stores bf16, but campplus embeddings and prompt tokens arrive as fp32, and
    # Qwen2's Linear layers will not mix them ("mat1 and mat2 must have the same dtype").
    # Cast the whole module to one dtype rather than casting call sites -- the inference
    # generator touches several entry points.
    import torch as _t
    llm = llm.to(device=device, dtype=(dtype or _t.float32)).eval()
    return llm, cfg


def qwen_tokenizer(model_dir: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(os.path.join(model_dir, "CosyVoice-BlankEN"))


import torch as _torch_mod


@_torch_mod.no_grad()
def cv2_generate_units(llm, tok, text: str, prompt_text: str, prompt_units, prompt_emb,
                       device="cuda:0", top_k=25, max_ratio=20, min_ratio=2, dtype=None):
    """Generate speech units with CosyVoice 2's own LM, with a transformers-5.x-correct cache.

    ⚠️ The vendored `Qwen2Encoder.forward_one_step` is INCOMPATIBLE with transformers >= 5:
    it passes `attention_mask = masks[:, -1, :]`, which on a 1-token decode step has shape
    (1, 1) and therefore covers only the NEW token. transformers 4.x expanded such a mask
    against the cache; 5.x takes it literally and masks the entire past, so the model sees a
    context-free single token every step. Symptom: the output distribution FREEZES after a
    few steps (identical top-k and identical p(EOS)=0.000118 at steps 5 through 339), the
    model emits one unit forever and never reaches EOS, so generation always runs to
    `max_token_text_ratio`.

    The fix is to pass a mask covering past+current. Everything else -- sos/task framing,
    RAS sampling, stop ids -- is the reference implementation's own.
    """
    import torch
    dt = dtype or next(llm.parameters()).dtype
    t_ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]], device=device)
    p_ids = torch.tensor([tok(prompt_text, add_special_tokens=False)["input_ids"]], device=device)
    pu = prompt_units.reshape(1, -1).to(device)
    text_all = torch.concat([p_ids, t_ids], dim=1)
    text_emb = llm.llm.model.model.embed_tokens(text_all)
    sos = llm.llm_embedding.weight[llm.sos].reshape(1, 1, -1)
    tid = llm.llm_embedding.weight[llm.task_id].reshape(1, 1, -1)
    pemb = llm.speech_embedding(pu)
    lm_input = torch.concat([sos, text_emb, tid, pemb], dim=1).to(dt)

    min_len, max_len = int(t_ids.shape[1] * min_ratio), int(t_ids.shape[1] * max_ratio)
    out, cache, total = [], None, lm_input.shape[1]
    for step in range(max_len):
        am = torch.ones(1, total, dtype=torch.long, device=device)
        o = llm.llm.model(inputs_embeds=lm_input, attention_mask=am,
                          output_hidden_states=True, return_dict=True,
                          use_cache=True, past_key_values=cache)
        cache = o.past_key_values
        logp = llm.llm_decoder(o.hidden_states[-1][:, -1]).log_softmax(dim=-1)
        _t = llm.sampling_ids(logp.squeeze(0), out, top_k, ignore_eos=step < min_len)
        tokid = int(_t.item()) if hasattr(_t, "item") else int(_t)
        if tokid in llm.stop_token_ids:
            break
        out.append(tokid)
        lm_input = llm.speech_embedding.weight[tokid].reshape(1, 1, -1).to(dt)
        total += 1
    return out
