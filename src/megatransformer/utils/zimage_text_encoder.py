"""Compute Z-Image's Qwen3-4B text conditioning from caption strings, for the
Z-Image adapter's regression target.

Loads ONLY Qwen3-4B (Z-Image's frozen text encoder, 4-bit ~2.5GB) from the Z-Image
repo -- NOT the full ~20GB pipeline (DiT/VAE) -- and replicates
ZImagePipeline.encode_prompt exactly:

  - wrap each caption in Qwen3's chat template (add_generation_prompt, enable_thinking)
  - tokenize (padding to max_length, truncation)
  - take the PENULTIMATE hidden state (hidden_states[-2]) of Qwen3-4B
  - keep only non-pad (attention-masked) tokens -> variable-length [L, 2560]

then linearly RESAMPLES each masked [L, 2560] to a fixed length K -> [B, K, 2560], so
the target aligns with the adapter's fixed K output queries and a token-wise MSE
(mirrors the SDXL adapter's fixed-77 target). Verified that a resampled-to-K
conditioning renders on par with the native variable-length one through Z-Image
(eval_output/zimage_cond_tolerance/fixedlen_check.png). No pooled vector, no
normalization (Z-Image uses none).

Frozen, inference-only. Lazily instantiated by the world trainer when the image
generator is a ZImageConditioningAdapter.
"""

import torch
import torch.nn.functional as F


class Qwen3TextTargetEncoder:
    def __init__(self, model_name="Tongyi-MAI/Z-Image-Turbo", seq_len=64,
                 device="cuda", max_length=512, load_in_4bit=True):
        from transformers import AutoTokenizer, Qwen3Model
        self.device = device
        self.seq_len = int(seq_len)
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        kw = dict(subfolder="text_encoder", torch_dtype=torch.bfloat16)
        if load_in_4bit:
            try:
                import bitsandbytes  # noqa: F401  (ensure the 4-bit backend is present)
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16)
                kw["device_map"] = {"": device}
            except Exception as e:  # bitsandbytes missing -> bf16 fallback (~8GB)
                print(f"[Qwen3TextTargetEncoder] 4-bit unavailable ({e}); loading bf16 "
                      f"(~8GB) instead. Run `uv sync` to install bitsandbytes.", flush=True)
                load_in_4bit = False
        self.model = Qwen3Model.from_pretrained(model_name, **kw)
        if not load_in_4bit:
            self.model = self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_native(self, captions, max_len=128):
        """Targets at their NATIVE length -- no resample. -> (B, Lmax, 2560) float32, (B, Lmax) bool.

        The AR head emits one token per real Qwen3 token, so it needs the true sequence and a
        mask rather than the K-slot interpolation `encode` performs. Lmax is the longest real
        length in the batch, clipped to max_len.
        """
        caps = [c if isinstance(c, str) else "" for c in captions]
        prompts = [self.tokenizer.apply_chat_template(
            [{"role": "user", "content": c}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True) for c in caps]
        t = self.tokenizer(prompts, padding="longest", truncation=True,
                           max_length=min(self.max_length, max_len), return_tensors="pt")
        ids = t.input_ids.to(self.device)
        attn = t.attention_mask.to(self.device)
        hs = self.model(input_ids=ids, attention_mask=attn,
                        output_hidden_states=True).hidden_states[-2]        # (B, T, 2560)
        return hs.float(), attn.bool()

    @torch.no_grad()
    def caption_length(self, caption, max_len=128):
        """Real token count for a caption under the same chat template -- the AR sample length.
        Deterministic from the CAPTION, so inference needs no length head and no stop token."""
        s = self.tokenizer.apply_chat_template([{"role": "user", "content": caption}],
                                               tokenize=False, add_generation_prompt=True,
                                               enable_thinking=True)
        return min(len(self.tokenizer(s).input_ids), max_len)

    @torch.no_grad()
    def encode(self, captions):
        """captions: list[str] (len B; None entries -> ""). Returns (B, K, 2560) float32."""
        caps = [c if isinstance(c, str) else "" for c in captions]
        prompts = [self.tokenizer.apply_chat_template(
            [{"role": "user", "content": c}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True) for c in caps]
        # padding="longest", NOT the "max_length"=512 that ZImagePipeline.encode_prompt
        # uses. The tokenizer RIGHT-pads and Qwen3 is causal, so a real token's penultimate
        # state is (numerically) identical regardless of trailing pad count -- we strip pads
        # before resampling anyway. Verified: vs max_length=512 the real-token features have
        # rel_L2 0.0049 / per-token cos 0.9997 (a single outlier dim shows ~64 abs diff, but
        # that's <1% relative), and renders are downstream-identical (ΔCLIP -0.006, same
        # scenes) -- 8x inside the already-safe 4-bit margin. This is ~7.6x less Qwen3
        # compute (captions ~20-40 tok vs 512): 41ms vs 312ms/batch.
        ti = self.tokenizer(prompts, padding="longest", max_length=self.max_length,
                            truncation=True, return_tensors="pt")
        ids = ti.input_ids.to(self.model.device)
        attn = ti.attention_mask.to(self.model.device)
        hs = self.model(input_ids=ids, attention_mask=attn,
                        output_hidden_states=True).hidden_states[-2]   # (B, T, 2560)
        mb = attn.bool()
        out = []
        for i in range(hs.shape[0]):
            real = hs[i][mb[i]]                                        # (L, 2560) masked real tokens
            if real.shape[0] == 0:                                     # empty-caption guard
                real = hs[i][:1]
            x = real.transpose(0, 1).unsqueeze(0).float()             # (1, 2560, L)
            r = F.interpolate(x, size=self.seq_len, mode="linear", align_corners=False)
            out.append(r[0].transpose(0, 1))                          # (K, 2560)
        return torch.stack(out, dim=0).to(self.device)                # (B, K, 2560) float32
