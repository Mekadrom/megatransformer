"""Compute SDXL's CLIP text conditioning from caption strings, for the SDXL
adapter's regression target.

Loads ONLY SDXL's two frozen CLIP text encoders + tokenizers (~820M), not the
full pipeline (UNet/VAE), and replicates diffusers' `encode_prompt`:

  - tokenize with both tokenizers (77 tokens)
  - take the PENULTIMATE hidden state of each encoder (SDXL uses hidden_states[-2])
  - concatenate on the channel dim -> (B, 77, 2048)  [CLIP-L 768 + bigG 1280]
  - pooled = the bigG (text_encoder_2) pooled text embed -> (B, 1280)

Frozen, inference-only. Lazily instantiated by the world trainer when the image
generator is an SDXLConditioningAdapter.
"""

import torch


class SDXLTextTargetEncoder:
    def __init__(self, model_name="stabilityai/stable-diffusion-xl-base-1.0",
                 device="cuda", dtype=torch.float16, max_length=77):
        from transformers import (
            CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer,
        )
        self.device = device
        self.dtype = dtype
        self.max_length = max_length
        self.tok1 = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.tok2 = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer_2")
        self.enc1 = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()
        self.enc2 = CLIPTextModelWithProjection.from_pretrained(
            model_name, subfolder="text_encoder_2", torch_dtype=dtype).to(device).eval()
        for p in self.enc1.parameters():
            p.requires_grad_(False)
        for p in self.enc2.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, captions):
        """captions: list[str] (len B, None entries allowed -> empty string).
        Returns (seq (B,77,2048) float32, pooled (B,1280) float32)."""
        caps = [c if isinstance(c, str) else "" for c in captions]
        seqs = []
        pooled = None
        for tok, enc, is_two in ((self.tok1, self.enc1, False), (self.tok2, self.enc2, True)):
            ids = tok(caps, padding="max_length", max_length=self.max_length,
                      truncation=True, return_tensors="pt").input_ids.to(self.device)
            out = enc(ids, output_hidden_states=True)
            seqs.append(out.hidden_states[-2])              # penultimate, (B,77,dim)
            if is_two:
                pooled = out.text_embeds                    # bigG pooled, (B,1280)
        seq = torch.cat(seqs, dim=-1).float()               # (B,77,2048)
        return seq, pooled.float()
