
# Mistral base tokenizer special tokens
EOS_TOKEN_ID = 2  # End of Sequence (Mistral-7B EOS)

# The 9 multimodal control tokens sit immediately after the "real" vocab, at
# base + 0..8 in a FIXED order. `base` = the real vocab size: 32000 for the
# Mistral tokenizer (the historical default below), or the pretrained LLM's
# native vocab (e.g. 49152 for SmolLM2) when config.special_token_base is set.
# Anything that emits/detects these tokens must use the SAME base as the data.
SPECIAL_TOKEN_BASE = 32_000  # Mistral vocab size (default)
N_SPECIAL_TOKENS = 9
_SPECIAL_ORDER = ("BOA", "EOA", "BOV", "EOV", "BOI", "EOI",
                  "AUDIO_PLACEHOLDER", "VOICE_PLACEHOLDER", "IMAGE_PLACEHOLDER")


class SpecialTokenIds:
    """Resolved control-token ids for a given base (= size of the real vocab)."""

    def __init__(self, base: int = SPECIAL_TOKEN_BASE):
        self.base = int(base)
        for offset, name in enumerate(_SPECIAL_ORDER):
            setattr(self, name, self.base + offset)


def special_token_ids(base: int = SPECIAL_TOKEN_BASE) -> "SpecialTokenIds":
    return SpecialTokenIds(base)


# Backward-compatible module-level constants (base 32000). Value-imports elsewhere
# keep working; code that must honor a different base uses special_token_ids(base).
BOA_TOKEN_ID = SPECIAL_TOKEN_BASE + 0  # Begin of Audio
EOA_TOKEN_ID = SPECIAL_TOKEN_BASE + 1  # End of Audio
BOV_TOKEN_ID = SPECIAL_TOKEN_BASE + 2  # Begin of Voice
EOV_TOKEN_ID = SPECIAL_TOKEN_BASE + 3  # End of Voice
BOI_TOKEN_ID = SPECIAL_TOKEN_BASE + 4  # Begin of Image
EOI_TOKEN_ID = SPECIAL_TOKEN_BASE + 5  # End of Image
AUDIO_PLACEHOLDER_TOKEN_ID = SPECIAL_TOKEN_BASE + 6
VOICE_PLACEHOLDER_TOKEN_ID = SPECIAL_TOKEN_BASE + 7
IMAGE_PLACEHOLDER_TOKEN_ID = SPECIAL_TOKEN_BASE + 8

BEGIN_AUDIO_TOKEN = "<|AUDIO|>"
END_AUDIO_TOKEN = "<|/AUDIO|>"

BEGIN_VOICE_TOKEN = "<|VOICE|>"
END_VOICE_TOKEN = "<|/VOICE|>"

BEGIN_IMAGE_TOKEN = "<|IMAGE|>"
END_IMAGE_TOKEN = "<|/IMAGE|>"
