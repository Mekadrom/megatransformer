
# Mistral base tokenizer special tokens
EOS_TOKEN_ID = 2  # End of Sequence (Mistral-7B EOS)

# The 9 multimodal control tokens sit immediately after the "real" vocab, at
# base + 0..8 in a FIXED order. `base` = the real vocab size: 32000 for the
# Mistral tokenizer (the historical default below), or the pretrained LLM's
# native vocab (e.g. 49152 for SmolLM2) when config.special_token_base is set.
# Anything that emits/detects these tokens must use the SAME base as the data.
SPECIAL_TOKEN_BASE = 32_000  # Mistral vocab size (default)

# DURATION BUCKETS (NAR voice synthesis). A masked-parallel voice decoder must know its
# block length BEFORE refinement starts -- unlike AR, where EOV simply halts and positions
# past it never exist. So the model emits a duration token right after BOV and the block is
# sized from it.
#
# These are ordinary control tokens: they ride the SAME trainable special_embed/special_head
# extension as BOV/EOV (feature_extractor splices ids >= llm_native_vocab through
# special_embed; generator concatenates special_head onto the frozen lm_head), so the
# pretrained LLM stays frozen and nothing about its vocab changes.
#
# LOG-spaced, not linear, and the range is measured rather than assumed. Sampled 8102 train
# utterances: min 25, median 102, mean 112, p90 204, and only 0.16% at the 250 cap (so the
# cap is not biasing the target). Linear 8-frame buckets would be ~3% wide at 250 frames but
# ~28% at 25 -- worst exactly where an oversized block hurts most. 32 log buckets over
# [25, 250] are 7.5% wide each and ALL 32 come out populated (occupancy 98-356).
#
# Appended AFTER the original 9 so every existing id keeps its value.
N_DURATION_BUCKETS = 32
DURATION_MIN_FRAMES = 25
DURATION_MAX_FRAMES = 250

# ⚠️ N_SPECIAL_TOKENS IS A MODEL SHAPE. It sizes special_embed/special_head, so changing the
# DEFAULT resizes those tensors for EVERY modality and breaks in-flight runs of unrelated
# models (this is not hypothetical -- raising it to 41 for the voice duration buckets broke a
# world-image run). The duration buckets are therefore OPT-IN: the default stays 9, and only a
# run that passes --voice_nar_duration_token gets N_SPECIAL_TOKENS_WITH_DURATION.
#
# The DUR_* names stay in _SPECIAL_ORDER unconditionally, which is safe: ids are assigned by
# offset, so appending names cannot move an existing id, and nothing is allocated from the
# order tuple alone.
N_SPECIAL_TOKENS = 9
N_SPECIAL_TOKENS_WITH_DURATION = 9 + N_DURATION_BUCKETS
_SPECIAL_ORDER = ("BOA", "EOA", "BOV", "EOV", "BOI", "EOI",
                  "AUDIO_PLACEHOLDER", "VOICE_PLACEHOLDER", "IMAGE_PLACEHOLDER"
                  ) + tuple(f"DUR_{i:02d}" for i in range(N_DURATION_BUCKETS))


def duration_bucket(n_frames: int,
                    lo: int = DURATION_MIN_FRAMES,
                    hi: int = DURATION_MAX_FRAMES,
                    n: int = N_DURATION_BUCKETS) -> int:
    """Frame count -> bucket index in [0, n). Clamped, so out-of-range lengths still land."""
    import math
    x = min(max(int(n_frames), lo), hi)
    frac = math.log(x / lo) / math.log(hi / lo)
    return min(n - 1, max(0, int(frac * n)))


def duration_bucket_frames(bucket: int,
                           lo: int = DURATION_MIN_FRAMES,
                           hi: int = DURATION_MAX_FRAMES,
                           n: int = N_DURATION_BUCKETS) -> int:
    """Bucket index -> representative frame count (the bucket's GEOMETRIC midpoint).

    Geometric, not arithmetic: the buckets are log-spaced, so the arithmetic midpoint sits
    above the middle of the bucket's mass and would bias every block slightly long.
    """
    import math
    b = min(n - 1, max(0, int(bucket)))
    ratio = hi / lo
    edge_lo = lo * ratio ** (b / n)
    edge_hi = lo * ratio ** ((b + 1) / n)
    return int(round(math.sqrt(edge_lo * edge_hi)))


def duration_bucket_alloc(bucket: int, margin: float = 0.05,
                          lo: int = DURATION_MIN_FRAMES,
                          hi: int = DURATION_MAX_FRAMES,
                          n: int = N_DURATION_BUCKETS,
                          cap: int = DURATION_MAX_FRAMES) -> int:
    """Block size to ALLOCATE for a predicted bucket -- the upper edge plus a margin.

    Deliberately not the midpoint: the two errors are not symmetric. Under-allocating
    truncates content with no way to recover it, while over-allocating only wastes
    refinement on positions that EOV then trims. So predict with duration_bucket_frames
    (an unbiased length estimate) but SIZE the block with this.
    """
    import math
    b = min(n - 1, max(0, int(bucket)))
    edge_hi = lo * (hi / lo) ** ((b + 1) / n)
    return int(min(cap, max(1, math.ceil(edge_hi * (1.0 + margin)))))


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
