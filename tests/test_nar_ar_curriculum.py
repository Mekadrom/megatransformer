"""Tests for the NAR→AR voice-attention curriculum (Variant B).

The world-TTS voice AR leans on the voice-history CRUTCH so text is under-used. The
curriculum down-scales voice→voice attention by a schedulable alpha in the prelude,
recurrent trunk, and coda: at alpha=0 a voice position attends to text and its own
shifted-TF input frame (t-1) ONLY. These tests pin the two load-bearing properties:

  1. PARITY: alpha>=1 / alpha=None is a bit-for-bit no-op (fast attention path).
  2. CRUTCH-INVARIANCE: at alpha=0, perturbing GT voice frame j changes ONLY the
     output at position j+1 (position t depends on frame t-1 alone) — proving the
     voice→voice history is severed in prelude + trunk + coda, while the intended
     weak 1-frame residual (the shifted-TF input) remains.

Determinism note: the recurrent block initializes its thought state randomly, so every
paired comparison seeds torch immediately before each forward — same random init, so
any output delta is attributable to the input/alpha change alone.
"""
import math

import pytest
import torch

from megatransformer.model.world.token_alignment import (
    MODALITY_PAD,
    MODALITY_TEXT,
    MODALITY_VOICE,
    VOICE_ATTN_MASK_NEG,
    build_all_voice_attn_bias,
    build_voice_voice_attn_bias,
    voice_attn_bias_value,
)

VOICE_PLACEHOLDER = 32007  # small_sum token_interleaver_config.voice_placeholder_token_id
SEED = 1234
B, C, T = 2, 128, 10


# ── Model fixture (heavy — build once) ─────────────────────────────────────────
@pytest.fixture(scope="module")
def world_model():
    from megatransformer.model.world.world_model import MegaTransformerWorldModel
    model = MegaTransformerWorldModel.from_config("small_sum")
    model.eval()
    return model


@pytest.fixture(scope="module")
def voice_batch():
    torch.manual_seed(0)
    text = torch.tensor([[5, 6, 7, VOICE_PLACEHOLDER],
                         [8, 9, 10, VOICE_PLACEHOLDER]])
    vin = torch.randn(B, 1, C, T)
    vlen = torch.tensor([[T], [T]])
    syn = torch.tensor([True, True])
    return text, vin, vlen, syn


def _fwd(model, text, voice, vlen, syn, alpha, labels=None):
    torch.manual_seed(SEED)  # match random thought-state init across paired calls
    with torch.no_grad():
        out = model(
            text_input_ids=text, voice_inputs=voice, voice_lengths=vlen,
            voice_latent_labels=labels, is_synthesis=syn, decode_outputs=False,
            voice_attn_alpha=alpha,
        )
    return out


# ── 1. Bias builders ───────────────────────────────────────────────────────────
def test_bias_value_math():
    assert voice_attn_bias_value(None) == 0.0
    assert voice_attn_bias_value(1.0) == 0.0
    assert voice_attn_bias_value(2.0) == 0.0            # >=1 clamps to no-op
    assert voice_attn_bias_value(0.0) == VOICE_ATTN_MASK_NEG
    assert math.isclose(voice_attn_bias_value(0.5), math.log(0.5), rel_tol=1e-6)


def test_trunk_bias_offdiagonal_only():
    mm = torch.tensor([[MODALITY_TEXT, MODALITY_VOICE, MODALITY_VOICE, MODALITY_PAD]])
    bias = build_voice_voice_attn_bias(mm, 0.0, torch.float32)
    # Only the two off-diagonal voice-voice pairs (1,2) and (2,1) are severed.
    assert bias[0, 1, 2].item() == VOICE_ATTN_MASK_NEG
    assert bias[0, 2, 1].item() == VOICE_ATTN_MASK_NEG
    assert bias[0, 1, 1].item() == 0.0                  # diagonal kept (no NaN rows)
    assert bias[0, 0, 1].item() == 0.0                  # text query untouched
    assert bias[0, 1, 3].item() == 0.0                  # pad key untouched


def test_bias_none_when_alpha_ge_one():
    mm = torch.tensor([[MODALITY_VOICE, MODALITY_VOICE]])
    assert build_voice_voice_attn_bias(mm, 1.0, torch.float32) is None
    assert build_voice_voice_attn_bias(mm, None, torch.float32) is None
    assert build_all_voice_attn_bias(4, 1.0, torch.device("cpu"), torch.float32) is None


def test_is_synthesis_gates_transcription_rows():
    mm = torch.tensor([[MODALITY_VOICE, MODALITY_VOICE],
                       [MODALITY_VOICE, MODALITY_VOICE]])
    iss = torch.tensor([True, False])
    bias = build_voice_voice_attn_bias(mm, 0.0, torch.float32, is_synthesis=iss)
    assert bias[0, 0, 1].item() == VOICE_ATTN_MASK_NEG   # synthesis row: severed
    assert bias[1, 0, 1].item() == 0.0                   # transcription row: untouched

    av = build_all_voice_attn_bias(3, 0.0, torch.device("cpu"), torch.float32,
                                   batch_size=2, is_synthesis=iss)
    assert av.shape == (2, 3, 3)
    assert av[0, 0, 1].item() == VOICE_ATTN_MASK_NEG
    assert av[1, 0, 1].item() == 0.0
    assert torch.all(av[0].diag() == 0.0)                # diagonal kept


# ── 2. Forward parity: alpha>=1 / None is a no-op ───────────────────────────────
def test_alpha_none_equals_no_arg(world_model, voice_batch):
    text, vin, vlen, syn = voice_batch
    with_none = _fwd(world_model, text, vin, vlen, syn, None)["voice_latent_preds"]
    torch.manual_seed(SEED)
    with torch.no_grad():
        no_arg = world_model(
            text_input_ids=text, voice_inputs=vin, voice_lengths=vlen,
            is_synthesis=syn, decode_outputs=False,
        )["voice_latent_preds"]
    assert torch.equal(with_none, no_arg)


def test_alpha_one_is_identity(world_model, voice_batch):
    text, vin, vlen, syn = voice_batch
    a_none = _fwd(world_model, text, vin, vlen, syn, None)["voice_latent_preds"]
    a_one = _fwd(world_model, text, vin, vlen, syn, 1.0)["voice_latent_preds"]
    assert torch.equal(a_none, a_one)


def test_alpha_zero_changes_output(world_model, voice_batch):
    text, vin, vlen, syn = voice_batch
    a_one = _fwd(world_model, text, vin, vlen, syn, 1.0)["voice_latent_preds"]
    a_zero = _fwd(world_model, text, vin, vlen, syn, 0.0)["voice_latent_preds"]
    # Severing history genuinely changes the prediction (the mechanism is active).
    assert not torch.allclose(a_zero, a_one, atol=1e-5)


# ── 3. ⭐ Crutch-invariance — the key test ───────────────────────────────────────
def test_crutch_invariance_alpha_zero(world_model, voice_batch):
    """At alpha=0, perturbing GT voice frame j changes ONLY output position j+1.

    Position t's shifted-TF input is frame t-1, so frame j feeds trunk position j+1;
    with voice→voice severed everywhere, that perturbation cannot reach any other
    output position. This proves the history crutch (frames != t-1) is fully starved
    while the intended weak 1-frame input residual is preserved.
    """
    text, vin, vlen, syn = voice_batch
    base = _fwd(world_model, text, vin, vlen, syn, 0.0)["voice_latent_preds"]  # (B,C,T)
    torch.manual_seed(0)
    for j in (2, 4, 6):
        vpert = vin.clone()
        vpert[:, :, :, j] = torch.randn(B, 1, C)          # perturb frame j only
        pert = _fwd(world_model, text, vpert, vlen, syn, 0.0)["voice_latent_preds"]
        per_pos = (pert - base).abs().sum(dim=1)          # (B, T), summed over channels
        changed = (per_pos[0] > 1e-5).nonzero().flatten().tolist()
        assert changed == [j + 1], f"frame {j}: expected only position {j+1}, got {changed}"


def test_frame_t_minus_1_residual_is_live(world_model, voice_batch):
    """The 1-frame residual (Variant B, not C): perturbing frame t-1 DOES move
    position t's output — that input crutch is intentionally left in place."""
    text, vin, vlen, syn = voice_batch
    base = _fwd(world_model, text, vin, vlen, syn, 0.0)["voice_latent_preds"]
    t = 5
    torch.manual_seed(1)
    vpert = vin.clone()
    vpert[:, :, :, t - 1] = torch.randn(B, 1, C)          # frame t-1 = position t's input
    pert = _fwd(world_model, text, vpert, vlen, syn, 0.0)["voice_latent_preds"]
    assert (pert[:, :, t] - base[:, :, t]).abs().max() > 1e-5


def test_alpha_one_uses_history(world_model, voice_batch):
    """Positive control: at alpha=1 an early-frame perturbation propagates causally
    to every downstream position (history is fully used)."""
    text, vin, vlen, syn = voice_batch
    base = _fwd(world_model, text, vin, vlen, syn, 1.0)["voice_latent_preds"]
    torch.manual_seed(2)
    vpert = vin.clone()
    vpert[:, :, :, 2] = torch.randn(B, 1, C)
    pert = _fwd(world_model, text, vpert, vlen, syn, 1.0)["voice_latent_preds"]
    per_pos = (pert - base).abs().sum(dim=1)[0]
    # Frame 2 feeds position 3; causal attention lets 3..T-1 (7 positions) see it.
    assert int((per_pos > 1e-5).sum()) >= T - 3


# ── 4. Integration: loss forward at alpha=0 and alpha=1 stays finite ─────────────
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_loss_forward_finite(world_model, voice_batch, alpha):
    text, vin, vlen, syn = voice_batch
    labels = vin.squeeze(1)  # (B, C, T) — coda L1/MSE target
    out = _fwd(world_model, text, vin, vlen, syn, alpha, labels=labels)
    for key in ("voice_latent_l1_loss", "voice_latent_mse_loss"):
        assert key in out
        assert torch.isfinite(out[key]).all()


# ── 5. Trainer alpha schedule (duck-typed — avoids constructing a full Trainer) ──
def _schedule(mask_steps, ramp_steps, floor, cap, step):
    from megatransformer.scripts.train.world.training import WorldModelTrainer

    class _Dummy:
        pass

    d = _Dummy()
    d.voice_ar_attn_mask_steps = mask_steps
    d.voice_ar_attn_ramp_steps = ramp_steps
    d.voice_ar_attn_floor = floor
    d.voice_ar_attn_cap = cap
    d._voice_ar_attn_enabled = (mask_steps > 0 or ramp_steps > 0)
    return WorldModelTrainer._voice_attn_alpha(d, step)


def test_schedule_off_by_default():
    assert _schedule(0, 0, 0.0, 1.0, 0) is None
    assert _schedule(0, 0, 0.0, 1.0, 50000) is None


def test_schedule_boundaries():
    ms, ramp, floor, cap = 10000, 20000, 0.0, 1.0
    assert _schedule(ms, ramp, floor, cap, 0) == floor            # NAR phase start
    assert _schedule(ms, ramp, floor, cap, ms - 1) == floor       # NAR phase end
    assert _schedule(ms, ramp, floor, cap, ms) == floor           # ramp start = floor
    mid = _schedule(ms, ramp, floor, cap, ms + ramp // 2)
    assert math.isclose(mid, 0.5, abs_tol=1e-6)                   # ramp midpoint
    assert _schedule(ms, ramp, floor, cap, ms + ramp) == cap      # ramp end = cap
    assert _schedule(ms, ramp, floor, cap, ms + ramp + 99999) == cap  # holds at cap


def test_schedule_respects_floor_and_cap():
    # A non-zero floor and a sub-1 cap (permanent partial suppression).
    assert _schedule(5000, 10000, 0.2, 0.8, 0) == 0.2
    assert math.isclose(_schedule(5000, 10000, 0.2, 0.8, 10000), 0.5, abs_tol=1e-6)
    assert _schedule(5000, 10000, 0.2, 0.8, 999999) == 0.8
