"""qk-clip (MuonClip) wiring + correctness.

The checks that matter here are the ones the inert-flag pattern in this repo keeps producing:
that OFF is bit-identical, and that ON actually moves the weights. The recurrent ones exist
because the trunk calls one attention module many times per forward, which breaks the naive
"reset the accumulator never / clip per call" implementations.
"""
import math

import pytest
import torch
import torch.nn as nn

from megatransformer.config.common import MegaTransformerBlockConfig
from megatransformer.model.transformer import MegaTransformerAttention
from megatransformer.model.qk_clip import QKClipController


def _cfg():
    return MegaTransformerBlockConfig(d_model=64, n_heads=4, n_query_groups=2,
                                      d_queries=16, d_values=16, causal=True)


def _net(cfg):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.a0 = MegaTransformerAttention(cfg)
            self.a1 = MegaTransformerAttention(cfg)
    return Net()


def test_default_is_off_and_bit_identical():
    torch.manual_seed(0)
    cfg = _cfg()
    assert cfg.qk_clip_tau is None, "qk-clip must default to OFF"
    net, x = _net(cfg), torch.randn(2, 12, 64)
    before = {k: v.clone() for k, v in net.state_dict().items()}
    with torch.no_grad():
        net.a0(x); net.a1(x)
    assert all(getattr(m, "_qk_max_logit", None) is None for m in (net.a0, net.a1))
    after = net.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_probe_matches_hand_computed_max_logit():
    torch.manual_seed(0)
    cfg = _cfg()
    net, x = _net(cfg), torch.randn(2, 12, 64)
    ctrl = QKClipController(net, tau=1e9, probe_every=1)
    assert len(ctrl) == 2
    ctrl.maybe_arm(0)
    with torch.no_grad():
        net.a0(x)
    S = net.a0._qk_max_logit
    assert S is not None and S.shape == (cfg.n_heads,) and torch.isfinite(S).all()
    with torch.no_grad():
        q = net.a0.q_proj(x).view(2, 12, cfg.n_heads, cfg.d_queries).permute(0, 2, 1, 3)
        k = net.a0.k_proj(x).view(2, 12, cfg.n_query_groups, cfg.d_queries).permute(0, 2, 1, 3)
        k = k.repeat_interleave(cfg.n_heads // cfg.n_query_groups, dim=1)
        s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(cfg.d_queries)
        s = s.masked_fill(torch.tril(torch.ones(12, 12)) == 0, float("-inf"))
        ref = s.amax(dim=3).amax(dim=2).amax(dim=0)
    assert torch.allclose(S, ref, atol=1e-4)


def test_tau_above_observed_is_a_noop():
    torch.manual_seed(0)
    cfg = _cfg()
    net, x = _net(cfg), torch.randn(2, 12, 64)
    ctrl = QKClipController(net, tau=1e9, probe_every=1)
    ctrl.maybe_arm(0)
    with torch.no_grad():
        net.a0(x); net.a1(x)
    wq, wk = net.a0.q_proj.weight.clone(), net.a0.k_proj.weight.clone()
    stats = ctrl.apply(0)
    assert torch.equal(net.a0.q_proj.weight, wq)
    assert torch.equal(net.a0.k_proj.weight, wk)
    assert stats["qk_clip/heads_clipped"] == 0.0


def test_clip_brings_every_head_to_tau():
    torch.manual_seed(0)
    cfg = _cfg()
    net, x = _net(cfg), torch.randn(2, 12, 64)
    probe = QKClipController(net, tau=1e9, probe_every=1)
    probe.maybe_arm(0)
    with torch.no_grad():
        net.a0(x)
    tau = float(net.a0._qk_max_logit.max()) / 4.0
    probe.apply(0)

    ctrl = QKClipController(net, tau=tau, probe_every=1)
    ctrl.maybe_arm(0)
    with torch.no_grad():
        net.a0(x); net.a1(x)
    stats = ctrl.apply(0)
    assert stats["qk_clip/heads_clipped"] > 0

    ctrl.maybe_arm(0)
    with torch.no_grad():
        net.a0(x)
    assert int((net.a0._qk_max_logit > tau * 1.01).sum()) == 0


def test_weight_shared_module_is_discovered_once():
    """Two references to one module must not compound gamma**n."""
    torch.manual_seed(0)
    cfg = _cfg()
    shared = MegaTransformerAttention(cfg)

    class Twice(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = shared
            self.again = shared

    assert len(QKClipController(Twice(), tau=1e9, probe_every=1)) == 1


def test_accumulator_spans_iterations_but_resets_between_steps():
    """The trunk calls one module per iteration: the max must span them, and must NOT
    survive into the next probe window (a stale max would clip forever)."""
    torch.manual_seed(0)
    cfg = _cfg()
    shared = MegaTransformerAttention(cfg)

    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.blk = shared

    ctrl = QKClipController(Holder(), tau=1e9, probe_every=1)
    ctrl.maybe_arm(0)
    xs = [torch.randn(2, 12, 64) * s for s in (1.0, 7.0)]
    with torch.no_grad():
        shared(xs[0])
        after_small = shared._qk_max_logit.max().clone()
        shared(xs[1])
        after_big = shared._qk_max_logit.max().clone()
    assert after_big > after_small, "max did not accumulate across iterations"

    ctrl.apply(0)
    ctrl.maybe_arm(1)
    assert shared._qk_max_logit is None, "stale max survived re-arm"


def test_disarms_so_eval_forward_cannot_pollute_the_measurement():
    torch.manual_seed(0)
    cfg = _cfg()
    net, x = _net(cfg), torch.randn(2, 12, 64)
    ctrl = QKClipController(net, tau=1e9, probe_every=1)
    ctrl.maybe_arm(0)
    with torch.no_grad():
        net.a0(x)
    ctrl.apply(0)
    assert getattr(net.a0, "_qk_probe_active", False) is False
    before = net.a0._qk_max_logit.clone()
    with torch.no_grad():
        net.a0(torch.randn(2, 12, 64) * 50)
    assert torch.equal(before, net.a0._qk_max_logit)
