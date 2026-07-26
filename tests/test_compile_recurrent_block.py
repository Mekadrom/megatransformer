"""Wiring tests for MegatransformerRecurrentBlock.compile_blocks.

The numerical correctness of block-level torch.compile (allclose to eager, no recompile
storm across sequence lengths, train+eval both fine) was validated manually against the
real inductor backend. These fast tests pin the wiring WITHOUT paying the inductor compile
cost: torch.compile is monkeypatched to a marker so we can assert each unique block's
forward is wrapped exactly once, the call is idempotent, shared blocks are deduplicated,
and — critically — state_dict keys stay clean (no `_orig_mod.` prefix that would break
every checkpoint/eval loader).
"""
import torch

from megatransformer.config.world.world_model import WORLD_MODEL_CONFIGS
from megatransformer.model.world.recurrent import MegatransformerRecurrentBlock


def _make_block(share):
    cfg = WORLD_MODEL_CONFIGS["small_sum"].recurrent_block_config
    cfg.share_block_weights = share
    return MegatransformerRecurrentBlock(cfg)


def test_compile_keeps_state_dict_keys_clean(monkeypatch):
    calls = {"n": 0}

    def fake_compile(fn, **kw):
        calls["n"] += 1
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    block = _make_block(share=False)
    keys_before = set(block.state_dict().keys())

    block.compile_blocks(dynamic=True)

    assert block._blocks_compiled is True
    assert set(block.state_dict().keys()) == keys_before          # no _orig_mod. prefix
    assert not any("_orig_mod" in k for k in block.state_dict().keys())
    # One compile per unique block in the bank.
    assert calls["n"] == len(block.recurrent_blocks)


def test_compile_is_idempotent(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: (calls.__setitem__("n", calls["n"] + 1), fn)[1])
    block = _make_block(share=False)
    block.compile_blocks()
    first = calls["n"]
    block.compile_blocks()                                        # second call is a no-op
    assert calls["n"] == first


def test_shared_blocks_compiled_once(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: (calls.__setitem__("n", calls["n"] + 1), fn)[1])
    block = _make_block(share=True)
    # share_block_weights => every ModuleList entry is the SAME object.
    assert len({id(b) for b in block.recurrent_blocks}) == 1
    block.compile_blocks()
    assert calls["n"] == 1                                        # compiled once, not N times
