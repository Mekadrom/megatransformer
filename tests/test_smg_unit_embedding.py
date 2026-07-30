"""SMG discrete-token (unit-id) embedding front-end: a pretrained VQ tokenizer (Mimi)
emits integer ids; the SMG embeds them via nn.Embedding(num_codes, sive_encoder_dim)
so the downstream [B, D, T'] float contract is unchanged. Covers the three init modes
and a full contour-mode forward/backward on ids."""
import torch

from megatransformer.model.smg.smg import SMG

CFG = "medium_decoder_only_1d_4x_mimicontour"
K, D = 2048, 256


def _centroids():
    torch.manual_seed(0)
    return torch.randn(K, D)


def _build(init_mode, centroids=None):
    return SMG.from_config(CFG, code_embed_init=init_mode,
                           unit_embed_centroids=centroids)


def test_learned_centroid_init_copies_and_trains():
    c = _centroids()
    m = _build("learned_centroid", c)
    assert m.unit_embedding.weight.shape == (K, D)
    assert torch.allclose(m.unit_embedding.weight.detach(), c, atol=1e-5)
    assert m.unit_embedding.weight.requires_grad is True


def test_frozen_init_copies_and_freezes():
    c = _centroids()
    m = _build("frozen", c)
    assert torch.allclose(m.unit_embedding.weight.detach(), c, atol=1e-5)
    assert m.unit_embedding.weight.requires_grad is False


def test_random_init_differs_and_trains():
    c = _centroids()
    m = _build("learned_random", c)
    assert not torch.allclose(m.unit_embedding.weight.detach(), c, atol=1e-2)
    assert m.unit_embedding.weight.requires_grad is True


def test_centroid_modes_require_centroids():
    for mode in ("learned_centroid", "frozen"):
        try:
            _build(mode, None)
            assert False, f"{mode} should require centroids"
        except ValueError:
            pass


def test_forward_on_ids_contour_mode():
    B, Tp, up = 2, 16, 4
    Tmel = Tp * up
    m = _build("learned_centroid", _centroids())
    ids = torch.randint(0, K, (B, Tp))                 # integer unit ids
    target = torch.randn(B, 80, Tmel)
    mask = torch.ones(B, Tmel, dtype=torch.bool)
    spk = torch.randn(B, 192)
    f0_contour = torch.randn(B, Tmel)
    tf0 = torch.randn(B, Tmel)
    tv = torch.randint(0, 2, (B, Tmel)).float()

    out = m(features=ids, target=target, mask=mask, speaker_embedding=spk,
            target_f0=tf0, target_voiced=tv, f0_contour=f0_contour)
    recon = out[0] if isinstance(out, tuple) else out
    if isinstance(recon, dict):
        recon = recon.get("recon_x", next(iter(recon.values())))
    assert recon.shape[0] == B and recon.shape[-1] == Tmel, recon.shape

    # gradient reaches the (trainable) embedding
    loss = recon.float().pow(2).mean()
    loss.backward()
    assert m.unit_embedding.weight.grad is not None
    assert m.unit_embedding.weight.grad.abs().sum() > 0


def test_vocos_8x_config_resamples_to_target_rate():
    # 24kHz Vocos variant: 100-mel, 8x decoder, then forward resamples the 12.5->100Hz
    # decoder output to the 93.75Hz Vocos mel target length (7.5x, fractional).
    m = SMG.from_config("medium_decoder_only_1d_8x_mimicontour_vocos",
                        code_embed_init="learned_random", unit_embed_centroids=_centroids())
    B, Tp = 2, 16
    Tmel = round(Tp * 7.5)  # 93.75Hz target from 12.5Hz units
    ids = torch.randint(0, K, (B, Tp))
    out = m(features=ids, target=torch.randn(B, 100, Tmel),
            mask=torch.ones(B, Tmel, dtype=torch.bool),
            speaker_embedding=torch.randn(B, 192), f0_contour=torch.randn(B, Tmel))
    recon = out[0] if isinstance(out, tuple) else out
    if isinstance(recon, dict):
        recon = recon.get("recon_x", next(iter(recon.values())))
    # 100 mel bands, resampled from 8*Tp=128 down to the 93.75Hz target Tmel=120
    assert recon.shape == (B, 100, Tmel), recon.shape


def test_float_features_still_pass_through():
    # num_codes>0 but given continuous features -> _embed_ids passes them through
    m = _build("learned_random", _centroids())
    B, Tp = 2, 16
    feats = torch.randn(B, D, Tp)          # float, channel-first
    out = m(features=feats, target=torch.randn(B, 80, Tp * 4),
            mask=torch.ones(B, Tp * 4, dtype=torch.bool),
            speaker_embedding=torch.randn(B, 192),
            f0_contour=torch.randn(B, Tp * 4))
    recon = out[0] if isinstance(out, tuple) else out
    if isinstance(recon, dict):
        recon = recon.get("recon_x", next(iter(recon.values())))
    assert recon.shape[0] == B
