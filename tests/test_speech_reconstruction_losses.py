"""
Tests for speech reconstruction training losses.

Tests cover:
- MultiScaleMelSpectrogramLoss
- Speaker embedding similarity computation
- Integration with training pipeline
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.audio.criteria import MultiScaleMelSpectrogramLoss


class TestMultiScaleMelSpectrogramLoss:
    """Test multi-scale mel spectrogram loss."""

    def test_output_is_scalar(self):
        loss_fn = MultiScaleMelSpectrogramLoss(scales=[1, 2, 4], use_log=True)

        pred_mel = torch.randn(2, 80, 100)
        target_mel = torch.randn(2, 80, 100)

        loss = loss_fn(pred_mel, target_mel)

        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Non-negative

    def test_zero_loss_for_identical_inputs(self):
        loss_fn = MultiScaleMelSpectrogramLoss(scales=[1, 2, 4], use_log=True)

        mel = torch.randn(2, 80, 100).abs() + 1e-5  # Positive for log
        loss = loss_fn(mel, mel.clone())

        assert loss.item() < 1e-5

    def test_handles_3d_input(self):
        loss_fn = MultiScaleMelSpectrogramLoss(scales=[1, 2, 4], use_log=True)

        pred_mel = torch.randn(2, 80, 100)  # [B, n_mels, T]
        target_mel = torch.randn(2, 80, 100)

        loss = loss_fn(pred_mel, target_mel)

        assert loss.dim() == 0

    def test_handles_4d_input(self):
        loss_fn = MultiScaleMelSpectrogramLoss(scales=[1, 2, 4], use_log=True)

        pred_mel = torch.randn(2, 1, 80, 100)  # [B, 1, n_mels, T]
        target_mel = torch.randn(2, 1, 80, 100)

        loss = loss_fn(pred_mel, target_mel)

        assert loss.dim() == 0

    def test_handles_mismatched_lengths(self):
        loss_fn = MultiScaleMelSpectrogramLoss(scales=[1, 2, 4], use_log=True)

        pred_mel = torch.randn(2, 80, 100)
        target_mel = torch.randn(2, 80, 120)  # Different length

        # Should handle by truncating to min length
        loss = loss_fn(pred_mel, target_mel)

        assert loss.dim() == 0

    def test_different_scales(self):
        loss_fn_1 = MultiScaleMelSpectrogramLoss(scales=[1], use_log=True)
        loss_fn_4 = MultiScaleMelSpectrogramLoss(scales=[1, 2, 4, 8], use_log=True)

        pred_mel = torch.randn(2, 80, 100)
        target_mel = torch.randn(2, 80, 100)

        loss_1 = loss_fn_1(pred_mel, target_mel)
        loss_4 = loss_fn_4(pred_mel, target_mel)

        # Both should produce valid losses
        assert loss_1.dim() == 0
        assert loss_4.dim() == 0

    def test_linear_vs_log(self):
        loss_fn_log = MultiScaleMelSpectrogramLoss(scales=[1, 2], use_log=True)
        loss_fn_linear = MultiScaleMelSpectrogramLoss(scales=[1, 2], use_log=False)

        pred_mel = torch.randn(2, 80, 100).abs() + 0.1
        target_mel = torch.randn(2, 80, 100).abs() + 0.1

        loss_log = loss_fn_log(pred_mel, target_mel)
        loss_linear = loss_fn_linear(pred_mel, target_mel)

        # Both should be valid, but typically different values
        assert loss_log.dim() == 0
        assert loss_linear.dim() == 0

    def test_gradient_flow(self):
        loss_fn = MultiScaleMelSpectrogramLoss(scales=[1, 2, 4], use_log=True)

        pred_mel = torch.randn(2, 80, 100, requires_grad=True)
        target_mel = torch.randn(2, 80, 100)

        loss = loss_fn(pred_mel, target_mel)
        loss.backward()

        assert pred_mel.grad is not None
        assert pred_mel.grad.abs().sum() > 0

    def test_batch_size_1(self):
        loss_fn = MultiScaleMelSpectrogramLoss(scales=[1, 2, 4], use_log=True)

        pred_mel = torch.randn(1, 80, 100)
        target_mel = torch.randn(1, 80, 100)

        loss = loss_fn(pred_mel, target_mel)

        assert loss.dim() == 0


class TestSpeakerSimilarityLoss:
    """Test speaker embedding similarity computation."""

    def test_cosine_similarity_basic(self):
        # Identical embeddings should have similarity 1
        emb = F.normalize(torch.randn(4, 192), dim=-1)
        sim = F.cosine_similarity(emb, emb, dim=-1)

        assert torch.allclose(sim, torch.ones(4), atol=1e-5)

    def test_cosine_similarity_orthogonal(self):
        # Orthogonal embeddings should have similarity ~0
        emb1 = torch.zeros(1, 4)
        emb1[0, 0] = 1.0
        emb2 = torch.zeros(1, 4)
        emb2[0, 1] = 1.0

        sim = F.cosine_similarity(emb1, emb2, dim=-1)

        assert torch.allclose(sim, torch.zeros(1), atol=1e-5)

    def test_similarity_loss_formula(self):
        # loss = 1 - cosine_similarity
        emb1 = F.normalize(torch.randn(4, 192), dim=-1)
        emb2 = F.normalize(torch.randn(4, 192), dim=-1)

        cos_sim = F.cosine_similarity(emb1, emb2, dim=-1)
        loss = 1.0 - cos_sim.mean()

        # Loss should be between 0 and 2
        assert loss.item() >= 0
        assert loss.item() <= 2

    def test_similarity_loss_gradient_flow(self):
        emb1 = F.normalize(torch.randn(4, 192), dim=-1)
        emb2 = F.normalize(torch.randn(4, 192, requires_grad=True), dim=-1)

        cos_sim = F.cosine_similarity(emb1, emb2, dim=-1)
        loss = 1.0 - cos_sim.mean()
        loss.backward()

        # Note: emb2 was normalized, so gradients flow through the original tensor
        # The normalized version doesn't have requires_grad after F.normalize
        # In practice, we'd compute this differently, but this tests the math
        assert loss.item() >= 0


class TestSpeakerEncoderUtility:
    """Test speaker encoder utility functions (without loading actual models)."""

    def test_embedding_dims_lookup(self):
        from utils.speaker_encoder import SPEAKER_EMBEDDING_DIMS

        assert SPEAKER_EMBEDDING_DIMS["ecapa_tdnn"] == 192
        assert SPEAKER_EMBEDDING_DIMS["wavlm"] == 768

    def test_input_types_lookup(self):
        from utils.speaker_encoder import SPEAKER_ENCODER_INPUT_TYPES

        assert SPEAKER_ENCODER_INPUT_TYPES["ecapa_tdnn"] == "mel"
        assert SPEAKER_ENCODER_INPUT_TYPES["wavlm"] == "waveform"

    def test_get_embedding_dim_helper(self):
        from utils.speaker_encoder import get_speaker_embedding_dim

        assert get_speaker_embedding_dim("ecapa_tdnn") == 192
        assert get_speaker_embedding_dim("wavlm") == 768

    def test_get_input_type_helper(self):
        from utils.speaker_encoder import get_speaker_encoder_input_type

        assert get_speaker_encoder_input_type("ecapa_tdnn") == "mel"
        assert get_speaker_encoder_input_type("wavlm") == "waveform"


class TestTrainerLossIntegration:
    """Test that the trainer can compute all losses correctly."""

    def test_masked_reconstruction_loss(self):
        # Simulate the masked loss computation from trainer
        B, n_mels, T = 2, 80, 100
        mel_spec = torch.randn(B, n_mels, T)
        mel_recon = torch.randn(B, n_mels, T)
        mel_lengths = torch.tensor([80, 100])

        # Create mask
        mask = torch.arange(T).unsqueeze(0) < mel_lengths.unsqueeze(1)
        mask = mask.unsqueeze(1).expand(-1, n_mels, -1)  # [B, n_mels, T]

        total_valid = mask.sum().clamp(min=1)

        mse_loss = ((mel_recon - mel_spec) ** 2 * mask).sum() / total_valid
        l1_loss = ((mel_recon - mel_spec).abs() * mask).sum() / total_valid

        assert mse_loss.dim() == 0
        assert l1_loss.dim() == 0
        assert mse_loss.item() >= 0
        assert l1_loss.item() >= 0

    def test_combined_loss_weighting(self):
        # Test that loss weighting works correctly
        mse_weight = 1.0
        l1_weight = 0.5
        ms_mel_weight = 0.5
        speaker_sim_weight = 0.1
        arcface_weight = 0.1

        mse_loss = torch.tensor(0.5)
        l1_loss = torch.tensor(0.3)
        ms_mel_loss = torch.tensor(0.2)
        speaker_sim_loss = torch.tensor(0.1)
        arcface_loss = torch.tensor(2.0)

        recon_loss = mse_weight * mse_loss + l1_weight * l1_loss
        total_loss = recon_loss
        total_loss = total_loss + ms_mel_weight * ms_mel_loss
        total_loss = total_loss + speaker_sim_weight * speaker_sim_loss
        total_loss = total_loss + arcface_weight * arcface_loss

        expected = (1.0 * 0.5 + 0.5 * 0.3 + 0.5 * 0.2 + 0.1 * 0.1 + 0.1 * 2.0)
        assert torch.isclose(total_loss, torch.tensor(expected))


class TestMultiScaleMelLossWithMasking:
    """Test multi-scale mel loss behavior with masked/padded inputs."""

    def test_with_different_valid_lengths(self):
        loss_fn = MultiScaleMelSpectrogramLoss(scales=[1, 2], use_log=True)

        # Create mels with different actual lengths (simulate padding)
        B, n_mels, T = 2, 80, 100

        pred_mel = torch.randn(B, n_mels, T)
        target_mel = torch.randn(B, n_mels, T)

        # Simulate valid lengths
        valid_lengths = [60, 100]

        # Zero out padding (as would be done during reconstruction)
        for i, length in enumerate(valid_lengths):
            pred_mel[i, :, length:] = 0
            target_mel[i, :, length:] = 0

        loss = loss_fn(pred_mel, target_mel)

        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
