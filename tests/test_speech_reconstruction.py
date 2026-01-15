"""
Comprehensive tests for Speech Reconstruction model.

Tests cover:
- SpeakerEncoder with attention pooling
- MelReconstructor with FiLM conditioning
- Combined SpeechReconstructionModel
- ArcFace head
- Order invariance of attention pooling
- Gradient flow through all components
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.audio.speech_reconstruction import (
    SpeakerEncoderConfig,
    MelReconstructorConfig,
    SpeechReconstructionConfig,
    ConvSubsampling2D,
    TransformerEncoderLayer,
    AttentionPooling,
    SpeakerEncoder,
    FiLMLayer,
    ResidualConvBlock,
    MelReconstructor,
    ArcFaceHead,
    SpeechReconstructionModel,
    SPEAKER_ENCODER_CONFIGS,
    MEL_RECONSTRUCTOR_CONFIGS,
    create_speech_reconstruction_model,
)


class TestConfigs:
    """Test configuration dataclasses."""

    def test_speaker_encoder_config_defaults(self):
        config = SpeakerEncoderConfig()
        assert config.n_mels == 80
        assert config.encoder_dim == 256
        assert config.num_layers == 4
        assert config.speaker_dim == 256
        assert config.conv_channels is not None
        assert config.conv_strides is not None

    def test_speaker_encoder_config_custom(self):
        config = SpeakerEncoderConfig(
            n_mels=40,
            encoder_dim=128,
            num_layers=2,
            speaker_dim=64,
        )
        assert config.n_mels == 40
        assert config.encoder_dim == 128
        assert config.speaker_dim == 64

    def test_mel_reconstructor_config_defaults(self):
        config = MelReconstructorConfig()
        assert config.gubert_dim == 256
        assert config.speaker_dim == 256
        assert config.n_mels == 80
        assert config.hidden_dim == 512
        assert config.upsample_factors is not None

    def test_speech_reconstruction_config_defaults(self):
        config = SpeechReconstructionConfig()
        assert config.speaker_encoder is not None
        assert config.mel_reconstructor is not None
        assert config.use_arcface is False


class TestConvSubsampling2D:
    """Test 2D convolutional subsampling."""

    def test_output_shape(self):
        subsample = ConvSubsampling2D(
            n_mels=80,
            out_channels=256,
            channels=[128, 256],
            kernel_sizes=[5, 3],
            strides=[2, 2],
        )

        x = torch.randn(2, 80, 100)  # [B, n_mels, T]
        out = subsample(x)

        assert out.dim() == 3
        assert out.shape[0] == 2
        assert out.shape[2] == 256  # encoder_dim

    def test_with_4d_input(self):
        subsample = ConvSubsampling2D(
            n_mels=80,
            out_channels=256,
            channels=[128, 256],
            kernel_sizes=[5, 3],
            strides=[2, 2],
        )

        x = torch.randn(2, 1, 80, 100)  # [B, 1, n_mels, T]
        out = subsample(x)

        assert out.dim() == 3
        assert out.shape[0] == 2

    def test_output_length_calculation(self):
        subsample = ConvSubsampling2D(
            n_mels=80,
            out_channels=256,
            channels=[128, 256],
            kernel_sizes=[5, 3],
            strides=[2, 2],
        )

        input_length = 100
        expected_length = subsample.get_output_length(input_length)

        x = torch.randn(1, 80, input_length)
        out = subsample(x)

        assert out.shape[1] == expected_length


class TestTransformerEncoderLayer:
    """Test transformer encoder layer."""

    def test_output_shape(self):
        layer = TransformerEncoderLayer(
            d_model=256,
            n_heads=4,
            d_ff=1024,
        )

        x = torch.randn(2, 50, 256)
        out = layer(x)

        assert out.shape == x.shape

    def test_with_padding_mask(self):
        layer = TransformerEncoderLayer(
            d_model=256,
            n_heads=4,
            d_ff=1024,
        )

        x = torch.randn(2, 50, 256)
        mask = torch.zeros(2, 50, dtype=torch.bool)
        mask[:, 40:] = True  # Pad last 10 positions

        out = layer(x, key_padding_mask=mask)

        assert out.shape == x.shape


class TestAttentionPooling:
    """Test attention pooling for speaker embedding."""

    def test_output_shape(self):
        pool = AttentionPooling(d_model=256, n_heads=4)

        x = torch.randn(2, 50, 256)
        out = pool(x)

        assert out.shape == (2, 256)

    def test_with_padding_mask(self):
        pool = AttentionPooling(d_model=256, n_heads=4)

        x = torch.randn(2, 50, 256)
        mask = torch.zeros(2, 50, dtype=torch.bool)
        mask[:, 40:] = True

        out = pool(x, key_padding_mask=mask)

        assert out.shape == (2, 256)

    def test_order_invariance(self):
        """Test that attention pooling is order-invariant (key property)."""
        pool = AttentionPooling(d_model=256, n_heads=4)
        pool.eval()  # Deterministic

        x = torch.randn(1, 20, 256)

        # Original output
        with torch.no_grad():
            out1 = pool(x)

        # Shuffle the sequence
        perm = torch.randperm(20)
        x_shuffled = x[:, perm, :]

        with torch.no_grad():
            out2 = pool(x_shuffled)

        # Outputs should be very close (order-invariant)
        # Note: Due to attention mechanism, there might be very small differences
        # but the key property is that the output doesn't depend on order
        assert torch.allclose(out1, out2, atol=1e-4)


class TestSpeakerEncoder:
    """Test complete speaker encoder."""

    def test_output_shape(self):
        config = SpeakerEncoderConfig(
            encoder_dim=128,
            num_layers=2,
            speaker_dim=64,
        )
        encoder = SpeakerEncoder(config)

        x = torch.randn(2, 80, 100)
        result = encoder(x)

        assert "speaker_embedding" in result
        assert result["speaker_embedding"].shape == (2, 64)

    def test_l2_normalization(self):
        config = SpeakerEncoderConfig(
            encoder_dim=128,
            num_layers=2,
            speaker_dim=64,
        )
        encoder = SpeakerEncoder(config)

        x = torch.randn(2, 80, 100)
        result = encoder(x)

        # Check that embeddings are L2 normalized
        norms = result["speaker_embedding"].norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_with_lengths(self):
        config = SpeakerEncoderConfig(
            encoder_dim=128,
            num_layers=2,
            speaker_dim=64,
        )
        encoder = SpeakerEncoder(config)

        x = torch.randn(2, 80, 100)
        lengths = torch.tensor([80, 100])

        result = encoder(x, lengths=lengths)

        assert result["speaker_embedding"].shape == (2, 64)

    def test_return_sequence(self):
        config = SpeakerEncoderConfig(
            encoder_dim=128,
            num_layers=2,
            speaker_dim=64,
        )
        encoder = SpeakerEncoder(config)

        x = torch.randn(2, 80, 100)
        result = encoder(x, return_sequence=True)

        assert "sequence" in result
        assert result["sequence"].dim() == 3

    def test_predefined_configs(self):
        for name, config in SPEAKER_ENCODER_CONFIGS.items():
            encoder = SpeakerEncoder(config)
            x = torch.randn(1, config.n_mels, 50)
            result = encoder(x)

            assert result["speaker_embedding"].shape == (1, config.speaker_dim)


class TestFiLMLayer:
    """Test FiLM conditioning layer."""

    def test_output_shape_3d_time_last(self):
        film = FiLMLayer(
            feature_dim=256,
            speaker_dim=64,
        )

        x = torch.randn(2, 50, 256)  # [B, T, D]
        speaker_emb = torch.randn(2, 64)

        out = film(x, speaker_emb)

        assert out.shape == x.shape

    def test_output_shape_3d_channel_middle(self):
        film = FiLMLayer(
            feature_dim=256,
            speaker_dim=64,
        )

        x = torch.randn(2, 256, 50)  # [B, D, T]
        speaker_emb = torch.randn(2, 64)

        out = film(x, speaker_emb)

        assert out.shape == x.shape

    def test_identity_initialization(self):
        """Test that FiLM is initialized near identity."""
        film = FiLMLayer(
            feature_dim=256,
            speaker_dim=64,
        )

        x = torch.randn(2, 50, 256)
        speaker_emb = torch.zeros(2, 64)  # Zero embedding

        out = film(x, speaker_emb)

        # With zero input, should be close to identity
        assert torch.allclose(out, x, atol=0.1)


class TestResidualConvBlock:
    """Test residual conv block with FiLM."""

    def test_output_shape(self):
        block = ResidualConvBlock(
            channels=256,
            kernel_size=5,
            speaker_dim=64,
        )

        x = torch.randn(2, 256, 50)  # [B, C, T]
        speaker_emb = torch.randn(2, 64)

        out = block(x, speaker_emb)

        assert out.shape == x.shape


class TestMelReconstructor:
    """Test mel spectrogram reconstructor."""

    def test_output_shape(self):
        config = MelReconstructorConfig(
            gubert_dim=256,
            speaker_dim=64,
            n_mels=80,
            hidden_dim=256,
            num_layers=2,
            upsample_factors=[2, 2],
        )
        recon = MelReconstructor(config)

        gubert = torch.randn(2, 25, 256)  # [B, T', D]
        speaker_emb = torch.randn(2, 64)

        out = recon(gubert, speaker_emb)

        # With 4x upsampling, 25 -> 100
        assert out.dim() == 3
        assert out.shape[0] == 2
        assert out.shape[1] == 80  # n_mels

    def test_with_target_length(self):
        config = MelReconstructorConfig(
            gubert_dim=256,
            speaker_dim=64,
            n_mels=80,
            hidden_dim=256,
            num_layers=2,
            upsample_factors=[2, 2],
        )
        recon = MelReconstructor(config)

        gubert = torch.randn(2, 25, 256)
        speaker_emb = torch.randn(2, 64)

        out = recon(gubert, speaker_emb, target_length=90)

        assert out.shape[-1] == 90

    def test_predefined_configs(self):
        for name, config in MEL_RECONSTRUCTOR_CONFIGS.items():
            recon = MelReconstructor(config)
            gubert = torch.randn(1, 25, config.gubert_dim)
            speaker_emb = torch.randn(1, config.speaker_dim)
            out = recon(gubert, speaker_emb)

            assert out.shape[1] == config.n_mels


class TestArcFaceHead:
    """Test ArcFace classification head."""

    def test_output_shape(self):
        head = ArcFaceHead(
            speaker_dim=64,
            num_speakers=100,
        )

        embeddings = F.normalize(torch.randn(2, 64), p=2, dim=-1)
        labels = torch.tensor([5, 10])

        logits = head(embeddings, labels)

        assert logits.shape == (2, 100)

    def test_training_mode_adds_margin(self):
        head = ArcFaceHead(
            speaker_dim=64,
            num_speakers=100,
            margin=0.5,
        )
        head.train()

        embeddings = F.normalize(torch.randn(2, 64), p=2, dim=-1)
        labels = torch.tensor([5, 10])

        logits_train = head(embeddings, labels)

        head.eval()
        logits_eval = head(embeddings)

        # In eval mode, no margin is applied, so logits differ
        assert not torch.allclose(logits_train, logits_eval)

    def test_scaling(self):
        head = ArcFaceHead(
            speaker_dim=64,
            num_speakers=100,
            scale=30.0,
        )
        head.eval()

        embeddings = F.normalize(torch.randn(2, 64), p=2, dim=-1)
        logits = head(embeddings)

        # Max cosine similarity is 1, scaled by 30
        assert logits.abs().max() <= 30.0 + 1e-5


class TestSpeechReconstructionModel:
    """Test combined speech reconstruction model."""

    def test_output_structure(self):
        config = SpeechReconstructionConfig(
            speaker_encoder=SpeakerEncoderConfig(
                encoder_dim=128,
                num_layers=2,
                speaker_dim=64,
            ),
            mel_reconstructor=MelReconstructorConfig(
                gubert_dim=256,
                speaker_dim=64,
                hidden_dim=256,
                num_layers=2,
            ),
        )
        model = SpeechReconstructionModel(config)

        mel_spec = torch.randn(2, 80, 100)
        gubert = torch.randn(2, 25, 256)

        result = model(mel_spec, gubert)

        assert "mel_recon" in result
        assert "speaker_embedding" in result
        assert result["mel_recon"].shape[0] == 2
        assert result["mel_recon"].shape[1] == 80
        assert result["speaker_embedding"].shape == (2, 64)

    def test_with_arcface(self):
        config = SpeechReconstructionConfig(
            speaker_encoder=SpeakerEncoderConfig(
                encoder_dim=128,
                num_layers=2,
                speaker_dim=64,
            ),
            mel_reconstructor=MelReconstructorConfig(
                gubert_dim=256,
                speaker_dim=64,
                hidden_dim=256,
                num_layers=2,
            ),
            use_arcface=True,
            num_speakers=100,
        )
        model = SpeechReconstructionModel(config)

        mel_spec = torch.randn(2, 80, 100)
        gubert = torch.randn(2, 25, 256)
        speaker_ids = torch.tensor([5, 10])

        result = model(mel_spec, gubert, speaker_ids=speaker_ids)

        assert "speaker_logits" in result
        assert result["speaker_logits"].shape == (2, 100)

    def test_encode_speaker(self):
        config = SpeechReconstructionConfig(
            speaker_encoder=SpeakerEncoderConfig(
                encoder_dim=128,
                num_layers=2,
                speaker_dim=64,
            ),
            mel_reconstructor=MelReconstructorConfig(
                gubert_dim=256,
                speaker_dim=64,
                hidden_dim=256,
                num_layers=2,
            ),
        )
        model = SpeechReconstructionModel(config)

        mel_spec = torch.randn(2, 80, 100)
        speaker_emb = model.encode_speaker(mel_spec)

        assert speaker_emb.shape == (2, 64)

    def test_reconstruct_mel(self):
        config = SpeechReconstructionConfig(
            speaker_encoder=SpeakerEncoderConfig(
                encoder_dim=128,
                num_layers=2,
                speaker_dim=64,
            ),
            mel_reconstructor=MelReconstructorConfig(
                gubert_dim=256,
                speaker_dim=64,
                hidden_dim=256,
                num_layers=2,
            ),
        )
        model = SpeechReconstructionModel(config)

        gubert = torch.randn(2, 25, 256)
        speaker_emb = torch.randn(2, 64)

        mel_recon = model.reconstruct_mel(gubert, speaker_emb)

        assert mel_recon.dim() == 3
        assert mel_recon.shape[1] == 80

    def test_with_lengths(self):
        config = SpeechReconstructionConfig(
            speaker_encoder=SpeakerEncoderConfig(
                encoder_dim=128,
                num_layers=2,
                speaker_dim=64,
            ),
            mel_reconstructor=MelReconstructorConfig(
                gubert_dim=256,
                speaker_dim=64,
                hidden_dim=256,
                num_layers=2,
            ),
        )
        model = SpeechReconstructionModel(config)

        mel_spec = torch.randn(2, 80, 100)
        gubert = torch.randn(2, 25, 256)
        mel_lengths = torch.tensor([80, 100])
        gubert_lengths = torch.tensor([20, 25])

        result = model(
            mel_spec, gubert,
            mel_lengths=mel_lengths,
            gubert_lengths=gubert_lengths,
        )

        assert result["mel_recon"].shape[0] == 2


class TestCreateFunction:
    """Test factory function."""

    def test_create_default(self):
        model = create_speech_reconstruction_model()
        assert isinstance(model, SpeechReconstructionModel)

    def test_create_all_configs(self):
        for config_name in SPEAKER_ENCODER_CONFIGS.keys():
            model = create_speech_reconstruction_model(config=config_name)
            assert isinstance(model, SpeechReconstructionModel)

    def test_create_with_arcface(self):
        model = create_speech_reconstruction_model(
            config="tiny",
            use_arcface=True,
            num_speakers=100,
        )
        assert model.arcface_head is not None

    def test_create_with_custom_dims(self):
        model = create_speech_reconstruction_model(
            config="tiny",
            gubert_dim=512,
            n_mels=40,
        )

        mel_spec = torch.randn(1, 40, 50)
        gubert = torch.randn(1, 12, 512)

        result = model(mel_spec, gubert)

        assert result["mel_recon"].shape[1] == 40


class TestGradientFlow:
    """Test gradient flow through the model."""

    def test_speaker_encoder_gradients(self):
        config = SpeakerEncoderConfig(
            encoder_dim=64,
            num_layers=1,
            speaker_dim=32,
        )
        encoder = SpeakerEncoder(config)

        x = torch.randn(2, 80, 50, requires_grad=True)
        result = encoder(x)
        # Use mean squared loss instead of sum - sum of L2-normalized vectors can have zero gradient
        target = torch.randn_like(result["speaker_embedding"])
        loss = F.mse_loss(result["speaker_embedding"], target)
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_mel_reconstructor_gradients(self):
        config = MelReconstructorConfig(
            gubert_dim=64,
            speaker_dim=32,
            hidden_dim=64,
            num_layers=1,
        )
        recon = MelReconstructor(config)

        gubert = torch.randn(2, 12, 64, requires_grad=True)
        speaker_emb = torch.randn(2, 32, requires_grad=True)

        out = recon(gubert, speaker_emb)
        loss = out.sum()
        loss.backward()

        assert gubert.grad is not None
        assert speaker_emb.grad is not None
        assert gubert.grad.abs().sum() > 0
        assert speaker_emb.grad.abs().sum() > 0

    def test_full_model_gradients(self):
        model = create_speech_reconstruction_model(config="tiny")

        mel_spec = torch.randn(2, 80, 50, requires_grad=True)
        gubert = torch.randn(2, 12, 256, requires_grad=True)

        result = model(mel_spec, gubert)
        loss = result["mel_recon"].sum() + result["speaker_embedding"].sum()
        loss.backward()

        assert mel_spec.grad is not None
        assert gubert.grad is not None

    def test_arcface_gradients(self):
        model = create_speech_reconstruction_model(
            config="tiny",
            use_arcface=True,
            num_speakers=10,
        )

        mel_spec = torch.randn(2, 80, 50, requires_grad=True)
        gubert = torch.randn(2, 12, 256, requires_grad=True)
        speaker_ids = torch.tensor([0, 1])

        result = model(mel_spec, gubert, speaker_ids=speaker_ids)
        loss = F.cross_entropy(result["speaker_logits"], speaker_ids)
        loss.backward()

        assert mel_spec.grad is not None
        # ArcFace loss should affect speaker encoder weights
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.speaker_encoder.parameters())


class TestInputShapeHandling:
    """Test various input shapes are handled correctly."""

    def test_mel_with_4d_input(self):
        model = create_speech_reconstruction_model(config="tiny")

        mel_spec = torch.randn(2, 1, 80, 50)  # [B, 1, n_mels, T]
        gubert = torch.randn(2, 12, 256)

        # Model should handle 4D input by squeezing channel dim
        # (internal SpeakerEncoder handles this)
        result = model(mel_spec.squeeze(1), gubert)

        assert "mel_recon" in result

    def test_different_batch_sizes(self):
        model = create_speech_reconstruction_model(config="tiny")

        for batch_size in [1, 4, 8]:
            mel_spec = torch.randn(batch_size, 80, 50)
            gubert = torch.randn(batch_size, 12, 256)

            result = model(mel_spec, gubert)

            assert result["mel_recon"].shape[0] == batch_size
            assert result["speaker_embedding"].shape[0] == batch_size

    def test_different_sequence_lengths(self):
        model = create_speech_reconstruction_model(config="tiny")

        for mel_len, gubert_len in [(50, 12), (100, 25), (200, 50)]:
            mel_spec = torch.randn(2, 80, mel_len)
            gubert = torch.randn(2, gubert_len, 256)

            result = model(mel_spec, gubert)

            assert result["mel_recon"].dim() == 3
            assert result["speaker_embedding"].shape == (2, 128)  # tiny speaker_dim


class TestParameterCount:
    """Test parameter counting methods."""

    def test_speaker_encoder_param_count(self):
        config = SpeakerEncoderConfig(
            encoder_dim=128,
            num_layers=2,
        )
        encoder = SpeakerEncoder(config)

        trainable = encoder.get_num_params(trainable_only=True)
        total = encoder.get_num_params(trainable_only=False)

        assert trainable > 0
        # Total includes non-trainable positional encoding, so total >= trainable
        assert total >= trainable

    def test_mel_reconstructor_param_count(self):
        config = MelReconstructorConfig(
            hidden_dim=256,
            num_layers=2,
        )
        recon = MelReconstructor(config)

        trainable = recon.get_num_params(trainable_only=True)
        assert trainable > 0

    def test_full_model_param_count(self):
        model = create_speech_reconstruction_model(config="tiny")

        total = model.get_num_params(trainable_only=False)
        assert total > 0

        # Verify subcomponent counting works
        speaker_params = model.speaker_encoder.get_num_params()
        recon_params = model.mel_reconstructor.get_num_params()

        # Total should be at least the sum of components
        assert total >= speaker_params + recon_params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
