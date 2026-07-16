from typing import Optional, Tuple

import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F


from megatransformer.config.voice.sive.sive import CONFIGS as SIVE_CONFIGS
from megatransformer.utils.megatransformer_utils import print_debug_tensor

from megatransformer.config.voice.sive.sive import SpeakerInvariantVoiceEncoderConfig
from megatransformer.model.norms import RMSNorm
from megatransformer.model.voice.sive.conv_subsampling import Conv2dSubsampling, ConvSubsampling
from megatransformer.model.voice.sive.grl import GradientReversalFunction, SpeakerClassifier
from megatransformer.model.voice.sive.mel_augment import MelFrequencyResponse, MelGaussianNoise, MelVTLP
from megatransformer.model.voice.sive.sive_block import SpeakerInvariantVoiceEncoderBlock
from megatransformer.model.voice.sive.spec_augment import SpecAugment


class SpeakerInvariantVoiceEncoder(nn.Module):
    """
    SIVE: Speaker-Invariant Voice Encoder

    Trained with proxy objective:
      - ASR (CTC loss) - forces encoding of linguistic content

    The resulting features are content-focused and speaker-agnostic,
    suitable as targets for a content VAE or direct use in TTS.
    """

    def __init__(self, config: SpeakerInvariantVoiceEncoderConfig):
        super().__init__()

        self.config = config

        # SpecAugment for data augmentation (only active during training)
        if config.use_spec_augment:
            self.spec_augment = SpecAugment(
                time_mask_param=config.spec_time_mask_param,
                freq_mask_param=config.spec_freq_mask_param,
                num_time_masks=config.spec_num_time_masks,
                num_freq_masks=config.spec_num_freq_masks,
            )
        else:
            self.spec_augment = None

        # Post-hoc VTLP — piecewise-linear warp of the mel-bin axis.
        # Applied first so subsequent EQ/noise/masking act on the warped mel,
        # matching the physical order (vocal-tract geometry → channel EQ →
        # ambient noise → training-only masks).
        if config.use_mel_vtlp:
            self.mel_vtlp = MelVTLP(
                strength=config.mel_vtlp_strength,
                prob=config.mel_vtlp_prob,
                boundary_frac=config.mel_vtlp_boundary_frac,
            )
        else:
            self.mel_vtlp = None

        # Mel-space frequency-response modulation (simulates channel/mic EQ).
        if config.use_mel_freq_response:
            self.mel_freq_response = MelFrequencyResponse(
                strength=config.mel_freq_response_strength,
                prob=config.mel_freq_response_prob,
                smoothing_kernel=config.mel_freq_response_smoothing,
            )
        else:
            self.mel_freq_response = None

        # Mel-space additive Gaussian noise at a random SNR (robustness prior).
        if config.use_mel_noise:
            self.mel_noise = MelGaussianNoise(
                snr_min_db=config.mel_noise_snr_min_db,
                snr_max_db=config.mel_noise_snr_max_db,
                prob=config.mel_noise_prob,
            )
        else:
            self.mel_noise = None

        # Convolutional subsampling frontend with optional Dropout1d
        if config.use_conv2d_frontend:
            self.conv_subsample = Conv2dSubsampling(
                out_channels=config.encoder_dim,
                n_mels=config.voice_n_mels,
                kernel_sizes=config.conv_kernel_sizes,
                strides=config.conv_strides,
                dropout=config.conv_dropout if config.conv_dropout > 0 else config.dropout,
                norm_type=config.downsample_norm_type,
            )
        else:
            self.conv_subsample = ConvSubsampling(
                in_channels=config.voice_n_mels,
                out_channels=config.encoder_dim,
                kernel_sizes=config.conv_kernel_sizes,
                strides=config.conv_strides,
                dropout=config.conv_dropout if config.conv_dropout > 0 else config.dropout,
                norm_type=config.downsample_norm_type,
            )

        # Transformer encoder blocks with architectural options
        # Stochastic depth: linearly scale drop_path from 0 to drop_path_rate
        drop_path_rates = [
            config.drop_path_rate * i / max(config.num_layers - 1, 1)
            for i in range(config.num_layers)
        ]
        self.encoder_blocks = nn.ModuleList([
            SpeakerInvariantVoiceEncoderBlock(
                d_model=config.encoder_dim,
                n_heads=config.num_heads,
                d_ff=config.ff_dim,
                dropout=config.dropout,
                head_drop_prob=config.attention_head_drop,
                drop_path_prob=drop_path_rates[i],
                conformer_kernel_size=config.conformer_kernel_size,
                activation=config.activation,
                norm_type=config.block_norm_type,
                conv_norm_type=config.conv_norm_type,
            )
            for i in range(config.num_layers)
        ])

        # Final norm on encoder output. LN does mean-subtraction across the dim
        # axis which creates inter-dim competition (a few high-magnitude dims
        # can suppress others); RMSNorm avoids this. Identity disables entirely.
        if config.final_norm_type == "layernorm":
            self.final_norm = nn.LayerNorm(config.encoder_dim)
        elif config.final_norm_type == "rmsnorm":
            self.final_norm = RMSNorm(config.encoder_dim)
        elif config.final_norm_type == "batchnorm":
            self.final_norm = nn.BatchNorm1d(config.encoder_dim)
        elif config.final_norm_type == "none":
            self.final_norm = nn.Identity()
        else:
            raise ValueError(
                f"Unknown final_norm_type: {config.final_norm_type}. "
                f"Expected one of 'layernorm', 'rmsnorm', 'batchnorm', 'none'."
            )

        # Optional VQ bottleneck on the post-final-norm features. Discretizes the
        # representation the CTC/GRL heads (and downstream SMG) consume.
        self.use_vq = getattr(config, "use_vq", False)
        if self.use_vq:
            from megatransformer.model.voice.sive.vector_quantizer import VectorQuantizerEMA
            self.vq = VectorQuantizerEMA(
                num_codes=config.vq_num_codes, dim=config.encoder_dim,
                commitment_weight=config.vq_commitment_weight, decay=config.vq_ema_decay,
                dead_code_threshold=config.vq_dead_code_threshold,
            )

        # Feature dropout (applied before heads)
        self.feature_dropout = nn.Dropout(config.feature_dropout) if config.feature_dropout > 0 else nn.Identity()

        self.asr_head = nn.Sequential(
            nn.Linear(config.encoder_dim, config.encoder_dim),
            nn.GELU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.encoder_dim, config.vocab_size),
        )

        # Speaker classifier (for GRL)
        self.speaker_classifier = SpeakerClassifier(
            d_model=config.encoder_dim,
            num_speakers=config.num_speakers,
            hidden_dim=config.speaker_classifier_hidden_dim,
            pooling=config.speaker_pooling,
            dropout=config.dropout,
            num_attention_heads=getattr(config, "speaker_classifier_num_heads", 4),
            adversary_target=getattr(config, "speaker_adversary_target", "speaker_id"),
            embedding_dim=getattr(config, "speaker_embedding_dim", 192),
        )

        # Optional gender adversary (GRL), parallel to the speaker adversary. A
        # binary pooled head reversed into its own tap (gender_grl_layer, default 10;
        # see forward), independent of the speaker adversary's layer. Reuses
        # SpeakerClassifier in id-mode with num_speakers=num_genders (CE, not ECAPA).
        self.gender_classifier = None
        if getattr(config, "use_gender_grl", False):
            self.gender_classifier = SpeakerClassifier(
                d_model=config.encoder_dim,
                num_speakers=getattr(config, "num_genders", 2),
                hidden_dim=config.speaker_classifier_hidden_dim,
                pooling=getattr(config, "gender_pooling", None) or config.speaker_pooling,
                dropout=config.dropout,
                num_attention_heads=getattr(config, "speaker_classifier_num_heads", 4),
                adversary_target="speaker_id",
            )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @classmethod
    def from_config(cls, config_name: str, **overrides) -> "SpeakerInvariantVoiceEncoder":
        """
        Create model from predefined config with optional overrides.

        Args:
            config_name: One of predefined configs
            **overrides: Override any config parameter

        Example:
            model = SpeakerInvariantVoiceEncoder.from_config("small", num_speakers=500)
        """
        if config_name not in SIVE_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(SIVE_CONFIGS.keys())}")

        config = SIVE_CONFIGS[config_name]
        # Apply overrides
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(overrides)
        config = SpeakerInvariantVoiceEncoderConfig(**config_dict)

        return cls(config)

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input mel spectrogram length."""
        return self.conv_subsample.get_output_length(input_length)

    @torch._dynamo.disable
    def _apply_mel_augmentations(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Training-only mel-space augmentations (all no-op in eval mode).

        Excluded from torch.compile via @torch._dynamo.disable: SpecAugment uses
        data-dependent random mask shapes (.item() on per-sample widths) that
        force dynamo graph breaks, and these are cheap augmentation not worth
        compiling — dynamo runs this region eagerly and resumes compiling the
        encoder after it. Order: VTLP (formant warp / vocal-tract geometry), then
        EQ-like gain (channel coloration), then additive noise (ambient), then
        SpecAugment masks. Roughly matches the physical signal path.
        """
        if self.mel_vtlp is not None:
            mel_spec = self.mel_vtlp(mel_spec)
        if self.mel_freq_response is not None:
            mel_spec = self.mel_freq_response(mel_spec)
        if self.mel_noise is not None:
            mel_spec = self.mel_noise(mel_spec)
        if self.spec_augment is not None:
            mel_spec = self.spec_augment(mel_spec)
        return mel_spec

    def forward(
        self,
        mel_spec: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        grl_alpha: float = 1.0,
        return_all_hiddens: bool = False,
    ) -> dict:
        """
        Forward pass through SIVE.

        Args:
            mel_spec: [B, n_mels, T] mel spectrogram
            lengths: [B] original lengths before padding (optional)
            grl_alpha: Gradient reversal strength (0=no reversal, 1=full reversal)
            return_all_hiddens: If True, return hidden states from all layers

        Returns:
            dict with keys:
                - features: [B, T', encoder_dim] normalized content features
                - features_unnorm: [B, T', encoder_dim] pre-LayerNorm features
                - asr_logits: [B, T', vocab_size] for CTC loss
                - speaker_logits: [B, num_speakers] for GRL speaker classification
                - feature_lengths: [B] output sequence lengths
                - variance_loss: scalar loss for variance regularization (if enabled)
                - temporal_smoothness: scalar metric (if variance_reg enabled and training)
                - all_hiddens: list of [B, T', D] if return_all_hiddens=True
        """
        # Mel-space augmentations (training-only; no-op in eval). Isolated in a
        # @torch._dynamo.disable method so --compile_model doesn't graph-break on
        # SpecAugment's data-dependent random mask shapes (.item() per sample).
        mel_spec = self._apply_mel_augmentations(mel_spec)

        # Convolutional frontend
        x: torch.Tensor = self.conv_subsample(mel_spec)  # [B, encoder_dim, T']

        x = x.permute(0, 2, 1)  # [B, T', encoder_dim]

        # InstanceNorm in conv_subsample upcasts to float32 internally; match
        # encoder block parameter dtype so LayerNorm doesn't error under bf16.
        x = x.to(next(self.encoder_blocks.parameters()).dtype)

        # Calculate output lengths
        feature_lengths = None
        padding_mask = None
        if lengths is not None:
            # Vectorized: was a per-sample `.item()` list comprehension, which
            # forced a torch.compile graph break. get_output_length is pure
            # integer arithmetic ((L + s - 1) // s), so it broadcasts over the
            # lengths tensor directly — also faster (no Python loop / per-element
            # host sync) even without compile.
            feature_lengths = self.get_output_length(lengths).to(
                device=mel_spec.device, dtype=torch.long
            )
            # Create padding mask [B, T'] - True for padded positions
            max_len = x.size(1)
            padding_mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= feature_lengths.unsqueeze(1)

        # Transformer encoder
        all_hiddens = [x] if return_all_hiddens else None
        # GRL attachment layer: capture the hidden the speaker adversary reverses into.
        # all_hiddens index convention: 0 = conv frontend output, k = output after block k-1.
        # -1 (default) => attach to the final features (below); >=0 => attach here so the
        # adversarial speaker-scrub lands where the SMG taps (e.g. layer 10).
        grl_layer = getattr(self.config, "grl_layer", -1)
        # The gender adversary can tap a DIFFERENT layer (default 10). Only capture
        # its hidden when the gender head exists.
        gender_grl_layer = getattr(self.config, "gender_grl_layer", 10)
        capture_gender = self.gender_classifier is not None
        grl_hidden = x if grl_layer == 0 else None
        gender_grl_hidden = x if (capture_gender and gender_grl_layer == 0) else None
        for i, block in enumerate(self.encoder_blocks):
            x = block(x, key_padding_mask=padding_mask)

            if return_all_hiddens:
                all_hiddens.append(x)
            if grl_layer == i + 1:   # all_hiddens[i+1] = output after block i
                grl_hidden = x
            if capture_gender and gender_grl_layer == i + 1:
                gender_grl_hidden = x

        # Store pre-LayerNorm features
        features_unnorm = x  # [B, T, encoder_dim]

        # Final normalization
        if isinstance(self.final_norm, nn.BatchNorm1d):
            # BatchNorm1d expects input of shape [B, C, T], so permute
            features = self.final_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            features = self.final_norm(x)

        # VQ bottleneck (post-norm): replace features with their quantized codes so every
        # consumer -- variance reg, CTC head, GRL adversary, feature extraction -- operates
        # on the discrete representation. valid_mask excludes padding from the codebook.
        vq_commitment_loss = None
        vq_indices = None
        vq_perplexity = None
        if self.use_vq:
            vq_mask = (~padding_mask) if padding_mask is not None else None
            features, vq_indices, vq_commitment_loss, vq_perplexity = self.vq(features, mask=vq_mask)

        # Compute variance regularization loss (for VAE-friendly features)
        variance_loss = torch.tensor(0.0, device=mel_spec.device)
        temporal_smoothness = None
        if self.config.use_variance_reg:
            # Use normalized features for variance computation
            feat_for_var = features

            # Temporal variance: want features to change over time (not constant)
            if feat_for_var.size(1) > 1:
                temporal_diff = feat_for_var[:, 1:, :] - feat_for_var[:, :-1, :]
                temporal_var = temporal_diff.var(dim=1).mean()
                temporal_loss = F.relu(self.config.temporal_var_min - temporal_var)

                # Temporal smoothness (for logging) - cosine similarity between adjacent frames
                with torch.no_grad():
                    feat_norm = F.normalize(feat_for_var, dim=-1)
                    cos_sim = (feat_norm[:, 1:, :] * feat_norm[:, :-1, :]).sum(dim=-1)
                    temporal_smoothness = cos_sim.mean()
            else:
                temporal_loss = torch.tensor(0.0, device=mel_spec.device)

            # Per-dimension variance: want each dimension to be used (not dead)
            dim_var = feat_for_var.var(dim=(0, 1))
            dim_loss = F.relu(self.config.dim_var_min - dim_var).mean()

            # Temporal smoothness penalty: penalize if features are too smooth
            smoothness_loss = torch.tensor(0.0, device=mel_spec.device)
            if temporal_smoothness is not None:
                smoothness_loss = F.relu(temporal_smoothness - self.config.temporal_smoothness_max)

            variance_loss = (
                self.config.temporal_var_weight * temporal_loss +
                self.config.dim_var_weight * dim_loss +
                self.config.temporal_smoothness_weight * smoothness_loss
            )

        # Std hinge + covariance regularization (independently togglable; both
        # disabled by default). Computed in float32 over padding-masked frames
        # so padded positions don't deflate the variance estimate. Training-
        # only — at eval the losses are zero.
        std_hinge_loss = torch.zeros((), device=mel_spec.device)
        cov_loss = torch.zeros((), device=mel_spec.device)
        run_feature_reg = self.training and (
            self.config.use_std_hinge or self.config.use_covariance_reg
        )
        if run_feature_reg:
            if padding_mask is not None:
                valid_positions = ~padding_mask  # [B, T'] True for valid
                flat_feat = features[valid_positions]  # [N_valid, D]
            else:
                flat_feat = features.reshape(-1, features.size(-1))
            flat_feat_fp32 = flat_feat.float()

            if flat_feat_fp32.size(0) >= 2:
                if self.config.use_std_hinge:
                    dim_std = flat_feat_fp32.std(dim=0)  # [D]
                    std_hinge_loss = self.config.dim_std_weight * F.relu(
                        self.config.dim_std_min - dim_std
                    ).mean()

                    if self.config.temporal_std_weight > 0 and features.size(1) > 1:
                        diffs = features[:, 1:, :] - features[:, :-1, :]  # [B, T-1, D]
                        if padding_mask is not None:
                            diff_valid = (~padding_mask[:, 1:]) & (~padding_mask[:, :-1])
                            flat_diffs = diffs[diff_valid]
                        else:
                            flat_diffs = diffs.reshape(-1, diffs.size(-1))
                        if flat_diffs.size(0) >= 2:
                            temporal_std = flat_diffs.float().std(dim=0)
                            std_hinge_loss = std_hinge_loss + self.config.temporal_std_weight * F.relu(
                                self.config.temporal_std_min - temporal_std
                            ).mean()

                if self.config.use_covariance_reg:
                    centered = flat_feat_fp32 - flat_feat_fp32.mean(dim=0, keepdim=True)
                    n_valid = centered.size(0)
                    d = centered.size(1)
                    cov = (centered.t() @ centered) / max(n_valid - 1, 1)  # [D, D]
                    off_diag_sq = cov.pow(2).sum() - cov.diag().pow(2).sum()
                    cov_loss = self.config.cov_weight * off_diag_sq / d

        # Apply feature dropout before heads
        features_for_heads = self.feature_dropout(features)

        # CTC upsampling: upsample features before ASR head to relax CTC length constraint
        # This keeps transformer efficient (operates on T/4) while giving CTC more frames
        ctc_lengths = feature_lengths
        if self.config.ctc_upsample_factor > 1:
            # Linear interpolation upsampling: [B, T, D] -> [B, T*factor, D]
            features_for_asr = features_for_heads.transpose(1, 2)  # [B, D, T]
            features_for_asr = F.interpolate(
                features_for_asr,
                scale_factor=float(self.config.ctc_upsample_factor),
                mode='linear',
                align_corners=False,
            )
            features_for_asr = features_for_asr.transpose(1, 2)  # [B, T*factor, D]

            # Update CTC lengths to reflect upsampling
            if feature_lengths is not None:
                ctc_lengths = feature_lengths * self.config.ctc_upsample_factor
        else:
            features_for_asr = features_for_heads

        # ASR head (operates on potentially upsampled features)
        asr_logits = self.asr_head(features_for_asr)

        # Speaker classifier with gradient reversal (operates on original resolution).
        # By default the adversary reverses into the FINAL features; with grl_layer>=0 it
        # reverses into that intermediate layer instead, so the speaker-scrub lands where the
        # SMG taps. CTC still trains the final layer, whose gradient flows back through the
        # scrubbed layer, so blocks above grl_layer keep refining content on cleaned features.
        valid_mask = ~padding_mask if padding_mask is not None else None
        grl_input = self.feature_dropout(grl_hidden) if grl_hidden is not None else features_for_heads
        reversed_features = GradientReversalFunction.apply(grl_input, grl_alpha)
        speaker_logits = self.speaker_classifier(reversed_features, mask=valid_mask)

        # Gender adversary reverses into its OWN tap (gender_grl_layer, default 10),
        # independent of the speaker adversary's layer. When both taps coincide, the
        # speaker adversary's already-reversed hidden is reused.
        gender_logits = None
        if self.gender_classifier is not None:
            if gender_grl_layer == grl_layer:
                reversed_gender = reversed_features
            else:
                gender_grl_input = (
                    self.feature_dropout(gender_grl_hidden)
                    if gender_grl_hidden is not None else features_for_heads
                )
                reversed_gender = GradientReversalFunction.apply(gender_grl_input, grl_alpha)
            gender_logits = self.gender_classifier(reversed_gender, mask=valid_mask)

        result = {
            "features": features,  # Normalized features (preferred for VAE)
            "features_unnorm": features_unnorm,  # Pre-LayerNorm (for comparison)
            "asr_logits": asr_logits,
            "speaker_logits": speaker_logits,
            "gender_logits": gender_logits,  # [B, num_genders] for GRL gender adversary (None if disabled)
            "feature_lengths": feature_lengths,  # Original feature lengths (for feature extraction)
            "ctc_lengths": ctc_lengths,  # CTC lengths (potentially upsampled, for CTC loss)
            "variance_loss": variance_loss,
            "temporal_smoothness": temporal_smoothness if self.config.use_variance_reg and self.training else None,
            "std_hinge_loss": std_hinge_loss,
            "cov_loss": cov_loss,
            # VQ (None when use_vq is False). vq_commitment_loss goes into the objective;
            # vq_perplexity is the code-usage health metric (near vq_num_codes = healthy,
            # collapsing toward 1 = codebook collapse); vq_indices [B, T] are the codes the
            # world model will predict and the SMG will consume.
            "vq_commitment_loss": vq_commitment_loss,
            "vq_indices": vq_indices,
            "vq_perplexity": vq_perplexity,
        }

        if return_all_hiddens:
            result["all_hiddens"] = all_hiddens

        return result

    def extract_features(
        self,
        mel_spec: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        layer: int = -1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract features without computing ASR head output.
        Useful for inference after training.

        Args:
            mel_spec: [B, n_mels, T]
            lengths: [B] optional lengths
            layer: Which layer to extract from (-1 = final)

        Returns:
            features: [B, T', encoder_dim]
            feature_lengths: [B] or None
        """
        with torch.no_grad():
            result = self.forward(
                mel_spec,
                lengths=lengths,
                return_all_hiddens=(layer != -1),
            )

        if layer == -1:
            features = result["features"]
        else:
            features = result["all_hiddens"][layer]

        return features, result["feature_lengths"]

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
