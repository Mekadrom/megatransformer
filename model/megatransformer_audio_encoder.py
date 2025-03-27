from model import megatransformer_modules

import megatransformer_utils
import torch
import torch.nn as nn


class AudioConv(nn.Module):
    def __init__(self, input_channels=1, base_channels=32, kernel_sizes=[3, 3, 3, 3, 3], dropout=0.1, activation="gelu"):
        super().__init__()
        self.conv_layers = nn.ModuleList()

        activation_type = megatransformer_utils.get_activation_type(activation)

        channels = [input_channels] + [base_channels * (2**i) for i in range(len(kernel_sizes))]
        for i in range(len(kernel_sizes)):
            out_channels = channels[i+1]

            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(channels[i], out_channels, kernel_size=kernel_sizes[i], stride=(2, 1), padding=1),
                nn.BatchNorm2d(out_channels),
                activation_type() if activation_type is not megatransformer_modules.SwiGLU else megatransformer_modules.SwiGLU(out_channels),
                nn.Dropout2d(dropout)
            ))
    
    def forward(self, features: torch.Tensor):
        # x: [batch_size, channels, height, width]
        for i, layer in enumerate(self.conv_layers):
            features = layer(features)
        return features

class AudioFeatureExtractor(nn.Module):
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__()
        self.config = config

        self.pos_encoding = nn.Parameter(torch.zeros(1, config.audio_max_frames, config.hidden_size))
        
        self.conv_feature_extractor = AudioConv(
            input_channels=1,
            base_channels=config.audio_encoder_base_channels,
            kernel_sizes=config.audio_encoder_kernel_sizes,
            dropout=config.audio_encoder_dropout,
            activation=config.audio_encoder_activation,
        )

        conv_output_channels = config.audio_encoder_base_channels * (2**(len(config.audio_encoder_kernel_sizes) - 1))
        self.conv_projection = nn.Linear(conv_output_channels * 2, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.prelude = megatransformer_modules.SimpleBlock(
            config.audio_prelude_config, config.audio_prelude_config.n_prelude_layers, config.hidden_dropout_prob
        )

    def forward(
        self,
        features: torch.Tensor,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        N, *_, T = features.shape

        features = self.conv_feature_extractor(features)

        features = features.permute(0, 3, 1, 2) # [batch_size, audio_seq_len, channels, n_mels]
        features = features.reshape(N, T, -1) # [batch_size, audio_seq_len, channels * hidden_size]

        features = self.conv_projection(features)

        features = features + self.pos_encoding[:, :T, :]

        features = self.dropout(features)

        features = self.prelude(
            features,
            attention_mask=torch.ones((N, T), device=features.device),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return features
