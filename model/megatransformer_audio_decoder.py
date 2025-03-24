from model import swiglu

import megatransformer_utils
import torch
import torch.nn as nn

class AudioEmbeddingUpsampleConv2dGenerator(nn.Module):
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__()
        self.config = config

        activation = config.image_decoder_activation
        dropout = config.image_decoder_dropout

        self.conv_layers = nn.ModuleList()

        activation_type = megatransformer_utils.get_activation_type(activation)

        channels = [64, 32, 16, 8, 1]
        sizes = [(1, 500), (4, 500), (16, 500), (64, 500), (128, 500)]
        
        for i in range(len(channels) - 1):
            out_channels = channels[i+1]
            upsample_target = sizes[i+1]

            self.conv_layers.append(nn.Sequential(
                nn.Upsample(size=upsample_target, mode="bilinear", align_corners=False),
                nn.Conv2d(channels[i], out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                activation_type() if activation_type is not swiglu.SwiGLU else swiglu.SwiGLU(out_channels),
                nn.Dropout2d(dropout)
            ))

    def forward(self, x: torch.Tensor):
        # naive approach; alternating conv2d and upsample layers to reach n_mels
        # x: [batch_size, channels, timestep, hidden_size]
        x = x.permute(0, 1, 3, 2) # [batch_size, channels, hidden_size, timestep]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], 1, x.shape[3]) # [batch_size, channels * hidden_size, 1, timestep]
        for layer in self.conv_layers:
            x = layer(x)
        return x
