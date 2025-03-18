
import torch
import torch.nn as nn
import megatransformer_utils
import swiglu


class ConvFeatureExtractor(nn.Module):
    def __init__(self, activation, input_channels=1, base_channels=32, kernel_sizes=[3, 3, 3, 3]):
        super().__init__()

        activation_type = megatransformer_utils.get_activation_function(activation)

        self.conv_layers = nn.ModuleList()

        channels = [input_channels] + [base_channels * (2**i) for i in range(len(kernel_sizes))]
        for i in range(len(kernel_sizes)):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], kernel_size=kernel_sizes[i], stride=2, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                swiglu.SwiGLU(channels[i+1]) if activation_type == swiglu.SwiGLU else activation_type()
            ))
    
    def forward(self, x):
        # x: [batch_size, channels, height, width]
        for layer in self.conv_layers:
            x = layer(x)
        return x

class AudioEmbedding(nn.Module):
    def __init__(self, 
                 activation,
                 input_channels=1,
                 max_frames=1024,
                 d_model=512, 
                 n_heads=8, 
                 n_layers=6,
                 inner_dim=2048,
                 dropout=0.1,
                 num_classes=None,
                 num_speakers=None,
                 num_genres=None):
        super().__init__()

        self.conv_extractor = ConvFeatureExtractor(input_channels=input_channels, activation=activation)

        self.pos_encoding = nn.Parameter(torch.zeros(1, max_frames, d_model))

        activation_type = megatransformer_utils.get_activation_function(activation)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=inner_dim,
                dropout=dropout,
                batch_first=True,
                activation=swiglu.SwiGLU(inner_dim) if activation_type == swiglu.SwiGLU else activation_type()
            ),
            num_layers=n_layers,
        )

        self.task_heads = nn.ModuleDict()
        if num_classes:
            self.task_heads['classification'] = nn.Linear(d_model, num_classes)
        if num_speakers:
            self.task_heads['speaker_id'] = nn.Linear(d_model, num_speakers)
        if num_genres:
            self.task_heads['genre'] = nn.Linear(d_model, num_genres)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.TransformerEncoderLayer):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    param.data.normal_(mean=0.0, std=0.02)
                elif 'bias' in name:
                    param.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(self, x: torch.Tensor, task=None, return_hidden_states=False):
        # x: [batch_size, channels, n_mels, frames]

        x = self.conv_extractor(x)

        # Reshape for transformer
        batch_size, _, height, width = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, -1)

        x = x + self.pos_encoding[:, :x.size(1), :]

        x = self.encoder(x)

        x_pooled = x.mean(dim=1)

        if task:
            # Single task mode
            logits = self.task_heads[task](x_pooled)
        else:
            logits = {}
            # Multi-task mode
            for task_name, head in self.task_heads.items():
                logits[task_name] = head(x_pooled)

        return logits if not return_hidden_states else (logits, x)
