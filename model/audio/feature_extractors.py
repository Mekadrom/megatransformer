from model import activations, megatransformer_attn, megatransformer_modules
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
                activation_type() if activation_type is not activations.SwiGLU else activations.SwiGLU(out_channels),
                nn.Dropout2d(dropout)
            ))

        self._init_weights()

    def _init_weights(self):
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
    
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
            config.audio_prelude_config, "audio_prelude", config.audio_prelude_config.n_prelude_layers, config.hidden_dropout_prob
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.conv_projection.weight)
        if self.conv_projection.bias is not None:
            nn.init.zeros_(self.conv_projection.bias)

    def forward(
        self,
        audio_raw_inputs: torch.Tensor,
        audio_waveform_labels: torch.Tensor,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        N, *_, T = audio_raw_inputs.shape

        audio_raw_inputs = self.conv_feature_extractor(audio_raw_inputs)

        audio_raw_inputs = audio_raw_inputs.permute(0, 3, 1, 2) # [batch_size, audio_seq_len, channels, n_mels]
        audio_raw_inputs = audio_raw_inputs.reshape(N, T, -1) # [batch_size, audio_seq_len, channels * hidden_size]

        audio_raw_inputs = self.conv_projection(audio_raw_inputs)

        audio_raw_inputs = audio_raw_inputs + self.pos_encoding[:, :T, :]

        audio_raw_inputs = self.dropout(audio_raw_inputs)

        audio_raw_inputs = self.prelude(
            audio_raw_inputs,
            attention_mask=torch.ones((N, T), device=audio_raw_inputs.device),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return audio_raw_inputs

class ImprovedAudioFeatureExtractor(nn.Module):
    """
    Audio encoder with multi-scale processing and optional temporal subsampling.
    Inspired by Conformer and AST architectures.
    """
    
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__()
        self.config = config
        
        # Temporal subsampling factor (e.g., 4 = 78 tokens for 10s instead of 312)
        self.time_subsample = getattr(config, 'audio_encoder_time_subsample', 1)
        
        # --- Patch embedding (AST-style) instead of sequential conv ---
        # Treats mel spectrogram more like ViT treats images
        self.patch_size = (16, 4)  # (freq_bins, time_frames)
        patch_dim = self.patch_size[0] * self.patch_size[1]
        
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, config.hidden_size, 
                      kernel_size=self.patch_size, 
                      stride=(self.patch_size[0], self.time_subsample)),
            nn.GELU(),
        )
        
        # --- Frequency band processing (multi-scale in frequency) ---
        # Process different frequency ranges with specialized convs
        self.freq_bands = nn.ModuleList([
            # Low frequencies (0-32 mels) - larger temporal context
            nn.Conv2d(1, config.hidden_size // 4, kernel_size=(32, 7), 
                      stride=(32, self.time_subsample), padding=(0, 3)),
            # Mid frequencies (32-80 mels) - medium context  
            nn.Conv2d(1, config.hidden_size // 4, kernel_size=(48, 5),
                      stride=(48, self.time_subsample), padding=(0, 2)),
            # High frequencies (80-128 mels) - fine temporal detail
            nn.Conv2d(1, config.hidden_size // 2, kernel_size=(48, 3),
                      stride=(48, self.time_subsample), padding=(0, 1)),
        ])
        
        # Combine frequency bands
        self.freq_combine = nn.Linear(config.hidden_size, config.hidden_size)
        
        # --- Conformer-style blocks ---
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(config.hidden_size, config.audio_encoder_dropout)
            for _ in range(getattr(config, 'audio_encoder_n_conformer_blocks', 2))
        ])
        
        # --- Positional encoding ---
        # Sinusoidal for better generalization to different lengths
        max_len = config.audio_max_frames // self.time_subsample
        self.register_buffer('pos_encoding', 
                             self._create_sinusoidal_encoding(max_len, config.hidden_size))
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self._init_weights()
    
    def _create_sinusoidal_encoding(self, max_len, hidden_size):
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                            (-math.log(10000.0) / hidden_size))
        pe = torch.zeros(1, max_len, hidden_size)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, audio_raw_inputs: torch.Tensor, **kwargs):
        """
        Args:
            audio_raw_inputs: [B, 1, n_mels, T] mel spectrogram
        Returns:
            [B, T', hidden_size] where T' = T // time_subsample
        """
        B, C, F, T = audio_raw_inputs.shape
        
        # Multi-scale frequency processing
        # Split mel into frequency bands
        low_freq = audio_raw_inputs[:, :, :32, :]    # 0-32
        mid_freq = audio_raw_inputs[:, :, 32:80, :]  # 32-80
        high_freq = audio_raw_inputs[:, :, 80:, :]   # 80-128
        
        low_feat = self.freq_bands[0](low_freq)      # [B, C/4, 1, T']
        mid_feat = self.freq_bands[1](mid_freq)      # [B, C/4, 1, T']
        high_feat = self.freq_bands[2](high_freq)    # [B, C/2, 1, T']
        
        # Concat along channel dim, squeeze freq dim
        freq_feat = torch.cat([low_feat, mid_feat, high_feat], dim=1)  # [B, C, 1, T']
        freq_feat = freq_feat.squeeze(2).permute(0, 2, 1)  # [B, T', C]
        
        features = self.freq_combine(freq_feat)  # [B, T', hidden]
        
        # Add positional encoding
        T_out = features.shape[1]
        features = features + self.pos_encoding[:, :T_out, :]
        features = self.dropout(features)
        
        # Conformer blocks
        for block in self.conformer_blocks:
            features = block(features)
        
        return features


class ConformerBlock(nn.Module):
    """
    Conformer block: FFN → Self-Attention → Conv → FFN
    Captures both local and global context.
    """
    
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        
        # Feed-forward modules (half-step each)
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        
        # Multi-head self-attention
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout, batch_first=True)
        self.attn = megatransformer_attn.MegaTransformerSelfAttention()
        self.attn_dropout = nn.Dropout(dropout)
        
        # Depthwise separable convolution (local context)
        self.conv_norm = nn.LayerNorm(hidden_size)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=1),  # Pointwise expand
            nn.GLU(dim=1),  # Gated
            nn.Conv1d(hidden_size, hidden_size, kernel_size=31, 
                      padding=15, groups=hidden_size),  # Depthwise
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),  # Pointwise project
            nn.Dropout(dropout),
        )
        
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        
        self.final_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        # FFN (half-step)
        x = x + 0.5 * self.ffn1(x)
        
        # Self-attention
        attn_in = self.attn_norm(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        x = x + self.attn_dropout(attn_out)
        
        # Convolution (local context)
        conv_in = self.conv_norm(x)
        conv_in = conv_in.permute(0, 2, 1)  # [B, C, T]
        conv_out = self.conv(conv_in)
        conv_out = conv_out.permute(0, 2, 1)  # [B, T, C]
        x = x + conv_out
        
        # FFN (half-step)
        x = x + 0.5 * self.ffn2(x)
        
        return self.final_norm(x)
