from torch import nn
from typing import Optional, Union

import admin_torch
import embedding_mlp, millions_moe, phi3_mlp, per_lang_embedding, positionwise_fcn, multihead_attn, sum
import math
import torch
import transformer_utils

def init_weights(model: nn.Module,
                 d_model: int,
                 init_weights_from: str = 'glorot_uniform',
                 init_weights_gain: float = 1.0,
                 tie_embeddings=False):
    for p in model.parameters():
        # glorot initialization needs at least two dimensions on the tensor
        if p.dim() > 1:
            if init_weights_from in ['glorot_uniform', 'xavier_uniform']:
                nn.init.xavier_uniform_(p, gain=init_weights_gain)
            elif init_weights_from in ['glorot_normal', 'xavier_normal']:
                nn.init.xavier_normal_(p, gain=init_weights_gain)
            elif init_weights_from == 'kaiming_uniform':
                nn.init.kaiming_uniform_(p)
            elif init_weights_from == 'kaiming_normal':
                nn.init.kaiming_normal_(p)
            elif init_weights_from == 'orthogonal':
                nn.init.orthogonal_(p)
            else:
                raise Exception(f"Unknown weight initialization method: {init_weights_from}")

    # share weights between the embedding layers and the logit layer
    if isinstance(model, Transformer):
        encoder: Encoder = model.encoder
        decoder: Decoder = model.decoder
        if isinstance(encoder.embed_tokens, nn.Embedding) and isinstance(decoder.embed_tokens, nn.Embedding):
            encoder_embedding: nn.Embedding = encoder.embed_tokens
            decoder_embedding: nn.Embedding = decoder.embed_tokens
            nn.init.normal_(encoder_embedding.weight, mean=0., std=d_model ** -0.5)
            if tie_embeddings:
                decoder_embedding.weight = encoder_embedding.weight
                decoder.lm_head.weight = decoder_embedding.weight
        elif isinstance(encoder.embed_tokens, embedding_mlp.EmbeddingMLP) \
            and isinstance(decoder.embed_tokens, embedding_mlp.EmbeddingMLP) \
            and isinstance(decoder.lm_head, nn.Sequential):
            encoder_embedding: embedding_mlp.EmbeddingMLP = encoder.embed_tokens
            decoder_embedding: embedding_mlp.EmbeddingMLP = decoder.embed_tokens

            nn.init.normal_(encoder_embedding.embedding.weight, mean=0., std=d_model ** -0.5)

            if tie_embeddings:
                decoder_embedding.embedding.weight = encoder_embedding.embedding.weight
                decoder.lm_head[-1].weight = decoder.embed_tokens.embedding.weight
    print("Model initialized.")

class EncoderLayer(nn.Module):
    def __init__(self, device, model_config, encoder_config):
        super(EncoderLayer, self).__init__()

        self_attn_config = encoder_config.self_attn_config
        ffn_config = encoder_config.ffn_config

        self.use_admin = model_config.use_admin
        self.norm = model_config.norm
        self.norm_eps = model_config.norm_eps

        self.n_layers = encoder_config.n_layers
        self.ffn_type = ffn_config.ffn_type
        self.moe_replace = ffn_config.moe_replace

        self.self_attn: multihead_attn.MultiHeadAttention = multihead_attn.MultiHeadAttention(device, model_config, self_attn_config, self_attn=True, in_decoder=False)

        self.self_attn_residual: nn.Module
        self.ffn_residual: nn.Module
        if self.use_admin:
            self.self_attn_residual = admin_torch.as_module(self.n_layers)
            self.ffn_residual = admin_torch.as_module(self.n_layers)
        else:
            self.self_attn_residual = sum.Sum()
            self.ffn_residual = sum.Sum()

        moe: Optional[nn.Module] = None
        self.ffn: nn.Module
        if self.ffn_type == 'millions':
            moe = millions_moe.MillionsMoE(model_config, ffn_config)
        elif self.ffn_type == "phi3":
            self.ffn = phi3_mlp.Phi3MLP(model_config, ffn_config)
        else:
            self.ffn = positionwise_fcn.PositionWiseFCNetwork(model_config, ffn_config)

        if moe is not None and bool(self.moe_replace):
            self.ffn = moe
        elif moe is not None:
            self.ffn = nn.Sequential(moe, self.ffn)

    def forward(self, encoder_sequences, key_padding_mask) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        self_attn, _ = self.self_attn(encoder_sequences, encoder_sequences, encoder_sequences, key_padding_mask)

        encoder_sequences = self.self_attn_residual(encoder_sequences, self_attn)
        fcn, gating_variances = self.ffn(encoder_sequences)
        encoder_sequences = self.ffn_residual(encoder_sequences, fcn)
            
        return encoder_sequences, gating_variances

class Encoder(nn.Module):
    def __init__(self, device, model_config, encoder_config):
        super(Encoder, self).__init__()

        self.device = device

        self.model_config = model_config
        self.encoder_config = encoder_config

        self.maxlen = model_config.maxlen
        self.d_model = model_config.d_model
        self.dropout = model_config.dropout
        self.positional_encoding_type = model_config.positional_encoding_type
        self.positional_encoding_dim = model_config.positional_encoding_dim
        self.learnable_positional_encoding = model_config.learnable_positional_encoding
        self.norm_eps = model_config.norm_eps
        self.norm = model_config.norm

        self.vocab_size = encoder_config.vocab_size
        self.n_layers = encoder_config.n_layers
        self.embedding_compression_dim = encoder_config.embedding_compression_dim
        self.per_lang_embedding_layers = encoder_config.per_lang_embedding_layers
        self.embedding_activation = encoder_config.embedding_activation
        self.param_sharing_type = encoder_config.param_sharing_type
        self.m_independent_layers = encoder_config.m_independent_layers

        self.embed_tokens: Union[embedding_mlp.EmbeddingMLP, per_lang_embedding.PerLangEmbedding, nn.Embedding]
        if self.embedding_compression_dim != 0:
            self.embed_tokens = embedding_mlp.EmbeddingMLP(self.vocab_size, self.embedding_compression_dim, self.d_model, transformer_utils.get_activation_function(self.embedding_activation) if self.embedding_activation != 'none' else nn.Identity)
        elif self.per_lang_embedding_layers > 1:
            self.embed_tokens = per_lang_embedding.PerLangEmbedding(self.vocab_size, self.d_model, self.per_lang_embedding_layers, self.embedding_activation)
        else:
            self.embed_tokens = nn.Embedding(self.vocab_size, self.d_model)

        self.encoder_dropout = nn.Dropout(self.dropout)
        self.post_encoder_norm = self.norm(self.d_model, self.norm_eps)
        self.encoder_layers = self.make_encoder_layers(self.n_layers, self.param_sharing_type, self.m_independent_layers)

        if self.positional_encoding_type != 'rotary':
            self.tensor_positional_encoding = nn.Parameter(transformer_utils.get_tensor_positional_encoding(device, self.d_model, self.positional_encoding_dim, self.learnable_positional_encoding, self.maxlen))

    def make_encoder_layers(self, n_layers, param_sharing_type, m_independent_layers) -> nn.ModuleList[EncoderLayer]:
        def new_encoder_layer():
            return EncoderLayer(self.device, self.model_config, self.encoder_config)

        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(new_encoder_layer())
            elif param_sharing_type == 'sequence':
                if (i - 1) % math.floor(n_layers / m_independent_layers) == 0:
                    layers.append(new_encoder_layer())
                else:
                    layers.append(layers[i - 1])
            elif param_sharing_type == 'cycle':
                if i <= m_independent_layers:
                    layers.append(new_encoder_layer())
                else:
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(layers[res_idx])
            elif param_sharing_type == 'cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_encoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(layers[res_idx])
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(layers[res_idx])
            elif param_sharing_type == 'ffn-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_encoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    new_layer = new_encoder_layer()
                    new_layer.ffn = layers[res_idx].fcn
                    if hasattr(layers[res_idx], 'fcn_residual'):
                        new_layer.ffn_residual = layers[res_idx].fcn_residual
                    layers.append(new_layer)
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    new_layer = new_encoder_layer()
                    new_layer.ffn = layers[res_idx].fcn
                    if hasattr(layers[res_idx], 'fcn_residual'):
                        new_layer.ffn_residual = layers[res_idx].fcn_residual
                    layers.append(new_layer)
            elif param_sharing_type == 'heads-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_encoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    new_layer = new_encoder_layer()
                    new_layer.self_attn = layers[res_idx].self_attn
                    if hasattr(layers[res_idx], 'self_attn_residual'):
                        new_layer.self_attn_residual = layers[res_idx].self_attn_residual
                    layers.append(new_layer)
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    new_layer = new_encoder_layer()
                    new_layer.self_attn = layers[res_idx].self_attn
                    if hasattr(layers[res_idx], 'self_attn_residual'):
                        new_layer.self_attn_residual = layers[res_idx].self_attn_residual
                    layers.append(new_layer)
            elif param_sharing_type == 'all':
                layers.append(layers[0])
            else:
                layers.append(new_encoder_layer())
        return nn.ModuleList(layers)

    def apply_embedding_transformation(self, encoder_sequences : torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(encoder_sequences) * math.sqrt(self.d_model)

    def apply_positional_embedding(self, encoder_sequences: torch.Tensor) -> torch.Tensor:
        # 1D buffer/sinusoidal encoding is applied here. 2D buffer/sinusoidal encoding and rotary encoding are applied in the MultiHeadAttention layer(s)
        if hasattr(self, 'tensor_positional_encoding'):
            return encoder_sequences + self.tensor_positional_encoding[:, :encoder_sequences.size(1), :]
        return encoder_sequences
    
    def apply_encoder_layer(self, encoder_sequences: torch.Tensor, key_padding_mask: torch.Tensor, encoder_layer: nn.Module) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return encoder_layer(encoder_sequences, key_padding_mask)

    def forward(self, encoder_sequences: torch.Tensor, key_padding_mask: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        assert torch.all(encoder_sequences < self.vocab_size), f"Encoder input is out of bounds: {torch.max(encoder_sequences)} >= {self.vocab_size}"

        encoder_sequences = encoder_sequences.to(self.device)
        key_padding_mask = key_padding_mask.to(self.device)

        encoder_sequences = self.apply_embedding_transformation(encoder_sequences)
        encoder_sequences = self.apply_positional_embedding(encoder_sequences)
        encoder_sequences = self.encoder_dropout(encoder_sequences)

        gating_variances: list[torch.Tensor] = []
        for encoder_layer in self.encoder_layers:
            encoder_sequences, gating_variance = self.apply_encoder_layer(encoder_sequences, key_padding_mask, encoder_layer)
            if gating_variance is not None:
                gating_variances.append(gating_variance)

        # post-LN
        encoder_sequences = self.post_encoder_norm(encoder_sequences)

        return encoder_sequences, gating_variances

class DecoderLayer(nn.Module):
    def __init__(self, device, model_config, decoder_config):
        super(DecoderLayer, self).__init__()

        self_attn_config = decoder_config.self_attn_config
        cross_attn_config = decoder_config.cross_attn_config
        use_cross_attn = cross_attn_config is not None
        ffn_config = decoder_config.ffn_config
        
        self.use_admin = model_config.use_admin

        self.n_layers = decoder_config.n_layers

        self.ffn_type = ffn_config.ffn_type
        self.moe_replace = ffn_config.moe_replace

        self.self_attn = multihead_attn.MultiHeadAttention(device, model_config, self_attn_config, self_attn=True, in_decoder=True)

        self.cross_attn: Optional[multihead_attn.MultiHeadAttention]
        if use_cross_attn:
            self.cross_attn = multihead_attn.MultiHeadAttention(device, model_config, cross_attn_config, self_attn=False, in_decoder=True)
        else:
            self.cross_attn = None

        if self.use_admin:
            self.self_attn_residual = admin_torch.as_module(self.n_layers)
            self.cross_attn_residual = admin_torch.as_module(self.n_layers)
            self.fcn_residual = admin_torch.as_module(self.n_layers)
        else:
            self.self_attn_residual = sum.Sum()
            self.cross_attn_residual = sum.Sum()
            self.fcn_residual = sum.Sum()

        self.ffn: nn.Module
        moe: Optional[nn.Module] = None
        if self.ffn_type == 'millions':
            moe = millions_moe.MillionsMoE(model_config, ffn_config)
        elif self.ffn_type == "phi3":
            self.ffn = phi3_mlp.Phi3MLP(model_config, ffn_config)
        else:
            self.ffn = positionwise_fcn.PositionWiseFCNetwork(model_config, ffn_config)

        if moe is not None and self.moe_replace:
            self.ffn = moe
        elif moe is not None:
            self.ffn = nn.Sequential(moe, self.ffn)

    def forward(self, decoder_sequences: torch.Tensor, encoder_sequences: torch.Tensor, src_key_padding_mask: torch.Tensor, tgt_key_padding_mask: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        self_attn, _ = self.self_attn(decoder_sequences, decoder_sequences, decoder_sequences, tgt_key_padding_mask)
        decoder_sequences = self.self_attn_residual(decoder_sequences, self_attn)

        if self.cross_attn is not None and encoder_sequences is not None and src_key_padding_mask is not None:
            cross_attn, _ = self.cross_attn(decoder_sequences, encoder_sequences, encoder_sequences, src_key_padding_mask)
            decoder_sequences = self.cross_attn_residual(decoder_sequences, cross_attn)

        fcn, gating_variances = self.ffn(decoder_sequences)
        decoder_sequences = self.fcn_residual(decoder_sequences, fcn)

        return decoder_sequences, gating_variances

class Decoder(nn.Module):
    def __init__(self, device, model_config, decoder_config):
        super(Decoder, self).__init__()

        self.device = device

        self.model_config = model_config
        self.decoder_config = decoder_config

        self.maxlen = model_config.maxlen
        self.d_model = model_config.d_model
        self.dropout = model_config.dropout
        self.positional_encoding_type = model_config.positional_encoding_type
        self.positional_encoding_dim = model_config.positional_encoding_dim
        self.learnable_positional_encoding = model_config.learnable_positional_encoding
        self.norm_eps = model_config.norm_eps
        self.norm = model_config.norm

        self.vocab_size = decoder_config.vocab_size
        self.n_layers = decoder_config.n_layers
        self.embedding_compression_dim = decoder_config.embedding_compression_dim
        self.per_lang_embedding_layers = decoder_config.per_lang_embedding_layers
        self.embedding_activation = decoder_config.embedding_activation
        self.param_sharing_type = decoder_config.param_sharing_type
        self.m_independent_layers = decoder_config.m_independent_layers

        self.embed_tokens: Union[embedding_mlp.EmbeddingMLP, per_lang_embedding.PerLangEmbedding, nn.Embedding]
        if self.embedding_compression_dim != 0:
            self.embed_tokens = embedding_mlp.EmbeddingMLP(self.vocab_size, self.embedding_compression_dim, self.d_model, transformer_utils.get_activation_function(self.embedding_activation) if self.embedding_activation != 'none' else nn.Identity)
        elif self.per_lang_embedding_layers > 1:
            self.embed_tokens = per_lang_embedding.PerLangEmbedding(self.vocab_size, self.d_model, self.per_lang_embedding_layers, self.embedding_activation)
        else:
            self.embed_tokens = nn.Embedding(self.vocab_size, self.d_model)

        self.decoder_dropout = nn.Dropout(self.dropout)
        self.post_decoder_norm = self.norm(self.d_model, self.norm_eps)
        self.decoder_layers = self.make_decoder_layers(self.n_layers, self.param_sharing_type, self.m_independent_layers)

        self.lm_head: nn.Module
        if self.embedding_compression_dim != 0:
            self.lm_head = nn.Sequential(
                nn.Linear(self.d_model, self.embedding_compression_dim),
                transformer_utils.create_activation_function(self.embedding_compression_dim, self.embedding_activation) if self.embedding_activation != 'none' else nn.Identity(),
                nn.Linear(self.embedding_compression_dim, self.vocab_size)
            )
        else:
            self.lm_head = nn.Linear(self.d_model, self.vocab_size)

        if self.positional_encoding_type != 'rotary':
            self.tensor_positional_encoding = nn.Parameter(transformer_utils.get_tensor_positional_encoding(self.decoder_device, self.d_model, self.positional_encoding_dim, self.learnable_positional_encoding, self.maxlen))

    def make_decoder_layers(self, n_layers, param_sharing_type, m_independent_layers) -> nn.ModuleList[DecoderLayer]:
        def new_decoder_layer():
            return DecoderLayer(self.device, self.model_config, self.decoder_config)
        
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(new_decoder_layer())
            elif param_sharing_type == 'sequence':
                if (i - 1) % math.floor(n_layers / m_independent_layers) == 0:
                    layers.append(new_decoder_layer())
                else:
                    layers.append(layers[i - 1])
            elif param_sharing_type == 'cycle':
                if i <= m_independent_layers:
                    layers.append(new_decoder_layer())
                else:
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(layers[res_idx])
            elif param_sharing_type == 'cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_decoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(layers[res_idx])
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(layers[res_idx])
            elif param_sharing_type == 'ffn-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_decoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    new_layer = new_decoder_layer()
                    new_layer.ffn = layers[res_idx].fcn
                    if hasattr(layers[res_idx], 'fcn_residual'):
                        new_layer.fcn_residual = layers[res_idx].fcn_residual
                    layers.append(new_layer)
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    new_layer = new_decoder_layer()
                    new_layer.ffn = layers[res_idx].fcn
                    if hasattr(layers[res_idx], 'fcn_residual'):
                        new_layer.fcn_residual = layers[res_idx].fcn_residual
                    layers.append(new_layer)
            elif param_sharing_type == 'heads-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_decoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    new_layer = new_decoder_layer()
                    new_layer.self_attn = layers[res_idx].self_attn
                    new_layer.cross_attn = layers[res_idx].cross_attn
                    if hasattr(layers[res_idx], 'self_attn_residual'):
                        new_layer.self_attn_residual = layers[res_idx].self_attn_residual
                    if hasattr(layers[res_idx], 'cross_attn_residual'):
                        new_layer.cross_attn_residual = layers[res_idx].cross_attn_residual
                    layers.append(new_layer)
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    new_layer = new_decoder_layer()
                    new_layer.self_attn = layers[res_idx].self_attn
                    new_layer.cross_attn = layers[res_idx].cross_attn
                    if hasattr(layers[res_idx], 'self_attn_residual'):
                        new_layer.self_attn_residual = layers[res_idx].self_attn_residual
                    if hasattr(layers[res_idx], 'cross_attn_residual'):
                        new_layer.cross_attn_residual = layers[res_idx].cross_attn_residual
                    layers.append(new_layer)
            elif param_sharing_type == 'all':
                layers.append(layers[0])
            else:
                layers.append(new_decoder_layer())
        return nn.ModuleList(layers)

    def apply_embedding_transformation(self, decoder_sequences: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(decoder_sequences) * math.sqrt(self.d_model)
    
    def apply_positional_embedding(self, decoder_sequences: torch.Tensor) -> torch.Tensor:
        # 1D buffer/sinusoidal encoding is applied here. 2D buffer/sinusoidal encoding and rotary encoding are applied in the MultiHeadAttention layer(s)
        if hasattr(self, 'tensor_positional_encoding'):
            return decoder_sequences + self.tensor_positional_encoding[:, :decoder_sequences.size(1), :]
        return decoder_sequences
    
    def apply_decoder_layer(self, decoder_sequences: torch.Tensor, encoder_sequences: torch.Tensor, src_key_padding_mask: torch.Tensor, tgt_key_padding_mask: torch.Tensor, decoder_layer: nn.Module) -> torch.Tensor:
        return decoder_layer(decoder_sequences, encoder_sequences, src_key_padding_mask, tgt_key_padding_mask)

    def forward(self, decoder_sequences: torch.Tensor, encoder_sequences: torch.Tensor, src_key_padding_mask: torch.Tensor, tgt_key_padding_mask: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        assert torch.all(encoder_sequences < self.vocab_size), f"Encoder input is out of bounds: {torch.max(encoder_sequences)} >= {self.vocab_size}"
        assert torch.all(decoder_sequences < self.vocab_size), f"Decoder input is out of bounds: {torch.max(decoder_sequences)} >= {self.vocab_size}"

        decoder_sequences = decoder_sequences.to(self.device)
        if encoder_sequences is not None:
            encoder_sequences = encoder_sequences.to(self.device)

        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.to(self.device)
        elif encoder_sequences is not None:
            src_key_padding_mask = encoder_sequences == 0

        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.to(self.device)
        else:
            tgt_key_padding_mask = decoder_sequences == 0

        if self.embed_tokens is not None:
            decoder_sequences = self.apply_embedding_transformation(decoder_sequences)
        decoder_sequences = self.apply_positional_embedding(decoder_sequences)
        decoder_sequences = self.decoder_dropout(decoder_sequences)

        gating_variances = []
        for decoder_layer in self.decoder_layers:
            decoder_sequences, gating_variance = self.apply_decoder_layer(decoder_sequences, encoder_sequences, src_key_padding_mask, tgt_key_padding_mask, decoder_layer)
            if gating_variance is not None:
                gating_variances.append(gating_variance)

        decoder_sequences = self.post_decoder_norm(decoder_sequences)
        decoder_sequences = self.lm_head(decoder_sequences)

        return decoder_sequences, gating_variances

class Transformer(nn.Module):
    def __init__(self, model_config):
        super(Transformer, self).__init__()

        self.encoder_config = model_config.encoder_config
        self.decoder_config = model_config.decoder_config

        self.encoder: Encoder = Encoder(self.encoder_config.device, model_config, self.encoder_config)
        self.decoder: Decoder = Decoder(self.decoder_config.device, model_config, self.decoder_config)

        init_weights(self, model_config.tie_embeddings)

    def forward(self, encoder_sequences: torch.Tensor, decoder_sequences: torch.Tensor, src_key_padding_mask: torch.Tensor, tgt_key_padding_mask: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        encoder_sequences = encoder_sequences.to(self.encoder_config.device)
        decoder_sequences = decoder_sequences.to(self.decoder_config.device)
        src_key_padding_mask = src_key_padding_mask.to(self.encoder_config.device)
        tgt_key_padding_mask = tgt_key_padding_mask.to(self.decoder_config.device)
        self.encoder.embed_tokens = self.encoder.embed_tokens.to(self.encoder_config.device)

        encoder_sequences, encoder_gating_variances = self.encoder(encoder_sequences, src_key_padding_mask)

        encoder_sequences = encoder_sequences.to(self.decoder_config.device)
        src_key_padding_mask = src_key_padding_mask.to(self.decoder_config.device)
        self.decoder.embed_tokens = self.decoder.embed_tokens.to(self.decoder_config.device)
        self.decoder.lm_head = self.decoder.lm_head.to(self.decoder_config.device)
        
        decoder_sequences, decoder_gating_variances = self.decoder(decoder_sequences, encoder_sequences, src_key_padding_mask, tgt_key_padding_mask)

        return decoder_sequences, encoder_gating_variances, decoder_gating_variances
