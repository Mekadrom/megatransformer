from torch import nn
from typing import Optional, Union, Callable

from . import config, embedding_mlp, infinite_multihead_attn, millions_moe, phi3_mlp, per_lang_embedding, positionwise_fcn, multihead_attn, sum, mult, transformer_utils, criteria, grouped_query_attn, huginn_criteria

import admin_torch
import copy
import math
import random
import torch
import warnings

class MegaTransformerOutput:
    def __init__(self,
                logits: torch.Tensor,
                loss: torch.Tensor,
                prediction_loss: torch.Tensor,
                moe_loss: torch.Tensor,
                decoder_gating_variances: list[torch.Tensor],
                encoder_gating_variances: Optional[list[torch.Tensor]]=None,
                block_stats: Optional[tuple]=None):
        self.logits = logits
        self.loss = loss
        self.prediction_loss = prediction_loss
        self.moe_loss = moe_loss
        self.decoder_gating_variances = decoder_gating_variances
        self.encoder_gating_variances = encoder_gating_variances
        self.block_stats = block_stats

class EncoderLayer(nn.Module):
    def __init__(self, device, model_config):
        super(EncoderLayer, self).__init__()

        encoder_config = model_config.encoder_config

        self_attn_config = encoder_config.self_attn_config
        ffn_config = model_config.ffn_config

        self.use_admin = model_config.use_admin
        norm = model_config.norm

        self.pre_self_attn_norm = norm(model_config.d_model, model_config.norm_eps) if encoder_config.pre_self_attn_norm else None
        self.post_self_attn_norm = norm(model_config.d_model, model_config.norm_eps) if encoder_config.post_self_attn_norm else None
        self.pre_ffn_norm = norm(model_config.d_model, model_config.norm_eps) if encoder_config.pre_ffn_norm else None
        self.post_ffn_norm = norm(model_config.d_model, model_config.norm_eps) if encoder_config.post_ffn_norm else None

        self.n_layers = encoder_config.n_layers
        self.ffn_type = ffn_config.ffn_type
        self.moe_replace = ffn_config.moe_replace

        if self_attn_config.attn_impl == 'gqa':
            self.self_attn: nn.Module = grouped_query_attn.GroupedQueryMultiHeadAttention(device, model_config, self_attn_config, self_attn=True, in_decoder=False)
        elif self_attn_config.attn_impl == 'infinite':
            self.self_attn: nn.Module = infinite_multihead_attn.InfiniteMultiHeadAttention(device, model_config, self_attn_config, self_attn=True, in_decoder=False)
        else:
            self.self_attn: nn.Module = multihead_attn.MultiHeadAttention(device, model_config, self_attn_config, self_attn=True, in_decoder=False)

        self.self_attn_residual: nn.Module
        self.ffn_residual: nn.Module
        if self.use_admin:
            if self_attn_config.attn_impl == 'infinite':
                self.self_attn_residual = transformer_utils.ReturnNthParameterModule()
            else:
                self.self_attn_residual = admin_torch.as_module(self.n_layers)
            self.ffn_residual = admin_torch.as_module(self.n_layers)
        else:
            if self_attn_config.attn_impl == 'infinite':
                self.self_attn_residual = transformer_utils.ReturnNthParameterModule()
            else:
                self.self_attn_residual = sum.Sum()
            self.ffn_residual = sum.Sum()

        moe: Optional[nn.Module] = None
        self.ffn: nn.Module
        if self.ffn_type == 'millions':
            moe = millions_moe.MillionsMoE(model_config)
        elif self.ffn_type == "phi3":
            self.ffn = phi3_mlp.Phi3MLP(model_config)
        else:
            self.ffn = positionwise_fcn.PositionWiseFCNetwork(model_config)

        if moe is not None and bool(self.moe_replace):
            self.ffn = moe
        elif moe is not None:
            self.ffn = nn.Sequential(moe, self.ffn)

    def forward(self, encoder_sequences, key_padding_mask) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # pre-LN
        if self.pre_self_attn_norm is not None:
            encoder_sequences = self.pre_self_attn_norm(encoder_sequences)

        self_attn, _ = self.self_attn(encoder_sequences, encoder_sequences, encoder_sequences, key_padding_mask)
        encoder_sequences = self.self_attn_residual(encoder_sequences, self_attn)

        if self.post_self_attn_norm is not None:
            encoder_sequences = self.post_self_attn_norm(encoder_sequences)

        if self.pre_ffn_norm is not None:
            encoder_sequences = self.pre_ffn_norm(encoder_sequences)

        ffn, gating_variances = self.ffn(encoder_sequences)
        encoder_sequences = self.ffn_residual(encoder_sequences, ffn)

        # post-LN
        if self.post_ffn_norm is not None:
            encoder_sequences = self.post_ffn_norm(encoder_sequences)
            
        return encoder_sequences, gating_variances

class Encoder(nn.Module):
    def __init__(self, device, model_config):
        super(Encoder, self).__init__()

        self.device = device

        self.model_config = model_config
        self.encoder_config = model_config.encoder_config

        self.maxlen = model_config.maxlen
        self.d_model = model_config.d_model
        self.dropout = model_config.dropout
        self.positional_encoding_type = model_config.positional_encoding_type
        self.positional_encoding_dim = model_config.positional_encoding_dim
        self.learnable_positional_encoding = model_config.learnable_positional_encoding

        self.vocab_size = self.encoder_config.vocab_size
        self.n_layers = self.encoder_config.n_layers
        self.embedding_compression_dim = self.encoder_config.embedding_compression_dim
        self.per_lang_embedding_layers = self.encoder_config.per_lang_embedding_layers
        self.embedding_activation = self.encoder_config.embedding_activation
        self.param_sharing_type = self.encoder_config.param_sharing_type
        self.m_independent_layers = self.encoder_config.m_independent_layers
        self.embed_scale = self.encoder_config.embed_scale

        self.embed_tokens: Union[embedding_mlp.EmbeddingMLP, per_lang_embedding.PerLangEmbedding, nn.Embedding]
        if self.embedding_compression_dim != 0:
            self.embed_tokens = embedding_mlp.EmbeddingMLP(self.vocab_size, self.embedding_compression_dim, self.d_model, transformer_utils.get_activation_function(self.embedding_activation) if self.embedding_activation != 'none' else nn.Identity)
        elif self.per_lang_embedding_layers > 1:
            self.embed_tokens = per_lang_embedding.PerLangEmbedding(self.vocab_size, self.d_model, self.per_lang_embedding_layers, self.embedding_activation)
        else:
            self.embed_tokens = nn.Embedding(self.vocab_size, self.d_model)

        self.encoder_dropout = nn.Dropout(self.dropout)
        self.encoder_layers = self.make_encoder_layers(self.n_layers, self.param_sharing_type, self.m_independent_layers)

        if self.positional_encoding_type == 'sinusoidal':
            self.tensor_positional_encoding = nn.Parameter(transformer_utils.get_tensor_positional_encoding(device, self.d_model, self.positional_encoding_dim, self.learnable_positional_encoding, self.maxlen))

    def make_encoder_layers(self, n_layers, param_sharing_type, m_independent_layers) -> nn.ModuleList:
        def new_encoder_layer():
            return EncoderLayer(self.device, self.model_config)

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
                    new_layer.ffn_residual = layers[res_idx].fcn_residual
                    layers.append(new_layer)
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    new_layer = new_encoder_layer()
                    new_layer.ffn = layers[res_idx].fcn
                    new_layer.ffn_residual = layers[res_idx].fcn_residual
                    layers.append(new_layer)
            elif param_sharing_type == 'heads-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_encoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    new_layer = new_encoder_layer()
                    new_layer.self_attn = layers[res_idx].self_attn
                    new_layer.self_attn_residual = layers[res_idx].self_attn_residual
                    layers.append(new_layer)
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    new_layer = new_encoder_layer()
                    new_layer.self_attn = layers[res_idx].self_attn
                    new_layer.self_attn_residual = layers[res_idx].self_attn_residual
                    layers.append(new_layer)
            elif param_sharing_type == 'all':
                layers.append(layers[0])
            else:
                layers.append(new_encoder_layer())
        return nn.ModuleList(layers)

    def apply_embedding_transformation(self, encoder_sequences : torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(encoder_sequences) * math.sqrt(self.d_model) * self.embed_scale

    def apply_positional_embedding(self, encoder_sequences: torch.Tensor) -> torch.Tensor:
        # 1D buffer/sinusoidal encoding is applied here. 2D buffer/sinusoidal encoding and rotary encoding are applied in the MultiHeadAttention layer(s)
        if hasattr(self, 'tensor_positional_encoding'):
            return encoder_sequences + self.tensor_positional_encoding[:, :encoder_sequences.size(1), :]
        return encoder_sequences
    
    def apply_encoder_layer(self, encoder_sequences: torch.Tensor, key_padding_mask: torch.Tensor, encoder_layer: nn.Module) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return encoder_layer(encoder_sequences, key_padding_mask)

    def forward(self, input_ids: torch.Tensor, key_padding_mask: torch.Tensor, inputs_embeds: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if inputs_embeds is None:
            assert torch.all(input_ids < self.vocab_size), f"Encoder input is out of bounds: {torch.max(input_ids)} >= {self.vocab_size}"
            input_ids = input_ids.to(self.device)
            if self.embed_tokens is not None:
                encoder_sequences = self.apply_embedding_transformation(input_ids)
            else:
                encoder_sequences = input_ids
        else:
            inputs_embeds = inputs_embeds.to(self.device)
            encoder_sequences = inputs_embeds

        encoder_sequences = self.apply_positional_embedding(encoder_sequences)
        encoder_sequences = self.encoder_dropout(encoder_sequences)

        key_padding_mask = key_padding_mask.to(self.device)

        gating_variances: list[torch.Tensor] = []
        for encoder_layer in self.encoder_layers:
            encoder_sequences, gating_variance = self.apply_encoder_layer(encoder_sequences, key_padding_mask, encoder_layer)
            if gating_variance is not None:
                gating_variances.append(gating_variance)

        return encoder_sequences, gating_variances

class DecoderLayer(nn.Module):
    def __init__(self, device, model_config):
        super(DecoderLayer, self).__init__()

        decoder_config = model_config.decoder_config

        self_attn_config = decoder_config.self_attn_config
        cross_attn_config = decoder_config.cross_attn_config
        use_cross_attn = cross_attn_config is not None
        ffn_config = model_config.ffn_config
        
        norm = model_config.norm

        self.pre_self_attn_norm = norm(model_config.d_model, model_config.norm_eps) if decoder_config.pre_self_attn_norm else None
        self.post_self_attn_norm = norm(model_config.d_model, model_config.norm_eps) if decoder_config.post_self_attn_norm else None
        self.pre_cross_attn_norm = norm(model_config.d_model, model_config.norm_eps) if decoder_config.pre_cross_attn_norm else None
        self.post_cross_attn_norm = norm(model_config.d_model, model_config.norm_eps) if decoder_config.post_cross_attn_norm else None
        self.pre_ffn_norm = norm(model_config.d_model, model_config.norm_eps) if decoder_config.pre_ffn_norm else None
        self.post_ffn_norm = norm(model_config.d_model, model_config.norm_eps) if decoder_config.post_ffn_norm else None
        
        self.use_admin = model_config.use_admin

        self.n_layers = decoder_config.n_layers

        self.ffn_type = ffn_config.ffn_type
        self.moe_replace = ffn_config.moe_replace

        if self_attn_config.attn_impl == 'gqa':
            self.self_attn: nn.Module = grouped_query_attn.GroupedQueryMultiHeadAttention(device, model_config, self_attn_config, self_attn=True, in_decoder=True)
        elif self_attn_config.attn_impl == 'infinite':
            self.self_attn: nn.Module = infinite_multihead_attn.InfiniteMultiHeadAttention(device, model_config, self_attn_config, self_attn=True, in_decoder=True)
        else:
            self.self_attn: nn.Module = multihead_attn.MultiHeadAttention(device, model_config, self_attn_config, self_attn=True, in_decoder=True)

        if use_cross_attn:
            if cross_attn_config.attn_impl == 'gqa':
                self.cross_attn: nn.Module = grouped_query_attn.GroupedQueryMultiHeadAttention(device, model_config, cross_attn_config, self_attn=False, in_decoder=True)
            elif cross_attn_config.attn_impl == 'infinite':
                self.cross_attn: nn.Module = infinite_multihead_attn.InfiniteMultiHeadAttention(device, model_config, cross_attn_config, self_attn=False, in_decoder=True)
            else:
                self.cross_attn: nn.Module = multihead_attn.MultiHeadAttention(device, model_config, cross_attn_config, self_attn=False, in_decoder=True)
        else:
            self.cross_attn = None

        if self.use_admin:
            if self_attn_config.attn_impl == 'infinite':
                self.self_attn_residual = transformer_utils.ReturnNthParameterModule()
            else:
                self.self_attn_residual = admin_torch.as_module(self.n_layers)

            if use_cross_attn and cross_attn_config.attn_impl == 'infinite':
                self.cross_attn_residual = transformer_utils.ReturnNthParameterModule()
            else:
                self.cross_attn_residual = admin_torch.as_module(self.n_layers)
            self.ffn_residual = admin_torch.as_module(self.n_layers)
        else:
            if self_attn_config.attn_impl == 'infinite':
                self.self_attn_residual = transformer_utils.ReturnNthParameterModule()
            else:
                self.self_attn_residual = sum.Sum()

            if use_cross_attn and cross_attn_config.attn_impl == 'infinite':
                self.cross_attn_residual = transformer_utils.ReturnNthParameterModule()
            else:
                self.cross_attn_residual = sum.Sum()
            self.ffn_residual = sum.Sum()

        self.ffn: nn.Module
        moe: Optional[nn.Module] = None
        if self.ffn_type == 'millions':
            moe = millions_moe.MillionsMoE(model_config)
        elif self.ffn_type == "phi3":
            self.ffn = phi3_mlp.Phi3MLP(model_config)
        else:
            self.ffn = positionwise_fcn.PositionWiseFCNetwork(model_config)

        if moe is not None and self.moe_replace:
            self.ffn = moe
        elif moe is not None:
            self.ffn = nn.Sequential(moe, self.ffn)

    def forward(self,
                decoder_sequence: torch.Tensor,
                encoder_sequence: Optional[torch.Tensor],
                decoder_attention_mask: Optional[torch.Tensor],
                encoder_attention_mask: Optional[torch.Tensor]=None,
                kv_cache: Optional[list[torch.Tensor]]=None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # pre-LN
        if self.pre_self_attn_norm is not None:
            decoder_sequence = self.pre_self_attn_norm(decoder_sequence)

        self_attn, _ = self.self_attn(decoder_sequence, decoder_sequence, decoder_sequence, decoder_attention_mask)
        decoder_sequence = self.self_attn_residual(decoder_sequence, self_attn)

        if self.post_self_attn_norm is not None:
            decoder_sequence = self.post_self_attn_norm(decoder_sequence)

        if self.cross_attn is not None and encoder_sequence is not None:
            if self.pre_cross_attn_norm is not None:
                decoder_sequence = self.pre_cross_attn_norm(decoder_sequence)

            cross_attn, _ = self.cross_attn(decoder_sequence, encoder_sequence, encoder_sequence, encoder_attention_mask, kv_cache=kv_cache)
            decoder_sequence = self.cross_attn_residual(decoder_sequence, cross_attn)

            if self.post_cross_attn_norm is not None:
                decoder_sequence = self.post_cross_attn_norm(decoder_sequence)

        if self.pre_ffn_norm is not None:
            decoder_sequence = self.pre_ffn_norm(decoder_sequence)

        ffn, gating_variances = self.ffn(decoder_sequence)
        decoder_sequence = self.ffn_residual(decoder_sequence, ffn)

        # post-LN
        if self.post_ffn_norm is not None:
            decoder_sequence = self.post_ffn_norm(decoder_sequence)

        return decoder_sequence, gating_variances

class Decoder(nn.Module):
    def __init__(self, device, model_config):
        super(Decoder, self).__init__()

        self.device = device

        self.model_config = model_config
        self.decoder_config = model_config.decoder_config

        self.maxlen = model_config.maxlen
        self.d_model = model_config.d_model
        self.dropout = model_config.dropout
        self.positional_encoding_type = model_config.positional_encoding_type
        self.positional_encoding_dim = model_config.positional_encoding_dim
        self.learnable_positional_encoding = model_config.learnable_positional_encoding

        self.vocab_size = self.decoder_config.vocab_size
        self.n_layers = self.decoder_config.n_layers
        self.embedding_compression_dim = self.decoder_config.embedding_compression_dim
        self.per_lang_embedding_layers = self.decoder_config.per_lang_embedding_layers
        self.embedding_activation = self.decoder_config.embedding_activation
        self.param_sharing_type = self.decoder_config.param_sharing_type
        self.m_independent_layers = self.decoder_config.m_independent_layers
        self.embed_scale = self.decoder_config.embed_scale

        self.embed_tokens: Union[embedding_mlp.EmbeddingMLP, per_lang_embedding.PerLangEmbedding, nn.Embedding]
        if self.embedding_compression_dim != 0:
            self.embed_tokens = embedding_mlp.EmbeddingMLP(self.vocab_size, self.embedding_compression_dim, self.d_model, transformer_utils.get_activation_function(self.embedding_activation) if self.embedding_activation != 'none' else nn.Identity)
        elif self.per_lang_embedding_layers > 1:
            self.embed_tokens = per_lang_embedding.PerLangEmbedding(self.vocab_size, self.d_model, self.per_lang_embedding_layers, self.embedding_activation)
        else:
            self.embed_tokens = nn.Embedding(self.vocab_size, self.d_model)

        self.decoder_dropout = nn.Dropout(self.dropout)
        self.init_decoder_layers()

        self.lm_head: nn.Module
        if self.embedding_compression_dim != 0:
            self.lm_head = nn.Sequential(
                nn.Linear(self.d_model, self.embedding_compression_dim),
                transformer_utils.create_activation_function(self.embedding_compression_dim, self.embedding_activation) if self.embedding_activation != 'none' else nn.Identity(),
                nn.Linear(self.embedding_compression_dim, self.vocab_size)
            )
        else:
            self.lm_head = nn.Linear(self.d_model, self.vocab_size)

        if self.positional_encoding_type == 'sinusoidal':
            self.tensor_positional_encoding = nn.Parameter(transformer_utils.get_tensor_positional_encoding(self.device, self.d_model, self.positional_encoding_dim, self.learnable_positional_encoding, self.maxlen))

        self.main_criteria = criteria.LMLoss(ignore_index=model_config.padding_value, eps=model_config.label_smoothing)
        self.moe_criteria = criteria.DecoderOnlyMoELoss(self.decoder_config.moe_diversity_loss_coefficient)

    def init_decoder_layers(self):
        self.decoder_layers = self.make_decoder_layers(self.n_layers, self.param_sharing_type, self.m_independent_layers)

    def make_decoder_layers(self, n_layers, param_sharing_type, m_independent_layers) -> nn.ModuleList:
        def new_decoder_layer():
            return DecoderLayer(self.device, self.model_config)
        
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
                    new_layer.ffn_residual = layers[res_idx].fcn_residual
                    layers.append(new_layer)
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    new_layer = new_decoder_layer()
                    new_layer.ffn = layers[res_idx].fcn
                    new_layer.ffn_residual = layers[res_idx].fcn_residual
                    layers.append(new_layer)
            elif param_sharing_type == 'heads-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_decoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    new_layer = new_decoder_layer()
                    new_layer.self_attn = layers[res_idx].self_attn
                    new_layer.self_attn_residual = layers[res_idx].self_attn_residual
                    new_layer.cross_attn = layers[res_idx].cross_attn
                    new_layer.cross_attn_residual = layers[res_idx].cross_attn_residual
                    layers.append(new_layer)
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    new_layer = new_decoder_layer()
                    new_layer.self_attn = layers[res_idx].self_attn
                    new_layer.self_attn_residual = layers[res_idx].self_attn_residual
                    new_layer.cross_attn = layers[res_idx].cross_attn
                    new_layer.cross_attn_residual = layers[res_idx].cross_attn_residual
                    layers.append(new_layer)
            elif param_sharing_type == 'all':
                layers.append(layers[0])
            else:
                layers.append(new_decoder_layer())
        return nn.ModuleList(layers)

    def apply_embedding_transformation(self, decoder_sequences: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(decoder_sequences) * math.sqrt(self.d_model) * self.embed_scale
    
    def apply_positional_embedding(self, labels: torch.Tensor) -> torch.Tensor:
        # 1D buffer/sinusoidal encoding is applied here. 2D buffer/sinusoidal encoding and rotary encoding are applied in the MultiHeadAttention layer(s)
        if hasattr(self, 'tensor_positional_encoding'):
            return labels + self.tensor_positional_encoding[:, :labels.size(1), :]
        return labels
    
    def apply_decoder_layer(self,
                            decoder_sequences: torch.Tensor,
                            encoder_sequences: Optional[torch.Tensor],
                            decoder_attention_mask: Optional[torch.Tensor],
                            decoder_layer: nn.Module,
                            encoder_attention_mask: Optional[torch.Tensor]=None,
                            kv_cache: Optional[list[torch.Tensor]]=None) -> torch.Tensor:
        return decoder_layer(decoder_sequences, encoder_sequences, decoder_attention_mask, encoder_attention_mask=encoder_attention_mask, kv_cache=kv_cache)

    def decoder_block_forward(self,
                              decoder_sequences: torch.Tensor,
                              encoder_sequences: Optional[torch.Tensor],
                              decoder_attention_mask: Optional[torch.Tensor],
                              kv_caches: Optional[list[list[torch.Tensor]]]) -> tuple:
        gating_variances = []
        if kv_caches:
            for decoder_layer, kv_cache in zip(self.decoder_layers, kv_caches):
                decoder_sequences, gating_variance = self.apply_decoder_layer(decoder_sequences, encoder_sequences, decoder_attention_mask, decoder_layer, kv_cache=kv_cache)
                if gating_variance is not None:
                    gating_variances.append(gating_variance)
        else:
            for decoder_layer in self.decoder_layers:
                decoder_sequences, gating_variance = self.apply_decoder_layer(decoder_sequences, encoder_sequences, decoder_attention_mask, decoder_layer)
                if gating_variance is not None:
                    gating_variances.append(gating_variance)

        return decoder_sequences, gating_variances

    def forward(self,
                input_ids: torch.Tensor,
                labels: Optional[torch.Tensor]=None,
                encoder_sequences: Optional[torch.Tensor]=None,
                targets_embeds: Optional[torch.Tensor]=None,
                decoder_attention_mask: Optional[torch.Tensor]=None,
                return_dict: bool=False,
                kv_caches: Optional[list[list[torch.Tensor]]]=None) -> Union[tuple, MegaTransformerOutput]:
        if targets_embeds is None:
            assert torch.all(input_ids < self.vocab_size), f"Decoder input is out of bounds: {torch.max(input_ids)} >= {self.vocab_size}"

            input_ids = input_ids.to(self.device)
            if self.embed_tokens is not None:
                decoder_sequences = self.apply_embedding_transformation(input_ids)
            else:
                decoder_sequences = input_ids
        else:
            targets_embeds = targets_embeds.to(self.device)
            decoder_sequences = targets_embeds

        if encoder_sequences is not None:
            encoder_sequences = encoder_sequences.to(self.device)

        decoder_sequences = self.apply_positional_embedding(decoder_sequences)
        decoder_sequences = self.decoder_dropout(decoder_sequences)
        block_results = self.decoder_block_forward(decoder_sequences, encoder_sequences, decoder_attention_mask, kv_caches)

        decoder_sequences, gating_variances = block_results[0], block_results[1]

        logits = self.lm_head(decoder_sequences)

        if labels is not None:
            main_loss = self.main_criteria(logits, labels)
            moe_loss, gating_variances_tensor = self.moe_criteria(gating_variances)
        else:
            main_loss = None
            moe_loss = None
            gating_variances_tensor = None

        if return_dict:
            return MegaTransformerOutput(
                logits=logits,
                loss=main_loss + moe_loss,
                prediction_loss=main_loss,
                moe_loss=moe_loss,
                decoder_gating_variances=gating_variances_tensor,
                block_stats=block_results[2:] if len(block_results) > 2 else None
            )
        return logits, gating_variances

class HuginnDecoder(Decoder):
    def __init__(self, device, model_config):
        super(HuginnDecoder, self).__init__(device, model_config)

        self.n_prelude_layers = self.decoder_config.n_huginn_prelude_layers
        self.n_thinking_layers = self.decoder_config.n_huginn_thinking_layers
        self.n_coda_layers = self.decoder_config.n_huginn_coda_layers

        self.mean_thinking_steps = self.decoder_config.mean_huginn_thinking_steps
        self.mean_backprop_depth = self.decoder_config.mean_huginn_backprop_depth
        self.thought_initialization_method = self.decoder_config.huginn_thought_initialization_method
        self.adapter_method = self.decoder_config.huginn_adapter_method
        self.exit_criteria = self.decoder_config.huginn_exit_criteria
        self.exit_criteria_threshold = self.decoder_config.huginn_exit_criteria_threshold

        self.adapter: nn.Module
        if self.adapter_method == 'sum':
            self.adapter = sum.Sum() # todo: implement other adapter methods
        elif self.adapter_method == "gate":
            self.adapter = mult.Mult()
        elif self.adapter_method == "linear":
            self.adapter = nn.Linear(model_config.d_model * 2, model_config.d_model)
        else:
            raise ValueError(f"Invalid adapter method: {self.adapter_method}")
        
        if self.exit_criteria == 'kl_divergence':
            self.exit_criteria = huginn_criteria.KLDivergenceCriteria(self.exit_criteria_threshold) # todo: implement other exit criteria
        else:
            raise ValueError(f"Invalid exit criteria: {self.exit_criteria}")

    def init_decoder_layers(self):
        self.prelude_layers = self.make_decoder_layers(self.n_prelude_layers, self.param_sharing_type, self.m_independent_layers)
        self.thinking_block = self.make_decoder_layers(self.n_thinking_layers, self.param_sharing_type, self.m_independent_layers)
        self.coda_layers = self.make_decoder_layers(self.n_coda_layers, self.param_sharing_type, self.m_independent_layers)

    def initialize_thinking_state(self, input_embeds):
        """
        Taken directly from the original Huginn implementation: https://github.com/seal-rg/recurrent-pretraining/blob/main/recpre/model_dynamic.py
        """
        if self.thought_initialization_method == "none":
            return input_embeds
        if self.thought_initialization_method == "normal":
            x = torch.randn_like(input_embeds)
        elif self.thought_initialization_method == "embed":
            x = torch.randn_like(input_embeds).mul(1 / math.sqrt(input_embeds.shape[-1]))
        elif self.thought_initialization_method == "like-init":
            x = torch.randn_like(input_embeds)
            std = self.embed_tokens.weight.std().float().item()
            torch.nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3 * std, b=3 * std)
            if self.embed_scale != 1:
                x = x * self.embed_scale
        elif self.thought_initialization_method == "zero":
            x = torch.zeros_like(input_embeds)
        elif self.thought_initialization_method == "unit":
            x = torch.randn_like(input_embeds)
            std, mean = torch.std_mean(x, dim=-1, keepdim=True)
            x = (x - mean) / std
        return x
    
    def n_k_steps(self, mean_steps, mean_backprop_depth):
        # todo: get seeding working
        n_generator = torch.Generator(device="cpu")
        n_generator.manual_seed(42 % (2**31 - 1))
        k_generator = torch.Generator(device="cpu")
        k_generator.manual_seed(42 % (2**31 - 1))

        t = max(mean_steps - mean_backprop_depth, 0)
        s = mean_backprop_depth

        if self.training:
            # poisson log normal filling
            sigma = 0.5
            mu = math.log(t + s) - (sigma**2 / 2)
            rate = torch.zeros((1,)).log_normal_(mean=mu, std=sigma, generator=n_generator)
            p = torch.poisson(torch.tensor([rate], dtype=torch.float), generator=n_generator) + 1
            n = torch.clamp(p - s, min=0)
            k = torch.as_tensor(torch.minimum(torch.as_tensor(s), p))
        else:
            n, k = torch.tensor(mean_steps), torch.tensor(0)
        return n.to(torch.long), k.to(torch.long)

    def apply_thinking_layers(self,
                              x: torch.Tensor,
                              last_thought_state: Optional[torch.Tensor],
                              n_steps: Union[int, torch.Tensor],
                              decoder_sequences: torch.Tensor,
                              encoder_sequences: Optional[torch.Tensor],
                              decoder_attention_mask: Optional[torch.Tensor],
                              encoder_attention_mask: Optional[torch.Tensor]=None,
                              kv_cache: Optional[list[torch.Tensor]]=None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        gating_variances = []
        for _ in range(n_steps):
            gating_variances_this_step = []
            if self.adapter_method == "linear":
                x = self.adapter(torch.cat([x, decoder_sequences], dim=-1))
            else:
                x = self.adapter(x, decoder_sequences)
            for thinking_layer in self.thinking_block:
                x, gating_variance = self.apply_decoder_layer(x, encoder_sequences, decoder_attention_mask, thinking_layer, encoder_attention_mask, kv_cache)
                gating_variances_this_step.append(gating_variance)
                if not self.training and self.exit_criteria.should_exit(last_thought_state, x):
                    break
                last_thought_state = x
            gating_variances.extend(gating_variances_this_step)

        return x, gating_variances
    
    def decoder_block_forward(self,
                              decoder_sequences: torch.Tensor,
                              encoder_sequences: Optional[torch.Tensor],
                              decoder_attention_mask: Optional[torch.Tensor],
                              kv_caches: Optional[list[list[torch.Tensor]]]) -> tuple:
        if kv_caches is not None:
            warnings.warn("KV caches are not yet supported in HuginnDecoder")

        gating_variances = []
        for decoder_layer in self.prelude_layers:
            decoder_sequences, gating_variance = self.apply_decoder_layer(decoder_sequences, encoder_sequences, decoder_attention_mask, decoder_layer)
            if gating_variance is not None:
                gating_variances.append(gating_variance)

        if self.training:
            n_steps_no_grad, k_steps_grad = self.n_k_steps(self.mean_thinking_steps, self.mean_backprop_depth)
        else:
            n_steps_no_grad, k_steps_grad = self.mean_thinking_steps, 0

        x = self.initialize_thinking_state(decoder_sequences)
        with torch.no_grad():
            x, _ = self.apply_thinking_layers(x, None, n_steps_no_grad - 1, decoder_sequences, encoder_sequences, decoder_attention_mask)
            last_thought_state = x
            x, _ = self.apply_thinking_layers(x, last_thought_state, 1, decoder_sequences, encoder_sequences, decoder_attention_mask)

        if k_steps_grad > 0:
            decoder_sequences, thinking_gating_variances = self.apply_thinking_layers(x, last_thought_state, k_steps_grad, decoder_sequences, encoder_sequences, decoder_attention_mask)

        # only add grad-required gating variances
        if thinking_gating_variances is not None:
            gating_variances.extend(thinking_gating_variances)

        for decoder_layer in self.coda_layers:
            decoder_sequences, gating_variance = self.apply_decoder_layer(decoder_sequences, encoder_sequences, decoder_attention_mask, decoder_layer)
            if gating_variance is not None:
                gating_variances.append(gating_variance)

        return decoder_sequences, gating_variances, n_steps_no_grad, k_steps_grad

class MegaTransformer(nn.Module):
    def __init__(self, model_config):
        super(MegaTransformer, self).__init__()

        self.encoder_config = model_config.encoder_config
        self.decoder_config = model_config.decoder_config

        self.padding_value = model_config.padding_value

        self.encoder: Encoder = Encoder(self.encoder_config.device, model_config)
        self.decoder: Decoder = Decoder(self.decoder_config.device, model_config)

        self.main_criteria = criteria.LMLoss(ignore_index=model_config.padding_value, eps=model_config.label_smoothing)
        self.moe_criteria = criteria.TransformerMoELoss(model_config.moe_diversity_loss_coefficient)

    def forward(
            self,
            input_ids: torch.Tensor,
            labels: torch.Tensor,
            inputs_embeds: Optional[torch.Tensor]=None,
            attention_mask: Optional[torch.Tensor]=None,
            decoder_attention_mask: Optional[torch.Tensor]=None,
            return_dict: bool=False) -> Union[tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]], MegaTransformerOutput]:
        """
        Full encoder-decoder transformer forward pass function, with device cast handling if necessary (training does not support this but inference should)
        """
        if self.padding_value is not None:
            if attention_mask is None and input_ids is not None:
                attention_mask = input_ids == self.padding_value
            if decoder_attention_mask is None and labels is not None:
                decoder_attention_mask = labels == self.padding_value

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.encoder_config.device)

        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.to(self.decoder_config.device)

        input_ids = input_ids.to(self.encoder_config.device)
        labels = labels.to(self.decoder_config.device)
        self.encoder.embed_tokens = self.encoder.embed_tokens.to(self.encoder_config.device)

        encoder_sequences, encoder_gating_variances = self.encoder(input_ids=input_ids, key_padding_mask=attention_mask, inputs_embeds=inputs_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.decoder_config.device)

        encoder_sequences = encoder_sequences.to(self.decoder_config.device)
        self.decoder.embed_tokens = self.decoder.embed_tokens.to(self.decoder_config.device)
        self.decoder.lm_head = self.decoder.lm_head.to(self.decoder_config.device)
        
        logits, decoder_gating_variances = self.decoder(labels, encoder_sequences, attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask)

        main_loss = self.main_criteria(logits, labels)
        moe_loss, encoder_gating_variances_tensor, decoder_gating_variances_tensor = self.moe_criteria(encoder_gating_variances, decoder_gating_variances)

        total_loss = main_loss + moe_loss

        if return_dict:
            return MegaTransformerOutput(
                logits=logits,
                loss=total_loss,
                prediction_loss=main_loss,
                moe_loss=moe_loss,
                encoder_gating_variances=encoder_gating_variances_tensor,
                decoder_gating_variances=decoder_gating_variances_tensor
            )
        return logits, encoder_gating_variances, decoder_gating_variances
