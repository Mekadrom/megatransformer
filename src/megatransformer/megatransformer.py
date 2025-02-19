from torch import nn
from typing import Optional, Union

from . import embedding_mlp, infinite_multihead_attn, millions_moe, phi3_mlp, per_lang_embedding, positionwise_fcn, multihead_attn, sum, transformer_utils, criteria, grouped_query_attn

import admin_torch
import math
import torch

class MegaTransformerOutput:
    def __init__(self, logits: torch.Tensor, loss: torch.Tensor, prediction_loss: torch.Tensor, moe_loss: torch.Tensor, decoder_gating_variances: list[torch.Tensor], encoder_gating_variances: Optional[list[torch.Tensor]]=None):
        self.logits = logits
        self.loss = loss
        self.prediction_loss = prediction_loss
        self.moe_loss = moe_loss
        self.decoder_gating_variances = decoder_gating_variances
        self.encoder_gating_variances = encoder_gating_variances

class EncoderLayer(nn.Module):
    def __init__(self, device, model_config):
        super(EncoderLayer, self).__init__()

        encoder_config = model_config.encoder_config

        self_attn_config = encoder_config.self_attn_config
        ffn_config = model_config.ffn_config

        self.use_admin = model_config.use_admin
        self.norm = model_config.norm
        self.norm_eps = model_config.norm_eps

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
            self.self_attn_residual = admin_torch.as_module(self.n_layers)
            self.ffn_residual = admin_torch.as_module(self.n_layers)
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
        self_attn, _ = self.self_attn(encoder_sequences, encoder_sequences, encoder_sequences, key_padding_mask)

        encoder_sequences = self.self_attn_residual(encoder_sequences, self_attn)
        ffn, gating_variances = self.ffn(encoder_sequences)
        encoder_sequences = self.ffn_residual(encoder_sequences, ffn)
            
        return encoder_sequences, gating_variances

class Encoder(nn.Module):
    def __init__(self, device, model_config):
        super(Encoder, self).__init__()

        self.device = device

        self.model_config = model_config
        encoder_config = model_config.encoder_config

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
        return self.embed_tokens(encoder_sequences) * math.sqrt(self.d_model)

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

        # post-LN
        encoder_sequences = self.post_encoder_norm(encoder_sequences)

        return encoder_sequences, gating_variances

class DecoderLayer(nn.Module):
    def __init__(self, device, model_config):
        super(DecoderLayer, self).__init__()

        decoder_config = model_config.decoder_config

        self_attn_config = decoder_config.self_attn_config
        cross_attn_config = decoder_config.cross_attn_config
        use_cross_attn = cross_attn_config is not None
        ffn_config = model_config.ffn_config
        
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
            self.self_attn_residual = admin_torch.as_module(self.n_layers)
            self.cross_attn_residual = admin_torch.as_module(self.n_layers)
            self.ffn_residual = admin_torch.as_module(self.n_layers)
        else:
            self.self_attn_residual = sum.Sum()
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
        # print(f"decoder_sequences before self attn shape: {decoder_sequence.shape}, min: {decoder_sequence.min()}, max: {decoder_sequence.max()}, mean: {decoder_sequence.mean()}, std: {decoder_sequence.std()}, norm: {decoder_sequence.norm()}")

        self_attn, _ = self.self_attn(decoder_sequence, decoder_sequence, decoder_sequence, decoder_attention_mask)
        decoder_sequence = self.self_attn_residual(decoder_sequence, self_attn)

        # print(f"decoder_sequences before cross attn shape: {decoder_sequence.shape}, min: {decoder_sequence.min()}, max: {decoder_sequence.max()}, mean: {decoder_sequence.mean()}, std: {decoder_sequence.std()}, norm: {decoder_sequence.norm()}")

        if self.cross_attn is not None and encoder_sequence is not None:
            cross_attn, _ = self.cross_attn(decoder_sequence, encoder_sequence, encoder_sequence, encoder_attention_mask, kv_cache=kv_cache)
            decoder_sequence = self.cross_attn_residual(decoder_sequence, cross_attn)

        ffn, gating_variances = self.ffn(decoder_sequence)

        decoder_sequence = self.ffn_residual(decoder_sequence, ffn)

        return decoder_sequence, gating_variances

class Decoder(nn.Module):
    def __init__(self, device, model_config):
        super(Decoder, self).__init__()

        self.device = device

        self.model_config = model_config
        decoder_config = model_config.decoder_config

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
            self.tensor_positional_encoding = nn.Parameter(transformer_utils.get_tensor_positional_encoding(self.device, self.d_model, self.positional_encoding_dim, self.learnable_positional_encoding, self.maxlen))

        self.main_criteria = criteria.LMLoss(ignore_index=model_config.padding_value, eps=model_config.label_smoothing)
        self.moe_criteria = criteria.DecoderOnlyMoELoss(decoder_config.moe_diversity_loss_coefficient)

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
        return self.embed_tokens(decoder_sequences) * math.sqrt(self.d_model)
    
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

    def forward(self,
                input_ids: torch.Tensor,
                labels: Optional[torch.Tensor]=None,
                encoder_sequences: Optional[torch.Tensor]=None,
                targets_embeds: Optional[torch.Tensor]=None,
                decoder_attention_mask: Optional[torch.Tensor]=None,
                return_dict: bool=False,
                kv_caches: Optional[list[list[torch.Tensor]]]=None) -> Union[tuple[torch.Tensor, list[torch.Tensor]], MegaTransformerOutput]:
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

        decoder_sequences = self.post_decoder_norm(decoder_sequences)
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
                decoder_gating_variances=gating_variances_tensor
            )
        return logits, gating_variances

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
