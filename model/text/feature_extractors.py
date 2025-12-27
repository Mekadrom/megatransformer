import torch
import torch.nn as nn

from typing import Optional, Union

from model import kv_cache, SimpleBlock
from utils import configuration
from utils.megatransformer_utils import embedding_weight_init
from utils.model_utils import create_sinusoidal_1d_pos_encoding


class TextFeatureExtractor(nn.Module):
    def __init__(self, config: configuration.MegaTransformerConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)

        self.wpe: Optional[Union[nn.Embedding, nn.Parameter]] = None
        if config.use_positional_embedding:
            self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        elif config.use_sinusoidal_embedding:
            self.wpe = nn.Parameter(create_sinusoidal_1d_pos_encoding(config.max_position_embeddings, config.hidden_size))
            self.wpe.requires_grad = config.sinusoidal_embedding_learnable

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.prelude = SimpleBlock(
            config.text_prelude_config, "text_prelude", config.text_prelude_config.n_prelude_layers, config.hidden_dropout_prob
        )

        self._init_weights()

    def _init_weights(self):
        self.apply(embedding_weight_init(self.config.hidden_size))

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values: list[kv_cache.KVCache]=None,
        use_cache=False,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        batch_size, seq_length = input_ids.shape
        inputs_embeds = self.wte(input_ids)
        if self.wpe is not None:
            if isinstance(self.wpe, nn.Parameter):
                position_embeds = self.wpe
                position_embeds = position_embeds[:, :seq_length, :].expand(batch_size, -1, -1)
            else:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
        else:
            # positional embedding likely applied in self attention block
            hidden_states = inputs_embeds
        
        hidden_states = self.dropout(hidden_states)

        prelude_outputs = self.prelude(
            hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return prelude_outputs
