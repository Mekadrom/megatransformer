from torch import nn
from transformers import GenerationMixin, PreTrainedModel, PreTrainedTokenizer
from typing import Optional, Union

from model import megatransformer_blocks, megatransformer_image_decoder, swiglu

import megatransformer_utils
import torch
import torch.nn.functional as F


BEGIN_IMAGE_TOKEN = "<|IMAGE|>"
END_IMAGE_TOKEN = "<|/IMAGE|>"

BEGIN_AUDIO_TOKEN = "<|AUDIO|>"
END_AUDIO_TOKEN = "<|/AUDIO|>"



class SimpleBlock(nn.Module):
    def __init__(self, config, n_layers: int, dropout: float):
        super().__init__()
        self.config = config
        self.prelude = nn.ModuleList([megatransformer_blocks.MegaTransformerBlock(config) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_values: list[megatransformer_utils.KVCache]=None,
        use_cache=False,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        all_hidden_states: Optional[list] = [] if output_hidden_states else None
        all_attentions: Optional[list] = [] if output_attentions else None

        for i, block in enumerate(self.prelude):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)

            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i] if head_mask is not None else None,
                past_key_values=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states
            )

            hidden_states = outputs.hidden_states
            attention_probs = outputs.attention_probs

            if all_attentions is not None:
                all_attentions.append(attention_probs)

        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        hidden_states = self.dropout(hidden_states)

        if not return_dict:
            return (
                hidden_states,
                past_key_values,
                all_hidden_states,
                all_attentions,
            )

        return megatransformer_utils.MegaTransformerCausalOutput(
            logits=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

class TextFeatureExtractor(nn.Module):
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)

        self.wpe: Optional[Union[nn.Embedding, nn.Parameter]] = None
        if config.use_positional_embedding:
            self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        elif config.use_sinusoidal_embedding:
            self.wpe = nn.Parameter(megatransformer_utils.create_sinusoidal_embedding(config.max_position_embeddings, config.hidden_size))
            self.wpe.requires_grad = config.sinusoidal_embedding_learnable

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.prelude = SimpleBlock(config, config.n_prelude_layers, config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values: list[megatransformer_utils.KVCache]=None,
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
                activation_type() if activation_type is not swiglu.SwiGLU else swiglu.SwiGLU(out_channels),
                nn.Dropout2d(dropout)
            ))
    
    def forward(self, x: torch.Tensor):
        # x: [batch_size, channels, height, width]
        for layer in self.conv_layers:
            x = layer(x)
        return x

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

        self.prelude = SimpleBlock(config, config.n_prelude_layers, config.hidden_dropout_prob)

    def forward(
        self,
        features: torch.Tensor,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        N, C, M, T = features.shape


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

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        assert H == W == self.img_size, f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"
        
        # (B, embed_dim, H/patch_size, W/patch_size) -> (B, embed_dim, n_patches)
        x = self.proj(x)
        x = x.flatten(2)
        # (B, embed_dim, n_patches) -> (B, n_patches, embed_dim)
        x = x.transpose(1, 2)
        
        return x

class ImageViTFeatureExtractor(nn.Module):
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__()
        self.config = config

        self.patch_embed = PatchEmbedding(config.image_image_size, config.image_encoder_patch_size, 3, config.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, self.config.hidden_size))
        
        self.dropout = nn.Dropout(config.image_encoder_pos_dropout)

        self.prelude = SimpleBlock(config, config.n_prelude_layers, config.hidden_dropout_prob)

    def forward(
        self,
        image_raw_inputs,
        past_key_values: list[megatransformer_utils.KVCache]=None,
        use_cache=False,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # image_raw_inputs: [batch_size, channels, height, width]
        B = image_raw_inputs.shape[0]

        # patches: [batch_size, n_patches, hidden_size] / [batch_size, (img_size // patch_size) ** 2, hidden_size]
        patches = self.patch_embed(image_raw_inputs)
        patches = patches + self.pos_embed

        patches = self.dropout(patches)

        attention_mask = torch.ones((B, patches.size(1)), device=patches.device)

        tokens = self.prelude(
            patches,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return tokens

class AudioEmbeddingUpsampleConv2dGenerator(nn.Module):
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__()
        self.config = config

        activation = config.image_decoder_activation
        dropout = config.image_decoder_dropout

        self.conv_layers = nn.ModuleList([
            nn.Unflatten(-1, (768, 1, 1)),
        ])

        activation_type = megatransformer_utils.get_activation_type(activation)

        channels = [768, 256, 128, 64, 1]
        image_sizes = [1, 4, 16, 64, 128]
        
        for i in range(len(channels) - 1):
            out_channels = channels[i+1]
            upsample_target = image_sizes[i+1]

            self.conv_layers.append(nn.Sequential(
                nn.Upsample(size=(upsample_target, upsample_target), mode="bilinear", align_corners=False),
                nn.Conv2d(channels[i], out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                activation_type() if activation_type is not swiglu.SwiGLU else swiglu.SwiGLU(out_channels),
                nn.Dropout2d(dropout)
            ))

    def forward(self, x: torch.Tensor):
        # naive approach; alternating conv2d and upsample layers to reach n_mels
        # x: [batch_size, timestep, hidden_size]
        x = x.permute(0, 2, 1).unsqueeze(1)  # [batch_size, channels, hidden_size, timestep]
        for layer in self.conv_layers:
            x = layer(x)
        return x

class MegaTransformerMultimodalEncoder(nn.Module):
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__()
        self.config = config

        self.text_embedding = TextFeatureExtractor(config)
        self.audio_embedding = AudioFeatureExtractor(config)
        self.image_embedding = ImageViTFeatureExtractor(config)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self,
                text_input_ids=None,
                audio_raw_inputs=None,
                image_raw_inputs=None,
                text_past_key_values=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if text_input_ids is None:
            text_prelude_outputs = None
        else:
            text_prelude_outputs = self.text_embedding(text_input_ids, past_key_values=text_past_key_values)

        if audio_raw_inputs is None:
            audio_prelude_outputs = None
        else:
            audio_prelude_outputs = self.audio_embedding(audio_raw_inputs)

        if image_raw_inputs is None:
            image_prelude_outputs = None
        else:
            image_prelude_outputs = self.image_embedding(image_raw_inputs)

        return text_prelude_outputs, audio_prelude_outputs, image_prelude_outputs

class MegaTransformerMultimodalDecoder(nn.Module):
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__()
        self.config = config

        self.text_coda = nn.ModuleList([megatransformer_blocks.MegaTransformerBlock(config) for _ in range(config.n_coda_layers)])
        self.text_decoder = nn.Linear(config.hidden_size, config.vocab_size)

        self.audio_coda = nn.ModuleList([megatransformer_blocks.MegaTransformerBlock(config) for _ in range(config.n_coda_layers)])
        self.audio_decoder =  AudioEmbeddingUpsampleConv2dGenerator(config),

        self.image_coda = nn.ModuleList([megatransformer_blocks.MegaTransformerBlock(config) for _ in range(config.n_coda_layers)])
        self.image_decoder = megatransformer_image_decoder.ConditionalGaussianDiffusion(config.image_decoder_activation, config.hidden_size)

    def forward(self,
                text_hidden_states=None,
                image_hidden_states=None,
                audio_hidden_states=None,
                image_labels=None):
        if text_hidden_states is None:
            text_logits = None
        else:
            text_hidden_states = self.text_coda(text_hidden_states)[0]
            text_logits = self.text_decoder(text_hidden_states)

        if audio_hidden_states is None:
            audio_outputs = None
        else:
            audio_hidden_states = self.audio_coda(audio_hidden_states)[0]
            audio_outputs = self.audio_decoder(audio_hidden_states)

        if image_hidden_states is None:
            image_outputs = None
            image_reconstruction_loss = None
        else:
            image_hidden_states = self.image_coda(image_hidden_states)[0]
            image_outputs, image_reconstruction_loss = self.image_decoder(image_labels, image_hidden_states)

        return text_logits, audio_outputs, image_outputs, image_reconstruction_loss

class MegaTransformerMultimodalRecurrentModel(PreTrainedModel, GenerationMixin):
    config_class = megatransformer_utils.MegaTransformerConfig
    
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__(config)
        self.config = config
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        prelude = nn.ModuleList([megatransformer_blocks.MegaTransformerBlock(config) for _ in range(config.n_prelude_layers)])
        recurrent = megatransformer_blocks.MegaTransformerRecurrentBlock(config)
        coda = nn.ModuleList([megatransformer_blocks.MegaTransformerBlock(config) for _ in range(config.n_coda_layers)])

        self.transformer = nn.ModuleList([*prelude, recurrent, *coda])
        
        if config.use_final_norm:
            self.norm_final = megatransformer_utils.create_norm(config.hidden_size, config.norm_type, config.norm_eps)
        else:
            self.norm_final = None
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights as in the original implementation"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        inputs_embeds,
        attention_mask=None,
        past_key_values: list[megatransformer_utils.KVCache]=None,
        use_cache=False,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        batch_size, seq_length = inputs_embeds.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=inputs_embeds.device)
        
        hidden_states = self.dropout(inputs_embeds)
        
        # Initialize lists to store outputs for each layer
        all_hidden_states: Optional[list] = [] if output_hidden_states else None
        all_attentions: Optional[list] = [] if output_attentions else None

        recurrent_outputs = None
        for i, block in enumerate(self.transformer):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states
            )

            if isinstance(block, megatransformer_blocks.MegaTransformerRecurrentBlock):
                recurrent_outputs = outputs

            hidden_states = outputs.hidden_states
            attention_probs = outputs.attention_probs

            if hasattr(outputs, "all_hidden_states") and all_hidden_states is not None:
                all_hidden_states.extend(outputs.all_hidden_states)
            elif hasattr(outputs, "hidden_states") and all_hidden_states is not None:
                all_hidden_states.append(outputs.hidden_states)
            
            if all_attentions:
                all_attentions.append(attention_probs)
        
        if self.norm_final is not None:
            hidden_states = self.norm_final(hidden_states)
        
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)
        
        if not return_dict:
            return (
                hidden_states,
                past_key_values,
                all_hidden_states,
                all_attentions,
                recurrent_outputs.n_steps_no_grad if recurrent_outputs is not None else None,
                recurrent_outputs.k_steps_grad if recurrent_outputs is not None else None,
            )
        
        return megatransformer_utils.MegaTransformerCausalOutput(
            logits=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            n_steps_no_grad=recurrent_outputs.n_steps_no_grad if recurrent_outputs is not None else None,
            k_steps_grad=recurrent_outputs.k_steps_grad if recurrent_outputs is not None else None,
        )
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask: torch.Tensor=None, **kwargs):
        if past_key_values is not None and past_key_values[0] is not None:
            input_ids = input_ids[:, -1:]

        use_cache = kwargs.get("use_cache", True)
        position_ids = kwargs.get("position_ids", None)
        
        if position_ids is not None:
            position_ids = position_ids[:, -1:] if position_ids is not None else None

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
    

class MegaTransformerCausalWMHeads(PreTrainedModel, GenerationMixin):
    config_class = megatransformer_utils.MegaTransformerConfig
    
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__(config)
        self.config = config

        self.input_transform = MegaTransformerMultimodalEncoder(config)
        self.world_model = MegaTransformerMultimodalRecurrentModel(config)
        self.output_transform = MegaTransformerMultimodalDecoder(config)

    def get_input_embeddings(self):
        return self.input_transform.text_embedding.wte
    
    def set_input_embeddings(self, new_embeddings):
        self.input_transform.text_embedding.wte = new_embeddings
    
    def get_output_embeddings(self):
        return self.output_transform.text_decoder
    
    def set_output_embeddings(self, new_embeddings):
        self.output_transform.text_decoder = new_embeddings

    def interleave_batch_aligned_embeds(self,
                                        text_input_ids,
                                        audio_raw_inputs,
                                        image_raw_inputs,
                                        text_past_key_values=None):
        # todo
        pass

    def extract_batch_aligned_multimodal_features(self, hidden_states, all_audio_positions, all_image_positions):
        # todo
        pass
    
    def forward(
        self,
        input_ids=None,
        audio_raw_inputs=None,
        image_raw_inputs=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        past_key_values: tuple[list[megatransformer_utils.KVCache], list[megatransformer_utils.KVCache], list[megatransformer_utils.KVCache], list[megatransformer_utils.KVCache]]=None,
        use_cache=False,
        inputs_embeds=None,
        labels=None,
        image_labels=None,
        audio_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass for the MegaTransformer model.
        Args:
            text_inputs: Input text IDs.
            image_inputs: Input image features. Where the IDs match <|IMAGE|>, the corresponding image features from the image at the same index in text_inputs will be concatenated.
            audio_inputs: Input audio features. Where the IDs match <|AUDIO|>, the corresponding audio features from the audio at the same index in audio_inputs will be concatenated.

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        past_key_values = past_key_values if past_key_values is not None else [None] * (self.config.n_prelude_layers + 1 + self.config.n_coda_layers)

        # if use_cache:
        #     # initialize kv caches for prelude layers as past_key_values[0:n_prelude_layers] = KVCache()
        #     for i in range(len(self.config.n_prelude_layers)):
        #         if past_key_values[i] is None:
        #             past_key_values[i] = megatransformer_utils.KVCache()

        #     # initialize kv caches for recurrent layers as past_key_values[n_prelude_layers] = list[KVCache()]
        #     if past_key_values[self.config.n_prelude_layers] is None:
        #         # use a list of KVCache() for the recurrent layer
        #         recurrent_kv_cache = [None] * (self.config.recurrent_mean_thinking_steps * 2)
        #         for j in range(len(recurrent_kv_cache)):
        #             recurrent_kv_cache[j] = megatransformer_utils.KVCache()
        #         past_key_values[self.config.n_prelude_layers] = recurrent_kv_cache

        #     # initialize kv caches for coda layers as past_key_values[n_prelude_layers+1:] = KVCache()
        #     for i in range(self.config.n_prelude_layers + 1, len(past_key_values)):
        #         if past_key_values[i] is None:
        #             past_key_values[i] = megatransformer_utils.KVCache()
        # print(f"initialized past_key_values: {past_key_values}")

        # batch_index, match_index

        if inputs_embeds is None:
            if (audio_raw_inputs is not None and audio_raw_inputs.shape[-1] != 0) or (image_raw_inputs is not None and image_raw_inputs.shape[-1] != 0) :
                # multimodal
                inputs_embeds, attention_mask, image_positions, audio_positions, text_prelude_outputs, audio_prelude_outputs, image_prelude_outputs = self.interleave_batch_aligned_embeds(
                    input_ids,
                    audio_raw_inputs,
                    image_raw_inputs,
                    text_past_key_values=past_key_values[:self.config.n_prelude_layers],
                )
            else:
                # text only
                text_prelude_outputs = self.input_transform(text_input_ids=input_ids)[0]
                inputs_embeds = text_prelude_outputs[0]  # [batch_size, seq_length, hidden_size]
                attention_mask = torch.ones((inputs_embeds.shape[0], inputs_embeds.shape[1]), device=inputs_embeds.device)
                audio_positions = []
                image_positions = []
        else:
            assert labels is None and image_labels is None and audio_labels is None, "If inputs_embeds is provided, labels, image_labels, and audio_labels should not be provided."
        
        transformer_outputs = self.world_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if return_dict:
            # dict
            hidden_states = transformer_outputs.logits
        else:
            # tuple
            hidden_states = transformer_outputs[0]
        
        if len(image_positions) != 0 and len(audio_positions) != 0:
            text_hidden_states, audio_hidden_states, image_hidden_states = self.extract_batch_aligned_multimodal_features(
                hidden_states,
                image_positions,
                audio_positions,
            )
        else:
            text_hidden_states = hidden_states
            audio_hidden_states = None
            image_hidden_states = None

        # hidden states is raw embedding space that hasn't been classified yet
        output_text_logits, output_audio, output_images, image_reconstruction_loss = self.output_transform(
            text_hidden_states=text_hidden_states,
            audio_hidden_states=audio_hidden_states,
            image_hidden_states=image_hidden_states,
            image_labels=image_labels,
        )

        text_loss = None
        image_loss = None
        audio_loss = None
        if labels is not None:
            shift_logits = output_text_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            text_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if audio_labels is not None:
            loss_fct = nn.MSELoss()
            audio_loss = loss_fct(output_audio, audio_labels)
        if image_labels is not None:
            loss_fct = nn.MSELoss()
            image_loss = loss_fct(output_images, image_labels)

        total_loss = None
        if text_loss is not None:
            total_loss = text_loss

        if audio_loss is not None:
            if total_loss is None:
                total_loss = audio_loss
            else:
                total_loss = total_loss + audio_loss

        if image_loss is not None:
            if total_loss is None:
                total_loss = image_loss
            else:
                total_loss = total_loss + image_loss

        if image_reconstruction_loss is not None:
            if total_loss is None:
                total_loss = image_reconstruction_loss
            else:
                total_loss = total_loss + image_reconstruction_loss
        
        if not return_dict:
            output = (
                output_text_logits,
                output_audio,
                output_images,
                *transformer_outputs[1:],
            )
            return ((total_loss,) + output) if total_loss is not None else output
        
        return megatransformer_utils.MegaTransformerMultimodalOutput(
            loss=total_loss,
            logits=output_text_logits,
            audio_raw_outputs=output_audio,
            image_raw_outputs=output_images,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            n_steps_no_grad=transformer_outputs.n_steps_no_grad,
            k_steps_grad=transformer_outputs.k_steps_grad,
        )
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return self.world_model.prepare_inputs_for_generation(input_ids, **kwargs)


def create_small_multimodal_model(tokenizer: PreTrainedTokenizer, max_position_embeddings):
    tokenizer.add_special_tokens({
        "additional_special_tokens": [BEGIN_AUDIO_TOKEN, END_AUDIO_TOKEN, BEGIN_IMAGE_TOKEN, END_IMAGE_TOKEN]
    })

    begin_audio_token_id = tokenizer.convert_tokens_to_ids(BEGIN_AUDIO_TOKEN)
    end_audio_token_id = tokenizer.convert_tokens_to_ids(END_AUDIO_TOKEN)
    begin_image_token_id = tokenizer.convert_tokens_to_ids(BEGIN_IMAGE_TOKEN)
    end_image_token_id = tokenizer.convert_tokens_to_ids(END_IMAGE_TOKEN)

    print(begin_audio_token_id, end_audio_token_id, begin_image_token_id, end_image_token_id)

    # uses a recurrent approach to emulate a deeper model (~317M params)
    return MegaTransformerCausalWMHeads(megatransformer_utils.MegaTransformerConfig(
        vocab_size=tokenizer.vocab_size + 4,
        max_position_embeddings=max_position_embeddings,
        n_layers=None,
        n_prelude_layers=2,
        n_recurrent_layers=4,
        n_coda_layers=2,
        intermediate_activation="swiglu",
        norm_type="rmsnorm",
        ffn_type="mlp",
        use_positional_embedding=False,
        use_sinusoidal_embedding=False,
        use_rotary_embedding=True,
        use_alibi_bias=False,
        use_qkv_bias=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,

        begin_audio_token_id=begin_audio_token_id,
        end_audio_token_id=end_audio_token_id,
        begin_image_token_id=begin_image_token_id,
        end_image_token_id=end_image_token_id,
        # defaults otherwise
    ))

def create_medium_multimodal_model(tokenizer: PreTrainedTokenizer, max_position_embeddings):
    tokenizer.add_special_tokens({
        "additional_special_tokens": [BEGIN_AUDIO_TOKEN, END_AUDIO_TOKEN, BEGIN_IMAGE_TOKEN, END_IMAGE_TOKEN]
    })

    begin_audio_token_id = tokenizer.convert_tokens_to_ids(BEGIN_AUDIO_TOKEN)
    end_audio_token_id = tokenizer.convert_tokens_to_ids(END_AUDIO_TOKEN)
    begin_image_token_id = tokenizer.convert_tokens_to_ids(BEGIN_IMAGE_TOKEN)
    end_image_token_id = tokenizer.convert_tokens_to_ids(END_IMAGE_TOKEN)

    print(begin_audio_token_id, end_audio_token_id, begin_image_token_id, end_image_token_id)

    # uses a recurrent approach to emulate a deeper model (~M params)
    return MegaTransformerCausalWMHeads(megatransformer_utils.MegaTransformerConfig(
        vocab_size=tokenizer.vocab_size + 4,
        max_position_embeddings=max_position_embeddings,
        hidden_size=1024,
        d_queries=64,
        d_values=64,
        n_query_groups=16,
        n_heads=16,
        intermediate_size=4096,
        n_layers=None,
        n_prelude_layers=2,
        n_recurrent_layers=4,
        n_coda_layers=2,
        intermediate_activation="swiglu",
        norm_type="rmsnorm",
        ffn_type="mlp",
        use_positional_embedding=False,
        use_sinusoidal_embedding=False,
        use_rotary_embedding=True,
        use_alibi_bias=False,
        use_qkv_bias=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,

        begin_audio_token_id=begin_audio_token_id,
        end_audio_token_id=end_audio_token_id,
        begin_image_token_id=begin_image_token_id,
        end_image_token_id=end_image_token_id,
        # defaults otherwise
    ))

def create_test_tiny_multimodal_model(tokenizer: PreTrainedTokenizer, max_position_embeddings):
    tokenizer.add_special_tokens({
        "additional_special_tokens": [BEGIN_AUDIO_TOKEN, END_AUDIO_TOKEN, BEGIN_IMAGE_TOKEN, END_IMAGE_TOKEN]
    })

    begin_audio_token_id = tokenizer.convert_tokens_to_ids(BEGIN_AUDIO_TOKEN)
    end_audio_token_id = tokenizer.convert_tokens_to_ids(END_AUDIO_TOKEN)
    begin_image_token_id = tokenizer.convert_tokens_to_ids(BEGIN_IMAGE_TOKEN)
    end_image_token_id = tokenizer.convert_tokens_to_ids(END_IMAGE_TOKEN)

    print(begin_audio_token_id, end_audio_token_id, begin_image_token_id, end_image_token_id)

    # uses a recurrent approach to emulate a deeper model (~M params)
    return MegaTransformerCausalWMHeads(megatransformer_utils.MegaTransformerConfig(
        vocab_size=tokenizer.vocab_size + 4,
        max_position_embeddings=max_position_embeddings,
        n_layers=None,
        hidden_size=64,
        d_queries=16,
        d_values=16,
        n_query_groups=4,
        n_heads=4,
        intermediate_size=256,
        n_prelude_layers=1,
        n_recurrent_layers=1,
        n_coda_layers=1,
        intermediate_activation="relu",
        norm_type="layernorm",
        ffn_type="mlp",
        use_positional_embedding=False,
        use_sinusoidal_embedding=False,
        use_rotary_embedding=True,
        use_alibi_bias=False,
        use_qkv_bias=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,

        begin_audio_token_id=begin_audio_token_id,
        end_audio_token_id=end_audio_token_id,
        begin_image_token_id=begin_image_token_id,
        end_image_token_id=end_image_token_id,
        # defaults otherwise
    ))

lookup = {
    "small_multimodal": create_small_multimodal_model,
    "test_tiny_multimodal": create_test_tiny_multimodal_model,
}

def model_config_lookup(config):
    if config not in lookup:
        raise ValueError(f"Unknown model configuration: {config}")
    return lookup[config]
