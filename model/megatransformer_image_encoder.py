from model import megatransformer_modules

import megatransformer_utils
import torch
import torch.nn as nn


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

        self._init_weights()

    def _init_weights(self):
        # Initialize weights for the Conv2d layer
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
    
    def forward(self, features: torch.Tensor):
        # x: (B, C, H, W)
        B, C, H, W = features.shape
        assert H == W == self.img_size, f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"
        
        # (B, embed_dim, H/patch_size, W/patch_size) -> (B, embed_dim, n_patches)
        features = self.proj(features)
        features = features.flatten(2)
        # (B, embed_dim, n_patches) -> (B, n_patches, embed_dim)
        features = features.transpose(1, 2)
        
        return features

class ImageViTFeatureExtractor(nn.Module):
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__()
        self.config = config

        self.patch_embed = PatchEmbedding(config.image_size, config.image_encoder_patch_size, 3, config.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, self.config.hidden_size))
        
        self.dropout = nn.Dropout(config.image_encoder_pos_dropout)

        self.prelude = megatransformer_modules.SimpleBlock(
            config.image_prelude_config, "image_prelude", config.image_prelude_config.n_prelude_layers, config.hidden_dropout_prob
        )

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

        outputs = self.prelude(
            patches,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs
