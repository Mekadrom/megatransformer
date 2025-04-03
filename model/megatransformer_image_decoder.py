from transformer import PreTrainedTokenizer
from typing import Union

from model import megatransformer_diffusion, megatransformer_recurrent, megatransformer_text_encoder

import megatransformer_utils
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageConditionalGaussianDiffusion(megatransformer_diffusion.GaussianDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def q_sample(self, x_start, t, noise=None, condition=None):
        # Same as before, condition not needed for forward process
        return super().q_sample(x_start, t, noise)
    
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, t_index, condition=None):
        """Single step of the reverse diffusion process with conditioning"""
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x.shape)
        
        # Model forward pass with condition
        model_output = self.unet(x, t.to(x.dtype), condition=condition)
        
        if self.predict_epsilon:
            # Model predicts noise ε
            pred_epsilon = model_output
            pred_x0 = sqrt_recip_alphas_t * x - sqrt_recip_alphas_t * sqrt_one_minus_alphas_cumprod_t * pred_epsilon
            pred_x0 = torch.clamp(pred_x0, -1., 1.)
            
            # Calculate posterior mean using betas_t
            denominator = torch.sqrt(1 - self._extract(self.alphas_cumprod, t, x.shape) + 1e-8)
            denominator = torch.clamp(denominator, min=1e-8)
            posterior_mean = (
                x * (1 - betas_t) / denominator +
                pred_x0 * betas_t / denominator
            )
        else:
            # Model directly predicts x_0
            pred_x0 = model_output
            pred_x0 = torch.clamp(pred_x0, -1., 1.)
            
            # Calculate posterior mean using betas_t
            denominator = torch.sqrt(1 - self._extract(self.alphas_cumprod, t, x.shape) + 1e-8)
            denominator = torch.clamp(denominator, min=1e-8)
            posterior_mean = (
                x * (1 - betas_t) / denominator +
                pred_x0 * betas_t / denominator
            )
        
        # Calculate posterior variance using betas_t
        posterior_variance = betas_t * (1 - self._extract(self.alphas_cumprod, t-1, x.shape)) / (1 - self._extract(self.alphas_cumprod, t, x.shape))
        
        if t_index == 0:
            # No noise at the last step (t=0)
            return posterior_mean
        else:
            noise = torch.randn_like(x)
            return posterior_mean + torch.sqrt(posterior_variance) * noise
    
    @torch.no_grad()
    def sample(self, device, condition, batch_size=1, image_size=224, return_intermediate=False) -> Union[tuple[torch.Tensor, list[torch.Tensor]], torch.Tensor]:
        """Sample with conditioning"""
        # Start from pure noise
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)
        
        # Iteratively denoise with conditioning
        time_step_outputs = []
        for time_step in reversed(range(1, self.num_timesteps)):
            t = torch.full((batch_size,), time_step, device=device, dtype=torch.long)
            x = self.p_sample(x, t, time_step, condition=condition)
            if return_intermediate:
                # perform same image normalization per intermediate step
                intermediate_output = (x + 1) / 2
                intermediate_output = torch.clamp(intermediate_output, 0.0, 1.0)
                time_step_outputs.append(intermediate_output)
            
        # Scale to [0, 1] range
        x = (x + 1) / 2
        x = torch.clamp(x, 0.0, 1.0)
        
        return x, time_step_outputs if return_intermediate else x
    
    def forward(self, x_0: torch.Tensor, condition=None):
        """Training forward pass with conditioning"""
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        
        # Model forward pass with condition
        model_output = self.unet(x_t, t.to(x_0.dtype), condition=condition)
        
        loss_dict = {}
        if self.predict_epsilon:
            # Loss to the added noise
            loss_dict["noise_mse"] = F.mse_loss(model_output, noise)

            pred_x0 = self.predict_start_from_noise(x_t, t, model_output)
            # megatransformer_utils.print_debug_tensor('initial pred_x0', pred_x0)
            pred_x0 = torch.clamp(pred_x0, -1., 1.)
            model_output = pred_x0
            
            # Loss to the original image
            loss_dict["pred_x0_mse"] = F.mse_loss(model_output, x_0)
        else:
            # Loss to the original image
            loss_dict["direct"] = F.mse_loss(model_output, x_0)

        total_loss = (
            1.0 * loss_dict.get("noise_mse", 0.0) +
            1.0 * loss_dict.get("pred_x0_mse", 0.0) +
            1.0 * loss_dict.get("direct", 0.0)
        )

        return model_output, total_loss

class ImageDiffusionSingleTaskModel(nn.Module):
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__()
        self.config = config

        self.text_recurrent = megatransformer_recurrent.MegaTransformerRecurrentCausalModel(config)
        self.diffuser = ImageConditionalGaussianDiffusion(
            hidden_size=config.hidden_size,
            activation=config.image_decoder_activation,
            scale_factor=(2, 2),
            stride=(2, 2),
            self_attn_class=megatransformer_diffusion.ImageDiffusionSelfAttentionBlock,
            cross_attn_class=megatransformer_diffusion.ImageDiffusionCrossAttentionBlock,
            in_channels=3,
            model_channels=config.image_decoder_model_channels,
            out_channels=3,
            time_embedding_dim=config.image_decoder_time_embedding_dim,
            num_res_blocks=config.image_decoder_num_res_blocks,
            has_condition=True,
            unet_dropout=config.image_decoder_unet_dropout,
            betas_schedule=config.image_decoder_betas_schedule,
            down_block_self_attn_n_heads=config.image_decoder_down_block_self_attn_n_heads,
            down_block_self_attn_d_queries=config.image_decoder_down_block_self_attn_d_queries,
            down_block_self_attn_d_values=config.image_decoder_down_block_self_attn_d_values,
            down_block_self_attn_use_flash_attention=config.image_decoder_down_block_self_attn_use_flash_attention,
            up_block_self_attn_n_heads=config.image_decoder_up_block_self_attn_n_heads,
            up_block_self_attn_d_queries=config.image_decoder_up_block_self_attn_d_queries,
            up_block_self_attn_d_values=config.image_decoder_up_block_self_attn_d_values,
            up_block_self_attn_use_flash_attention=config.image_decoder_up_block_self_attn_use_flash_attention,
            cross_attn_n_heads=config.image_decoder_cross_attn_n_heads,
            cross_attn_d_queries=config.image_decoder_cross_attn_d_queries,
            cross_attn_d_values=config.image_decoder_cross_attn_d_values,
            cross_attn_use_flash_attention=config.image_decoder_cross_attn_use_flash_attention,
        )

    def forward(self,
                input_ids,
                image_labels,
                attention_mask=None,
                past_key_values: list[megatransformer_utils.KVCache]=None,
                use_cache=False):
        text_embeddings = self.text_prelude(input_ids)

        # recurrent model (for enriching the text embeddings as conditioning for the image diffuser)
        text_embeddings = self.text_recurrent(
            text_embeddings, 
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )

        # reconstruction, loss
        return self.diffuser(
            image_labels,
            condition=text_embeddings,
        )

def create_small_multimodal_model(tokenizer: PreTrainedTokenizer, max_position_embeddings):
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            megatransformer_utils.BEGIN_AUDIO_TOKEN,
            megatransformer_utils.END_AUDIO_TOKEN,
            megatransformer_utils.BEGIN_IMAGE_TOKEN,
            megatransformer_utils.END_IMAGE_TOKEN
        ]
    })

    begin_audio_token_id = tokenizer.convert_tokens_to_ids(megatransformer_utils.BEGIN_AUDIO_TOKEN)
    end_audio_token_id = tokenizer.convert_tokens_to_ids(megatransformer_utils.END_AUDIO_TOKEN)
    begin_image_token_id = tokenizer.convert_tokens_to_ids(megatransformer_utils.BEGIN_IMAGE_TOKEN)
    end_image_token_id = tokenizer.convert_tokens_to_ids(megatransformer_utils.END_IMAGE_TOKEN)

    print(begin_audio_token_id, end_audio_token_id, begin_image_token_id, end_image_token_id)

    # uses a recurrent approach to emulate a deeper model (~317M params)
    config = megatransformer_utils.MegaTransformerConfig(
        vocab_size=tokenizer.vocab_size + 4,
        max_position_embeddings=max_position_embeddings,
        n_layers=None,
        n_prelude_layers=2,
        n_recurrent_layers=2,
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

        audio_decoder_model_channels=64,
        audio_decoder_time_embedding_dim=64,
        audio_decoder_num_res_blocks=2,
        audio_decoder_betas_schedule="cosine",
        audio_decoder_down_block_self_attn_n_heads=4,
        audio_decoder_up_block_self_attn_n_heads=4,
        audio_decoder_cross_attn_n_heads=4,

        image_decoder_model_channels=64,
        image_decoder_time_embedding_dim=64,
        image_decoder_num_res_blocks=2,
        image_decoder_betas_schedule="cosine",
    )

    config.text_prelude_config = config
    config.audio_prelude_config = config
    config.image_prelude_config = config

    config.text_coda_config = config
    config.audio_coda_config = config
    config.image_coda_config = config

    return ImageDiffusionSingleTaskModel(config)

def create_test_tiny_multimodal_model(tokenizer: PreTrainedTokenizer, max_position_embeddings):
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            megatransformer_utils.BEGIN_AUDIO_TOKEN,
            megatransformer_utils.END_AUDIO_TOKEN,
            megatransformer_utils.BEGIN_IMAGE_TOKEN,
            megatransformer_utils.END_IMAGE_TOKEN
        ]
    })

    begin_audio_token_id = tokenizer.convert_tokens_to_ids(megatransformer_utils.BEGIN_AUDIO_TOKEN)
    end_audio_token_id = tokenizer.convert_tokens_to_ids(megatransformer_utils.END_AUDIO_TOKEN)
    begin_image_token_id = tokenizer.convert_tokens_to_ids(megatransformer_utils.BEGIN_IMAGE_TOKEN)
    end_image_token_id = tokenizer.convert_tokens_to_ids(megatransformer_utils.END_IMAGE_TOKEN)

    print(begin_audio_token_id, end_audio_token_id, begin_image_token_id, end_image_token_id)

    # uses a recurrent approach to emulate a deeper model (~M params)
    config = megatransformer_utils.MegaTransformerConfig(
        vocab_size=tokenizer.vocab_size + 4,
        max_position_embeddings=max_position_embeddings,
        n_layers=None,
        hidden_size=8,
        d_queries=4,
        d_values=4,
        n_query_groups=2,
        n_heads=2,
        intermediate_size=64,
        n_prelude_layers=1,
        n_recurrent_layers=1,
        n_coda_layers=1,
        intermediate_activation="relu",
        norm_type="layernorm",
        ffn_type="mlp",
        use_positional_embedding=False,
        use_sinusoidal_embedding=False,
        use_rotary_embedding=True,
        rotary_embedding_dim=4,
        use_alibi_bias=False,
        use_qkv_bias=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,

        begin_audio_token_id=begin_audio_token_id,
        end_audio_token_id=end_audio_token_id,
        begin_image_token_id=begin_image_token_id,
        end_image_token_id=end_image_token_id,

        audio_decoder_model_channels=32,

        audio_decoder_down_block_self_attn_n_heads=2,
        audio_decoder_down_block_self_attn_d_queries=16,
        audio_decoder_down_block_self_attn_d_values=16,
        audio_decoder_up_block_self_attn_n_heads=2,
        audio_decoder_up_block_self_attn_d_queries=16,
        audio_decoder_up_block_self_attn_d_values=16,
        audio_decoder_cross_attn_n_heads=2,
        audio_decoder_cross_attn_d_queries=16,
        audio_decoder_cross_attn_d_values=16,

        audio_vocoder_hidden_channels=16,
        audio_vocoder_n_residual_layers=1,

        image_decoder_model_channels=32,

        image_decoder_down_block_self_attn_n_heads=2,
        image_decoder_down_block_self_attn_d_queries=16,
        image_decoder_down_block_self_attn_d_values=16,
        image_decoder_up_block_self_attn_n_heads=2,
        image_decoder_up_block_self_attn_d_queries=16,
        image_decoder_up_block_self_attn_d_values=16,
        image_decoder_cross_attn_n_heads=2,
        image_decoder_cross_attn_d_queries=16,
        image_decoder_cross_attn_d_values=16,
    )

    config.text_prelude_config = config
    config.audio_prelude_config = config
    config.image_prelude_config = config

    config.text_coda_config = config
    config.audio_coda_config = config
    config.image_coda_config = config

    return ImageDiffusionSingleTaskModel(config)

lookup = {
    "small_multimodal": create_small_image_diffusion_model,
    "test_tiny_multimodal": create_test_tiny_image_diffusion_model,
}

def model_config_lookup(config):
    if config not in lookup:
        raise ValueError(f"Unknown model configuration: {config}")
    return lookup[config]
