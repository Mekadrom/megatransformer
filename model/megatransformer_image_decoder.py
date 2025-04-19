from transformers import PreTrainedTokenizer

from model import megatransformer_diffusion, megatransformer_recurrent

import megatransformer_utils

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[-1] ** 0.5)

class ImageSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, d_queries, d_values, use_flash_attention=True, dropout=0.1, is_linear_attention=True):
        super().__init__()
        self.hidden_dim = hidden_size
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.use_flash_attention = use_flash_attention
        self.dropout_p = dropout  # Store dropout probability for flash attention
        self.is_linear_attention = is_linear_attention

        self.q_proj = nn.Conv2d(hidden_size, d_queries * n_heads, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(hidden_size, d_queries * n_heads, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(hidden_size, d_values * n_heads, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(self.d_values * n_heads, hidden_size, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        self.apply(megatransformer_utils.transformer_weight_init())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-head self attention for image data, treating spatial dimensions (H,W) as the sequence length.
        Args:
            x: [B, C, H, W] where B is batch size, C is channels, H is height, W is width.
        Returns:
            output: [B, C, H, W] where attention is applied along the spatial dimensions.
        """
        B, C, H, W = x.shape

        q: torch.Tensor = self.q_proj(x)  # [B, n_heads*d_queries, H, W]
        k: torch.Tensor = self.k_proj(x)  # [B, n_heads*d_queries, H, W]
        v: torch.Tensor = self.v_proj(x)  # [B, n_heads*d_values, H, W]
        
        q = q.view(B, self.n_heads, self.d_queries, -1)  # [B, n_heads, d_queries, H*W]
        k = k.view(B, self.n_heads, self.d_queries, -1)  # [B, n_heads, d_queries, H*W]
        v = v.view(B, self.n_heads, self.d_values, -1)  # [B, n_heads, d_values, H*W]

        if self.is_linear_attention:
            q = q.softmax(dim=-2)
            k = k.softmax(dim=-1)
        
        output: torch.Tensor
        if self.use_flash_attention:
            output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False
            )  # [B, n_heads, H*W, d_values]
        else:
            scale = 1.0 / math.sqrt(self.d_queries)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, n_heads, H*W, H*W]
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            output = torch.matmul(attn_weights, v)  # [B, n_heads, H*W, d_values]
        
        output = output.contiguous().view(B, -1, H, W)  # [B, n_heads*d_values, H, W]
        output = self.out_proj(output)  # [B, H*W, C]

        return output

class ImageCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, d_queries, d_values, context_dim=None, use_flash_attention=True, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.context_dim = context_dim or hidden_size  # If None, use hidden_dim
        self.use_flash_attention = use_flash_attention

        self.q_proj = nn.Linear(hidden_size, n_heads*d_queries)
        self.k_proj = nn.Linear(self.context_dim, n_heads*d_queries)
        self.v_proj = nn.Linear(self.context_dim, n_heads*d_values)
        
        self.out_proj = nn.Linear(n_heads*d_values, hidden_size)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        self.apply(megatransformer_utils.transformer_weight_init())
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        tgt_seq_len = H * W
        BC, T, CC = context.size()
        ctxt_seq_len = T

        assert B == BC, f"Batch size mismatch: {B} vs {BC}"

        x = x.contiguous().view(B, C, tgt_seq_len)    # [B, C, tgt_seq_len]
        x = x.permute(0, 2, 1)  # [B, tgt_seq_len, C]

        q: torch.Tensor = self.q_proj(x)        # [B, tgt_seq_len, n_heads*d_queries]
        k: torch.Tensor = self.k_proj(context)  # [B, ctxt_seq_len, n_heads*d_queries]
        v: torch.Tensor = self.v_proj(context)  # [B, ctxt_seq_len, n_heads*d_values]
        
        q = q.view(B, tgt_seq_len, self.n_heads, self.d_queries).transpose(1, 2)   # [B, n_heads, tgt_seq_len, d_queries]
        k = k.view(B, ctxt_seq_len, self.n_heads, self.d_queries).transpose(1, 2)  # [B, n_heads, ctxt_seq_len, d_queries]
        v = v.view(B, ctxt_seq_len, self.n_heads, self.d_values).transpose(1, 2)   # [B, n_heads, ctxt_seq_len, d_values]
        
        output: torch.Tensor
        if self.use_flash_attention:
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )  # [B, n_heads, tgt_seq_len, d_values]
        else:
            scale = 1.0 / math.sqrt(self.d_queries)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, n_heads, tgt_seq_len, ctxt_seq_len]
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            output = torch.matmul(attn_weights, v)  # [B, n_heads, tgt_seq_len, d_values]
        
        output = output.transpose(1, 2).contiguous()  # [B, tgt_seq_len, n_heads, d_values]
        output = output.view(B, tgt_seq_len, self.n_heads*self.d_values)  # [B, tgt_seq_len, n_heads*d_values]
        
        output = self.out_proj(output)  # [B, tgt_seq_len, C]

        output = output.permute(0, 2, 1)  # [B, C, tgt_seq_len]

        # restore input shape by splitting the hidden dim into width and height
        output = output.view(B, C, H, W)

        return output

class ImageDiffusionSingleTaskModel(nn.Module):
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__()
        self.config = config

        self.text_recurrent = megatransformer_recurrent.MegaTransformerRecurrentCausalModel(config)
        self.diffuser = megatransformer_diffusion.GaussianDiffusion(
            config,
            hidden_size=config.hidden_size,
            activation=config.image_decoder_activation,
            scale_factor=(2, 2),
            stride=(2, 2),
            self_attn_class=ImageSelfAttentionBlock,
            cross_attn_class=ImageCrossAttentionBlock,
            norm_class=ImageRMSNorm,
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

    def gradient_checkpointing_enable(self, **kwargs):
        self.diffuser.unet.use_gradient_checkpointing = True

    def get_input_embeddings(self):
        return self.text_recurrent.wte
    
    def set_input_embeddings(self, new_embeddings):
        self.text_recurrent.wte = new_embeddings
    
    def forward(
        self,
        input_ids=None,
        image_raw_inputs=None,
        image_labels=None,
        attention_mask=None,
        past_key_values: list[megatransformer_utils.KVCache]=None,
        use_cache=False,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):
        # recurrent model (for enriching the text embeddings as conditioning for the image diffuser)
        recurrent_outputs = self.text_recurrent(
            input_ids, 
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        logits = recurrent_outputs.logits
        all_hidden_states = recurrent_outputs.hidden_states
        all_attentions = recurrent_outputs.attentions

        if len(image_labels.shape) == 5:
            # for singular image diffusion, example dimension is unnecessary
            B, E, C, H, W = image_labels.shape
            image_labels = image_labels.view(B, C, H, W)

        # reconstruction, loss
        image_outputs, loss = self.diffuser(
            image_labels,
            condition=logits,
        )

        if not return_dict:
            outputs = (
                image_outputs,
                past_key_values,
                all_hidden_states,
                all_attentions,
                recurrent_outputs.n_steps_no_grad,
                recurrent_outputs.k_steps_grad,
            )
            outputs = ((loss,) + outputs) if loss is not None else outputs
        else:
            outputs = megatransformer_utils.MegaTransformerMultimodalOutput(
                loss=loss,
                logits=image_outputs,
                image_raw_outputs=image_outputs,
                past_key_values=past_key_values,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
                n_steps_no_grad=recurrent_outputs.n_steps_no_grad,
                k_steps_grad=recurrent_outputs.k_steps_grad,
            )
        return outputs
    
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        num_samples=1,
        override_ddim_sampling_steps=None,
        past_key_values: list[megatransformer_utils.KVCache]=None,
        use_cache=False,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict_in_generate=None,
        diffusion_generator=None,
    ):

        # recurrent model (for enriching the text embeddings as conditioning for the image diffuser)
        text_embeddings = self.text_recurrent(
            input_ids, 
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )[0]

        if num_samples > 1 and text_embeddings.shape[0] == 1:
            # repeat text embeddings for each sample along the batch dimension if num_samples > 1
            text_embeddings = text_embeddings.repeat_interleave(num_samples, dim=0)

        pred_x0, noise_preds, x_start_preds = self.diffuser.sample(
            device=text_embeddings.device,
            condition=text_embeddings,
            batch_size=num_samples,
            image_size=64,
            return_intermediate=True,
            override_ddim_sampling_steps=override_ddim_sampling_steps,
            generator=diffusion_generator,
        )
        if return_dict_in_generate:
            return megatransformer_utils.MultimodalGenerationOutput(
                image_outputs=[pred_x0],
                intermediate_image_outputs=(noise_preds, x_start_preds),
            )
        return ([pred_x0], (noise_preds, x_start_preds))

def create_small_image_diffusion_model(tokenizer: PreTrainedTokenizer, max_position_embeddings, use_gradient_checkpointing):
    # uses a recurrent approach to emulate a deeper model (~317M params)
    config = megatransformer_utils.MegaTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=max_position_embeddings,
        n_layers=None,
        n_prelude_layers=2,
        n_recurrent_layers=2,
        n_coda_layers=1,
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

        image_decoder_model_channels=72,
        image_decoder_time_embedding_dim=72,
        image_decoder_num_res_blocks=5,
        image_decoder_betas_schedule="cosine",

        use_gradient_checkpointing=use_gradient_checkpointing,
    )

    config.text_prelude_config = config
    config.audio_prelude_config = config
    config.image_prelude_config = config

    config.text_coda_config = config
    config.audio_coda_config = config
    config.image_coda_config = config

    return ImageDiffusionSingleTaskModel(config)

def create_test_tiny_image_diffusion_model(tokenizer: PreTrainedTokenizer, max_position_embeddings, use_gradient_checkpointing):
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
        vocab_size=tokenizer.vocab_size,
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

        use_gradient_checkpointing=use_gradient_checkpointing,
    )

    config.text_prelude_config = config
    config.audio_prelude_config = config
    config.image_prelude_config = config

    config.text_coda_config = config
    config.audio_coda_config = config
    config.image_coda_config = config

    return ImageDiffusionSingleTaskModel(config)

lookup = {
    "small": create_small_image_diffusion_model,
    "test_tiny": create_test_tiny_image_diffusion_model,
}

def model_config_lookup(config):
    if config not in lookup:
        raise ValueError(f"Unknown model configuration: {config}")
    return lookup[config]
