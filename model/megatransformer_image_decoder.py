from diffusers import DiffusionPipeline
from pytorch_msssim import MS_SSIM
from transformers import PreTrainedTokenizer
from typing import Optional

from model import megatransformer_diffusion, megatransformer_modules, megatransformer_recurrent, megatransformer_text_encoder
import megatransformer_utils

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SDXLLatentLoss(torch.nn.Module):
    def __init__(self, vae_model):
        """
        Pure latent space loss for SDXL fine-tuning.
        
        This directly compares images in the latent space that SDXL's diffusion model
        operates in, which is ideal for ensuring semantic equivalence without
        enforcing pixel-level similarity.
        
        Args:
            vae_model: The VAE model from SDXL
        """
        super().__init__()
        self.vae = vae_model
        
        # Freeze VAE model
        self.vae.requires_grad_(False)
        self.vae.eval()
    
    def forward(self, generated_image, reference_image):
        # megatransformer_utils.print_debug_tensor("generated_image", generated_image)
        # megatransformer_utils.print_debug_tensor("reference_image", reference_image)
        """
        Compare images in VAE latent space.
        """
        with torch.no_grad():
            # Convert images to latent representations
            gen_latents = self.vae.encode(generated_image.to(self.vae.encoder.conv_in.weight.data.dtype)).latent_dist.sample()
            ref_latents = self.vae.encode(reference_image.to(self.vae.encoder.conv_in.weight.data.dtype)).latent_dist.sample()
            
            # Compute similarity in latent space
            latent_loss = torch.nn.functional.mse_loss(gen_latents, ref_latents)
            
            return latent_loss

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ms_ssim = MS_SSIM(data_range=1.0)

    def forward(self, reconstructed_image, image_labels):
        return 1 - self.ms_ssim(reconstructed_image, image_labels)

class PreTrainedImageDecoderWrapper(nn.Module):
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__()
        self.config = config

        self.pipe = DiffusionPipeline.from_pretrained(
            "lambdalabs/miniSD-diffusers",
            torch_dtype=torch.float16,
        ).to(torch.bfloat16)
        self.model = self.pipe.unet
        
        # self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        #     self.pipe.scheduler.config,
        #     algorithm_type="sde-dpmsolver++",
        #     use_karras_sigmas=True
        # )

        self.proj = nn.Linear(config.hidden_size, 768)
        self.pool_proj = nn.Linear(config.hidden_size, 1280)

        self.loss_fn = SDXLLatentLoss(self.pipe.vae)

        self.init_weights()

    def to(self, device):
        super().to(device)
        self.pipe.to(device)
        self.pipe.unet.to(device)
        self.pipe.vae.to(device)
        self.model.to(device)

    def init_layer_to_match_distribution(self, layer, mean, std):
        """Initialize a linear layer to match a specific mean and standard deviation."""
        # Initialize weights with normal distribution
        nn.init.normal_(layer.weight, mean=0, std=std)
        if layer.bias is not None:
            # Initialize biases to achieve the target mean
            nn.init.constant_(layer.bias, mean)

    def init_weights(self):
        clip_seq_std = 0.75
        clip_seq_mean = 0.0
        clip_pooled_std = 0.75
        clip_pooled_mean = 0.0
        self.init_layer_to_match_distribution(self.proj, clip_seq_mean, clip_seq_std)
        self.init_layer_to_match_distribution(self.pool_proj, clip_pooled_mean, clip_pooled_std)

    def forward(self, image_label, condition):
        # megatransformer_utils.print_debug_tensor("image_label", image_label)
        # megatransformer_utils.print_debug_tensor("condition", condition)
        prompt_embeds = self.proj(condition)
        # megatransformer_utils.print_debug_tensor("prompt_embeds", prompt_embeds)
        pooled_prompt_embeds = self.pool_proj(condition[:, 0, :])
        # megatransformer_utils.print_debug_tensor("pooled_prompt_embeds", pooled_prompt_embeds)

        bsz = image_label.shape[0]

        print(type(self.pipe.text_encoder))

        outputs = []
        for i in range(bsz):
            outputs.append(self.pipe(
                prompt_embeds=prompt_embeds[None, i],
                pooled_prompt_embeds=pooled_prompt_embeds[None, i],
                width=self.config.image_size,
                height=self.config.image_size,
                guidance_scale=7.0,
                num_inference_steps=20,
                output_type="pt",
                disable_progress_bar=True,
            ).images[0])

        outputs = torch.stack(outputs, dim=0).to(condition.dtype)

        loss = self.loss_fn(outputs, image_label)
        return outputs, loss

    def sample(self, condition, batch_size, image_size, **kwargs):
        megatransformer_utils.print_debug_tensor("condition", condition)
        prompt_embeds = self.proj(condition)
        megatransformer_utils.print_debug_tensor("prompt_embeds", prompt_embeds)
        pooled_prompt_embeds = self.pool_proj(condition[:, 0, :])
        megatransformer_utils.print_debug_tensor("pooled_prompt_embeds", pooled_prompt_embeds)

        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(batch_size, 1)

        megatransformer_utils.print_debug_tensor("repeated_prompt_embeds", prompt_embeds)
        megatransformer_utils.print_debug_tensor("repeated_pooled_prompt_embeds", pooled_prompt_embeds)

        # Generate images
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            width=image_size,
            height=image_size,
            guidance_scale=7.0,
            num_inference_steps=20,
            output_type="pt"
        ).images[0]

        return images, None, None

class ImageRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        # x: [B, H, W, C]
        x = x.permute(0, 3, 1, 2).contiguous()  # to [B, C, H, W]
        x_norm = F.normalize(x, dim=1)
        shift = x.shape[-1] ** 0.5
        norm = x_norm * self.g * shift
        norm = norm.permute(0, 2, 3, 1).contiguous()  # back to [B, H, W, C]
        return norm

class ImageSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, d_queries, d_values, use_flash_attention=True, dropout_p=0.1, is_linear_attention=False):
        super().__init__()
        self.hidden_dim = hidden_size
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.use_flash_attention = use_flash_attention
        self.dropout_p = dropout_p
        self.is_linear_attention = is_linear_attention

        self.q_proj = nn.Conv2d(hidden_size, d_queries * n_heads, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(hidden_size, d_queries * n_heads, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(hidden_size, d_values * n_heads, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(self.d_values * n_heads, hidden_size, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout_p)

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
        
        q = q.view(B, self.n_heads, self.d_queries, -1).transpose(-2, -1)  # [B, n_heads, H*W, d_queries]
        k = k.view(B, self.n_heads, self.d_queries, -1).transpose(-2, -1)  # [B, n_heads, H*W, d_queries]
        v = v.view(B, self.n_heads, self.d_values, -1).transpose(-2, -1)  # [B, n_heads, H*W, d_values]

        if self.is_linear_attention:
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-2)
        
        output: torch.Tensor
        if self.use_flash_attention:
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False
            )  # [B, n_heads, H*W, d_values]
        else:
            if self.is_linear_attention:
                # Linear attention: (Q^T)·(K·V) - more efficient for long sequences
                kv = torch.matmul(k.transpose(-2, -1), v)  # [B, n_heads, d_queries, d_values]
                output = torch.matmul(q, kv)  # [B, n_heads, seq_len, d_values]
            else:
                # Standard dot-product attention
                scale = 1.0 / math.sqrt(self.d_queries)
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, n_heads, seq_len, seq_len]
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                output = torch.matmul(attn_weights, v)  # [B, n_heads, seq_len, d_values]
        
        # Reshape back to image format
        output = output.transpose(2, 3).contiguous()  # [B, n_heads, d_values, seq_len]
        output = output.view(B, self.n_heads * self.d_values, H, W)  # [B, n_heads*d_values, H, W]
        output = self.out_proj(output)  # [B, C, H, W]

        return output

class PatchSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, n_heads=8, d_queries=64, d_values=64, use_flash_attention=False, dropout_p=0.1, is_linear_attention=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.use_flash_attention = use_flash_attention
        self.dropout_p = dropout_p
        self.is_linear_attention = is_linear_attention

        self.qk_pos_embed = megatransformer_utils.create_sinusoidal_2d_pos_encoding(n_heads*d_queries)
        self.v_pos_embed = megatransformer_utils.create_sinusoidal_2d_pos_encoding(n_heads*d_values)

        # patch embeddings
        self.q_proj = nn.Conv2d(hidden_size, n_heads*d_queries, kernel_size=16, stride=16, bias=False)
        self.k_proj = nn.Conv2d(hidden_size, n_heads*d_queries, kernel_size=16, stride=16, bias=False)
        self.v_proj = nn.Conv2d(hidden_size, n_heads*d_values, kernel_size=16, stride=16, bias=False)
        self.out_proj = nn.ConvTranspose2d(n_heads*d_values, hidden_size, kernel_size=16, stride=16)

        self.dropout = nn.Dropout(dropout_p)

        self._init_weights()

    def _init_weights(self):
        self.apply(megatransformer_utils.conv2d_weight_init())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()

        # megatransformer_utils.print_debug_tensor("x", x)

        q: torch.Tensor = self.q_proj(x)  # [B, n_heads*d_queries, H, W]
        q = q + self.qk_pos_embed(q)  # positional embedding after patch embeddings/query extraction

        k: torch.Tensor = self.k_proj(x)  # [B, n_heads*d_queries, H, W]
        k = k + self.qk_pos_embed(k)  # positional embedding after patch embeddings/query extraction

        v: torch.Tensor = self.v_proj(x)  # [B, n_heads*d_queries, H, W]
        v = v + self.v_pos_embed(v)  # positional embedding after patch embeddings/query extraction

        q = q.view(B, self.n_heads, self.d_queries, -1).transpose(-2, -1)  # [B, n_heads, seq_len, d_queries]
        k = k.view(B, self.n_heads, self.d_queries, -1).transpose(-2, -1)  # [B, n_heads, seq_len, d_queries]
        v = v.view(B, self.n_heads, self.d_values, -1).transpose(-2, -1)   # [B, n_heads, seq_len, d_values]
        
        output: torch.Tensor
        if self.use_flash_attention:
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False
            )  # [B, n_heads, seq_len, d_values]
        else:
            if self.is_linear_attention:
                # Linear attention: (Q^T)·(K·V) - more efficient for long sequences
                kv = torch.matmul(k.transpose(-2, -1), v)  # [B, n_heads, d_queries, d_values]
                output = torch.matmul(q, kv)  # [B, n_heads, seq_len, d_values]
            else:
                # Standard dot-product attention
                scale = 1.0 / math.sqrt(self.d_queries)
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, n_heads, seq_len, seq_len]
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                output = torch.matmul(attn_weights, v)  # [B, n_heads, seq_len, d_values]

        # megatransformer_utils.print_debug_tensor("output", output)

        # Reshape back to image format
        output = output.transpose(-2, -1).contiguous()  # [B, n_heads, d_values, seq_len]

        # megatransformer_utils.print_debug_tensor("output_tranposed", output)

        seq_len = output.size(-1)

        H = W = math.floor(math.sqrt(seq_len))  # Assuming square patches

        output = output.view(B, self.n_heads * self.d_values, H, W)  # [B, n_heads*d_values, H, W]
        output = self.out_proj(output)  # [B, C, H, W]

        return output

class ImageCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size, n_heads=8, d_queries=64, d_values=64, context_dim=None, use_flash_attention=False, dropout_p=0.1, is_linear_attention=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.context_dim = context_dim or hidden_size  # If None, use hidden_dim
        self.use_flash_attention = use_flash_attention
        self.dropout_p = dropout_p
        self.is_linear_attention = is_linear_attention

        self.q_pos_embed = megatransformer_utils.create_sinusoidal_2d_pos_encoding(n_heads*d_queries)

        self.q_proj = nn.Conv2d(hidden_size, n_heads*d_queries, kernel_size=16, stride=16, bias=False)
        self.k_proj = nn.Linear(self.context_dim, n_heads*d_queries, bias=False)
        self.v_proj = nn.Linear(self.context_dim, n_heads*d_values, bias=False)
        self.out_proj = nn.ConvTranspose2d(n_heads*d_values, hidden_size, kernel_size=16, stride=16)

        self.dropout = nn.Dropout(dropout_p)

        self._init_weights()

    def _init_weights(self):
        self.apply(megatransformer_utils.transformer_weight_init())

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        BC, T, CC = context.size()
        ctxt_seq_len = T

        assert B == BC, f"Batch size mismatch: {B} vs {BC}"

        q: torch.Tensor = self.q_proj(x)        # [B, n_heads*d_queries, H, W]

        q = q + self.q_pos_embed(q) # positional embedding after patch embeddings/query extraction

        k: torch.Tensor = self.k_proj(context)  # [B, ctxt_seq_len, n_heads*d_queries]
        v: torch.Tensor = self.v_proj(context)  # [B, ctxt_seq_len, n_heads*d_values]

        q = q.view(B, self.n_heads, self.d_queries, -1).transpose(-2, -1)  # [B, n_heads, seq_len, d_queries]
        k = k.view(B, ctxt_seq_len, self.n_heads, self.d_queries).transpose(1, 2)  # [B, n_heads, ctxt_seq_len, d_queries]
        v = v.view(B, ctxt_seq_len, self.n_heads, self.d_values).transpose(1, 2)   # [B, n_heads, ctxt_seq_len, d_values]
        
        output: torch.Tensor
        if self.use_flash_attention:
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False
            )  # [B, n_heads, seq_len, d_values]
        else:
            if self.is_linear_attention:
                # Linear attention: (Q^T)·(K·V) - more efficient for long sequences
                kv = torch.matmul(k.transpose(-2, -1), v)  # [B, n_heads, d_queries, d_values]
                output = torch.matmul(q, kv)  # [B, n_heads, seq_len, d_values]
            else:
                # Standard dot-product attention
                scale = 1.0 / math.sqrt(self.d_queries)
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, n_heads, seq_len, seq_len]
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                output = torch.matmul(attn_weights, v)  # [B, n_heads, seq_len, d_values]

        # Reshape back to image format
        output = output.transpose(-2, -1).contiguous()  # [B, n_heads, d_values, seq_len]

        seq_len = output.size(-1)

        H = W = math.floor(math.sqrt(seq_len))  # Assuming square patches

        output = output.view(B, self.n_heads * self.d_values, H, W)  # [B, n_heads*d_values, H, W]
        output = self.out_proj(output)  # [B, C, H, W]

        return output


class ImageCrossAttentionBlockSimple(nn.Module):
    """
    Simpler cross-attention block for image data that works with any spatial size.
    Uses 1x1 convolutions for query projection (instead of large patch embeddings).
    This is the standard approach used in UNet-based diffusion models like Stable Diffusion.
    """
    def __init__(self, hidden_size, n_heads=8, d_queries=64, d_values=64, context_dim=None, use_flash_attention=True, dropout_p=0.1, is_linear_attention=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.context_dim = context_dim or hidden_size
        self.use_flash_attention = use_flash_attention
        self.dropout_p = dropout_p
        self.is_linear_attention = is_linear_attention

        # Use 1x1 conv for query (works with any spatial size)
        self.q_proj = nn.Conv2d(hidden_size, n_heads * d_queries, kernel_size=1, bias=False)
        # Linear projections for context (key/value)
        self.k_proj = nn.Linear(self.context_dim, n_heads * d_queries, bias=False)
        self.v_proj = nn.Linear(self.context_dim, n_heads * d_values, bias=False)
        # 1x1 conv for output projection
        self.out_proj = nn.Conv2d(n_heads * d_values, hidden_size, kernel_size=1)

        self.dropout = nn.Dropout(dropout_p)

        self._init_weights()

    def _init_weights(self):
        self.apply(megatransformer_utils.transformer_weight_init())

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Cross-attention between image features and context (e.g., text embeddings).

        Args:
            x: Image features [B, C, H, W]
            context: Context embeddings [B, T, D] (e.g., text embeddings)

        Returns:
            output: [B, C, H, W] with cross-attended features
        """
        B, C, H, W = x.size()
        BC, T, CC = context.size()

        assert B == BC, f"Batch size mismatch: {B} vs {BC}"

        # Query from image features
        q = self.q_proj(x)  # [B, n_heads*d_queries, H, W]
        q = q.view(B, self.n_heads, self.d_queries, -1).transpose(-2, -1)  # [B, n_heads, H*W, d_queries]

        # Key and value from context
        k = self.k_proj(context)  # [B, T, n_heads*d_queries]
        v = self.v_proj(context)  # [B, T, n_heads*d_values]

        k = k.view(B, T, self.n_heads, self.d_queries).transpose(1, 2)  # [B, n_heads, T, d_queries]
        v = v.view(B, T, self.n_heads, self.d_values).transpose(1, 2)   # [B, n_heads, T, d_values]

        output: torch.Tensor
        if self.use_flash_attention:
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False
            )  # [B, n_heads, H*W, d_values]
        else:
            if self.is_linear_attention:
                # Linear attention
                kv = torch.matmul(k.transpose(-2, -1), v)  # [B, n_heads, d_queries, d_values]
                output = torch.matmul(q, kv)  # [B, n_heads, H*W, d_values]
            else:
                # Standard attention
                scale = 1.0 / math.sqrt(self.d_queries)
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, n_heads, H*W, T]
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                output = torch.matmul(attn_weights, v)  # [B, n_heads, H*W, d_values]

        # Reshape back to image format
        output = output.transpose(2, 3).contiguous()  # [B, n_heads, d_values, H*W]
        output = output.view(B, self.n_heads * self.d_values, H, W)  # [B, n_heads*d_values, H, W]
        output = self.out_proj(output)  # [B, C, H, W]

        return output


class ImageVAE(nn.Module):
    def __init__(self, config, encoder, decoder, kl_beta=1e-4):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.decoder = decoder

        self.kl_beta = kl_beta

        self.mu_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU()
        )

        self.logvar_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, image_labels=None, condition=None):
        if image_labels is not None:
            mu, logvar = self.encoder(image_labels)
            encoder_z = self.reparameterize(mu, logvar)
            if condition is not None:
                pooled_embeds = torch.mean(condition, dim=1)
                mu_pooled_embeds = self.mu_proj(pooled_embeds)
                logvar_pooled_embeds = self.logvar_proj(pooled_embeds)
                pooled_embeds_z = self.reparameterize(mu_pooled_embeds, logvar_pooled_embeds)
            else:
                pooled_embeds_z = None
        else:
            assert condition is not None, "Condition must be provided if image_labels is None"
            encoder_z = None
            pooled_embeds = torch.mean(condition, dim=1)
            mu_pooled_embeds = self.mu_proj(pooled_embeds)
            logvar_pooled_embeds = self.logvar_proj(pooled_embeds)
            pooled_embeds_z = self.reparameterize(mu_pooled_embeds, logvar_pooled_embeds)
        
        if encoder_z is not None:
            reconstructed_image, reconstruction_loss, ssim_loss = self.decoder(encoder_z, condition=condition, image_labels=image_labels)
        else:
            reconstructed_image = None
            reconstruction_loss = None
            ssim_loss = None

        if pooled_embeds_z is not None:
            conditioned_image, _, _ = self.decoder(pooled_embeds_z, condition=condition)
        else:
            conditioned_image = None

        all_losses = []
        if reconstruction_loss is not None:
            all_losses.append(reconstruction_loss)
        if ssim_loss is not None:
            all_losses.append(ssim_loss)
        
        if encoder_z is not None:
            kl_loss = self.kl_beta * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
            all_losses.append(kl_loss)

            if pooled_embeds_z is not None:
                # loss between mu and logvar for encoder and condition pooler
                mu_mse = F.mse_loss(mu, mu_pooled_embeds)
                logvar_mse = F.mse_loss(logvar, logvar_pooled_embeds)
                all_losses.append(mu_mse)
                all_losses.append(logvar_mse)

        if len(all_losses) > 0:
            # add device num dimension for gather to concat across; requires custom trainer for handling
            all_losses = torch.stack(all_losses).unsqueeze(0)

        return all_losses, reconstructed_image, conditioned_image
    
    def sample(self, device, batch_size: int, condition: Optional[torch.Tensor]=None, **kwargs) -> torch.Tensor:
        if condition is None:
            condition = torch.randn(batch_size, self.config.hidden_size, device=device)
        
        pooled_embeds = torch.mean(condition, dim=1)
        mu_pooled_embeds = self.mu_proj(pooled_embeds)
        logvar_pooled_embeds = self.logvar_proj(pooled_embeds)
        pooled_embeds_z = self.reparameterize(mu_pooled_embeds, logvar_pooled_embeds)

        conditioned_image, _, _ = self.decoder(pooled_embeds_z, condition=condition)

        return None, conditioned_image, None

class ImageVAEEncoder(nn.Module):
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig, features=[16, 32, 64, 128, 256, 512], self_attns=None, dropout_p=0.1):
        super().__init__()
        self.config = config

        if self_attns is None:
            self_attns = [False] * len(features)

        self.expected_image_sizes = [config.image_size // (2 ** (i+1)) for i in range(0, len(features))]

        def conv_downsample_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Dropout2d(dropout_p),
                nn.BatchNorm2d(out_channels)
            )

        self.initial_conv = conv_downsample_block(3, features[0])

        self.convs = nn.ModuleList([conv_downsample_block(in_f, out_f) for in_f, out_f in zip(features[:-1], features[1:])])
        self.self_attns = nn.ModuleList([PatchSelfAttentionBlock(out_f, config.image_decoder_down_block_self_attn_n_heads, dropout_p=dropout_p) if use else nn.Identity() for out_f, use in zip(features, self_attns)])
        self.pool = megatransformer_modules.AvgMaxAdaptivePool2d()

        # AvgMaxAdaptivePool2d will output a tensor of shape (B, features[-1]*2, 1, 1) due to concatentating max and avg pools
        self.fc_mu = nn.Linear(features[-1]*2, config.hidden_size)
        self.fc_logvar = nn.Linear(features[-1]*2, config.hidden_size)

    def _init_weights(self):
        self.apply(megatransformer_utils.conv2d_weight_init())
        self.fc_mu.weight.data.normal_(mean=0.0, std=0.02)
        self.fc_logvar.weight.data.normal_(mean=0.0, std=0.02)
        self.fc_mu.bias.data.zero_()
        self.fc_logvar.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        x = self.initial_conv(x)

        if self.self_attns[0] is not None and not isinstance(self.self_attns[0], nn.Identity):
            x = x + self.self_attns[0](x)

        assert x.shape[2:] == (self.expected_image_sizes[0], self.expected_image_sizes[0]), f"Expected {self.expected_image_sizes[0]}x{self.expected_image_sizes[0]} but got {x.shape[2:]} from layer {self.initial_conv}"

        for conv, self_attn, expected_image_size in zip(self.convs, self.self_attns[1:], self.expected_image_sizes[1:]):
            x = conv(x)

            assert x.shape[2:] == (expected_image_size, expected_image_size), f"Expected {expected_image_size}x{expected_image_size} but got {x.shape[2:]} from layer {conv}"

            if self_attn is not None and not isinstance(self_attn, nn.Identity):
                x = x + self_attn(x)

        pooled_features = self.pool(x)
        flattened_features = pooled_features.view(pooled_features.size(0), -1)
        mu = self.fc_mu(flattened_features)
        logvar = self.fc_logvar(flattened_features)
        return mu, logvar

class ImageVAEDecoder(nn.Module):
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig, features=[512, 256, 128, 64, 32, 16], self_attns=None, cross_attns=None, self_attn_patch_sizes=None, dropout_p=0.1, recon_beta=100.0, ssim_beta=50.0):
        super().__init__()
        self.config = config

        self.recon_beta = recon_beta
        self.ssim_beta = ssim_beta

        if self_attns is None:
            self_attns = [False] * len(features)

        if cross_attns is None:
            self_attns = [True] * len(features)

        if self_attn_patch_sizes is None:
            self_attn_patch_sizes = [4] * len(self_attns)

        self.expected_image_sizes = [config.image_size // (2 ** i) for i in range(len(features) - 1, -1, -1)]
        print(f"expected image sizes: {self.expected_image_sizes}")

        print(f"creating decoder with {features} features, {self_attns} self-attentions, {cross_attns} cross-attentions, {self_attn_patch_sizes} self-attn patch sizes")

        def conv_upsample_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GELU(),
                nn.Dropout2d(dropout_p),
                nn.BatchNorm2d(out_channels)
            )
        
        def conv_refine_block(channels):
            return nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Dropout2d(dropout_p),
                nn.BatchNorm2d(channels)
            )

        self.fc = nn.Linear(config.hidden_size, features[0] * self.expected_image_sizes[0] * self.expected_image_sizes[0])
        self.unflatten = nn.Unflatten(1, (features[0], self.expected_image_sizes[0], self.expected_image_sizes[0]))
        self.convs = nn.ModuleList([conv_upsample_block(in_f, out_f) for in_f, out_f in zip(features[:-1], features[1:])])
        self.self_attns = nn.ModuleList([
            PatchSelfAttentionBlock(
                out_f,
                n_heads=config.image_decoder_up_block_self_attn_n_heads,
                dropout_p=dropout_p
            )
            if use else nn.Identity() 
            for out_f, use in zip(features, self_attns)
        ])
        self.cross_attns = nn.ModuleList([
            ImageCrossAttentionBlock(
                out_f,
                n_heads=config.image_decoder_cross_attn_n_heads,
                context_dim=config.hidden_size,
                dropout_p=dropout_p
            )
            if use else nn.Identity()
            for out_f, use in zip(features, cross_attns)
        ])
        self.refines = nn.ModuleList([conv_refine_block(out_f) for out_f in features[1:]])

        self.final_conv = nn.Conv2d(features[-1], 3, kernel_size=3, stride=1, padding=1)
        self.clamp_act = nn.Sigmoid()

        self.vae_loss_fn = F.mse_loss
        self.perceptual_loss = PerceptualLoss()

    def forward(self, z, condition=None, image_labels=None, speaker_embedding=None):
        features = self.fc(z)
        features = self.unflatten(features)

        assert features.shape[2:] == (self.expected_image_sizes[0], self.expected_image_sizes[0]), f"Expected {self.expected_image_sizes[0]}x{self.expected_image_sizes[0]} but got {features.shape[2:]} from layer {self.unflatten}"

        assert len(self.convs) == len(self.self_attns) - 1 == len(self.cross_attns) - 1 == len(self.refines), f"Mismatch in number of layers: {len(self.convs)}, {len(self.self_attns)}, {len(self.cross_attns)}, {len(self.refines)}"

        if self.self_attns[0] is not None and not isinstance(self.self_attns[0], nn.Identity):
            features = features + self.self_attns[0](features)
        if condition is not None and self.cross_attns[0] is not None and not isinstance(self.cross_attns[0], nn.Identity):
            features = features + self.cross_attns[0](features, condition)

        for layer, self_attn, cross_attn, refine, expected_image_size in zip(self.convs, self.self_attns[1:], self.cross_attns[1:], self.refines, self.expected_image_sizes[1:]):
            features = layer(features)

            assert features.shape[2:] == (expected_image_size, expected_image_size), f"Expected {expected_image_size}x{expected_image_size} but got {features.shape[2:]} from layer {layer}"

            if self_attn is not None and not isinstance(self_attn, nn.Identity):
                features = features + self_attn(features)
            if condition is not None and cross_attn is not None and not isinstance(cross_attn, nn.Identity):
                features = features + cross_attn(features, condition)
            if refine is not None:
                features = features + refine(features)

        reconstructed_image = self.final_conv(features)
        reconstructed_image = self.clamp_act(reconstructed_image)

        # output is on [-1, 1] range; scale to [0, 1]
        if isinstance(self.clamp_act, nn.Tanh):
            reconstructed_image = (reconstructed_image + 1) / 2
            reconstructed_image = torch.clamp(reconstructed_image, 0, 1)

        if image_labels is not None:
            assert reconstructed_image.shape == image_labels.shape, f"Shape mismatch: {reconstructed_image.shape} vs {image_labels.shape}"
            recon_loss = self.recon_beta * self.vae_loss_fn(reconstructed_image, image_labels)
            perceptual_loss = self.ssim_beta * self.perceptual_loss(reconstructed_image, image_labels)
            return reconstructed_image, recon_loss, perceptual_loss
        return reconstructed_image, None, None

class ImageReconstructionSingleTaskModel(nn.Module):
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig, text_recurrent, image_recon):
        super().__init__()
        self.config = config

        self.text_encoder = text_recurrent
        self.image_recon = image_recon

    def gradient_checkpointing_enable(self, **kwargs):
        self.image_recon.unet.use_gradient_checkpointing = True

    def get_input_embeddings(self):
        return self.text_encoder.get_input_embeddings()
    
    def set_input_embeddings(self, new_embeddings):
        self.text_encoder.set_input_embeddings(new_embeddings)
    
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

        text_outputs = self.text_encoder(
            input_ids, 
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        logits = text_outputs['logits']
        all_hidden_states = text_outputs['hidden_states']
        all_attentions = text_outputs['attentions']

        if len(image_labels.shape) == 5:
            # for singular image diffusion, example dimension is unnecessary
            B, E, C, H, W = image_labels.shape
            image_labels = image_labels.view(B, C, H, W)

        loss, reconstruction, conditional_generation = self.image_recon(
            image_labels,
            condition=logits,
        )

        if not return_dict:
            outputs = (
                reconstruction,
                past_key_values,
                all_hidden_states,
                all_attentions,
                text_outputs['n_steps_no_grad'] if 'n_steps_no_grad' in text_outputs else None,
                text_outputs['k_steps_grad'] if 'k_steps_grad' in text_outputs else None,
            )
            outputs = ((loss,) + outputs) if loss is not None else outputs
        else:
            outputs = megatransformer_utils.MegaTransformerMultimodalOutput(
                loss=loss,
                logits=reconstruction,
                image_raw_outputs=[reconstruction, conditional_generation],
                past_key_values=past_key_values,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
                n_steps_no_grad=text_outputs['n_steps_no_grad'] if 'n_steps_no_grad' in text_outputs else None,
                k_steps_grad=text_outputs['k_steps_grad'] if 'k_steps_grad' in text_outputs else None,
            )
        return outputs
    
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        num_samples=1,
        image_size=64,
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
        text_embeddings = self.text_encoder(
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

        if isinstance(self.image_recon, ImageVAE):
            _, conditioned_image_recon, _ = self.image_recon.sample(
                device=text_embeddings.device,
                condition=text_embeddings,
                batch_size=num_samples,
                image_size=image_size,
                return_intermediate=True,
                override_ddim_sampling_steps=override_ddim_sampling_steps,
                generator=diffusion_generator,
            )
            noise_preds = None
            x_start_preds = None
        else:
            conditioned_image_recon, noise_preds, x_start_preds = self.image_recon.sample(
                device=text_embeddings.device,
                condition=text_embeddings,
                batch_size=num_samples,
                image_size=image_size,
                return_intermediate=True,
                override_ddim_sampling_steps=override_ddim_sampling_steps,
                generator=diffusion_generator,
            )
        if return_dict_in_generate:
            return megatransformer_utils.MultimodalGenerationOutput(
                image_outputs=[conditioned_image_recon],
                intermediate_image_outputs=(noise_preds, x_start_preds),
            )
        return ([conditioned_image_recon], (noise_preds, x_start_preds))

def gaussian_diffusion_image_model(config: megatransformer_utils.MegaTransformerConfig):
    return megatransformer_diffusion.GaussianDiffusion(
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
        unet_dropout=config.image_decoder_unet_dropout_p,
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

    return ImageReconstructionSingleTaskModel(config, gaussian_diffusion_image_model(config))

def create_test_tiny_image_diffusion_model(tokenizer: PreTrainedTokenizer, max_position_embeddings, use_gradient_checkpointing):
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            megatransformer_utils.BEGIN_AUDIO_TOKEN,
            megatransformer_utils.END_AUDIO_TOKEN,
            megatransformer_utils.BEGIN_IMAGE_TOKEN,
            megatransformer_utils.END_IMAGE_TOKEN,
            megatransformer_utils.BEGIN_VOICE_TOKEN,
            megatransformer_utils.END_VOICE_TOKEN,
        ]
    })

    begin_audio_token_id = tokenizer.convert_tokens_to_ids(megatransformer_utils.BEGIN_AUDIO_TOKEN)
    end_audio_token_id = tokenizer.convert_tokens_to_ids(megatransformer_utils.END_AUDIO_TOKEN)
    begin_image_token_id = tokenizer.convert_tokens_to_ids(megatransformer_utils.BEGIN_IMAGE_TOKEN)
    end_image_token_id = tokenizer.convert_tokens_to_ids(megatransformer_utils.END_IMAGE_TOKEN)
    begin_voice_token_id = tokenizer.convert_tokens_to_ids(megatransformer_utils.BEGIN_VOICE_TOKEN)
    end_voice_token_id = tokenizer.convert_tokens_to_ids(megatransformer_utils.END_VOICE_TOKEN)

    print(begin_audio_token_id, end_audio_token_id, begin_image_token_id, end_image_token_id, begin_voice_token_id, end_voice_token_id)

    # uses a recurrent approach to emulate a deeper model (~M params)
    config = megatransformer_utils.MegaTransformerConfig(
        vocab_size=tokenizer.vocab_size + 6,
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
        begin_voice_token_id=begin_voice_token_id,
        end_voice_token_id=end_voice_token_id,

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

    return ImageReconstructionSingleTaskModel(config, gaussian_diffusion_image_model(config))

def create_conv_transpose_model(tokenizer: PreTrainedTokenizer, text_model_cls, max_position_embeddings, use_gradient_checkpointing):
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            megatransformer_utils.BEGIN_AUDIO_TOKEN,
            megatransformer_utils.END_AUDIO_TOKEN,
            megatransformer_utils.BEGIN_IMAGE_TOKEN,
            megatransformer_utils.END_IMAGE_TOKEN,
            megatransformer_utils.BEGIN_VOICE_TOKEN,
            megatransformer_utils.END_VOICE_TOKEN,
        ]
    })

    begin_audio_token_id = tokenizer.convert_tokens_to_ids(megatransformer_utils.BEGIN_AUDIO_TOKEN)
    end_audio_token_id = tokenizer.convert_tokens_to_ids(megatransformer_utils.END_AUDIO_TOKEN)
    begin_image_token_id = tokenizer.convert_tokens_to_ids(megatransformer_utils.BEGIN_IMAGE_TOKEN)
    end_image_token_id = tokenizer.convert_tokens_to_ids(megatransformer_utils.END_IMAGE_TOKEN)
    begin_voice_token_id = tokenizer.convert_tokens_to_ids(megatransformer_utils.BEGIN_VOICE_TOKEN)
    end_voice_token_id = tokenizer.convert_tokens_to_ids(megatransformer_utils.END_VOICE_TOKEN)

    print(begin_audio_token_id, end_audio_token_id, begin_image_token_id, end_image_token_id, begin_voice_token_id, end_voice_token_id)

    # uses a recurrent approach to emulate a deeper model (~M params)
    config = megatransformer_utils.MegaTransformerConfig(
        vocab_size=tokenizer.vocab_size + 6,
        max_position_embeddings=max_position_embeddings,
        n_layers=None,
        hidden_size=256,
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
        begin_voice_token_id=begin_voice_token_id,
        end_voice_token_id=end_voice_token_id,

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

    features = [32, 64, 128, 256, 512, 1024, 2048]

    return ImageReconstructionSingleTaskModel(
        config,
        text_model_cls(config),
        ImageVAE(
            config,
            ImageVAEEncoder(
                config,
                features=features,
                self_attns=[True, True, True] + ([False] * (len(features) - 3)),
            ),
            ImageVAEDecoder(
                config,
                features=list(reversed(features)),
                self_attns=[False, False, False, False] + ([True] * (len(features) - 4)),
                cross_attns=[False, False, False, False] + ([True] * (len(features) - 4)),
            )
        )
    )

def create_test_conv_transpose_model(tokenizer: PreTrainedTokenizer, max_position_embeddings, use_gradient_checkpointing):
    return create_conv_transpose_model(
        tokenizer,
        megatransformer_recurrent.MegaTransformerRecurrentCausalModel,
        max_position_embeddings,
        use_gradient_checkpointing
    )

def create_test_t5_conv_transpose_model(tokenizer: PreTrainedTokenizer, max_position_embeddings, use_gradient_checkpointing):
    return create_conv_transpose_model(
        tokenizer,
        megatransformer_text_encoder.T5TextEncoder,
        max_position_embeddings,
        use_gradient_checkpointing
    )

lookup = {
    "small": create_small_image_diffusion_model,
    "test_tiny": create_test_tiny_image_diffusion_model,
    "test_conv_transpose": create_test_conv_transpose_model,
    "test_t5_conv_transpose": create_test_t5_conv_transpose_model,
}

def model_config_lookup(config):
    if config not in lookup:
        raise ValueError(f"Unknown model configuration: {config}")
    return lookup[config]
