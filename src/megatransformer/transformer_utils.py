from positional_encodings.torch_encodings import PositionalEncoding2D
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from typing import Union

from . import swiglu, megatransformer, multihead_attn, positionwise_fcn, grouped_query_attn, phi3_mlp, millions_moe, infinite_multihead_attn

import math
import torch

def get_activation_function(activation_function_name):
    if activation_function_name == 'relu':
        return nn.ReLU
    elif activation_function_name == 'gelu':
        return nn.GELU
    elif activation_function_name == 'elu':
        return nn.ELU
    elif activation_function_name == 'selu':
        return nn.SELU
    elif activation_function_name == 'prelu':
        return nn.PReLU
    elif activation_function_name == 'leaky_relu':
        return nn.LeakyReLU
    elif activation_function_name == 'silu':
        return nn.SiLU
    elif activation_function_name == 'none':
        return nn.Identity
    else:
        raise Exception(f"Unknown activation function {activation_function_name}")

def create_activation_function(d_in, activation_function_name):
    if activation_function_name == 'swiglu':
        return swiglu.SwiGLU(d_in)
    return get_activation_function(activation_function_name)()

def get_buffered_positional_encoding(device, d_model, positional_encoding_dim: int, maxlen=100, num_dims=1):
    if num_dims == 1:
        positional_encoding = torch.zeros((maxlen, d_model)) # (max_length, d_model)
        for i in range(maxlen):
            for k in range(d_model):
                if k % 2 == 0:
                    positional_encoding[i, k] = math.sin(i / math.pow(10000, k / d_model))
                else:
                    positional_encoding[i, k] = math.cos(i / math.pow(10000, (k - 1) / d_model))
        positional_encoding = positional_encoding.unsqueeze(0) # (1, max_length, d_model)
    elif num_dims == 2:
        positional_encoding_2d = PositionalEncoding2D(positional_encoding_dim).to(device)
        positional_encoding = torch.zeros((1, maxlen, maxlen, positional_encoding_dim))
        positional_encoding = positional_encoding_2d(positional_encoding.to(device))
    return positional_encoding  # (1, max_length, d_model) or (1, max_length, max_length, d_model)

def get_tensor_positional_encoding(device, d_model: int, positional_encoding_dim: int, learnable_positional_encoding: bool, maxlen: int):
    positional_encoding = get_buffered_positional_encoding(
        device,
        d_model,
        positional_encoding_dim,
        maxlen=maxlen + 1,
    ).to(device)
    positional_encoding.requires_grad = learnable_positional_encoding
    return positional_encoding

def sanitize_model(model):
    if hasattr(model, '_orig_mod'):
        return sanitize_model(model._orig_mod)
    
    return model

def init_weights(model: nn.Module,
                 d_model: int,
                 init_weights_from: str = 'glorot_uniform',
                 init_weights_gain: float = 1.0,
                 tie_embeddings=False):
    if isinstance(model, megatransformer.MegaTransformer):
        init_weights(model.encoder, d_model, init_weights_from, init_weights_gain, tie_embeddings)
        init_weights(model.decoder, d_model, init_weights_from, init_weights_gain, tie_embeddings)
    elif isinstance(model, megatransformer.Encoder):
        encoder: megatransformer.Encoder = model
        init_weights(encoder.embed_tokens, d_model, init_weights_from, init_weights_gain, tie_embeddings)
        for encoder_layer in encoder.encoder_layers:
            init_weights(encoder_layer, d_model, init_weights_from, init_weights_gain, tie_embeddings)
        init_weights(encoder.post_encoder_norm, d_model, init_weights_from, init_weights_gain, tie_embeddings)
    elif isinstance(model, megatransformer.EncoderLayer):
        encoder_layer: megatransformer.EncoderLayer = model
        init_weights(encoder_layer.self_attn, d_model, init_weights_from, init_weights_gain, tie_embeddings)
        init_weights(encoder_layer.ffn, d_model, init_weights_from, init_weights_gain, tie_embeddings)
    elif isinstance(model, megatransformer.Decoder):
        decoder: megatransformer.Decoder = model
        init_weights(decoder.embed_tokens, d_model, init_weights_from, init_weights_gain, tie_embeddings)
        for decoder_layer in decoder.decoder_layers:
            init_weights(decoder_layer, d_model, init_weights_from, init_weights_gain, tie_embeddings)
        init_weights(decoder.post_decoder_norm, d_model, init_weights_from, init_weights_gain, tie_embeddings)
        if tie_embeddings:
            decoder.lm_head.weight = decoder.embed_tokens.weight
        else:
            init_weights(decoder.lm_head, d_model, init_weights_from, init_weights_gain, tie_embeddings)
    elif isinstance(model, megatransformer.DecoderLayer):
        decoder_layer: megatransformer.DecoderLayer = model
        init_weights(decoder_layer.self_attn, d_model, init_weights_from, init_weights_gain, tie_embeddings)
        if decoder_layer.cross_attn is not None:
            init_weights(decoder_layer.cross_attn, d_model, init_weights_from, init_weights_gain, tie_embeddings)
        init_weights(decoder_layer.ffn, d_model, init_weights_from, init_weights_gain, tie_embeddings)
    elif isinstance(model, multihead_attn.MultiHeadAttention) or isinstance(model, grouped_query_attn.GroupedQueryMultiHeadAttention):
        mha: Union[multihead_attn.MultiHeadAttention, grouped_query_attn.GroupedQueryMultiHeadAttention] = model
        init_weights(mha.q_proj, d_model, init_weights_from, init_weights_gain, tie_embeddings)
        init_weights(mha.k_proj, d_model, init_weights_from, init_weights_gain, tie_embeddings)
        init_weights(mha.v_proj, d_model, init_weights_from, init_weights_gain, tie_embeddings)
        init_weights(mha.qkv_norm, d_model, init_weights_from, init_weights_gain, tie_embeddings)
        init_weights(mha.o_proj, d_model, init_weights_from, init_weights_gain, tie_embeddings)
    elif isinstance(model, infinite_multihead_attn.InfiniteMultiHeadAttention):
        raise NotImplementedError("Weight initialization for InfiniteMultiHeadAttention not implemented.")
    elif isinstance(model, positionwise_fcn.PositionWiseFCNetwork) or isinstance(model, phi3_mlp.Phi3MLP):
        fcn: Union[positionwise_fcn.PositionWiseFCNetwork, phi3_mlp.Phi3MLP] = model
        init_weights(fcn.layer_norm, d_model, init_weights_from, init_weights_gain, tie_embeddings)
        init_weights(fcn.expand, d_model, init_weights_from, init_weights_gain, tie_embeddings)
        init_weights(fcn.condense, d_model, init_weights_from, init_weights_gain, tie_embeddings)
    elif isinstance(model, millions_moe.MillionsMoE):
        raise NotImplementedError("Weight initialization for MillionsMoE not implemented.")
    elif isinstance(model, nn.Embedding):
        embedding: nn.Embedding = model
        nn.init.normal_(embedding.weight, mean=0., std=d_model ** -0.5)
    elif isinstance(model, nn.Linear):
        linear: nn.Linear = model
        if init_weights_from == 'glorot_uniform':
            nn.init.xavier_uniform_(linear.weight, gain=init_weights_gain)
        elif init_weights_from == 'glorot_normal':
            nn.init.xavier_normal_(linear.weight, gain=init_weights_gain)
        elif init_weights_from == 'kaiming_uniform':
            nn.init.kaiming_uniform_(linear.weight)
        elif init_weights_from == 'kaiming_normal':
            nn.init.kaiming_normal_(linear.weight)
        elif init_weights_from == 'orthogonal':
            nn.init.orthogonal_(linear.weight)
        else:
            raise Exception(f"Unknown weight initialization method: {init_weights_from}")
        
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)
    elif isinstance(model, nn.LayerNorm):
        layer_norm: nn.LayerNorm = model
        nn.init.ones_(layer_norm.weight)
        nn.init.zeros_(layer_norm.bias)
    else:
        raise Exception(f"Unknown model type: {type(model)}")
        
    print(f"{type(model)} initialized.")

def record_model_param_stats(args, summary_writer: SummaryWriter, model, step, prefix='', embedding_tokens=None):
    model = sanitize_model(model)
    if isinstance(model, megatransformer.MegaTransformer):
        record_model_param_stats(args, summary_writer, model.encoder, step, prefix='/'.join([prefix, 'encoder']), embedding_tokens=embedding_tokens)
        record_model_param_stats(args, summary_writer, model.decoder, step, prefix='/'.join([prefix, 'decoder']), embedding_tokens=embedding_tokens)
    elif isinstance(model, megatransformer.Encoder):
        encoder: megatransformer.Encoder = model
        record_model_param_stats(args, summary_writer, encoder.embed_tokens, step, prefix='/'.join([prefix, 'embed_tokens']))
        for i, encoder_layer in enumerate(model.encoder_layers):
            record_model_param_stats(args, summary_writer, encoder_layer, step, prefix=f'/'.join([prefix, 'encoder_layer', str(i)]))
        record_model_param_stats(args, summary_writer, encoder.post_encoder_norm, step, prefix='/'.join([prefix, 'post_encoder_norm']))
    elif isinstance(model, megatransformer.Decoder):
        decoder: megatransformer.Decoder = model
        record_model_param_stats(args, summary_writer, decoder.embed_tokens, step, prefix='/'.join([prefix, 'embed_tokens']), embedding_tokens=embedding_tokens)
        for i, decoder_layer in enumerate(model.decoder_layers):
            record_model_param_stats(args, summary_writer, decoder_layer, step, prefix=f'/'.join([prefix, 'decoder_layer', str(i)]))
        record_model_param_stats(args, summary_writer, decoder.post_decoder_norm, step, prefix='/'.join([prefix, 'post_decoder_norm']))
        record_model_param_stats(args, summary_writer, decoder.lm_head, step, prefix='/'.join([prefix, 'lm_head']))
    elif isinstance(model, megatransformer.EncoderLayer):
        encoder_layer: megatransformer.EncoderLayer = model
        self_attn: nn.Module = encoder_layer.self_attn
        ffn: nn.Module = encoder_layer.ffn
        record_model_param_stats(args, summary_writer, self_attn, step, prefix='/'.join([prefix, 'self_attn']))
        record_model_param_stats(args, summary_writer, ffn, step, prefix='/'.join([prefix, 'ffn']))
    elif isinstance(model, megatransformer.DecoderLayer):
        decoder_layer: megatransformer.DecoderLayer = model
        self_attn: nn.Module = decoder_layer.self_attn
        cross_attn: nn.Module = decoder_layer.cross_attn
        ffn: nn.Module = decoder_layer.ffn
        record_model_param_stats(args, summary_writer, self_attn, step, prefix='/'.join([prefix, 'self_attn']))
        if cross_attn is not None:
            record_model_param_stats(args, summary_writer, cross_attn, step, prefix='/'.join([prefix, 'cross_attn']))
        record_model_param_stats(args, summary_writer, ffn, step, prefix='/'.join([prefix, 'ffn']))
    elif isinstance(model, multihead_attn.MultiHeadAttention) or isinstance(model, grouped_query_attn.GroupedQueryMultiHeadAttention):
        mha: Union[multihead_attn.MultiHeadAttention, grouped_query_attn.GroupedQueryMultiHeadAttention] = model
        q_proj: nn.Linear = mha.q_proj
        k_proj: nn.Linear = mha.k_proj
        v_proj: nn.Linear = mha.v_proj
        qkv_norm: nn.LayerNorm = mha.qkv_norm
        out_proj: nn.Linear = mha.o_proj
        record_model_param_stats(args, summary_writer, q_proj, step, prefix='/'.join([prefix, 'q_proj']))
        record_model_param_stats(args, summary_writer, k_proj, step, prefix='/'.join([prefix, 'k_proj']))
        record_model_param_stats(args, summary_writer, v_proj, step, prefix='/'.join([prefix, 'v_proj']))
        record_model_param_stats(args, summary_writer, qkv_norm, step, prefix='/'.join([prefix, 'qkv_norm']))
        record_model_param_stats(args, summary_writer, out_proj, step, prefix='/'.join([prefix, 'o_proj']))
    elif isinstance(model, infinite_multihead_attn.InfiniteMultiHeadAttention):
        raise NotImplementedError("Weight visualization for InfiniteMultiHeadAttention not implemented.")
    elif isinstance(model, positionwise_fcn.PositionWiseFCNetwork) or isinstance(model, phi3_mlp.Phi3MLP):
        ffn: Union[positionwise_fcn.PositionWiseFCNetwork, phi3_mlp.Phi3MLP] = model
        record_model_param_stats(args, summary_writer, ffn.layer_norm, step, prefix='/'.join([prefix, 'layer_norm']))
        record_model_param_stats(args, summary_writer, ffn.expand, step, prefix='/'.join([prefix, 'expand']))
        record_model_param_stats(args, summary_writer, ffn.condense, step, prefix='/'.join([prefix, 'condense']))
    elif isinstance(model, millions_moe.MillionsMoE):
        raise NotImplementedError("Weight visualization for MillionsMoE not implemented.")
    elif isinstance(model, nn.Embedding):
        embedding: nn.Embedding = model
        if embedding_tokens is not None:
            metadata = embedding_tokens
            print(f"Metadata for {prefix}: {len(metadata)}")
        else:
            metadata = None
        summary_writer.add_histogram(f'{prefix}/weight', embedding.weight, step)
        if embedding.weight.grad is not None:
            summary_writer.add_histogram(f'{prefix}/weight_grad', embedding.weight.grad, step)
        summary_writer.add_embedding(embedding.weight, global_step=step, tag=f'{prefix}/embedding', metadata=metadata, metadata_header=['token'])
    elif isinstance(model, nn.Linear):
        linear: nn.Linear = model
        summary_writer.add_histogram(f'{prefix}/weight', linear.weight, step)
        if linear.bias is not None:
            summary_writer.add_histogram(f'{prefix}/bias', linear.bias, step)
        if linear.weight.grad is not None:
            summary_writer.add_histogram(f'{prefix}/weight_grad', linear.weight.grad, step)
            if linear.bias is not None:
                summary_writer.add_histogram(f'{prefix}/bias_grad', linear.bias.grad, step)
    elif isinstance(model, nn.LayerNorm):
        layer_norm: nn.LayerNorm = model
        summary_writer.add_histogram(f'{prefix}/weight', layer_norm.weight, step)
        summary_writer.add_histogram(f'{prefix}/bias', layer_norm.bias, step)
        if layer_norm.weight.grad is not None:
            summary_writer.add_histogram(f'{prefix}/weight_grad', layer_norm.weight.grad, step)
            summary_writer.add_histogram(f'{prefix}/bias_grad', layer_norm.bias.grad, step)

    print(f"Posted statistics for {prefix}")