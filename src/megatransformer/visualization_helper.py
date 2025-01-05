from . import megatransformer, multihead_attn

import io
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def viz_attn_weight(stack_name, layer_num, n_head, activation_weights, attendee_tokens, attending_tokens):
    fig, ax = plt.subplots(figsize=(10, 10))
    s = sns.heatmap(activation_weights, square=True, annot=True, annot_kws={"fontsize":6}, fmt=".4f", xticklabels=attendee_tokens, yticklabels=attending_tokens, ax=ax)
    s.set(xlabel="Attending Tokens", ylabel="Attended Tokens", title=f"{stack_name}-Attn Layer {layer_num} Head {n_head} Weights")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    return buf

def viz_attn_weights(attn: multihead_attn.MultiHeadAttention, attn_residual, layer_num, src_seq, src_seq_len, key_padding_mask, tgt_seq, tgt_seq_len, src_tokens, tgt_tokens, log_callback, step):
    decoder_or_encoder = 'decoder' if attn.in_decoder else 'encoder'
    self_or_cross = 'self' if attn.self_attn else 'cross'
    stack_name = f"{decoder_or_encoder}-{self_or_cross}"

    residual, attention_weights = attn(tgt_seq, src_seq, src_seq, key_padding_mask)
    tgt_seq = attn_residual(tgt_seq, residual)

    for a, attention_weight_grid in enumerate(attention_weights):
        attention_weight_grid = attention_weight_grid.cpu().contiguous()
        for head_num in range(attention_weight_grid.size(1)):
            image_data = viz_attn_weight(stack_name, layer_num, head_num, attention_weight_grid[:, head_num, :tgt_seq_len, :src_seq_len].transpose(-2, -1).squeeze(0).cpu().detach().numpy(), tgt_tokens, src_tokens)
            log_callback(f"{decoder_or_encoder}/viz/layer_{layer_num}/segment_{a}/head_{head_num}/{self_or_cross}-attn", plt.imread(image_data), global_step=step, dataformats='HWC')

    return tgt_seq

def viz_encoder_layer(encoder_layer: megatransformer.EncoderLayer, src_seq, src_seq_len, src_key_padding_mask, src_tokens, log_callback, step, layer_num):
    print(f"Visualizing encoder layer {layer_num}...")
    residual = viz_attn_weights(encoder_layer.self_attn, encoder_layer.self_attn_residual, layer_num, src_seq, src_seq_len, src_key_padding_mask, src_seq, src_seq_len, src_tokens, src_tokens, log_callback, step)
    fcn_out, _ = encoder_layer.fcn(residual)
    return encoder_layer.fcn_residual(residual, fcn_out)

def viz_encoder_layers(encoder_layers: list[megatransformer.EncoderLayer], seq, seq_len, key_padding_mask, tokens, log_callback, step):
    for e, encoder_layer in enumerate(encoder_layers):
        seq = viz_encoder_layer(encoder_layer, seq, seq_len, key_padding_mask, tokens, log_callback, step, e)
    return seq

def viz_encoder(device, encoder: megatransformer.Encoder, seq, seq_len, key_padding_mask, tokens, log_callback, step):
    print("Visualizing encoder...")
    if (hasattr(encoder, 'embed_tokens') and encoder.embed_tokens is not None) or (hasattr(encoder, 'embedding') and encoder.embed_tokens is not None):
        seq = encoder.apply_embedding_transformation(seq)
    seq = encoder.apply_positional_embedding(seq)
    seq = viz_encoder_layers(encoder.encoder_layers, seq, seq_len, key_padding_mask, tokens, log_callback, step)
    return encoder.post_encoder_norm(seq).to(device)

def viz_decoder_layer(decoder_layer: megatransformer.DecoderLayer, src_seq, src_seq_len, src_key_padding_mask, tgt_seq, tgt_seq_len, tgt_key_padding_mask, src_tokens, tgt_tokens, log_callback, step, layer_num):
    print(f"Visualizing decoder layer {layer_num}...")
    tgt_seq = viz_attn_weights(decoder_layer.self_attn, decoder_layer.self_attn_residual, layer_num, tgt_seq, tgt_seq_len, tgt_key_padding_mask, tgt_seq, tgt_seq_len, tgt_tokens, tgt_tokens, log_callback, step)
    residual = viz_attn_weights(decoder_layer.cross_attn, decoder_layer.cross_attn_residual, layer_num, src_seq, src_seq_len, src_key_padding_mask, tgt_seq, tgt_seq_len, src_tokens, tgt_tokens, log_callback, step)
    fcn_out, _ = decoder_layer.fcn(residual)
    return decoder_layer.ffn_residual(residual, fcn_out)

def viz_decoder_layers(decoder_layers, src_seq, src_seq_len, src_key_padding_mask, tgt_seq, tgt_seq_len, tgt_key_padding_mask, src_tokens, tgt_tokens, log_callback, step):
    for d, decoder_layer in enumerate(decoder_layers):
        tgt_seq = viz_decoder_layer(decoder_layer, src_seq, src_seq_len, src_key_padding_mask, tgt_seq, tgt_seq_len, tgt_key_padding_mask, src_tokens, tgt_tokens, log_callback, step, d)
    return tgt_seq

def viz_decoder(device, decoder: megatransformer.Decoder, src_seq, src_seq_len, src_key_padding_mask, tgt_seq, tgt_seq_len, tgt_key_padding_mask, src_tokens, tgt_tokens, log_callback, step):
    print("Visualizing decoder...")
    if (hasattr(decoder, 'embed_tokens') and decoder.embed_tokens is not None) or (hasattr(decoder, 'embedding') and decoder.embed_tokens is not None):
        tgt_seq = decoder.apply_embedding_transformation(tgt_seq)
    tgt_seq = decoder.apply_positional_embedding(tgt_seq.to(device))
    return viz_decoder_layers(decoder.decoder_layers, src_seq, src_seq_len, src_key_padding_mask, tgt_seq, tgt_seq_len, tgt_key_padding_mask, src_tokens, tgt_tokens, log_callback, step)

def viz_model(encoder_device, decoder_device, model: megatransformer.MegaTransformer, log_callback, step, maxlen, src, src_labels, tgt=None, tgt_labels=None):
    model.eval()
    with torch.no_grad():
        src_seq_len = src.size(1)
        tgt_seq_len = tgt.size(1)
        
        src = torch.cat([src, torch.zeros([1, maxlen - src.size(1)], dtype=torch.long, device=src.device)], dim=1)
        tgt = torch.cat([tgt, torch.zeros([1, maxlen - tgt.size(1)], dtype=torch.long, device=tgt.device)], dim=1)

        src_key_padding_mask = src == 0
        tgt_key_padding_mask = tgt.to(decoder_device) == 0

        # permanent device assignments
        src = src.to(encoder_device)
        tgt = tgt.to(decoder_device)
        src_key_padding_mask = src_key_padding_mask.to(encoder_device)
        tgt_key_padding_mask = tgt_key_padding_mask.to(decoder_device)
        model.encoder.embed_tokens = model.encoder.embed_tokens.to(encoder_device)

        src = viz_encoder(encoder_device, model.encoder, src, src_seq_len, src_key_padding_mask, src_labels, log_callback, step)

        # things that need to move to decoder device
        src = src.to(decoder_device)
        src_key_padding_mask = src_key_padding_mask.to(decoder_device)
        model.decoder.embed_tokens = model.decoder.embed_tokens.to(decoder_device)
        model.decoder.lm_head = model.decoder.lm_head.to(decoder_device)

        tgt = viz_decoder(decoder_device, model.decoder, src, src_seq_len, src_key_padding_mask, tgt, tgt_seq_len, tgt_key_padding_mask, src_labels, tgt_labels, log_callback, step)
