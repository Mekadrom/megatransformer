from torch import nn
from transformers import GenerationMixin, PreTrainedModel, PreTrainedTokenizer, GPT2LMHeadModel, GPT2Config

from model import megatransformer_audio_decoder, megatransformer_recurrent, megatransformer_audio_encoder, megatransformer_diffusion, megatransformer_image_encoder, megatransformer_image_decoder, megatransformer_modules, megatransformer_text_encoder

import megatransformer_utils
import torch
import torch.nn.functional as F


class MegaTransformerMultimodalEncoder(nn.Module):
    def __init__(self, config, text_embedding, audio_embedding, image_embedding):
        super().__init__()
        self.config = config

        self.text_embedding = text_embedding
        self.audio_embedding = audio_embedding
        self.image_embedding = image_embedding

    def forward(self,
                text_input_ids=None,
                audio_raw_inputs=None,
                image_raw_inputs=None,
                text_past_key_values=None,
                audio_waveform_labels=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if text_input_ids is None:
            text_prelude_outputs = None
        else:
            text_prelude_outputs = self.text_embedding(text_input_ids, past_key_values=text_past_key_values)

        if audio_raw_inputs is None or audio_waveform_labels is None:
            audio_prelude_outputs = None
        else:
            B, N, C, M, T = audio_raw_inputs.shape

            # megatransformer_utils.print_debug_tensor('audio_raw_inputs', audio_raw_inputs)

            # combine batch and n_audios dimensions for easier processing
            audio_raw_inputs = audio_raw_inputs.view(-1, C, M, T)

            # Find which samples in the batch have valid audio content
            valid_samples = audio_raw_inputs.sum(dim=(1, 2)) != 0  # [batch_size, audio_max_frames]
            valid_batch_indices = valid_samples.any(dim=1).nonzero().squeeze(-1)  # [num_valid_samples]

            if len(valid_batch_indices) > 0:
                # Process only valid audio samples through the embedding
                valid_audio = audio_raw_inputs[valid_batch_indices]
                valid_waveform_labels = audio_waveform_labels[valid_batch_indices]
                valid_outputs = self.audio_embedding(valid_audio, valid_waveform_labels)
                
                # Create output tensor with zeros
                batch_size = audio_raw_inputs.shape[0]
                output_shape = list(valid_outputs[0].shape)
                output_shape[0] = batch_size  # Adjust to full batch size
                
                # Initialize output with zeros
                audio_logits = torch.zeros(output_shape, dtype=valid_outputs[0].dtype, device=audio_raw_inputs.device)
                
                # Place valid outputs in the appropriate positions
                audio_logits[valid_batch_indices] = valid_outputs[0]

                # restore n_audios dimension
                audio_logits = audio_logits.view(B, N, T, -1)

                # Reconstruct the full outputs tuple
                other_outputs = [audio_logits]
                for i in range(1, len(valid_outputs)):
                    # For other outputs in the tuple, also create tensors with appropriate shape
                    if valid_outputs[i] is None:
                        other_outputs.append(None)
                        continue
                    other_shape = list(valid_outputs[i].shape)
                    other_shape[0] = batch_size
                    other_tensor = torch.zeros(other_shape, dtype=valid_outputs[i].dtype, device=audio_raw_inputs.device)
                    other_tensor[valid_batch_indices] = valid_outputs[i]
                    other_outputs.append(other_tensor)
                
                audio_prelude_outputs = tuple(other_outputs)
            else:
                audio_prelude_outputs = None
        if image_raw_inputs is None:
            image_prelude_outputs = None
        else:
            B, N, C, H, W = image_raw_inputs.shape

            # megatransformer_utils.print_debug_tensor('image_raw_inputs', image_raw_inputs)

            # combine batch and n_images dimensions for easier processing
            image_raw_inputs = image_raw_inputs.view(-1, C, H, W)

            # megatransformer_utils.print_debug_tensor('image_raw_inputs', image_raw_inputs)

            # Find which samples in the batch have valid image content
            valid_samples = image_raw_inputs.sum(dim=(1, 2, 3)) != 0  # [batch_size]
            valid_batch_indices = valid_samples.nonzero().squeeze(-1)  # [num_valid_samples]

            # print(f"valid_batch_indices: {valid_batch_indices}")

            if len(valid_batch_indices) > 0:
                # Process only valid image samples through the embedding
                valid_images = image_raw_inputs[valid_batch_indices]
                valid_outputs = self.image_embedding(valid_images)

                if isinstance(valid_outputs, tuple):
                    image_features = valid_outputs[0]
                elif isinstance(valid_outputs, torch.Tensor):
                    image_features = valid_outputs
                else:
                    raise ValueError(f"Invalid output from image embedding, expected tuple, got {type(valid_outputs)}")
                

                # Create output tensor with zeros
                batch_size = image_raw_inputs.shape[0]
                output_shape = list(image_features.shape)
                output_shape[0] = batch_size  # Adjust to full batch size
                
                # Initialize output with zeros
                image_logits = torch.zeros(output_shape, dtype=image_features.dtype, device=image_raw_inputs.device)
                
                # megatransformer_utils.print_debug_tensor('image_logits', image_logits)
                # megatransformer_utils.print_debug_tensor('valid_batch_indices', valid_batch_indices)
                # megatransformer_utils.print_debug_tensor('image_features', image_features)

                # Place valid outputs in the appropriate positions
                image_logits[valid_batch_indices] = image_features

                # megatransformer_utils.print_debug_tensor('image_logits', image_logits)

                _, T, _ = image_logits.shape

                # restore n_images dimension
                image_logits = image_logits.view(B, N, T, -1)
                
                if isinstance(valid_outputs, tuple):
                    # Reconstruct the full outputs tuple
                    other_outputs = [image_logits]
                    for i in range(1, len(valid_outputs)):
                        # For other outputs in the tuple, also create tensors with appropriate shape
                        valid_output = valid_outputs[i]
                        if valid_output is None:
                            other_outputs.append(None)
                            continue

                        megatransformer_utils.print_debug_tensor('valid_output', valid_output)
                        megatransformer_utils.print_debug_tensor('valid_batch_indices', valid_batch_indices)

                        other_shape = list(valid_output.shape)
                        other_shape[0] = batch_size
                        other_tensor = torch.zeros(other_shape, dtype=valid_output.dtype, device=image_raw_inputs.device)
                        other_tensor[valid_batch_indices] = valid_output
                        other_outputs.append(other_tensor)
                
                    image_prelude_outputs = tuple(other_outputs)
                else:
                    image_prelude_outputs = image_logits,
            else:
                image_prelude_outputs = None

        return text_prelude_outputs, audio_prelude_outputs, image_prelude_outputs

class MegaTransformerMultimodalDecoder(nn.Module):
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig, text_decoder, audio_decoder, image_decoder):
        super().__init__()
        self.config = config

        self.text_coda = megatransformer_modules.SimpleBlock(config.text_coda_config, "text_coda", config.text_coda_config.n_coda_layers, config.hidden_dropout_prob)
        self.text_decoder = text_decoder

        self.audio_coda = megatransformer_modules.SimpleBlock(config.audio_coda_config, "audio_coda", config.audio_coda_config.n_coda_layers, config.audio_decoder_dropout)
        self.audio_decoder = audio_decoder

        self.image_coda = megatransformer_modules.SimpleBlock(config.image_coda_config, "image_coda", config.image_coda_config.n_coda_layers, config.image_decoder_dropout)
        self.image_decoder = image_decoder

    def audio_coda_forward(self, audio_hidden_states, audio_mel_spec_labels=None, audio_waveform_labels=None):
        _, _, H1, W1 = audio_hidden_states.shape
        if audio_mel_spec_labels is not None:
            _, _, C2, H2, W2 = audio_mel_spec_labels.shape

        # combine batch and n_audios/example # dimensions for easier processing
        audio_hidden_states = audio_hidden_states.view(-1, H1, W1)
        if audio_mel_spec_labels is not None:
            audio_mel_spec_labels = audio_mel_spec_labels.view(-1, C2, H2, W2)
        if audio_waveform_labels is not None:
            audio_waveform_labels = audio_waveform_labels.view(-1, audio_waveform_labels.shape[-1])

        return self.audio_coda(audio_hidden_states)[0], audio_mel_spec_labels, audio_waveform_labels

    def image_coda_forward(self, image_hidden_states, image_labels=None):
        _, _, H1, W1 = image_hidden_states.shape
        if image_labels is not None:
            _, _, C2, H2, W2 = image_labels.shape

        # combine batch and n_images dimensions for easier processing
        image_hidden_states = image_hidden_states.view(-1, H1, W1)
        if image_labels is not None:
            image_labels = image_labels.view(-1, C2, H2, W2)

        return self.image_coda(image_hidden_states)[0], image_labels

    def forward(self,
                text_hidden_states=None,
                image_hidden_states=None,
                audio_hidden_states=None,
                audio_mel_spec_labels=None,
                audio_waveform_labels=None,
                image_labels=None):
        if text_hidden_states is None:
            text_logits = None
        else:
            text_hidden_states = self.text_coda(text_hidden_states)[0]
            text_logits = self.text_decoder(text_hidden_states)

        if audio_hidden_states is None:
            mel_spec_reconstructions = None
            audio_waveforms = None
            audio_reconstruction_loss = None
        else:
            # (batch, example, seq_len, hidden_size)
            B1, E1, H1, W1 = audio_hidden_states.shape
            audio_hidden_states, audio_mel_spec_labels, audio_waveform_labels = self.audio_coda_forward(
                audio_hidden_states,
                audio_mel_spec_labels,
                audio_waveform_labels,
            )

            megatransformer_utils.print_debug_tensor('audio_mel_spec_labels', audio_mel_spec_labels)
            megatransformer_utils.print_debug_tensor('audio_hidden_states', audio_hidden_states)

            mel_spec_reconstructions, audio_reconstruction_loss, audio_waveforms = self.audio_decoder(
                audio_mel_spec_labels,
                condition=audio_hidden_states,
                waveform_labels=audio_waveform_labels,
            )

            # restore n_audios dimension
            # (batch, example, channels, hidden_size, seq_len)
            mel_spec_reconstructions = mel_spec_reconstructions.view(B1, E1, 1, self.config.audio_n_mels, H1)
        if image_hidden_states is None:
            image_outputs = None
            image_reconstruction_loss = None
        else:
            # (batch, example, seq_len, hidden_size)
            B1, E1, H1, W1 = image_hidden_states.shape
            image_hidden_states, image_labels = self.image_coda_forward(
                image_hidden_states,
                image_labels,
            )

            image_outputs, image_reconstruction_loss = self.image_decoder(
                image_labels,
                condition=image_hidden_states,
            )

            # restore n_images dimension
            # (batch, example, channels, hidden_size, seq_len)
            image_outputs = image_outputs.view(B1, E1, 3, self.config.image_size, self.config.image_size)

        return text_logits, mel_spec_reconstructions, audio_waveforms, audio_reconstruction_loss, image_outputs, image_reconstruction_loss

class MegaTransformerCausalWMHeads(PreTrainedModel, GenerationMixin):
    config_class = megatransformer_utils.MegaTransformerConfig
    
    def __init__(self,
                 config: megatransformer_utils.MegaTransformerConfig,
                 text_embedding, audio_embedding, image_embedding,
                 world_model,
                 text_decoder, audio_decoder, image_decoder):
        super().__init__(config)
        self.config = config

        self.input_transform = MegaTransformerMultimodalEncoder(config, text_embedding, audio_embedding, image_embedding)
        self.world_model = world_model
        self.output_transform = MegaTransformerMultimodalDecoder(config, text_decoder, audio_decoder, image_decoder)

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
                                        text_past_key_values=None,
                                        audio_waveform_labels=None):
        """
        Interleaves text, audio, and image embeddings based on special tokens in the text sequence.
        
        Args:
            text_input_ids: Tensor of shape [batch_size, seq_length]
            audio_raw_inputs: Tensor of shape [batch_size, 1, n_mels, max_audio_frames]
            image_raw_inputs: Tensor of shape [batch_size, 3, image_size, image_size]
            text_past_key_values: Optional past key values for text embedding
            
        Returns:
            embeds: Tensor of shape [batch_size, total_seq_length, hidden_size]
            attention_mask: Tensor of shape [batch_size, total_seq_length]
            image_positions: List of positions where image features start
            audio_positions: List of positions where audio features start
            text_prelude_outputs: Output from text embedding
            audio_prelude_outputs: Output from audio embedding
            image_prelude_outputs: Output from image embedding
        """
        # Process each modality through their embedding functions
        text_prelude_outputs, audio_prelude_outputs, image_prelude_outputs = self.input_transform(
            text_input_ids=text_input_ids,
            audio_raw_inputs=audio_raw_inputs,
            image_raw_inputs=image_raw_inputs,
            text_past_key_values=text_past_key_values,
            audio_waveform_labels=audio_waveform_labels,
        )

        # Extract the embeddings
        text_embeds = text_prelude_outputs[0] if isinstance(text_prelude_outputs, tuple) else text_prelude_outputs
        
        audio_embeds = None
        if audio_prelude_outputs is not None:
            audio_embeds = audio_prelude_outputs[0] if isinstance(audio_prelude_outputs, tuple) else audio_prelude_outputs
        
        image_embeds = None
        if image_prelude_outputs is not None:
            image_embeds = image_prelude_outputs[0] if isinstance(image_prelude_outputs, tuple) else image_prelude_outputs

        # megatransformer_utils.print_debug_tensor('text_embeds', text_embeds)
        # megatransformer_utils.print_debug_tensor('audio_embeds', audio_embeds)
        # megatransformer_utils.print_debug_tensor('image_embeds', image_embeds)

        # Get batch size and device
        batch_size = text_input_ids.shape[0]
        device = text_input_ids.device
        
        # Track where audio and image features are placed
        # start_pos, length
        audio_positions = []
        image_positions = []
        
        all_embeddings = []
        all_masks = []
        
        for b in range(batch_size):
            audio_positions.append([])
            image_positions.append([])

            batch_ids = text_input_ids[b]
            batch_embeds = text_embeds[b]
            
            # Find begin token positions
            begin_audio_mask = (batch_ids == self.config.begin_audio_token_id) | (batch_ids == self.config.begin_voice_token_id)
            begin_audio_pos = begin_audio_mask.nonzero(as_tuple=True)[0]
            begin_image_pos = (batch_ids == self.config.begin_image_token_id).nonzero(as_tuple=True)[0]

            # print(f"Batch {b}: Begin audio positions: {begin_audio_pos}")
            # print(f"Batch {b}: Begin image positions: {begin_image_pos}")
            
            segments = []
            current_pos = 0
            
            # (n_mode_exmaple, begin pos, length, type) for this batch
            pre_filtered_special_positions = []
            for pos in begin_audio_pos:
                pre_filtered_special_positions.append([None, pos.item(), 0])

            for pos in begin_image_pos:
                pre_filtered_special_positions.append([None, pos.item(), 1])
            
            # Sort by begin position
            pre_filtered_special_positions.sort(key=lambda x: x[1])

            # Assign indices to each special position after sorting
            audio_idx = 0
            image_idx = 0
            special_positions = []
            for i, special_position in enumerate(pre_filtered_special_positions):
                if special_position[-1] == 0:
                    if audio_raw_inputs is None or audio_raw_inputs[audio_idx].count_nonzero() == 0:
                        audio_idx += 1
                        continue
                    special_position[0] = audio_idx
                    audio_idx += 1
                else:
                    if image_raw_inputs is None or image_raw_inputs[image_idx].count_nonzero() == 0:
                        image_idx += 1
                        continue
                    special_position[0] = image_idx
                    image_idx += 1
                special_positions.append(tuple(special_position))

            # print(f"Batch {b}: Special positions: {special_positions}")

            new_pos = 0
            # iterate over only valid special positions
            for i, (idx, begin_pos, modal_type) in enumerate(special_positions):
                # Add text up to begin token
                if begin_pos > current_pos:
                    text_segment = batch_embeds[current_pos:begin_pos]
                    segments.append(text_segment)
                    new_pos += text_segment.shape[0]
                
                # Add begin token embed
                begin_token_embed = batch_embeds[begin_pos:begin_pos+1]
                segments.append(begin_token_embed)
                new_pos += 1
                
                if modal_type == 0 and audio_embeds is not None and b < audio_embeds.shape[0] and idx < audio_embeds.shape[1]:
                    audio_segment = audio_embeds[b][idx]

                    segments.append(audio_segment)

                    # Track the starting position for this audio segment
                    audio_start_pos = new_pos
                    audio_positions[b].append((audio_start_pos, audio_segment.shape[0]))

                    # Update position
                    new_pos += audio_segment.shape[0]

                    # Add end token embed
                    end_token_embed = batch_embeds[begin_pos:begin_pos+1]
                    segments.append(end_token_embed)
                    new_pos += 1
                elif modal_type == 1 and image_embeds is not None and b < image_embeds.shape[0] and idx < image_embeds.shape[1]:
                    image_segment = image_embeds[b][idx]

                    segments.append(image_segment)

                    # Track the starting position for this image segment
                    image_start_pos = new_pos
                    image_positions[b].append((image_start_pos, image_segment.shape[0]))

                    # Update position
                    new_pos += image_segment.shape[0]

                    # Add end token embed
                    end_token_embed = batch_embeds[begin_pos:begin_pos+1]
                    segments.append(end_token_embed)
                    new_pos += 1

                # Update current position
                current_pos = begin_pos + 2
            
            # Add any remaining text
            if current_pos < batch_embeds.shape[0]:
                # print(f"adding remaining text: {current_pos} to {batch_embeds.shape[0]}")
                text_segment = batch_embeds[current_pos:]
                segments.append(text_segment)
            
            total_length = 0
            for s, segment in enumerate(segments):
                total_length += segment.shape[0]

            # print(f"Batch {b}: Total length of segments: {total_length}")

            # Concatenate all segments
            if segments:
                batch_embeddings = torch.cat(segments, dim=0)
                # Create attention mask (1 for all positions)
                batch_mask = torch.ones(batch_embeddings.shape[0], device=device)
                
                all_embeddings.append(batch_embeddings)
                all_masks.append(batch_mask)
            else:
                # No special tokens, use original embeddings
                all_embeddings.append(batch_embeds)
                all_masks.append(torch.ones(batch_embeds.shape[0], device=device))
        
        # Pad to max length
        max_length = max([emb.shape[0] for emb in all_embeddings])
        padded_embeddings = []
        padded_masks = []
        
        for emb, mask in zip(all_embeddings, all_masks):
            # Pad if needed
            if emb.shape[0] < max_length:
                padding = torch.zeros(max_length - emb.shape[0], emb.shape[1], dtype=emb.dtype, device=device)
                padded_emb = torch.cat([emb, padding], dim=0)
                
                mask_padding = torch.zeros(max_length - mask.shape[0], dtype=mask.dtype, device=device)
                padded_mask = torch.cat([mask, mask_padding], dim=0)
            else:
                padded_emb = emb
                padded_mask = mask
            
            padded_embeddings.append(padded_emb)
            padded_masks.append(padded_mask)
        
        # Stack into batch tensors
        combined_embeddings = torch.stack(padded_embeddings, dim=0)
        combined_masks = torch.stack(padded_masks, dim=0)

        return combined_embeddings, combined_masks, audio_positions, image_positions, text_prelude_outputs, audio_prelude_outputs, image_prelude_outputs

    def extract_batch_aligned_multimodal_features(self, hidden_states, longest_text_sample, audio_positions, image_positions):
        batch_size, seq_length, hidden_dim = hidden_states.shape
        device = hidden_states.device

        assert len(audio_positions) == batch_size, "Audio positions must match batch size, include empty lists for batch examples with no audio."
        assert len(image_positions) == batch_size, "Image positions must match batch size, include empty lists for batch examples with no images."
        
        max_audio_samples = -1
        longest_audio_sample = -1

        max_image_samples = -1
        longest_image_sample = -1

        # Extract audio hidden states
        audio_hidden_states = []
        for b in range(batch_size):
            if len(audio_positions[b]) > 0:
                positions = sorted(audio_positions[b], key=lambda x: x[0])

                max_audio_samples = max(max_audio_samples, len(positions))
                
                # For each starting position, find where the segment ends
                batch_audio_examples = []
                for i, (start_pos, length) in enumerate(positions):
                    end_pos = start_pos + length
                    audio_example = hidden_states[b, start_pos:end_pos]
                    batch_audio_examples.append(audio_example)

                    T, _ = audio_example.shape
                    longest_audio_sample = max(longest_audio_sample, T)

                audio_hidden_states.append(batch_audio_examples)
            else:
                # No audio for this batch
                audio_hidden_states.append([])

        if longest_audio_sample > 0:
            # Pad audio hidden states to the longest sample along length dimension
            for j, batch_audio_examples in enumerate(audio_hidden_states):
                if len(batch_audio_examples) > 0:
                    for i, audio_example in enumerate(batch_audio_examples):
                        T, _ = audio_example.shape
                        if T < longest_audio_sample:
                            padding = torch.zeros(longest_audio_sample - T, hidden_dim, dtype=hidden_states.dtype, device=device)
                            batch_audio_examples[i] = torch.cat([audio_example, padding], dim=0)
                    audio_hidden_states[j] = torch.stack(batch_audio_examples, dim=0)
                else:
                    audio_hidden_states[j] = torch.zeros(0, longest_audio_sample, hidden_dim, dtype=hidden_states.dtype, device=device)

                # Pad to same number of examples
                if len(audio_hidden_states[j]) < max_audio_samples:
                    padding = torch.zeros(max_audio_samples - len(audio_hidden_states[j]), longest_audio_sample, hidden_dim, dtype=hidden_states.dtype, device=device)
                    audio_hidden_states[j] = torch.cat([audio_hidden_states[j], padding], dim=0)

            # Stack the audio hidden states
            audio_hidden_states = torch.stack(audio_hidden_states, dim=0)
        else:
            # No audio for this batch
            audio_hidden_states = torch.zeros(batch_size, 0, 0, hidden_dim, dtype=hidden_states.dtype, device=device)

        # Extract image hidden states with the same approach
        image_hidden_states = []
        for b in range(batch_size):
            if len(image_positions[b]) > 0:
                positions = sorted(image_positions[b], key=lambda x: x[0])

                max_image_samples = max(max_image_samples, len(positions))
                
                # For each starting position, find where the segment ends
                batch_image_examples = []
                for i, (start_pos, length) in enumerate(positions):
                    end_pos = start_pos + length
                    image_example = hidden_states[b, start_pos:end_pos]
                    batch_image_examples.append(image_example)

                    T, _ = image_example.shape
                    longest_image_sample = max(longest_image_sample, T)
                    
                image_hidden_states.append(batch_image_examples)
            else:
                # No images for this batch
                image_hidden_states.append([])

        if longest_image_sample > 0:
            # Pad image hidden states to the longest sample along length dimension
            for j, batch_image_examples in enumerate(image_hidden_states):
                if len(batch_image_examples) > 0:
                    for i, image_example in enumerate(batch_image_examples):
                        T, _ = image_example.shape
                        if T < longest_image_sample:
                            padding = torch.zeros(longest_image_sample - T, hidden_dim, dtype=hidden_states.dtype, device=device)
                            batch_image_examples[i] = torch.cat([image_example, padding], dim=0)
                    image_hidden_states[j] = torch.stack(batch_image_examples, dim=0)
                else:
                    image_hidden_states[j] = torch.zeros(0, longest_image_sample, hidden_dim, dtype=hidden_states.dtype, device=device)

                # Pad to same number of examples
                if len(image_hidden_states[j]) < max_image_samples:
                    padding = torch.zeros(max_image_samples - len(image_hidden_states[j]), longest_image_sample, hidden_dim, dtype=hidden_states.dtype, device=device)
                    image_hidden_states[j] = torch.cat([image_hidden_states[j], padding], dim=0)

            # Stack the image hidden states
            image_hidden_states = torch.stack(image_hidden_states, dim=0)
        else:
            # No image for this batch
            image_hidden_states = torch.zeros(batch_size, 0, 0, hidden_dim, dtype=hidden_states.dtype, device=device)

        # For text, we'll cut up the original hidden states based on the masks
        text_hidden_states = []
        for b in range(batch_size):
            segments = []
            current_pos = 0
            for position in sorted(audio_positions[b] + image_positions[b], key=lambda x: x[0]):
                start_pos = position[0]
                end_pos = position[0] + position[1]
                
                # Get the text segment
                if current_pos < start_pos:
                    example = hidden_states[b, current_pos:start_pos]
                    segments.append(example)
                
                # Update current position
                current_pos = end_pos

            # Add any remaining text
            if current_pos < seq_length:
                example = hidden_states[b, current_pos:]
                segments.append(example)

            example = torch.cat(segments, dim=0)
            example = example[:longest_text_sample]  # Truncate to longest text sample
            text_hidden_states.append(example)

        # Stack the hidden states for text
        text_hidden_states = torch.stack(text_hidden_states, dim=0)

        # print(f"audio_hidden_states: {audio_hidden_states.shape}, image_hidden_states: {image_hidden_states.shape}, text_hidden_states: {text_hidden_states.shape}")
        
        return text_hidden_states, audio_hidden_states, image_hidden_states
    
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
        audio_mel_spec_labels=None,
        audio_waveform_labels=None,
        image_labels=None,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=False,
    ):
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

        if inputs_embeds is None:
            if (audio_raw_inputs is not None and audio_raw_inputs.shape[-1] != 0) or (image_raw_inputs is not None and image_raw_inputs.shape[-1] != 0) :
                # multimodal
                inputs_embeds, attention_mask, audio_positions, image_positions, text_prelude_outputs, audio_prelude_outputs, image_prelude_outputs = self.interleave_batch_aligned_embeds(
                    input_ids,
                    audio_raw_inputs,
                    image_raw_inputs,
                    text_past_key_values=past_key_values[:self.config.n_prelude_layers],
                    audio_waveform_labels=audio_waveform_labels,
                )
            else:
                # text only shortcut
                text_prelude_outputs = self.input_transform(text_input_ids=input_ids)[0]
                inputs_embeds = text_prelude_outputs[0]  # [batch_size, seq_length, hidden_size]
                attention_mask = torch.ones((inputs_embeds.shape[0], inputs_embeds.shape[1]), device=inputs_embeds.device)
                audio_positions = []
                image_positions = []
        else:
            assert labels is None and image_labels is None and audio_mel_spec_labels is None and audio_waveform_labels is None, "If inputs_embeds is provided, labels, image_labels, and audio_labels should not be provided."

        transformer_outputs = self.world_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
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
        
        # during training, the positions of multimodal features are known from the interleaving process
        if len(audio_positions) != 0 and len(image_positions) != 0:
            text_hidden_states, audio_hidden_states, image_hidden_states = self.extract_batch_aligned_multimodal_features(
                hidden_states,
                input_ids.shape[1],
                audio_positions,
                image_positions,
            )
        else:
            text_hidden_states = hidden_states
            audio_hidden_states = None
            image_hidden_states = None

        if torch.count_nonzero(text_hidden_states) != 0:
            output_text_logits, *_ = self.output_transform(
                text_hidden_states=text_hidden_states,
            )
        else:
            output_text_logits = None

        if torch.count_nonzero(audio_hidden_states) != 0:
            _, output_audio, _, audio_reconstruction_loss, *_ = self.output_transform(
                audio_hidden_states=audio_hidden_states,
                audio_mel_spec_labels=audio_mel_spec_labels,
                audio_waveform_labels=audio_waveform_labels,
            )
        else:
            # output dummy audio so huggingface data parallelization doesn't have a conniption
            output_audio = torch.zeros_like(audio_raw_inputs)
            audio_reconstruction_loss = None

        if torch.count_nonzero(image_hidden_states) != 0:
            *_, output_images, image_reconstruction_loss = self.output_transform(
                image_hidden_states=image_hidden_states,
                image_labels=image_labels,
            )
        else:
            # output dummy images so huggingface data parallelization doesn't have a conniption
            output_images = torch.zeros_like(image_raw_inputs)
            image_reconstruction_loss = None

        text_loss = None
        if labels is not None:
            shift_logits = output_text_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            text_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        total_loss = None
        if text_loss is not None:
            total_loss = text_loss

        if audio_reconstruction_loss is not None:
            if total_loss is None:
                total_loss = audio_reconstruction_loss
            else:
                total_loss = total_loss + audio_reconstruction_loss

        if image_reconstruction_loss is not None:
            if total_loss is None:
                total_loss = image_reconstruction_loss
            else:
                total_loss = total_loss + image_reconstruction_loss
        
        if not return_dict:
            outputs = (
                output_text_logits,
                output_audio,
                output_images,
                *transformer_outputs[1:]
            )
            outputs = ((total_loss,) + outputs) if total_loss is not None else outputs
        else:
            outputs = megatransformer_utils.MegaTransformerMultimodalOutput(
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
        return outputs
    
    def generate(
        self,
        input_ids=None,
        audio_raw_inputs=None,
        image_raw_inputs=None,
        attention_mask=None,
        audio_waveform_labels=None,
        max_length=None,
        min_length=None,
        do_sample=None,
        early_stopping=None,
        num_beams=None,
        audio_override_ddim_sampling_steps=None,
        image_override_ddim_sampling_steps=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        bad_words_ids=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        length_penalty=None,
        no_repeat_ngram_size=None,
        encoder_no_repeat_ngram_size=None,
        num_return_sequences=None,
        max_time=None,
        output_scores=None,
        return_dict_in_generate=None,
        **model_kwargs,
    ):
        """
        Custom generation function that handles multimodal tokens.
        """
        # Set generation parameters if not provided
        max_length = max_length if max_length is not None else self.config.max_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        
        # Initialize generation tracking
        batch_size = input_ids.shape[0]
        current_length = input_ids.shape[1]
        
        # We'll build up these outputs during generation
        all_input_ids = input_ids.clone()

        model_inputs = self.prepare_inputs_for_generation(
            all_input_ids,
            **model_kwargs
        )
        past_key_values = model_inputs.get("past_key_values", None)

        inputs_embeds = self.interleave_batch_aligned_embeds(
            input_ids,
            audio_raw_inputs,
            image_raw_inputs,
            text_past_key_values=model_kwargs.get("past_key_values", None),
            audio_waveform_labels=audio_waveform_labels,
        )[0]

        all_audio_outputs = []
        all_image_outputs = []
        all_audio_mel_specs = []
        noise_outputs = []
        x_start_outputs = []
        
        # Tracking state for modal generation
        in_audio_generation = [False] * batch_size
        in_image_generation = [False] * batch_size
        current_modal_tokens = {}

        for i in range(batch_size):
            last_token = input_ids[i, -1].item()
            if last_token == self.config.begin_audio_token_id:
                in_audio_generation[i] = True
                current_modal_tokens[i] = []
            elif last_token == self.config.begin_image_token_id:
                in_image_generation[i] = True
                current_modal_tokens[i] = []
            else:
                in_audio_generation[i] = False
                in_image_generation[i] = False

        # Configure model kwargs for the first step
        model_kwargs["attention_mask"] = attention_mask.clone() if attention_mask is not None else None
        
        while current_length < max_length or any(in_audio_generation) or any(in_image_generation):
            outputs = self.world_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=model_inputs.get("use_cache", None),
                output_attentions=model_inputs.get("output_attentions", None),
                output_hidden_states=model_inputs.get("output_hidden_states", None),
                return_dict=True,
            )
            
            raw_multimodal_logits = outputs.logits[:, -1, :]

            text_token_logits = self.output_transform.text_decoder(raw_multimodal_logits)
            
            # Apply logit processing (temperature, top-k, top-p, etc.)
            text_token_logits = self.adjust_logits_during_generation(
                text_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
            )
            
            # Sample or greedy select next token
            if do_sample:
                # Temperature (optional, already applied in adjust_logits_during_generation)
                if temperature != 1.0:
                    text_token_logits = text_token_logits / temperature
                
                # Top-p/top-k filtering
                if top_k > 0 or top_p < 1.0:
                    text_token_logits = self.top_k_top_p_filtering(
                        text_token_logits, top_k=top_k, top_p=top_p
                    )
                
                # Sample
                probs = F.softmax(text_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(text_token_logits, dim=-1)
            
            # Check for special tokens and handle modal generation
            for i in range(batch_size):
                text_token_id = next_tokens[i].item()

                # Handle modal generation state transitions
                if not in_audio_generation[i] and not in_image_generation[i] and text_token_id == self.config.begin_audio_token_id:
                    in_audio_generation[i] = True
                    current_modal_tokens[i] = []
                elif in_audio_generation[i] and i in current_modal_tokens and (text_token_id == self.config.end_audio_token_id or len(current_modal_tokens[i]) > self.config.audio_max_frames):
                    in_audio_generation[i] = False

                    # Process collected audio tokens through audio generator
                    # add batch dimension and example dimension
                    audio_hidden_states = torch.cat(current_modal_tokens[i]).unsqueeze(0).unsqueeze(0)

                    audio_coda_outputs, _, _ = self.output_transform.audio_coda_forward(audio_hidden_states)

                    outputs = self.output_transform.audio_decoder.sample(
                        condition=audio_coda_outputs,
                        batch_size=1,
                        n_mels=self.config.audio_n_mels,
                        device=text_token_logits.device,
                        override_ddim_sampling_steps=audio_override_ddim_sampling_steps,
                    )
                    if isinstance(outputs, tuple):
                        audio_mel_specs, audio_waveforms = outputs
                    else:
                        # remove singleton channel dimension from mel specs
                        audio_waveforms = self.output_transform.audio_decoder.vocoder(audio_mel_specs.squeeze(1), audio_coda_outputs.permute(0, 2, 1))

                    all_audio_mel_specs.append(audio_mel_specs)

                    if torch.isnan(audio_waveforms).any():
                        print("NaN in audio waveforms")
                    if torch.isinf(audio_waveforms).any():
                        print("Inf in audio waveforms")

                    all_audio_outputs.append(audio_waveforms)

                    next_tokens[i] = self.config.end_audio_token_id
                elif not in_audio_generation[i] and not in_image_generation[i] and text_token_id == self.config.begin_image_token_id:
                    in_image_generation[i] = True
                    current_modal_tokens[i] = []
                elif in_image_generation[i] and i in current_modal_tokens and (text_token_id == self.config.end_image_token_id or len(current_modal_tokens[i]) >= (self.config.image_size // self.config.image_encoder_patch_size) ** 2):
                    in_image_generation[i] = False
                    
                    # Process collected image tokens through image generator
                    # add batch dimension and example dimension
                    noise_preds = torch.cat(current_modal_tokens[i]).unsqueeze(0).unsqueeze(0)

                    image_coda_outputs, _ = self.output_transform.image_coda_forward(noise_preds)

                    image, noise_preds, x_start_preds = self.output_transform.image_decoder.sample(
                        batch_size=1,
                        image_size=self.config.image_size,
                        device=image_coda_outputs.device,
                        condition=image_coda_outputs,
                        return_intermediate=True,
                        override_ddim_sampling_steps=image_override_ddim_sampling_steps,
                    )
                    all_image_outputs.append(image)
                    noise_outputs.append(noise_preds)
                    x_start_outputs.append(x_start_preds)

                    next_tokens[i] = self.config.end_image_token_id
                # Collect tokens for modal generation
                elif (in_audio_generation[i] or in_image_generation[i]) and i in current_modal_tokens:
                    current_modal_tokens[i].append(raw_multimodal_logits)
            
            # Add the next tokens to the growing sequences
            all_input_ids = torch.cat([all_input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Check if we've reached the desired length
            current_length = all_input_ids.shape[1]
            
            # Check for EOS
            if eos_token_id is not None and (next_tokens == eos_token_id).any():
                # Early stopping if all sequences have generated EOS
                if (next_tokens == eos_token_id).all():
                    break
        
        # Prepare outputs
        if return_dict_in_generate:
            return megatransformer_utils.MultimodalGenerationOutput(
                sequences=all_input_ids,
                audio_outputs=all_audio_outputs if all_audio_outputs else None,
                audio_mel_specs=all_audio_mel_specs if all_audio_mel_specs else None,
                image_outputs=all_image_outputs if all_image_outputs else None,
                intermediate_image_outputs=(noise_outputs, x_start_outputs) if noise_outputs else None,
            )
        else:
            return (all_input_ids, all_audio_outputs, all_audio_mel_specs, all_image_outputs, noise_outputs, x_start_outputs)

    def adjust_logits_during_generation(
        self,
        logits,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        no_repeat_ngram_size=None,
        bad_words_ids=None,
        **kwargs
    ):
        """Apply various processing to logits during generation."""
        # Apply repetition penalty
        if repetition_penalty is not None and repetition_penalty != 1.0:
            logits = self.enforce_repetition_penalty(logits, repetition_penalty)
        
        # Apply bad words filtering
        if bad_words_ids is not None:
            logits = self.filter_bad_words(logits, bad_words_ids)
        
        # Apply no repeat ngram
        if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
            logits = self.enforce_no_repeat_ngram(logits, no_repeat_ngram_size)
        
        return logits

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=1.0, filter_value=-float("Inf")):
        """Filter a distribution of logits using top-k and/or top-p (nucleus) filtering."""
        top_k = min(top_k, logits.size(-1))  # Safety check
        
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
            
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value
            
        return logits
    
    def enforce_repetition_penalty(self, logits, penalty):
        """Apply repetition penalty to logits."""
        # Implementation would penalize tokens that have already been generated
        return logits

    def filter_bad_words(self, logits, bad_words_ids):
        """Filter out bad words from generation."""
        # Implementation would set probabilities of bad words to very low values
        return logits

    def enforce_no_repeat_ngram(self, logits, no_repeat_ngram_size):
        """Prevent repeating n-grams."""
        # Implementation would check for repeating n-grams and penalize them
        return logits

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs):
        """Update the model kwargs for the next generation step."""
        # Update cache
        if "past_key_values" in outputs:
            model_kwargs["past_key_values"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past_key_values"] = outputs.mems
            
        # Update attention mask if needed
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            if attention_mask is not None and attention_mask.shape[1] < outputs.logits.shape[1]:
                new_attn_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                    dim=1
                )
                model_kwargs["attention_mask"] = new_attn_mask
                
        return model_kwargs

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        inputs = self.world_model.prepare_inputs_for_generation(input_ids, **kwargs)

        if "audio_raw_inputs" in kwargs:
            inputs["audio_raw_inputs"] = kwargs["audio_raw_inputs"]

        if "image_raw_inputs" in kwargs:
            inputs["image_raw_inputs"] = kwargs["image_raw_inputs"]

        return inputs


def make_audio_decoder(config):
    return megatransformer_audio_decoder.AudioConditionalGaussianDiffusion(
        config=config,  # for vocoder to grab its details from
        hidden_size=config.hidden_size,
        activation=config.audio_decoder_activation,
        scale_factor=(2, 1),
        stride=(2, 1),
        self_attn_class=megatransformer_audio_decoder.AudioDiffusionSelfAttentionBlock,
        cross_attn_class=megatransformer_audio_decoder.AudioDiffusionCrossAttentionBlock,
        norm_class=megatransformer_modules.RMSNorm,
        in_channels=1,
        model_channels=config.audio_decoder_model_channels,
        out_channels=1,
        time_embedding_dim=config.audio_decoder_time_embedding_dim,
        num_res_blocks=config.audio_decoder_num_res_blocks,
        has_condition=True,
        unet_dropout=config.audio_decoder_unet_dropout,
        betas_schedule=config.audio_decoder_betas_schedule,
        down_block_self_attn_n_heads=config.audio_decoder_down_block_self_attn_n_heads,
        down_block_self_attn_d_queries=config.audio_decoder_down_block_self_attn_d_queries,
        down_block_self_attn_d_values=config.audio_decoder_down_block_self_attn_d_values,
        down_block_self_attn_use_flash_attention=config.audio_decoder_down_block_self_attn_use_flash_attention,
        up_block_self_attn_n_heads=config.audio_decoder_up_block_self_attn_n_heads,
        up_block_self_attn_d_queries=config.audio_decoder_up_block_self_attn_d_queries,
        up_block_self_attn_d_values=config.audio_decoder_up_block_self_attn_d_values,
        up_block_self_attn_use_flash_attention=config.audio_decoder_up_block_self_attn_use_flash_attention,
        cross_attn_n_heads=config.audio_decoder_cross_attn_n_heads,
        cross_attn_d_queries=config.audio_decoder_cross_attn_d_queries,
        cross_attn_d_values=config.audio_decoder_cross_attn_d_values,
        cross_attn_use_flash_attention=config.audio_decoder_cross_attn_use_flash_attention,
    )

def make_image_decoder(config):
    return megatransformer_diffusion.GaussianDiffusion(
        config=config,
        hidden_size=config.hidden_size,
        activation=config.image_decoder_activation,
        scale_factor=(2, 2),
        stride=(2, 2),
        self_attn_class=megatransformer_image_decoder.ImageSelfAttentionBlock,
        cross_attn_class=megatransformer_image_decoder.ImageCrossAttentionBlock,
        norm_class=megatransformer_image_decoder.ImageRMSNorm,
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

def create_small_multimodal_model(tokenizer: PreTrainedTokenizer, max_position_embeddings):
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

    # uses a recurrent approach to emulate a deeper model (~317M params)
    config = megatransformer_utils.MegaTransformerConfig(
        vocab_size=tokenizer.vocab_size + 6,
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
        begin_voice_token_id=begin_voice_token_id,
        end_voice_token_id=end_voice_token_id,

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

    text_embedding = megatransformer_text_encoder.TextFeatureExtractor(config)
    audio_embedding = megatransformer_audio_encoder.AudioFeatureExtractor(config)
    image_embedding = megatransformer_image_encoder.ImageViTFeatureExtractor(config)
    world_model = megatransformer_recurrent.MegaTransformerRawEmbedsRecurrentCausalModel(config)
    text_decoder = nn.Linear(config.hidden_size, config.vocab_size)
    audio_decoder = make_audio_decoder(config)
    image_decoder = make_image_decoder(config)
    return MegaTransformerCausalWMHeads(config, text_embedding, audio_embedding, image_embedding, world_model, text_decoder, audio_decoder, image_decoder)

def create_medium_multimodal_model(tokenizer: PreTrainedTokenizer, max_position_embeddings):
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

    # uses a recurrent approach to emulate a deeper model (~946M params)
    config = megatransformer_utils.MegaTransformerConfig(
        vocab_size=tokenizer.vocab_size + 6,
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
        begin_voice_token_id=begin_voice_token_id,
        end_voice_token_id=end_voice_token_id,
        # defaults otherwise
    )

    config.text_prelude_config = config
    config.audio_prelude_config = config
    config.image_prelude_config = config

    config.text_coda_config = config
    config.audio_coda_config = config
    config.image_coda_config = config

    text_embedding = megatransformer_text_encoder.TextFeatureExtractor(config)
    audio_embedding = megatransformer_audio_encoder.AudioFeatureExtractor(config)
    image_embedding = megatransformer_image_encoder.ImageViTFeatureExtractor(config)
    world_model = megatransformer_recurrent.MegaTransformerRawEmbedsRecurrentCausalModel(config)
    text_decoder = nn.Linear(config.hidden_size, config.vocab_size)
    audio_decoder = make_audio_decoder(config)
    image_decoder = make_image_decoder(config)
    return MegaTransformerCausalWMHeads(config, text_embedding, audio_embedding, image_embedding, world_model, text_decoder, audio_decoder, image_decoder)

def create_normal_multimodal_model(tokenizer: PreTrainedTokenizer, max_position_embeddings):
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

    # uses a recurrent approach to emulate a deeper model (~13.4B params)
    config = megatransformer_utils.MegaTransformerConfig(
        vocab_size=tokenizer.vocab_size + 6,
        max_position_embeddings=max_position_embeddings,
        hidden_size=5280,
        d_queries=96,
        d_values=96,
        n_query_groups=55,
        n_heads=55,
        intermediate_size=21120,
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
        begin_voice_token_id=begin_voice_token_id,
        end_voice_token_id=end_voice_token_id,
    )

    config.text_prelude_config = config
    config.audio_prelude_config = config
    config.image_prelude_config = config

    config.text_coda_config = config
    config.audio_coda_config = config
    config.image_coda_config = config

    text_embedding = megatransformer_text_encoder.TextFeatureExtractor(config)
    audio_embedding = megatransformer_audio_encoder.AudioFeatureExtractor(config)
    image_embedding = megatransformer_image_encoder.ImageViTFeatureExtractor(config)
    world_model = megatransformer_recurrent.MegaTransformerRawEmbedsRecurrentCausalModel(config)
    text_decoder = nn.Linear(config.hidden_size, config.vocab_size)
    audio_decoder = make_audio_decoder(config)
    image_decoder = make_image_decoder(config)
    return MegaTransformerCausalWMHeads(config, text_embedding, audio_embedding, image_embedding, world_model, text_decoder, audio_decoder, image_decoder)

def create_test_tiny_multimodal_model(tokenizer: PreTrainedTokenizer, max_position_embeddings):
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
    )

    config.text_prelude_config = config
    config.audio_prelude_config = config
    config.image_prelude_config = config

    config.text_coda_config = config
    config.audio_coda_config = config
    config.image_coda_config = config

    text_embedding = megatransformer_text_encoder.TextFeatureExtractor(config)
    audio_embedding = megatransformer_audio_encoder.AudioFeatureExtractor(config)
    image_embedding = megatransformer_image_encoder.ImageViTFeatureExtractor(config)
    world_model = megatransformer_recurrent.MegaTransformerRawEmbedsRecurrentCausalModel(config)
    text_decoder = nn.Linear(config.hidden_size, config.vocab_size)
    audio_decoder = make_audio_decoder(config)
    image_decoder = make_image_decoder(config)
    return MegaTransformerCausalWMHeads(config, text_embedding, audio_embedding, image_embedding, world_model, text_decoder, audio_decoder, image_decoder)

class SumEmbeddings(nn.Module):
    def __init__(self, wte: nn.Embedding, wpe: nn.Embedding):
        super().__init__()
        self.wte = wte
        self.wpe = wpe

    def forward(self, input_ids: torch.Tensor, **kwargs):
        embeddings = self.wte(input_ids)
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)
        position_embeddings = self.wpe(position_ids)
        embeddings += position_embeddings
        return embeddings

def split_model(model: GPT2LMHeadModel, config):
    text_encoder = SumEmbeddings(model.transformer.wte, model.transformer.wpe)
    text_decoder = model.lm_head
    world_model = model.transformer
    return text_encoder, world_model, text_decoder

def create_frankenstein_model(tokenizer: PreTrainedTokenizer, max_position_embeddings):
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
        d_queries=8,
        d_values=8,
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
    )

    config.text_prelude_config = config
    config.audio_prelude_config = config
    config.image_prelude_config = config

    config.text_coda_config = config
    config.audio_coda_config = config
    config.image_coda_config = config

    gpt2_config = GPT2Config(
        vocab_size=tokenizer.vocab_size + 6,
        n_embd=config.hidden_size,
        n_layer=config.n_recurrent_layers,
        n_head=config.n_heads,
        n_inner=config.intermediate_size,
        activation_function=config.intermediate_activation,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=config.norm_eps,
        initializer_range=config.initializer_range,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    text_embedding = megatransformer_text_encoder.TextFeatureExtractor(config)
    # audio_embedding = megatransformer_audio_encoder.AudioFeatureExtractor(config)
    # image_embedding = megatransformer_image_encoder.ImageViTFeatureExtractor(config)
    world_model = megatransformer_recurrent.MegaTransformerRawEmbedsRecurrentCausalModel(config)
    text_decoder = nn.Linear(config.hidden_size, config.vocab_size)
    image_decoder = make_image_decoder(config)

    # gpt2 = GPT2LMHeadModel(gpt2_config)
    # text_embedding, world_model, text_decoder = split_model(gpt2, config)
    audio_embedding = megatransformer_audio_encoder.PreTrainedAudioFeatureExtractorWrapper(config)
    image_embedding = megatransformer_image_encoder.PreTrainedImageFeatureExtractorWrapper(config)
    audio_decoder = megatransformer_audio_decoder.PreTrainedAudioDecoderWrapper(config)
    image_decoder = megatransformer_image_decoder.PreTrainedImageDecoderWrapper(config)
    return MegaTransformerCausalWMHeads(config, text_embedding, audio_embedding, image_embedding, world_model, text_decoder, audio_decoder, image_decoder)    

lookup = {
    "normal_multimodal": create_normal_multimodal_model,
    "medium_multimodal": create_medium_multimodal_model,
    "small_multimodal": create_small_multimodal_model,
    "test_tiny_multimodal": create_test_tiny_multimodal_model,
    "frankenstein_multimodal": create_frankenstein_model,
}

def model_config_lookup(config):
    if config not in lookup:
        raise ValueError(f"Unknown model configuration: {config}")
    return lookup[config]


if __name__ == '__main__':
    from transformers import AutoTokenizer
    import unittest


    class TestMegaTransformerMultimodalUtilityFunctions(unittest.TestCase):
        def setUp(self):
            # text max sequence length
            self.max_position_embeddings = 32

            self.n_mels = 128
            # audio max sequence length
            self.audio_max_frames = 312

            # image conditions static sequence length
            self.image_size = 128
            self.patch_size = 16
            self.image_embeds_length = (self.image_size // self.patch_size) ** 2  # 64

            self.hidden_size = 8

            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.object_instance = create_test_tiny_multimodal_model(
                tokenizer=self.tokenizer,
                max_position_embeddings=32
            )

            self.begin_audio_token_id = self.object_instance.config.begin_audio_token_id
            self.end_audio_token_id = self.object_instance.config.end_audio_token_id

            self.begin_image_token_id = self.object_instance.config.begin_image_token_id
            self.end_image_token_id = self.object_instance.config.end_image_token_id

        def test_interleave_batch_aligned_embeds(self):
            text_input_ids = torch.tensor([
                [32, 32, 32, 32, self.tokenizer.eos_token_id, self.tokenizer.eos_token_id, self.tokenizer.eos_token_id, self.tokenizer.eos_token_id, self.tokenizer.eos_token_id],  # raw text example
                [self.object_instance.config.begin_audio_token_id, self.object_instance.config.end_audio_token_id, 32, 32, 32, 32, self.tokenizer.eos_token_id, self.tokenizer.eos_token_id, self.tokenizer.eos_token_id],  # audio transcription
                [32, 32, 32, 32, self.object_instance.config.begin_audio_token_id, self.object_instance.config.end_audio_token_id, self.tokenizer.eos_token_id, self.tokenizer.eos_token_id, self.tokenizer.eos_token_id],  # audio generation
                [self.object_instance.config.begin_image_token_id, self.object_instance.config.end_image_token_id, 32, 32, 32, 32, self.tokenizer.eos_token_id, self.tokenizer.eos_token_id, self.tokenizer.eos_token_id],  # image description
                [32, 32, 32, 32, self.object_instance.config.begin_image_token_id, self.object_instance.config.end_image_token_id, self.tokenizer.eos_token_id, self.tokenizer.eos_token_id, self.tokenizer.eos_token_id],  # image generation
                [32, 32, self.object_instance.config.begin_audio_token_id, self.object_instance.config.end_audio_token_id, 32, 32, self.tokenizer.eos_token_id, self.tokenizer.eos_token_id, self.tokenizer.eos_token_id],  # audio in the middle
                [32, 32, self.object_instance.config.begin_image_token_id, self.object_instance.config.end_image_token_id, 32, 32, self.tokenizer.eos_token_id, self.tokenizer.eos_token_id, self.tokenizer.eos_token_id],  # image in the middle
                [32, self.object_instance.config.begin_audio_token_id, self.object_instance.config.end_audio_token_id, 32, 32, self.object_instance.config.begin_image_token_id, self.object_instance.config.end_image_token_id, 32, self.tokenizer.eos_token_id],  # audio followed by text and then image
                [32, self.object_instance.config.begin_image_token_id, self.object_instance.config.end_image_token_id, 32, 32, self.object_instance.config.begin_audio_token_id, self.object_instance.config.end_audio_token_id, 32, self.tokenizer.eos_token_id],  # image followed by text and then audio
                [32, self.object_instance.config.begin_audio_token_id, self.object_instance.config.end_audio_token_id, 32, 32, self.object_instance.config.begin_audio_token_id, self.object_instance.config.end_audio_token_id, 32, self.tokenizer.eos_token_id],  # multiple audios
                [32, self.object_instance.config.begin_image_token_id, self.object_instance.config.end_image_token_id, 32, 32, self.object_instance.config.begin_image_token_id, self.object_instance.config.end_image_token_id, 32, self.tokenizer.eos_token_id],  # multiple images
            ]).to(torch.long)

            print(f"unpadded text_input_ids: {text_input_ids.shape}")
            text_input_ids = F.pad(text_input_ids, (0, self.max_position_embeddings - text_input_ids.shape[1]), value=self.tokenizer.pad_token_id)
            print(f"padded text_input_ids: {text_input_ids.shape}")

            # batch_size, seq_len
            self.assertEqual(tuple(text_input_ids.shape), (11, self.max_position_embeddings))

            audio_raw_inputs = torch.stack([
                torch.zeros((2, 1, self.n_mels, self.audio_max_frames)),  # text only so zeros
                # audio transcription
                torch.cat([
                    torch.randn((1, 1, self.n_mels, self.audio_max_frames)),
                    torch.zeros((1, 1, self.n_mels, self.audio_max_frames)),
                ], dim=0),
                # audio generation
                torch.cat([
                    torch.randn((1, 1, self.n_mels, self.audio_max_frames)),
                    torch.zeros((1, 1, self.n_mels, self.audio_max_frames)),
                ], dim=0),
                torch.zeros((2, 1, self.n_mels, self.audio_max_frames)),  # image description
                torch.zeros((2, 1, self.n_mels, self.audio_max_frames)),  # image generation
                # audio in the middle
                torch.cat([
                    torch.randn((1, 1, self.n_mels, self.audio_max_frames)),
                    torch.zeros((1, 1, self.n_mels, self.audio_max_frames)),
                ], dim=0),
                torch.zeros((2, 1, self.n_mels, self.audio_max_frames)),  # image in the middle
                # audio followed by text and then image
                torch.cat([
                    torch.randn((1, 1, self.n_mels, self.audio_max_frames)),
                    torch.zeros((1, 1, self.n_mels, self.audio_max_frames)),
                ], dim=0),
                # image followed by text and then audio
                torch.cat([
                    torch.randn((1, 1, self.n_mels, self.audio_max_frames)),
                    torch.zeros((1, 1, self.n_mels, self.audio_max_frames)),
                ], dim=0),
                # multiple audios
                torch.cat([
                    torch.randn((1, 1, self.n_mels, self.audio_max_frames)),
                    torch.randn((1, 1, self.n_mels, self.audio_max_frames)),
                ], dim=0),
                torch.zeros((2, 1, self.n_mels, self.audio_max_frames)),  # multiple images
            ], dim=0).to(torch.float32)

            # batch_size, n_audios, channels, n_mels, max_audio_frames
            self.assertEqual(tuple(audio_raw_inputs.shape), (11, 2, 1, self.n_mels, self.audio_max_frames))

            image_raw_inputs = torch.stack([
                torch.zeros((2, 3, self.image_size, self.image_size)),  # text only so zeros
                torch.zeros((2, 3, self.image_size, self.image_size)),  # audio transcription
                torch.zeros((2, 3, self.image_size, self.image_size)),  # audio generation
                # image description
                torch.cat([
                    torch.randn((1, 3, self.image_size, self.image_size)),
                    torch.zeros((1, 3, self.image_size, self.image_size)),
                ], dim=0),
                # image generation
                torch.cat([
                    torch.randn((1, 3, self.image_size, self.image_size)),
                    torch.zeros((1, 3, self.image_size, self.image_size)),
                ], dim=0),
                torch.zeros((2, 3, self.image_size, self.image_size)),  # audio in the middle
                # image in the middle
                torch.cat([
                    torch.randn((1, 3, self.image_size, self.image_size)),
                    torch.zeros((1, 3, self.image_size, self.image_size)),
                ], dim=0),
                # audio followed by text and then image
                torch.cat([
                    torch.randn((1, 3, self.image_size, self.image_size)),
                    torch.zeros((1, 3, self.image_size, self.image_size)),
                ]),
                # image followed by text and then audio
                torch.cat([
                    torch.randn((1, 3, self.image_size, self.image_size)),
                    torch.zeros((1, 3, self.image_size, self.image_size)),
                ], dim=0),
                torch.zeros((2, 3, self.image_size, self.image_size)),  # multiple audios
                # multiple images
                torch.cat([
                    torch.randn((1, 3, self.image_size, self.image_size)),
                    torch.randn((1, 3, self.image_size, self.image_size)),
                ]),
            ], dim=0).to(torch.float32)

            # batch_size, n_images, channels, height, width
            self.assertEqual(tuple(image_raw_inputs.shape), (11, 2, 3, self.image_size, self.image_size))

            embeddings, masks, audio_positions, image_positions, _, _, _ = self.object_instance.interleave_batch_aligned_embeds(
                text_input_ids=text_input_ids,
                audio_raw_inputs=audio_raw_inputs,
                image_raw_inputs=image_raw_inputs,
            )

            # batch returned will be padded to largest example's embeds length, which is text total length + audio + audio
            batch_max_sequence_length = self.max_position_embeddings + self.audio_max_frames + self.audio_max_frames

            self.assertEqual(tuple(embeddings.shape), (11, batch_max_sequence_length, self.hidden_size))

            # audio positions
            self.assertEqual(len(audio_positions), 11)
            self.assertEqual(len(audio_positions[0]), 0)  # text only
            self.assertEqual(len(audio_positions[1]), 1)
            self.assertEqual(audio_positions[1][0], (1, self.audio_max_frames))  # audio transcription

            self.assertEqual(len(audio_positions[2]), 1)
            self.assertEqual(audio_positions[2][0], (5, self.audio_max_frames))  # audio generation

            self.assertEqual(len(audio_positions[3]), 0)  # image description
            self.assertEqual(len(audio_positions[4]), 0)  # image generation

            self.assertEqual(len(audio_positions[5]), 1)
            self.assertEqual(audio_positions[5][0], (3, self.audio_max_frames))  # audio in the middle

            self.assertEqual(len(audio_positions[6]), 0)  # image in the middle

            self.assertEqual(len(audio_positions[7]), 1)
            self.assertEqual(audio_positions[7][0], (2, self.audio_max_frames))  # audio followed by text and then image

            self.assertEqual(len(audio_positions[8]), 1)
            self.assertEqual(audio_positions[8][0], (6+self.image_embeds_length, self.audio_max_frames))  # image followed by text and then audio

            self.assertEqual(len(audio_positions[9]), 2)
            self.assertEqual(audio_positions[9][0], (2, self.audio_max_frames))  # multiple audios
            self.assertEqual(audio_positions[9][1], (2+self.audio_max_frames+4, self.audio_max_frames))

            self.assertEqual(len(audio_positions[10]), 0)  # multiple images

            # image positions
            self.assertEqual(len(image_positions), 11)
            self.assertEqual(len(image_positions[0]), 0)  # text only

            self.assertEqual(len(image_positions[1]), 0)  # audio transcription

            self.assertEqual(len(image_positions[2]), 0)  # audio generation

            self.assertEqual(len(image_positions[3]), 1)
            self.assertEqual(image_positions[3][0], (1, self.image_embeds_length))  # image description

            self.assertEqual(len(image_positions[4]), 1)
            self.assertEqual(image_positions[4][0], (5, self.image_embeds_length))  # image generation

            self.assertEqual(len(image_positions[5]), 0)  # audio in the middle

            self.assertEqual(len(image_positions[6]), 1)
            self.assertEqual(image_positions[6][0], (3, self.image_embeds_length))  # image in the middle

            self.assertEqual(len(image_positions[7]), 1)
            self.assertEqual(image_positions[7][0], (6+self.audio_max_frames, self.image_embeds_length))  # audio followed by text and then image

            self.assertEqual(len(image_positions[8]), 1)
            self.assertEqual(image_positions[8][0], (2, self.image_embeds_length))  # image followed by text and then audio

            self.assertEqual(len(image_positions[9]), 0)  # multiple audios

            self.assertEqual(len(image_positions[10]), 2)
            self.assertEqual(image_positions[10][0], (2, self.image_embeds_length))
            self.assertEqual(image_positions[10][1], (2+self.image_embeds_length+4, self.image_embeds_length))  # multiple images

            text_hidden_states, audio_hidden_states, image_hidden_states = self.object_instance.extract_batch_aligned_multimodal_features(
                embeddings,
                self.max_position_embeddings,
                audio_positions=audio_positions,
                image_positions=image_positions,
            )

            self.assertEqual(tuple(text_hidden_states.shape), (11, self.max_position_embeddings, self.hidden_size))
            self.assertEqual(tuple(audio_hidden_states.shape), (11, 2, self.audio_max_frames, self.hidden_size))
            self.assertEqual(tuple(image_hidden_states.shape), (11, 2, self.image_embeds_length, self.hidden_size))

            # case of a batch with no audio or images
            text_input_ids = torch.tensor([
                [32, 32, 32, 32, self.tokenizer.eos_token_id],
                [self.tokenizer.eos_token_id, self.tokenizer.eos_token_id, self.tokenizer.eos_token_id, self.tokenizer.eos_token_id, self.tokenizer.eos_token_id]
            ]).to(torch.long)

            text_input_ids = torch.cat([text_input_ids, torch.zeros((text_input_ids.shape[0], self.max_position_embeddings - text_input_ids.shape[1]), dtype=torch.long)], dim=1)

            # batch_size, seq_len
            self.assertEqual(tuple(text_input_ids.shape), (2, self.max_position_embeddings))

            audio_raw_inputs = torch.zeros((2, 1, 1, self.n_mels, self.audio_max_frames), dtype=torch.float32)

            # batch_size, n_audios, channels, n_mels, max_audio_frames
            self.assertEqual(tuple(audio_raw_inputs.shape), (2, 1, 1, self.n_mels, self.audio_max_frames))

            image_raw_inputs = torch.zeros((2, 1, 3, self.image_size, self.image_size), dtype=torch.float32)

            # batch_size, n_images, channels, height, width
            self.assertEqual(tuple(image_raw_inputs.shape), (2, 1, 3, self.image_size, self.image_size))

            embeddings, masks, audio_positions, image_positions, _, _, _ = self.object_instance.interleave_batch_aligned_embeds(
                text_input_ids=text_input_ids,
                audio_raw_inputs=audio_raw_inputs,
                image_raw_inputs=image_raw_inputs,
            )

            # batch returned will be padded to largest example's embeds length, which is text total length because no modal inputs
            batch_max_sequence_length = self.max_position_embeddings
            self.assertEqual(tuple(embeddings.shape), (2, batch_max_sequence_length, self.hidden_size))

            self.assertEqual(len(audio_positions), 2)
            self.assertEqual(len(audio_positions[0]), 0)
            self.assertEqual(len(audio_positions[1]), 0)

            self.assertEqual(len(image_positions), 2)
            self.assertEqual(len(image_positions[0]), 0)
            self.assertEqual(len(image_positions[1]), 0)

            text_hidden_states, audio_hidden_states, image_hidden_states = self.object_instance.extract_batch_aligned_multimodal_features(
                embeddings,
                self.max_position_embeddings,
                audio_positions=audio_positions,
                image_positions=image_positions,
            )

            self.assertEqual(tuple(text_hidden_states.shape), (2, self.max_position_embeddings, self.hidden_size))
            self.assertEqual(tuple(audio_hidden_states.shape), (2, 0, 0, self.hidden_size))
            self.assertEqual(tuple(image_hidden_states.shape), (2, 0, 0, self.hidden_size))

    unittest.main()
