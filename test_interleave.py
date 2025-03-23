import torch
import unittest
from unittest.mock import MagicMock, patch
import torch.nn as nn

class TestMultimodalEmbedding(unittest.TestCase):
    def setUp(self):
        # Create a mock config with special token IDs
        self.config = MagicMock()
        self.config.begin_image_token_id = 50000
        self.config.end_image_token_id = 50001
        self.config.begin_audio_token_id = 50002
        self.config.end_audio_token_id = 50003
        
        # Mock the embedding dimensions
        self.hidden_size = 32
        
        # Create a mock input_transform
        self.input_transform = MagicMock()
        
        # Set up the text embedding behavior
        self.input_transform.text_embedding = MagicMock()
        def mock_text_embed(text_input_ids):
            if isinstance(text_input_ids, int) or (isinstance(text_input_ids, torch.Tensor) and text_input_ids.dim() == 0):
                # Single token case
                token = text_input_ids.item() if isinstance(text_input_ids, torch.Tensor) else text_input_ids
                # Create a unique embedding for each token
                embedding = torch.ones(1, self.hidden_size) * token / 10000
                return embedding
            else:
                # Batch of tokens case
                batch_size, seq_len = text_input_ids.shape
                embeddings = torch.zeros(batch_size, seq_len, self.hidden_size)
                for b in range(batch_size):
                    for s in range(seq_len):
                        embeddings[b, s] = torch.ones(self.hidden_size) * text_input_ids[b, s].item() / 10000
                return embeddings
        
        self.input_transform.text_embedding.side_effect = mock_text_embed
        
        # Mock the image embedding behavior
        self.input_transform.image_embedding = MagicMock()
        def mock_image_embed(images):
            num_images = images.shape[0]
            # Create a unique embedding for each image (single token for simplicity)
            embeddings = torch.zeros(num_images, 1, self.hidden_size)
            for i in range(num_images):
                # Use a value that's easily identifiable in tests
                embeddings[i, 0] = torch.ones(self.hidden_size) * (0.8 + i * 0.01)
            return embeddings
        
        self.input_transform.image_embedding.side_effect = mock_image_embed
        
        # Mock the audio embedding behavior
        self.input_transform.audio_embedding = MagicMock()
        def mock_audio_embed(audio):
            num_audio = audio.shape[0]
            # Create a unique embedding for each audio clip (single token for simplicity)
            embeddings = torch.zeros(num_audio, 1, self.hidden_size)
            for i in range(num_audio):
                # Use a value that's easily identifiable in tests
                embeddings[i, 0] = torch.ones(self.hidden_size) * (0.9 + i * 0.01)
            return embeddings
        
        self.input_transform.audio_embedding.side_effect = mock_audio_embed
        
        # Create the class with the interleave_embeds method
        class MultimodalModel(nn.Module):
            def __init__(self, input_transform, config):
                super().__init__()
                self.input_transform = input_transform
                self.config = config
            
            # Copy the implementation of interleave_embeds here
            def interleave_embeds(self, text_inputs_ids, image_raw_inputs, audio_raw_inputs):
                """
                Args:
                    text_inputs_ids: (batch_size, seq_length) containing text token IDs and special tokens
                    image_raw_inputs: (num_images, channels, height, width) 
                    audio_raw_inputs: (num_audio, channels, mels, frames)
                returns:
                    inputs_embeds: (batch_size, new_seq_length, hidden_size)
                """
                batch_size, seq_length = text_inputs_ids.shape
                device = text_inputs_ids.device
                
                # Process all modality embeddings
                text_embeds = self.input_transform.text_embedding(text_inputs_ids)  # [batch_size, seq_length, hidden_size]
                image_embeds = self.input_transform.image_embedding(image_raw_inputs)  # [num_images, img_tokens, hidden_size]
                audio_embeds = self.input_transform.audio_embedding(audio_raw_inputs)  # [num_audio, audio_tokens, hidden_size]
                
                # Create masks for special tokens
                is_image_token = (text_inputs_ids == self.config.begin_image_token_id)  # [batch_size, seq_length]
                is_audio_token = (text_inputs_ids == self.config.begin_audio_token_id)  # [batch_size, seq_length]
                
                # Count how many images and audio clips per example
                images_per_example = is_image_token.sum(dim=1)  # [batch_size]
                audio_per_example = is_audio_token.sum(dim=1)  # [batch_size]
                
                # Get indices of image and audio tokens in flattened batch
                image_indices = torch.nonzero(is_image_token, as_tuple=True)  # Returns (batch_indices, seq_indices)
                audio_indices = torch.nonzero(is_audio_token, as_tuple=True)  # Returns (batch_indices, seq_indices)
                
                # Reorder image and audio embeddings to match their order in the text
                image_batch_map = torch.zeros(images_per_example.sum().item(), dtype=torch.long, device=device)
                audio_batch_map = torch.zeros(audio_per_example.sum().item(), dtype=torch.long, device=device)
                
                # Fill image and audio mapping tensors
                if images_per_example.sum() > 0:
                    for i, (batch_idx, _) in enumerate(zip(*image_indices)):
                        image_batch_map[i] = batch_idx
                        
                if audio_per_example.sum() > 0:
                    for i, (batch_idx, _) in enumerate(zip(*audio_indices)):
                        audio_batch_map[i] = batch_idx
                
                # Create output embeddings list
                outputs = []
                
                # Process each batch item (still need one loop but over batch dimension only)
                img_counter = 0
                audio_counter = 0
                
                for b in range(batch_size):
                    # Find special token positions in this example
                    example_img_positions = (image_indices[0] == b).nonzero(as_tuple=True)[0]
                    example_audio_positions = (audio_indices[0] == b).nonzero(as_tuple=True)[0]
                    
                    # Extract token positions
                    img_seq_positions = image_indices[1][example_img_positions]  # positions in sequence
                    audio_seq_positions = audio_indices[1][example_audio_positions]  # positions in sequence
                    
                    # Current batch embeddings
                    batch_embeds = text_embeds[b]  # [seq_length, hidden_size]
                    
                    # Process example with special cases for each modality
                    
                    # Process sequence segments between special tokens
                    segments = []
                    last_pos = 0
                    
                    # Combine all special token positions and sort
                    all_special_pos = torch.cat([img_seq_positions, audio_seq_positions], dim=0)
                    if len(all_special_pos) > 0:
                        special_types = torch.cat([
                            torch.ones_like(img_seq_positions),  # 1 for images
                            torch.ones_like(audio_seq_positions) * 2  # 2 for audio
                        ], dim=0)
                        
                        # Sort by position
                        sorted_indices = torch.argsort(all_special_pos)
                        all_special_pos = all_special_pos[sorted_indices]
                        special_types = special_types[sorted_indices]
                        
                        # Process segments
                        for pos, type_id in zip(all_special_pos, special_types):
                            # Add text before this special token
                            if pos > last_pos:
                                segments.append(batch_embeds[last_pos:pos])
                            
                            # Add the special token embedding
                            if type_id == 1:  # Image
                                # Add image token + image embedding + end token
                                img_embed = image_embeds[img_counter]
                                img_counter += 1
                                begin_token_embed = self.input_transform.text_embedding(torch.tensor([self.config.begin_image_token_id], device=device).unsqueeze(0)).squeeze(0)
                                end_token_embed = self.input_transform.text_embedding(torch.tensor([self.config.end_image_token_id], device=device).unsqueeze(0)).squeeze(0)
                                segments.append(torch.cat([begin_token_embed, img_embed, end_token_embed], dim=0))
                            else:  # Audio
                                # Add audio token + audio embedding + end token
                                audio_embed = audio_embeds[audio_counter]
                                audio_counter += 1
                                begin_token_embed = self.input_transform.text_embedding(torch.tensor([self.config.begin_audio_token_id], device=device).unsqueeze(0)).squeeze(0)
                                end_token_embed = self.input_transform.text_embedding(torch.tensor([self.config.end_audio_token_id], device=device).unsqueeze(0)).squeeze(0)
                                segments.append(torch.cat([begin_token_embed, audio_embed, end_token_embed], dim=0))
                            
                            # Update last position (skip the special token)
                            last_pos = pos + 1
                        
                        # Add remaining text after last special token
                        if last_pos < seq_length:
                            segments.append(batch_embeds[last_pos:])
                    else:
                        # No special tokens, just use the text embeddings
                        segments.append(batch_embeds)
                    
                    # Concatenate all segments
                    outputs.append(torch.cat(segments, dim=0))
                
                # Pad sequences to the same length and stack
                max_length = max(x.size(0) for x in outputs)
                padded_outputs = []
                
                for emb in outputs:
                    if emb.size(0) < max_length:
                        padding = torch.zeros(max_length - emb.size(0), emb.size(1), device=device)
                        padded_outputs.append(torch.cat([emb, padding], dim=0))
                    else:
                        padded_outputs.append(emb)
                
                return torch.stack(padded_outputs, dim=0)
        
        # Initialize the model
        self.model = MultimodalModel(self.input_transform, self.config)

    def test_text_only(self):
        """Test with text only (no special tokens)"""
        # Create input with just text tokens
        text_inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
        image_inputs = torch.zeros((0, 3, 224, 224))  # Empty tensor
        audio_inputs = torch.zeros((0, 1, 80, 100))   # Empty tensor
        
        # Get embeddings
        outputs = self.model.interleave_embeds(text_inputs, image_inputs, audio_inputs)
        
        # Check output shape
        self.assertEqual(outputs.shape, (2, 3, self.hidden_size))
        
        # Check values - should be the original token embeddings
        for b in range(2):
            for s in range(3):
                expected_value = text_inputs[b, s].item() / 10000
                self.assertTrue(torch.allclose(outputs[b, s], torch.ones(self.hidden_size) * expected_value))

    def test_with_images(self):
        """Test with text and image tokens"""
        # Create input with text and image tokens
        text_inputs = torch.tensor([
            [1, 2, self.config.begin_image_token_id, 3],
            [4, self.config.begin_image_token_id, 5, 6]
        ])
        image_inputs = torch.zeros((2, 3, 224, 224))  # 2 images
        audio_inputs = torch.zeros((0, 1, 80, 100))   # No audio
        
        # Get embeddings
        outputs = self.model.interleave_embeds(text_inputs, image_inputs, audio_inputs)
        
        # Check output shape (each image adds 3 tokens: begin_token, image, end_token)
        # The shape should be (2, max_seq_length, hidden_size)
        # Original sequences: [1, 2, IMG, 3] and [4, IMG, 5, 6]
        # After expansion: [1, 2, IMG_BEGIN, IMG_EMBED, IMG_END, 3] and [4, IMG_BEGIN, IMG_EMBED, IMG_END, 5, 6]
        # Lengths: 6 and 6, so max_length = 6
        self.assertEqual(outputs.shape, (2, 6, self.hidden_size))
        
        # Check values for first batch
        # Sequence: [1, 2, IMG_BEGIN, IMG_EMBED, IMG_END, 3]
        self.assertTrue(torch.allclose(outputs[0, 0], torch.ones(self.hidden_size) * 0.0001))  # Token 1
        self.assertTrue(torch.allclose(outputs[0, 1], torch.ones(self.hidden_size) * 0.0002))  # Token 2
        self.assertTrue(torch.allclose(outputs[0, 2], torch.ones(self.hidden_size) * 5.0000))    # IMG_BEGIN (50000/10000)
        self.assertTrue(torch.allclose(outputs[0, 3], torch.ones(self.hidden_size) * 0.8))     # IMG_EMBED (first image)
        self.assertTrue(torch.allclose(outputs[0, 4], torch.ones(self.hidden_size) * 5.0001)) # IMG_END (50001/10000)
        self.assertTrue(torch.allclose(outputs[0, 5], torch.ones(self.hidden_size) * 0.0003))  # Token 3
        
        # Check values for second batch
        # Sequence: [4, IMG_BEGIN, IMG_EMBED, IMG_END, 5, 6]
        self.assertTrue(torch.allclose(outputs[1, 0], torch.ones(self.hidden_size) * 0.0004))  # Token 4
        self.assertTrue(torch.allclose(outputs[1, 1], torch.ones(self.hidden_size) * 5.0000))    # IMG_BEGIN
        self.assertTrue(torch.allclose(outputs[1, 2], torch.ones(self.hidden_size) * 0.81))    # IMG_EMBED (second image)
        self.assertTrue(torch.allclose(outputs[1, 3], torch.ones(self.hidden_size) * 5.0001)) # IMG_END
        self.assertTrue(torch.allclose(outputs[1, 4], torch.ones(self.hidden_size) * 0.0005))  # Token 5
        self.assertTrue(torch.allclose(outputs[1, 5], torch.ones(self.hidden_size) * 0.0006))  # Token 6

    def test_with_audio(self):
        """Test with text and audio tokens"""
        # Create input with text and audio tokens
        text_inputs = torch.tensor([
            [1, 2, self.config.begin_audio_token_id, 3],
            [4, 5, self.config.begin_audio_token_id, 6]
        ])
        image_inputs = torch.zeros((0, 3, 224, 224))  # No images
        audio_inputs = torch.zeros((2, 1, 80, 100))   # 2 audio clips
        
        # Get embeddings
        outputs = self.model.interleave_embeds(text_inputs, image_inputs, audio_inputs)
        
        # Check values for first batch
        # Sequence: [1, 2, AUDIO_BEGIN, AUDIO_EMBED, AUDIO_END, 3]
        self.assertTrue(torch.allclose(outputs[0, 0], torch.ones(self.hidden_size) * 0.0001))  # Token 1
        self.assertTrue(torch.allclose(outputs[0, 1], torch.ones(self.hidden_size) * 0.0002))  # Token 2
        self.assertTrue(torch.allclose(outputs[0, 2], torch.ones(self.hidden_size) * 5.0002), f"outputs[0, 2] was shape: {outputs[0, 2].shape} and value: {outputs[0, 2]}, but expected shape: {torch.ones(self.hidden_size).shape} and value: {torch.ones(self.hidden_size) * 5.0002}") # AUDIO_BEGIN
        self.assertTrue(torch.allclose(outputs[0, 3], torch.ones(self.hidden_size) * 0.9))     # AUDIO_EMBED (first audio)
        self.assertTrue(torch.allclose(outputs[0, 4], torch.ones(self.hidden_size) * 5.0003)) # AUDIO_END
        self.assertTrue(torch.allclose(outputs[0, 5], torch.ones(self.hidden_size) * 0.0003))  # Token 3

    def test_with_mixed_modalities(self):
        """Test with text, image, and audio tokens mixed"""
        # Create input with text, image, and audio tokens
        text_inputs = torch.tensor([
            [1, self.config.begin_image_token_id, 2, self.config.begin_audio_token_id, 3],
            [4, self.config.begin_audio_token_id, 5, self.config.begin_image_token_id, 6]
        ])
        image_inputs = torch.zeros((2, 3, 224, 224))  # 2 images
        audio_inputs = torch.zeros((2, 1, 80, 100))   # 2 audio clips
        
        # Get embeddings
        outputs = self.model.interleave_embeds(text_inputs, image_inputs, audio_inputs)
        
        # Check output shape
        # Sequence 1: [1, IMG_BEGIN, IMG_EMBED, IMG_END, 2, AUDIO_BEGIN, AUDIO_EMBED, AUDIO_END, 3]
        # Sequence 2: [4, AUDIO_BEGIN, AUDIO_EMBED, AUDIO_END, 5, IMG_BEGIN, IMG_EMBED, IMG_END, 6]
        # Both have length 9
        self.assertEqual(outputs.shape, (2, 9, self.hidden_size))
        
        # Check first sequence values at key positions
        self.assertTrue(torch.allclose(outputs[0, 0], torch.ones(self.hidden_size) * 0.0001))  # Token 1
        self.assertTrue(torch.allclose(outputs[0, 3], torch.ones(self.hidden_size) * 5.0001)) # IMG_END
        self.assertTrue(torch.allclose(outputs[0, 4], torch.ones(self.hidden_size) * 0.0002))  # Token 2
        self.assertTrue(torch.allclose(outputs[0, 6], torch.ones(self.hidden_size) * 0.9))     # AUDIO_EMBED
        
        # Check second sequence values at key positions
        self.assertTrue(torch.allclose(outputs[1, 0], torch.ones(self.hidden_size) * 0.0004))  # Token 4
        self.assertTrue(torch.allclose(outputs[1, 2], torch.ones(self.hidden_size) * 0.91))    # AUDIO_EMBED (second audio)
        self.assertTrue(torch.allclose(outputs[1, 4], torch.ones(self.hidden_size) * 0.0005))  # Token 5
        self.assertTrue(torch.allclose(outputs[1, 6], torch.ones(self.hidden_size) * 0.81))    # IMG_EMBED (second image)

    def test_consecutive_special_tokens(self):
        """Test with consecutive special tokens"""
        # Create input with consecutive special tokens
        text_inputs = torch.tensor([
            [1, self.config.begin_image_token_id, self.config.begin_image_token_id, 2],
        ])
        image_inputs = torch.zeros((2, 3, 224, 224))  # 2 images
        audio_inputs = torch.zeros((0, 1, 80, 100))   # No audio
        
        # Get embeddings
        outputs = self.model.interleave_embeds(text_inputs, image_inputs, audio_inputs)
        
        # Expected sequence: [1, IMG1_BEGIN, IMG1_EMBED, IMG1_END, IMG2_BEGIN, IMG2_EMBED, IMG2_END, 2]
        self.assertEqual(outputs.shape[1], 8)  # Check sequence length
        
        # Check values at key positions
        self.assertTrue(torch.allclose(outputs[0, 0], torch.ones(self.hidden_size) * 0.0001))  # Token 1
        self.assertTrue(torch.allclose(outputs[0, 2], torch.ones(self.hidden_size) * 0.8))     # First IMG_EMBED
        self.assertTrue(torch.allclose(outputs[0, 5], torch.ones(self.hidden_size) * 0.81))    # Second IMG_EMBED
        self.assertTrue(torch.allclose(outputs[0, 7], torch.ones(self.hidden_size) * 0.0002))  # Token 2

    def test_padding(self):
        """Test padding when sequences have different lengths after interleaving"""
        # Create input with different numbers of special tokens
        text_inputs = torch.tensor([
            [1, 2, 3, 4],
            [5, self.config.begin_image_token_id, self.config.begin_audio_token_id, 6]
        ])
        image_inputs = torch.zeros((1, 3, 224, 224))  # 1 image
        audio_inputs = torch.zeros((1, 1, 80, 100))   # 1 audio
        
        # Get embeddings
        outputs = self.model.interleave_embeds(text_inputs, image_inputs, audio_inputs)
        
        # Sequence 1: [1, 2, 3, 4] -> length 4
        # Sequence 2: [5, IMG_BEGIN, IMG_EMBED, IMG_END, AUDIO_BEGIN, AUDIO_EMBED, AUDIO_END, 6] -> length 8
        # After padding: max_length = 8
        self.assertEqual(outputs.shape, (2, 8, self.hidden_size))
        
        # Check values in first sequence (should have zeros for padding)
        for i in range(4):
            expected = (i + 1) / 10000
            self.assertTrue(torch.allclose(outputs[0, i], torch.ones(self.hidden_size) * expected))
        
        # Check padding values
        for i in range(4, 8):
            self.assertTrue(torch.allclose(outputs[0, i], torch.zeros(self.hidden_size)))

if __name__ == "__main__":
    unittest.main()
