from io import BytesIO
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

import gradio as gr
import os
import requests
import torch

model_id = "Salesforce/blip2-opt-2.7b"  # ViT-G/14 variant
processor = Blip2Processor.from_pretrained(model_id)
model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def caption_image(image, prompt="", max_length=30, num_beams=5):
    """
    Generate a caption for the input image
    
    Args:
        image: Input image (PIL Image or path)
        prompt: Optional text prompt to guide captioning
        max_length: Maximum length of generated caption
        num_beams: Number of beams for beam search
        
    Returns:
        Generated caption
    """
    try:
        if isinstance(image, str):
            if image.startswith('http'):
                response = requests.get(image)
                pil_image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                pil_image = Image.open(image).convert('RGB')
        else:
            pil_image = image.convert('RGB')
        
        if prompt:
            inputs = processor(pil_image, prompt, return_tensors="pt").to(device, torch.float16)
        else:
            inputs = processor(pil_image, return_tensors="pt").to(device, torch.float16)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_beams=num_beams,
            )
        
        generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
        
        # Access intermediate features if needed
        # hidden_states = model.get_image_features(**inputs)
        return generated_text
    
    except Exception as e:
        raise RuntimeError(f"Error generating caption: {str(e)}")

with gr.Blocks(title="BLIP-2 Image Captioning") as demo:
    gr.Markdown("# BLIP-2 Image Captioning")
    gr.Markdown("Upload an image to get an automatic caption, or provide a prompt for guided captioning.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="pil", 
                label="Upload Image"
            )
            
            prompt_input = gr.Textbox(
                label="Optional Prompt (e.g., 'a photo of')", 
                placeholder="Leave empty for free-form captioning"
            )
            
            with gr.Row():
                max_length_slider = gr.Slider(
                    minimum=5, 
                    maximum=75, 
                    value=30, 
                    step=1, 
                    label="Max Caption Length"
                )
                
                num_beams_slider = gr.Slider(
                    minimum=1, 
                    maximum=10, 
                    value=5, 
                    step=1, 
                    label="Beam Search Size"
                )
            
            caption_button = gr.Button("Generate Caption")
        
        with gr.Column():
            caption_output = gr.Textbox(
                label="Generated Caption", 
                lines=3
            )
            
            with gr.Accordion("Advanced Output", open=False):
                # Extra output for features visualization if needed
                features_output = gr.Textbox(
                    label="Image Features (hidden states info)", 
                    visible=False
                )
    
    caption_button.click(
        fn=caption_image,
        inputs=[image_input, prompt_input, max_length_slider, num_beams_slider],
        outputs=caption_output
    )
    
    gr.Examples(
        examples=[
            [os.path.join('inference', 'examples', 'test_vlm1.png'), "a photo of", 30, 5],
            [os.path.join('inference', 'examples', 'test_vlm2.png'), "an image showing", 40, 5],
            [os.path.join('inference', 'examples', 'test_vlm1.png'), "", 30, 5],  # No prompt example
        ],
        inputs=[image_input, prompt_input, max_length_slider, num_beams_slider],
        outputs=caption_output,
        fn=caption_image,
        cache_examples=True,
    )
    
    gr.Markdown("""
    ## Usage Instructions:
    1. Upload an image using the upload button or by dragging and dropping
    2. Optionally provide a text prompt to guide the captioning (e.g., "a photo of", "an image showing")
    3. Adjust the maximum caption length and beam search parameters if desired
    4. Click "Generate Caption" to process the image
    
    ## Model Information:
    This demo uses Salesforce's BLIP-2 model with ViT-G/14 as the vision encoder (~400M parameters).
    BLIP-2 (Bootstrapping Language-Image Pre-training) is a state-of-the-art vision-language model
    that excels at image understanding tasks including captioning and visual question answering.
    """)

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False, allowed_paths=[os.path.join('inference', 'examples')])
