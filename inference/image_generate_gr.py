from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

import gradio as gr
import torch
import random
import os

# Create directory for saved images
os.makedirs("generated_images", exist_ok=True)

# Load SDXL model
def load_model():
    pipe = DiffusionPipeline.from_pretrained(
        "lambdalabs/miniSD-diffusers",
        torch_dtype=torch.float16,
    ).to(torch.bfloat16)
    
    # # Use DPM solver for faster inference
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    #     pipe.scheduler.config,
    #     algorithm_type="sde-dpmsolver++",
    #     use_karras_sigmas=True
    # )
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    
    # Enable memory efficient attention if using CUDA
    if device == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
    
    return pipe

# Load model
model = load_model()

def generate_image(
    prompt, 
    negative_prompt, 
    width, 
    height, 
    guidance_scale, 
    num_inference_steps, 
    seed
):
    """
    Generate an image using SDXL model
    
    Args:
        prompt: Text prompt for image generation
        negative_prompt: Negative prompt to specify what to avoid
        width: Image width (must be multiple of 8)
        height: Image height (must be multiple of 8)
        guidance_scale: How strictly to follow the prompt (higher = more literal)
        num_inference_steps: Number of denoising steps (more = higher quality, slower)
        seed: Random seed for reproducibility
        
    Returns:
        Generated image
    """
    try:
        # Make sure dimensions are multiples of 8
        width = width - (width % 8)
        height = height - (height % 8)
        
        # Set seed for reproducibility
        if seed == -1:
            seed = random.randint(0, 2147483647)
        generator = torch.Generator().manual_seed(seed)
        
        # Generate the image
        image = model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images[0]
        
        # Save the image
        timestamp = torch.cuda.current_device() if torch.cuda.is_available() else 0
        filename = os.path.join("inference", "generated", f"sdxl_{seed}_{timestamp}.png")
        image.save(filename)
        
        # Return image and generation settings
        return image, f"Settings: Steps: {num_inference_steps}, Scale: {guidance_scale}, Size: {width}x{height}, Seed: {seed}"
    
    except Exception as e:
        return None, f"Error generating image: {str(e)}"

# Define Gradio interface
with gr.Blocks(title="Stable Diffusion XL Image Generator") as demo:
    gr.Markdown("# Stable Diffusion XL Image Generator")
    gr.Markdown("Generate high-quality images using Stable Diffusion XL")
    
    with gr.Row():
        with gr.Column():
            # Input components
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter a detailed description of the image you want to generate...",
                lines=3
            )
            
            negative_prompt_input = gr.Textbox(
                label="Negative Prompt",
                placeholder="What you don't want to see in the image...",
                value="low quality, blurry, distorted, disfigured, bad anatomy, watermark",
                lines=2
            )
            
            with gr.Row():
                width_slider = gr.Slider(
                    minimum=64, maximum=1024, step=8, value=1024,
                    label="Width"
                )
                height_slider = gr.Slider(
                    minimum=64, maximum=1024, step=8, value=1024,
                    label="Height"
                )
            
            with gr.Row():
                guidance_slider = gr.Slider(
                    minimum=1.0, maximum=15.0, step=0.5, value=7.5,
                    label="Guidance Scale"
                )
                steps_slider = gr.Slider(
                    minimum=20, maximum=100, step=1, value=30,
                    label="Inference Steps"
                )
            
            seed_number = gr.Number(
                label="Seed (-1 for random)",
                value=-1
            )
            
            generate_button = gr.Button("Generate Image")
        
        with gr.Column():
            # Output components
            image_output = gr.Image(
                label="Generated Image",
                type="pil"
            )
            
            settings_output = gr.Textbox(
                label="Generation Settings",
                lines=1
            )
    
    # Set up the action when the button is clicked
    generate_button.click(
        fn=generate_image,
        inputs=[
            prompt_input, 
            negative_prompt_input, 
            width_slider, 
            height_slider, 
            guidance_slider, 
            steps_slider, 
            seed_number
        ],
        outputs=[image_output, settings_output]
    )
    
    # Examples
    gr.Examples(
        examples=[
            [
                "A breathtaking landscape with mountains, forest, and a serene lake, hyper-detailed, photorealistic, trending on artstation",
                "low quality, blurry, distorted", 
                1024, 768, 7.5, 30, 42
            ],
            [
                "Portrait of a cyberpunk character with neon lights, detailed facial features, futuristic background, cinematic lighting",
                "low quality, weird anatomy, extra limbs", 
                768, 1024, 9.0, 40, 123
            ],
        ],
        inputs=[
            prompt_input,
            negative_prompt_input,
            width_slider,
            height_slider,
            guidance_slider,
            steps_slider,
            seed_number
        ],
        outputs=[image_output, settings_output],
        fn=generate_image,
        cache_examples=True,
    )
    
    gr.Markdown("""
    ## Usage Tips:
    1. **Detailed prompts** work better than short ones
    2. **Style keywords** can help (photorealistic, oil painting, concept art, etc.)
    3. **Negative prompts** help avoid unwanted elements
    4. Use **higher guidance** for more literal interpretations (7-9 is good)
    5. **More steps** generally improves quality (30-50 is a good range)
    6. Save the **seed number** to recreate similar images later
    
    ## Model Information:
    This demo uses Stability AI's Stable Diffusion XL Base 1.0 (~900M parameters).
    Images are saved in the 'generated_images' folder.
    """)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False, allowed_paths=[os.path.join('inference', 'generated')])
