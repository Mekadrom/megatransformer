import gradio as gr
import soundfile as sf
import torch
import os

from datetime import datetime
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset


processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
vocoder = vocoder.to(device)
speaker_embeddings = speaker_embeddings.to(device)

def generate_speech(text):
    """
    Generate speech from text using the above model and vocoder.
    Args:
        text: Text to synthesize
    Returns:
        Path to generated audio file
    """
    try:
        inputs = processor(text=text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        
        speech = output.cpu().numpy().squeeze()
        sampling_rate = 16000
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs(os.path.join("inference", "generated"), exist_ok=True)
        output_path = os.path.join("inference", "generated", f"speech_{timestamp}.wav")

        sf.write(output_path, speech, sampling_rate)
        
        print(sampling_rate, speech)
        return (sampling_rate, speech)
    except Exception as e:
        raise RuntimeError(f"Error generating speech: {e}")

with gr.Blocks(title="Text-to-Speech") as demo:
    gr.Markdown("# Text-to-Speech")
    gr.Markdown("Enter text to generate natural-sounding speech.")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                lines=5,
                placeholder="Enter the text you want to convert to speech...",
                label="Text Input"
            )
            
            generate_button = gr.Button("Generate Speech")
            
        with gr.Column():
            audio_output = gr.Audio(
                label="Generated Speech",
                type="numpy"
            )
    
    generate_button.click(
        fn=generate_speech,
        inputs=[text_input],
        outputs=audio_output
    )
    
    gr.Examples(
        examples=[
            ["Hello, how are you?"],
            ["This is a test of the text-to-speech system."],
            ["I love programming in Python!"],
            ["Gradio makes it easy to create demos."],
        ],
        inputs=[text_input],
        outputs=audio_output,
        fn=generate_speech,
        cache_examples=True,
    )
    
    gr.Markdown("""
    ## Usage Instructions:
    1. Enter the text you want to convert to speech in the text box
    2. Click 'Generate Speech' to create audio
    
    ## Model Information:
    This demo uses the SpeechT5 model for text-to-speech synthesis.
    It generates natural-sounding English speech with speaker embedding options.
    """)

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False, allowed_paths=[os.path.join('inference', 'generated')])
