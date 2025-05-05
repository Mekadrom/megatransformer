from transformers import WhisperProcessor, WhisperForConditionalGeneration

import gradio as gr
import numpy as np
import os
import torch

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def transcribe_audio(audio_file, task="transcribe", language=None):
    """
    Transcribe audio using Whisper Small model
    
    Args:
        audio_file: Path to audio file or tuple of (sample_rate, samples)
        task: Either "transcribe" or "translate" (translate to English)
        language: Source language code (optional, if None it will be auto-detected)
        
    Returns:
        Transcribed text
    """
    try:
        if isinstance(audio_file, tuple):
            sampling_rate, waveform = audio_file
            if len(waveform.shape) > 1 and waveform.shape[1] > 1:
                waveform = waveform.mean(axis=1)
        else:
            import librosa
            waveform, sampling_rate = librosa.load(audio_file, sr=16000)

        if not isinstance(waveform, np.ndarray):
            waveform = np.array(waveform)

        input_features = processor(
            waveform, 
            sampling_rate=sampling_rate, 
            return_tensors="pt"
        ).input_features.to(device)
        
        forced_decoder_ids = None
        if language is not None:
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language=language, task=task
            )
            
        predicted_ids = model.generate(
            input_features, 
            forced_decoder_ids=forced_decoder_ids,
            max_length=1024
        )
        
        transcription = processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription
    
    except Exception as e:
        raise RuntimeError(f"Error transcribing audio: {str(e)}")

with gr.Blocks(title="Whisper Small Audio Transcription") as demo:
    gr.Markdown("# Whisper Small Audio Transcription")
    gr.Markdown("Upload an audio file or record audio to get a transcription.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["upload", "microphone"], 
                type="filepath",
                label="Audio Input"
            )
            
            with gr.Row():
                task_dropdown = gr.Dropdown(
                    choices=["transcribe", "translate"], 
                    value="transcribe", 
                    label="Task"
                )
                language_dropdown = gr.Dropdown(
                    choices=[None, "en", "es", "fr", "de", "it", "pt", "nl", 
                             "ru", "zh", "ar", "hi", "ja", "ko"],
                    value=None, 
                    label="Source Language (optional)"
                )
            
            transcribe_button = gr.Button("Transcribe")
            
        with gr.Column():
            transcription_output = gr.Textbox(
                label="Transcription", 
                lines=10
            )
    
    transcribe_button.click(
        fn=transcribe_audio,
        inputs=[audio_input, task_dropdown, language_dropdown],
        outputs=transcription_output
    )
    
    gr.Examples(
        examples=[
            [os.path.join('inference', 'examples', 'test_alm.mp3'), "transcribe", None],
        ],
        inputs=[audio_input, task_dropdown, language_dropdown],
        outputs=transcription_output,
        fn=transcribe_audio,
        cache_examples=True,
    )
    
    gr.Markdown("""
    ## Usage Instructions:
    1. Upload an audio file or record using your microphone
    2. Select the task: 'transcribe' (keep original language) or 'translate' (to English)
    3. Optionally select the source language (if known), or leave as 'None' for auto-detection
    4. Click 'Transcribe' to process the audio
    
    ## Model Information:
    This demo uses OpenAI's Whisper Small model (~244M parameters).
    It supports multiple languages and can both transcribe and translate to English.
    """)

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False, allowed_paths=[os.path.join('inference', 'examples')])
