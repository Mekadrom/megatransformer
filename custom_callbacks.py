from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from transformers import PreTrainedTokenizer, Trainer, TrainerCallback
from transformers.integrations import TensorBoardCallback
from typing import Any, Optional

from model import megatransformer_multimodal

import math
import numpy as np
import os
import torch
import torchaudio


def get_writer(trainer: Trainer):
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, TensorBoardCallback):
            if callback.tb_writer is not None:
                return callback.tb_writer
            
    return None


class GenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, prompts, generation_steps=2000):
        self.trainer: Optional[Trainer] = None
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.generation_steps = generation_steps
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if ((state.global_step == 1) or (state.global_step % self.generation_steps == 0)) and state.is_world_process_zero:
            writer = get_writer(self.trainer)
            if writer is None:
                print("No TensorBoard writer found, skipping generation...")
                return

            inputs = self.tokenizer(self.prompts, padding=True, return_tensors="pt").to(model.device)

            print(f"Generating text at step {state.global_step}...")
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=False,
                    max_length=100,
                    num_return_sequences=1,
                    do_sample=True,
                    top_p=0.92,
                    temperature=0.7,
                )
        
            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for i, text in enumerate(generated_texts):
                writer.add_text(f"generation/sample_{i}", text, state.global_step)

class MultimodalGenerationCallback(TrainerCallback):
    def __init__(self, tokenizer: PreTrainedTokenizer, text_only_prompts, generation_steps=2000):
        self.trainer: Optional[Trainer] = None
        self.tokenizer = tokenizer
        self.text_only_prompts = text_only_prompts
        self.generation_steps = generation_steps

        self.test_audio, self.sample_rate = torchaudio.load("test_alm.mp3")
        self.test_audio = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)(self.test_audio)
        self.test_audio_prompt_text = "It is from Westport, above the villages of Murrisk and Lecanvey."
        self.test_audio_prompt = tokenizer(self.test_audio_prompt_text, return_tensors="pt")

        self.test_image: Any = Image.open("test_vlm.png").convert("RGB")
        self.test_image = transforms.ToTensor()(self.test_image)
        self.test_image_prompt_text = "A man ironing a shirt while strapped to the back of a taxi."
        self.test_image_prompt = tokenizer(self.test_image_prompt_text, return_tensors="pt")

    def on_step_end(self, args, state, control, model: megatransformer_multimodal.MegaTransformerCausalWMHeads=None, **kwargs):
        if ((state.global_step == 1) or (state.global_step % self.generation_steps == 0)) and state.is_world_process_zero:
            writer = get_writer(self.trainer)
            if writer is None:
                print("No TensorBoard writer found, skipping generation...")
                return

            test_audio_prompt = self.test_audio_prompt.to(model.device)
            test_audio = self.test_audio.unsqueeze(0).to(model.device)

            test_image_prompt = self.test_image_prompt.to(model.device)
            test_image = self.test_image.unsqueeze(0).to(model.device)

            print(f"Generating at step {state.global_step}...")
            
            text_only_inputs = self.tokenizer(self.prompts, padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=text_only_inputs["input_ids"],
                    attention_mask=text_only_inputs["attention_mask"],
                    use_cache=False,
                    max_length=100,
                    num_return_sequences=1,
                    do_sample=True,
                    top_p=0.92,
                    temperature=0.7,
                )

                generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                for i, text in enumerate(generated_texts):
                    writer.add_text(f"generation/sample_{i}", text, state.global_step)

                # Generate audio
                audio_generation_outputs = model.generate(
                    input_ids=torch.cat([test_audio_prompt["input_ids"], torch.tensor(model.config.begin_audio_token_id).unsqueeze(0).to(model.device)], dim=1),
                    attention_mask=torch.cat([test_audio_prompt["attention_mask"], torch.ones(1, 1).to(model.device)], dim=1),
                    use_cache=False,
                    max_length=100,
                    num_return_sequences=1,
                    do_sample=True,
                    top_p=0.92,
                    temperature=0.7,
                    return_dict_in_generate=True,
                )

                audio_mel_specs = audio_generation_outputs.audio_outputs[0]

                # Transform from mel specs to audio
                audio_waveforms = self.reconstruct_audio_from_mel(
                    audio_mel_specs,
                    sample_rate=model.config.audio_decoder_sample_rate,
                    n_fft=model.config.audio_n_fft,
                    n_stft=model.config.audio_n_stft,
                    hop_length=model.config.audio_hop_length,
                    n_mels=model.config.audio_n_mels,
                )

                # Save audio
                audio_filepath = os.path.join(self.trainer.args.output_dir, f"generated_audio_step_{state.global_step}.wav")
                self.save_audio_to_file(
                    audio_waveforms[0],
                    audio_filepath,
                    sample_rate=model.config.audio_decoder_sample_rate,
                    normalize=True,
                )
                writer.add_text(
                    "generated_audio/prompt",
                    self.test_audio_prompt_text,
                    state.global_step,
                )
                writer.add_audio(
                    f"generated_audio/sample",
                    audio_waveforms[0],
                    state.global_step,
                    sample_rate=model.config.audio_decoder_sample_rate,
                )

                # transcribe audio
                audio_transcription_outputs = model.generate(
                    # interleave in generator will splice audio in between begin and end token embeddings
                    input_ids=torch.tensor([model.config.begin_audio_token_id, model.config.end_audio_token_id]).unsqueeze(0).to(model.device),
                    audio_raw_inputs=test_audio,
                    attention_mask=torch.ones(1, 1).to(model.device),
                    use_cache=False,
                    max_length=100,
                    num_return_sequences=2,
                    do_sample=True,
                    top_p=0.92,
                    temperature=0.7,
                )

                audio_transcription_texts = audio_transcription_outputs.sequences
                audio_transcription_texts = self.tokenizer.batch_decode(audio_transcription_texts, skip_special_tokens=True)
                writer.add_audio(
                    f"audio_transcription/sample",
                    self.test_audio,
                    state.global_step,
                    sample_rate=model.config.audio_decoder_sample_rate,
                )
                for i, text in enumerate(audio_transcription_texts):
                    writer.add_text(f"audio_transcription/sample_{i}", text, state.global_step)

                # Generate image
                image_generation_outputs = model.generate(
                    input_ids=torch.cat([test_image_prompt["input_ids"], torch.tensor(model.config.begin_image_token_id).unsqueeze(0).to(model.device)], dim=1),
                    attention_mask=torch.cat([test_image_prompt["attention_mask"], torch.ones(1, 1).to(model.device)], dim=1),
                    use_cache=False,
                    max_length=100,
                    num_return_sequences=1,
                    do_sample=True,
                    top_p=0.92,
                    temperature=0.7,
                    return_dict_in_generate=True,
                )
                image_generation_outputs = image_generation_outputs.image_outputs[0]
                # Save image
                image_filepath = os.path.join(self.trainer.args.output_dir, f"generated_image_step_{state.global_step}.png")
                image = transforms.ToPILImage()(image_generation_outputs[0].cpu())
                image.save(image_filepath)
                writer.add_image(
                    f"generated_image/sample",
                    image_generation_outputs[0].cpu(),
                    state.global_step,
                )
                writer.add_text(
                    "generated_image/prompt",
                    self.test_image_prompt_text,
                    state.global_step,
                )

                # transcribe image
                image_transcription_outputs = model.generate(
                    # interleave in generator will splice image in between begin and end token embeddings
                    input_ids=torch.tensor([model.config.begin_image_token_id, model.config.end_image_token_id]).unsqueeze(0).to(model.device),
                    image_raw_inputs=test_image,
                    attention_mask=torch.ones(1, 1).to(model.device),
                    use_cache=False,
                    max_length=100,
                    num_return_sequences=2,
                    do_sample=True,
                    top_p=0.92,
                    temperature=0.7,
                )
                image_transcription_texts = image_transcription_outputs.sequences
                image_transcription_texts = self.tokenizer.batch_decode(image_transcription_texts, skip_special_tokens=True)
                for i, text in enumerate(image_transcription_texts):
                    writer.add_text(f"image_transcription/sample_{i}", text, state.global_step)
                writer.add_image(
                    f"image_transcription/sample",
                    test_image,
                    state.global_step,
                )

                # holy shit this is a lot of code

    def reconstruct_audio_from_mel(
        self,
        mel_spec,
        sample_rate,
        n_fft,
        n_stft,
        hop_length,
        n_mels,
        power=2.0,
        is_log_scale=True,
        n_iter=32,
    ):
        # Ensure mel_spec is a torch tensor
        if not isinstance(mel_spec, torch.Tensor):
            mel_spec = torch.tensor(mel_spec)
        
        # Add channel dimension if needed
        if mel_spec.dim() == 2:
            mel_spec = mel_spec.unsqueeze(0)
        
        # If mel_spec is in log scale (dB), convert back to power
        if is_log_scale:
            mel_spec = torch.exp(mel_spec * 0.115129) # 10.0 / np.log(10.0) ≈ 4.342, then divided by 10 for dB
        
        # Define the inverse mel scale transform
        inverse_mel_scale = torchaudio.transforms.InverseMelScale(
            n_stft=n_stft,
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=0.0,
            f_max=sample_rate // 2,
            max_iter=200,  # Convergence iterations for the inverse mel transform
            tolerance_loss=1e-5,
            tolerance_change=1e-8,
        )
        
        # Apply inverse mel scale to get linear spectrogram
        linear_spec = inverse_mel_scale(mel_spec)
        
        # Define Griffin-Lim algorithm
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            power=power,
            n_iter=n_iter,
            momentum=0.99,
            rand_init=False,
        )
        
        # Apply Griffin-Lim to get audio
        waveform = griffin_lim(linear_spec)
        
        return waveform
    
    def save_audio_to_file(
        self,
        waveform, 
        filepath, 
        sample_rate, 
        normalize=True, 
        bits_per_sample=16
    ):
        # Ensure waveform is a torch tensor
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)
        
        # Add channel dimension if needed
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Normalize if requested
        if normalize:
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save audio file
        torchaudio.save(
            filepath,
            waveform,
            sample_rate,
            bits_per_sample=bits_per_sample,
            format="wav"
        )
        print(f"Audio saved to {filepath}")

    def log_audio_to_tensorboard(
        self,
        writer: SummaryWriter, 
        waveform, 
        tag, 
        global_step, 
        sample_rate=16000, 
        normalize=True,
        max_outputs=4
    ):
        # Ensure waveform is a torch tensor
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)
        
        # Add batch dimension if needed
        if waveform.dim() == 1:  # [samples]
            waveform = waveform.unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
        elif waveform.dim() == 2:  # [channels, samples] or [batch, samples]
            # Determine if the first dimension is channels or batch
            if waveform.size(0) <= 2:  # Likely channels
                waveform = waveform.unsqueeze(0)  # [1, channels, samples]
            else:  # Likely batch
                waveform = waveform.unsqueeze(1)  # [batch, 1, samples]
        
        # Limit batch size
        batch_size = min(waveform.size(0), max_outputs)
        
        # Normalize if requested (per sample in batch)
        if normalize:
            for i in range(batch_size):
                waveform[i] = waveform[i] / (torch.max(torch.abs(waveform[i])) + 1e-8)
        
        # Log each audio sample in the batch
        for i in range(batch_size):
            writer.add_audio(
                f"{tag}/{i}", 
                waveform[i],
                global_step, 
                sample_rate=sample_rate
            )
        
        print(f"Logged {batch_size} audio samples to TensorBoard with tag '{tag}'")

class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.trainer: Optional[Trainer] = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            print("No logs found, skipping...")
            return

        writer = get_writer(self.trainer)
        if writer is None:
            print("No TensorBoard writer found, skipping...")
            return

        self.add_perplexity(writer, logs, state)

        model = kwargs.get("model", None)
        tokenizer = kwargs.get("processing_class", None)
        if model is not None and tokenizer is not None:
            embedding_weights = model.get_input_embeddings().weight.data.clone().to(torch.float32).cpu().numpy()
            vocab = tokenizer.get_vocab()
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
            tokens = [token for token, _ in sorted_vocab]

            assert len(tokens) == embedding_weights.shape[0], f"Mismatch between tokens and embedding weights: {len(tokens)} vs {embedding_weights.shape[0]}"
            writer.add_embedding(
                mat=embedding_weights,
                metadata=tokens,
                tag='token_embeddings',
                global_step=state.global_step,
            )
        else:
            print("Model or tokenizer not found, skipping embedding logging...")

    def add_perplexity(self, writer, logs, state):
        is_eval = any(key.startswith("eval_") for key in logs.keys())
        loss_key = "eval_loss" if is_eval else "loss"

        if loss_key in logs:
            perplexity = math.exp(logs[loss_key])
            tag = "eval/perplexity" if is_eval else "train/perplexity"
            writer.add_scalar(tag, perplexity, state.global_step)
