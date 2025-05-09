from PIL import Image
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from transformers import PreTrainedTokenizer, Trainer, TrainerCallback
from transformers.integrations import TensorBoardCallback
from typing import Any, Optional

from dataset_loading import audio_loading
from model import megatransformer_audio_decoder, megatransformer_image_decoder, megatransformer_multimodal

import librosa
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchaudio
import torchvision


def get_writer(trainer: Trainer):
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, TensorBoardCallback):
            if callback.tb_writer is not None:
                return callback.tb_writer
    return None


class GenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, prompts, step_offset, generation_steps=2000):
        self.trainer: Optional[Trainer] = None
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.generation_steps = generation_steps
        self.step_offset = step_offset
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        global_step = state.global_step + self.step_offset

        if ((global_step == 1) or (global_step % self.generation_steps == 0)) and state.is_world_process_zero:
            writer = get_writer(self.trainer)
            if writer is None:
                print("No TensorBoard writer found, skipping generation...")
                return

            inputs = self.tokenizer(self.prompts, padding=True, return_tensors="pt").to(model.device)

            print(f"Generating text at step {global_step}...")
            
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
                writer.add_text(f"generation/sample_{i}", text, global_step)

class MultimodalGenerationCallback(TrainerCallback):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 text_only_prompts,
                 step_offset,
                 audio_sample_rate: int = 16000,
                 audio_n_mels: int = 128,
                 audio_n_fft: int = 1024,
                 audio_hop_length: int = 512,
                 generation_steps=2000):
        self.trainer: Optional[Trainer] = None
        self.tokenizer = tokenizer
        self.text_only_prompts = text_only_prompts
        self.step_offset = step_offset
        self.generation_steps = generation_steps

        self.test_audio_waveforms, self.sample_rate = torchaudio.load(os.path.join('inference', 'examples', 'test_alm.mp3'))
        self.test_audio_waveforms = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)(self.test_audio_waveforms)
        self.test_audio_mels = audio_loading.extract_mels(
            self.test_audio_waveforms,
            audio_sample_rate,
            audio_n_mels,
            audio_n_fft,
            audio_hop_length,
        )[0]
        self.test_audio_prompt_text = "It is from Westport, above the villages of Murrisk and Lecanvey."
        self.test_audio_prompt = tokenizer(self.test_audio_prompt_text, return_tensors="pt")

        self.test_image: Any = Image.open(os.path.join('inference', 'examples', 'test_vlm1.png')).convert("RGB")
        self.test_image = transforms.ToTensor()(self.test_image)
        self.test_image_prompt_text = "A man ironing a shirt while strapped to the back of a taxi."
        self.test_image_prompt = tokenizer(self.test_image_prompt_text, return_tensors="pt")

    def on_step_end(self, args, state, control, model: megatransformer_multimodal.MegaTransformerCausalWMHeads=None, **kwargs):
        global_step = state.global_step + self.step_offset

        if ((global_step == 1) or (global_step % self.generation_steps == 0)) and state.is_world_process_zero:
            writer = get_writer(self.trainer)
            if writer is None:
                print("No TensorBoard writer found, skipping generation...")
                return

            test_audio_prompt = self.test_audio_prompt.to(model.device)
            test_audio = self.test_audio_mels.unsqueeze(0).unsqueeze(0).to(model.device)

            test_image_prompt = self.test_image_prompt.to(model.device)
            test_image = self.test_image.unsqueeze(0).unsqueeze(0).to(model.device)

            print(f"Generating at step {global_step}...")
            
            text_only_inputs = self.tokenizer(self.text_only_prompts, padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                with autocast('cuda' if model.device.type == 'cuda' else 'cpu', dtype=torch.bfloat16 if bool(args.bf16) else torch.float16 if args.fp16 else torch.float32):
                    outputs = model.generate(
                        input_ids=text_only_inputs["input_ids"],
                        attention_mask=text_only_inputs["attention_mask"],
                        use_cache=False,
                        max_length=100,
                        num_return_sequences=1,
                        do_sample=True,
                        top_p=0.92,
                        temperature=0.7,
                        return_dict_in_generate=False
                    )

                    generated_texts = self.tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
                    
                    for i, text in enumerate(generated_texts):
                        writer.add_text(f"generation/sample_{i}", text, global_step)

                    begin_audio_token = torch.tensor(model.config.begin_audio_token_id).unsqueeze(0).unsqueeze(0).to(model.device)

                    # Generate audio
                    audio_generation_outputs = model.generate(
                        input_ids=torch.cat([test_audio_prompt["input_ids"], begin_audio_token], dim=1),
                        attention_mask=torch.cat([test_audio_prompt["attention_mask"], torch.ones(1, 1).to(model.device)], dim=1),
                        use_cache=False,
                        max_length=100,
                        num_return_sequences=1,
                        do_sample=True,
                        top_p=0.92,
                        temperature=0.7,
                        return_dict_in_generate=True,
                    )

                    audio_mel_specs = audio_generation_outputs.audio_mel_specs[0].cpu()
                    audio_waveforms = audio_generation_outputs.audio_outputs[0].to(torch.float64).cpu()

                    # Save audio
                    audio_filepath = os.path.join(self.trainer.args.output_dir, f"generated_audio_step_{global_step}.wav")
                    self.save_audio_to_file(
                        audio_waveforms,
                        audio_filepath,
                        sample_rate=model.config.audio_sample_rate,
                        normalize=True,
                    )
                    writer.add_text(
                        "generated_audio/prompt",
                        self.test_audio_prompt_text,
                        global_step,
                    )

                    # clip waveforms
                    audio_waveforms = torch.clamp(audio_waveforms, -1.0, 1.0)

                    writer.add_audio(
                        f"generated_audio/sample",
                        audio_waveforms,
                        global_step,
                        sample_rate=model.config.audio_sample_rate,
                    )

                    try:
                        # after vocoder mel spec
                        writer.add_image(
                            f"generated_audio/waveform_mel_spec",
                            self.viz_waveform(audio_waveforms[0].squeeze(0).cpu().numpy(), model.config.audio_sample_rate),
                            global_step,
                        )
                    except Exception as e:
                        writer.add_text(
                            f"generated_audio/waveform_mel_spec/error",
                            f"ERROR: probably NaN in waveforms {e}",
                            global_step,
                        )

                    try:
                        # before vocoder mel spec
                        writer.add_image(
                            f"generated_audio/mel_spec",
                            self.viz_mels(audio_mel_specs[0].squeeze(0).cpu().numpy(), model.config.audio_sample_rate),
                            global_step,
                        )
                    except Exception as e:
                        writer.add_text(
                            f"generated_audio/mel_spec/error",
                            f"ERROR: probably librosa error {e}",
                            global_step,
                        )

                    # transcribe audio
                    audio_transcription_outputs = model.generate(
                        # interleave in generator will splice audio in between begin and end token embeddings
                        input_ids=torch.tensor([model.config.begin_audio_token_id, model.config.end_audio_token_id]).unsqueeze(0).to(model.device),
                        audio_raw_inputs=test_audio,
                        use_cache=False,
                        max_length=100,
                        num_return_sequences=2,
                        do_sample=True,
                        top_p=0.92,
                        temperature=0.7,
                        return_dict_in_generate=True,
                    )

                    audio_transcription_texts = audio_transcription_outputs.sequences
                    audio_transcription_texts = self.tokenizer.batch_decode(audio_transcription_texts, skip_special_tokens=True)
                    writer.add_audio(
                        f"audio_transcription/sample",
                        self.test_audio_waveforms,
                        global_step,
                        sample_rate=model.config.audio_sample_rate,
                    )
                    for i, text in enumerate(audio_transcription_texts):
                        writer.add_text(f"audio_transcription/sample_{i}", text, global_step)

                    # test just vocoder by inputing ground truth mel specs and taking output as waveform reconstruction
                    # no idea what the conditioning should be for this
                    if not isinstance(model.output_transform.audio_decoder, megatransformer_audio_decoder.PreTrainedAudioDecoderWrapper):
                        audio_waveforms = model.output_transform.audio_decoder.vocoder(test_audio.squeeze(1).view(-1, test_audio.shape[-2], test_audio.shape[-1]))
                        audio_waveforms = torch.clamp(audio_waveforms, -1.0, 1.0)[0].to(torch.float64).cpu()

                        audio_waveforms_filepath = os.path.join(self.trainer.args.output_dir, f"generated_audio_vocoder_step_{global_step}.wav")
                        self.save_audio_to_file(
                            audio_waveforms,
                            audio_waveforms_filepath,
                            sample_rate=model.config.audio_sample_rate,
                            normalize=True,
                        )

                        writer.add_audio(
                            f"generated_audio_vocoder/sample",
                            audio_waveforms,
                            global_step,
                            sample_rate=model.config.audio_sample_rate,
                        )

                    begin_image_token = torch.tensor(model.config.begin_image_token_id).unsqueeze(0).unsqueeze(0).to(model.device)

                    # Generate image
                    image_generation_outputs = model.generate(
                        input_ids=torch.cat([test_image_prompt["input_ids"], begin_image_token], dim=1),
                        attention_mask=torch.cat([test_image_prompt["attention_mask"], torch.ones(1, 1).to(model.device)], dim=1),
                        use_cache=False,
                        max_length=100,
                        num_return_sequences=1,
                        do_sample=True,
                        top_p=0.92,
                        temperature=0.7,
                        return_dict_in_generate=True,
                    )

                    # images are in shape (batch_size, channels, height, width)
                    image_output = image_generation_outputs.image_outputs[0].cpu()

                    # Save image
                    image_filepath = os.path.join(self.trainer.args.output_dir, f"generated_image_step_{global_step}.png")
                    image = transforms.ToPILImage()(image_output.squeeze(0))
                    image = image.convert("RGB")
                    image.save(image_filepath)
                    writer.add_image(
                        f"generated_image/sample",
                        image_output.squeeze(0),
                        global_step,
                    )

                    if image_generation_outputs.intermediate_image_outputs is not None:
                        noise_preds = image_generation_outputs.intermediate_image_outputs[0]
                        x_start_preds = image_generation_outputs.intermediate_image_outputs[1]
                        if noise_preds is not None and x_start_preds is not None:
                            noise_preds = noise_preds[0]
                            x_start_preds = x_start_preds[0]
                            if noise_preds is not None and x_start_preds is not None:
                                for i, noise_pred, x_start_pred in zip(reversed(range(len(noise_preds))), noise_preds, x_start_preds):
                                    noise_pred = noise_pred.cpu()
                                    x_start_pred = x_start_pred.cpu()

                                    noise_pred = torchvision.utils.make_grid(noise_pred, nrow=2, padding=1)
                                    x_start_pred = torchvision.utils.make_grid(x_start_pred, nrow=2, padding=1)

                                    writer.add_image(
                                        f"generated_image/timestep_{i}_noise",
                                        noise_pred.squeeze(0),
                                        global_step,
                                    )

                                    writer.add_image(
                                        f"generated_image/timestep_{i}_x_start",
                                        x_start_pred.squeeze(0),
                                        global_step,
                                    )

                    writer.add_text(
                        "generated_image/prompt",
                        self.test_image_prompt_text,
                        global_step,
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
                        return_dict_in_generate=True,
                    )
                    image_transcription_texts = image_transcription_outputs.sequences
                    image_transcription_texts = self.tokenizer.batch_decode(image_transcription_texts, skip_special_tokens=True)
                    for i, text in enumerate(image_transcription_texts):
                        writer.add_text(f"image_transcription/sample_{i}", text, global_step)
                    
                    writer.add_image(
                        f"image_transcription/sample",
                        test_image[0].squeeze(0).cpu(),
                        global_step,
                    )

                    # holy shit this is a lot of code

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
            waveform.cpu(),
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

    def viz_waveform(self, audio_array, sr=16000, n_mels=128):
        """
        Generate a visualization of a mel spectrogram for TensorBoard
        
        Parameters:
        audio_array: numpy array of audio samples, shape [frames]
        sample_rate: audio sample rate in Hz
        n_mels: number of mel bands
        
        Returns:
        A numpy array with shape [1, height, width, 3] for TensorBoard
        """
        audio_array = np.clip(audio_array, -1.0, 1.0)

        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_array, 
            sr=sr,
            n_mels=n_mels,
            fmax=8000,
        )
        
        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return self.viz_mels(log_mel_spec, sr)

    def viz_mels(self, log_mel_spec, sr=16000):
        # Normalize to [0, 1] range for visualization
        log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
        
        # Create a figure with no padding
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(
            log_mel_spec, 
            x_axis='time',
            y_axis='mel', 
            sr=sr,
            fmax=8000,
            ax=ax
        )
        plt.colorbar(img, format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()
        
        # Convert figure to numpy array
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Close the figure to free memory
        plt.close(fig)
        
        # Add batch dimension
        data = np.expand_dims(data, axis=0)
        
        return data

class ImageGenerationCallback(TrainerCallback):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 step_offset,
                 generation_steps=2000):
        self.trainer: Optional[Trainer] = None
        self.tokenizer = tokenizer
        self.step_offset = step_offset
        self.generation_steps = generation_steps

        self.test_image1: Any = Image.open(os.path.join('inference', 'examples', 'test_vlm1_x256.png')).convert("RGB")
        self.test_image2: Any = Image.open(os.path.join('inference', 'examples', 'test_vlm2_x256.png')).convert("RGB")
        self.test_images = [
            transforms.ToTensor()(self.test_image1),
            transforms.ToTensor()(self.test_image1),
            transforms.ToTensor()(self.test_image1),
            transforms.ToTensor()(self.test_image2),
        ]
        self.test_image_prompt_text = [
            "A man ironing a shirt while strapped to the back of a taxi on a busy street.",
            "A man strapped to the back of a taxi ironing a shirt on a busy street.",
            "A man on the back of a taxi ironing a shirt. The taxi is driving on a busy street.",
            "bedroom minimalist home interior storage for kids bedroom design",
        ]
        self.test_image_prompt = tokenizer(self.test_image_prompt_text, return_tensors="pt", padding=True)

    def on_step_end(self, args, state, control, model: megatransformer_image_decoder.ImageReconstructionSingleTaskModel=None, **kwargs):
        global_step = state.global_step + self.step_offset

        if ((global_step == 1) or (global_step % self.generation_steps == 0)) and state.is_world_process_zero:
            writer = get_writer(self.trainer)
            if writer is None:
                print("No TensorBoard writer found, skipping generation...")
                return

            print(f"Generating at step {global_step}...")

            if torch.distributed.is_initialized():
                device = torch.device(f"cuda:{torch.distributed.get_rank()}")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            
            with torch.no_grad():
                with autocast(device.type, dtype=torch.bfloat16 if bool(args.bf16) else torch.float16 if args.fp16 else torch.float32):
                    self.log_generate(model, state, writer, None, device, "linear")
                    if not isinstance(model.image_recon, megatransformer_image_decoder.ImageVAE):
                        self.log_generate(model, state, writer, 50, device, "ddim")
                    label_grid = torchvision.utils.make_grid(self.test_images, nrow=2, padding=1)
                    writer.add_image(
                        f"generated_image/label",
                        label_grid,
                        global_step,
                    )

    def log_generate(self, model: megatransformer_image_decoder.ImageReconstructionSingleTaskModel, state, writer, ddim_steps, device, sample_type):
        global_step = state.global_step + self.step_offset

        diffusion_generator = torch.Generator(device=device)
        diffusion_generator.manual_seed(42)  # same seed for all samples throughout training

        # Generate image
        image_generation_outputs = model.generate(
            input_ids=self.test_image_prompt["input_ids"].to(device),
            attention_mask=self.test_image_prompt["attention_mask"].to(device),
            num_samples=4,
            override_ddim_sampling_steps=ddim_steps,
            use_cache=False,
            return_dict_in_generate=True,
            diffusion_generator=diffusion_generator,
        )

        # images are in shape (batch_size, channels, height, width)
        image_output = image_generation_outputs.image_outputs[0].cpu()
        noise_preds, x_start_preds = image_generation_outputs.intermediate_image_outputs

        image_output = torchvision.utils.make_grid(image_output, nrow=2, padding=1)

        # Save image
        image_filepath = os.path.join(self.trainer.args.output_dir, f"generated_image_step_{global_step}_{sample_type}.png")
        image = transforms.ToPILImage()(image_output.squeeze(0).to(torch.float32))
        image = image.convert("RGB")
        image.save(image_filepath)

        writer.add_image(
            f"generated_image/sample_{sample_type}",
            image_output.squeeze(0),
            global_step,
        )

        if noise_preds is not None and x_start_preds is not None:
            for i, noise_pred, x_start_pred in zip(reversed(range(len(noise_preds))), noise_preds, x_start_preds):
                noise_pred = noise_pred.cpu()
                x_start_pred = x_start_pred.cpu()

                noise_pred = torchvision.utils.make_grid(noise_pred, nrow=2, padding=1)
                x_start_pred = torchvision.utils.make_grid(x_start_pred, nrow=2, padding=1)

                writer.add_image(
                    f"generated_image/timestep_{i}_{sample_type}_noise",
                    noise_pred.squeeze(0),
                    global_step,
                )

                writer.add_image(
                    f"generated_image/timestep_{i}_{sample_type}_x_start",
                    x_start_pred.squeeze(0),
                    global_step,
                )

        for i, text in enumerate(self.test_image_prompt_text):
            writer.add_text(
                f"generated_image/prompt_{i}_{sample_type}",
                text,
                global_step,
            )

class MetricsCallback(TrainerCallback):
    def __init__(self, step_offset, is_add_perplexity=True):
        self.trainer: Optional[Trainer] = None
        self.step_offset = step_offset
        self.is_add_perplexity = is_add_perplexity

    def on_log(self, args, state, control, logs=None, **kwargs):
        global_step = state.global_step + self.step_offset

        if logs is None:
            print("No logs found, skipping...")
            return

        writer = get_writer(self.trainer)
        if writer is None:
            print("No TensorBoard writer found, skipping...")
            return

        if self.is_add_perplexity:
            self.add_perplexity(writer, logs, global_step)

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
                global_step=global_step,
            )
        else:
            print("Model or tokenizer not found, skipping embedding logging...")

    def add_perplexity(self, writer, logs, global_step):
        is_eval = any(key.startswith("eval_") for key in logs.keys())
        loss_key = "eval_loss" if is_eval else "loss"

        if loss_key in logs:
            perplexity = math.exp(logs[loss_key])
            tag = "eval/perplexity" if is_eval else "train/perplexity"
            writer.add_scalar(tag, perplexity, global_step)
