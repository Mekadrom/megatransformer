## TODO: this needs rewritten and cleaned up

import math
import os

os.environ["DEEPSPEED_UNIT_TEST"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ['NCCL_TIMEOUT'] = '1200000'

import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
import torchvision

from typing import Any, Optional

from PIL import Image
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, PreTrainedTokenizer, Trainer, TrainingArguments, TrainerCallback

from dataset_loading import audio_loading, multimodal_dataset
from model import causal, multimodal, recurrent
from utils import megatransformer_utils, training_utils
from utils.audio_utils import SharedWindowBuffer
from utils.model_loading_utils import load_model
from utils.training_utils import create_multimodal_optimizer, get_writer, trainer_lookup, setup_int8_training


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
                 shared_window_buffer: SharedWindowBuffer,
                 tokenizer: PreTrainedTokenizer,
                 text_only_prompts,
                 step_offset,
                 audio_sample_rate: int = 16000,
                 audio_n_mels: int = 80,
                 audio_n_fft: int = 1024,
                 audio_hop_length: int = 256,
                 generation_steps=2000):
        self.trainer: Optional[Trainer] = None
        self.tokenizer = tokenizer
        self.text_only_prompts = text_only_prompts
        self.step_offset = step_offset
        self.generation_steps = generation_steps

        self.test_audio_waveforms, self.sample_rate = torchaudio.load(os.path.join('inference', 'examples', 'test_alm_1.mp3'))
        self.test_audio_waveforms = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)(self.test_audio_waveforms)
        self.test_audio_mels = audio_loading.extract_mels(
            shared_window_buffer,
            self.test_audio_waveforms,
            audio_sample_rate,
            audio_n_mels,
            audio_n_fft,
            audio_hop_length,
        )[0]
        self.test_audio_prompt_text = "It is from Westport, above the villages of Murrisk and Lecanvey."
        self.test_audio_prompt = tokenizer(self.test_audio_prompt_text, return_tensors="pt")

        self.test_image: Any = Image.open(os.path.join('inference', 'examples', 'test_vlm1_x256.png')).convert("RGB")
        self.test_image = transforms.ToTensor()(self.test_image)
        self.test_image_prompt_text = "A man ironing a shirt while strapped to the back of a taxi."
        self.test_image_prompt = tokenizer(self.test_image_prompt_text, return_tensors="pt")

    def on_step_end(self, args, state, control, model: multimodal.MegaTransformerCausalWMHeads=None, **kwargs):
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
                with autocast(device_type=model.device.type, dtype=torch.bfloat16 if bool(args.bf16) else torch.float16 if args.fp16 else torch.float32):
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

            if len(tokens) != embedding_weights.shape[0]:
                print(f"Mismatch between tokens and embedding weights: {len(tokens)} vs {embedding_weights.shape[0]}, not logging embeddings")
            else:
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


args, unk = megatransformer_utils.parse_args()
run_dir = os.path.join(args.logging_base_dir, args.run_name)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, add_bos_token=False)
print(f"default tokenizer.padding_side: {tokenizer.padding_side}")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.bos_token = tokenizer.eos_token
tokenizer.padding_side = "right"

if len(args.include_modes) > 1 or "text" not in args.include_modes:
    # multimodal supports mixed mode training, but also single mode for audio transcription/generation and image description/generation
    model_maker = multimodal
elif 'recurrent' in args.config:
    model_maker = recurrent
else:
    model_maker = causal

shared_window_buffer = SharedWindowBuffer()

model = model_maker.model_config_lookup(args.config)(tokenizer, args.max_position_embeddings)
model, model_loaded = load_model(False, model, run_dir)

if args.local_rank == 0 or not args.use_deepspeed:
    print(f"model structure: {model}")
    print(f"model parameters: {(sum(p.numel() for p in model.parameters())):,}")
    print(f"trainable model parameters: {(sum(p.numel() for p in model.parameters() if p.requires_grad)):,}")

    if len(args.include_modes) > 1 or "text" not in args.include_modes:
        print(f"\tmodel.input_transform parameters: {(sum(p.numel() for p in model.input_transform.parameters())):,}")
        print(f"\t\tmodel.input_transform.text_embedding parameters: {(sum(p.numel() for p in model.input_transform.text_embedding.parameters())):,}")
        print(f"\t\t\tmodel.input_transform.text_embedding.wte parameters: {(sum(p.numel() for p in model.input_transform.text_embedding.wte.parameters())):,}")
        print(f"\t\t\tmodel.input_transform.text_embedding.prelude parameters: {(sum(p.numel() for p in model.input_transform.text_embedding.prelude.parameters())):,}")

        print(f"\t\tmodel.input_transform.audio_embedding parameters: {(sum(p.numel() for p in model.input_transform.audio_embedding.parameters())):,}")
        print(f"\t\t\tmodel.input_transform.audio_embedding.conv_feature_extractor parameters: {(sum(p.numel() for p in model.input_transform.audio_embedding.conv_feature_extractor.parameters())):,}")
        print(f"\t\t\tmodel.input_transform.audio_embedding.conv_projection parameters: {(sum(p.numel() for p in model.input_transform.audio_embedding.conv_projection.parameters())):,}")
        print(f"\t\t\tmodel.input_transform.audio_embedding.prelude parameters: {(sum(p.numel() for p in model.input_transform.audio_embedding.prelude.parameters())):,}")

        print(f"\t\tmodel.input_transform.image_embedding parameters: {(sum(p.numel() for p in model.input_transform.image_embedding.parameters())):,}")
        print(f"\t\t\tmodel.input_transform.image_embedding.patch_embed parameters: {(sum(p.numel() for p in model.input_transform.image_embedding.patch_embed.parameters())):,}")
        print(f"\t\t\tmodel.input_transform.image_embedding.prelude parameters: {(sum(p.numel() for p in model.input_transform.image_embedding.prelude.parameters())):,}")

        print(f"\tmodel.world_model parameters: {(sum(p.numel() for p in model.world_model.parameters())):,}")

        print(f"\tmodel.output_transform parameters: {(sum(p.numel() for p in model.output_transform.parameters())):,}")
        print(f"\t\tmodel.output_transform.text_coda parameters: {(sum(p.numel() for p in model.output_transform.text_coda.parameters())):,}")
        print(f"\t\tmodel.output_transform.text_decoder parameters: {(sum(p.numel() for p in model.output_transform.text_decoder.parameters())):,}")

        print(f"\t\tmodel.output_transform.audio_coda parameters: {(sum(p.numel() for p in model.output_transform.audio_coda.parameters())):,}")
        print(f"\t\tmodel.output_transform.audio_decoder parameters: {(sum(p.numel() for p in model.output_transform.audio_decoder.parameters())):,}")
        if hasattr(model.output_transform.audio_decoder, "vocoder"):
            print(f"\t\t\tmodel.output_transform.audio_decoder.vocoder parameters: {(sum(p.numel() for p in model.output_transform.audio_decoder.vocoder.parameters())):,}")
        print(f"\t\t\tmodel.output_transform.audio_decoder.unet parameters: {(sum(p.numel() for p in model.output_transform.audio_decoder.unet.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.audio_decoder.unet.time_transform parameters: {(sum(p.numel() for p in model.output_transform.audio_decoder.unet.time_transform.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.audio_decoder.unet.init_conv parameters: {(sum(p.numel() for p in model.output_transform.audio_decoder.unet.init_conv.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.audio_decoder.unet.down_blocks parameters: {(sum(p.numel() for p in model.output_transform.audio_decoder.unet.down_blocks.parameters())):,}")
        for d, down_block in enumerate(model.output_transform.audio_decoder.unet.down_blocks):
            print(f"\t\t\t\t\tmodel.output_transform.audio_decoder.unet.down_blocks[{d}] parameters: {(sum(p.numel() for p in down_block.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.audio_decoder.unet.middle_res_block parameters: {(sum(p.numel() for p in model.output_transform.audio_decoder.unet.middle_res_block.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.audio_decoder.unet.middle_attn_block parameters: {(sum(p.numel() for p in model.output_transform.audio_decoder.unet.middle_attn_block.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.audio_decoder.unet.middle_res_block2 parameters: {(sum(p.numel() for p in model.output_transform.audio_decoder.unet.middle_res_block2.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.audio_decoder.unet.up_blocks parameters: {(sum(p.numel() for p in model.output_transform.audio_decoder.unet.up_blocks.parameters())):,}")
        for u, up_block in enumerate(model.output_transform.audio_decoder.unet.up_blocks):
            print(f"\t\t\t\t\tmodel.output_transform.audio_decoder.unet.up_blocks[{u}] parameters: {(sum(p.numel() for p in up_block.parameters())):,}")
            print(f"\t\t\t\t\t\tmodel.output_transform.audio_decoder.unet.up_blocks[{u}].upsample parameters: {(sum(p.numel() for p in up_block.upsample.parameters())):,}")
            print(f"\t\t\t\t\t\tmodel.output_transform.audio_decoder.unet.up_blocks[{u}].res_blocks parameters: {(sum(p.numel() for p in up_block.res_blocks.parameters())):,}")
            print(f"\t\t\t\t\t\tmodel.output_transform.audio_decoder.unet.up_blocks[{u}].attn_blocks parameters: {(sum(p.numel() for p in up_block.attn_blocks.parameters())):,}")
            print(f"\t\t\t\t\t\tmodel.output_transform.audio_decoder.unet.up_blocks[{u}].cross_attn_blocks parameters: {(sum(p.numel() for p in up_block.cross_attn_blocks.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.audio_decoder.unet.final_res_block parameters: {(sum(p.numel() for p in model.output_transform.audio_decoder.unet.final_res_block.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.audio_decoder.unet.final_conv parameters: {(sum(p.numel() for p in model.output_transform.audio_decoder.unet.final_conv.parameters())):,}")

        print(f"\t\tmodel.output_transform.image_coda parameters: {(sum(p.numel() for p in model.output_transform.image_coda.parameters())):,}")
        print(f"\t\tmodel.output_transform.image_decoder parameters: {(sum(p.numel() for p in model.output_transform.image_decoder.parameters())):,}")
        print(f"\t\t\tmodel.output_transform.image_decoder.unet parameters: {(sum(p.numel() for p in model.output_transform.image_decoder.unet.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.image_decoder.unet.time_transform parameters: {(sum(p.numel() for p in model.output_transform.image_decoder.unet.time_transform.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.image_decoder.unet.init_conv parameters: {(sum(p.numel() for p in model.output_transform.image_decoder.unet.init_conv.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.image_decoder.unet.down_blocks parameters: {(sum(p.numel() for p in model.output_transform.image_decoder.unet.down_blocks.parameters())):,}")
        for d, down_block in enumerate(model.output_transform.image_decoder.unet.down_blocks):
            print(f"\t\t\t\t\tmodel.output_transform.image_decoder.unet.down_blocks[{d}] parameters: {(sum(p.numel() for p in down_block.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.image_decoder.unet.middle_res_block parameters: {(sum(p.numel() for p in model.output_transform.image_decoder.unet.middle_res_block.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.image_decoder.unet.middle_attn_block parameters: {(sum(p.numel() for p in model.output_transform.image_decoder.unet.middle_attn_block.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.image_decoder.unet.middle_res_block2 parameters: {(sum(p.numel() for p in model.output_transform.image_decoder.unet.middle_res_block2.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.image_decoder.unet.up_blocks parameters: {(sum(p.numel() for p in model.output_transform.image_decoder.unet.up_blocks.parameters())):,}")
        for u, up_block in enumerate(model.output_transform.image_decoder.unet.up_blocks):
            print(f"\t\t\t\t\tmodel.output_transform.image_decoder.unet.up_blocks[{u}] parameters: {(sum(p.numel() for p in up_block.parameters())):,}")
            print(f"\t\t\t\t\t\tmodel.output_transform.image_decoder.unet.up_blocks[{u}].upsample parameters: {(sum(p.numel() for p in up_block.upsample.parameters())):,}")
            print(f"\t\t\t\t\t\tmodel.output_transform.image_decoder.unet.up_blocks[{u}].res_blocks parameters: {(sum(p.numel() for p in up_block.res_blocks.parameters())):,}")
            print(f"\t\t\t\t\t\tmodel.output_transform.image_decoder.unet.up_blocks[{u}].attn_blocks parameters: {(sum(p.numel() for p in up_block.attn_blocks.parameters())):,}")
            print(f"\t\t\t\t\t\tmodel.output_transform.image_decoder.unet.up_blocks[{u}].cross_attn_blocks parameters: {(sum(p.numel() for p in up_block.cross_attn_blocks.parameters())):,}")
            for c, cross_attn_block in enumerate(up_block.cross_attn_blocks):
                print(f"\t\t\t\t\t\t\tmodel.output_transform.image_decoder.unet.up_blocks[{u}].cross_attn_blocks[{c}] parameters: {(sum(p.numel() for p in cross_attn_block.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.image_decoder.unet.final_res_block parameters: {(sum(p.numel() for p in model.output_transform.image_decoder.unet.final_res_block.parameters())):,}")
        print(f"\t\t\t\tmodel.output_transform.image_decoder.unet.final_conv parameters: {(sum(p.numel() for p in model.output_transform.image_decoder.unet.final_conv.parameters())):,}")

    print(f"modified tokenizer: {tokenizer}")
    print(f"special tokens: {tokenizer.special_tokens_map}")

    print(f"DeepSpeed config path: {args.deepspeed_config}")
    print(f"DeepSpeed enabled: {args.use_deepspeed}")
    print(f"XLA enabled: {args.use_xla}")

model = setup_int8_training(args, model)

if not os.path.exists(run_dir):
    os.makedirs(run_dir)

training_args = TrainingArguments(
    tpu_num_cores=8 if args.use_xla else None,
    output_dir=run_dir,
    overwrite_output_dir=True,
    lr_scheduler_type="cosine",
    warmup_ratio=args.warmup_ratio,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=1 if args.config == 'huginn' else args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    num_train_epochs=args.num_train_epochs if args.num_train_epochs > 0 else 1,
    max_steps=args.max_steps if args.max_steps > 0 else -1,
    weight_decay=args.weight_decay,
    report_to="tensorboard",
    logging_dir=run_dir,
    logging_steps=args.logging_steps,
    # eval_strategy="steps",
    eval_strategy="no",
    # eval_steps=args.eval_steps,
    save_safetensors=False,
    save_steps=args.save_steps,
    gradient_checkpointing=args.use_gradient_checkpointing,
    bf16=args.bf16,
    fp16=args.fp16,
    max_grad_norm=args.max_grad_norm,
    torch_compile=args.compile_model and not args.use_deepspeed and not args.use_xla,
    deepspeed=args.deepspeed_config if args.use_deepspeed and not args.use_xla else None,
    use_cpu=args.cpu,
    log_level=args.log_level,
    logging_first_step=True,
    local_rank=args.local_rank,
)

# print(f"Training arguments: {training_args}")

text_weight = 1.0 if "text" in args.include_modes else 0.0
audio_weight = 1.0 if "audio" in args.include_modes else 0.0
image_weight = 1.0 if "image" in args.include_modes else 0.0

train_dataset = multimodal_dataset.MultimodalDataset(
    model.config,
    approximated_length=300_000,
    tokenizer=tokenizer,
    sample_rate=model.config.audio_sample_rate,
    n_mels=model.config.audio_n_mels,
    n_fft=model.config.audio_n_fft,
    hop_length=model.config.audio_hop_length,
    audio_max_frames=model.config.audio_max_frames,
    image_size=model.config.image_size,
    cache_dir=args.dataset_cache_dir,
    text_weight=text_weight,
    audio_weight=audio_weight,
    image_weight=image_weight,
    split="train",
    seed=args.seed,
    max_position_embeddings=args.max_position_embeddings,
    shared_window_buffer=shared_window_buffer
)

validation_dataset = multimodal_dataset.MultimodalDataset(
    model.config,
    approximated_length=200_000,
    tokenizer=tokenizer,
    sample_rate=model.config.audio_sample_rate,
    n_mels=model.config.audio_n_mels,
    n_fft=model.config.audio_n_fft,
    hop_length=model.config.audio_hop_length,
    audio_max_frames=model.config.audio_max_frames,
    image_size=model.config.image_size,
    cache_dir=args.dataset_cache_dir,
    text_weight=0.0, # no text in validation
    audio_weight=audio_weight,
    image_weight=image_weight,
    split="validation",
    seed=args.seed,
    max_position_embeddings=args.max_position_embeddings,
)

data_collator: DataCollatorForLanguageModeling
if 'multimodal' in args.config.lower():
    data_collator = multimodal_dataset.DataCollatorForMultimodalLanguageModeling(
        tokenizer=tokenizer,
        max_position_embeddings=args.max_position_embeddings,
        image_size=model.config.image_size,
        audio_max_frames=model.config.audio_max_frames,
        audio_max_waveform_length=model.config.audio_max_waveform_length,
        modes=args.include_modes,
        mlm=False,
    )
else:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

optimizer = None
if "multimodal" in args.config.lower() and not "frankenstein" in args.config.lower():
    optimizer = create_multimodal_optimizer(model, args.weight_decay)

trainer = trainer_lookup(args, args.trainer)(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=tokenizer,
    optimizers=(optimizer, None)
)

model.config.include_modes = args.include_modes.split(",") if isinstance(args.include_modes, str) else args.include_modes

prompts = [
    "In this paper, we propose a novel approach to",
    "The Higgs boson, sometimes called the Higgs particle, is",
    "The capital of France is",
    "2 + 2 ="
]

generation_callback: TrainerCallback
if 'multimodal' in args.config.lower():
    generation_callback = training_utils.MultimodalGenerationCallback(
        tokenizer=tokenizer,
        text_only_prompts=prompts,
        step_offset=args.start_step,
        generation_steps=args.generation_steps,
    )
    trainer.add_callback(generation_callback)
    generation_callback.trainer = trainer
else:
    # todo: implement for multimodal
    generation_callback = training_utils.GenerationCallback(
        tokenizer=tokenizer,
        prompts=prompts,
        step_offset=args.start_step,
        generation_steps=args.generation_steps,
    )
    trainer.add_callback(generation_callback)
    generation_callback.trainer = trainer

metrics_callback = training_utils.MetricsCallback(step_offset=args.start_step)
trainer.add_callback(metrics_callback)
metrics_callback.trainer = trainer

print(f"Starting training with {sum(p.numel() for p in model.parameters()):,} parameters")
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
