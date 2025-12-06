from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor, WhisperForConditionalGeneration, WhisperProcessor
from dataset_loading import load_dataset
import megatransformer_utils
from model.audio.criteria import ASRFeatureLoss
import torch
import torch.nn as nn


class PreTrainedAudioDecoderWrapper(nn.Module):
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__()
        self.config = config

        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        loss_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").model.encoder

        self.loss_fn = ASRFeatureLoss(loss_model, sample_rate=config.audio_sample_rate)

        self.embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embeddings = torch.tensor(self.embeddings_dataset[7306]["xvector"]).unsqueeze(0)

        self.decoder_cast = nn.Linear(self.config.hidden_size, 80)
        self.encoder_cast = nn.Linear(self.config.hidden_size, 768)

        self.model.requires_grad_(False)
        self.vocoder.requires_grad_(False)
        self.speaker_embeddings.requires_grad_(False)

    def forward(self,
                mel_spec_labels,
                condition,
                waveform_labels,
                threshold: float = 0.5,
                minlenratio=0.0,
                maxlenratio=20.0,
                attention_mask=None):
        if attention_mask is None:
            # 1 for attending, 0 for pad
            encoder_attention_mask = torch.ones(condition.shape[:-1], device=condition.device, dtype=torch.long)
        else:
            encoder_attention_mask = attention_mask

        bsz = condition.size(0)

        encoder_last_hidden_state = self.encoder_cast(condition)
        
        maxlen = int(encoder_last_hidden_state.size(1) * maxlenratio / self.model.config.reduction_factor)
        minlen = int(encoder_last_hidden_state.size(1) * minlenratio / self.model.config.reduction_factor)

        # Start the output sequence with a mel spectrum that is all zeros.
        output_sequence = encoder_last_hidden_state.new_zeros(bsz, 1, self.model.config.num_mel_bins)

        spectrogram = []
        idx = 0
        result_spectrogram = {}

        while True and idx < self.model.config.max_speech_positions:
            idx += 1

            # megatransformer_utils.print_debug_tensor('output_sequence', output_sequence)
            # megatransformer_utils.print_debug_tensor('self.speaker_embeddings', self.speaker_embeddings)

            # Run the decoder prenet on the entire output sequence.
            decoder_hidden_states = self.model.speecht5.decoder.prenet(output_sequence, self.speaker_embeddings.expand(bsz, -1))
            # Run the decoder layers on the last element of the prenet output.
            decoder_out = self.model.speecht5.decoder.wrapped_decoder(
                hidden_states=decoder_hidden_states[:, -1:],
                attention_mask=None,
                encoder_hidden_states=encoder_last_hidden_state,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
                return_dict=True,
            )

            last_decoder_output = decoder_out.last_hidden_state.squeeze(1)

            # Predict the new mel spectrum for this step in the sequence.
            spectrum = self.model.speech_decoder_postnet.feat_out(last_decoder_output)
            spectrum = spectrum.view(bsz, self.model.config.reduction_factor, self.model.config.num_mel_bins)
            spectrogram.append(spectrum)

            # Extend the output sequence with the new mel spectrum.
            new_spectrogram = spectrum[:, -1, :].view(bsz, 1, self.model.config.num_mel_bins)
            output_sequence = torch.cat((output_sequence, new_spectrogram), dim=1)
            # Predict the probability that this is the stop token.
            prob = torch.sigmoid(self.model.speech_decoder_postnet.prob_out(last_decoder_output))

            if idx < minlen:
                continue
            else:
                # If the generation loop is less than maximum length time, check the ones in the batch that have met
                # the prob threshold. Otherwise, assume all have met thresholds and fill other spectrograms for the batch.
                if idx < maxlen:
                    meet_thresholds = torch.sum(prob, dim=-1) >= threshold
                    meet_indexes = torch.where(meet_thresholds)[0].tolist()
                else:
                    meet_indexes = range(len(prob))
                meet_indexes = [i for i in meet_indexes if i not in result_spectrogram]
                if len(meet_indexes) > 0:
                    spectrograms = torch.stack(spectrogram)
                    spectrograms = spectrograms.transpose(0, 1).flatten(1, 2)
                    spectrograms = self.model.speech_decoder_postnet.postnet(spectrograms)
                    for meet_index in meet_indexes:
                        result_spectrogram[meet_index] = spectrograms[meet_index]
                if len(result_spectrogram) >= bsz:
                    break

        spectrograms = [result_spectrogram[i] for i in range(len(result_spectrogram))]
        spectrogram = spectrograms[0] if bsz == 1 else torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
        outputs = self.vocoder(spectrogram)

        similarity_loss = self.loss_fn(outputs, waveform_labels)
        snr_loss = self.snr_loss(outputs, waveform_labels)
        total_loss = similarity_loss + 0.1 * snr_loss

        return spectrogram, total_loss, outputs

    def snr_loss(self, generated_audio, reference_audio):
        """Calculate SNR loss between generated and reference audio"""
        signal_power = torch.mean(reference_audio**2)
        noise_power = torch.mean((generated_audio - reference_audio)**2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-5))
        return -snr  # Negative because we want to maximize SNR

    def sample(self,
               condition,
               device,
               n_mels,
               threshold: float = 0.5,
               minlenratio=0.0,
               maxlenratio=20.0,
               attention_mask=None,
               **kwargs):
        if attention_mask is None:
            # 1 for attending, 0 for pad
            encoder_attention_mask = torch.ones(condition.shape[:-1], device=condition.device, dtype=torch.long)
        else:
            encoder_attention_mask = attention_mask

        bsz = condition.size(0)

        encoder_last_hidden_state = self.encoder_cast(condition)
        
        maxlen = 1876 # int(encoder_last_hidden_state.size(1) * maxlenratio / self.model.config.reduction_factor)
        minlen = int(encoder_last_hidden_state.size(1) * minlenratio / self.model.config.reduction_factor)

        # Start the output sequence with a mel spectrum that is all zeros.
        output_sequence = encoder_last_hidden_state.new_zeros(bsz, 1, self.model.config.num_mel_bins)

        spectrogram = []
        idx = 0
        result_spectrogram = {}

        while True:
            idx += 1

            # megatransformer_utils.print_debug_tensor('output_sequence', output_sequence)
            # megatransformer_utils.print_debug_tensor('self.speaker_embeddings', self.speaker_embeddings)

            # Run the decoder prenet on the entire output sequence.
            decoder_hidden_states = self.model.speecht5.decoder.prenet(output_sequence, self.speaker_embeddings.to(output_sequence.device))
            # Run the decoder layers on the last element of the prenet output.
            decoder_out = self.model.speecht5.decoder.wrapped_decoder(
                hidden_states=decoder_hidden_states[:, -1:],
                attention_mask=None,
                encoder_hidden_states=encoder_last_hidden_state,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=None,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )

            last_decoder_output = decoder_out.last_hidden_state.squeeze(1)

            # Predict the new mel spectrum for this step in the sequence.
            spectrum = self.model.speech_decoder_postnet.feat_out(last_decoder_output)
            spectrum = spectrum.view(bsz, self.model.config.reduction_factor, self.model.config.num_mel_bins)
            spectrogram.append(spectrum)

            # Extend the output sequence with the new mel spectrum.
            new_spectrogram = spectrum[:, -1, :].view(bsz, 1, self.model.config.num_mel_bins)
            output_sequence = torch.cat((output_sequence, new_spectrogram), dim=1)
            # Predict the probability that this is the stop token.
            prob = torch.sigmoid(self.model.speech_decoder_postnet.prob_out(last_decoder_output))

            if idx < minlen:
                continue
            else:
                # If the generation loop is less than maximum length time, check the ones in the batch that have met
                # the prob threshold. Otherwise, assume all have met thresholds and fill other spectrograms for the batch.
                # print(f"idx: {idx}, maxlen: {maxlen}, minlen: {minlen}")
                if idx < maxlen-1:
                    meet_thresholds = torch.sum(prob, dim=-1) >= threshold
                    meet_indexes = torch.where(meet_thresholds)[0].tolist()
                else:
                    meet_indexes = range(len(prob))
                meet_indexes = [i for i in meet_indexes if i not in result_spectrogram]
                if len(meet_indexes) > 0:
                    spectrograms = torch.stack(spectrogram)
                    spectrograms = spectrograms.transpose(0, 1).flatten(1, 2)
                    spectrograms = self.model.speech_decoder_postnet.postnet(spectrograms)
                    for meet_index in meet_indexes:
                        result_spectrogram[meet_index] = spectrograms[meet_index]
                if len(result_spectrogram) >= bsz:
                    break

        # print(f"result_spectrogram: {result_spectrogram}")

        spectrograms = [result_spectrogram[i] for i in range(len(result_spectrogram))]
        spectrogram = spectrograms[0] if bsz == 1 else torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
        outputs = self.vocoder(spectrogram)
        return spectrogram, outputs

class PreTrainedAudioFeatureExtractorWrapper(nn.Module):
    def __init__(self, config: megatransformer_utils.MegaTransformerConfig):
        super().__init__()
        self.config = config

        model_id = "openai/whisper-small"
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        self.model.requires_grad = False

    def forward(self, audio_raw_inputs, audio_waveform_labels):
        # audio_raw_inputs: [batch_size, channels, length]
        # audio_waveform_labels = audio_waveform_labels.permute(0, 2, 1)  # [batch_size, length, channels]
        audio_waveform_labels = audio_waveform_labels.squeeze(1)  # [batch_size, length]
        megatransformer_utils.print_debug_tensor('audio_waveform_labels', audio_waveform_labels)
        inputs = self.processor(audio_waveform_labels.detach().cpu().numpy(), sampling_rate=self.config.audio_sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        outputs = self.model(**inputs)
        audio_features = outputs.last_hidden_state
        return audio_features
