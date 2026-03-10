import torch
import torch.nn as nn

from model.world.world_model import MegaTransformerWorldModel
from scripts.train.trainer import CommonTrainer
from utils import model_loading_utils


class WorldModelTrainer(CommonTrainer):
    """
    Trainer for the multimodal world model.

    Handles the full forward pass through:
    1. Modality-specific feature extractors
    2. Token interleaving (text as driver with media placeholders)
    3. Recurrent transformer
    4. Token uninterleaving
    5. Modality-specific codas (predictions only)
    6. Loss computation per modality (in this trainer, not the model)

    Each modality is optional per-batch — the model gracefully handles batches
    where only some modalities are present.
    """

    def __init__(
        self,
        *args,
        cmdline: str = "",
        git_commit_hash: str = "",
        step_offset: int = 0,
        # Loss weights
        text_loss_weight: float = 1.0,
        audio_latent_loss_weight: float = 1.0,
        voice_latent_loss_weight: float = 1.0,
        image_latent_loss_weight: float = 1.0,
        # Modality flags
        include_text: bool = True,
        include_audio: bool = True,
        include_voice: bool = False,
        include_image: bool = True,
        # Whether data provides precomputed VAE latents
        precomputed_latents: bool = True,
        # Text loss label smoothing
        text_label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.cmdline = cmdline
        self.git_commit_hash = git_commit_hash
        self.step_offset = step_offset or 0

        self.text_loss_weight = text_loss_weight
        self.audio_latent_loss_weight = audio_latent_loss_weight
        self.voice_latent_loss_weight = voice_latent_loss_weight
        self.image_latent_loss_weight = image_latent_loss_weight

        self.include_text = include_text
        self.include_audio = include_audio
        self.include_voice = include_voice
        self.include_image = include_image
        self.precomputed_latents = precomputed_latents

        self.has_logged_cli = False

        # Shard-aware sampler for efficient data loading
        self._shard_sampler = None
        if hasattr(self.train_dataset, 'get_sampler'):
            self._shard_sampler = self.train_dataset.get_sampler(shuffle=True, seed=42)

        # GAN support stubs (required by CommonTrainer.is_gan_enabled)
        self.discriminator = None
        self.gan_already_started = False
        self.gan_start_condition_key = None
        self.gan_start_condition_value = None

        # Loss functions
        self.text_loss_fn = nn.CrossEntropyLoss(label_smoothing=text_label_smoothing)
        self.latent_l1_loss = nn.L1Loss()
        self.latent_mse_loss = nn.MSELoss()

        self.writer = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        global_step = self.state.global_step + self.step_offset

        self._ensure_tensorboard_writer()

        if not self.has_logged_cli and self.writer is not None:
            self.writer.add_text("training/command_line", self.cmdline, global_step)
            self.writer.add_text("training/git_commit_hash", self.git_commit_hash, global_step)
            self.has_logged_cli = True

        # ── Prepare inputs for world model forward ──────────────────────

        # Text: the collator produces text_token_ids [B, T] which include
        # boundary tokens (BOV, EOV, etc.) and placeholder tokens (VOICE_PH, etc.).
        # The interleaver replaces placeholder positions with media embeddings and
        # marks everything else as text. The text coda sees all tokens EXCEPT
        # placeholders.
        #
        # To build aligned targets: remove placeholders from the full sequence,
        # then do the standard causal shift. This ensures e.g. the target for BOV
        # is EOV (the next text token), not VOICE_PH.
        #
        # The model still receives the full text_input_ids (with placeholders)
        # because the interleaver needs them to locate media insertion points.
        text_input_ids = inputs.get("text_token_ids")  # [B, T]
        text_targets = None
        if text_input_ids is not None and self.include_text:
            from utils.constants import (
                AUDIO_PLACEHOLDER_TOKEN_ID,
                VOICE_PLACEHOLDER_TOKEN_ID,
                IMAGE_PLACEHOLDER_TOKEN_ID,
            )
            placeholder_ids = {AUDIO_PLACEHOLDER_TOKEN_ID, VOICE_PLACEHOLDER_TOKEN_ID, IMAGE_PLACEHOLDER_TOKEN_ID}

            # The model sees text_input_ids[:, :-1] as input (standard causal shift).
            # The interleaver removes placeholder positions, so the text coda
            # produces logits only at non-placeholder positions.
            #
            # For targets: remove placeholders from the FULL sequence, then shift.
            # This way BOV's target is EOV (next text token), not VOICE_PH.
            #
            # Example: full = [hello, world, BOV, VOICE_PH, EOV]
            #   Model input (:-1): [hello, world, BOV, VOICE_PH] → 3 text logits
            #   Full no-PH: [hello, world, BOV, EOV] → shift → [world, BOV, EOV] = 3 targets ✓

            full_ids = text_input_ids  # [B, T_full] before :-1

            # Remove placeholders from full sequence, then shift by 1
            non_ph_mask_full = torch.ones_like(full_ids, dtype=torch.bool)
            for pid in placeholder_ids:
                non_ph_mask_full &= (full_ids != pid)

            # Model input: keep placeholders for the interleaver
            text_input_ids = full_ids[:, :-1].contiguous()

            # Count non-PH positions in model input (= number of logits per item)
            non_ph_input = torch.ones_like(text_input_ids, dtype=torch.bool)
            for pid in placeholder_ids:
                non_ph_input &= (text_input_ids != pid)

            target_list = []
            for b in range(full_ids.shape[0]):
                clean = full_ids[b][non_ph_mask_full[b]]  # non-PH tokens in order
                shifted = clean[1:]  # causal shift: predict next non-PH token
                K = non_ph_input[b].sum().item()  # number of logits this item will produce
                # Truncate or pad targets to exactly K
                if shifted.shape[0] >= K:
                    target_list.append(shifted[:K])
                else:
                    target_list.append(torch.cat([shifted, shifted.new_zeros(K - shifted.shape[0])]))

            # Pad and stack targets across batch
            max_len = max(t.shape[0] for t in target_list)
            padded_targets = []
            for t in target_list:
                if t.shape[0] < max_len:
                    padded_targets.append(torch.cat([t, t.new_zeros(max_len - t.shape[0])]))
                else:
                    padded_targets.append(t)
            text_targets = torch.stack(padded_targets)  # [B, T_text]

        # Audio inputs: SIVE features shaped [B, C, T].
        # The world model expects (B, n_audio, C, T) where n_audio is the number
        # of audio clips per batch item. Since the dataset provides one clip per
        # item we unsqueeze to n_audio=1.
        audio_inputs = None
        audio_lengths = None
        audio_latent_labels = None
        if self.include_audio:
            audio_data = inputs.get("audio_features")  # [B, C, T]
            if audio_data is not None:
                audio_inputs = audio_data.unsqueeze(1)  # [B, 1, C, T]
                # Labels: squeeze n_audio dim to match coda output shape [B, C, T]
                audio_latent_labels = audio_data.clone()

                # Lengths: per-clip lengths, shape (B, n_audio=1)
                feat_lengths = inputs.get("audio_feature_lengths")
                if feat_lengths is not None:
                    audio_lengths = feat_lengths.unsqueeze(1)  # [B, 1]

        # Voice: same shape contract as audio (SIVE features). Separate placeholder token.
        voice_inputs = None
        voice_lengths = None
        voice_latent_labels = None
        if self.include_voice:
            voice_data = inputs.get("voice_features")  # [B, C, T]
            if voice_data is not None:
                voice_inputs = voice_data.unsqueeze(1)  # [B, 1, C, T]
                voice_latent_labels = voice_data.clone()

                voice_feat_lengths = inputs.get("voice_feature_lengths")
                if voice_feat_lengths is not None:
                    voice_lengths = voice_feat_lengths.unsqueeze(1)

        # Image inputs: collator provides image_images [B, C, H, W] (raw or latent).
        # World model expects (B, n_images, ...).
        image_inputs = None
        image_latent_labels = None
        if self.include_image:
            image_data = inputs.get("image_images")  # [B, C, H, W]
            if image_data is not None:
                image_inputs = image_data.unsqueeze(1)  # [B, 1, C, H, W]
                # Labels: keep without n_images dim to match coda output [B, C, H, W]
                image_latent_labels = image_data.clone()

        # ── Forward pass (predictions only, no loss) ────────────────────

        outputs = model(
            text_input_ids=text_input_ids,
            audio_inputs=audio_inputs,
            audio_lengths=audio_lengths,
            voice_inputs=voice_inputs,
            voice_lengths=voice_lengths,
            image_inputs=image_inputs,
            precomputed_latents=self.precomputed_latents,
            decode_outputs=False,
        )

        # ── Compute losses ──────────────────────────────────────────────

        device = text_input_ids.device if text_input_ids is not None else next(model.parameters()).device
        total_loss = torch.tensor(0.0, device=device)
        loss_components = {}

        # Text: cross-entropy on logits vs shifted targets (placeholders already removed)
        logits = outputs.get("logits")
        if logits is not None and text_targets is not None:
            B, T, V = logits.size()
            # Align logits and targets (may differ by at most 1 due to
            # uninterleaver padding vs target padding across batch items)
            T_min = min(T, text_targets.shape[1])
            logits = logits[:, :T_min, :].contiguous()
            text_targets = text_targets[:, :T_min].contiguous()
            B, T, V = logits.size()
            text_loss = self.text_loss_fn(
                logits.reshape(B * T, V),
                text_targets.reshape(B * T),
            )
            total_loss = total_loss + self.text_loss_weight * text_loss
            loss_components["text_loss"] = text_loss.detach()

        # Audio: L1 + MSE on latent predictions vs latent labels
        audio_latent_preds = outputs.get("audio_latent_preds")
        if audio_latent_preds is not None and audio_latent_labels is not None:
            audio_l1 = self.latent_l1_loss(audio_latent_preds, audio_latent_labels)
            audio_mse = self.latent_mse_loss(audio_latent_preds, audio_latent_labels)
            audio_loss = audio_l1 + audio_mse
            total_loss = total_loss + self.audio_latent_loss_weight * audio_loss
            loss_components["audio_latent_l1"] = audio_l1.detach()
            loss_components["audio_latent_mse"] = audio_mse.detach()

        # Voice: L1 + MSE (same as audio)
        voice_latent_preds = outputs.get("voice_latent_preds")
        if voice_latent_preds is not None and voice_latent_labels is not None:
            voice_l1 = self.latent_l1_loss(voice_latent_preds, voice_latent_labels)
            voice_mse = self.latent_mse_loss(voice_latent_preds, voice_latent_labels)
            voice_loss = voice_l1 + voice_mse
            total_loss = total_loss + self.voice_latent_loss_weight * voice_loss
            loss_components["voice_latent_l1"] = voice_l1.detach()
            loss_components["voice_latent_mse"] = voice_mse.detach()

        # Image: L1 + MSE on latent predictions vs latent labels
        image_latent_preds = outputs.get("image_latent_preds")
        if image_latent_preds is not None and image_latent_labels is not None:
            image_l1 = self.latent_l1_loss(image_latent_preds, image_latent_labels)
            image_mse = self.latent_mse_loss(image_latent_preds, image_latent_labels)
            image_loss = image_l1 + image_mse
            total_loss = total_loss + self.image_latent_loss_weight * image_loss
            loss_components["image_latent_l1"] = image_l1.detach()
            loss_components["image_latent_mse"] = image_mse.detach()

        # ── TensorBoard logging ─────────────────────────────────────────

        if self.writer is not None and global_step % self.args.logging_steps == 0:
            self._log_scalar("world/total_loss", total_loss, global_step)
            for name, value in loss_components.items():
                self._log_scalar(f"world/{name}", value, global_step)

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to route eval through compute_loss (same as training).

        The default Trainer.prediction_step calls model(**inputs), passing raw
        collator keys as kwargs. Our model expects different arg names, so we
        reuse compute_loss which handles the mapping.
        """
        model.eval()
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss, None, None)

    def start_train_print(self, args):
        model = self.model
        print(f"World model structure: {model.__class__.__name__}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"Active modalities: text={self.include_text}, audio={self.include_audio}, "
              f"voice={self.include_voice}, image={self.include_image}")
        print(f"Precomputed latents: {self.precomputed_latents}")
        print(f"Loss weights: text={self.text_loss_weight}, audio={self.audio_latent_loss_weight}, "
              f"voice={self.voice_latent_loss_weight}, image={self.image_latent_loss_weight}")


def load_model(args):
    return model_loading_utils.load_model(
        MegaTransformerWorldModel,
        args.config,
        checkpoint_path=args.resume_from_checkpoint,
    )


def create_trainer(
    args,
    model,
    optimizer,
    training_args,
    data_collator,
    train_dataset,
    eval_dataset,
):
    include_modes = [m.strip() for m in args.include_modes.split(",")]

    return WorldModelTrainer(
        model=model,
        optimizers=(optimizer, None),
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        cmdline=args.cmdline,
        git_commit_hash=args.commit_hash or "",
        step_offset=args.start_step,
        text_loss_weight=args.text_loss_weight,
        audio_latent_loss_weight=args.audio_latent_loss_weight,
        voice_latent_loss_weight=args.voice_latent_loss_weight,
        image_latent_loss_weight=args.image_latent_loss_weight,
        include_text="text" in include_modes,
        include_audio="audio" in include_modes,
        include_voice="voice" in include_modes,
        include_image="image" in include_modes,
        precomputed_latents=args.precomputed_latents,
        text_label_smoothing=args.text_label_smoothing,
    )


def add_cli_args(subparsers):
    sub_parser = subparsers.add_parser(
        "world", help="Train the multimodal world model (text + audio + image)"
    )

    # Data directories
    sub_parser.add_argument("--text_cache_dir", type=str, default=None,
                            help="Directory for preprocessed text shards")
    sub_parser.add_argument("--audio_cache_dir", type=str, default=None,
                            help="Directory for preprocessed audio shards")
    sub_parser.add_argument("--voice_cache_dir", type=str, default=None,
                            help="Directory for preprocessed voice shards")
    sub_parser.add_argument("--image_cache_dir", type=str, default=None,
                            help="Directory for preprocessed image shards")
    sub_parser.add_argument("--cache_dir", type=str, default=None,
                            help="Unused for world model — use per-modality cache dirs instead")

    # Modality loss weights
    sub_parser.add_argument("--text_loss_weight", type=float, default=1.0,
                            help="Weight for text classification loss")
    sub_parser.add_argument("--audio_latent_loss_weight", type=float, default=1.0,
                            help="Weight for audio latent prediction losses")
    sub_parser.add_argument("--voice_latent_loss_weight", type=float, default=1.0,
                            help="Weight for voice latent prediction losses")
    sub_parser.add_argument("--image_latent_loss_weight", type=float, default=1.0,
                            help="Weight for image latent prediction losses")

    # Text loss
    sub_parser.add_argument("--text_label_smoothing", type=float, default=0.0,
                            help="Label smoothing for text cross-entropy loss")

    # Precomputed latents flag
    sub_parser.add_argument("--precomputed_latents", action="store_true", default=True,
                            help="Whether media inputs are precomputed VAE latents (default: True)")
    sub_parser.add_argument("--no_precomputed_latents", action="store_false", dest="precomputed_latents",
                            help="Media inputs are raw (mel specs / images), not VAE latents")

    # Max sequence length for text collator
    sub_parser.add_argument("--max_seq_len", type=int, default=2048,
                            help="Maximum token sequence length for text")

    # Visualization callback dependencies
    sub_parser.add_argument("--vocoder_checkpoint_path", type=str, default=None,
                            help="Path to vocoder checkpoint for visualization")
    sub_parser.add_argument("--vocoder_config", type=str, default=None,
                            help="Vocoder config name (e.g. 'hifigan' for pretrained SpeechBrain HiFi-GAN, no checkpoint needed)")
    sub_parser.add_argument("--image_vae_decoder_path", type=str, default=None,
                            help="Path to image VAE decoder checkpoint (not needed for litevae)")
    sub_parser.add_argument("--image_vae_decoder_config", type=str, default=None,
                            help="Image VAE decoder config. Use 'litevae' for pretrained LiteVAE (auto-downloaded)")
    sub_parser.add_argument("--voice_cvae_checkpoint_path", type=str, default=None,
                            help="Path to voice CVAE decoder checkpoint for decoding SIVE latents to mel specs")
    sub_parser.add_argument("--voice_cvae_config", type=str, default="small",
                            help="Voice CVAE decoder config name")
    sub_parser.add_argument("--voice_cvae_latent_channels", type=int, default=None,
                            help="Override latent_channels for voice CVAE (must match what it was trained with)")
    sub_parser.add_argument("--static_speaker_embedding_path", type=str, default=None,
                            help="Path to a .pt file containing a speaker embedding tensor for static-speaker voice decoding")
    sub_parser.add_argument("--num_eval_samples", type=int, default=4,
                            help="Number of samples per visualization scenario")

    # Audio collator settings
    sub_parser.add_argument("--audio_max_seconds", type=float, default=10.0,
                            help="Maximum audio length in seconds")
    sub_parser.add_argument("--audio_sample_rate", type=int, default=16000,
                            help="Audio sample rate")
    sub_parser.add_argument("--audio_hop_length", type=int, default=256,
                            help="Audio hop length")
    sub_parser.add_argument("--sive_total_stride", type=int, default=4,
                            help="Total temporal downsampling stride of the SIVE encoder (e.g. 4 for 4x, 3 for 3x)")

    return sub_parser
