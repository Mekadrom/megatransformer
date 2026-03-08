import torch

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
    5. Modality-specific codas with per-modality losses

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

        self.writer = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        global_step = self.state.global_step + self.step_offset

        self._ensure_tensorboard_writer()

        if not self.has_logged_cli and self.writer is not None:
            self.writer.add_text("training/command_line", self.cmdline, global_step)
            self.writer.add_text("training/git_commit_hash", self.git_commit_hash, global_step)
            self.has_logged_cli = True

        # ── Prepare inputs for world model forward ──────────────────────

        # Text: the collator produces text_token_ids [B, T] which become both
        # the input (with placeholder tokens) and the shifted target.
        text_input_ids = inputs.get("text_token_ids")  # [B, T]
        text_targets = None
        if text_input_ids is not None and self.include_text:
            # Standard causal LM target: shift right by 1
            text_targets = text_input_ids[:, 1:].contiguous()
            text_input_ids = text_input_ids[:, :-1].contiguous()

        # Audio inputs: the collator provides audio_features or audio_mel_specs.
        # For precomputed latent training we expect VAE latents in audio_features
        # shaped [B, C, H, T].  The world model expects (B, n_audio, C, H, T)
        # where n_audio is the number of audio clips per batch item.  Since the
        # multimodal dataset provides one clip per item we unsqueeze to n_audio=1.
        audio_inputs = None
        audio_lengths = None
        audio_latent_labels = None
        if self.include_audio:
            audio_data = inputs.get("audio_features")  # [B, C, T] or [B, C, H, T]
            if audio_data is not None:
                if audio_data.dim() == 3:
                    # [B, C, T] -> [B, 1, C, 1, T] — mono feature channel
                    audio_data = audio_data.unsqueeze(1).unsqueeze(3)
                elif audio_data.dim() == 4:
                    # [B, C, H, T] -> [B, 1, C, H, T]
                    audio_data = audio_data.unsqueeze(1)
                audio_inputs = audio_data
                audio_latent_labels = audio_data.clone()

                # Lengths: per-clip lengths, shape (B, n_audio=1)
                feat_lengths = inputs.get("audio_feature_lengths")
                if feat_lengths is not None:
                    audio_lengths = feat_lengths.unsqueeze(1)  # [B, 1]

            # Fall back to mel specs when features are unavailable
            if audio_inputs is None:
                mel_data = inputs.get("audio_mel_specs")  # [B, mel_bins, T]
                if mel_data is not None:
                    mel_data = mel_data.unsqueeze(1)  # [B, 1, mel_bins, T]
                    audio_inputs = mel_data
                    audio_latent_labels = mel_data.clone()
                    mel_lengths = inputs.get("audio_mel_lengths")
                    if mel_lengths is not None:
                        audio_lengths = mel_lengths.unsqueeze(1)

        # Voice: same shape contract as audio. Separate placeholder token.
        voice_inputs = None
        voice_lengths = None
        voice_latent_labels = None
        # Voice data would come from a voice shard dir; same keys with voice_ prefix.
        # Not wired yet, but the plumbing is ready.

        # Image inputs: collator provides image_images [B, 3, H, W].
        # World model expects (B, n_images, ...).
        image_inputs = None
        image_latent_labels = None
        if self.include_image:
            image_data = inputs.get("image_images")  # [B, 3, H, W]
            if image_data is not None:
                image_data = image_data.unsqueeze(1)  # [B, 1, 3, H, W]
                image_inputs = image_data
                image_latent_labels = image_data.clone()

        # ── Forward pass ────────────────────────────────────────────────

        outputs = model(
            text_input_ids=text_input_ids,
            audio_inputs=audio_inputs,
            audio_lengths=audio_lengths,
            audio_latent_labels=audio_latent_labels,
            voice_inputs=voice_inputs,
            voice_lengths=voice_lengths,
            voice_latent_labels=voice_latent_labels,
            image_inputs=image_inputs,
            image_latent_labels=image_latent_labels,
            text_targets=text_targets,
            precomputed_latents=self.precomputed_latents,
            decode_outputs=False,
        )

        # ── Aggregate losses ────────────────────────────────────────────

        total_loss = torch.tensor(0.0, device=text_input_ids.device)
        loss_components = {}

        # Text classification loss
        text_loss = outputs.get("text_classification_loss")
        if text_loss is not None:
            weighted_text = self.text_loss_weight * text_loss
            total_loss = total_loss + weighted_text
            loss_components["text_loss"] = text_loss.detach()

        # Audio latent losses (prefixed with "audio_" by AudioCodaAndVAEWithLoss)
        audio_l1 = outputs.get("audio_latent_l1_loss")
        audio_mse = outputs.get("audio_latent_mse_loss")
        if audio_l1 is not None and audio_mse is not None:
            audio_loss = audio_l1 + audio_mse
            weighted_audio = self.audio_latent_loss_weight * audio_loss
            total_loss = total_loss + weighted_audio
            loss_components["audio_latent_l1"] = audio_l1.detach()
            loss_components["audio_latent_mse"] = audio_mse.detach()

        # Voice latent losses
        voice_l1 = outputs.get("voice_latent_l1_loss")
        voice_mse = outputs.get("voice_latent_mse_loss")
        if voice_l1 is not None and voice_mse is not None:
            voice_loss = voice_l1 + voice_mse
            weighted_voice = self.voice_latent_loss_weight * voice_loss
            total_loss = total_loss + weighted_voice
            loss_components["voice_latent_l1"] = voice_l1.detach()
            loss_components["voice_latent_mse"] = voice_mse.detach()

        # Image latent losses
        image_l1 = outputs.get("image_latent_l1_loss")
        image_mse = outputs.get("image_latent_mse_loss")
        if image_l1 is not None and image_mse is not None:
            image_loss = image_l1 + image_mse
            weighted_image = self.image_latent_loss_weight * image_loss
            total_loss = total_loss + weighted_image
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
    sub_parser.add_argument("--image_cache_dir", type=str, default=None,
                            help="Directory for preprocessed image shards")
    sub_parser.add_argument("--cache_dir", type=str, default=None,
                            help="Unused for world model — use per-modality cache dirs instead")

    # Audio columns to load from shards
    sub_parser.add_argument("--audio_columns", type=str, default="features,mel_specs,text",
                            help="Comma-separated list of audio shard columns to load")

    # Modality loss weights
    sub_parser.add_argument("--text_loss_weight", type=float, default=1.0,
                            help="Weight for text classification loss")
    sub_parser.add_argument("--audio_latent_loss_weight", type=float, default=1.0,
                            help="Weight for audio latent prediction losses")
    sub_parser.add_argument("--voice_latent_loss_weight", type=float, default=1.0,
                            help="Weight for voice latent prediction losses")
    sub_parser.add_argument("--image_latent_loss_weight", type=float, default=1.0,
                            help="Weight for image latent prediction losses")

    # Precomputed latents flag
    sub_parser.add_argument("--precomputed_latents", action="store_true", default=True,
                            help="Whether media inputs are precomputed VAE latents (default: True)")
    sub_parser.add_argument("--no_precomputed_latents", action="store_false", dest="precomputed_latents",
                            help="Media inputs are raw (mel specs / images), not VAE latents")

    # Max sequence length for text collator
    sub_parser.add_argument("--max_seq_len", type=int, default=2048,
                            help="Maximum token sequence length for text")

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
