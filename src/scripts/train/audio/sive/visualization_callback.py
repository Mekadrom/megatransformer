import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from model.audio.sive.ctc_vocab import CTCVocab
from model.audio.vocoder.vocoder import Vocoder
from scripts.train.visualization_callback import VisualizationCallback
from transformers.trainer import Trainer

from utils.train_utils import get_writer


def levenshtein_distance(s1, s2) -> int:
    """Compute Levenshtein distance between two sequences."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def compute_wer_cer(pred: str, target: str) -> tuple:
    """Compute Word Error Rate and Character Error Rate."""
    # CER
    cer = levenshtein_distance(pred, target) / max(len(target), 1)

    # WER
    pred_words = pred.split()
    target_words = target.split()
    wer = levenshtein_distance(pred_words, target_words) / max(len(target_words), 1)

    return wer, cer


class SIVEVisualizationCallback(VisualizationCallback):
    """
    Callback for logging CTC SIVE visualizations to TensorBoard.

    Logs:
    - PCA of features colored by speaker
    - Feature heatmaps with CTC alignment
    - Sample transcriptions with CER/WER
    - Feature health metrics (global_mean, std, temporal_smoothness)
    - Audio samples via vocoder
    """

    def __init__(
        self,
        vocoder: Vocoder,
        vocab: CTCVocab,
        num_tsne_samples: int = 150,
        num_transcription_samples: int = 8,
        max_speakers_for_tsne: int = 15,
        # Audio settings
        audio_sample_rate: int = 16000,
        audio_n_fft: int = 1024,
        audio_hop_length: int = 256,
        num_audio_samples: int = 4,
        # LM decoder settings (beam search with optional language model)
        kenlm_model_path: Optional[str] = None,
        lm_alpha: float = 0.5,
        lm_beta: float = 1.0,
        beam_width: int = 100,
    ):
        # Shared window buffer for vocoder
        self.vocoder = vocoder
        self.vocab = vocab
        self.num_tsne_samples = num_tsne_samples
        self.num_transcription_samples = num_transcription_samples
        self.max_speakers_for_tsne = max_speakers_for_tsne
        self.trainer: Optional[Trainer] = None

        # Audio settings
        self.audio_sample_rate = audio_sample_rate
        self.audio_n_fft = audio_n_fft
        self.audio_hop_length = audio_hop_length
        self.num_audio_samples = num_audio_samples

        # LM decoder settings
        self.kenlm_model_path = kenlm_model_path
        self.lm_alpha = lm_alpha
        self.lm_beta = lm_beta
        self.beam_width = beam_width
        self.ctc_decoder = None
        self._decoder_build_attempted = False

        self.trainer = None

    def _build_ctc_decoder(self):
        """Lazily build CTC decoder with optional LM on first use."""
        if self._decoder_build_attempted:
            return
        self._decoder_build_attempted = True

        # Build decoder (with or without LM)
        try:
            self.ctc_decoder = self.vocab.build_ctc_decoder(
                kenlm_model_path=self.kenlm_model_path,
                alpha=self.lm_alpha,
                beta=self.lm_beta,
            )
            if self.ctc_decoder is not None:
                if self.kenlm_model_path:
                    print(f"Built CTC decoder with LM from {self.kenlm_model_path}")
                    print(f"  alpha={self.lm_alpha}, beta={self.lm_beta}, beam_width={self.beam_width}")
                else:
                    print(f"Built CTC beam decoder (no LM), beam_width={self.beam_width}")
        except Exception as e:
            print(f"Failed to build CTC decoder: {e}")
            self.ctc_decoder = None

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Run visualizations in lockstep with eval (if enabled)."""
        if not state.is_world_process_zero:
            return

        # Model may be passed directly or in kwargs
        actual_model = model if model is not None else kwargs.get("model")
        # Fall back to trainer's model if available
        if actual_model is None and self.trainer is not None:
            actual_model = self.trainer.model
        self._run_visualizations(actual_model, state, kwargs)

    def _run_visualizations(self, model, state, kwargs):
        """Run all visualizations and log to TensorBoard."""
        if model is None:
            print("  [Visualization] Skipping: model is None")
            return

        writer = get_writer(kwargs.get("trainer", self.trainer))
        if writer is None:
            print("  [Visualization] Skipping: TensorBoard writer is None")
            return

        # Lazily load CTC decoder
        self._build_ctc_decoder()

        model.eval()
        device = next(model.parameters()).device

        try:
            self._log_tsne(model, writer, state.global_step, device)
            self._log_transcriptions_with_alignment(model, writer, state.global_step, device)
            self._log_feature_health(model, writer, state.global_step, device)
            self._log_audio_samples(model, writer, state.global_step, device)
        except Exception as e:
            print(f"Visualization error at step {state.global_step}: {e}")
            import traceback
            traceback.print_exc()

        model.train()

    @torch.no_grad()
    def _log_tsne(self, model, writer, step, device):
        """Log PCA and t-SNE visualization of features colored by speaker."""
        features_list = []
        speakers_list = []
        speaker_counts = {}

        indices = list(range(len(self.trainer.eval_dataset)))
        np.random.shuffle(indices)

        for idx in indices:
            if len(features_list) >= self.num_tsne_samples:
                break

            sample = self.trainer.eval_dataset[idx]
            speaker_id = sample["speaker_id"].item()

            if speaker_counts.get(speaker_id, 0) >= self.num_tsne_samples // self.max_speakers_for_tsne:
                continue

            if len(speaker_counts) >= self.max_speakers_for_tsne and speaker_id not in speaker_counts:
                continue

            mel_spec = sample["mel_spec"].unsqueeze(0).to(device)
            mel_length = sample["mel_length"].unsqueeze(0)

            result = model(mel_spec, lengths=mel_length, grl_alpha=0.0)
            feat = result["features"]

            feat_pooled = feat.mean(dim=1).cpu().numpy()

            features_list.append(feat_pooled[0])
            speakers_list.append(speaker_id)
            speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1

        if len(features_list) < 10:
            return

        features = np.array(features_list)
        speakers = np.array(speakers_list)

        # Map speakers to color indices
        unique_speakers = np.unique(speakers)
        speaker_to_idx = {s: i for i, s in enumerate(unique_speakers)}
        color_indices = np.array([speaker_to_idx[s] for s in speakers])

        # PCA
        t_pca = time.time()
        pca = PCA(n_components=2, random_state=42)
        pca_2d = pca.fit_transform(features)
        print(f"      PCA compute: {time.time() - t_pca:.1f}s")

        # t-SNE
        t_tsne = time.time()
        tsne = TSNE(n_components=2, perplexity=min(30, len(features) - 1), random_state=42)
        tsne_2d = tsne.fit_transform(features)
        print(f"      t-SNE compute: {time.time() - t_tsne:.1f}s")

        # Create side-by-side figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].scatter(pca_2d[:, 0], pca_2d[:, 1], c=color_indices, cmap='tab20', alpha=0.7, s=20)
        axes[0].set_title(f"PCA (Step {step})")
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")

        axes[1].scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=color_indices, cmap='tab20', alpha=0.7, s=20)
        axes[1].set_title(f"t-SNE (Step {step})")
        axes[1].set_xlabel("t-SNE 1")
        axes[1].set_ylabel("t-SNE 2")

        plt.tight_layout()
        writer.add_figure("visualizations/pca_tsne_by_speaker", fig, step)
        plt.close(fig)

    @torch.no_grad()
    def _log_transcriptions_with_alignment(self, model, writer, step, device):
        """Log sample transcriptions with CER/WER and CTC alignment visualization."""
        transcriptions = []
        total_cer = 0
        total_wer = 0

        indices = np.random.choice(
            len(self.trainer.eval_dataset),
            min(self.num_transcription_samples, len(self.trainer.eval_dataset)),
            replace=False,
        )

        for i, idx in enumerate(indices):
            sample = self.trainer.eval_dataset[idx]

            mel_spec = sample["mel_spec"].unsqueeze(0).to(device)
            mel_length = sample["mel_length"].unsqueeze(0)
            actual_mel_len = sample["mel_length"].item()
            ctc_tokens = sample["ctc_tokens"]
            ctc_length = sample["ctc_length"].item()

            result = model(mel_spec, lengths=mel_length, grl_alpha=0.0)
            asr_logits = result["asr_logits"]  # [1, T, vocab]
            features = result["features"]  # [1, T, D]
            feature_length = result["feature_lengths"][0].item() if result["feature_lengths"] is not None else asr_logits.size(1)

            # Get probabilities
            asr_probs = F.softmax(asr_logits[0, :feature_length], dim=-1).cpu().numpy()  # [T, vocab]

            # Get ground truth
            target_text = self.vocab.decode(
                ctc_tokens[:ctc_length].tolist(),
                remove_blanks=True,
                collapse_repeats=False,
            )

            # Always compute greedy decode for base metrics (maintains consistency with previous runs)
            pred_text_greedy = self.vocab.ctc_decode_greedy(asr_logits)[0]
            wer_greedy, cer_greedy = compute_wer_cer(pred_text_greedy, target_text)
            length_ratio_greedy = len(pred_text_greedy) / max(len(target_text), 1)
            total_wer += wer_greedy
            total_cer += cer_greedy

            # Optionally compute beam+LM decode for separate metrics
            if self.ctc_decoder is not None:
                try:
                    pred_text_lm = self.vocab.ctc_decode_beam(
                        asr_logits[0, :feature_length],
                        decoder=self.ctc_decoder,
                        beam_width=self.beam_width,
                    )[0]
                    wer_lm, cer_lm = compute_wer_cer(pred_text_lm, target_text)
                    length_ratio_lm = len(pred_text_lm) / max(len(target_text), 1)
                except Exception as e:
                    print(f"Warning: Beam+LM decode failed: {e}")
                    pred_text_lm = None
                    wer_lm, cer_lm, length_ratio_lm = None, None, None
            else:
                pred_text_lm = None
                wer_lm, cer_lm, length_ratio_lm = None, None, None

            transcriptions.append({
                "target": target_text,
                "pred_greedy": pred_text_greedy,
                "pred_lm": pred_text_lm,
                "cer": cer_greedy,
                "wer": wer_greedy,
                "length_ratio": length_ratio_greedy,
                "cer_lm": cer_lm,
                "wer_lm": wer_lm,
                "length_ratio_lm": length_ratio_lm,
            })

            # Create alignment visualization (mel + CTC probs + blank prob + text comparison)
            if i < 4:  # Only visualize first 4
                fig, axes = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 2, 1.5, 0.8]})

                # 1. Mel spectrogram
                mel_np = sample["mel_spec"][:, :actual_mel_len].numpy()
                im0 = axes[0].imshow(mel_np, aspect="auto", origin="lower", cmap="viridis")
                axes[0].set_title(f"Mel Spectrogram (T={actual_mel_len})")
                axes[0].set_ylabel("Mel Bin")
                plt.colorbar(im0, ax=axes[0])

                # 2. Feature heatmap
                feat_np = features[0, :feature_length].cpu().numpy().T  # [D, T]
                im1 = axes[1].imshow(feat_np, aspect="auto", origin="lower", cmap="viridis")
                axes[1].set_title(f"SIVE Features (T'={feature_length}, D={feat_np.shape[0]})")
                axes[1].set_ylabel("Feature Dim")
                plt.colorbar(im1, ax=axes[1])

                # 3. CTC probability heatmap (top-k characters)
                # Get top-k most likely chars at each timestep
                top_k = 10
                top_indices = np.argsort(asr_probs, axis=-1)[:, -top_k:][:, ::-1]  # [T, top_k]
                top_probs = np.take_along_axis(asr_probs, top_indices, axis=-1)  # [T, top_k]

                im2 = axes[2].imshow(top_probs.T, aspect="auto", origin="lower", cmap="hot", vmin=0, vmax=1)
                axes[2].set_title("CTC Probabilities (top-10 chars per frame)")
                axes[2].set_ylabel("Char Rank")
                axes[2].set_yticks(range(top_k))
                plt.colorbar(im2, ax=axes[2])

                # 4. Blank probability over time
                blank_probs = asr_probs[:, self.vocab.blank_idx]  # [T]
                axes[3].plot(blank_probs, color='blue', linewidth=1)
                axes[3].fill_between(range(len(blank_probs)), blank_probs, alpha=0.3)
                axes[3].set_xlim(0, feature_length)
                axes[3].set_ylim(0, 1)
                axes[3].set_ylabel("P(blank)")
                axes[3].set_xlabel("Frame")
                axes[3].set_title(f"Blank Probability (mean={np.mean(blank_probs):.3f})")
                axes[3].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

                # Add CER/WER, length ratio in upper right (greedy metrics, with LM if available)
                # Length ratio > 1 means prediction is too long, < 1 means too short
                metrics_text = f"Greedy: CER={cer_greedy:.3f} WER={wer_greedy:.3f} Len={length_ratio_greedy:.2f}x"
                if cer_lm is not None:
                    metrics_text += f"\nBeam+LM: CER={cer_lm:.3f} WER={wer_lm:.3f} Len={length_ratio_lm:.2f}x"
                fig.text(0.98, 0.98, metrics_text, fontsize=9,
                        ha='right', verticalalignment='top', fontweight='bold', family='monospace')

                # Add text comparison below figure with manual line wrapping
                def wrap_text(text, max_chars=120):
                    """Wrap text to multiple lines."""
                    lines = []
                    while len(text) > max_chars:
                        # Find last space before max_chars
                        wrap_at = text.rfind(' ', 0, max_chars)
                        if wrap_at == -1:
                            wrap_at = max_chars
                        lines.append(text[:wrap_at])
                        text = text[wrap_at:].lstrip()
                    lines.append(text)
                    return '\n'.join(lines)

                target_wrapped = wrap_text(f"Target:  {target_text}")
                greedy_wrapped = wrap_text(f"Greedy:  {pred_text_greedy}")

                # Count lines needed for proper spacing
                target_lines = target_wrapped.count('\n') + 1
                greedy_lines = greedy_wrapped.count('\n') + 1
                total_lines = target_lines + greedy_lines + 1

                # Add LM prediction if available
                if pred_text_lm is not None:
                    lm_wrapped = wrap_text(f"Beam+LM: {pred_text_lm}")
                    lm_lines = lm_wrapped.count('\n') + 1
                    total_lines += lm_lines + 1
                else:
                    lm_wrapped = None
                    lm_lines = 0

                # Calculate bottom margin based on number of lines
                line_height = 0.018
                bottom_margin = max(0.12, total_lines * line_height + 0.02)

                y_pos = bottom_margin - 0.01
                fig.text(0.02, y_pos, target_wrapped, fontsize=8, family='monospace', verticalalignment='top')
                y_pos -= (target_lines * line_height) + 0.01
                fig.text(0.02, y_pos, greedy_wrapped, fontsize=8, family='monospace', verticalalignment='top')
                if lm_wrapped is not None:
                    y_pos -= (greedy_lines * line_height) + 0.01
                    fig.text(0.02, y_pos, lm_wrapped, fontsize=8, family='monospace', verticalalignment='top', color='blue')

                # rect=[left, bottom, right, top] - leave room at top for metrics, bottom for text
                top_margin = 0.94 if cer_lm is not None else 0.96  # More room if LM metrics shown
                plt.tight_layout(rect=[0, bottom_margin, 1, top_margin])
                writer.add_figure(f"ctc_alignment/sample_{i}", fig, step)
                plt.close(fig)

        # Log greedy metrics (consistent with previous runs)
        avg_cer = total_cer / len(transcriptions) if transcriptions else 0
        avg_wer = total_wer / len(transcriptions) if transcriptions else 0
        avg_length_ratio = sum(t['length_ratio'] for t in transcriptions) / len(transcriptions) if transcriptions else 1.0
        writer.add_scalar("eval/cer", avg_cer, step)
        writer.add_scalar("eval/wer", avg_wer, step)
        writer.add_scalar("eval/length_ratio", avg_length_ratio, step)

        # Log LM metrics if available
        if self.ctc_decoder is not None:
            lm_cers = [t['cer_lm'] for t in transcriptions if t['cer_lm'] is not None]
            lm_wers = [t['wer_lm'] for t in transcriptions if t['wer_lm'] is not None]
            lm_ratios = [t['length_ratio_lm'] for t in transcriptions if t['length_ratio_lm'] is not None]
            if lm_cers:
                avg_cer_lm = sum(lm_cers) / len(lm_cers)
                avg_wer_lm = sum(lm_wers) / len(lm_wers)
                avg_length_ratio_lm = sum(lm_ratios) / len(lm_ratios)
                writer.add_scalar("eval/cer_lm", avg_cer_lm, step)
                writer.add_scalar("eval/wer_lm", avg_wer_lm, step)
                writer.add_scalar("eval/length_ratio_lm", avg_length_ratio_lm, step)

        # Log sample transcriptions as text
        text_summary = f"**Step {step} Sample Transcriptions**\n\n"
        text_summary += f"**Greedy:** CER={avg_cer:.3f} | WER={avg_wer:.3f} | Len={avg_length_ratio:.2f}x\n"
        if self.ctc_decoder is not None and lm_cers:
            text_summary += f"**Beam+LM:** CER={avg_cer_lm:.3f} | WER={avg_wer_lm:.3f} | Len={avg_length_ratio_lm:.2f}x\n"
        text_summary += "\n"
        for i, t in enumerate(transcriptions[:4]):
            text_summary += f"**Sample {i + 1}**\n"
            text_summary += f"  Target:  `{t['target'][:100]}`\n"
            text_summary += f"  Greedy:  `{t['pred_greedy'][:100]}` (CER={t['cer']:.3f})\n"
            if t['pred_lm'] is not None:
                text_summary += f"  Beam+LM: `{t['pred_lm'][:100]}` (CER={t['cer_lm']:.3f})\n"
            text_summary += "\n"

        writer.add_text("transcriptions/samples", text_summary, step)

    @torch.no_grad()
    def _log_feature_health(self, model, writer, step, device, num_samples: int = 50):
        """Log feature health metrics and visualization (global stats, temporal smoothness, etc.)."""
        all_features = []
        temporal_sims_norm = []
        temporal_sims_unnorm = []

        indices = list(range(min(len(self.trainer.eval_dataset), num_samples)))

        for idx in indices:
            sample = self.trainer.eval_dataset[idx]
            mel_spec = sample["mel_spec"].unsqueeze(0).to(device)
            mel_length = sample["mel_length"].unsqueeze(0)

            result = model(mel_spec, lengths=mel_length, grl_alpha=0.0)
            features_norm = result["features"][0].cpu()  # [T, D]
            features_unnorm = result["features_unnorm"][0].cpu()  # [T, D]

            all_features.append(features_norm)

            # Temporal smoothness
            if features_norm.shape[0] > 1:
                feat_n = F.normalize(features_unnorm, dim=-1)
                sim = (feat_n[:-1] * feat_n[1:]).sum(dim=-1)
                temporal_sims_unnorm.append(sim.mean().item())

                feat_n = F.normalize(features_norm, dim=-1)
                sim = (feat_n[:-1] * feat_n[1:]).sum(dim=-1)
                temporal_sims_norm.append(sim.mean().item())

        all_features = torch.cat(all_features, dim=0).numpy()
        feat_dim = all_features.shape[1]

        # Global statistics
        global_mean = np.mean(all_features)
        global_std = np.std(all_features)
        writer.add_scalar("feature_health/global_mean", global_mean, step)
        writer.add_scalar("feature_health/global_std", global_std, step)

        # Feature norms
        feature_norms = np.linalg.norm(all_features, axis=-1)
        writer.add_scalar("feature_health/mean_norm", np.mean(feature_norms), step)
        writer.add_scalar("feature_health/std_norm", np.std(feature_norms), step)

        # Per-dimension statistics
        dim_means = np.mean(all_features, axis=0)  # [D]
        dim_stds = np.std(all_features, axis=0)    # [D]

        # Dead dimension detection (std < 0.01)
        dead_dims = np.sum(dim_stds < 0.01)
        writer.add_scalar("feature_health/dead_dimensions", dead_dims, step)
        writer.add_scalar("feature_health/dead_dim_ratio", dead_dims / feat_dim, step)

        # Effective dimensionality (via explained variance ratio using entropy)
        var_per_dim = dim_stds ** 2
        var_normalized = var_per_dim / (var_per_dim.sum() + 1e-8)
        entropy = -np.sum(var_normalized * np.log(var_normalized + 1e-8))
        effective_dim = np.exp(entropy)
        writer.add_scalar("feature_health/effective_dimensionality", effective_dim, step)
        writer.add_scalar("feature_health/dim_utilization_ratio", effective_dim / feat_dim, step)

        # Temporal smoothness (log both pre-norm and post-norm for comparison)
        if temporal_sims_unnorm:
            mean_smoothness_unnorm = np.mean(temporal_sims_unnorm)
            writer.add_scalar("feature_health/temporal_smoothness_unnorm", mean_smoothness_unnorm, step)
            writer.add_scalar("feature_health/temporal_smoothness", mean_smoothness_unnorm, step)
        if temporal_sims_norm:
            mean_smoothness_norm = np.mean(temporal_sims_norm)
            writer.add_scalar("feature_health/temporal_smoothness_norm", mean_smoothness_norm, step)

        # Debug print to console
        if temporal_sims_unnorm and temporal_sims_norm:
            print(f"      [Smoothness] pre-norm={mean_smoothness_unnorm:.4f}, post-norm={mean_smoothness_norm:.4f}")

        # === Visualizations ===
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Per-dimension mean distribution
        axes[0, 0].hist(dim_means, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(0, color='red', linestyle='--', label='zero')
        axes[0, 0].set_title(f"Per-Dimension Means (global μ={global_mean:.3f})")
        axes[0, 0].set_xlabel("Mean")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].legend()

        # 2. Per-dimension std distribution (dimension utilization)
        axes[0, 1].hist(dim_stds, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(0.01, color='red', linestyle='--', label='dead threshold')
        axes[0, 1].set_title(f"Per-Dimension Stds ({dead_dims}/{feat_dim} dead)")
        axes[0, 1].set_xlabel("Std")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].legend()

        # 3. Feature norms distribution
        axes[0, 2].hist(feature_norms, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 2].set_title(f"Feature Norms (μ={np.mean(feature_norms):.2f}, σ={np.std(feature_norms):.2f})")
        axes[0, 2].set_xlabel("L2 Norm")
        axes[0, 2].set_ylabel("Count")

        # 4. Overall activation histogram
        flat_features = all_features.flatten()
        # Clip for visualization (avoid extreme outliers dominating)
        clip_val = np.percentile(np.abs(flat_features), 99)
        clipped = np.clip(flat_features, -clip_val, clip_val)
        axes[1, 0].hist(clipped, bins=100, edgecolor='black', alpha=0.7)
        axes[1, 0].set_title(f"Activation Distribution (clipped to 99th %ile)")
        axes[1, 0].set_xlabel("Activation Value")
        axes[1, 0].set_ylabel("Count")

        # 5. Dimension correlation matrix (subsample dimensions for visibility)
        max_dims_to_show = 64
        if feat_dim > max_dims_to_show:
            dim_indices = np.linspace(0, feat_dim - 1, max_dims_to_show, dtype=int)
            features_subset = all_features[:, dim_indices]
        else:
            features_subset = all_features

        corr_matrix = np.corrcoef(features_subset.T)
        im = axes[1, 1].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        axes[1, 1].set_title(f"Dimension Correlations ({min(feat_dim, max_dims_to_show)} dims)")
        axes[1, 1].set_xlabel("Dimension")
        axes[1, 1].set_ylabel("Dimension")
        plt.colorbar(im, ax=axes[1, 1])

        # 6. Temporal smoothness histogram (compare pre-norm vs post-norm)
        if temporal_sims_unnorm and temporal_sims_norm:
            axes[1, 2].hist(temporal_sims_unnorm, bins=30, edgecolor='black', alpha=0.5,
                           label=f'pre-norm (μ={np.mean(temporal_sims_unnorm):.3f})', color='blue')
            axes[1, 2].hist(temporal_sims_norm, bins=30, edgecolor='black', alpha=0.5,
                           label=f'post-norm (μ={np.mean(temporal_sims_norm):.3f})', color='orange')
            axes[1, 2].axvline(np.mean(temporal_sims_unnorm), color='blue', linestyle='--')
            axes[1, 2].axvline(np.mean(temporal_sims_norm), color='orange', linestyle='--')
            axes[1, 2].set_title("Temporal Smoothness (adj. frame cos sim)")
            axes[1, 2].set_xlabel("Cosine Similarity")
            axes[1, 2].set_ylabel("Count")
            axes[1, 2].legend()
        elif temporal_sims_unnorm:
            axes[1, 2].hist(temporal_sims_unnorm, bins=30, edgecolor='black', alpha=0.7)
            axes[1, 2].axvline(np.mean(temporal_sims_unnorm), color='red', linestyle='--',
                              label=f'mean={np.mean(temporal_sims_unnorm):.3f}')
            axes[1, 2].set_title("Temporal Smoothness (pre-norm)")
            axes[1, 2].set_xlabel("Cosine Similarity")
            axes[1, 2].set_ylabel("Count")
            axes[1, 2].legend()
        else:
            axes[1, 2].text(0.5, 0.5, "No temporal data", ha='center', va='center')

        plt.suptitle(f"Feature Space Health (Step {step})", fontsize=14)
        plt.tight_layout()
        writer.add_figure("visualizations/feature_space_health", fig, step)
        plt.close(fig)

        # Log summary to console
        print(f"      Feature health: mean={global_mean:.3f}, std={global_std:.3f}, "
              f"norm={np.mean(feature_norms):.3f}, dead={dead_dims}/{feat_dim}, "
              f"effective_dim={effective_dim:.1f}")

    @torch.no_grad()
    def _log_audio_samples(self, model, writer, step, device):
        """Log audio samples with vocoder output aligned with transcription."""
        if self.vocoder is None:
            return

        num_samples = min(self.num_audio_samples, len(self.trainer.eval_dataset))

        for i in range(num_samples):
            sample = self.trainer.eval_dataset[i]

            mel_spec = sample["mel_spec"]  # [n_mels, T]
            mel_length = sample["mel_length"].item()
            ctc_tokens = sample["ctc_tokens"]
            ctc_length = sample["ctc_length"].item()

            # Crop to actual length
            mel_cropped = mel_spec[:, :mel_length]

            # Get ground truth text
            target_text = self.vocab.decode(
                ctc_tokens[:ctc_length].tolist(),
                remove_blanks=True,
                collapse_repeats=False,
            )

            # Generate audio from mel spectrogram
            self._log_vocoder_audio(
                writer, mel_cropped, step,
                tag=f"audio_samples/sample_{i}"
            )

            # Log the transcription for this sample
            writer.add_text(
                f"audio_samples/sample_{i}_text",
                f"**Sample {i}**: {target_text}",
                step
            )
