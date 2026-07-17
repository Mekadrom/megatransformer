"""Waveform-level augmentations for SIVE training.

Unlike the mel-space augmentations in ``mel_augment.py``, these operate on the
raw waveform and so can only run when the dataset surfaces ``waveforms`` (not
when it only carries pre-extracted mel spectrograms). The motivating case is
pitch / F0 shifting, which is a fundamentally waveform-domain operation and
cannot be reproduced by perturbing a mel spectrogram.

Augmentations are applied per-sample with a configurable probability and a
fresh random draw each call, so the same example is perturbed differently on
each epoch. Pitch shift preserves length; speed perturbation changes it, so
``forward`` returns both the (re-padded) waveform batch and the new per-sample
lengths — the caller must recompute mel lengths / masks from those.

Performance: per-sample n_steps / factor are quantized to a small grid so
samples sharing a quantized value can share a single batched call to
``AF.pitch_shift`` / ``AF.speed``. With Kaldi-style defaults
(``pitch_quantize_step=1`` semitone, ``speed_quantize_step=0.05``) and the
default strengths, this caps the worst-case call count at ~9 pitch groups +
~5 speed groups regardless of batch size, vs B-per-step in a naive loop.
Setting either step to 0 disables quantization for that axis and falls back
to per-sample calls. All randomness and length reads are hoisted to CPU
before the per-group loop, so the GPU isn't pinged for syncs each iteration.

This module is a no-op outside training mode.
"""

from fractions import Fraction

import torch
import torch.nn as nn
import torchaudio.functional as AF
import torchaudio.transforms as AT

# AF.pitch_shift = phase-vocoder stretch + resample(int(sr/rate), sr). The
# resample dominates: sr/rate is coprime with sr, so its polyphase sinc kernel
# is ~sr rows (a ~1GB alloc, rebuilt every call — profiled at ~90% of a
# consistency-run step). We reuse torchaudio's stretch helper but cache a
# small-rational Resample per n_steps: approximating the ratio to a bounded
# denominator shrinks the kernel ~1000x (6GB->~50MB) for a >40dB-SNR,
# pitch-exact result. If the private helpers ever move, fall back to functional.
try:
    from torchaudio.functional.functional import _stretch_waveform, _fix_waveform_shape
    _HAS_STRETCH_HELPERS = True
except Exception:  # pragma: no cover - torchaudio internal moved
    _HAS_STRETCH_HELPERS = False

_PITCH_RATIO_MAX_DEN = 2048  # rational-approx denominator cap (SNR vs kernel size)


class WaveformAugment(nn.Module):
    """Per-sample pitch shift + speed perturbation on a padded waveform batch.

    Args:
        sample_rate: Waveform sample rate (Hz).
        pitch_shift_semitones: Half-width of the uniform pitch-shift range in
            semitones; a per-sample shift is drawn from
            ``[-pitch_shift_semitones, +pitch_shift_semitones]``. 0 disables.
        pitch_shift_prob: Per-sample probability of applying pitch shift.
        speed_perturb: Half-width of the uniform speed range; a per-sample
            factor is drawn from ``[1 - speed_perturb, 1 + speed_perturb]``
            (factor > 1 = faster/shorter). 0 disables.
        speed_perturb_prob: Per-sample probability of applying speed perturb.
        pitch_quantize_step: Quantization step in semitones for n_steps;
            samples sharing a quantized value are dispatched in one batched
            ``AF.pitch_shift`` call. 0 disables quantization (per-sample
            calls). Default 1.0 (Kaldi-style integer semitones).
        speed_quantize_step: Quantization step for the speed factor (same
            grouping logic as pitch). 0 disables. Default 0.05.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        pitch_shift_semitones: float = 0.0,
        pitch_shift_prob: float = 0.0,
        speed_perturb: float = 0.0,
        speed_perturb_prob: float = 0.0,
        pitch_quantize_step: float = 1.0,
        speed_quantize_step: float = 0.05,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.pitch_shift_semitones = pitch_shift_semitones
        self.pitch_shift_prob = pitch_shift_prob
        self.speed_perturb = speed_perturb
        self.speed_perturb_prob = speed_perturb_prob
        self.pitch_quantize_step = pitch_quantize_step
        self.speed_quantize_step = speed_quantize_step
        # Caches for the fast cached-resample pitch shift, keyed by
        # (n_steps, device, dtype) / (device, dtype). Plain dicts, NOT
        # submodules: the resample kernels are deterministic and must stay out of
        # the model state_dict (they'd otherwise add non-checkpoint buffers).
        self._resample_bank: dict = {}
        self._hann_cache: dict = {}

    @property
    def enabled(self) -> bool:
        return (
            (self.pitch_shift_semitones > 0 and self.pitch_shift_prob > 0)
            or (self.speed_perturb > 0 and self.speed_perturb_prob > 0)
        )

    # ---- randomness helpers (CPU; no device syncs) -------------------------

    def _draw_pitch_steps(self, B: int) -> list:
        """Per-sample quantized n_steps; ``None`` for skipped/no-op samples."""
        if self.pitch_shift_semitones <= 0 or self.pitch_shift_prob <= 0:
            return [None] * B
        # All randomness on CPU in one shot — no per-sample sync.
        apply = torch.rand(B).lt(self.pitch_shift_prob).tolist()
        raw = (torch.rand(B) * 2 - 1).mul_(self.pitch_shift_semitones).tolist()
        step = self.pitch_quantize_step
        out = []
        for i in range(B):
            if not apply[i]:
                out.append(None)
                continue
            v = raw[i] if step <= 0 else round(raw[i] / step) * step
            out.append(None if v == 0.0 else float(v))
        return out

    def _draw_speed_factors(self, B: int) -> list:
        """Per-sample quantized speed factor; ``None`` for skipped/no-op samples."""
        if self.speed_perturb <= 0 or self.speed_perturb_prob <= 0:
            return [None] * B
        apply = torch.rand(B).lt(self.speed_perturb_prob).tolist()
        raw = (torch.rand(B) * 2 - 1).mul_(self.speed_perturb).add_(1.0).tolist()
        step = self.speed_quantize_step
        out = []
        for i in range(B):
            if not apply[i]:
                out.append(None)
                continue
            v = raw[i] if step <= 0 else round((raw[i] - 1.0) / step) * step + 1.0
            out.append(None if v == 1.0 else float(v))
        return out

    # ---- batched, grouped passes -------------------------------------------

    def _get_resampler(self, ns: float, device, dtype) -> "AT.Resample":
        """Cached small-rational Resample that restores pitch after the stretch.

        AF.pitch_shift resamples by int(sr/rate)->sr (coprime => huge kernel).
        We approximate that ratio with a bounded-denominator rational so the
        kernel is tiny and reusable; the shift lands on the same pitch (>40dB
        SNR vs functional at the default cap).
        """
        key = (round(float(ns), 6), device, dtype)
        r = self._resample_bank.get(key)
        if r is None:
            rate = 2.0 ** (-float(ns) / 12.0)
            orig_freq = int(self.sample_rate / rate)
            frac = Fraction(self.sample_rate, orig_freq).limit_denominator(_PITCH_RATIO_MAX_DEN)
            # resample ratio = new/orig = frac.numerator/frac.denominator
            r = AT.Resample(frac.denominator, frac.numerator).to(device=device, dtype=dtype)
            self._resample_bank[key] = r
        return r

    def _hann(self, device, dtype) -> torch.Tensor:
        key = (device, dtype)
        w = self._hann_cache.get(key)
        if w is None:
            w = torch.hann_window(512, device=device, dtype=dtype)
            self._hann_cache[key] = w
        return w

    def _pitch_shift_cached(self, batch: torch.Tensor, ns: float) -> torch.Tensor:
        """Fast cached-resample equivalent of AF.pitch_shift (length preserved)."""
        stretched = _stretch_waveform(
            batch, ns, 12, 512, 512, 128, self._hann(batch.device, batch.dtype),
        )
        shifted = self._get_resampler(ns, batch.device, batch.dtype)(stretched)
        return _fix_waveform_shape(shifted, batch.size())

    def _apply_grouped_pitch(self, work: torch.Tensor, steps: list) -> torch.Tensor:
        """Pitch shift, grouped by quantized n_steps. Length preserved."""
        groups: dict = {}
        for i, ns in enumerate(steps):
            if ns is None:
                continue
            groups.setdefault(ns, []).append(i)
        if not groups:
            return work
        # Padded zeros pass through STFT -> phase vocoder -> ISTFT -> resample
        # as zeros, so we can shift the full padded [G, T] batch and the valid
        # region of each sample comes back correctly. Length is preserved by
        # pitch_shift, so we can write back in place.
        # Quantized draws (the default) reuse a cached small-rational resampler
        # so the sinc kernel isn't rebuilt each call. Unquantized (step<=0) draws
        # are unique floats with no reuse, so fall back to functional pitch_shift;
        # ditto if torchaudio's stretch helpers ever move.
        cache_ok = (
            _HAS_STRETCH_HELPERS
            and self.pitch_quantize_step is not None
            and self.pitch_quantize_step > 0
        )
        out = work.clone()
        for ns, idxs in groups.items():
            idx_t = torch.tensor(idxs, device=work.device, dtype=torch.long)
            batch = work.index_select(0, idx_t)  # [G, T]
            if cache_ok:
                shifted = self._pitch_shift_cached(batch, ns)
            else:
                shifted = AF.pitch_shift(batch, self.sample_rate, ns)
            out.index_copy_(0, idx_t, shifted)
        return out

    def _apply_grouped_speed(
        self,
        work: torch.Tensor,
        lengths_cpu: list,
        factors: list,
    ) -> tuple[torch.Tensor, list]:
        """Speed perturb, grouped by quantized factor. Returns (out, new_lengths)."""
        B, T = work.shape
        device = work.device

        groups: dict = {None: []}
        for i, f in enumerate(factors):
            groups.setdefault(f, []).append(i)
        # Fast path: nothing to do.
        if all(f is None for f in groups):
            return work, lengths_cpu

        per_sample_out = [None] * B           # tensor refs (views into batched outputs)
        per_sample_len = [0] * B
        for f, idxs in groups.items():
            if not idxs:
                continue
            if f is None:
                for i in idxs:
                    per_sample_out[i] = work[i]
                    per_sample_len[i] = lengths_cpu[i]
                continue
            idx_t = torch.tensor(idxs, device=device, dtype=torch.long)
            group_in = work.index_select(0, idx_t)  # [G, T]
            group_lens = torch.tensor(
                [lengths_cpu[i] for i in idxs], device=device, dtype=torch.long,
            )
            group_out, group_new_lens = AF.speed(
                group_in, self.sample_rate, f, lengths=group_lens,
            )
            group_new_lens_cpu = group_new_lens.tolist()
            for j, i in enumerate(idxs):
                per_sample_out[i] = group_out[j]
                per_sample_len[i] = int(group_new_lens_cpu[j])

        max_len = max(per_sample_len) if per_sample_len else T
        out = work.new_zeros(B, max_len)
        for i in range(B):
            L = per_sample_len[i]
            if L > 0:
                out[i, :L] = per_sample_out[i][:L]
        return out, per_sample_len

    # ---- entry point --------------------------------------------------------

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            waveforms: [B, T] padded waveform batch.
            lengths: [B] valid (unpadded) length of each waveform.

        Returns:
            (waveforms, lengths): re-padded augmented batch [B, T'] and the new
            per-sample lengths [B]. Returned unchanged in eval mode or when no
            augmentation is enabled.
        """
        if not self.training or not self.enabled:
            return waveforms, lengths

        B = waveforms.size(0)
        device = waveforms.device
        work = waveforms.to(torch.float32)
        lengths_cpu = lengths.tolist()  # single host transfer; small tensor

        # Phase 1: pitch shift, grouped by quantized n_steps. Length preserved.
        pitch_steps = self._draw_pitch_steps(B)
        if any(s is not None for s in pitch_steps):
            work = self._apply_grouped_pitch(work, pitch_steps)

        # Phase 2: speed perturb, grouped by quantized factor. Length changes.
        speed_factors = self._draw_speed_factors(B)
        if any(f is not None for f in speed_factors):
            work, lengths_cpu = self._apply_grouped_speed(work, lengths_cpu, speed_factors)

        new_lengths = torch.tensor(lengths_cpu, device=device, dtype=torch.long)
        return work, new_lengths
