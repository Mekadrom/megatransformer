from typing import Optional
import torch


class CTCVocab:
    """Character-level vocabulary for CTC ASR."""

    # Standard character set for English ASR
    CHARS = " 'abcdefghijklmnopqrstuvwxyz"
    BLANK = "<blank>"
    UNK = "<unk>"

    def __init__(self, chars: str = None):
        chars = chars or self.CHARS
        self.blank_idx = 0
        self.unk_idx = 1
        self.chars = chars

        # Build vocab: [blank, unk, ...chars...]
        self.idx_to_char = [self.BLANK, self.UNK] + list(chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.idx_to_char)}
        self.vocab_size = len(self.idx_to_char)

    def encode(self, text: str) -> list:
        """Convert text to token indices."""
        text = text.lower()
        return [self.char_to_idx.get(c, self.unk_idx) for c in text]

    def decode(self, indices: list, remove_blanks: bool = True, collapse_repeats: bool = True) -> str:
        """
        Convert token indices back to text.

        Args:
            indices: List of token indices
            remove_blanks: Remove blank tokens
            collapse_repeats: Collapse repeated characters (CTC decoding)
        """
        if collapse_repeats:
            # CTC collapse: remove consecutive duplicates
            collapsed = []
            prev = None
            for idx in indices:
                if idx != prev:
                    collapsed.append(idx)
                    prev = idx
            indices = collapsed

        chars = []
        for idx in indices:
            if remove_blanks and idx == self.blank_idx:
                continue
            if idx == self.unk_idx:
                chars.append('?')
            elif idx < len(self.idx_to_char):
                char = self.idx_to_char[idx]
                if char not in [self.BLANK, self.UNK]:
                    chars.append(char)

        return ''.join(chars)

    def ctc_decode_greedy(self, logits: torch.Tensor) -> list:
        """
        Greedy CTC decoding.

        Args:
            logits: [T, vocab_size] or [B, T, vocab_size]

        Returns:
            List of decoded strings
        """
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)

        # Greedy: take argmax at each timestep
        predictions = logits.argmax(dim=-1)  # [B, T]

        decoded = []
        for pred in predictions:
            text = self.decode(pred.tolist())
            decoded.append(text)

        return decoded

    def build_ctc_decoder(self, kenlm_model_path: Optional[str] = None, alpha: float = 0.5, beta: float = 1.0):
        """
        Build a pyctcdecode decoder with optional LM support.

        Args:
            kenlm_model_path: Path to KenLM .arpa or .bin file (optional)
            alpha: LM weight (higher = more LM influence)
            beta: Word insertion bonus (higher = longer words)

        Returns:
            pyctcdecode decoder or None if pyctcdecode not available
        """
        import os

        try:
            from pyctcdecode import build_ctcdecoder
        except ImportError:
            print("pyctcdecode not installed. Install with: pip install pyctcdecode")
            return None

        # Check if LM file exists
        if kenlm_model_path and not os.path.exists(kenlm_model_path):
            print(f"WARNING: KenLM model not found at {kenlm_model_path}")
            print("  CTC decoder will be built WITHOUT language model")
            print("  Download from: https://www.openslr.org/11/ (e.g., 4-gram.arpa.gz)")
            kenlm_model_path = None

        # Build labels list in vocab order (pyctcdecode expects this)
        labels = self.idx_to_char.copy()

        # pyctcdecode expects "" for blank token
        labels[self.blank_idx] = ""

        # Extract unigrams from ARPA file for word-level LM
        unigrams = None
        if kenlm_model_path and kenlm_model_path.endswith('.arpa'):
            print(f"Loading LM from {kenlm_model_path}...")
            unigrams = self._extract_unigrams_from_arpa(kenlm_model_path)

        decoder = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=kenlm_model_path,
            unigrams=unigrams,
            alpha=alpha,
            beta=beta,
        )

        return decoder

    def _extract_unigrams_from_arpa(self, arpa_path: str) -> list:
        """Extract unigrams from ARPA file for pyctcdecode.

        Note: We extract ALL words, not just those matching our char vocab.
        pyctcdecode handles the character-to-word mapping internally.
        """
        unigrams = []
        in_unigrams = False

        try:
            with open(arpa_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()

                    if line == '\\1-grams:':
                        in_unigrams = True
                        continue
                    elif line.startswith('\\') and in_unigrams:
                        break  # Done with unigrams

                    if in_unigrams and line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            word = parts[1]
                            # Skip special tokens
                            if word and word not in ('<s>', '</s>', '<unk>', '<UNK>'):
                                unigrams.append(word.lower())

            print(f"  Extracted {len(unigrams):,} unigrams from ARPA")
        except Exception as e:
            print(f"Warning: Failed to extract unigrams from ARPA: {e}")
            return None

        return unigrams if unigrams else None

    def ctc_decode_beam(
        self,
        logits: torch.Tensor,
        decoder=None,
        beam_width: int = 100,
    ) -> list:
        """
        Beam search CTC decoding with optional LM.

        Args:
            logits: [T, vocab_size] or [B, T, vocab_size]
            decoder: pyctcdecode decoder (from build_ctc_decoder)
            beam_width: Beam width for search

        Returns:
            List of decoded strings
        """
        if decoder is None:
            # Fall back to greedy if no decoder
            return self.ctc_decode_greedy(logits)

        import numpy as np

        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()

        if logits.ndim == 2:
            logits = logits[np.newaxis, ...]

        # Convert to log probabilities (pyctcdecode expects log probs)
        # Softmax then log
        logits_max = logits.max(axis=-1, keepdims=True)
        logits_stable = logits - logits_max
        probs = np.exp(logits_stable) / np.exp(logits_stable).sum(axis=-1, keepdims=True)
        log_probs = np.log(probs + 1e-10)

        decoded = []
        for log_prob in log_probs:
            text = decoder.decode(log_prob, beam_width=beam_width)
            decoded.append(text)

        return decoded
