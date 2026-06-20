"""
build_gender_lookup.py — generate a SPEAKERS.TXT-compatible gender lookup file
by running an off-the-shelf gender classifier over a HuggingFace audio dataset,
voting at the speaker level, and emitting one row per speaker.

The output is consumable by ``preprocess voice --gender_lookup_path <path>``
(it shares the LibriSpeech SPEAKERS.TXT format: pipe-delimited, ``;``-comment
header, ``ID | SEX | …``). Use this for datasets that don't ship a per-row
gender column (LibriSpeech itself, parler-tts/mls_eng, AMI, VoxPopuli, etc.).

Defaults to ``alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech``
because it loads via standard ``AutoModelForAudioClassification`` and was
trained in-domain (LibriVox read speech). Pass ``--model_id`` to swap in any
other HF audio gender classifier whose model config has a clean ``id2label``
mapping (e.g. ``prithivMLmods/Common-Voice-Gender-Detection`` for noisier
domains). Models with custom architectures (audeering's age+gender regression
model) are NOT supported here — they need custom loader code.

Speaker-level voting: by default each speaker is classified from up to 3
utterances; we majority-vote among utterances whose top-class confidence
exceeds ``--confidence_threshold`` (default 0.8). Speakers whose votes all
fall below the threshold get ``?`` (which the preprocess-side normalizer
treats as ``-1``/unknown).

Example:
    python -m megatransformer.scripts.data.voice.build_gender_lookup \\
        --dataset_name parler-tts/mls_eng --split test \\
        --speaker_id_column speaker_id --audio_column audio \\
        --output_path cached_datasets/mls_eng_gender.txt \\
        --utterances_per_speaker 3 --confidence_threshold 0.8
"""

import argparse
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Reuse the same canonical-gender mapping the preprocessor uses so the
# emitted file round-trips cleanly through _parse_speakers_txt later.
from megatransformer.scripts.data.voice.preprocess import _normalize_gender


DEFAULT_MODEL_ID = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"


def _load_classifier(model_id: str, device: str):
    """Load a HF audio gender classifier that follows the standard
    AutoModelForAudioClassification pattern. Returns (feature_extractor, model,
    idx_to_canonical_gender) where idx_to_canonical_gender maps model output
    index to {0=male, 1=female, -1=other/unknown} via the existing normalizer.
    """
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    print(f"Loading classifier '{model_id}' on {device}...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id).to(device).eval()
    id2label = model.config.id2label
    if id2label is None or len(id2label) == 0:
        raise RuntimeError(
            f"Model '{model_id}' has no id2label in its config; can't determine "
            "which output index is male/female. Use a model with a populated "
            "id2label or pick a different one."
        )
    idx_to_canonical: Dict[int, int] = {}
    for idx, label_str in id2label.items():
        idx = int(idx)
        canonical = _normalize_gender(label_str)
        idx_to_canonical[idx] = canonical
        print(f"  index {idx} → label {label_str!r} → canonical {canonical}")
    # Sanity check: should have at least one male and one female index
    canonicals = set(idx_to_canonical.values())
    if 0 not in canonicals or 1 not in canonicals:
        print(
            f"  WARNING: id2label doesn't contain a clear male/female pair "
            f"(canonical set = {canonicals}). Speakers may be marked unknown "
            "frequently. Verify the model is a gender classifier."
        )
    return feature_extractor, model, idx_to_canonical


@torch.no_grad()
def _scan_and_classify(
    dataset_iter,
    feature_extractor,
    model,
    idx_to_canonical: Dict[int, int],
    speaker_id_column: str,
    audio_column: str,
    utterances_per_speaker: int,
    max_speakers: Optional[int],
    max_examples: Optional[int],
    max_hours: Optional[float],
    saturation_grace_period: int,
    max_audio_samples: int,
    target_sample_rate: int,
    batch_size: int,
    device: str,
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Walk the dataset and collect up to ``utterances_per_speaker`` clips per
    speaker. Audio is clipped to ``max_audio_samples`` samples to bound peak
    memory. Stops early on whichever of these fires first:

      - **--max_speakers** cap reached AND all known speakers saturated
        (immediate exit; no new speaker can be added so no point continuing)
      - **all known speakers saturated** AND no new speaker has been added
        for the last ``saturation_grace_period`` iterations (covers the
        common case where the user didn't pin a speaker count up front —
        prevents the loop from running to dataset exhaustion when every
        speaker is already full)
      - **--max_examples**: stop after this many dataset rows have been
        iterated (counts every row, including those we cheaply skip)
      - **--max_hours**: stop after this many cumulative hours of
        *accepted* utterance audio have been collected. NOTE: this is an
        accepted-audio budget (matches the preprocessor's semantics), NOT
        an iteration-time cap. With small ``utterances_per_speaker``, this
        ceiling can be unreachable on huge datasets — use ``--max_examples``
        if you want a hard iteration cap.

    Saturated-speaker and beyond-cap rows are skipped *before* the audio
    field is decoded, so they cost no I/O beyond the dataset's normal
    iteration overhead.

    Classification is inline-streaming: accepted waveforms are buffered into
    batches of size ``batch_size``, the gender classifier runs on each filled
    batch, only the per-utterance ``(canonical_gender, confidence)`` tuple is
    retained, and the waveform tensor is discarded. Peak memory is therefore
    bounded by ``batch_size`` waveforms regardless of how many total hours
    of audio are processed.

    Returns ``{speaker_id: [(canonical, confidence), ...]}`` — one list of
    per-utterance vote tuples per speaker. Aggregation into a final per-
    speaker label happens downstream in ``_aggregate_votes``.
    """
    # utterances_per_speaker <= 0 disables the per-speaker cap entirely —
    # every utterance is accepted, and termination relies on --max_hours
    # and/or --max_examples. This is the "I just want N hours of audio,
    # don't be clever about it" mode.
    unlimited = utterances_per_speaker <= 0

    votes_per_speaker: Dict[str, List[Tuple[int, float]]] = {}
    accepted_counts: Dict[str, int] = {}
    n_seen = 0
    n_saturated = 0  # O(1) tracker so the all-saturated check stays O(1)
    iters_since_new_speaker = 0
    duration_seconds = 0.0
    max_seconds = max_hours * 3600.0 if max_hours is not None else None

    # Streaming inference buffer. Cleared on every flush so peak memory is
    # bounded to ~batch_size waveforms (~few MB at 16kHz, 10s clips).
    batch_arrs: List[np.ndarray] = []
    batch_speakers: List[str] = []

    def flush_batch() -> None:
        if not batch_arrs:
            return
        inputs = feature_extractor(
            batch_arrs,
            sampling_rate=target_sample_rate,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits  # [B, n_classes]
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        for spk, p in zip(batch_speakers, probs):
            top_idx = int(np.argmax(p))
            top_conf = float(p[top_idx])
            canonical = idx_to_canonical.get(top_idx, -1)
            votes_per_speaker[spk].append((canonical, top_conf))
        batch_arrs.clear()
        batch_speakers.clear()

    pbar = tqdm(desc="Scanning + classifying", unit="ex")
    stop_reason: Optional[str] = None

    def _all_saturated() -> bool:
        return (
            not unlimited
            and len(accepted_counts) > 0
            and n_saturated == len(accepted_counts)
        )

    for example in dataset_iter:
        if max_examples is not None and n_seen >= max_examples:
            stop_reason = f"reached --max_examples {max_examples}"
            break
        n_seen += 1
        pbar.update(1)
        iters_since_new_speaker += 1

        # Periodic status so the user can see whether we're stuck in
        # saturated-skip land vs. still discovering speakers. Throttled to
        # avoid formatting overhead on every iteration.
        if n_seen % 1000 == 0:
            pbar.set_postfix(
                {
                    "speakers": len(accepted_counts),
                    "sat": "off" if unlimited else f"{n_saturated}/{len(accepted_counts)}",
                    "hrs": f"{duration_seconds / 3600:.2f}",
                    "no_new": iters_since_new_speaker,
                },
                refresh=False,
            )

        spk_raw = example.get(speaker_id_column)
        if spk_raw is None:
            continue
        spk = str(spk_raw)

        existing_count = accepted_counts.get(spk, 0)
        is_new_speaker = spk not in accepted_counts

        # Saturated-speaker fast skip (no audio decode). When unlimited, every
        # utterance is accepted regardless of count, so this block is skipped.
        if not unlimited and existing_count >= utterances_per_speaker:
            # Hard exit: --max_speakers cap reached + everyone full
            if max_speakers is not None and len(accepted_counts) >= max_speakers and _all_saturated():
                stop_reason = "all known speakers saturated and --max_speakers cap reached"
                break
            # Soft exit: every known speaker is full and we haven't seen
            # a new one in a while. Default safety net for the common case
            # where the user didn't set --max_speakers.
            if _all_saturated() and iters_since_new_speaker >= saturation_grace_period:
                stop_reason = (
                    f"all {len(accepted_counts)} known speakers saturated and "
                    f"{saturation_grace_period:,} iterations without a new speaker"
                )
                break
            continue

        # New-speaker fast skip if we've hit the --max_speakers cap.
        if is_new_speaker and max_speakers is not None and len(accepted_counts) >= max_speakers:
            continue

        # Audio decode (cost lives here).
        audio_field = example.get(audio_column)
        if not isinstance(audio_field, dict):
            # Some dataset shapes return numpy array directly — skip these,
            # we need the sampling rate to be guaranteed-correct.
            continue
        arr = audio_field.get("array")
        sr = audio_field.get("sampling_rate", target_sample_rate)
        if arr is None or len(arr) == 0:
            continue
        if sr != target_sample_rate:
            # Caller should have cast_column(Audio(sampling_rate=…)); warn
            # and skip if not, since the feature extractor is sample-rate
            # sensitive.
            print(
                f"  Skipping utterance with sr={sr} != target {target_sample_rate}. "
                "Cast your dataset via Audio(sampling_rate=target_sample_rate).",
                file=sys.stderr,
            )
            continue

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=0)  # mono
        if arr.shape[0] > max_audio_samples:
            arr = arr[:max_audio_samples]

        # Bookkeeping — count BEFORE buffering since saturation logic uses it.
        if is_new_speaker:
            accepted_counts[spk] = 1
            votes_per_speaker[spk] = []
            iters_since_new_speaker = 0
            # First utterance saturates only if utterances_per_speaker == 1
            if not unlimited and utterances_per_speaker == 1:
                n_saturated += 1
        else:
            accepted_counts[spk] += 1
            if not unlimited and accepted_counts[spk] == utterances_per_speaker:
                n_saturated += 1

        # Buffer for inline classification; flush when full so memory stays
        # bounded regardless of total accepted hours.
        batch_arrs.append(arr)
        batch_speakers.append(spk)
        if len(batch_arrs) >= batch_size:
            flush_batch()

        # Track *accepted* audio duration for the --max_hours budget so the
        # semantics align with the preprocessor's "stored-audio budget."
        duration_seconds += arr.shape[0] / target_sample_rate
        if max_seconds is not None and duration_seconds >= max_seconds:
            stop_reason = f"reached --max_hours {max_hours}"
            break
    pbar.close()

    # Drain final partial batch so its predictions land in votes_per_speaker.
    flush_batch()

    print(
        f"  Saw {n_seen:,} examples, classified utterances across {len(accepted_counts)} speakers "
        f"({n_saturated} fully saturated), {duration_seconds/3600:.3f}h of audio"
        + (f" (stop: {stop_reason})" if stop_reason else " (dataset exhausted)")
    )
    return votes_per_speaker


def _aggregate_votes(
    votes_per_speaker: Dict[str, List[Tuple[int, float]]],
    confidence_threshold: float,
) -> Dict[str, Tuple[int, float, int, int]]:
    """
    Collapse per-utterance ``(canonical, confidence)`` vote lists into a
    single per-speaker label. Per-utterance predictions below
    ``confidence_threshold`` (or with canonical == -1) are dropped from the
    vote. Returns ``(canonical_gender, confidence, n_winning_votes, n_total_votes)``
    per speaker; speakers with no surviving votes get ``(-1, 0.0, 0, n_total)``.
    """
    results: Dict[str, Tuple[int, float, int, int]] = {}
    for spk, per_utt in votes_per_speaker.items():
        kept = [(c, conf) for c, conf in per_utt if conf >= confidence_threshold and c != -1]
        if not kept:
            results[spk] = (-1, 0.0, 0, len(per_utt))
            continue
        tally: Dict[int, List[float]] = defaultdict(list)
        for c, conf in kept:
            tally[c].append(conf)
        winner = max(tally.keys(), key=lambda c: (len(tally[c]), sum(tally[c])))
        winner_conf = float(np.mean(tally[winner]))
        results[spk] = (winner, winner_conf, len(tally[winner]), len(per_utt))
    return results


def _write_speakers_txt(
    output_path: str,
    predictions: Dict[str, Tuple[int, float, int, int]],
    model_id: str,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    confidence_threshold: float,
    utterances_per_speaker: int,
) -> None:
    """Emit a SPEAKERS.TXT-compatible file. The trailing CONFIDENCE / N_VOTES /
    N_TOTAL columns are transparency-only; the preprocess-side parser only
    reads SPEAKER_ID and SEX."""
    n_male = sum(1 for v in predictions.values() if v[0] == 0)
    n_female = sum(1 for v in predictions.values() if v[0] == 1)
    n_unknown = sum(1 for v in predictions.values() if v[0] == -1)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(";Gender labels auto-generated by build_gender_lookup.py\n")
        f.write(f";Model: {model_id}\n")
        f.write(
            f";Dataset: {dataset_name}"
            + (f" config={dataset_config}" if dataset_config else "")
            + f" split={split}\n"
        )
        f.write(
            f";Voting: up to {utterances_per_speaker} utterances/speaker, "
            f"per-utt confidence_threshold = {confidence_threshold}\n"
        )
        f.write(
            f";Coverage: male={n_male}, female={n_female}, unknown={n_unknown}, "
            f"total={len(predictions)}\n"
        )
        f.write(";Format: SPEAKER_ID | SEX | CONFIDENCE | N_VOTES | N_TOTAL\n")
        f.write(";ID    | SEX | CONFIDENCE | N_VOTES | N_TOTAL\n")
        # Sort speakers: numeric IDs ascending if possible, else lexicographic
        def sort_key(s: str):
            try:
                return (0, int(s))
            except ValueError:
                return (1, s)
        for spk in sorted(predictions.keys(), key=sort_key):
            canonical, conf, n_votes, n_total = predictions[spk]
            sex = "M" if canonical == 0 else "F" if canonical == 1 else "?"
            f.write(f"{spk:<6}| {sex}  | {conf:.4f}     | {n_votes:>2}      | {n_total:>2}\n")
    print(
        f"\nWrote {output_path}: "
        f"male={n_male}, female={n_female}, unknown={n_unknown}, "
        f"total={len(predictions)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a SPEAKERS.TXT-compatible gender lookup file by "
        "running a HuggingFace audio gender classifier over a dataset and "
        "voting at the speaker level."
    )
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="HuggingFace dataset name, e.g. 'parler-tts/mls_eng'")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset config (e.g. 'en'). Optional.")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split (default: train)")
    parser.add_argument("--speaker_id_column", type=str, default="speaker_id",
                        help="Per-row speaker ID column name")
    parser.add_argument("--audio_column", type=str, default="audio",
                        help="Per-row audio column name (must decode to "
                             "{array, sampling_rate})")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output SPEAKERS.TXT-compatible file path")

    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID,
                        help="HuggingFace audio gender classifier model ID. "
                             "Must follow AutoModelForAudioClassification pattern "
                             "with a populated id2label.")
    parser.add_argument("--utterances_per_speaker", type=int, default=3,
                        help="Max utterances sampled per speaker for voting (default: 3). "
                             "Set to 0 to disable the per-speaker cap entirely and accept "
                             "every utterance — termination then relies on --max_hours and/or "
                             "--max_examples. Use 0 when you want a flat 'process N hours of "
                             "audio' scan without speaker-saturation logic.")
    parser.add_argument("--max_speakers", type=int, default=None,
                        help="Optional cap on the number of speakers to classify "
                             "(useful for spot checks; default: no cap)")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Stop iterating the dataset after this many rows. Counts "
                             "every row including those we cheaply skip (saturated "
                             "speakers, etc.). Use to bound total iteration cost on "
                             "huge corpora like MLS (~10M rows). Default: no cap.")
    parser.add_argument("--max_hours", type=float, default=None,
                        help="Stop after this many cumulative hours of *accepted* "
                             "utterance audio have been collected (post-clipping). "
                             "Matches the preprocessor's --max_hours semantics. "
                             "WARNING: this is an accepted-audio cap, not an iteration "
                             "cap — with small --utterances_per_speaker on huge corpora "
                             "this ceiling may be unreachable. Use --max_examples for a "
                             "hard iteration cap. Default: no cap.")
    parser.add_argument("--saturation_grace_period", type=int, default=50000,
                        help="Stop scanning once every known speaker is saturated AND "
                             "this many iterations have passed without finding a new "
                             "speaker. Guards against running to dataset exhaustion when "
                             "--max_speakers isn't pinned. Set lower (e.g. 5000) for "
                             "shuffled datasets where new speakers appear quickly; set "
                             "higher (e.g. 200000) for chapter-ordered datasets like MLS "
                             "where consecutive utterances often share a speaker. "
                             "Default: 50000.")
    parser.add_argument("--confidence_threshold", type=float, default=0.8,
                        help="Per-utterance confidence below which a vote is "
                             "discarded; speakers with no surviving votes get "
                             "labeled '?' (default: 0.8)")
    parser.add_argument("--max_audio_seconds", type=float, default=10.0,
                        help="Per-utterance audio clip cap in seconds (default: 10)")
    parser.add_argument("--target_sample_rate", type=int, default=16000,
                        help="Target sample rate (default: 16000)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Inference batch size (default: 8)")
    parser.add_argument("--streaming", action="store_true",
                        help="Stream dataset (avoids downloading full corpus). "
                             "Recommended for MLS-scale datasets.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Inference device")
    args = parser.parse_args()

    max_audio_samples = int(args.max_audio_seconds * args.target_sample_rate)

    # ---- 1. Load model ----
    feature_extractor, model, idx_to_canonical = _load_classifier(
        args.model_id, args.device,
    )

    # ---- 2. Load + cast dataset ----
    from datasets import load_dataset, Audio

    print(
        f"\nLoading dataset {args.dataset_name}"
        + (f" config={args.dataset_config}" if args.dataset_config else "")
        + f" split={args.split} (streaming={args.streaming})..."
    )
    ds = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
        streaming=args.streaming,
    )
    ds = ds.cast_column(args.audio_column, Audio(sampling_rate=args.target_sample_rate))

    # ---- 3. Stream-scan + classify in one pass (bounded memory) ----
    votes_per_speaker = _scan_and_classify(
        ds,
        feature_extractor=feature_extractor,
        model=model,
        idx_to_canonical=idx_to_canonical,
        speaker_id_column=args.speaker_id_column,
        audio_column=args.audio_column,
        utterances_per_speaker=args.utterances_per_speaker,
        max_speakers=args.max_speakers,
        max_examples=args.max_examples,
        max_hours=args.max_hours,
        saturation_grace_period=args.saturation_grace_period,
        max_audio_samples=max_audio_samples,
        target_sample_rate=args.target_sample_rate,
        batch_size=args.batch_size,
        device=args.device,
    )
    if not votes_per_speaker:
        print("ERROR: no usable utterances found. Check --speaker_id_column and "
              "--audio_column are correct for this dataset.")
        sys.exit(1)

    # ---- 4. Aggregate per-utterance votes into per-speaker labels ----
    predictions = _aggregate_votes(votes_per_speaker, args.confidence_threshold)

    # ---- 5. Emit lookup file ----
    _write_speakers_txt(
        args.output_path,
        predictions,
        model_id=args.model_id,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        confidence_threshold=args.confidence_threshold,
        utterances_per_speaker=args.utterances_per_speaker,
    )


if __name__ == "__main__":
    main()
