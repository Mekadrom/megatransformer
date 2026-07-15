"""Shared plumbing so the diagnostics suite works on a contour/units SMG.

The suite was written for the continuous-feature, `f0_predictor_input="features"` SMG:
it built its dataset with no codebook and called `decode(feat, speaker_embedding=emb,
features=feat)`. Both assumptions break on a quantized-units + contour SMG:

  * no codebook  -> `features` stay CONTINUOUS ContentVec, but the SMG trained on
                    centroids. The decode silently runs off-manifold and every metric
                    measures a model/input pair that never existed.
  * no f0_contour -> `f0_predictor_input="contour"` raises, because its F0 trunk reads a
                    1-channel contour and there is no prosody in its input to fall back on.

Both are the SAME artifact: the codebook ships the centroids AND the per-speaker F0 stats
that define the contour, so passing `--voice_codebook_path` fixes both at once.

Note what an embedding swap means here. The contour is speaker-NORMALIZED and comes from
the SOURCE utterance; the SMG's F0 trunk denormalizes it with the GIVEN ECAPA embedding.
So swapping the embedding re-pitches the source's prosodic shape into the target
speaker's range -- which is exactly the conversion the metric wants to measure, and it
routes through ECAPA by construction rather than by hope.
"""
from typing import Optional

import torch


def add_codebook_arg(ap):
    ap.add_argument("--voice_codebook_path", default=None,
                    help="k-means codebook .pt (from scripts.data.voice.fit_codebook). REQUIRED "
                         "for an SMG trained on quantized units: it snaps features to centroids "
                         "(matching training) and carries the per-speaker F0 stats that define "
                         "the speaker-normalized contour an f0_predictor_input='contour' SMG "
                         "reads. Omit only for a continuous-feature SMG.")


def needs_contour(model) -> bool:
    return getattr(model, "f0_predictor_input", "features") == "contour"


def check_compat(model, codebook_path):
    """Fail before burning a GPU sweep on a model/input pair that never existed."""
    if needs_contour(model) and not codebook_path:
        raise SystemExit(
            "This SMG has f0_predictor_input='contour' but no --voice_codebook_path was "
            "given. Its F0 trunk reads a 1-channel speaker-normalized contour, which is "
            "derived from the per-speaker F0 stats inside the codebook artifact; without "
            "it decode() raises. Pass the SAME codebook the SMG was trained with."
        )
    if codebook_path and not needs_contour(model):
        print("[note] --voice_codebook_path given to a 'features' SMG: features will be "
              "quantized (matching a units-trained SMG) but F0 still comes from content.")


def sample_contour(sample, model, device, dtype=torch.float32) -> Optional[torch.Tensor]:
    """(1, T) contour for `decode(f0_contour=...)`, or None for a features SMG."""
    if not needs_contour(model):
        return None
    c = sample.get("f0_contour")
    if c is None:
        raise SystemExit(
            "Contour SMG, but the dataset produced no 'f0_contour'. Ensure the shard has an "
            "'f0' column, that 'f0' is in --columns, and that the codebook carries F0 stats."
        )
    return c.float().reshape(1, -1).to(device=device, dtype=dtype)


def decode(model, feat, emb, contour):
    """decode() with the contour attached when the model wants one, trimmed to align.

    Trims rather than interpolates: features and contour come off the same 50 Hz frame
    grid, so a length mismatch is a bug to surface, not a resampling job.
    """
    kwargs = {}
    if contour is not None:
        if contour.shape[-1] != feat.shape[-1]:
            T = min(contour.shape[-1], feat.shape[-1])
            contour, feat = contour[..., :T], feat[..., :T]
        kwargs["f0_contour"] = contour
    out = model.decode(feat, speaker_embedding=emb, features=feat, **kwargs)
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, dict):
        out = out.get("reconstructed", next(iter(out.values())))
    return out
