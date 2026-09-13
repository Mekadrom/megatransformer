"""CosyVoice 2 content + speaker encoders, with MINIMAL dependencies.

Split out of scripts/data/voice/preprocess.py so that consumers which only need the FSQ
tokenizer and campplus -- the chat UI's Reference voice control, any eval that encodes a
reference clip -- do not import the whole preprocessing stack. That module pulls `datasets`,
`torchcrepe`, SIVE and the speaker-encoder zoo at import time, and each of those failed in
turn on a machine that only wanted to tokenise three seconds of audio.

Everything here needs only: torch, torchaudio, numpy, onnxruntime, and whisper's bundled
mel_filters.npz asset (read directly, never imported -- see whisper_log_mel).
"""
import os
import sys

import numpy as np
import torch
import torchaudio

from megatransformer.scripts.data.preprocessor import BatchProcessor


def _preload_venv_cuda_libs() -> int:
    """dlopen the venv's bundled NVIDIA libs so onnxruntime's CUDA EP can resolve them.

    onnxruntime-gpu links against cuDNN 9 / CUDA 12 but does NOT ship them; torch's cu124
    wheels DO, under site-packages/nvidia/*/lib. Without this the CUDA provider fails to
    load and onnxruntime SILENTLY falls back to CPU -- `get_providers()` still lists
    CUDAExecutionProvider as available while the session runs on CPU at ~50x the cost. That
    is a throughput bug with no error message, so preload here rather than relying on the
    caller having exported LD_LIBRARY_PATH.
    """
    import ctypes, glob
    base = os.path.join(os.path.dirname(sys.executable), "..", "lib",
                        f"python{sys.version_info.major}.{sys.version_info.minor}",
                        "site-packages", "nvidia")
    n = 0
    for pat in ("cudnn/lib/libcudnn*.so*", "cublas/lib/libcublas*.so*",
                "cufft/lib/libcufft*.so*", "curand/lib/libcurand*.so*"):
        for f in sorted(glob.glob(os.path.join(base, pat))):
            try:
                ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
                n += 1
            except OSError:
                pass
    return n


_WHISPER_MEL_CACHE: dict = {}


def whisper_log_mel(wav16: "torch.Tensor", n_mels: int = 128) -> "torch.Tensor":
    """whisper.log_mel_spectrogram without importing whisper.

    openai-whisper imports numba (for its timing module), which caps NumPy at 2.4 and
    hard-fails on a newer venv -- and the FSQ tokenizer needs exactly one function from it.
    The mel filterbank is a bundled asset, so read the .npz directly: importlib.util.find_spec
    locates the package WITHOUT executing its __init__, which is what pulls numba.

    Constants are whisper's own (audio.py): SAMPLE_RATE 16000, N_FFT 400, HOP_LENGTH 160,
    magnitude SQUARED, log10, an 8-decade dynamic-range floor, then (x + 4) / 4.
    """
    import importlib.util
    import numpy as np
    key = int(n_mels)
    if key not in _WHISPER_MEL_CACHE:
        spec = importlib.util.find_spec("whisper")
        if spec is None or not spec.origin:
            raise ImportError("openai-whisper must be installed (its mel_filters.npz asset is "
                              "read directly; the package itself is never imported)")
        npz = os.path.join(os.path.dirname(spec.origin), "assets", "mel_filters.npz")
        with np.load(npz, allow_pickle=False) as z:
            _WHISPER_MEL_CACHE[key] = torch.from_numpy(z[f"mel_{key}"]).float()
    filters = _WHISPER_MEL_CACHE[key].to(wav16.device)
    window = torch.hann_window(400, device=wav16.device)
    stft = torch.stft(wav16.reshape(-1), 400, 160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    mel = filters @ magnitudes
    log_spec = torch.clamp(mel, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).unsqueeze(0)


class CosyVoice2BatchProcessor(BatchProcessor):
    """CosyVoice 2 speech_tokenizer_v2 (FSQ) as a DISCRETE content tokenizer.

    Emits INTEGER unit ids at 25 Hz -- the same ids the frozen CosyVoice 2 flow decoder
    consumes, so a predicted id indexes straight into `flow.input_embedding`. This is the
    encoder the world-voice runs use; previously it lived in an out-of-repo script
    (cosyvoice-runtime/scripts/extract_libritts_full.py) whose output then needed a SECOND
    pass through assemble_cosyvoice_cache.py. Both stages now happen here.

    Ported faithfully from that script, including two details that matter:
      - the tokenizer wants whisper's 128-bin log-mel at 16 kHz, and takes the frame count
        as a second int32 input;
      - `cudnn_conv_algo_search: HEURISTIC` avoids EXHAUSTIVE per-shape cudnn autotune,
        which the reference measured at 343 ms/utterance against ~4 ms with the heuristic,
        because every new sequence length triggers a fresh search.
    """

    def __init__(self, model_dir: str, voice_max_frames: int, mel_frame_rate: float,
                 device: str = "cuda", source_sr: int = 16000, cpu_threads: int = 0):
        import onnxruntime as ort
        # NO `import whisper` here. We only need its mel_filters.npz asset, which
        # whisper_log_mel() reads via importlib.util.find_spec WITHOUT executing the package
        # __init__ -- that __init__ pulls numba, which caps NumPy at 2.4. Fail early with a
        # useful message if the asset is absent, rather than at the first process_batch call.
        whisper_log_mel(torch.zeros(400), n_mels=128)
        self.frame_rate = 25.0
        self.source_sr = source_sr
        # voice_max_frames counts MEL frames; convert to 25 Hz unit frames the way
        # MimiBatchProcessor does, or every row is padded to the mel width instead.
        self.max_id_frames = int(voice_max_frames * self.frame_rate / float(mel_frame_rate)) + 1
        self.encoder_dim = 512          # CosyVoice 2 flow.input_embedding width
        self.num_layers = 1
        self._resamplers = {}

        want_cuda = str(device).startswith("cuda")
        if want_cuda:
            k = _preload_venv_cuda_libs()
            print(f"  preloaded {k} bundled CUDA libs for onnxruntime")
        if not cpu_threads:
            cpu_threads = int(os.environ.get("ONNX_CPU_THREADS",
                                             os.environ.get("OMP_NUM_THREADS", 6)))
        opt = ort.SessionOptions()
        opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opt.intra_op_num_threads = 1 if want_cuda else max(1, int(cpu_threads))
        providers = ([("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC",
                                                 "do_copy_in_default_stream": True}),
                      "CPUExecutionProvider"] if want_cuda else ["CPUExecutionProvider"])
        self.tok = ort.InferenceSession(
            os.path.join(model_dir, "speech_tokenizer_v2.onnx"), sess_options=opt,
            providers=providers)
        got = self.tok.get_providers()
        print(f"  CosyVoice 2 tokenizer providers: {got}")
        if want_cuda and "CUDAExecutionProvider" not in got:
            print("  [warn] CUDA provider did NOT load -- running on CPU, expect ~50x slower. "
                  "onnxruntime-gpu needs cuDNN 9 / CUDA 12 visible.")
        self._in_feats = self.tok.get_inputs()[0].name
        self._in_len = self.tok.get_inputs()[1].name

    def _resample(self, wav_t, sr):
        import torchaudio
        if sr == 16000:
            return wav_t
        if sr not in self._resamplers:
            self._resamplers[sr] = torchaudio.transforms.Resample(sr, 16000)
        return self._resamplers[sr](wav_t)

    @torch.no_grad()
    def process_batch(self, waveforms, waveform_lengths, mel_spec_lengths=None):
        import numpy as np
        B = len(waveforms)
        unit_ids = torch.zeros(B, self.max_id_frames, dtype=torch.long)
        feat_lengths = torch.zeros(B, dtype=torch.long)
        for i in range(B):
            wlen = int(waveform_lengths[i].item())
            wav = waveforms[i][:wlen].float().reshape(1, -1).cpu()      # trim pad first
            wav16 = self._resample(wav, self.source_sr)
            feat = whisper_log_mel(wav16, n_mels=128).numpy()
            ids = self.tok.run(None, {self._in_feats: feat,
                                      self._in_len: np.array([feat.shape[2]], dtype=np.int32)}
                               )[0].flatten().astype("int64")
            L = min(int(ids.shape[0]), self.max_id_frames)
            unit_ids[i, :L] = torch.from_numpy(ids[:L])
            feat_lengths[i] = max(L, 1)
        return {"unit_ids": unit_ids, "feature_lengths": feat_lengths}


class CampplusBatchProcessor(BatchProcessor):
    """CosyVoice 2's campplus.onnx speaker encoder -> 192-d embedding.

    The world-voice cache stores THESE, not ECAPA/WavLM: the frozen flow decoder is
    conditioned on campplus vectors, so any other speaker encoder would be off-manifold at
    decode. Runs on CPU in the reference implementation and is cheap enough to leave there.
    """

    def __init__(self, model_dir: str, source_sr: int = 16000, cpu_threads: int = 0):
        import onnxruntime as ort
        # Thread budget: hardcoding this oversubscribes badly when several preprocessing
        # processes share a box. Measured 2026-08-31: 4 processes on 32 cores, each with
        # torch's default 16 intra-op threads plus 6 here, gave load average 70 and
        # 5.4 utt/s TOTAL -- less than half the 12.5 utt/s a single process reached alone.
        # Respect OMP_NUM_THREADS (which torch also honours) so one env var caps the whole
        # process, and set OMP_NUM_THREADS ~= cores/n_processes when sharding.
        if not cpu_threads:
            cpu_threads = int(os.environ.get("ONNX_CPU_THREADS",
                                             os.environ.get("OMP_NUM_THREADS", 6)))
        opt = ort.SessionOptions()
        opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opt.intra_op_num_threads = max(1, int(cpu_threads))
        self.spk = ort.InferenceSession(os.path.join(model_dir, "campplus.onnx"),
                                        sess_options=opt, providers=["CPUExecutionProvider"])
        self._in0 = self.spk.get_inputs()[0].name
        self.source_sr = source_sr
        self.embedding_dim = 192
        self._resamplers = {}

    def _resample(self, wav_t, sr):
        import torchaudio
        if sr == 16000:
            return wav_t
        if sr not in self._resamplers:
            self._resamplers[sr] = torchaudio.transforms.Resample(sr, 16000)
        return self._resamplers[sr](wav_t)

    @torch.no_grad()
    def process_batch(self, waveforms, waveform_lengths, mel_spec_lengths=None):
        import torchaudio.compliance.kaldi as kaldi
        out = torch.zeros(len(waveforms), self.embedding_dim, dtype=torch.float32)
        for i in range(len(waveforms)):
            wlen = int(waveform_lengths[i].item())
            wav = waveforms[i][:wlen].float().reshape(1, -1).cpu()
            wav16 = self._resample(wav, self.source_sr)
            fb = kaldi.fbank(wav16, num_mel_bins=80, dither=0, sample_frequency=16000)
            fb = fb - fb.mean(dim=0, keepdim=True)
            emb = self.spk.run(None, {self._in0: fb.unsqueeze(0).numpy()})[0].flatten()
            out[i] = torch.from_numpy(emb.astype("float32"))
        return {"speaker_embeddings": out}


