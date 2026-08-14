"""All-core CPU soak: one single-threaded FP-heavy worker per core.

Companion to stress_gpus.py for a COMBINED system-power test. Real training loads the CPU
hard via dataloader workers (image decode, audio STFT, augmentation) at the same time the
GPUs draw slot power -- both sit on the board's power domain, and the original crash logged
a CPU-temp event alongside the GPU. This burns every core with vectorized (AVX) matmuls to
push CPU package power near its ceiling, standing in for that dataloader draw.

One process per core, each pinned to a SINGLE BLAS thread, so N procs = N busy cores with no
oversubscription (matches N independent dataloader workers rather than one BLAS storm).

  python cpu_soak.py [seconds] [workers] [size]     # default 300s, nproc workers, 512
"""
import os, sys, time

# Must be set BEFORE numpy imports: one thread per worker process.
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import multiprocessing as mp
import numpy as np


def burn(seconds: float, size: int):
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)
    t0 = time.time()
    iters = 0
    while time.time() - t0 < seconds:
        c = a @ b                      # AVX FP matmul = high package power
        a = c * (1.0 / (np.abs(c).max() + 1e-6))  # renormalize -> stays finite, no inf/nan
        iters += 1
    return iters


if __name__ == "__main__":
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 300
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    print(f"CPU soak: {workers} single-thread workers for {seconds:.0f}s, {size}x{size} f32 matmul",
          flush=True)
    procs = [mp.Process(target=burn, args=(seconds, size)) for _ in range(workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print("CPU soak complete.", flush=True)
