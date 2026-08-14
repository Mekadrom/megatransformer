"""4-GPU thermal soak: sustained bf16 tensor-core matmuls on every GPU at once.

Highest-power, highest-heat workload (bf16 GEMM pins the tensor cores near the
450W limit). One process per GPU for true simultaneous saturation.

  python stress_gpus.py [seconds] [matrix_size]     # default 300s, 8192

Watch thermals in another shell:
  nvidia-smi dmon -s puct          # power / util / clocks / temp, 1Hz
  watch -n2 nvidia-smi --query-gpu=index,temperature.gpu,power.draw,clocks.sm,pcie.link.gen.current --format=csv
"""
import sys, time
import torch
import torch.multiprocessing as mp


def burn(rank: int, seconds: int, size: int):
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"
    a = torch.randn(size, size, device=dev, dtype=torch.bfloat16)
    b = torch.randn(size, size, device=dev, dtype=torch.bfloat16)
    t0 = time.time()
    iters = 0
    while time.time() - t0 < seconds:
        for _ in range(100):                 # queue 100 async matmuls, then sync
            c = torch.matmul(a, b)           # a,b fixed -> no bf16 overflow, full tensor-core load
        torch.cuda.synchronize(dev)
        iters += 100
    # rough TFLOPs: 2*N^3 per matmul
    tflops = (2 * size**3 * iters) / (time.time() - t0) / 1e12
    print(f"[gpu{rank}] {iters} matmuls, ~{tflops:.0f} TFLOP/s sustained", flush=True)


if __name__ == "__main__":
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
    n = torch.cuda.device_count()
    print(f"soaking {n} GPUs for {seconds}s at {size}x{size} bf16", flush=True)
    mp.set_start_method("spawn")
    procs = [mp.Process(target=burn, args=(r, seconds, size)) for r in range(n)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print("soak complete", flush=True)
