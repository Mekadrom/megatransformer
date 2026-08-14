"""PCIe/chipset heat load: sustained bidirectional host<->device transfers.

Unlike a GEMM soak (compute stays ON the GPU, minimal PCIe), this ping-pongs large pinned
buffers H2D/D2H in a tight loop -> maximal PCIe-lane + chipset controller activity, with
near-zero GPU compute and near-zero CPU. Purpose: heat the WRX80 chipset (the component the
dead grate fan cools) to see if that fan spins up -- WITHOUT touching the CPU thermal wall.

  python pcie_soak.py [seconds] [n_gpus] [buffer_mb]     # default 300s, 2 gpus, 512MB
"""
import sys, time
import torch
import torch.multiprocessing as mp


def hammer(rank, seconds, mb):
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"
    n = mb * 1024 * 1024 // 4
    h = torch.empty(n, dtype=torch.float32).pin_memory()   # pinned host buffer
    d = torch.empty(n, dtype=torch.float32, device=dev)
    t0 = time.time()
    it = 0
    while time.time() - t0 < seconds:
        for _ in range(20):
            d.copy_(h, non_blocking=True)   # H2D over PCIe
            h.copy_(d, non_blocking=True)   # D2H over PCIe
        torch.cuda.synchronize(dev)
        it += 20
    gb = it * 2 * mb / 1024.0
    print(f"[gpu{rank}] {it} round-trips, ~{gb/ (time.time()-t0):.1f} GB/s PCIe", flush=True)


if __name__ == "__main__":
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    ngpu = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    mb = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    print(f"PCIe soak: {ngpu} GPU(s), {mb}MB buffers, {seconds}s (heats the chipset, not the CPU)", flush=True)
    mp.set_start_method("spawn")
    procs = [mp.Process(target=hammer, args=(r, seconds, mb)) for r in range(ngpu)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print("PCIe soak complete.", flush=True)
