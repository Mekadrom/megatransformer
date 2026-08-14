"""4-GPU synchronized load-transient (di/dt) stress: square-wave idle<->full cycling.

A steady soak tests sustained rail current; it does NOT test the PSU's transient
response. The failure mode that actually trips a marginal supply is the RISING EDGE --
all cards slamming from idle to ~450W at once, a large di/dt the regulation loop and
bulk caps must absorb in microseconds. OPP (over-power protection) reacts to these
sub-millisecond spikes even when the 1Hz average looks safe (see crash_telemetry.py).

This cycles every GPU between a full-tilt bf16 GEMM burst and a true idle, SYNCHRONIZED
across all cards via the wall clock (no IPC): every process computes its on/off phase
from the same absolute time, so all four transition together -- the worst case for the
shared 12V/3.3V rails (the transients ADD instead of averaging out).

  python spike_gpus.py [seconds] [period_ms] [duty] [size]
      seconds    total run (default 300)
      period_ms  one on+off cycle (default 200 => 5 Hz)
      duty       fraction of each cycle spent at FULL load (default 0.5)
      size       GEMM dim (default 8192; bigger = higher peak W)

Aggressive edge-count example (fast cycling, brief bursts):
  python spike_gpus.py 300 120 0.4 8192      # ~8.3 Hz, 48ms bursts

Watch the PEAK_W column of crash_telemetry.py (200Hz instant sampler) -- the average
will read low while the per-second peak rides near the cap on every burst.
"""
import sys, time
import torch
import torch.multiprocessing as mp


def spike(rank: int, seconds: float, period: float, duty: float, size: int):
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"
    a = torch.randn(size, size, device=dev, dtype=torch.bfloat16)
    b = torch.randn(size, size, device=dev, dtype=torch.bfloat16)
    # warm up cublas so the first ON edge isn't diluted by kernel autotuning
    torch.matmul(a, b); torch.cuda.synchronize(dev)

    t0 = time.time()
    cycles = full_iters = 0
    on_len = period * duty
    while True:
        now = time.time() - t0
        if now >= seconds:
            break
        phase = now % period
        if phase < on_len:
            # ON: queue a short burst of matmuls, then re-check phase. Kept short so the
            # OFF edge lands crisply on time rather than overrunning by a whole big GEMM.
            for _ in range(4):
                c = torch.matmul(a, b)
            torch.cuda.synchronize(dev)
            full_iters += 4
        else:
            # OFF: no CUDA work at all -> clocks gate down, power falls toward idle, so the
            # next ON edge is a genuine idle->full transient. Sleep to the next period start.
            torch.cuda.synchronize(dev)
            cycles += 1
            time.sleep(max(0.0, period - phase))
    torch.cuda.synchronize(dev)
    print(f"[gpu{rank}] {cycles} cycles, {full_iters} matmuls over {seconds:.0f}s "
          f"({period*1000:.0f}ms period, duty {duty})", flush=True)


if __name__ == "__main__":
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 300
    period = (float(sys.argv[2]) if len(sys.argv) > 2 else 200) / 1000.0
    duty = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    size = int(sys.argv[4]) if len(sys.argv) > 4 else 8192
    n = torch.cuda.device_count()
    print(f"spiking {n} GPUs for {seconds:.0f}s: {period*1000:.0f}ms period, duty {duty}, "
          f"{size}x{size} bf16 (synchronized idle<->full square wave)", flush=True)
    mp.set_start_method("spawn")
    procs = [mp.Process(target=spike, args=(r, seconds, period, duty, size)) for r in range(n)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print("spike test complete (box survived the transients).", flush=True)
