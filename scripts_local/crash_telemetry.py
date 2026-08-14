#!/usr/bin/env python3
"""Crash-forensics telemetry: sample CPU + GPU state every second and fsync each line.

The point of the fsync: on a hard power cut the OS log loses whatever is still in the
page cache, which is why an early crash left no evidence. fsync-per-line means the last
sample is physically on disk, so the tail of this file IS the state at the cutoff.

POWER: the 1 Hz averaged reading (and nvtop) can look normal (<300W) while the box still
OPP-trips, because the PSU reacts to sub-millisecond current spikes. So when pynvml is
available we poll INSTANT power in-process at ~200 Hz and log the per-second PEAK next to
the average — this catches millisecond-scale excursions (expect 4090 peaks well above the
average) that the averaged field misses. NOTE: true sub-ms PCIe transients are below even
NVML's instant resolution; catching those needs a hardware clamp/scope on the 12V rails.
Falls back to nvidia-smi (average only) if pynvml is missing, so logging never breaks.

  python3 crash_telemetry.py [outfile] [interval_s] [peak_hz]

NVML line:  ts, cpu_tctl_C | per-GPU: idx,avg_W,PEAK_W,temp_C,sm_MHz,util%,throttle_hex
smi  line:  ts, cpu_tctl_C | per-GPU: idx,power_W,temp_C,sm_MHz,util%,throttle_hex
throttle bitmask: 0x1=GpuIdle 0x4=SwPowerCap 0x8=HwSlowdown 0x20=SwThermalSlowdown
                  0x40=HwThermalSlowdown 0x80=HwPowerBrakeSlowdown
"""
import glob
import os
import subprocess
import sys
import time

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/crash_telemetry.csv")
INTERVAL = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
PEAK_HZ = float(sys.argv[3]) if len(sys.argv) > 3 else 200.0
GPU_Q = "index,power.draw,temperature.gpu,clocks.sm,utilization.gpu,clocks_event_reasons.active"


def _find_tctl():
    """Path to the CPU Tctl/Tdie sysfs input (k10temp). A THERMTRIP cuts power in
    hardware with no log, so a climbing Tctl right before the tail ends is its
    fingerprint (ruled out for the 2026-07-17 crash — Tctl was flat ~67C)."""
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            if open(os.path.join(h, "name")).read().strip() != "k10temp":
                continue
            for lbl in glob.glob(os.path.join(h, "temp*_label")):
                if open(lbl).read().strip() in ("Tctl", "Tdie"):
                    return lbl.replace("_label", "_input")
        except Exception:
            pass
    return None


TCTL = _find_tctl()


def cpu_temp():
    if not TCTL:
        return float("nan")
    try:
        return int(open(TCTL).read()) / 1000.0
    except Exception:
        return float("nan")


# ---- NVML path (preferred): in-process, high-rate instant-power peak-hold ----
try:
    import pynvml as N
    N.nvmlInit()
    _H = [N.nvmlDeviceGetHandleByIndex(i) for i in range(N.nvmlDeviceGetCount())]
    NVML = True
except Exception:
    NVML = False


def gpus_nvml(dt):
    """Peak-hold instant power over dt seconds at ~PEAK_HZ; slower metrics sampled once.
    This call also IS the per-line wait, so no extra sleep is needed."""
    n = len(_H)
    pmax = [0.0] * n
    psum = [0.0] * n
    cnt = 0
    period = 1.0 / PEAK_HZ
    t_end = time.time() + dt
    while time.time() < t_end:
        for i, h in enumerate(_H):
            try:
                w = N.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                w = float("nan")
            if w == w:  # not NaN
                if w > pmax[i]:
                    pmax[i] = w
                psum[i] += w
        cnt += 1
        time.sleep(period)
    parts = []
    for i, h in enumerate(_H):
        avg = (psum[i] / cnt) if cnt else float("nan")
        try:
            temp = N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU)
            clk = N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM)
            util = N.nvmlDeviceGetUtilizationRates(h).gpu
        except Exception:
            temp = clk = util = -1
        try:
            thr = N.nvmlDeviceGetCurrentClocksThrottleReasons(h)
        except Exception:
            thr = 0
        parts.append(f"{i}, {avg:.1f}, {pmax[i]:.1f}, {temp}, {clk}, {util}, 0x{thr:016x}")
    return " | ".join(parts)


def gpus_smi():
    """Fallback: single 1 Hz averaged snapshot via nvidia-smi (no peak)."""
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_Q}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        return " | ".join(l.strip() for l in r.stdout.strip().splitlines())
    except Exception as e:
        return f"GPU_ERR:{e}"


def main():
    f = open(OUT, "a")
    mode = f"NVML peak-hold {PEAK_HZ:.0f}Hz" if NVML else "nvidia-smi avg-only"
    f.write(f"# start {time.strftime('%F %T')} interval={INTERVAL}s mode={mode} tctl_src={TCTL}\n")
    f.flush()
    os.fsync(f.fileno())
    while True:
        if NVML:
            gpu = gpus_nvml(INTERVAL)          # this call spans ~INTERVAL seconds
        else:
            gpu = gpus_smi()
            time.sleep(INTERVAL)
        line = f"{time.strftime('%F %T')},cpu_tctl={cpu_temp():.1f} | {gpu}\n"
        f.write(line)
        f.flush()
        os.fsync(f.fileno())   # survives a hard power cut


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
