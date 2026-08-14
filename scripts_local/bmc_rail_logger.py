"""Continuous BMC rail witness: poll ipmitool for the power rails and fsync each line.

Companion to crash_telemetry.py (which logs CPU/GPU). This logs the RAILS the BMC reads --
especially +3.3V, the rail the Add2PSU was sagging. fsync-per-line so the last reading before
a hard cut is physically on disk. Requires /dev/ipmi0 readable (udev rule already in place).

  python bmc_rail_logger.py [outfile] [interval_s]     # default ~/bmc_rails.csv, 2s
"""
import os, sys, time, subprocess

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/bmc_rails.csv")
INTERVAL = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
RAILS = ["+3.3V", "+3.3V_ALW", "+CPU_3.3V", "+5V", "+5V_ALW", "+12V", "+VCORE", "+VSOC"]


def read_rails():
    try:
        out = subprocess.run(["ipmitool", "sdr", "type", "Voltage"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return {}
    vals = {}
    for line in out.splitlines():
        p = [x.strip() for x in line.split("|")]
        if len(p) >= 5 and p[0] in RAILS:
            try:
                vals[p[0]] = float(p[4].replace("Volts", "").strip())
            except ValueError:
                pass
    return vals


hdr = "timestamp," + ",".join(RAILS)
newfile = not os.path.exists(OUT)
with open(OUT, "a", buffering=1) as f:
    if newfile:
        f.write(hdr + "\n"); f.flush(); os.fsync(f.fileno())
    while True:
        v = read_rails()
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        row = ts + "," + ",".join(f"{v.get(r, 'NA')}" for r in RAILS)
        f.write(row + "\n"); f.flush(); os.fsync(f.fileno())
        time.sleep(INTERVAL)
