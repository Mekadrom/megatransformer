#!/bin/bash
# Installs the GPU power-limit systemd unit (-pl 300 at boot) + the login warning.
# Run with sudo:   sudo bash scripts_local/install_powerlimit.sh
set -e
SRC="$(cd "$(dirname "$0")" && pwd)"

install -m 644 "$SRC/gpu-powerlimit.service" /etc/systemd/system/gpu-powerlimit.service
install -m 755 "$SRC/99-gpu-powerlimit"      /etc/update-motd.d/99-gpu-powerlimit

systemctl daemon-reload
systemctl enable gpu-powerlimit.service
systemctl start  gpu-powerlimit.service   # applies -pl 300 right now, too

echo "=== gpu-powerlimit.service status ==="
systemctl --no-pager --lines=0 status gpu-powerlimit.service || true
echo "=== enforced power limits now ==="
nvidia-smi --query-gpu=index,enforced.power.limit --format=csv
# refresh the cached MOTD so the banner shows on the next shell without a full re-login
run-parts /etc/update-motd.d/ >/dev/null 2>&1 || true
echo
echo "Installed. The login banner: /etc/update-motd.d/99-gpu-powerlimit (run it directly to preview)."
