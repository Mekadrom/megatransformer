#!/bin/bash
# usage: ./tensorboard.sh <port> <subdir>     e.g. ./tensorboard.sh 6008 world_image

# rustboard (tensorboard_data_server) opens and HOLDS every .tfevents file under --logdir.
# runs/world_image passed 1000 event files, against the inherited soft limit of 1024, which
# surfaces as:
#     WARN rustboard_core::disk_logdir] While walking log directory:
#     IO error for operation on runs/world_image: Too many open files (os error 24)
# It is not an FD leak, so restarting TB does not clear it -- a fresh process inherits the same
# 1024 and re-opens the same files. The hard limit is already 1048576, so raising the SOFT limit
# needs no root. Every eval writes its own event file (2 per checkpoint at w=1.0/w=3.0), so the
# count only grows; 65536 buys a very long runway.
ulimit -n 65536 2>/dev/null || echo "[tensorboard.sh] WARN: could not raise open-file limit; \
large logdirs may hit 'Too many open files'" >&2

python3 -m tensorboard.main --logdir "runs/$2" --port $1 --host 0.0.0.0 --reload_interval 1 --samples_per_plugin scalars=999999,images=999999,audio=999999
