#!/bin/bash

python3 -m tensorboard.main --logdir "runs/$2" --port $1 --host 0.0.0.0 --reload_interval 1 --samples_per_plugin scalars=999999,images=999999,audio=999999
