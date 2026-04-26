#!/bin/bash
pkill vmtouch 2>/dev/null || true
sudo umount /mnt/shards_ram 2>/dev/null || true
sudo rmdir /mnt/shards_ram 2>/dev/null || true
echo "Cleaned up. Free RAM:"
free -h
