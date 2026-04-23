#!/bin/bash
set -e
RAM_DIR=/mnt/shards_ram
SRC_DIR=./cached_datasets
sudo mkdir -p $RAM_DIR
sudo mount -t tmpfs -o size=200G tmpfs $RAM_DIR
sudo chown $USER:$USER $RAM_DIR
cp -r $SRC_DIR/voice_sive_tiny_deep_3xdownsample_conv2d_batchnorm_0_0_layer10_train $RAM_DIR/
cp -r $SRC_DIR/image_vae_train $RAM_DIR/
cp -r $SRC_DIR/text_pile_train $RAM_DIR/
echo "Train shards copied to $RAM_DIR"
du -sh $RAM_DIR/*
