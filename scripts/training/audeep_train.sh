#!/bin/sh

# Trains an auDeep model using the given spectrograms as input.
# This script must be run in the audeep docker container.

spectrograms=$1
name=$(basename $spectrograms)

audeep t-rae train \
    --input $spectrograms \
    --run-name logs/audeep/2x256_u_b/${name%.nc}_b64_l0.001 \
    --num-epochs 100 \
    --batch-size 64 \
    --checkpoints-to-keep 1 \
    --learning-rate 0.001 \
    --keep-prob 0.8 \
    --num-layers 2 \
    --num-units 256 \
    --bidirectional-decoder
