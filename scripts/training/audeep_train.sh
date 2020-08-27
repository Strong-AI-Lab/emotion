#!/bin/sh

# Trains an auDeep model using the given spectrograms as input.
# This script must be run in the audeep docker container.

spectrograms=$1

audeep t-rae train \
    --input $spectrograms \
    --run-name logs/audeep/2x256_u_b/all-0.05-0.025-120-60_b64_l0.001 \
    --num-epochs 50 \
    --batch-size 64 \
    --checkpoints-to-keep 1 \
    --learning-rate 0.001 \
    --keep-prob 0.8 \
    --num-layers 2 \
    --num-units 256 \
    --bidirectional-decoder
