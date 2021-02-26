#!/bin/bash

# Trains an auDeep model using the given spectrograms as input.
# This script must be run in the audeep docker container.

spectrograms=$1
name=$(basename $spectrograms)
name=${name%.nc}
name=${name/spectrograms_audeep/all}

bs=128
lr=0.001
audeep t-rae train \
    --input $spectrograms \
    --run-name logs/audeep/2x256_u_b/${name}_b${bs}_l${lr} \
    --num-epochs 250 \
    --batch-size $bs \
    --checkpoints-to-keep 5 \
    --learning-rate $lr \
    --keep-prob 0.8 \
    --num-layers 2 \
    --num-units 256 \
    --bidirectional-decoder
