#!/bin/bash

# Trains an auDeep model using the given spectrograms as input.
# This script must be run in the audeep docker container.

input=$1
shift
run_name=$1
shift

audeep t-rae train \
    --input $input \
    --run-name $run_name \
    --checkpoints-to-keep 5 \
    --keep-prob 0.8 \
    --num-layers 2 \
    --num-units 256 \
    --bidirectional-decoder \
    $*
