#!/bin/sh

# Generates representations for corpus spectrograms using model trained on
# combined spectrograms.
# This script must be run in the audeep docker container.

corpus=$1

audeep t-rae generate \
    --model-dir logs/audeep/2x256_u_b/all-0.05-0.025-120-60_b64_l0.001/logs \
    --input datasets/spectrograms/${corpus}-0.05-0.025-120-60.nc \
    --output datasets/${corpus}/output/audeep.nc
echo "Wrote representations to datasets/${corpus}/output/audeep.nc"
