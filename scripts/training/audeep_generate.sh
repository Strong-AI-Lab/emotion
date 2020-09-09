#!/bin/sh

# Generates representations for corpus spectrograms using model trained on
# combined spectrograms.
# This script must be run in the audeep docker container.

corpus=$1

audeep t-rae generate \
    --model-dir logs/audeep/2x256_u_b/all-0.05-0.025-240-60_b64_l0.001/logs \
    --input spectrograms/${corpus}-0.05-0.025-240-60.nc \
    --output output/${corpus}/audeep-0.05-0.025-240-60_b64_l0.001.nc
echo "Wrote representations to datasets/${corpus}/output/audeep.nc"
python scripts/utils/audeep_corpus_name.py output/${corpus}/audeep-0.05-0.025-240-60_b64_l0.001.nc $corpus
