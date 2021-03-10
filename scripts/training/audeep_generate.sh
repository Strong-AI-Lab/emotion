#!/bin/sh

# Generates representations for corpus spectrograms using model trained on
# combined spectrograms.
# This script must be run in the audeep docker container.

corpus=$1

$params=5-0.025-0.010-240-60_b128_l0.001
audeep t-rae generate \
    --model-dir logs/audeep/2x256_u_b/all-$params/logs \
    --input output/$corpus/spectrograms_audeep-5-0.025-0.010-240-60.nc \
    --output output/$corpus/audeep-$params.nc
echo "Wrote representations to datasets/$corpus/output/audeep.nc"
python scripts/utils/audeep_corpus_name.py output/$corpus/audeep-$params.nc $corpus
