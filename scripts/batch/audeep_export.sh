#!/bin/sh

# Exports each dataset using the auDeep model trained on combined spectrograms.
# This script must be run in the audeep docker container.

for corpus in cafe crema-d demos emodb emofilm enterface iemocap jl msp-improv portuguese ravdess savee shemo smartkom tess; do
    echo "Generating $corpus"
    scripts/training/audeep_generate.sh $corpus
    python scripts/utils/audeep_corpus_name.py datasets/${corpus}/output/audeep.nc $corpus
done
