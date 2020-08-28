#!/bin/sh

# Exports spectrograms using auDeep spectrogram extraction.
# This script must be run in the audeep docker container.

for corpus in cafe crema-d demos emodb emofilm enterface iemocap jl msp-improv portuguese ravdess savee shemo smartkom tess; do
    scripts/preprocessing/audeep_spectrograms.sh $corpus
done
