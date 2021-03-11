#!/bin/sh

# Generates spectrograms using the auDeep code.
# This script must be run in the audeep docker container.

for corpus in cafe crema-d demos emodb emofilm enterface iemocap jl msp-improv portuguese ravdess savee shemo smartkom tess urdu venec; do
    echo "Extracting auDeep spectrograms for $corpus"
    audeep preprocess \
        --basedir datasets/$corpus \
        --parser audeep.backend.parsers.filelist.FileListParser \
        --fixed-length 5 \
        --window-width 0.025 \
        --window-overlap 0.010 \
        --mel-spectrum 240 \
        --clip-below -60 \
        --output output/$corpus/spectrograms_audeep-5-0.025-0.010-240-60.nc
    echo "Wrote spectrograms to output/$corpus/spectrograms_audeep-5-0.025-0.010-240-60.nc"
done
