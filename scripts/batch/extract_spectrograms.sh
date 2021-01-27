#!/bin/sh

for corpus in cafe crema-d demos emodb emofilm enterface iemocap jl msp-improv portuguese ravdess savee shemo smartkom tess urdu venec; do
    echo "Extracting spectrograms for $corpus"
    python scripts/preprocessing/extract_spectrograms.py \
        datasets/$corpus/files.txt \
        --corpus $corpus \
        --labels datasets/$corpus/labels.csv \
        --audeep output/$corpus/spectrograms-0.025-0.010-240-60.nc \
        --length 5 \
        --skip 0 \
        --clip 60 \
        --window_size 0.05 \
        --window_shift 0.025 \
        --mel_bands 240 \
        --pre_emphasis 0.95 \
        --channels mean

    python scripts/preprocessing/extract_spectrograms.py \
        datasets/$corpus/files.txt \
        --corpus $corpus \
        --labels datasets/$corpus/labels.csv \
        --audeep output/$corpus/spectrograms-0.025-0.010-40-60.nc \
        --length 5 \
        --skip 0 \
        --clip 60 \
        --window_size 0.05 \
        --window_shift 0.025 \
        --mel_bands 40 \
        --pre_emphasis 0.95 \
        --channels mean
done
