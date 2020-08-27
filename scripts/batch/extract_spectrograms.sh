#!/bin/sh

for corpus in cafe crema-d demos emodb emofilm enterface iemocap jl msp-improv portuguese ravdess savee shemo smartkom tess; do
    echo "Extracting spectrograms for $corpus"
    TF_CPP_MIN_LOG_LEVEL=1 python scripts/preprocessing/extract_spectrograms.py \
        --corpus $corpus \
        --labels datasets/$corpus/labels.csv \
        --batch_size 128 \
        --audeep datasets/$corpus/spectrograms-0.05-0.025-120-60.nc \
        --length 5 \
        --skip 0 \
        --clip 60 \
        --window_size 0.05 \
        --window_shift 0.025 \
        --mel_bands 120 \
        --pre_emphasis 0.95 \
        --channels mean \
        datasets/$corpus/files.txt
done
