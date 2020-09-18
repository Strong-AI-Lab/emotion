#!/usr/bin/sh

# This script runs all combinations of classifier, feature set and corpus.
# Must be run from the root directory of the project.

export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=1

for corpus in cafe crema-d demos emodb emofilm enterface iemocap jl msp-improv portuguese ravdess savee shemo smartkom tess; do
    for features in IS09 IS13 eGeMAPS GeMAPS boaw_20_500 boaw_50_1000 boaw_100_5000 audeep; do
        for kind in linear poly2 poly3 rbf; do
            python scripts/training/comparative.py --clf svm --kind $kind --data output/$corpus/$features.nc
        done
        for kind in basic 1layer 2layer 3layer; do
            python scripts/training/comparative.py --clf dnn --kind $kind --data output/$corpus/$features.nc
        done
    done
    python scripts/training/comparative.py --clf cnn --kind aldeneh --data output/$corpus/logmel.nc --datatype frame
done
