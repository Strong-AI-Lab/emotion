#!/bin/sh

# This script runs all combinations of classifier, feature set and corpus.
# Must be run from the root directory of the project.

export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=1

for corpus in cafe crema-d demos emodb emofilm enterface iemocap jl msp-improv portuguese ravdess savee shemo smartkom tess urdu venec; do
    for features in IS09 IS13 eGeMAPS GeMAPS boaw_20_500 boaw_50_1000 boaw_100_5000 audeep-0.05-0.025-240-60_b64_l0.001; do
        for kind in linear poly2 poly3 rbf; do
            python scripts/training/comparative.py --kind svm/$kind --reps 5 --data output/$corpus/$features.nc --results results/comparative2020/$corpus/svm/$kind/$features.csv
        done
        for kind in 1layer 2layer 3layer; do
            python scripts/training/comparative.py --kind mlp/$kind --reps 5 --data output/$corpus/$features.nc --results results/comparative2020/$corpus/mlp/$kind/$features.csv
        done
        python scripts/training/comparative.py --kind rf --reps 5 --data output/$corpus/$features.nc --results results/comparative2020/$corpus/rf/$features.csv
    done
    python scripts/training/comparative.py --kind aldeneh2017 --reps 3 --data output/$corpus/logmel_40.nc --pad 64 --results results/comparative2020/$corpus/aldeneh2017/logmel_40.csv
    python scripts/training/comparative.py --kind aldeneh2017 --reps 3 --data output/$corpus/logmel_240.nc --pad 64 --results results/comparative2020/$corpus/aldeneh2017/logmel_240.csv
    python scripts/training/comparative.py --kind aldeneh2017 --reps 3 --data output/$corpus/spectrograms-0.025-0.010-240-60.nc --pad 64 --results results/comparative2020/$corpus/aldeneh2017/spectrograms_240.csv

    python scripts/training/comparative.py --kind latif2019 --reps 3 --data output/$corpus/raw_audio.nc --clip 80000 --results results/comparative2020/$corpus/latif2019/raw_audio.csv

    python scripts/training/comparative.py --kind zhang2019 --reps 1 --data output/$corpus/raw_audio.nc --clip 80000 --results results/comparative2020/$corpus/zhang2019/raw_audio.csv

    python scripts/training/comparative.py --kind zhao2019 --reps 3 --data output/$corpus/spectrograms-0.025-0.010-40-60.nc --pad 512 --results results/comparative2020/$corpus/zhao2019/spectrograms_40.csv
done
