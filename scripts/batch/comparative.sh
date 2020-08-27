#!/usr/bin/sh

# This script runs all combinations of classifier, feature set and corpus.

for corpus in cafe crema-d demos emodb emofilm enterface iemocap jl msp-improv portuguese ravdess savee shemo smartkom tess; do
    for features in IS09_emotion IS13_ComParE eGeMAPS GeMAPS boaw_mfcc_le_20_500 boaw_mfcc_le_50_1000 boaw_mfcc_le_100_5000 boaw_eGeMAPS_20_500 boaw_eGeMAPS_50_1000 boaw_eGeMAPS_100_5000; do
        python scripts/training/comparative.py --corpus $corpus --clf svm --kind linear --data $corpus/output/$features.arff
        python scripts/training/comparative.py --corpus $corpus --clf svm --kind poly2  --data $corpus/output/$features.arff
        python scripts/training/comparative.py --corpus $corpus --clf svm --kind poly3  --data $corpus/output/$features.arff
        python scripts/training/comparative.py --corpus $corpus --clf svm --kind rbf    --data $corpus/output/$features.arff
        python scripts/training/comparative.py --corpus $corpus --clf dnn --kind basic  --data $corpus/output/$features.arff
        python scripts/training/comparative.py --corpus $corpus --clf dnn --kind 1layer --data $corpus/output/$features.arff
        python scripts/training/comparative.py --corpus $corpus --clf dnn --kind 2layer --data $corpus/output/$features.arff
        python scripts/training/comparative.py --corpus $corpus --clf dnn --kind 3layer --data $corpus/output/$features.arff
    done
done

for corpus in cafe crema-d demos emodb emofilm enterface iemocap jl msp-improv portuguese ravdess savee shemo smartkom tess; do
    python scripts/training/comparative.py --corpus $corpus --clf svm --kind linear --data $corpus/output/audeep.nc
    python scripts/training/comparative.py --corpus $corpus --clf svm --kind poly2  --data $corpus/output/audeep.nc
    python scripts/training/comparative.py --corpus $corpus --clf svm --kind poly3  --data $corpus/output/audeep.nc
    python scripts/training/comparative.py --corpus $corpus --clf svm --kind rbf    --data $corpus/output/audeep.nc
    python scripts/training/comparative.py --corpus $corpus --clf dnn --kind basic  --data $corpus/output/audeep.nc
    python scripts/training/comparative.py --corpus $corpus --clf dnn --kind 1layer --data $corpus/output/audeep.nc
    python scripts/training/comparative.py --corpus $corpus --clf dnn --kind 2layer --data $corpus/output/audeep.nc
    python scripts/training/comparative.py --corpus $corpus --clf dnn --kind 3layer --data $corpus/output/audeep.nc
done

for corpus in cafe crema-d demos emodb emofilm enterface iemocap jl msp-improv portuguese ravdess savee shemo smartkom tess; do
    python scripts/training/comparative.py --corpus $corpus --clf cnn --data $corpus/output/logmel.arff --kind aldeneh
done
