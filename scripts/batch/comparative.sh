#!/usr/bin/sh

# This script runs all combinations of classifier, feature set and corpus.

for corpus in cafe crema-d demos emodb emofilm enterface iemocap jl msp-improv portuguese ravdess savee shemo smartkom tess; do
    for features in IS09_emotion IS13_ComParE eGeMAPS GeMAPS boaw_mfcc_le_20_500 boaw_mfcc_le_50_1000 boaw_mfcc_le_100_5000 boaw_eGeMAPS_20_500 boaw_eGeMAPS_50_1000 boaw_eGeMAPS_100_5000; do
        python scripts/training/comparative.py --corpus $corpus --clf svm --kind linear --data datasets/$corpus/output/$features.arff --datatype utterance
        python scripts/training/comparative.py --corpus $corpus --clf svm --kind poly2  --data datasets/$corpus/output/$features.arff --datatype utterance
        python scripts/training/comparative.py --corpus $corpus --clf svm --kind poly3  --data datasets/$corpus/output/$features.arff --datatype utterance
        python scripts/training/comparative.py --corpus $corpus --clf svm --kind rbf    --data datasets/$corpus/output/$features.arff --datatype utterance
        python scripts/training/comparative.py --corpus $corpus --clf dnn --kind basic  --data datasets/$corpus/output/$features.arff --datatype utterance
        python scripts/training/comparative.py --corpus $corpus --clf dnn --kind 1layer --data datasets/$corpus/output/$features.arff --datatype utterance
        python scripts/training/comparative.py --corpus $corpus --clf dnn --kind 2layer --data datasets/$corpus/output/$features.arff --datatype utterance
        python scripts/training/comparative.py --corpus $corpus --clf dnn --kind 3layer --data datasets/$corpus/output/$features.arff --datatype utterance
    done
done

for corpus in cafe crema-d demos emodb emofilm enterface iemocap jl msp-improv portuguese ravdess savee shemo smartkom tess; do
    python scripts/training/comparative.py --corpus $corpus --clf svm --kind linear --data datasets/$corpus/output/audeep.nc --datatype netCDF
    python scripts/training/comparative.py --corpus $corpus --clf svm --kind poly2  --data datasets/$corpus/output/audeep.nc --datatype netCDF
    python scripts/training/comparative.py --corpus $corpus --clf svm --kind poly3  --data datasets/$corpus/output/audeep.nc --datatype netCDF
    python scripts/training/comparative.py --corpus $corpus --clf svm --kind rbf    --data datasets/$corpus/output/audeep.nc --datatype netCDF
    python scripts/training/comparative.py --corpus $corpus --clf dnn --kind basic  --data datasets/$corpus/output/audeep.nc --datatype netCDF
    python scripts/training/comparative.py --corpus $corpus --clf dnn --kind 1layer --data datasets/$corpus/output/audeep.nc --datatype netCDF
    python scripts/training/comparative.py --corpus $corpus --clf dnn --kind 2layer --data datasets/$corpus/output/audeep.nc --datatype netCDF
    python scripts/training/comparative.py --corpus $corpus --clf dnn --kind 3layer --data datasets/$corpus/output/audeep.nc --datatype netCDF
done

for corpus in cafe crema-d demos emodb emofilm enterface iemocap jl msp-improv portuguese ravdess savee shemo smartkom tess; do
    python scripts/training/comparative.py --corpus $corpus --clf cnn --kind aldeneh --data $corpus/output/logmel.arff --datatype frame
done
