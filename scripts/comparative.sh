#!/usr/bin/sh

cd ~/src/emotion
source venv/bin/activate
source scripts/env.sh

for corpus in cafe crema-d demos emodb emofilm enterface iemocap jl msp-improv portuguese ravdess savee shemo smartkom tess; do
    for config in IS09_emotion IS13_ComParE eGeMAPS GeMAPS boaw_mfcc_le_20_500 boaw_mfcc_le_50_1000 boaw_mfcc_le_100_5000 boaw_eGeMAPS_20_500 boaw_eGeMAPS_50_1000 boaw_eGeMAPS_100_5000; do
        python scripts/comparative.py --corpus $corpus --clf svm --data $corpus/output/$config.arff --kind linear
        python scripts/comparative.py --corpus $corpus --clf svm --data $corpus/output/$config.arff --kind poly2
        python scripts/comparative.py --corpus $corpus --clf svm --data $corpus/output/$config.arff --kind poly3
        python scripts/comparative.py --corpus $corpus --clf svm --data $corpus/output/$config.arff --kind rbf
        python scripts/comparative.py --corpus $corpus --clf dnn --data $corpus/output/$config.arff --kind basic
        python scripts/comparative.py --corpus $corpus --clf dnn --data $corpus/output/$config.arff --kind 1layer
        python scripts/comparative.py --corpus $corpus --clf dnn --data $corpus/output/$config.arff --kind 2layer
        python scripts/comparative.py --corpus $corpus --clf dnn --data $corpus/output/$config.arff --kind 3layer
    done
done

for corpus in cafe crema-d demos emodb emofilm enterface iemocap jl msp-improv portuguese ravdess savee shemo smartkom tess; do
    python scripts/comparative.py --corpus $corpus --clf svm --data $corpus/output/audeep.nc --kind linear
    python scripts/comparative.py --corpus $corpus --clf svm --data $corpus/output/audeep.nc --kind poly2
    python scripts/comparative.py --corpus $corpus --clf svm --data $corpus/output/audeep.nc --kind poly3
    python scripts/comparative.py --corpus $corpus --clf svm --data $corpus/output/audeep.nc --kind rbf
    python scripts/comparative.py --corpus $corpus --clf dnn --data $corpus/output/audeep.nc --kind basic
    python scripts/comparative.py --corpus $corpus --clf dnn --data $corpus/output/audeep.nc --kind 1layer
    python scripts/comparative.py --corpus $corpus --clf dnn --data $corpus/output/audeep.nc --kind 2layer
    python scripts/comparative.py --corpus $corpus --clf dnn --data $corpus/output/audeep.nc --kind 3layer
done

for corpus in cafe crema-d demos emodb emofilm enterface iemocap jl msp-improv portuguese ravdess savee shemo smartkom tess; do
    python scripts/comparative.py --corpus $corpus --clf cnn --data $corpus/output/logmel.arff --kind aldeneh
done

deactivate
