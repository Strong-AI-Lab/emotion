#!/bin/sh

# This script runs all tests on each corpus.
# It must be run from the root directory of the repository.


# Ignore 'INFO' level messages as TF tends to be quite verbose.
export TF_CPP_MIN_LOG_LEVEL=1

for corpus in cafe crema-d demos emodb emofilm enterface iemocap jl msp-improv portuguese ravdess savee shemo smartkom tess; do
    ./run_preprocessing.sh $1
    ./run_experiments $1
done
