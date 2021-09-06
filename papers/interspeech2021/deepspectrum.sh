#!/usr/bin/env bash

# DeepSpectrum features. This must be run wherever deepspectrum is
# installed.

export TF_CPP_MIN_LOG_LEVEL=1

# Change this as required depending on GPU memory
batch_size=${batch_size:=16}
nmels=128

for corpus in CaFE CREMA-D DEMoS EMO-DB EmoFilm eNTERFACE IEMOCAP JL MSP-IMPROV Portuguese RAVDESS SAVEE ShEMO SmartKom TESS URDU VENEC; do
    for net in densenet201 densenet169 densenet121; do
        echo "Extracting DeepSpectrum for $corpus using $net"
        deepspectrum features \
            --extraction-network $net \
            --feature-layer avg_pool \
            --no-labels \
            --mode mel \
            --frequency-scale mel \
            --number-of-melbands $nmels \
            --batch-size $batch_size \
            datasets/$corpus/resampled \
            --output features/$corpus/$net.csv
    done
done
