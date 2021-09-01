#!/bin/sh

# This scripts extracts BoAW features for all corpora from the MFCCs and
# log frame energy features previously extracted with openSMILE.

for corpus in CaFE CREMA-D DEMoS EMO-DB EmoFilm eNTERFACE IEMOCAP JL MSP-IMPROV Portuguese RAVDESS SAVEE ShEMO SmartKom TESS URDU VENEC; do
    python scripts/preprocessing/opensmile.py \
        --config third_party/opensmile/conf/mfcc_log_energy.conf \
        $corpus \
        datasets/$corpus/files.txt \
        features/$corpus/mfcc_log_energy.nc
    python scripts/preprocessing/openxbow.py \
        --closest 20 \
        --codebook 500 \
        features/$corpus/mfcc_log_energy.nc \
        features/$corpus/boaw_20_500.nc
    python scripts/preprocessing/openxbow.py \
        --closest 50 \
        --codebook 1000 \
        features/$corpus/mfcc_log_energy.nc \
        features/$corpus/boaw_50_1000.nc
    python scripts/preprocessing/openxbow.py \
        --closest 100 \
        --codebook 5000 \
        features/$corpus/mfcc_log_energy.nc \
        features/$corpus/boaw_100_5000.nc
done
