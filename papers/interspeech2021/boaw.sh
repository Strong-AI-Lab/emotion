#!/usr/bin/env bash

# This scripts extracts BoAW features for all corpora from the MFCCs and
# log frame energy features previously extracted with openSMILE.

for corpus in CaFE CREMA-D DEMoS EMO-DB EmoFilm eNTERFACE IEMOCAP JL MSP-IMPROV Portuguese RAVDESS SAVEE ShEMO SmartKom TESS URDU VENEC; do
    python scripts/preprocessing/opensmile.py \
        --config third_party/opensmile/conf/mfcc_log_energy.conf \
        $corpus \
        datasets/$corpus/files_all.txt \
        features/$corpus/mfcc_log_energy.nc || { exit 1; }
    python scripts/preprocessing/openxbow.py \
        features/$corpus/mfcc_log_energy.nc \
        features/$corpus/boaw_20_500.nc \
        -- \
        -a 20 \
        -size 500 \
        -norm 1
    python scripts/preprocessing/openxbow.py \
        features/$corpus/mfcc_log_energy.nc \
        features/$corpus/boaw_50_1000.nc \
        -- \
        -a 50 \
        -size 1000 \
        -norm 1
    python scripts/preprocessing/openxbow.py \
        features/$corpus/mfcc_log_energy.nc \
        features/$corpus/boaw_100_5000.nc \
        -- \
        -a 100 \
        -size 5000 \
        -norm 1
done
