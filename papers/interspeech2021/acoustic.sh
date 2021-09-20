#!/usr/bin/env bash

# This script extracts all acoustic feature sets for all corpora.

for corpus in CaFE CREMA-D DEMoS EMO-DB EmoFilm eNTERFACE IEMOCAP JL MSP-IMPROV Portuguese RAVDESS SAVEE ShEMO SmartKom TESS URDU VENEC; do
    for features in IS09 IS13 eGeMAPS GeMAPS; do
        python scripts/preprocessing/opensmile.py \
            --config third_party/opensmile/conf/$features.conf \
            $corpus \
            datasets/$corpus/files_all.txt \
            features/$corpus/$features.nc
    done
    python scripts/preprocessing/opensmile.py \
        --config third_party/opensmile/conf/mean_mfcc.conf \
        $corpus \
        datasets/$corpus/files_all.txt \
        features/$corpus/mean_mfcc_64.nc \
        -- \
        -nMfcc 64
done
