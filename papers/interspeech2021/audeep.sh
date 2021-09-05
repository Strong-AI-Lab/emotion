#!/usr/bin/env bash

# Generates spectrograms using the auDeep code.

# Change batch_size as necessary for your GPU memory
batch_size=${batch_size:=128}

for corpus in CaFE CREMA-D DEMoS EMO-DB EmoFilm eNTERFACE IEMOCAP JL MSP-IMPROV Portuguese RAVDESS SAVEE ShEMO SmartKom TESS URDU VENEC; do
    echo "Extracting auDeep spectrograms for $corpus"
    audeep preprocess \
        --basedir datasets/$corpus/resampled \
        --parser audeep.backend.parsers.no_metadata.NoMetadataParser \
        --fixed-length 5 \
        --window-width 0.025 \
        --window-overlap 0.015 \
        --mel-spectrum 240 \
        --clip-below -60 \
        --output features/$corpus/spectrograms_audeep-5sec-0.025-0.010-240-60.nc
done

echo "Training auDeep model"
audeep t-rae train \
    --batch-size 128 \
    --num-epochs 250 \
    --run-name logs/audeep \
    --checkpoints-to-keep 5 \
    --num-layers 2 \
    --num-units 256 \
    --bidirectional-decoder \
    --keep-prob 0.8 \
    --input features/{CaFE,CREMA-D,DEMoS,EMO-DB,EmoFilm,eNTERFACE,IEMOCAP,JL,MSP-IMPROV,Portuguese,RAVDESS,SAVEE,ShEMO,SmartKom,TESS,URDU,VENEC}/spectrograms_audeep-5sec-0.025-0.010-240-60.nc

for corpus in CaFE CREMA-D DEMoS EMO-DB EmoFilm eNTERFACE IEMOCAP JL MSP-IMPROV Portuguese RAVDESS SAVEE ShEMO SmartKom TESS URDU VENEC; do
    echo "Generating auDeep features for $corpus"
    audeep t-rae generate \
        --model-dir logs/audeep/2x256_u_b/all-5-0.025-0.010-240-60_b128_l0.001/logs \
        --batch-size $batch_size \
        --input features/$corpus/spectrograms_audeep-5sec-0.025-0.010-240-60.nc \
        --output features/$corpus/audeep.nc
done

echo "Note that the auDeep features still need to be converted to our " \
     "dataset format to be of use."
