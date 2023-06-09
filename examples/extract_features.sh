#!/bin/bash

for dataset in CREMA-D EMO-DB RAVDESS; do
    ertk-dataset process \
        --processor opensmile \
        --n_jobs -1 \
        --sample_rate 16000 \
        --corpus $dataset \
        $dataset/files_all.txt \
        $dataset/features/eGeMAPS.nc \
        opensmile_config=eGeMAPS
    ertk-dataset process \
        --processor fairseq \
        --sample_rate 16000 \
        --corpus $dataset \
        $dataset/files_all.txt \
        $dataset/features/wav2vec_c_mean.nc \
        model_type=wav2vec \
        checkpoint=/path/to/wav2vec_large.pt \
        layer=context \
        aggregate=MEAN
    ertk-dataset process \
        --processor spectrogram \
        --n_jobs -1 \
        --sample_rate 16000 \
        --corpus $dataset \
        $dataset/files_all.txt \
        $dataset/features/logmel-0.05-0.025-80.nc \
        kind=mel \
        window_size=0.05 \
        window_shift=0.025 \
        n_mels=80 \
        to_log=log
done
