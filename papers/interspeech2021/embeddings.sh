#!/bin/sh

# This script extracts neural embeddings for all corpora


# Change this for DeepSpectrum as required depending on GPU memory
ds_batch_size=16

# Change this for VGGish/YAMNet as required depending on GPU memory
as_batch_size=256

# Change to path to wav2vec models dir
wav2vec_models_dir=$HOME/src/fairseq/examples/wav2vec/

nmels=128

for corpus in CaFE CREMA-D DEMoS EMO-DB EmoFilm eNTERFACE IEMOCAP JL MSP-IMPROV Portuguese RAVDESS SAVEE ShEMO SmartKom TESS URDU VENEC; do
    # auDeep dataset conversion
    python scripts/utils/convert_audeep_dataset.py \
        output/$corpus/audeep.nc \
        $corpus

    # DeepSpectrum features
    for net in densenet201 densenet169 densenet121; do
        echo "Extracting for $corpus using $net"

        deepspectrum features \
            --extraction-network $net \
            --feature-layer avg_pool \
            --no-labels \
            --mode mel \
            --frequency-scale mel \
            --number-of-melbands $nmels \
            --batch-size $ds_batch_size \
            datasets/$corpus/wav \
            --output features/$corpus/$net.csv
        python scripts/utils/convert.py \
            features/$corpus/$net.csv \
            features/$corpus/$net.nc \
            --corpus $corpus
        rm features/$corpus/$net.csv
    done

    # Spectrograms
    python scripts/preprocessing/spectrograms.py \
        datasets/$corpus/files.txt \
        --corpus $corpus \
        --labels datasets/$corpus/labels.csv \
        --window_size 0.025 \
        --window_shift 0.01 \
        --mel_bands 64 \
        --fmin 125 \
        --fmax 7500 \
        --netcdf features/$corpus/spectrograms-0.025-0.010-64.nc

    # YAMNet and VGGish mean embeddings
    python scripts/preprocessing/yamnet_features.py \
        features/$corpus/spectrograms-0.025-0.010-64.nc \
        features/$corpus/yamnet.nc \
        --batch_size $as_batch_size
    python scripts/preprocessing/vggish_features.py \
        features/$corpus/spectrograms-0.025-0.010-64.nc \
        features/$corpus/vggish.nc \
        --batch_size $as_batch_size

    # (VQ-)Wav2Vec mean embeddings
    python scripts/preprocessing/wav2vec_features.py \
        --checkpoint $wav2vec_models_dir/wav2vec_large.pt \
        datasets/$corpus/files.txt \
        features/$corpus/wav2vec.nc
    python scripts/preprocessing/wav2vec_features.py \
        --checkpoint $wav2vec_models_dir/vq-wav2vec.pt \
        datasets/$corpus/files.txt \
        features/$corpus/vq-wav2vec.nc
    # Wav2Vec 2.0
    python scripts/preprocessing/wav2vec_features.py \
        --checkpoint $wav2vec_models_dir/libri960_big.pt \
        --type 2 \
        datasets/$corpus/files.txt \
        features/$corpus/wav2vec2.nc
done
