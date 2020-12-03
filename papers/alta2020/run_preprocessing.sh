#!/usr/bin/sh

# This script runs all the data preprocessing scripts (feature
# generation) on a given corpus.


# IS09 features
python scripts/preprocessing/opensmile.py --corpus $1 --config third_party/opensmile/conf/IS09.conf --annotations datasets/$1/labels.csv --input datasets/$1/files.txt --output output/$1/IS09.nc

# Variable length unclipped log-mel spectrograms
python scripts/preprocessing/opensmile.py --corpus $1 --config third_party/opensmile/conf/logmel.conf --annotations datasets/$1/labels.csv --input datasets/$1/files.txt --output output/$1/logmel_40.nc -nMelBands 40
python scripts/preprocessing/opensmile.py --corpus $1 --config third_party/opensmile/conf/logmel.conf --annotations datasets/$1/labels.csv --input datasets/$1/files.txt --output output/$1/logmel_240.nc -nMelBands 240

# Fixed-size clipped log-mel spectrograms
python scripts/preprocessing/extract_spectrograms.py --corpus $1 --length 5 --clip 60 --window_size 0.025 --window_shift 0.010 --pre_emphasis 0.97 --labels datasets/$1/labels.csv --input datasets/$1/files.txt --netcdf output/$1/spectrograms-0.025-0.010-40-60.nc --mel_bands 40
python scripts/preprocessing/extract_spectrograms.py --corpus $1 --length 5 --clip 60 --window_size 0.025 --window_shift 0.010 --pre_emphasis 0.97 --labels datasets/$1/labels.csv --input datasets/$1/files.txt --netcdf output/$1/spectrograms-0.025-0.010-240-60.nc --mel_bands 240
