#!/bin/sh

# Generates spectrograms using the auDeep code.
# This script must be run in the audeep docker container.

corpus="$1"

output=datasets/spectrograms/audeep/${corpus}-0.05-0.025-120-60.nc

audeep preprocess \
    --basedir $corpus/wav_corpus \
    --parser audeep.backend.parsers.no_metadata.NoMetadataParser \
    --window-width 0.05 \
    --window-overlap 0.025 \
    --mel-spectrum 120 \
    --fixed-length 5 \
    --clip-below -60 \
    --output $output 2>/dev/null
echo "Wrote spectrograms to $output"
