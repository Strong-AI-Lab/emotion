#!/bin/sh

# Generates spectrograms using the auDeep code.
# This script must be run in the audeep docker container.

corpus=$1

output=output/${corpus}/spectrograms_audeep-5-0.05-0.025-240-60.nc

audeep preprocess \
    --basedir datasets/$corpus/wav_corpus \
    --parser audeep.backend.parsers.no_metadata.NoMetadataParser \
    --window-width 0.05 \
    --window-overlap 0.025 \
    --mel-spectrum 240 \
    --fixed-length 5 \
    --clip-below -60 \
    --output $output
echo "Wrote spectrograms to $output"
