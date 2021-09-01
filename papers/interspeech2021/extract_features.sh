#!/bin/sh

# This script extracts all features used in our INTERSPEECH 2021 paper.
# Note that the audeep.sh script should be run from the Docker container
# (or other appropriate environment) initially to train the audeep
# model.

SCRIPT_DIR=`dirname $0`

$SCRIPT_DIR/acoustic.sh
$SCRIPT_DIR/boaw.sh
$SCRIPT_DIR/embeddings.sh
