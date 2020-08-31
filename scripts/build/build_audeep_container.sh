#!/bin/sh

# Builds the auDeep docker image using the code from the auDeep repository.

docker build                 \
    --file Dockerfile.audeep \
    --tag audeep             \
    third_party/audeep
