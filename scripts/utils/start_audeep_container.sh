#!/bin/sh

# Starts the auDeep docker container. Must be run from the root of the
# repository in order to access the datsets properly.

docker run                                 \
    --gpus device=1                        \
    --rm                                   \
    --interactive                          \
    --tty                                  \
    --user $(id -u):$(id -g)               \
    --mount type=bind,src=$(pwd),dst=/work \
    --workdir /work                        \
    audeep
