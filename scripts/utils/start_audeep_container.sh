#!/bin/sh

# Starts the auDeep docker container. Must be run from the root of the
# repository in order to access the datsets properly.

realuser=${SUDO_USER:-$(whoami)}
uid=`id -u $realuser`
gid=`id -g $realuser`

docker run                                  \
    --gpus device=0                         \
    --rm                                    \
    --interactive                           \
    --tty                                   \
    --user $uid:$gid                        \
    --mount type=bind,src="$PWD",dst=/work \
    --workdir /work                         \
    audeep
