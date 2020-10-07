#!/bin/sh

# Acts like the mv equivalent of cp --parents -t DIR

set -e

[ $# -lt 2 ] && echo "Usage: $0 DIR FILE [FILE ...]" && exit 1

DIR=$1
shift

mkdir --parents "$DIR"
for x in "$@"; do
    dir=$(dirname "$x")
    mv -t "$DIR/$dir" "$x"
done
