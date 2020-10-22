#!/bin/sh

# Recursively downloads files from a server. This requires the server to
# list files and directories in an HTML page.

[ $# -ne 2 ] && echo "Usage: $0 DEPTH URL" && exit 1

depth=$1
url=$2

wget --execute robots=off \
    --reject "index.html*" \
    --timestamping \
    --no-host-directories \
    --level $depth \
    --recursive \
    --no-parent \
    --no-verbose \
    $url
