#!/bin/sh

# Recursively downloads files from a server. This requires the server to
# list files and directories in an HTML page.

[ "$1" = "" ] && echo "Usage: $0 URL" && exit 1

wget --execute robots=off \
    --reject "index.html*" \
    --timestamping \
    --no-host-directories \
    --level inf \
    --recursive \
    --no-parent \
    --no-verbose \
    $1
