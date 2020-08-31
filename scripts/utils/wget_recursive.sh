#!/bin/sh

# Recursively downloads files from a server. This requires the index page to
# list files and directories.

wget --execute robots=off \
    --reject "index.html*" \
    --timestamping \
    --no-host-directories \
    --level inf \
    --recursive \
    --no-parent \
    --no-verbose \
    $1
