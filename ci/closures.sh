#!/bin/bash
set -x -o pipefail

# Enable printing of all potential GC allocation sources to stdout
export DFLAGS=-vgc

# Passes only if `make` succeeds and `grep` successfully finds no
# `closure` (exit status 1); fails if either `make` or `grep` fail
# or if `grep` finds at least one `closure`
beaver dlang make fasttest 2>&1 | (grep -e "closure"; test $? -eq 1)
