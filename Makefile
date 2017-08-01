# include the top-level makefile
include submodules/makd/Makd.mak

# dmxnet is a library, so the default
# build goal is tests only
.DEFAULT_GOAL := test
