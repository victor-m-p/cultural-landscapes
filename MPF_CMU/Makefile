SHELL=/bin/bash

# Needs GSL and (optionally, for speed) a compiler with OpenMP.
#
# Linux / PSC (run `module load gcc` first):      make
# Mac with MacPorts clang (Apple's clang has no OpenMP):
#     make CC=/opt/local/bin/clang-mp-19
# AMD compiler on PSC (run `module load aocc aocl` first):
#     make CC=clang CFLAGS="-O3 -zopt -fopenmp" LDLIBS="-lalm -lm `gsl-config --libs`"
# Without OpenMP (single-threaded, works with any C compiler):
#     make OPENMP=
# make predefines CC=cc, so `CC ?= gcc` would do nothing; set gcc unless CC was given explicitly
ifeq ($(origin CC),default)
CC = gcc
endif
OPENMP ?= -fopenmp
# plain `=`: a CFLAGS or LDLIBS exported by the environment (conda, HPC modules) must not silently drop the
# OpenMP and GSL flags; to change them, pass them on the make command line as in the examples above
CFLAGS = -O3 -Wall $(OPENMP) `gsl-config --cflags`
LDLIBS = -lm `gsl-config --libs`

OBJ = mpf.o mcmc.o nlin.o nsb.o

.PHONY: all test clean

all: mpf

mpf: main.o $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

%.o: %.c mpf.h
	$(CC) $(CFLAGS) -c $< -o $@

# unit tests (objective against brute-force references, gradients, enumeration, sampling, CV) and
# command-line tests; see tests/README.md
tests/test_mpf: tests/test_mpf.c $(OBJ) mpf.h
	$(CC) $(CFLAGS) -I. tests/test_mpf.c $(OBJ) -o $@ $(LDLIBS)

test: mpf tests/test_mpf
	./tests/test_mpf
	bash tests/test_cli.sh

clean:
	rm -f *.o mpf tests/test_mpf
