SHELL=/bin/bash

# Needs only a C compiler (C99) and the C maths library. OpenMP is used if the compiler supports it
# (cross-validation folds then run in parallel); without it the program is single-threaded and gives
# the same results.
#
#     make                 build ./mpf
#     make test            build and run the unit and command-line tests (about a minute with OpenMP)
#
# Linux:                            make
# Mac, Apple's clang (no OpenMP):   make                     (single-threaded)
# Mac, with OpenMP:                 make CC=/opt/local/bin/clang-mp-19      (MacPorts)   or   make CC=gcc-14   (Homebrew)
# Force a build without OpenMP:     make OPENMP=
# Other flags:                      make CFLAGS="-O2 -g -fopenmp"
#
# make predefines CC=cc, so `CC ?= gcc` would do nothing; use gcc unless CC was given explicitly
ifeq ($(origin CC),default)
CC = gcc
endif
# use -fopenmp if this compiler accepts it, unless OPENMP was given explicitly
ifeq ($(origin OPENMP),undefined)
# (the test program has no preprocessor line because old versions of make read a hash sign here as a comment)
OPENMP := $(shell echo 'int omp_get_max_threads(void); int main(void){return omp_get_max_threads() > 0 ? 0 : 1;}' | $(CC) -fopenmp -x c - -o /dev/null 2>/dev/null && echo -fopenmp)
ifeq ($(OPENMP),)
$(info mpf: $(CC) has no OpenMP support; building the single-threaded program)
endif
endif
# plain `=`: a CFLAGS or LDLIBS exported by the environment (conda, HPC modules) must not silently drop
# flags; to change them, pass them on the make command line as in the examples above
CFLAGS = -O3 -Wall -Wno-unknown-pragmas $(OPENMP)
LDLIBS = -lm

OBJ = mpf.o mcmc.o nlin.o nsb.o rng.o

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
