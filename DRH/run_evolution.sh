#!/bin/bash
END=261
for sim in `seq 1 10 1`; do
	for slice in `seq 1 100 261`; do
		julia -t 35 compute_evolution.jl $sim $slice
	done 
done
