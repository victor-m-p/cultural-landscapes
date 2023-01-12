#!/bin/bash
END=261
for i in `seq 1 20 261`; do
	julia -t 8 compute_evolution.jl $i
done
