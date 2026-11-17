#!/bin/bash
# usage: run_table.sh script.py first_id last_id workers outfile [extra args]
# Runs `python3 script.py ID BETA_LO BETA_HI [extra args]` for every trial id in the range that is not already
# in outfile (so an interrupted run can be resumed), `workers` at a time, appending one JSON line per trial.
# Trial t is in coupling regime t mod 4, which fixes its range of beta (common.REGIME_BOUNDS). Each trial
# records the fingerprint of the sources, binary and scripts that produced it, and the manifest with that
# fingerprint is written next to outfile (common.emit); manifests are never overwritten.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}   # threads per fit; the folds of a cross-validation run in parallel
S=$1; A=$2; B=$3; W=$4; OUT=$5; EXTRA=${6:-}
export MPF_PROVENANCE="$(pwd)/$OUT.provenance.json"; case "$OUT" in /*) MPF_PROVENANCE="$OUT.provenance.json";; esac
python3 -c "
import common
done = common.done_ids('$OUT')
for t in range($A, $B + 1):
    if t not in done: print(t, *common.REGIME_BOUNDS[t % 4])" | xargs -P $W -L 1 sh -c "python3 $S \$0 \$1 \$2 $EXTRA" >> $OUT
