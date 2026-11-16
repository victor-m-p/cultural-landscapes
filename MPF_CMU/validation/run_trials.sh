#!/bin/bash
# usage: run_trials.sh first_id last_id workers outfile [jobs] [n] [nan]
# Trial t is assigned to coupling regime t mod 4: dispersed, ordered, near-critical, critical.
# Trials already present in outfile are skipped, so an interrupted run can be resumed without duplicates.
# A provenance file (source and binary hashes, environment) is written next to outfile.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}   # threads per fit; the folds of a cross-validation run in parallel
A=$1; B=$2; W=$3; OUT=$4; JOBS=${5:-base,full,naive,partial_legacy,partial_marginal}; N=${6:-20}; NAN=${7:-5}
python3 -c "import common; common.write_provenance('$OUT.provenance.json')"
bounds=("0.01 0.125" "0.125 0.25" "0.25 0.5" "0.5 1.0")
python3 -c "import common; done=common.done_ids('$OUT'); print(' '.join(str(t) for t in range($A, $B+1) if t not in done))" | tr ' ' '\n' | grep -v '^$' | while read t; do echo "$t ${bounds[$((t % 4))]}"; done \
  | xargs -P $W -L 1 sh -c "python3 partial_trial.py \$0 \$1 \$2 $N $NAN $JOBS" >> $OUT
