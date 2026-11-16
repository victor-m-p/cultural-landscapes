#!/bin/bash
# usage: run_table.sh script.py first_id last_id workers outfile [extra args]
# Trials already present in outfile are skipped; a provenance file is written next to outfile.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}   # threads per fit; the folds of a cross-validation run in parallel
S=$1; A=$2; B=$3; W=$4; OUT=$5; EXTRA=${6:-}
python3 -c "import common; common.write_provenance('$OUT.provenance.json')"
bounds=("0.01 0.125" "0.125 0.25" "0.25 0.5" "0.5 1.0")
python3 -c "import common; done=common.done_ids('$OUT'); print(' '.join(str(t) for t in range($A, $B+1) if t not in done))" | tr ' ' '\n' | grep -v '^$' | while read t; do echo "$t ${bounds[$((t % 4))]}"; done \
  | xargs -P $W -L 1 sh -c "python3 $S \$0 \$1 \$2 $EXTRA" >> $OUT
