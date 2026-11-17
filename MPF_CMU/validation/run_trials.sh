#!/bin/bash
# usage: run_trials.sh first_id last_id workers outfile [jobs] [n] [nan]
# The Table 3 protocol (partial_trial.py) through run_table.sh; see there.
exec "$(dirname "$0")/run_table.sh" partial_trial.py "$1" "$2" "$3" "$4" "${6:-20} ${7:-5} ${5:-base,full,naive,partial_legacy,partial_marginal}"
