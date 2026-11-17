#!/bin/bash
# Refit the DRH data in several ways. Run from anywhere after `make` in MPF_CMU/src. Inputs are copied from
# the committed data file, so this works in a clean checkout; nothing outside this directory is written.
#   legacy            the published procedure (reproduces data/mdl_experiments/..._params.dat)
#   marginal          marginal objective, weighted row-wise cross-validation (the default)
#   marginal_entrycv  marginal objective, leave-one-ENTRY-out cross-validation
#   frozen_entrycv    frozen-weight iteration, leave-one-ENTRY-out cross-validation
#   frozen_entrycv_nn2, frozen_entrycv_nn3   the same with two- and three-flip neighbourhoods (NN=2, 3)
#   legacy_nn2        the published procedure with NN=2
set -euo pipefail
cd "$(dirname "$0")"
MPF=../../../src/mpf
SRC=../../../../data/mdl_experiments/matrix_questions_20_maxna_5_nrows_455_entries_407.txt.mpf
REF=../../../../data/reference/direct_reference_questions_20_maxna_5_nrows_455_entries_407.csv
# one entry id per data row (the reference file lists the rows in the same order as the data file)
tail -n +2 "$REF" | cut -d, -f1 > entry_groups.txt
[ "$(wc -l < entry_groups.txt | tr -d ' ')" = "$(head -1 "$SRC" | tr -d ' ')" ] || { echo "group file and data file disagree on the number of rows" >&2; exit 1; }
python3 -c "import sys; sys.path.insert(0, '../..'); import common; common.write_provenance('provenance.json')"
fit() { # name, NN, flags...
	local name=$1 nn=$2; shift 2
	cp "$SRC" "drh_$name.mpf"
	"$MPF" -c "drh_$name.mpf" "$nn" "$@" > "drh_$name.log" 2> "drh_$name.err"
	echo "$name: $(grep 'Best log_sparsity' "drh_$name.log")  $(grep 'Wall time' "drh_$name.log")  warnings: $(grep -c warning "drh_$name.err" || true)"
}
# usage: refit.sh [name ...]   (no names: all of them)
want() { [ $# -eq 0 ] && return 0; local n=$1; shift; for x in "$@"; do [ "$x" = "$n" ] && return 0; done; return 1; }
run() { local name=$1; shift; if [ ${#ONLY[@]} -eq 0 ] || want "$name" "${ONLY[@]}"; then fit "$name" "$@"; fi; }
ONLY=("$@")
run legacy 1 --legacy
run marginal 1
run marginal_entrycv 1 --groups entry_groups.txt
# the frozen-weight fits with whole-entry cross-validation, for single-, two- and three-flip neighbourhoods
run frozen_entrycv 1 --frozen --groups entry_groups.txt
run frozen_entrycv_nn2 2 --frozen --groups entry_groups.txt
run frozen_entrycv_nn3 3 --frozen --groups entry_groups.txt
# the published procedure with two-flip neighbours
run legacy_nn2 2 --legacy
