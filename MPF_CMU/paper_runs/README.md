# The original runs for the paper (2022–23)

The scripts, logs and scratch files from the work behind "Inferring Cultural Landscapes with the Inverse
Ising Model", as they were. They are kept for the record; nothing here is needed to use the program, which
is in [`../src`](../src).

| what | files |
|---|---|
| Simulation scripts behind the paper's tables | `new_cv_scan.rb` (Table 1), `bias_correction.rb` (Table 2), `big_test_nans.rb` (Table 3); `bias_tables.rb`, `nan_table.rb`, `analyze_cv_tests.rb` summarise their logs |
| Their logs | `BIAS_TESTS/`, `NAN_TESTS/`; simulated datasets, parameters and cross-validation scans in `DATA/` |
| Scripts and logs for the DRH fits | `final_process.rb`, `cleaned_fits.rb`, `holdout_fits.rb`, `process.rb`, `best_fit_DRH*.rb`, `cv_data.rb`; `UPDATED_FITS`, `CLEANED_FITS*`, `HOLDOUT_FITS*`, `TEST_ONE*`, `TEST_TWO*` |
| Other scripts, plots and scratch files | the remaining `*.rb`, `*.pro`, `*.sh`, `PLOTS/`, `*.pdf`, `*.gephi`, `temp*`, `test*.txt` |
| Old C files that are not part of the program | `junk.c`, `older_mcmc.c`, `prob.h` |

## Running the scripts today

The scripts call `./mpf` and use relative paths (`DATA/`, `NAN_TESTS/`, ...), so they expect to be run from
this directory with the program linked here:

```
(cd ../src && make) && ln -s ../src/mpf mpf
```

They were written for the published version of `mpf`. Run against the current one they use its new
defaults: the marginal objective for missing data, cross-validation that builds each fold from its own
training rows and weights held-out rows, exact sampling in `-g` for up to 24 nodes, and real multi-flip
neighbours for NN > 1 (the published code ran NN=1 whatever NN was given). To repeat the published
procedure, add `--legacy` to their `-c` / `-o` / `-t` / `-l` calls and `--mcmc` to their `-g` calls, and use
NN=1. Each script that calls `mpf` carries a note saying so.

Three scripts (`best_fit_DRH_2.rb`, `cv_scan.rb`, `param_scan.rb`) also call `-c` and `-t` with the argument
order of an earlier version of the program and will not run as they stand. Several scripts contain
absolute paths from the machines they were written on.

Re-runs of the three simulation tables with the current program are in
[`../validation`](../validation) (`table1_trial.py`, `table2_trial.py`, `partial_trial.py`).
