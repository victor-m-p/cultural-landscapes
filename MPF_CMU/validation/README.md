# Validation experiments

The experiments behind `CLAUDE_IMPROVEMENTS_SUMMARY.md` (at the top of the repository). All of them use
`../mpf` (or the binary named in `MPF_BIN`), so run `make` in `MPF_CMU` first. The Python scripts need
numpy (scipy for `population_consistency.py` and `drh_missingness.py`; scikit-learn for
`clustering_sensitivity.py`). Outputs are in `results/`.

Every simulation result file `X.jsonl` has a companion `X.jsonl.provenance.json` recording the SHA-256 of
each C source file and of the binary, the git commit, and the Python environment, and every trial carries a
short fingerprint of the sources (`source`). The runners skip trial ids that are already in the output file,
so an interrupted run can be resumed, and the summarisers refuse files with duplicate trial ids.

## Missing data

| script | what it does | output |
|---|---|---|
| `population_consistency.py [n] [n_blank]` | Infinite-data test, by exact enumeration and independent of the C code, of whether each treatment of missing data recovers the true parameters when entries are missing completely at random. | `results/population_n6.txt` |
| `large_sample.py beta M seed [mar] [nnK]` | n=20, 2M rows: KL from the truth for complete-rows-only, legacy, marginal and frozen-weight fits, each at its best lambda on a grid; with entries missing completely at random, or (`mar`) depending on observed answers; single flips or (`nn2`) larger neighbourhoods. | `results/large_sample.txt`, `results/large_sample_nn2.txt` |
| `partial_oracle_trial.py` | The same data as the Table 3 protocol, but every method scored at its best lambda on a grid, without cross-validation. | `results/table3_oracle.jsonl` |
| `partial_trial.py`, `run_trials.sh`, `summarize_table3.py` | The paper's Table 3 protocol (128 complete; 128 + 128 partial under each treatment; naive imputation; 256 complete; lambda by cross-validation) with exact sampling. One result file per group of columns. `summarize_trials.py FILE` summarises a single file. | `results/table3_complete_columns.jsonl`, `table3_legacy.jsonl`, `table3_marginal.jsonl`, `table3_frozen.jsonl`, `table3_summary.txt` |
| `drh_missingness.py` | Tests whether the blanks in the curated DRH data are related to the answers (they are). | `results/drh_missingness.txt` |
| `drh_bootstrap.py` | Parametric bootstrap at the size of the DRH data with its real patterns of blanks, assigned at random or according to an observed answer: KL from the truth for each treatment. | `results/drh_bootstrap.jsonl` |

## Regularisation, cross-validation, neighbourhoods, sampler

| script | what it does | output |
|---|---|---|
| `table1_trial.py`, `table2_trial.py`, `run_table.sh`, `summarize_tables.py` | The paper's Table 1 (optimal vs cross-validated lambda vs none) and Table 2 (re-weighting) protocols, with the published Monte Carlo sampler and with exact sampling; Table 1 also on the same landscapes with one-, two- and three-flip neighbourhoods. | `results/table1_mcmc.jsonl`, `table1_exact.jsonl`, `table1_nn1.jsonl`, `table1_nn2.jsonl`, `table1_nn3.jsonl`, `table2_exact.jsonl`, `tables_summary.txt` |
| `check_sampler.py beta n_samples seed` | Compares samples from `mpf -g` (Monte Carlo and exact) with the exact distribution. | `results/sampler.txt` |
| `repo_logs_table.py` | Recomputes the paper's Table 3 from the logs committed in `../NAN_TESTS`. | `results/repo_logs_table.txt` |

## The DRH fit

| script | what it does | output |
|---|---|---|
| `results/drh_refits/refit.sh [name ...]` | Refits the DRH data from the committed input: published procedure (`legacy`), `marginal`, `marginal_entrycv`, frozen-weight with entry-wise cross-validation at NN = 1, 2, 3 (`frozen_entrycv`, `frozen_entrycv_nn2`, `frozen_entrycv_nn3`), and `legacy_nn2`. | `results/drh_refits/` (parameters, logs, stderr, provenance; `entry_groups.txt` is the `--groups` file) |
| `drh_compare.py` | Compares each refit with the parameter file used in the paper. The table, including each lambda, is generated from the logs. | `results/drh_compare.txt` |
| `clustering_sensitivity.py` | Stability of the three groups of Figure 4A to the number of configurations clustered and to the fit. | `results/clustering_sensitivity.txt` |

The analysis regenerated from these fits is in `regenerated/` at the top of the repository.

## Checks on the code itself

| script | what it does | output |
|---|---|---|
| `compare_with_published_code.py` | Builds the published C code from git (commit 23dce37, or the directory named in `MPF_PUBLISHED_SRC`) and compares it with the rewritten program in `--legacy` mode on random inputs, on the DRH input, and — with the published two- and three-flip blocks re-enabled — at NN = 2 and 3. | `results/compare_with_published_code.txt` |
| `mutation_test.py` | Injects deliberate bugs into a temporary copy of the sources, one at a time, and checks that `make test` fails. Takes over an hour. | `results/mutation_test.txt` |

## Notes on the stored results

* The simulation tables were run on a 256-core Linux server (`results/wright_driver.log` is the driver's
  log); `results/local_partial/table1_nn1_local.jsonl` is a partial run of the NN=1 column of Table 1 on a
  Mac, kept as a cross-platform check (KL agrees to 2e-4).
* `results/large_sample.txt` (NN=1) was produced before neighbourhoods larger than one flip were
  implemented; the NN=1 code path it exercises was not changed by that work.
* Runs differ in the `source` fingerprint they carry because the C sources changed between them (the
  NN > 1 work, then a final code review). The last review changed one thing that can alter a number: how
  `--legacy` cross-validation starts its folds when rows have missing answers (it now matches the published
  code). `results/table3_legacy.jsonl` was therefore re-run after that fix; the fixed-lambda and
  complete-data runs are unaffected, and the DRH fits and the comparison with the published code were
  redone after it.
* The Table 1 and Table 2 trials call `mpf -o` and `mpf -c` on complete data with `--legacy`, i.e. the
  published protocol; for complete data the only difference from the default is that the search for lambda
  stays within [-2, 4].
