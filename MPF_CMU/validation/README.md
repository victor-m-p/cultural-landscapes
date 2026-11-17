# Validation experiments

The experiments behind `CLAUDE_IMPROVEMENTS_SUMMARY.md` (at the top of the repository). All of them use
`../src/mpf` (or the binary named in `MPF_BIN`), so run `make` in `../src` first. The Python scripts need
numpy (scipy for `population_consistency.py` and `drh_missingness.py`; scikit-learn for
`clustering_sensitivity.py`). Outputs are in `results/`.

## How the simulations are run

Every trial script takes a trial number and prints one JSON line. One runner drives them all:

```
./run_table.sh SCRIPT FIRST LAST WORKERS OUTFILE ["EXTRA ARGS"]
```

It runs `python3 SCRIPT ID BETA_LO BETA_HI EXTRA ARGS` for every id in the range that is not already in
OUTFILE (so an interrupted run can be resumed), WORKERS at a time. Trial t is in coupling regime t mod 4
(dispersed, ordered, near-critical, critical), which fixes its range of beta. The data of a trial depend
only on its number, and the generators live in one place (`common.py`), so the results of different
methods pair by trial id. `run_trials.sh` is the same thing for the Table 3 protocol. The summarisers
refuse files with duplicate trial ids.

Each trial records a fingerprint (`source`) of the C sources, the binary and every script in this folder,
and a manifest with those hashes, the git commit and the Python environment is written beside the result
file, named after the fingerprint (`OUTFILE.provenance.<fingerprint>.json`) and never overwritten; a file
produced by more than one version of the code therefore has one manifest per version. (Result files from
before this scheme have a single `X.jsonl.provenance.json` and a fingerprint of the C sources only.) The
helpers refuse to run with a binary that is not up to date with its sources.

## Missing data

| script | what it does | output |
|---|---|---|
| `population_consistency.py [n] [n_blank]` | Infinite-data test, by exact enumeration and independent of the C code, of whether each treatment of missing data recovers the true parameters when entries are missing completely at random. | `results/population_n6.txt` |
| `hidden_node_check.py [n_visible]` | The same kind of test for a node that is never observed (a hidden node): gradient of each objective at the truth, and what its minimiser does to the couplings to the hidden node and to the distribution of the visible nodes. | `results/hidden_node.txt` |
| `large_sample.py beta M seed [mar] [nnK]` | n=20, 2M rows: KL from the truth for complete-rows-only, legacy, marginal and frozen-weight fits, each at its best lambda on a grid; with entries missing completely at random, or (`mar`) depending on observed answers; single flips or (`nn2`) larger neighbourhoods. | `results/large_sample.txt`, `results/large_sample_nn2.txt` |
| `partial_oracle_trial.py` | The same data as the Table 3 protocol, but every method scored at its best lambda on a grid, without cross-validation. | `results/table3_oracle.jsonl` |
| `partial_trial.py`, `run_trials.sh`, `summarize_table3.py` | The paper's Table 3 protocol (128 complete; 128 + 128 partial under each treatment; naive imputation; 256 complete; lambda by cross-validation) with exact sampling. One result file per group of columns. | `results/table3_complete_columns.jsonl`, `table3_legacy.jsonl`, `table3_marginal.jsonl`, `table3_frozen.jsonl`, `table3_summary.txt` |
| `drh_missingness.py` | Tests whether the blanks in the curated DRH data are related to the answers (they are). | `results/drh_missingness.txt` |
| `drh_bootstrap.py` | Parametric bootstrap at the size of the DRH data with its real patterns of blanks, assigned at random or according to an observed answer: KL from the truth for each treatment. | `results/drh_bootstrap.jsonl` |

## Regularisation, cross-validation, neighbourhoods, sampler

| script | what it does | output |
|---|---|---|
| `table1_trial.py`, `table2_trial.py`, `run_table.sh`, `summarize_tables.py` | The paper's Table 1 (optimal vs cross-validated lambda vs none) and Table 2 (re-weighting) protocols, with the published Monte Carlo sampler and with exact sampling; Table 1 also on the same landscapes with one-, two- and three-flip neighbourhoods. | `results/table1_mcmc.jsonl`, `table1_exact.jsonl`, `table1_nn1.jsonl`, `table1_nn2.jsonl`, `table1_nn3.jsonl`, `table2_exact.jsonl`, `tables_summary.txt` |
| `check_sampler.py beta n_samples seed` | Compares samples from `mpf -g` (Monte Carlo and exact) with the exact distribution. | `results/sampler.txt` |
| `repo_logs_table.py` | Recomputes the paper's Table 3 from the logs committed in `../paper_runs/NAN_TESTS`. | `results/repo_logs_table.txt` |

## The DRH fit

| script | what it does | output |
|---|---|---|
| `results/drh_refits/refit.sh [name ...]` | Refits the DRH data from the committed input: published procedure (`legacy`), `marginal`, `marginal_entrycv`, frozen-weight with entry-wise cross-validation at NN = 1, 2, 3 (`frozen_entrycv`, `frozen_entrycv_nn2`, `frozen_entrycv_nn3`), and `legacy_nn2`. | `results/drh_refits/` (parameters, logs, stderr, provenance; `entry_groups.txt` is the `--groups` file) |
| `drh_compare.py` | Compares each refit with the parameter file used in the paper. The table, including each lambda, is generated from the logs. | `results/drh_compare.txt` |
| `clustering_sensitivity.py` | Stability of the three groups of Figure 4A to the number of configurations clustered and to the fit. | `results/clustering_sensitivity.txt` |

The analysis regenerated from these fits is in `regenerated/` at the top of the repository.

## Comparison with the R package bgms

In `bgms/`; results in `results/bgms/` (`summary.txt` is the table of everything). The summary and what we
take from it are in the README at the top of the repository. Needs R with bgms (0.2.0.0 was used; set
`BGMS_LIB` to the library that holds it if it is not on R's default path). The runs were:

```
./run_table.sh bgms/bgms_trial.py 0 63 14 results/bgms/table1.jsonl "table1 4"
./run_table.sh bgms/bgms_trial.py 0 63 14 results/bgms/table3.jsonl "table3 4"
./run_table.sh bgms/bgms_trial.py 0 39 8 results/bgms/drhboot.jsonl "drhboot 4"
./run_table.sh bgms/mpf_timing.py 0 15 4 results/bgms/mpf_timing.jsonl 4
./run_table.sh bgms/bgms_drh.py 0 9 10 results/bgms/drh_cv.jsonl "cv 4"
python3 bgms/bgms_drh.py fit 4 > results/bgms/drh_fit.json
python3 bgms/summarize_bgms.py
```

| script | what it does | output |
|---|---|---|
| `bgms/bgms_fit.R` | Fits bgms to a data file in `mpf`'s format and writes the posterior mean, intervals and draws in `mpf`'s parameterisation. | — |
| `bgms/bgms_check.py` | Checks that conversion against a known model. | — |
| `bgms/bgms_trial.py ID LO HI table1\|table3\|drhboot` | bgms on exactly the datasets of the Table 1 and Table 3 re-runs (paired with the stored `mpf` results by trial id), and bgms and cross-validated `mpf` on DRH-like bootstrap data: KL from the truth, coverage of bgms's intervals, time. | `results/bgms/table1.jsonl`, `table3.jsonl`, `drhboot.jsonl` |
| `bgms/mpf_timing.py` | Times `mpf -c` on the Table 1 datasets with the same four cores bgms is given. | `results/bgms/mpf_timing.jsonl` |
| `bgms/bgms_drh.py fit` and `FOLD _ _ cv` | Both programs on the real DRH data, one row per entry: fits to all entries, and a 10-fold cross-validation scored by the probability of held-out entries. | `results/bgms/drh_fit.json`, `drh_cv.jsonl`, `drh/` |
| `bgms/summarize_bgms.py` | The summary. | `results/bgms/summary.txt` |

## Checks on the code itself

| script | what it does | output |
|---|---|---|
| `compare_with_published_code.py` | Builds the published C code (which needs GSL) from git (commit 23dce37, or the directory named in `MPF_PUBLISHED_SRC`) and compares it with the rewritten program in `--legacy` mode on random inputs, on the DRH input, and — with the published two- and three-flip blocks re-enabled — at NN = 2 and 3. | `results/compare_with_published_code.txt` |
| `mutation_test.py` | Injects deliberate bugs into a temporary copy of the sources, one at a time, and checks that `make test` fails. Takes over an hour. | `results/mutation_test.txt` |
| `gsl_reference/check_against_gsl.c` | The program no longer uses the GNU Scientific Library. This compares the code that replaced it — random numbers, digamma and trigamma functions, the NSB entropy — with GSL itself (build line in the file; the only thing here that needs GSL besides the published code). | `results/check_against_gsl.txt` |

## Notes on the stored results

* The simulation tables were run on a 256-core Linux server (`results/wright_driver.log` is the driver's
  log); `results/local_partial/table1_nn1_local.jsonl` is a partial run of the NN=1 column of Table 1 on a
  Mac, kept as a cross-platform check (KL agrees to 2e-4).
* `results/large_sample.txt` (NN=1) was produced before neighbourhoods larger than one flip were
  implemented; the NN=1 code path it exercises was not changed by that work.
* Runs differ in the `source` fingerprint they carry because the code changed between them (the NN > 1
  work, two code reviews and two external reviews, the removal of GSL). Results that a change could alter
  were re-run after it: every cross-validated fit with missing data (the DRH refits, `table3_marginal.jsonl`,
  `table3_frozen.jsonl`, the `mpf` side of `bgms/`) is from the final program, as are `table3_legacy.jsonl`
  and the bootstrap. Table 1, Table 2, the complete-data columns of Table 3 and the fixed-lambda
  experiments are from earlier builds; nothing that changed since affects them (Table 1 was re-run once to
  confirm it, with identical numbers, and removing GSL leaves parameter files byte-identical).
* The Table 1 and Table 2 trials call `mpf -o` and `mpf -c` on complete data with `--legacy`, i.e. the
  published protocol; for complete data the only difference from the default is that the search for lambda
  stays within [-2, 4].
