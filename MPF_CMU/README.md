# General notes

`mpf` fits an Ising model (pairwise couplings `J`, local fields `h`) to binary data with Minimum
Probability Flow, with L2 regularisation chosen by leave-one-out cross-validation, per-row weights, and
missing entries ("Partial-MPF").

The code was revised after publication; **`CLAUDE_IMPROVEMENTS_SUMMARY.md` at the top of the repository
describes every change, the errors found, and which of them affect the paper**. In short: the results in
the paper are reproduced by `--legacy`; the default treats missing data differently (see "Missing data" below).

A cross-validated fit of the DRH data (n=20, 455 rows, up to 5 missing entries per row) takes one to three
minutes on a laptop, depending on the options; a fit at a fixed lambda takes a few hundredths of a second.

# Building

Needs GSL, and a compiler with OpenMP if you want the cross-validation folds to run in parallel.

```
make                                    # gcc + OpenMP (Linux, PSC after `module load gcc`)
make CC=/opt/local/bin/clang-mp-19      # Mac: Apple's clang has no OpenMP; use a MacPorts/Homebrew clang
make OPENMP=                            # any C compiler, single-threaded
make test                               # unit tests + command-line tests (about a minute)
```

# Data format

```
3            <- number of rows
4            <- number of nodes (questions)
0101 1.0     <- one row: n characters (0 = no, 1 = yes, X = missing), then a weight
1X10 1.0
0011 0.5
```

Identical rows are merged (their weights add). If the file contains more rows than the header says, only
the first ones are used. Up to 62 nodes and 20 missing entries per row are supported.

# Modes

`NN` selects the neighbourhood: flows go from each data state to every state within Hamming distance
`NN` — 1 is single flips (the paper's main results), 2 adds every pair of flips, 3 every triple, up to 6.
The cost per state grows with the number of such neighbours (20, 210, 1350, ... for 20 nodes), and the
penalty is scaled by that number, as in the published code. With missing entries, only observed entries are
flipped (except by `--frozen` with `NN > 1`; see "Missing data" below).
Parameter files hold the n(n-1)/2 couplings in the order (0,1), (0,2), ..., (1,2), ... followed by the n
fields; the model is p(s) proportional to exp(sum J_ab s_a s_b + sum h_a s_a) with s = +1 (yes) / -1 (no).

`mpf -l [filename] [logsparsity] [NN]`
Fit at a fixed regularisation strength lambda = 10^logsparsity; writes `[filename]_params.dat`.
(The penalty is lambda * n * (sum of weights) / (number of parameters) * sum(theta^2) / 2.)

`mpf -c [filename] [NN]`
Choose lambda by leave-one-out cross-validation, then fit; writes `[filename]_params.dat`. If some rows
have missing entries, lambda is first chosen (and a fit made) on the complete rows only, and that fit is
the starting point for the second stage on all rows. Only rows without missing entries are held out, and
each is scored by its exact log-likelihood (n <= 24) or its log pseudo-likelihood (larger n). Each fold
repeats the same two stages on its own training rows, so the held-out row cannot influence its own fold
through the starting point. lambda is searched on a grid over log10(lambda) in [-2, 4], extended downwards
(to -8) while the score keeps improving, then refined by golden section around the best grid point. A fit
that fails numerically makes that lambda invalid; fits that do not converge are reported on stderr.

`mpf -g [filename] [n_nodes] [n_obs] [beta]`
Draw J and h from a Gaussian with standard deviation beta and generate data; writes
`[filename]_params.dat` and `[filename]_data.dat`. For n <= 24 the data are exact, independent draws;
for larger n a Monte Carlo sampler is used (see the warning it prints).

`mpf -t [filename] [paramfile] [NN]`
Cross-validated fit, then the KL divergence of the fit from the true parameters in `paramfile`.

`mpf -o [filename_prefix] [NN]`
For simulated data (`_data.dat` and `_params.dat`): the best lambda judged by KL from the truth, the
cross-validated lambda, and the KL of each (and of the unregularised fit) from the truth.

`mpf -k [filename] [paramfile_truth] [paramfile_inferred]`
KL divergence (nats) of the inferred distribution from the true one. n <= 30.

`mpf -z [paramfile] [n_nodes]`
Write every configuration, its energy and its probability to `[paramfile]_probs.dat`. n <= 30.

# Options

| option | meaning |
|---|---|
| `--marginal` | (default) missing data handled by MPF on the marginal distribution of the observed nodes |
| `--frozen` | missing data handled by freezing the completion weights while minimising, re-estimating them, and iterating to a fixed point (about three times slower) |
| `--legacy` | reproduce the published fits: the original Partial-MPF objective; cross-validation that counts every held-out row equally regardless of its weight and starts every fold from the full-data fit; lambda confined to [-2, 4] |
| `--legacy-cv` | only the cross-validation half of `--legacy` |
| `--groups FILE` | one integer per data row; rows with the same id (e.g. the expansions of one inconsistently-coded entry) are held out together |
| `--start FILE` | (`-l`) starting parameters |
| `--seed N` | (`-g`) seed, for reproducible simulated data |
| `--mcmc` | (`-g`) force the Monte Carlo sampler used for the published simulations |
| `--verbose` | print every cross-validation evaluation |

# Missing data: what each option assumes

All three options expand a row with m missing entries into its 2^m completions and use flows along the
observed entries only. They differ in how the completions are combined, and in what has to be true about
*why* entries are missing for the method to converge on the right answer as data accumulate
(derivations and tests in `CLAUDE_IMPROVEMENTS_SUMMARY.md`; unit tests in `tests/test_mpf.c`):

| option | what is minimised | consistent if entries are missing... |
|---|---|---|
| `--marginal` | a single objective: MPF on the marginal distribution of the observed entries | **completely at random** (the pattern of blanks is unrelated to any answer) |
| `--frozen` | no single objective: the fixed point of "weights frozen, flows minimised, weights re-estimated" | **at random** (the pattern may depend on answers that are observed in that row) |
| `--legacy` | the published Partial-MPF objective | neither (biased even when blanks are completely at random) |

With `NN > 1`, `--frozen` also sends flows to neighbours that flip a missing entry together with an
observed one (weighted by the frozen completion weights). That is what keeps its "at random" guarantee
for larger neighbourhoods; `--marginal` and `--legacy` flip observed entries only.

None of them is correct if whether an answer is missing depends on that answer itself, and none is convex
when entries are missing: the result is a local solution, found from the complete-rows fit. If the
blanks in your data plausibly depend on other answers (for example, cultures known only archaeologically
lack answers about belief), `--frozen` is the better-justified choice, and comparing it with `--marginal`
is a useful check.

To reproduce the parameter file used in the paper:

```
./mpf -c ../data/mdl_experiments/matrix_questions_20_maxna_5_nrows_455_entries_407.txt.mpf 1 --legacy
```

(this overwrites the `_params.dat` next to the data file with a result that agrees with the committed
one to about 7e-4; `-l [file] 0.205745 1 --legacy` agrees to 2e-6. `validation/results/drh_refits/refit.sh`
does the same on a copy.)

# Layout

* `mpf.c` data handling, the objective and its gradient, the L-BFGS minimiser
* `mcmc.c` exact enumeration (partition function, KL, probabilities), exact and Monte Carlo sampling
* `nlin.c` cross-validation and the search over lambda
* `nsb.c` entropy estimate for the diagnostic output of `-o`
* `main.c` command line
* `tests/` unit and command-line tests (`make test`)
* `validation/` the experiments behind `CLAUDE_IMPROVEMENTS_SUMMARY.md`, and their outputs
* `*.rb`, `*.pro` the scripts used on the Pittsburgh Supercomputing Center for the paper, kept for the
  record. They were written for the published version of `mpf`: run against the current one they use its
  new defaults (marginal objective, training-rows-only weighted cross-validation, exact sampling, real
  multi-flip neighbours for NN > 1). To repeat the published procedure add `--legacy` to their
  `-c`/`-o`/`-t`/`-l` calls and `--mcmc` to their `-g` calls, and use NN=1; each script that calls `mpf`
  carries a note saying so. Three of them (`best_fit_DRH_2.rb`, `cv_scan.rb`, `param_scan.rb`) also call
  `-c` and `-t` with an argument order from an earlier version of the program and will not run as they are.

# final parameter fits

* based on the lists that appear in /data/reference/main_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv

* questions have the same order as in /data/reference/main_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv

* the following entry entry ids have been removed, because they are tagged by DRH as religious texts
[1025, 1042, 1050, 1054, 1055, 1056, 1057, 1058, 1062, 1063, 1067, 1074, 1075, 1078, 1079, 1080, 1098, 1100, 1112, 1117, 1120, 1121, 1124, 1141, 1144, 1158, 1160, 1170, 1176, 1177, 1182, 1188, 1203, 1207, 1224, 1232, 1239, 1243, 1252, 1253, 1255, 1260, 1265, 1269, 1270, 1271, 1272, 1290, 1294, 1297, 1324, 1326, 1328, 1330, 1338, 1342, 1345, 1347, 1356, 1367, 1375, 1387, 1389, 1393, 1401, 1405, 1406, 1417, 1418, 1421, 1422, 1424, 1427, 1429, 1432, 1442, 1443, 1452, 1471, 1473, 1484, 1486, 1508, 1509, 1513, 1525]

* the fits are with one nearest neighbour, and have the suffix _params.dat; J's and then h's order.

* right now, we have NOT characterized the stability for N=20 when NA > 5; we have the CPU-hours, so I've run the fits on the Pittsburgh Supercomputer Center for NAs larger, just so we have them.

In part because they focus on the content of a text, which might not reflect the full spectrum of attitudes within the religion as a whole, text questions end up having quite different statistics from the group cases:

Q0: Does the religion have official political support
Text average: 0.297 vs. Non-text average: 0.567 ± 0.021
Q1: Does the religious group have scriptures:
Text average: 0.276 vs. Non-text average: 0.612 ± 0.021
Q2: Is monumental religious architecture present:
Text average: NaN vs. Non-text average: 0.519 ± 0.021
Q3: Is a spirit-body distinction present:
Text average: 0.408 vs. Non-text average: 0.868 ± 0.013
Q4: Belief in afterlife:
Text average: 0.486 vs. Non-text average: 0.866 ± 0.013
Q5: Reincarnation in this world:
Text average: 0.085 vs. Non-text average: 0.322 ± 0.020
Q6: Are there special treatments for adherents’ corpses:
Text average: 0.121 vs. Non-text average: 0.689 ± 0.020
Q7: Are co-sacrifices present in tomb/burial:
Text average: 0.028 vs. Non-text average: 0.181 ± 0.018
Q8: Are grave goods present:
Text average: 0.087 vs. Non-text average: 0.455 ± 0.023
Q9: Are formal burials present:
Text average: 0.137 vs. Non-text average: 0.739 ± 0.018
Q10: Are supernatural beings present:
Text average: 0.710 vs. Non-text average: 0.870 ± 0.013
Q11: Is supernatural monitoring present:
Text average: 0.310 vs. Non-text average: 0.697 ± 0.020
Q12: Do supernatural beings mete out punishment:
Text average: 0.304 vs. Non-text average: 0.719 ± 0.019
Q13: Does membership in this religious group require castration:
Text average: 0.000 vs. Non-text average: 0.003 ± 0.002
Q14: Does membership in this religious group require sacrifice of adults:
Text average: 0.000 vs. Non-text average: 0.049 ± 0.009
Q15: Does membership in this religious group require sacrifice of children:
Text average: 0.009 vs. Non-text average: 0.023 ± 0.007
Q16: Does membership in this religious group require self-sacrifice (suicide):
Text average: 0.000 vs. Non-text average: 0.020 ± 0.006
Q17: Does membership in this religious group require participation in small-scale rituals (private, household):
Text average: 0.325 vs. Non-text average: 0.576 ± 0.024
Q18: Does membership in this religious group require participation in large-scale rituals:
Text average: 0.217 vs. Non-text average: 0.670 ± 0.021
Q19: Does the religious group in question possess its own distinct written language:
Text average: 0.214 vs. Non-text average: 0.226 ± 0.019

