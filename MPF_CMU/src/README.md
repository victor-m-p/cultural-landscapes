# mpf: Ising models by Minimum Probability Flow

`mpf` fits an Ising model (pairwise couplings `J`, local fields `h`) to binary data with Minimum
Probability Flow, with L2 regularisation chosen by leave-one-out cross-validation, per-row weights, and
missing entries ("Partial-MPF"). It is the program behind "Inferring Cultural Landscapes with the Inverse
Ising Model" (Poulsen & DeDeo, *Entropy* 25(2), 264, 2023), as revised in 2026.

This directory is self-contained: C99 sources, a Makefile and tests, with no external libraries. Copy it
wherever you need it.

The 2026 revision — what changed, the errors found in the published code, and which of them affect the
paper — is described in `CLAUDE_IMPROVEMENTS_SUMMARY.md` at the top of
https://github.com/victor-m-p/cultural-landscapes. In short: the results in the paper are reproduced by
`--legacy`; the default treats missing data differently (see "Missing data" below).

A cross-validated fit of the paper's data (20 questions, 455 rows, up to 5 missing entries per row) takes
one to three minutes on a laptop, depending on the options; a fit at a fixed lambda takes a few hundredths
of a second.

# Building

```
make            # builds ./mpf
make test       # unit tests + command-line tests (about a minute with OpenMP)
```

Only a C compiler and the C maths library are needed. If the compiler supports OpenMP the
cross-validation folds run in parallel (set `OMP_NUM_THREADS` to limit the cores used); if not — Apple's
clang, for example — `make` says so and builds a single-threaded program that gives the same results.

```
make CC=/opt/local/bin/clang-mp-19      # Mac with OpenMP: a MacPorts clang (or CC=gcc-14 from Homebrew)
make OPENMP=                            # force a single-threaded build
make CFLAGS="-O2 -g -fopenmp"           # other compiler flags
```

Earlier versions needed the GNU Scientific Library. It was used only for random numbers and for one
diagnostic (the NSB entropy printed by `-o`); `rng.c` and `nsb.c` now do both, and `mpf -g --seed N`
writes exactly the same files as before.

# Data format

```
3            <- number of rows
4            <- number of nodes (questions)
0101 1.0     <- one row: n characters (0 = no, 1 = yes, X = missing), then a weight
1X10 1.0
0011 0.5
```

Identical rows are merged (their weights add). If the file contains more rows than the header says, only
the first ones are used. Up to 62 nodes and 20 missing entries per row are supported. Numbers are read
strictly, in files and on the command line: the whole field must be a number in range (`1.0abc`, `nan`, a
row count that overflows, or a parameter file with more values than the model has are errors).

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
(The penalty is lambda * N * (sum of weights) / (number of parameters) * sum(theta^2) / 2, where N is the
number of neighbours of a state: n for NN=1, n + n(n-1)/2 for NN=2, and so on.)

`mpf -c [filename] [NN]`
Choose lambda by leave-one-out cross-validation, then fit; writes `[filename]_params.dat`. Only rows
without missing entries are held out, and each is scored by its exact log-likelihood (n <= 24) or its log
pseudo-likelihood (larger n). lambda is searched on a grid over log10(lambda) in [-2, 4], extended
downwards (to -8) while the score keeps improving, then refined by golden section around the best grid
point. A fit that fails numerically makes that lambda invalid; fits that do not converge are reported on
stderr, and a line on stdout says how many fold fits did not converge at the lambda finally chosen.

If some rows have missing entries the problem is not convex, and every fit is made in two stages: the
complete rows first (a convex problem), then all rows starting from that solution, both at the same
lambda. Each cross-validation fold repeats the two stages on its own training rows, so nothing in a fold
depends on the row it holds out. (`--legacy` instead does what the published code did: it chooses a
separate lambda for the first stage from all the complete rows and starts every fold from that one fit.)

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
| `--legacy` | reproduce the published fits: the original Partial-MPF objective; cross-validation that counts every held-out row equally regardless of its weight and starts every fold from one fit to all the complete rows, made at a lambda of its own; lambda confined to [-2, 4] |
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
(derivations and tests in the 2026 analysis, see below; unit tests in `tests/test_mpf.c`):

| option | what is minimised | with unlimited data the true parameters are a solution if entries are missing... |
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
lack answers about belief), `--frozen` is the only one of the three with a justification that could
apply, and comparing it with `--marginal` is a useful check. These are statements about what happens with
unlimited data at the true parameters; they do not show that the solution is unique, that the iteration
reaches it, or what happens with the amount of regularisation a small data set needs.

To reproduce the parameter file used in the paper (`tests/drh_data.mpf` is the paper's data and
`tests/drh_published_params.dat` the parameters published with it):

```
cp tests/drh_data.mpf /tmp/drh.mpf
./mpf -c /tmp/drh.mpf 1 --legacy          # writes /tmp/drh.mpf_params.dat
```

The result agrees with the published parameters to about 7e-4; `./mpf -l /tmp/drh.mpf 0.205745 1 --legacy`
(the published lambda) agrees to 2e-6.

# Files

* `mpf.h` data structures, limits, and the interface between the files below
* `mpf.c` data handling, the objective and its gradient, the L-BFGS minimiser
* `mcmc.c` exact enumeration (partition function, KL, probabilities), exact and Monte Carlo sampling
* `nlin.c` cross-validation and the search over lambda
* `rng.c` random numbers
* `nsb.c` entropy estimate for the diagnostic output of `-o` (with the special functions and quadrature it needs)
* `main.c` command line
* `tests/` unit and command-line tests (`make test`), and the data they use
