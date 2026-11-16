# Tests

Run with `make test` from `MPF_CMU` (about a minute).

* `test_mpf.c` unit tests. Every numerical routine is compared with a slow implementation written
  directly from the definitions (explicit bit flips, energies computed from scratch):
  * the objective for all three treatments of missing data, n from 2 to 62, with blanks, weights and
    duplicate rows; gradients by finite differences
  * neighbourhoods of two, three and four flips (NN = 2..4), for all three treatments, including the
    frozen-weight flows that flip a missing answer together with an observed one
  * partition function, KL divergence and pseudo-likelihood, including parameters far too large for
    `exp(H)` to be finite
  * the `-z` output, the exact sampler and the Monte Carlo sampler (chi-square against exact probabilities)
  * the minimiser (stationarity, independence of the starting point)
  * infinite-data consistency: on exact population data with entries missing completely at random, the
    marginal objective and the frozen-weight iteration recover the true parameters and the legacy objective
    does not; with missingness that depends on observed answers only the frozen-weight iteration does
    (checked at NN = 1, 2 and 3)
  * extreme parameters: the objective stays correct in the log domain, and a fit whose flows overflow is
    reported as a numerical failure rather than as convergence
  * cross-validation against explicit refits (row-wise, weighted, grouped, with a row mask), and, with
    missing data, against explicit training-rows-only staged refits and independence of the starting vector
  * the search over lambda (interior minima, minima at and below the edge of the range, two minima, a
    function that is invalid everywhere or in part)
* `test_cli.sh` command-line tests (97 checks): exit codes, malformed input, file formats, determinism
  (same answer for any number of threads), all seven modes, NN = 2 and 3, more than 31 nodes, a fit that
  runs out of iterations being reported as unconverged (`unconverged_data.dat`), and reproduction of the
  committed DRH parameter file with `--legacy`.

Of the 201,303 checks in `test_mpf.c`, 200,000 are range checks on individual draws of the exact sampler;
the rest are the comparisons above. Both suites pass on macOS (clang) and Linux (gcc), and under an
address + undefined-behaviour sanitizer build.

The suite is itself checked by `validation/mutation_test.py`, which injects deliberate bugs one at a time
and confirms that `make test` fails (results in `validation/results/mutation_test.txt`).
