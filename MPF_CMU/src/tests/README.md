# Tests

Run with `make test` from the directory above this one (about a minute with OpenMP). Nothing outside
that directory is needed.

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
    missing data, against the two-stage fit made by hand from each fold's training rows alone (on two data
    sets, at three values of lambda), and independence of any starting vector
  * the search over lambda (interior minima, minima at and below the edge of the range, two minima, a
    function that is invalid everywhere or in part)
  * the random number generator (known answers recorded from the GNU Scientific Library, which earlier
    versions used), the digamma and trigamma functions, the quadrature and the NSB entropy
* `test_cli.sh` command-line tests (111 checks): exit codes, malformed input, file formats, determinism
  (same answer for any number of threads), all seven modes, NN = 2 and 3, more than 31 nodes, a fit that
  runs out of iterations being reported as unconverged (`unconverged_data.dat`), and reproduction of the
  published DRH parameter file with `--legacy` (`drh_data.mpf`, `drh_published_params.dat`).

Of the 201,361 checks in `test_mpf.c`, 200,000 are range checks on individual draws of the exact sampler;
the rest are the comparisons above. Both suites pass on Linux (gcc, with OpenMP) and macOS (Apple clang,
without OpenMP), and under an address + undefined-behaviour sanitizer build.

The suite has itself been checked by mutation testing: deliberate bugs are injected one at a time and
`make test` must fail (`MPF_CMU/validation/mutation_test.py` in
https://github.com/victor-m-p/cultural-landscapes).
