"""Paper-scale comparison of the three treatments of missing data without cross-validation: the same data as
partial_trial.py (128 complete + 128 rows with 5 of 20 entries blanked completely at random), each method
fit on a grid of lambda and scored at its best lambda (an oracle choice). Cheap, so it can include the
frozen-weight iteration for every trial. Prints one JSON line.
Usage: python3 partial_oracle_trial.py trial_id beta_lo beta_hi
"""
import os, sys, tempfile
from common import table3_dataset, write_params, write_data, kl, fit_fixed, emit

n = 20
tid, lo, hi = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
beta, theta, lines, part = table3_dataset(tid, lo, hi, n)
res = dict(tid=tid, beta=beta)
grid = [-1.0 + 0.25 * i for i in range(13)]            # log10(lambda) from -1 to 2
with tempfile.TemporaryDirectory() as td:
    os.chdir(td)
    write_params('truth.dat', theta)
    write_data('base.dat', n, lines[:128]); write_data('full.dat', n, lines); write_data('partial.dat', n, lines[:128] + part)
    for key, fn, flags in (('base', 'base.dat', ()), ('full', 'full.dat', ()), ('legacy', 'partial.dat', ('--legacy',)),
                           ('marginal', 'partial.dat', ('--marginal',)), ('frozen', 'partial.dat', ('--frozen',))):
        best = None; bad = 0
        for lam in grid:
            f, status = fit_fixed(fn, lam, flags, tag=key)
            bad += (status != 'converged')
            k = kl(fn, 'truth.dat', f)
            if best is None or k < best[0]: best = (k, lam)
        res[key] = dict(kl=best[0], lam=best[1], not_converged=bad)
emit(res)
