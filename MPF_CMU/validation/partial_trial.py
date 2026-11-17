"""One simulation trial following ../paper_runs/big_test_nans.rb (the script behind the paper's Partial-MPF table), but
with exact sampling instead of `mpf -g`'s Monte Carlo sampler:

   128 complete rows | 128 complete + 128 rows with `nan` of n entries missing | 256 complete rows | naive imputation

Entries are blanked completely at random (independently of every answer). Every fit chooses lambda by
leave-one-out cross-validation. Prints one JSON line; each fit records the warnings printed during its
search and how many fold fits did not converge at the lambda it chose.

Usage: python3 partial_trial.py trial_id beta_lo beta_hi n nan [jobs]
  jobs: comma-separated subset of base,full,naive,partial_legacy,partial_marginal,partial_frozen (default: all but partial_frozen)
"""
import os, sys, tempfile, time
from common import table3_dataset, write_params, write_data, kl, fit_cv, emit

tid, lo, hi, n, nan = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
wanted = sys.argv[6].split(',') if len(sys.argv) > 6 else ['base', 'full', 'naive', 'partial_legacy', 'partial_marginal']
beta, theta, lines, part = table3_dataset(tid, lo, hi, n, nan)
avg = [sum(int(l[k]) for l in lines[:128]) / 128.0 for k in range(n)]
imputed = ["".join((str(int(avg[i] + 0.5)) if c == 'X' else c) for i, c in enumerate(q[:n])) + " 1.0" for q in part]
res = dict(tid=tid, beta=beta, n=n, nan=nan)
jobs = dict(base=('base.dat', ()), full=('full.dat', ()), naive=('naive.dat', ()),
            partial_legacy=('partial.dat', ('--legacy',)), partial_marginal=('partial.dat', ('--marginal',)),
            partial_frozen=('partial.dat', ('--frozen',)))
with tempfile.TemporaryDirectory() as td:
    os.chdir(td)
    write_params('truth.dat', theta)
    write_data('base.dat', n, lines[:128]); write_data('full.dat', n, lines)
    write_data('partial.dat', n, lines[:128] + part); write_data('naive.dat', n, lines[:128] + imputed)
    for key in wanted:
        fn, flags = jobs[key]
        t = time.time(); f, lam, cv = fit_cv(fn, flags, tag=key)
        res[key] = dict(kl=kl(fn, 'truth.dat', f), lam=lam, secs=round(time.time() - t, 1), **cv)
emit(res)
