"""One simulation trial following ../big_test_nans.rb (the script behind the paper's Partial-MPF table), but
with exact sampling instead of `mpf -g`'s Monte Carlo sampler:

   128 complete rows | 128 complete + 128 rows with `nan` of n entries missing | 256 complete rows | naive imputation

Entries are blanked completely at random (independently of every answer). Every fit chooses lambda by
leave-one-out cross-validation. Prints one JSON line.

Usage: python3 partial_trial.py trial_id beta_lo beta_hi n nan [jobs]
  jobs: comma-separated subset of base,full,naive,partial_legacy,partial_marginal,partial_frozen (default: all but partial_frozen)
"""
import json, os, random, sys, tempfile, time
import numpy as np
from common import draw_model, all_probabilities, state_string, write_params, write_data, kl, fit_cv, source_tag

tid, lo, hi, n, nan = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
wanted = sys.argv[6].split(',') if len(sys.argv) > 6 else ['base', 'full', 'naive', 'partial_legacy', 'partial_marginal']
rng = random.Random(1000 + tid)
while True:
    beta = rng.random() ** 2          # same draw as big_test_nans.rb, restricted to the regime
    if lo <= beta < hi and beta >= 0.01: break
nrng = np.random.default_rng(5000 + tid)
theta = draw_model(n, beta, nrng)
p, _ = all_probabilities(n, theta)
lines = [state_string(d, n) + " 1.0" for d in nrng.choice(1 << n, size=256, p=p)]
del p
part = []
for l in lines[128:]:
    code = list(l)
    for i in rng.sample(range(n), nan): code[i] = 'X'
    part.append("".join(code))
avg = [sum(int(l[k]) for l in lines[:128]) / 128.0 for k in range(n)]
imputed = ["".join((str(int(avg[i] + 0.5)) if c == 'X' else c) for i, c in enumerate(q[:n])) + " 1.0" for q in part]
res = dict(tid=tid, beta=beta, n=n, nan=nan, source=source_tag())
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
        t = time.time(); f, lam, warnings = fit_cv(fn, flags, tag=key)
        res[key] = dict(kl=kl(fn, 'truth.dat', f), lam=lam, secs=round(time.time() - t, 1), warnings=warnings)
print(json.dumps(res), flush=True)
