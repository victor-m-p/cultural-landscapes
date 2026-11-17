"""One trial of the paper's Table 2 (re-weighting to correct sampling bias), following ../paper_runs/bias_correction.rb,
with exact sampling: 256 observations in which one question splits 128/128; the biased sample drops every
other "yes" row; the corrected fit gives the "no" rows weight 1/2. Prints one JSON line.
Usage: python3 table2_trial.py trial_id beta_lo beta_hi
"""
import os, random, sys, tempfile
import numpy as np
from common import draw_model, all_probabilities, state_string, write_params, write_data, kl, fit_cv, emit

n = 20
tid, lo, hi = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
rng = random.Random(4000 + tid); nrng = np.random.default_rng(8000 + tid)
while True:
    beta = rng.random() ** 2 + 0.01
    if lo <= beta < hi: break
for attempt in range(200):
    theta = draw_model(n, beta, nrng)
    p, _ = all_probabilities(n, theta)
    rows = [state_string(d, n) for d in nrng.choice(1 << n, size=256, p=p)]
    balanced = [k for k in range(n) if sum(r[k] == '1' for r in rows) == 128]
    if balanced: break
else:
    sys.exit(0)        # no balanced question found for this beta (as in the original script, the trial is skipped)
q = balanced[0]
biased, weighted, counter = [], [], 0
for r in rows:
    if r[q] == '0':
        biased.append(r + " 1.0"); weighted.append(r + " 0.5")
    else:
        if counter % 2 == 0: biased.append(r + " 1.0"); weighted.append(r + " 1.0")
        counter += 1
idx = np.arange(1 << n)
def odds(theta_fit):                       # p(question q = no) / p(question q = yes)
    pf, _ = all_probabilities(n, theta_fit)
    m = pf[((idx >> q) & 1) == 0].sum()
    return m / (1 - m)
res = dict(tid=tid, beta=beta)
with tempfile.TemporaryDirectory() as td:
    os.chdir(td)
    write_params('truth.dat', theta)
    write_data('full.dat', n, [r + " 1.0" for r in rows]); write_data('biased.dat', n, biased); write_data('weighted.dat', n, weighted)
    true_odds = odds(theta)
    for key, fn in (('ideal', 'full.dat'), ('naive', 'biased.dat'), ('corrected', 'weighted.dat')):
        f, lam, cv = fit_cv(fn, ('--legacy',), tag=key)
        res[key] = dict(kl=kl('full.dat', 'truth.dat', f), lam=lam, log_bias=float(np.log(true_odds / odds(np.loadtxt(f)))), **cv)
emit(res)
