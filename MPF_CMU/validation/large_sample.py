"""With sampling noise suppressed, does each treatment of missing data converge on the truth?
n=20, 2M rows. For each method, KL from the truth at the best lambda on a coarse grid (an oracle choice, not
cross-validation; the grid extends below the default search range).

  mcar (default): M complete rows + M rows with 5 of the 20 entries blanked at random, independently of every answer.
  mar:            every row has questions 15-19 blanked with probability 0.85 if at least three of questions 0-4
                  are "yes" and 0.15 otherwise. Missingness depends only on answers that are always observed
                  (missing at random), but not completely at random.

Usage: python3 large_sample.py beta M seed [mar] [nn2]        e.g.  python3 large_sample.py 0.3 8192 1
"""
import os, sys, tempfile
import numpy as np
from common import draw_model, all_probabilities, state_string, write_params, write_data, kl, fit_fixed

n = 20; beta = float(sys.argv[1]); M = int(sys.argv[2]); seed = int(sys.argv[3]); mar = 'mar' in sys.argv[4:]
nn = next((int(a[2:]) for a in sys.argv[4:] if a.startswith('nn')), 1)     # neighbourhood, e.g. nn2
rng = np.random.default_rng(seed); nJ = n * (n - 1) // 2
theta = draw_model(n, beta, rng)
p, _ = all_probabilities(n, theta)
draw = rng.choice(1 << n, size=2 * M, p=p)
rows, complete = [], []
for k, d in enumerate(draw):
    s = list(state_string(d, n))
    if mar:
        if rng.random() < (0.85 if sum(c == '1' for c in s[:5]) >= 3 else 0.15):
            for i in range(15, 20): s[i] = 'X'
    elif k >= M:
        for i in rng.choice(n, 5, replace=False): s[i] = 'X'
    rows.append("".join(s) + " 1.0")
    if 'X' not in s: complete.append(rows[-1])
with tempfile.TemporaryDirectory() as td:
    os.chdir(td)
    write_params('truth.dat', theta); write_data('all.dat', n, rows); write_data('complete.dat', n, complete)
    for name, fn, flags in (('complete rows only', 'complete.dat', ()), ('legacy', 'all.dat', ('--legacy',)),
                            ('marginal', 'all.dat', ('--marginal',)), ('frozen', 'all.dat', ('--frozen',))):
        best = None
        for lam in (-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0):
            f, status = fit_fixed(fn, lam, flags, tag=name.split()[0], nn=nn)
            k = kl(fn, 'truth.dat', f); th = np.loadtxt(f)
            slope = float(th[:nJ] @ theta[:nJ] / (theta[:nJ] @ theta[:nJ]))
            if best is None or k < best[0]: best = (k, lam, slope, status)
        print(f"{'MAR ' if mar else 'MCAR'} NN={nn} beta={beta} seed={seed} rows={len(complete)} complete + {len(rows) - len(complete)} partial  {name:20s} KL={best[0]:.5f} at log-lambda={best[1]:+.1f}  slope of J_fit on J_true={best[2]:.3f}  ({best[3]})", flush=True)
