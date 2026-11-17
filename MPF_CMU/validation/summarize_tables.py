"""Summarise the Table 1 and Table 2 re-runs.   Usage: python3 summarize_tables.py"""
import os
from common import load_trials, REGIME_NAMES as names, se
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
def load(f):
    p = os.path.join(here, 'results', f)
    return load_trials(p) if os.path.exists(p) else []
print("Table 1 (n=20, 128 observations): mean KL from truth ± standard error; paper: optimal / CV / standard MPF = 0.22/0.23/1.2, 0.55/0.56/2.3, 0.62/0.63/19.4, 0.50/0.54/9.5")
for f, label in (('table1_mcmc.jsonl', 'published Monte Carlo sampler'), ('table1_exact.jsonl', 'exact sampling')):
    R = load(f)
    if not R: continue
    print(f"  {label} ({len(R)} trials)")
    print(f"    {'regime':14s} {'N':>3s} {'optimal':>13s} {'CV':>13s} {'unregularised (mean)':>21s} {'(median)':>9s} {'CV - optimal':>14s}")
    for i, nm in enumerate(names):
        r = [x for x in R if x['tid'] % 4 == i]
        o = np.array([x['kl_opt'] for x in r]); c = np.array([x['kl_cv'] for x in r]); u = np.array([x['kl_unreg'] for x in r])
        print(f"    {nm:14s} {len(r):3d} {o.mean():7.3f}±{se(o):.3f} {c.mean():7.3f}±{se(c):.3f} {u.mean():21.2f} {np.median(u):9.2f} {(c - o).mean():+8.3f}±{se(c - o):.3f}")
R = load('table2_exact.jsonl')
if R:
    print(f"Table 2 (re-weighting; exact sampling; {len(R)} trials): paper: ideal / corrected / naive KL = 0.13/0.16/0.16, 0.34/0.45/0.46, 0.43/0.55/0.60, 0.48/0.57/0.71; bias corrected / naive = -0.2%/-14%, -0.1%/-40%, 9.6%/-51%, 0.1%/-65%")
    print(f"    {'regime':14s} {'N':>3s} {'ideal KL':>9s} {'corrected':>10s} {'naive':>8s} {'bias corrected':>15s} {'bias naive':>11s}")
    edges = [(0.01, 0.125), (0.125, 0.25), (0.25, 0.5), (0.5, 1.02)]
    for (lo, hi), nm in zip(edges, names):
        r = [x for x in R if lo <= x['beta'] < hi]
        if not r: continue
        m = lambda k, q: np.mean([x[k][q] for x in r])
        print(f"    {nm:14s} {len(r):3d} {m('ideal','kl'):9.2f} {m('corrected','kl'):10.2f} {m('naive','kl'):8.2f} {100 * (np.exp(m('corrected','log_bias')) - 1):14.1f}% {100 * (np.exp(m('naive','log_bias')) - 1):10.1f}%")
