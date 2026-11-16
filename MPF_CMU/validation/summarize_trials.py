"""Summarise the JSON lines written by run_trials.sh.   Usage: python3 summarize_trials.py results/file.jsonl"""
import json, sys
import numpy as np
from common import load_trials
R = load_trials(sys.argv[1])
names = ['dispersed', 'ordered', 'near-critical', 'critical']
keys = [k for k in ('base', 'partial_legacy', 'partial_marginal', 'partial_frozen', 'naive', 'full') if k in R[0]]
label = dict(base='128 full', partial_legacy='legacy', partial_marginal='marginal', partial_frozen='frozen', naive='naive', full='256 full')
print(f"{len(R)} trials; mean KL from truth ± standard error")
print(f"{'regime':14s} {'N':>3s} " + " ".join(f"{label[k]:>14s}" for k in keys) + "   marginal - legacy")
for i, nm in enumerate(names):
    r = [x for x in R if x['tid'] % 4 == i]
    if not r: continue
    m = {k: np.array([x[k]['kl'] for x in r]) for k in keys}
    d = m['partial_marginal'] - m['partial_legacy'] if 'partial_marginal' in m and 'partial_legacy' in m else np.array([float('nan')])
    se = lambda a: a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float('nan')
    print(f"{nm:14s} {len(r):3d} " + " ".join(f"{m[k].mean():8.3f}±{se(m[k]):.3f}" for k in keys) + f"   {d.mean():+.4f} ± {se(d):.4f}")
