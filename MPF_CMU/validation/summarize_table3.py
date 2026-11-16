"""Summarise the Table 3 re-run from its per-column result files (results/table3_*.jsonl).
Usage: python3 summarize_table3.py      (output also in results/table3_summary.txt)"""
import os
import numpy as np
from common import load_trials
here = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(here, 'results')
cols = {}
for f, keys in (('table3_complete_columns.jsonl', ('base', 'full', 'naive')), ('table3_legacy.jsonl', ('partial_legacy',)),
                ('table3_marginal.jsonl', ('partial_marginal',)), ('table3_frozen.jsonl', ('partial_frozen',))):
    for r in load_trials(os.path.join(R, f)):
        for k in keys: cols.setdefault(k, {})[r['tid']] = r[k]
names = ['dispersed', 'ordered', 'near-critical', 'critical']
label = dict(base='128 full', partial_legacy='published', partial_marginal='marginal', partial_frozen='frozen', naive='naive', full='256 full')
order = ['base', 'partial_legacy', 'partial_marginal', 'partial_frozen', 'naive', 'full']
se = lambda a: a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float('nan')
out = []
def say(s):
    print(s); out.append(s)
say("Table 3 protocol, exact sampling, lambda by cross-validation; mean KL from truth ± standard error (number of landscapes)")
say(f"{'regime':14s} " + " ".join(f"{label[k]:>20s}" for k in order))
for i, nm in enumerate(names):
    cells = []
    for k in order:
        a = np.array([v['kl'] for t, v in cols[k].items() if t % 4 == i])
        cells.append(f"{a.mean():7.3f} ± {se(a):.3f} ({len(a):2d})")
    say(f"{nm:14s} " + " ".join(f"{c:>20s}" for c in cells))
say("paired differences in KL (landscapes present in both columns)")
for a, b in (('partial_marginal', 'partial_legacy'), ('partial_frozen', 'partial_legacy'), ('partial_frozen', 'partial_marginal'), ('partial_legacy', 'base'), ('partial_legacy', 'full'), ('naive', 'base')):
    cells = []
    for i in range(4):
        t = [x for x in cols[a] if x in cols[b] and x % 4 == i]
        d = np.array([cols[a][x]['kl'] - cols[b][x]['kl'] for x in t])
        cells.append(f"{d.mean():+.4f} ± {se(d):.4f}")
    say(f"  {label[a]:>9s} minus {label[b]:<9s} " + "   ".join(cells))
say("fits with at least one warning (a cross-validation fold that did not converge at some lambda): " + ", ".join(f"{label[k]} {sum(v.get('warnings', 0) > 0 for v in cols[k].values())}/{len(cols[k])}" for k in order))
open(os.path.join(R, 'table3_summary.txt'), 'w').write("\n".join(out) + "\n")
