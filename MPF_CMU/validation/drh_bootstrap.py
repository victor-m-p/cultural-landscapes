"""Which treatment of missing data does best on data like the DRH?
Parametric bootstrap: take a fitted DRH model as the truth, draw 407 entries from it exactly, give them the
real patterns of missing answers, fit with each method, and measure KL from the truth.

  mcar: each simulated entry gets the blank pattern of a randomly chosen real entry (unrelated to its answers)
  mar:  the blank pattern is drawn from real entries with the same answer to "Scriptures" (the answer most
        strongly related to missingness in the real data; see drh_missingness.py). "Scriptures" itself is
        then always observed, so the pattern depends only on an observed answer.

Each method is scored at its best lambda on a grid (an oracle choice; no cross-validation), starting from the
complete-rows fit as the program's staged procedure does.

Needs numpy and pandas.  Usage: python3 drh_bootstrap.py first_rep last_rep        (appends JSON lines to stdout)
"""
import json, os, sys, tempfile
import numpy as np, pandas as pd
from common import all_probabilities, state_string, write_params, write_data, kl, run, source_tag

here = os.path.dirname(os.path.abspath(__file__)); repo = os.path.join(here, '..', '..'); n = 20
truth = np.loadtxt(os.path.join(here, 'results', 'drh_refits', 'drh_frozen_entrycv.mpf_params.dat'))
ref = pd.read_csv(os.path.join(repo, 'data', 'reference', 'direct_reference_questions_20_maxna_5_nrows_455_entries_407.csv'))
q = [c for c in ref.columns if c not in ('entry_id', 'weight')]
first = ref.groupby('entry_id')[q].first()
masks = (first.values == 0)                                  # 407 real blank patterns
SCRIPT = 1                                                   # column of "Scriptures" (question 4729)
scr_obs = ~masks[:, SCRIPT]
scr_yes = ref.groupby('entry_id')[q[SCRIPT]].max().values > 0   # an entry counts as "yes" if any of its rows says yes
pool = {True: masks[scr_obs & scr_yes], False: masks[scr_obs & ~scr_yes]}
p, _ = all_probabilities(n, truth)
grid = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
for rep in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
    rng = np.random.default_rng(90000 + rep)
    draws = rng.choice(1 << n, size=407, p=p)
    res = dict(tid=rep, source=source_tag())
    for mech in ('mcar', 'mar'):
        rows = []
        for d in draws:
            s = list(state_string(d, n))
            if mech == 'mcar': m = masks[rng.integers(len(masks))]
            else:
                pl = pool[s[SCRIPT] == '1']; m = pl[rng.integers(len(pl))]
            for i in np.flatnonzero(m): s[i] = 'X'
            rows.append("".join(s) + " 1.0")
        complete = [r for r in rows if 'X' not in r]
        with tempfile.TemporaryDirectory() as td:
            os.chdir(td)
            write_params('truth.dat', truth); write_data('all.dat', n, rows); write_data('complete.dat', n, complete)
            out = dict(n_complete=len(complete), scriptures_complete=float(np.mean([r[SCRIPT] == '1' for r in complete])), scriptures_all=float(np.mean([state_string(d, n)[SCRIPT] == '1' for d in draws])))
            for key, fn, flags in (('complete', 'complete.dat', ()), ('legacy', 'all.dat', ('--legacy',)), ('marginal', 'all.dat', ('--marginal',)), ('frozen', 'all.dat', ('--frozen',))):
                best = None
                for lam in grid:
                    run(['-l', 'complete.dat', lam, 1])                           # stage 1: complete rows (convex)
                    start = 'complete.dat_params.dat'
                    if key == 'complete': f = start
                    else:
                        run(['-l', fn, lam, 1, '--start', start, *flags]); f = fn + '_params.dat'
                    k = kl(fn, 'truth.dat', f)
                    if best is None or k < best[0]: best = (k, lam)
                out[key] = dict(kl=best[0], lam=best[1])
        res[mech] = out
    print(json.dumps(res), flush=True)
