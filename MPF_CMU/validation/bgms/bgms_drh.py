"""mpf and the R package bgms on the real DRH data.

Both programs are given the same file: one row per DRH entry (407 rows, 20 questions, weight 1), with an
answer coded missing where the entry's answers are missing or inconsistent. (The fits for the paper instead
expand an inconsistently coded entry into several weighted rows; bgms has no row weights.)

  fit                fit each method to all 407 entries (4 cores each); time, parameters, intervals
  FOLD _ _ cv        one fold of a 10-fold cross-validation: fit to nine tenths of the entries, and score the
                     held-out entries by the exact log-probability of their observed answers (summing over
                     the missing ones). (The argument order of every trial script, so that ../run_table.sh
                     can run it; the two arguments after the fold are ignored.)

Methods: mpf --frozen, mpf --marginal (lambda by leave-one-out cross-validation within the training data);
bgms with missing answers imputed, with edge selection (`select`, the default) and without (`full`); bgms
with its default of dropping incomplete entries (`listwise`). For bgms the held-out score is given for the
posterior-mean parameters and for the posterior predictive distribution (average over 100 draws).

Usage: BGMS_LIB=/path/to/R/library python3 bgms_drh.py fit [cores]  |  FOLD _ _ cv [cores]     (prints JSON; files go to ../results/bgms/drh)
"""
import os, shutil, sys, tempfile, time
import numpy as np
from bgms_common import bgms_fit, log_probs, predictive_log_probs, HERE
from common import drh_reference, write_data, fit_cv, emit

n = 20
OUT = os.path.join(HERE, '..', 'results', 'bgms', 'drh'); os.makedirs(OUT, exist_ok=True)
ref, q = drh_reference()
rows = []
for _, g in ref.groupby('entry_id', sort=True):
    v = g[q].values
    rows.append("".join(('1' if v[0, k] > 0 else '0') if (v[:, k] == v[0, k]).all() and v[0, k] != 0 else 'X' for k in range(n)))
rows = np.array(rows)

def heldout_logp(lq, row):
    """log p(observed answers of one entry): sum over the states that agree with them."""
    obs = [k for k in range(n) if row[k] != 'X']; mis = [k for k in range(n) if row[k] == 'X']
    base = sum(1 << k for k in obs if row[k] == '1')
    idx = np.array([base + sum(1 << mis[j] for j in range(len(mis)) if (c >> j) & 1) for c in range(1 << len(mis))])
    return float(np.logaddexp.reduce(lq[idx]))

def fit_all(train, cores, tag):
    """Fit every method to the given rows; returns {name: dict(theta, draws, secs, ...)}."""
    write_data('train.dat', n, [r + " 1.0" for r in train])
    res = {}
    for key, flags in (('mpf_frozen', ('--frozen',)), ('mpf_marginal', ('--marginal',))):
        t = time.time(); f, lam, cv = fit_cv('train.dat', flags, tag=key, threads=cores)
        res[key] = dict(theta=np.loadtxt(f).ravel(), secs=time.time() - t, lam=lam, **cv)
    for name, es, action in (('bgms_select', 1, 'impute'), ('bgms_full', 0, 'impute'), ('bgms_listwise', 1, 'listwise')):
        fit = bgms_fit('train.dat', 'train.' + name, es, action, 7, cores=cores)
        res[name] = dict(theta=fit['theta'], draws=fit['draws'], secs=float(fit['meta']['elapsed_seconds']), cpu=float(fit['meta']['cpu_seconds']),
                         max_rhat=float(fit['meta']['max_rhat']), divergences=fit['meta']['divergences'], rows=int(fit['meta']['rows']), lo=fit['lo'], hi=fit['hi'], inclusion=fit['inclusion'])
        if tag: shutil.copy('train.' + name + '_summary.tsv', os.path.join(OUT, f'{tag}_{name}_summary.tsv'))
    if tag:
        for k, v in res.items(): np.savetxt(os.path.join(OUT, f'{tag}_{k}_params.dat'), v['theta'][None, :], fmt='%.10e')
    return res

with tempfile.TemporaryDirectory() as td:
    os.chdir(td)
    if sys.argv[1] == 'fit':
        cores = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        write_data(os.path.join(OUT, 'drh_entries.dat'), n, [r + " 1.0" for r in rows])
        res = fit_all(rows, cores, 'all')
        out = dict(mode='fit', entries=len(rows), complete=int(sum('X' not in r for r in rows)), missing_answers=int(sum(r.count('X') for r in rows)),
                   max_missing=int(max(r.count('X') for r in rows)), cores=cores)
        for k, v in res.items():
            out[k] = {a: (v[a] if not isinstance(v[a], np.ndarray) else None) for a in v if a not in ('theta', 'draws', 'lo', 'hi', 'inclusion')}
        emit(out)
    else:
        fold = int(sys.argv[1]); cores = int(sys.argv[5]) if len(sys.argv) > 5 else 4
        perm = np.random.default_rng(2026).permutation(len(rows)); test = perm[fold::10]; train = np.setdiff1d(perm, test)
        res = fit_all(rows[train], cores, None)
        out = dict(mode='cv', tid=fold, n_test=len(test))
        for k, v in res.items():
            lq = log_probs(n, v['theta'])
            out[k] = dict(logp=[heldout_logp(lq, r) for r in rows[test]], secs=round(v['secs'], 1))
            if v.get('draws') is not None:
                lq = predictive_log_probs(n, v['draws'])
                out[k]['logp_predictive'] = [heldout_logp(lq, r) for r in rows[test]]
        out['test_complete'] = [bool('X' not in r) for r in rows[test]]
        emit(out)
