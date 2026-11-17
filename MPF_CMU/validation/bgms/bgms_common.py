"""Helpers for the comparison with the R package bgms."""
import os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))

def all_H(n, theta):
    """H(s) for all 2^n states in O(2^n) (bit i of the state index is node i, as in the C code)."""
    nJ = n * (n - 1) // 2
    J = np.zeros((n, n)); J[np.triu_indices(n, 1)] = theta[:nJ]; h = theta[nJ:]
    H = np.zeros(1)
    for k in range(n):
        phi = np.zeros(1)                       # sum_{a<k} J[a,k] s_a over the 2^k states of nodes 0..k-1
        for a in range(k):
            phi = np.concatenate([phi - J[a, k], phi + J[a, k]])
        f = h[k] + phi
        H = np.concatenate([H - f, H + f])
    return H

def log_probs(n, theta):
    H = all_H(n, theta)
    return H - np.logaddexp.reduce(H)

def kl_from(lp_true, lq):
    """KL(true || q) in nats, from log-probabilities over all states."""
    p = np.exp(lp_true)
    return float(np.sum(p * (lp_true - lq)))

def predictive_log_probs(n, draws):
    """Log of the posterior predictive distribution over all states: the average of the draws' distributions."""
    acc = None
    for d in draws:
        lq = log_probs(n, d)
        acc = lq if acc is None else np.logaddexp(acc, lq)
    return acc - np.log(len(draws))

def read_summary(path):
    """The per-parameter table written by bgms_fit.R: (2.5% quantiles, 97.5% quantiles, inclusion probabilities), in
    the order of the parameter file (couplings, then fields); NaN where bgms gave none."""
    rows = [l.split('\t') for l in open(path).read().strip().split('\n')[1:]]
    num = lambda v: float('nan') if v in ('NA', 'NaN') else float(v)
    return tuple(np.array([num(r[k]) for r in rows]) for k in (4, 5, 6))

def bgms_fit(datafile, prefix, edge_selection, na_action, seed, cores=4, scale=None, draws=100):
    """Run bgms_fit.R; returns a dict with the parameter vector, summary table, metadata and draws."""
    cmd = ['Rscript', os.path.join(HERE, 'bgms_fit.R'), datafile, prefix, str(int(edge_selection)), na_action, str(seed), f'cores={cores}', f'draws={draws}']
    if scale is not None: cmd.append(f'scale={scale}')
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"bgms failed: {r.stderr[-2000:]}")
    meta = dict(l.split(' ', 1) for l in open(prefix + '_meta.txt').read().strip().split('\n'))
    lo, hi, inclusion = read_summary(prefix + '_summary.tsv')
    return dict(theta=np.loadtxt(prefix + '_params.dat').ravel(), lo=lo, hi=hi, inclusion=inclusion,
                draws=np.atleast_2d(np.loadtxt(prefix + '_draws.txt')) if draws else None, meta=meta)

def score(n, theta_true, lp_true, fit):
    """Accuracy of a bgms fit against a known truth: KL of the posterior-mean model, KL of the posterior
    predictive distribution (average over draws), and coverage / width of the 95% intervals."""
    nJ = n * (n - 1) // 2
    out = dict(kl_mean=kl_from(lp_true, log_probs(n, fit['theta'])))
    if fit['draws'] is not None:
        out['kl_predictive'] = kl_from(lp_true, predictive_log_probs(n, fit['draws']))
    ok = ~np.isnan(fit['lo'])
    inside = (fit['lo'] <= theta_true) & (theta_true <= fit['hi'])
    isJ = np.arange(len(theta_true)) < nJ
    frac = lambda m: float(inside[m].mean()) if m.any() else float('nan')       # no intervals if bgms could not be run
    out.update(cover_J=frac(ok & isJ), cover_h=frac(ok & ~isJ),
               width_J=float((fit['hi'] - fit['lo'])[ok & isJ].mean()) if (ok & isJ).any() else float('nan'), rmse_J=float(np.sqrt(np.mean((fit['theta'] - theta_true)[isJ] ** 2))),
               secs=float(fit['meta']['elapsed_seconds']), cpu=float(fit['meta']['cpu_seconds']), max_rhat=float(fit['meta']['max_rhat']),
               divergences=fit['meta']['divergences'], constant_columns=int(fit['meta']['constant_columns']), rows=int(fit['meta']['rows']))
    return out

def mpf_score(n, theta_true, lp_true, paramfile):
    th = np.loadtxt(paramfile).ravel(); nJ = n * (n - 1) // 2
    return dict(kl=kl_from(lp_true, log_probs(n, th)), rmse_J=float(np.sqrt(np.mean((th - theta_true)[:nJ] ** 2))))
