"""Compare the DRH refits in results/drh_refits with the parameter file used in the paper. Everything in the
table, including the chosen lambda, is read from the named log and parameter files."""
import os, re
import numpy as np
here = os.path.dirname(os.path.abspath(__file__)); n = 20; nJ = 190
com = np.loadtxt(os.path.join(here, '..', '..', 'data', 'mdl_experiments', 'matrix_questions_20_maxna_5_nrows_455_entries_407.txt.mpf_params.dat'))
fits = [('committed (used in the paper)', com, None, None)]
for key, label in (('legacy', 'refit, --legacy'), ('marginal', 'refit, marginal'), ('marginal_entrycv', 'refit, marginal + entry-wise CV'), ('frozen_entrycv', 'refit, frozen + entry-wise CV'),
                   ('frozen_entrycv_nn2', 'refit, frozen + entry CV, NN=2'), ('frozen_entrycv_nn3', 'refit, frozen + entry CV, NN=3'), ('legacy_nn2', 'refit, --legacy, NN=2')):
    base = os.path.join(here, 'results', 'drh_refits', f'drh_{key}')
    if not os.path.exists(base + '.log'): continue
    log = open(base + '.log').read()
    fits.append((label, np.loadtxt(base + '.mpf_params.dat'), float(re.findall(r'Best log_sparsity: (\S+)', log)[-1]), float(re.search(r'Wall time: (\S+)', log).group(1))))
idx = np.arange(1 << n)
S = np.where((idx[:, None] >> np.arange(n - 1, -1, -1)) & 1, 1.0, -1.0)       # the analysis scripts' state ordering
def probs(par):
    J = np.zeros((n, n)); J[np.triu_indices(n, 1)] = par[:nJ]
    H = np.einsum('ij,ij->i', S @ J, S) + S @ par[nJ:]
    q = np.exp(H - H.max()); return q / q.sum()
def edges(top):                                # single-flip edges among a set of configurations
    s = set(int(t) for t in top)
    return sum(1 for t in s for b in range(n) if (t ^ (1 << b)) in s) // 2
pc = probs(com)
spartans = [765847, 765831, 765843, 765827]     # +ritual +sacrifice, +ritual -sacrifice, -ritual +sacrifice, -ritual -sacrifice
top15 = set(np.argsort(-np.abs(com[:nJ]))[:15]); top150 = set(np.argsort(-pc)[:150])
print(f"{'fit':32s} {'log10 lambda':>12s} {'seconds':>8s} {'max|dtheta|':>11s} {'corr(J)':>8s} {'top-15 J':>9s} {'top-150 configs':>15s} {'edges':>6s} {'mass top 150':>12s} {'KL(paper||fit)':>14s}   Spartans (%)")
for label, par, lam, secs in fits:
    p = probs(par); s = 100 * p[spartans] / p[spartans].sum(); t150 = np.argsort(-p)[:150]
    print(f"{label:32s} {'' if lam is None else f'{lam:.6f}':>12s} {'' if secs is None else f'{secs:.0f}':>8s} {np.abs(par - com).max():11.2e} {np.corrcoef(par[:nJ], com[:nJ])[0, 1]:8.4f} "
          f"{len(set(np.argsort(-np.abs(par[:nJ]))[:15]) & top15):6d}/15 {len(set(t150) & top150):11d}/150 {edges(t150):6d} {np.sort(p)[-150:].sum():12.3f} {float(np.sum(pc * (np.log(pc) - np.log(p)))):14.4f}   {s[0]:.1f} / {s[1]:.1f} / {s[2]:.1f} / {s[3]:.1f}")
