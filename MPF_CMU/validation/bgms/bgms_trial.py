"""One trial of the comparison between mpf and the R package bgms, on data with a known truth.
Each protocol regenerates exactly the dataset of the corresponding mpf trial (same seeds), so the bgms
results pair with the stored mpf results by trial id.

  table1  TID   the paper's Table 1 setting: 20 nodes, 128 complete rows       (mpf: ../results/table1_nn1.jsonl)
  table3  TID   the paper's Table 3 setting: 128 complete rows + 128 rows with 5 of 20 answers blanked at
                random                                                          (mpf: ../results/table3_*.jsonl)
  drhboot REP   407 rows drawn from a fitted DRH model, with the real patterns of blanks assigned at random
                (mcar) or according to the simulated answer to "Scriptures" (mar); here mpf is also run,
                with cross-validation, because the stored bootstrap used an oracle lambda

Trial t of table1 / table3 is in coupling regime t mod 4 (dispersed, ordered, near-critical, critical).
bgms variants: `select` = package defaults (edge selection on); `full` = edge selection off; `matched` =
edge selection off with the prior standard deviation of the pairwise parameters set to that of the true
ones (an oracle choice, to show what tuning the prior could achieve). With missing answers bgms imputes
(`na_action = "impute"`); `listwise` is its default of dropping incomplete rows.

Usage: BGMS_LIB=/path/to/R/library python3 bgms_trial.py ID BETA_LO BETA_HI PROTOCOL [cores]     (prints one JSON line)
(the argument order of every trial script, so that ../run_table.sh can run it; drhboot ignores the two bounds)
"""
import os, sys, tempfile, time
import numpy as np
from bgms_common import bgms_fit, score, mpf_score, log_probs
from common import table1_dataset, table3_dataset, DrhBootstrap, write_data, fit_cv, emit

tid, lo, hi, protocol = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
cores = int(sys.argv[5]) if len(sys.argv) > 5 else 4
n = 20
res = dict(tid=tid, protocol=protocol)

def variants(datafile, theta, lp, missing, beta=None):
    out = {}
    na = 'impute' if missing else 'listwise'
    todo = [('select', 1, na, None), ('full', 0, na, None)]
    if beta is not None: todo.append(('matched', 0, na, 2 * beta))          # bgms's pairwise parameter is 2 J
    if missing: todo.append(('listwise', 1, 'listwise', None))
    for name, es, action, scale in todo:
        fit = bgms_fit(datafile, f'{datafile}.{name}', es, action, 100 + tid, cores=cores, scale=scale)
        out[name] = score(n, theta, lp, fit)
    return out

with tempfile.TemporaryDirectory() as td:
    os.chdir(td)
    if protocol == 'table1':
        beta = table1_dataset(tid, lo, hi)
        theta = np.loadtxt('t_params.dat').ravel()
        res.update(beta=beta, bgms=variants('t_data.dat', theta, log_probs(n, theta), False, beta))
    elif protocol == 'table3':
        beta, theta, lines, part = table3_dataset(tid, lo, hi, n)
        write_data('partial.dat', n, lines[:128] + part)
        res.update(beta=beta, bgms=variants('partial.dat', theta, log_probs(n, theta), True, beta))
    elif protocol == 'drhboot':
        boot = DrhBootstrap()
        lp = log_probs(n, boot.truth)
        for mech, rows in boot.rows(tid)[1].items():
            fn = f'{mech}.dat'; write_data(fn, n, rows)
            out = dict(n_complete=sum('X' not in r for r in rows), bgms=variants(fn, boot.truth, lp, True))
            for key, flags in (('frozen', ('--frozen',)), ('marginal', ('--marginal',)), ('legacy', ('--legacy',))):
                t = time.time(); f, lam, cv = fit_cv(fn, flags, tag=key, threads=cores)
                out['mpf_' + key] = dict(mpf_score(n, boot.truth, lp, f), lam=lam, secs=round(time.time() - t, 1), **cv)
            res[mech] = out
    else:
        sys.exit('unknown protocol')
emit(res)
