"""Which treatment of missing data does best on data like the DRH?
Parametric bootstrap (common.DrhBootstrap): take a fitted DRH model as the truth, draw 407 entries from it
exactly, give them the real patterns of missing answers, fit with each method, and measure KL from the truth.

  mcar: each simulated entry gets the blank pattern of a randomly chosen real entry (unrelated to its answers)
  mar:  the blank pattern is drawn from real entries with the same answer to "Scriptures" (the answer most
        strongly related to missingness in the real data; see drh_missingness.py). "Scriptures" itself is
        then always observed, so the pattern depends only on an observed answer.

Each method is scored at its best lambda on a grid (an oracle choice; no cross-validation), starting from the
complete-rows fit at the same lambda, as the program's staged procedure does. For the chosen lambda the record
keeps how both fits ended (`status`: the complete-rows stage and the final fit) and the final residual, and,
per method, how many of the candidate fits did not converge (`not_converged`).

Needs numpy and pandas.  Usage: python3 drh_bootstrap.py replicate [ignored ...]     (prints one JSON line)
"""
import os, shutil, sys, tempfile
from common import DrhBootstrap, write_params, write_data, kl, run, fit_end, emit

rep = int(sys.argv[1])
boot = DrhBootstrap(); n = boot.n
draws, rows_of = boot.rows(rep)
grid = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
res = dict(tid=rep)
for mech in ('mcar', 'mar'):
    rows = rows_of[mech]
    complete = [r for r in rows if 'X' not in r]
    with tempfile.TemporaryDirectory() as td:
        os.chdir(td)
        write_params('truth.dat', boot.truth); write_data('all.dat', n, rows); write_data('complete.dat', n, complete)
        out = dict(n_complete=len(complete), scriptures_complete=sum(r[boot.SCRIPTURES] == '1' for r in complete) / len(complete),
                   scriptures_all=sum((int(d) >> boot.SCRIPTURES) & 1 for d in draws) / len(draws))
        stage1 = {}                                                               # the complete-rows fit at each lambda (convex), made once
        for lam in grid:
            stage1[lam] = (f'stage1_{lam}.dat', fit_end(run(['-l', 'complete.dat', lam, 1])[0]))
            shutil.copy('complete.dat_params.dat', stage1[lam][0])
        for key, flags in (('complete', None), ('legacy', ('--legacy',)), ('marginal', ('--marginal',)), ('frozen', ('--frozen',))):
            best = None; bad = 0
            for lam in grid:
                f, st1 = stage1[lam]; st2 = st1
                if flags is not None:
                    st2 = fit_end(run(['-l', 'all.dat', lam, 1, '--start', f, *flags])[0]); f = 'all.dat_params.dat'
                k = kl('all.dat', 'truth.dat', f)
                bad += (st1[0] != 'converged') + (flags is not None and st2[0] != 'converged')
                if best is None or k < best[0]: best = (k, lam, [st1[0], st2[0]], st2[1])
            out[key] = dict(kl=best[0], lam=best[1], status=best[2], residual=best[3], not_converged=bad)
    res[mech] = out
emit(res)
