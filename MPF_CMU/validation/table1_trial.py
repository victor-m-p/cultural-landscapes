"""One trial of the paper's Table 1 (optimal lambda vs cross-validated lambda vs no regularisation), following
../paper_runs/new_cv_scan.rb: n=20, 128 observations, `mpf -o`. Prints one JSON line.
Usage: python3 table1_trial.py trial_id beta_lo beta_hi [mcmc] [nnK]     (mcmc: the published Monte Carlo sampler; nn2, nn3: neighbourhood)
"""
import os, re, sys, tempfile
from common import run, emit, table1_dataset, count_warnings

tid, lo, hi = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
mcmc = 'mcmc' in sys.argv[4:]
nn = next((int(a[2:]) for a in sys.argv[4:] if a.startswith('nn')), 1)      # e.g. nn2
with tempfile.TemporaryDirectory() as td:
    os.chdir(td)
    beta = table1_dataset(tid, lo, hi, mcmc)
    out, err = run(['-o', 't', nn, '--legacy'])
v = [float(x) for x in re.search(r'val=\[(.*)\]', out).group(1).split(',')]
unreg = re.search(r'Unregularised fit: (.*?) \(max \|gradient\| (\S+) after (\d+) iterations\)', out)
emit(dict(tid=tid, beta=beta, sampler='mcmc' if mcmc else 'exact', nn=nn, kl_opt=v[2], lam_opt=v[3], kl_cv=v[4], lam_cv=v[5], kl_unreg=v[6],
          unreg_status=unreg.group(1) if unreg else None, unreg_gmax=float(unreg.group(2)) if unreg else None, warnings=count_warnings(err)))
