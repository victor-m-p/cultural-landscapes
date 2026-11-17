"""Time `mpf -c` on the Table 1 datasets with the same number of cores bgms is given, so that the two can be
compared like for like (the stored Table 1 results did not record times).
Usage: python3 mpf_timing.py ID BETA_LO BETA_HI [cores]      (prints one JSON line)"""
import os, sys, tempfile, time
import numpy as np
from bgms_common import mpf_score, log_probs
from common import table1_dataset, fit_cv, emit
tid, lo, hi = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]); cores = int(sys.argv[4]) if len(sys.argv) > 4 else 4
with tempfile.TemporaryDirectory() as td:
    os.chdir(td)
    beta = table1_dataset(tid, lo, hi)
    theta = np.loadtxt('t_params.dat').ravel()
    t = time.time(); f, lam, cv = fit_cv('t_data.dat', (), tag='cv', threads=cores); secs = time.time() - t
    emit(dict(tid=tid, beta=beta, cores=cores, secs=round(secs, 2), lam=lam, **mpf_score(20, theta, log_probs(20, theta), f), **cv))
