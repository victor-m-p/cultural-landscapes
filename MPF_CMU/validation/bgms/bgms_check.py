"""Check the conversion between bgms's parameters and mpf's: fit bgms to 20,000 exact draws from a known
6-node model and compare the converted posterior mean with the truth.
Usage: BGMS_LIB=/path/to/R/library python3 bgms_check.py"""
import os, tempfile
import numpy as np
from bgms_common import bgms_fit
from common import run
with tempfile.TemporaryDirectory() as td:
    os.chdir(td)
    run(['-g', 's', 6, 20000, 0.5, '--seed', 3])
    truth = np.loadtxt('s_params.dat').ravel()
    fit = bgms_fit('s_data.dat', 'chk', 0, 'listwise', 1, cores=4, draws=0)
    d = fit['theta'] - truth
    print(f"bgms {fit['meta']['bgms']}; 6 nodes, 20000 rows, no edge selection: converted posterior mean vs truth: max |J error| {np.abs(d[:15]).max():.3f}, max |h error| {np.abs(d[15:]).max():.3f}; "
          f"slope of fitted J on true J {np.polyfit(truth[:15], fit['theta'][:15], 1)[0]:.3f}; 95% intervals cover {np.mean((fit['lo'] <= truth) & (truth <= fit['hi'])):.2f} of the 21 parameters")
