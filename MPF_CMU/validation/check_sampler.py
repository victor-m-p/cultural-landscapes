"""Does `mpf -g` draw from the exact Boltzmann distribution?  Compares the sample mean of H and the frequency
of the ten most probable states with their exact values, for the Monte Carlo sampler (--mcmc, what the
published simulations used) and for the exact sampler (the new default for n <= 24).

Usage: python3 check_sampler.py beta n_samples seed
"""
import os, sys, tempfile
import numpy as np
from common import all_probabilities, run

n = 20; beta = float(sys.argv[1]); m = int(sys.argv[2]); seed = sys.argv[3]
with tempfile.TemporaryDirectory() as td:
    os.chdir(td)
    for label, flags in (('Monte Carlo', ['--mcmc']), ('exact', [])):
        run(['-g', 's', n, m, beta, '--seed', seed, *flags])
        theta = np.loadtxt('s_params.dat')
        p, H = all_probabilities(n, theta)
        EH = float(p @ H); VH = float(p @ (H - EH) ** 2)
        ids = np.array([int(l.split()[0][::-1], 2) for l in open('s_data.dat').read().split('\n')[2:] if l])
        z = (H[ids].mean() - EH) / np.sqrt(VH / len(ids))
        top = np.argsort(p)[-10:]; ptop = p[top].sum(); ftop = np.isin(ids, top).mean()
        ztop = (ftop - ptop) / np.sqrt(ptop * (1 - ptop) / len(ids))
        print(f"beta={beta:.2f} seed={seed} {label:11s} <H> exact={EH:8.3f} sample={H[ids].mean():8.3f} z={z:+6.1f}   top-10 states: exact mass={ptop:.4f} sample={ftop:.4f} z={ztop:+6.1f}")
