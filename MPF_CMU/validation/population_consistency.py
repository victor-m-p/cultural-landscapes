"""Infinite-data (population) check of the three treatments of missing data, by exact enumeration.
Independent of the C code: pure numpy/scipy.

Convention as in the paper: p(s) ~ exp(H(s)), H = sum_{a<b} J_ab s_a s_b + sum_a h_a s_a.
For an observation o with missing part c:

  legacy   : K = sum_o p*(o) sum_c p_theta(c|o) sum_{k observed} exp((H(o^k,c) - H(o,c))/2)   weights differentiated through
  frozen   : same expression, weights p_theta(c|o) frozen at the previous theta, iterated to a fixed point
  marginal : K = sum_o p*(o) sum_{k observed} sqrt( p_theta(o^k) / p_theta(o) )               MPF on the marginal

Usage: python3 population_consistency.py [n] [n_blank]      (defaults 6 2; n=8 takes a few minutes)
"""
import itertools, sys
import numpy as np
from scipy.optimize import minimize

def setup(n):
    N = 1 << n
    idx = np.arange(N)
    S = np.where((idx[:, None] >> np.arange(n)) & 1, 1.0, -1.0)
    iu = np.triu_indices(n, 1)
    F = np.concatenate([S[:, iu[0]] * S[:, iu[1]], S], axis=1)   # sufficient statistics
    return N, idx, F

def make_objectives(n, theta_true, masks, pis):
    N, idx, F = setup(n)
    H_true = F @ theta_true
    lp_true = H_true - np.logaddexp.reduce(H_true)
    p_true = np.exp(lp_true)
    info = []
    for B, pi in zip(masks, pis):
        bmask = sum(1 << b for b in B)
        gid = idx & ~bmask                                           # group = observed part
        pobs = np.bincount(gid, weights=p_true, minlength=N)[gid]    # p*(o) for each full state
        info.append((pi, gid, pobs, [k for k in range(n) if k not in B]))

    def K_legacy(theta, wtheta=None):
        H = F @ theta
        Hw = H if wtheta is None else F @ wtheta
        tot = 0.0
        for pi, gid, pobs, obs_bits in info:
            Z = np.bincount(gid, weights=np.exp(Hw), minlength=N)[gid]
            w = pobs * np.exp(Hw) / Z                                # p*(o) p_theta(c|o)
            g = np.zeros(N)
            for k in obs_bits:
                g += np.exp(0.5 * (H[idx ^ (1 << k)] - H))
            tot += pi * np.sum(w * g)
        return tot

    def K_marginal(theta):
        eH = np.exp(F @ theta)
        tot = 0.0
        for pi, gid, pobs, obs_bits in info:
            Zo = np.bincount(gid, weights=eH, minlength=N)
            reps = np.unique(gid)
            po = np.bincount(gid, weights=p_true, minlength=N)
            for k in obs_bits:
                tot += pi * np.sum(po[reps] * np.sqrt(Zo[reps ^ (1 << k)] / Zo[reps]))
        return tot

    def kl(theta):
        H = F @ theta
        return float(np.sum(p_true * (lp_true - (H - np.logaddexp.reduce(H)))))
    return K_legacy, K_marginal, kl

def numgrad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = eps
        g[i] = (f(x + e) - f(x - e)) / (2 * eps)
    return g

def run(n, beta, nblank, frac_complete, seed, do_frozen=True):
    rng = np.random.default_rng(seed)
    npar = n * (n + 1) // 2; nJ = n * (n - 1) // 2
    theta = rng.normal(0, beta, npar)
    masks = [()] + list(itertools.combinations(range(n), nblank))
    pis = [frac_complete] + [(1 - frac_complete) / (len(masks) - 1)] * (len(masks) - 1)
    K_legacy, K_marginal, kl = make_objectives(n, theta, masks, pis)
    opt = dict(method="BFGS", options=dict(gtol=1e-10, maxiter=5000))
    out = []
    fit = minimize(K_legacy, theta, jac=lambda t: numgrad(K_legacy, t), **opt).x
    out.append(("legacy", np.linalg.norm(numgrad(K_legacy, theta)), fit))
    if do_frozen:
        t = theta + rng.normal(0, 0.05, npar)                        # start away from the truth
        for _ in range(200):
            f = lambda x, w=t.copy(): K_legacy(x, wtheta=w)
            new = minimize(f, t, jac=lambda x: numgrad(f, x), **opt).x
            d = np.max(np.abs(new - t)); t = new
            if d < 1e-9: break
        out.append(("frozen", np.linalg.norm(numgrad(lambda x: K_legacy(x, wtheta=theta), theta)), t))
    fit = minimize(K_marginal, theta, jac=lambda t: numgrad(K_marginal, t), **opt).x
    out.append(("marginal", np.linalg.norm(numgrad(K_marginal, theta)), fit))
    return [(name, g, np.sqrt(np.mean((th - theta) ** 2)), np.mean(np.abs(th[:nJ])) / np.mean(np.abs(theta[:nJ])), kl(th)) for name, g, th in out]

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    nblank = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    print(f"n={n}; half the rows complete, half with {nblank} of {n} entries missing at random; infinite data")
    print(f"{'beta':>5} {'objective':<9} {'|grad K| at truth':>18} {'RMS param err':>14} {'mean|J_fit|/mean|J|':>20} {'KL(true||fit)':>14}")
    for beta in (0.1, 0.25, 0.5, 1.0):
        for seed in ((1, 2) if n <= 6 else (11,)):
            for name, g, rms, ratio, k in run(n, beta, nblank, 0.5, seed, do_frozen=(n <= 6)):
                print(f"{beta:5.2f} {name:<9} {g:18.2e} {rms:14.2e} {ratio:20.4f} {k:14.2e}", flush=True)
            print()
