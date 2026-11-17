"""Hidden nodes: does each treatment of missing data recover the truth when one node is never observed?

A hidden node is a column that is missing in every row. With infinite data from a known model, this
computes (by exact enumeration, independently of the C code, using population_consistency.py):

  * the gradient of each objective at the true parameters (zero if the truth is a stationary point), and
  * where an optimiser started at the truth ends up: whether it reports convergence, the gradient there, the
    KL divergence between the true and the fitted distribution of the VISIBLE nodes (the couplings to a
    hidden node are not identified individually, e.g. its sign is arbitrary, so the visible distribution is
    what can be compared), and how much the couplings to the hidden node grew.

The first is the evidence about bias and does not depend on any optimisation. The second shows the direction
and rough size of the effect; where the optimiser does not report convergence its end point is not a
minimum, and the legacy objective may have none at finite parameters (see the last column).

  legacy   : the published Partial-MPF objective (completion weights differentiated through)
  frozen   : the same flows with the weights held fixed (the estimating equation of `mpf --frozen`)
  marginal : MPF on the marginal distribution of the visible nodes (`mpf --marginal`, the default)

Usage: python3 hidden_node_check.py [n_visible]      (default 5; output in results/hidden_node.txt)
"""
import sys
import numpy as np
from scipy.optimize import minimize
from population_consistency import make_objectives, numgrad, setup

def visible_kl(n, theta_true, theta_fit):
    N, idx, F = setup(n)
    out = []
    for th in (theta_true, theta_fit):
        H = F @ th
        p = np.exp(H - np.logaddexp.reduce(H))
        out.append(np.bincount(idx & ~(1 << (n - 1)), weights=p, minlength=N)[: N >> 1])   # sum over the hidden (last) node
    return float(np.sum(out[0] * np.log(out[0] / out[1])))

nv = int(sys.argv[1]) if len(sys.argv) > 1 else 5
n = nv + 1; npar = n * (n + 1) // 2
iu = np.triu_indices(n, 1); to_hidden = (iu[1] == n - 1)          # couplings between a visible node and the hidden one
print(f"{nv} visible nodes + 1 hidden node (missing in every row); infinite data; optimiser (BFGS, numerical gradient) started at the truth")
print(f"{'beta':>5} {'seed':>4} {'objective':<9} {'|gradient| at truth':>20} {'optimiser':>14} {'|gradient| at end':>18} {'KL of visible distribution':>26} {'|J to hidden|: end / true':>26} {'largest |parameter| at end':>27}")
opt = dict(method="BFGS", options=dict(gtol=1e-8, maxiter=5000))
ended = lambda r: 'converged' if r.success else f'not converged ({r.status})'
for beta in (0.25, 0.5, 1.0):
    for seed in (1, 2):
        theta = np.random.default_rng(seed).normal(0, beta, npar)
        K_legacy, K_marginal, _ = make_objectives(n, theta, [(n - 1,)], [1.0])
        frozen = lambda x: K_legacy(x, wtheta=theta)                # weights frozen at the truth
        for name, K in (("legacy", K_legacy), ("frozen", frozen), ("marginal", K_marginal)):
            g = np.linalg.norm(numgrad(K, theta))
            if name == "frozen":
                t = theta.copy(); status = 'converged'              # iterate: freeze, minimise, re-freeze
                for _ in range(100):
                    f = lambda x, w=t.copy(): K_legacy(x, wtheta=w)
                    r = minimize(f, t, jac=lambda x: numgrad(f, x), **opt)
                    if not r.success: status = ended(r)
                    d = np.max(np.abs(r.x - t)); t = r.x
                    if d < 1e-9: break
                else: status = 'no fixed point'
                fit = t; gend = np.linalg.norm(numgrad(lambda x: K_legacy(x, wtheta=fit), fit))    # residual of the estimating equation
            else:
                r = minimize(K, theta, jac=lambda x: numgrad(K, x), **opt)
                fit = r.x; status = ended(r); gend = np.linalg.norm(numgrad(K, fit))
            ratio = np.mean(np.abs(fit[: len(to_hidden)][to_hidden])) / np.mean(np.abs(theta[: len(to_hidden)][to_hidden]))
            print(f"{beta:5.2f} {seed:4d} {name:<9} {g:20.2e} {status:>14} {gend:18.2e} {visible_kl(n, theta, fit):26.2e} {ratio:26.3f} {np.abs(fit).max():27.1f}", flush=True)
        r = minimize(K_legacy, theta, jac=lambda x: numgrad(K_legacy, x), **opt)
        with np.errstate(all='ignore'):                             # far along the ray the reference's raw exponentials overflow
            ray = [K_legacy(theta + c * (r.x - theta)) for c in (0.0, 1.0, 2.0, 4.0)]
        show = lambda v: f"{v:.6f}" if np.isfinite(v) else "(overflows)"
        print(f"{'':10} legacy objective at the truth {show(ray[0])}, at the optimiser's end point {show(ray[1])}, and 2 and 4 times as far along the same direction {show(ray[2])}, {show(ray[3])}")
        print()
