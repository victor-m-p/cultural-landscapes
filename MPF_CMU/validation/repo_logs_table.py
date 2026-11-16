"""Recompute the paper's Partial-MPF table from the simulation logs committed in ../NAN_TESTS."""
import ast, glob, os, statistics as st
rows = []
for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'NAN_TESTS', 'big_NAN_scan_experimental_20_5_*'))):
    m = [l for l in open(f).read().split('\n') if l.startswith('[[')]
    if m: rows += ast.literal_eval(m[-1])
print(f"{len(rows)} trials in the committed logs (n=20, 5 missing); mean KL from truth")
print(f"{'regime':15s} {'N':>4s} {'128 full':>9s} {'Partial':>9s} {'naive':>9s} {'256 full':>9s}")
for lo, hi, nm in ((0.01, 0.125, 'dispersed'), (0.125, 0.25, 'ordered'), (0.25, 0.5, 'near-critical'), (0.5, 1.0, 'critical')):
    s = [r for r in rows if lo < r[0] < hi and len(r) >= 5]
    f = lambda k: st.mean(r[k] for r in s)
    print(f"{nm:15s} {len(s):4d} {f(1):9.3f} {f(3):9.3f} {f(4):9.3f} {f(2):9.3f}")
