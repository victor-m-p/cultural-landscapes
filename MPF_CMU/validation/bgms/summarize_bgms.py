"""Summarise the comparison between mpf and bgms (results/bgms/*).   Usage: python3 summarize_bgms.py
Output also in ../results/bgms/summary.txt."""
import json, os
import numpy as np
from bgms_common import log_probs, read_summary, HERE
from common import load_trials, REGIME_NAMES as names, se, DRH_PUBLISHED_PARAMS
R = os.path.join(HERE, '..', 'results'); B = os.path.join(R, 'bgms')
out = []
def say(s=''):
    print(s); out.append(s)
load = lambda path: {r['tid']: r for r in load_trials(path)}
ms = lambda a: f"{np.mean(a):.3f} ± {se(a):.3f}"
def paired(a, b):
    d = np.array(a) - np.array(b); return f"{d.mean():+.3f} ± {se(d):.3f}"

def block(title, T, mpf_cols, variants, group=lambda t: t % 4, group_names=names):
    """T: tid -> bgms dict; mpf_cols: label -> {tid: kl}."""
    say(title)
    labels = list(mpf_cols) + [f'bgms {v}' for v in variants] + [f'bgms {v}, predictive' for v in variants]
    say(f"  {'':14s} {'n':>3s} " + " ".join(f"{l:>26s}" for l in labels))
    for gi, gname in enumerate(group_names):
        tids = sorted(t for t in T if group(t) == gi and all(t in c for c in mpf_cols.values()))
        if not tids: continue
        cells = [ms([c[t] for t in tids]) for c in mpf_cols.values()]
        cells += [ms([T[t][v]['kl_mean'] for t in tids]) for v in variants] + [ms([T[t][v]['kl_predictive'] for t in tids]) for v in variants]
        say(f"  {gname:14s} {len(tids):3d} " + " ".join(f"{c:>26s}" for c in cells))
    return labels

if os.path.exists(os.path.join(B, 'table1.jsonl')):
    T = load(os.path.join(B, 'table1.jsonl')); M = load(os.path.join(R, 'table1_nn1.jsonl'))
    Tb = {t: r['bgms'] for t, r in T.items()}
    NN = {k: load(os.path.join(R, f'table1_nn{k}.jsonl')) for k in (2, 3)}        # the same landscapes with larger neighbourhoods
    block("1. Complete data (the paper's Table 1 setting: 20 nodes, 128 rows). KL from the truth, mean ± standard error. (mpf with single-flip neighbours unless N2 or N3 is given.)",
          Tb, {'mpf, cross-validated': {t: M[t]['kl_cv'] for t in M}, 'mpf, best possible lambda': {t: M[t]['kl_opt'] for t in M},
               'mpf N2, cross-validated': {t: NN[2][t]['kl_cv'] for t in NN[2]}, 'mpf N3, cross-validated': {t: NN[3][t]['kl_cv'] for t in NN[3]}}, ['select', 'full', 'matched'])
    for k in (2, 3):
        say(f"  paired difference in KL, bgms select (predictive) minus cross-validated mpf N{k}:   " + "   ".join(f"{names[i]}: {paired([Tb[t]['select']['kl_predictive'] for t in sorted(Tb) if t % 4 == i], [NN[k][t]['kl_cv'] for t in sorted(Tb) if t % 4 == i])}" for i in range(4)))
    say("  paired differences in KL, bgms minus cross-validated mpf with single-flip neighbours (negative: bgms closer to the truth)")
    for v, key in (('select', 'kl_mean'), ('select', 'kl_predictive'), ('full', 'kl_predictive'), ('matched', 'kl_predictive')):
        say(f"    {'bgms ' + v + (', predictive' if key == 'kl_predictive' else ''):28s} " + "   ".join(f"{names[i]}: {paired([Tb[t][v][key] for t in sorted(Tb) if t % 4 == i], [M[t]['kl_cv'] for t in sorted(Tb) if t % 4 == i])}" for i in range(4)))
    say("  bgms 95% credible intervals: fraction that contain the true value (couplings / fields), and mean width for couplings")
    for v in ('select', 'full', 'matched'):
        say(f"    {v:10s} " + "   ".join(f"{names[i]}: {np.nanmean([Tb[t][v]['cover_J'] for t in Tb if t % 4 == i]):.2f} / {np.nanmean([Tb[t][v]['cover_h'] for t in Tb if t % 4 == i]):.2f}, width {np.nanmean([Tb[t][v]['width_J'] for t in Tb if t % 4 == i]):.2f}" for i in range(4)))
    say(f"  bgms wall time per fit on 4 cores: select {np.median([Tb[t]['select']['secs'] for t in Tb]):.1f} s, full {np.median([Tb[t]['full']['secs'] for t in Tb]):.1f} s (medians); "
        f"largest R-hat {np.nanmax([Tb[t][v]['max_rhat'] for t in Tb for v in Tb[t]]):.3f}; fits with divergent transitions: {sum(Tb[t][v]['divergences'] not in ('0', 'NA') for t in Tb for v in Tb[t])} of {sum(len(Tb[t]) for t in Tb)}; "
        f"datasets with a question answered identically by every row: {sum(Tb[t]['select']['constant_columns'] > 0 for t in Tb)} of {len(Tb)}")
    if os.path.exists(os.path.join(B, 'mpf_timing.jsonl')):
        P = load(os.path.join(B, 'mpf_timing.jsonl'))
        if P: say(f"  mpf -c wall time on 4 cores, same datasets ({len(P)} of them): median {np.median([P[t]['secs'] for t in P]):.1f} s, range {min(P[t]['secs'] for t in P):.1f}-{max(P[t]['secs'] for t in P):.1f} s")
    say()

if os.path.exists(os.path.join(B, 'table3.jsonl')):
    T = load(os.path.join(B, 'table3.jsonl')); Tb = {t: r['bgms'] for t, r in T.items()}
    cols = {}
    for f, key, label in (('table3_complete_columns.jsonl', 'base', 'mpf, 128 complete rows only'), ('table3_legacy.jsonl', 'partial_legacy', 'mpf published'),
                          ('table3_marginal.jsonl', 'partial_marginal', 'mpf marginal'), ('table3_frozen.jsonl', 'partial_frozen', 'mpf frozen')):
        L = load(os.path.join(R, f)); cols[label] = {t: L[t][key]['kl'] for t in L}
    block("2. Missing data (the paper's Table 3 setting: 128 complete rows + 128 rows with 5 of 20 answers blanked at random). bgms imputes, except `listwise`.",
          Tb, cols, ['select', 'full', 'matched', 'listwise'])
    say("  paired differences in KL, bgms minus mpf marginal (cross-validated)")
    for v, key in (('select', 'kl_predictive'), ('full', 'kl_predictive'), ('matched', 'kl_predictive'), ('listwise', 'kl_predictive')):
        say(f"    {'bgms ' + v + ', predictive':28s} " + "   ".join(f"{names[i]}: {paired([Tb[t][v][key] for t in sorted(Tb) if t % 4 == i], [cols['mpf marginal'][t] for t in sorted(Tb) if t % 4 == i])}" for i in range(4)))
    say("  bgms 95% credible intervals: coverage (couplings / fields)")
    for v in ('select', 'full'):
        say(f"    {v:10s} " + "   ".join(f"{names[i]}: {np.nanmean([Tb[t][v]['cover_J'] for t in Tb if t % 4 == i]):.2f} / {np.nanmean([Tb[t][v]['cover_h'] for t in Tb if t % 4 == i]):.2f}" for i in range(4)))
    say(f"  bgms wall time per fit on 4 cores (256 rows, imputing): select {np.median([Tb[t]['select']['secs'] for t in Tb]):.1f} s, full {np.median([Tb[t]['full']['secs'] for t in Tb]):.1f} s")
    say()

if os.path.exists(os.path.join(B, 'drhboot.jsonl')):
    T = load(os.path.join(B, 'drhboot.jsonl'))
    say(f"3. Data like the DRH ({len(T)} datasets of 407 rows drawn from a fitted DRH model, with the real patterns of blanks). KL from the truth.")
    for mech, desc in (('mcar', 'blanks assigned at random'), ('mar', 'blanks depend on the answer to "Scriptures"')):
        say(f"  {desc}  (complete rows per dataset: {np.mean([T[t][mech]['n_complete'] for t in T]):.0f})")
        for key, label in (('mpf_frozen', 'mpf --frozen, cross-validated'), ('mpf_marginal', 'mpf --marginal, cross-validated'), ('mpf_legacy', 'mpf --legacy, cross-validated')):
            say(f"    {label:38s} {ms([T[t][mech][key]['kl'] for t in T])}    median time on 4 cores {np.median([T[t][mech][key]['secs'] for t in T]):.0f} s")
        for v in ('select', 'full', 'listwise'):
            b = [T[t][mech]['bgms'][v] for t in T]
            say(f"    {'bgms ' + v + (' (imputing)' if v != 'listwise' else ' (complete rows only)'):38s} {ms([x['kl_mean'] for x in b])}    predictive {ms([x['kl_predictive'] for x in b])}    "
                f"median time {np.median([x['secs'] for x in b]):.0f} s    coverage {np.nanmean([x['cover_J'] for x in b]):.2f} / {np.nanmean([x['cover_h'] for x in b]):.2f}")
        say(f"    paired, bgms select predictive minus mpf --frozen: {paired([T[t][mech]['bgms']['select']['kl_predictive'] for t in T], [T[t][mech]['mpf_frozen']['kl'] for t in T])};  "
            f"bgms full predictive minus mpf --frozen: {paired([T[t][mech]['bgms']['full']['kl_predictive'] for t in T], [T[t][mech]['mpf_frozen']['kl'] for t in T])}")
    say()

fitp = os.path.join(B, 'drh_fit.json')
if os.path.exists(fitp) and os.path.getsize(fitp) > 0:
    F = json.loads(open(fitp).read().strip().split('\n')[-1]); D = os.path.join(B, 'drh')
    say(f"4. The real DRH data, one row per entry ({F['entries']} entries, {F['complete']} complete, {F['missing_answers']} missing answers, at most {F['max_missing']} per entry); {F['cores']} cores each.")
    P = {k: np.loadtxt(os.path.join(D, f'all_{k}_params.dat')).ravel() for k in ('mpf_frozen', 'mpf_marginal', 'bgms_select', 'bgms_full', 'bgms_listwise')}
    P['published'] = np.loadtxt(DRH_PUBLISHED_PARAMS).ravel()
    lp = {k: log_probs(20, v) for k, v in P.items()}; top = {k: set(np.argsort(-v)[:150]) for k, v in lp.items()}
    say(f"  {'':16s} {'wall time':>10s} {'cpu time':>9s} {'corr. of couplings with mpf --frozen':>38s} {'15 strongest shared':>20s} {'top 150 configurations shared':>30s} {'mass of top 150':>16s}")
    ref = P['mpf_frozen']; strongest = lambda v: set(np.argsort(-np.abs(v[:190]))[:15])
    for k in ('mpf_frozen', 'mpf_marginal', 'bgms_select', 'bgms_full', 'bgms_listwise', 'published'):
        t = f"{F[k]['secs']:.0f} s" if k in F else ''; c = f"{F[k]['cpu']:.0f} s" if k in F and 'cpu' in F[k] else ''
        say(f"  {k:16s} {t:>10s} {c:>9s} {np.corrcoef(P[k][:190], ref[:190])[0, 1]:38.4f} {len(strongest(P[k]) & strongest(ref)):20d} {len(top[k] & top['mpf_frozen']):30d} {np.exp(np.sort(lp[k])[-150:]).sum():16.3f}")
    say(f"  mpf chose log10 lambda {F['mpf_frozen']['lam']:.3f} (frozen) and {F['mpf_marginal']['lam']:.3f} (marginal); bgms listwise used {F['bgms_listwise']['rows']} entries; largest R-hat {max(F[k]['max_rhat'] for k in F if k.startswith('bgms')):.3f}")
    for k in ('bgms_select', 'bgms_full'):
        lo, hi, inc = (v[:190] for v in read_summary(os.path.join(D, f'all_{k}_summary.tsv')))
        extra = ''
        if k == 'bgms_select':
            extra = f"; edges with posterior inclusion probability above 0.5: {int((inc > 0.5).sum())} of 190, above 0.9: {int((inc > 0.9).sum())}"
        say(f"  {k}: mpf --frozen's coupling lies inside the 95% interval for {int(((lo <= ref[:190]) & (ref[:190] <= hi)).sum())} of 190 couplings; RMS difference of couplings {np.sqrt(np.mean((P[k][:190] - ref[:190]) ** 2)):.3f} (RMS coupling {np.sqrt(np.mean(ref[:190] ** 2)):.3f}){extra}")
    cvp = os.path.join(B, 'drh_cv.jsonl')
    if os.path.exists(cvp):
        C = load(cvp)
        say(f"  10-fold cross-validation ({len(C)} folds done): mean log-probability of a held-out entry's observed answers (higher is better), and the paired difference from mpf --frozen")
        cat = lambda key, sub='logp': np.concatenate([C[t][key][sub] for t in sorted(C)])
        base = cat('mpf_frozen'); comp = np.concatenate([C[t]['test_complete'] for t in sorted(C)])
        for key, sub, label in (('mpf_frozen', 'logp', 'mpf --frozen'), ('mpf_marginal', 'logp', 'mpf --marginal'), ('bgms_select', 'logp', 'bgms select, posterior mean'), ('bgms_select', 'logp_predictive', 'bgms select, predictive'),
                                ('bgms_full', 'logp', 'bgms full, posterior mean'), ('bgms_full', 'logp_predictive', 'bgms full, predictive'), ('bgms_listwise', 'logp_predictive', 'bgms listwise, predictive')):
            v = cat(key, sub)
            say(f"    {label:30s} {v.mean():8.3f}   difference {paired(v, base)}   (complete entries only: {v[comp].mean():.3f}; entries with missing answers: {v[~comp].mean():.3f})")
        say(f"    median time per training fit on 4 cores: mpf --frozen {np.median([C[t]['mpf_frozen']['secs'] for t in C]):.0f} s, mpf --marginal {np.median([C[t]['mpf_marginal']['secs'] for t in C]):.0f} s, "
            f"bgms select {np.median([C[t]['bgms_select']['secs'] for t in C]):.0f} s, bgms full {np.median([C[t]['bgms_full']['secs'] for t in C]):.0f} s")
    for nn in (2, 3):
        p = os.path.join(B, f'drh_cv_nn{nn}.jsonl')
        if os.path.exists(cvp) and os.path.exists(p):
            N = load(p)
            if sorted(N) == sorted(C):
                v = np.concatenate([N[t]['logp'] for t in sorted(N)]); sel = cat('bgms_select', 'logp_predictive')
                say(f"    {'mpf --frozen, NN=' + str(nn):30s} {v.mean():8.3f}   difference {paired(v, base)}   (complete entries only: {v[comp].mean():.3f}; entries with missing answers: {v[~comp].mean():.3f})"
                    f"   bgms select predictive minus this: {paired(sel, v)};  median time per training fit {np.median([N[t]['secs'] for t in N]):.0f} s (6 cores); mean log10 lambda {np.mean([N[t]['lam'] for t in N]):.3f}")
open(os.path.join(B, 'summary.txt'), 'w').write("\n".join(out) + "\n")
