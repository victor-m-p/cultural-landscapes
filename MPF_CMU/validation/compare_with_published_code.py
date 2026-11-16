"""The rewritten program (with --legacy) against the published one on random inputs: fixed-lambda fits, KL,
probabilities of all configurations, and cross-validated fits.

The published sources are taken from git (commit 23dce37, the last commit before the rewrite) and built
in a temporary directory.

Usage: CC=/path/to/compiler-with-openmp python3 compare_with_published_code.py     (output: results/compare_with_published_code.txt)
(set MPF_PUBLISHED_SRC to a directory containing the published sources if the git history is not available)
"""
import os, random, re, shutil, subprocess, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.join(HERE, '..', '..')
NEW = os.environ.get('MPF_BIN', os.path.join(HERE, '..', 'mpf')); CC = os.environ.get('CC', 'gcc'); COMMIT = '23dce37'
out_lines = []
def say(s):
    print(s, flush=True); out_lines.append(s)
with tempfile.TemporaryDirectory() as td:
    for f in ('main.c', 'mpf.c', 'mcmc.c', 'nlin.c', 'nsb.c', 'mpf.h'):
        if os.environ.get('MPF_PUBLISHED_SRC'):     # a directory holding the published sources, for machines without the git history
            shutil.copy(os.path.join(os.environ['MPF_PUBLISHED_SRC'], f), os.path.join(td, f))
        else:
            open(os.path.join(td, f), 'w').write(subprocess.run(['git', '-C', REPO, 'show', f'{COMMIT}:MPF_CMU/{f}'], capture_output=True, text=True, check=True).stdout)
    gsl_c = subprocess.run(['gsl-config', '--cflags'], capture_output=True, text=True).stdout.split()
    gsl_l = subprocess.run(['gsl-config', '--libs'], capture_output=True, text=True).stdout.split()
    subprocess.run([CC, '-O3', '-fopenmp', *gsl_c, 'main.c', 'mpf.c', 'mcmc.c', 'nlin.c', 'nsb.c', '-o', 'mpf_published', '-lm', *gsl_l], cwd=td, check=True, capture_output=True)
    ORIG = os.path.join(td, 'mpf_published'); os.chdir(td)
    rng = random.Random(11); env = dict(os.environ, OMP_NUM_THREADS='4')
    def run(cmd): return subprocess.run(cmd, capture_output=True, text=True, env=env)
    def make(name, n, m, blanks, weighted, dup):
        rows = []
        for i in range(m):
            if dup and rows and rng.random() < 0.3: rows.append(rng.choice(rows)); continue
            s = [rng.choice('01') for _ in range(n)]
            if blanks and rng.random() < 0.5:
                for k in rng.sample(range(n), rng.randint(1, min(blanks, n - 1))): s[k] = 'X'
            rows.append("".join(s) + " " + (str(rng.choice([0.125, 0.25, 0.5, 1.0, 2.0])) if weighted else "1.0"))
        open(name, 'w').write(f"{m}\n{n}\n" + "\n".join(rows) + "\n")
    def P(f): return np.array([float(x) for x in open(f).read().split()])
    res = {'complete': [], 'missing': [], 'missing_from_orig': []}
    for t in range(60):
        n = rng.randint(3, 20); m = rng.randint(20, 300); ls = round(rng.uniform(-0.5, 2.0), 3); blanks = 0 if t % 2 == 0 else rng.randint(1, 5)
        make('a.dat', n, m, blanks, t % 3 == 0, t % 4 == 0); shutil.copy('a.dat', 'b.dat')
        run([ORIG, '-l', 'a.dat', str(ls), '1']); o = run([NEW, '-l', 'b.dat', str(ls), '1', '--legacy'])
        assert o.returncode == 0, o.stderr
        a = P('a.dat_params.dat'); d = np.abs(a - P('b.dat_params.dat')).max()
        if blanks == 0: res['complete'].append(d)
        else:
            res['missing'].append(d)   # not convex: also start the new code from the published code's answer
            run([NEW, '-l', 'b.dat', str(ls), '1', '--legacy', '--start', 'a.dat_params.dat']); res['missing_from_orig'].append(np.abs(a - P('b.dat_params.dat')).max())
    say(f"fixed-lambda fits, complete data ({len(res['complete'])} random datasets, n=3..20): max |new - published| = {max(res['complete']):.2e}")
    say(f"fixed-lambda fits, missing data ({len(res['missing'])} datasets): max |new (zero start) - published (random start)| = {max(res['missing']):.2e}; median {np.median(res['missing']):.2e}")
    say(f"   new code started from the published code's answer: largest movement = {max(res['missing_from_orig']):.2e}")
    dk = []; dz = []
    for t in range(20):
        n = rng.randint(3, 16); npar = n * (n + 1) // 2
        for f in ('p1.dat', 'p2.dat'): open(f, 'w').write(" ".join(f"{rng.gauss(0, 0.4):.10e}" for _ in range(npar)) + " ")
        make('k.dat', n, 5, 0, False, False)
        ko = float(re.search(r'KL: (\S+)', run([ORIG, '-k', 'k.dat', 'p1.dat', 'p2.dat']).stdout).group(1)); kn = float(re.search(r'KL: (\S+)', run([NEW, '-k', 'k.dat', 'p1.dat', 'p2.dat']).stdout).group(1))
        dk.append(abs(ko - kn))
        shutil.copy('p1.dat', 'q1.dat'); run([ORIG, '-z', 'p1.dat', str(n)]); run([NEW, '-z', 'q1.dat', str(n)])
        A = [l.split() for l in open('p1.dat_probs.dat')]; B = [l.split() for l in open('q1.dat_probs.dat')]
        assert [x[0] for x in A] == [x[0] for x in B]
        dz.append(max(abs(float(x[2]) - float(y[2])) for x, y in zip(A, B)))
    say(f"KL (-k), 20 random parameter pairs: max |new - published| = {max(dk):.2e} (the published code prints 6 decimals)")
    say(f"probabilities (-z), 20 random models: same state order; max |new - published| = {max(dz):.2e}")
    dl = []; dp = []; dkl = []
    for t in range(8):
        n = rng.randint(5, 9); beta = rng.choice([0.1, 0.2, 0.4])
        run([NEW, '-g', 'g', str(n), '80', str(beta), '--seed', str(100 + t)])
        rows = open('g_data.dat').read().split('\n')[2:82]
        if t % 2:
            for i in range(40, 80):
                s = list(rows[i])
                for k in rng.sample(range(n), 2): s[k] = 'X'
                rows[i] = "".join(s)
        open('c1.dat', 'w').write(f"80\n{n}\n" + "\n".join(rows) + "\n"); shutil.copy('c1.dat', 'c2.dat')
        o = run([ORIG, '-c', 'c1.dat', '1']).stdout; nw = run([NEW, '-c', 'c2.dat', '1', '--legacy']).stdout
        lo = float(re.findall(r'Best log_sparsity: (\S+)', o)[-1]); ln = float(re.findall(r'Best log_sparsity: (\S+)', nw)[-1])
        dl.append(abs(lo - ln)); dp.append(np.abs(P('c1.dat_params.dat') - P('c2.dat_params.dat')).max())
        k1 = float(re.search(r'KL: (\S+)', run([NEW, '-k', 'c1.dat', 'g_params.dat', 'c1.dat_params.dat']).stdout).group(1)); k2 = float(re.search(r'KL: (\S+)', run([NEW, '-k', 'c1.dat', 'g_params.dat', 'c2.dat_params.dat']).stdout).group(1))
        dkl.append(abs(k1 - k2)); say(f"   CV fit {t}: n={n} {'with' if t % 2 else 'no'} missing data: log-lambda published {lo:.4f} new {ln:.4f}; max |param diff| {dp[-1]:.1e}; KL from truth {k1:.4f} vs {k2:.4f}")
    say(f"cross-validated fits (8 problems): max |log-lambda diff| = {max(dl):.4f}; max |param diff| = {max(dp):.2e}; max |KL diff| = {max(dkl):.4f}")
    # Larger neighbourhoods. In the published mpf.c the two- and three-flip blocks of create_near() are
    # commented out (so NN=2 silently meant NN=1). Re-enable them in a copy and compare on complete data
    # (those blocks pre-date the handling of missing entries and cannot be used with blanks).
    lines = open('mpf.c').read().split('\n')
    def uncomment(first, last):          # 1-based line numbers in the published file
        for i in range(first - 1, last):
            assert lines[i].lstrip().startswith('//'), (i + 1, lines[i])
            lines[i] = lines[i].replace('// ', '', 1) if '// ' in lines[i] else lines[i].replace('//', '', 1)
    assert 'if (n_step >= 2)' in lines[392] and 'if (n_step >= 3)' in lines[467], 'published mpf.c is not laid out as expected'
    uncomment(393, 398); uncomment(443, 492)
    os.makedirs('nn'); open('nn/mpf.c', 'w').write('\n'.join(lines))
    for f in ('main.c', 'mcmc.c', 'nlin.c', 'nsb.c', 'mpf.h'): shutil.copy(f, 'nn/' + f)
    subprocess.run([CC, '-O3', '-fopenmp', *gsl_c, 'main.c', 'mpf.c', 'mcmc.c', 'nlin.c', 'nsb.c', '-o', 'mpf_published_nn', '-lm', *gsl_l], cwd='nn', check=True, capture_output=True)
    ORIG_NN = os.path.join(td, 'nn', 'mpf_published_nn')
    for nn in (1, 2, 3):
        diffs = []
        for t in range(12):
            n = rng.randint(max(3, nn + 1), 9); m = rng.randint(10, 60); ls = round(rng.uniform(-0.5, 1.5), 3)
            make('a.dat', n, m, 0, t % 3 == 0, t % 4 == 0); shutil.copy('a.dat', 'b.dat')
            run([ORIG_NN, '-l', 'a.dat', str(ls), str(nn)]); o = run([NEW, '-l', 'b.dat', str(ls), str(nn), '--legacy'])
            assert o.returncode == 0, o.stderr
            diffs.append(np.abs(P('a.dat_params.dat') - P('b.dat_params.dat')).max())
        say(f"NN={nn}, published code with its multi-flip blocks re-enabled, 12 random complete datasets (n up to 9): max |new - published| = {max(diffs):.2e}")
    # the DRH complete rows with NN=2
    drh0 = os.path.join(REPO, 'data', 'mdl_experiments', 'matrix_questions_20_maxna_0_nrows_201_entries_181.txt.mpf')
    shutil.copy(drh0, 'e1.mpf'); shutil.copy(drh0, 'e2.mpf')
    run([ORIG_NN, '-l', 'e1.mpf', '0.4', '2']); run([NEW, '-l', 'e2.mpf', '0.4', '2', '--legacy'])
    say(f"NN=2 on the 201 complete DRH rows (n=20) at log-lambda 0.4: max |new - published (re-enabled)| = {np.abs(P('e1.mpf_params.dat') - P('e2.mpf_params.dat')).max():.2e}")

    # the DRH input at the published lambda
    drh = os.path.join(REPO, 'data', 'mdl_experiments', 'matrix_questions_20_maxna_5_nrows_455_entries_407.txt.mpf')
    shutil.copy(drh, 'd1.mpf'); shutil.copy(drh, 'd2.mpf')
    run([ORIG, '-l', 'd1.mpf', '0.205745', '1']); run([NEW, '-l', 'd2.mpf', '0.205745', '1', '--legacy'])
    com = P(drh + '_params.dat')
    say(f"DRH at log-lambda 0.205745: published build vs committed file {np.abs(P('d1.mpf_params.dat') - com).max():.2e}; new --legacy vs committed file {np.abs(P('d2.mpf_params.dat') - com).max():.2e}; published build vs new {np.abs(P('d1.mpf_params.dat') - P('d2.mpf_params.dat')).max():.2e}")
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
open(os.path.join(HERE, 'results', 'compare_with_published_code.txt'), 'w').write("\n".join(out_lines) + "\n")
