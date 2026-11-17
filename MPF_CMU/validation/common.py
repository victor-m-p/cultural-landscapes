"""Shared helpers for the validation experiments: the simulated data sets, calls to ../src/mpf, provenance."""
import hashlib, json, os, platform, random, re, shutil, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'src')
REPO = os.path.join(HERE, '..', '..')
MPF = os.environ.get('MPF_BIN', os.path.join(SRC, 'mpf'))
SOURCES = ['mpf.h', 'mpf.c', 'mcmc.c', 'nlin.c', 'nsb.c', 'rng.c', 'main.c']

# Simulated trials are numbered; trial t is in coupling regime t mod 4. The runner hands each trial script
# its regime's range of beta (the standard deviation of the couplings and fields) as its second and third
# arguments.
REGIME_NAMES = ['dispersed', 'ordered', 'near-critical', 'critical']
REGIME_BOUNDS = [(0.01, 0.125), (0.125, 0.25), (0.25, 0.5), (0.5, 1.0)]

DRH_TAG = 'questions_20_maxna_5_nrows_455_entries_407'
DRH_PUBLISHED_PARAMS = os.path.join(REPO, 'data', 'mdl_experiments', f'matrix_{DRH_TAG}.txt.mpf_params.dat')
DRH_REFERENCE_CSV = os.path.join(REPO, 'data', 'reference', f'direct_reference_{DRH_TAG}.csv')

se = lambda a: np.std(a, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float('nan')     # standard error of a mean

def check_binary_is_current():
    """Refuse to run with a binary that is not up to date with its sources (the provenance would then name
    sources that did not produce the result). `make` knows what the binary depends on. Only applies to the
    binary in ../src; one named by MPF_BIN is the caller's choice."""
    if 'MPF_BIN' in os.environ: return
    if subprocess.run(['make', '-q', '-C', SRC, 'mpf'], capture_output=True).returncode != 0:
        sys.exit(f'{MPF} is missing or older than its sources: run make in {SRC}')

def _sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def _scripts():
    """The experiment scripts (they generate the data, choose what to fit and compute the scores, so they are
    part of what produced a result)."""
    out = {}
    for d in (HERE, os.path.join(HERE, 'bgms')):
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(('.py', '.sh', '.R')): out[os.path.relpath(os.path.join(d, f), HERE)] = _sha(os.path.join(d, f))
    return out

def provenance():
    """What produced a result: hashes of the C sources, the binary and the experiment scripts, the git commit and
    the environment, and a short fingerprint of all the hashes."""
    check_binary_is_current()
    try:
        head = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except OSError:
        head = ''
    p = dict(sources={s: _sha(os.path.join(SRC, s)) for s in SOURCES}, binary=_sha(MPF), scripts=_scripts())
    fp = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:12]
    return dict(fingerprint=fp, git_head=head, **p, python=sys.version.split()[0], numpy=np.__version__, platform=platform.platform())

def write_provenance(path):
    """Record what is about to produce results. `path` ends in .json; the manifest is written to
    <path without .json>.<fingerprint>.json and is never overwritten, so a result file produced by more than
    one version of the code has one manifest per version. Returns the fingerprint."""
    p = provenance()
    out = (path[:-5] if path.endswith('.json') else path) + f".{p['fingerprint']}.json"
    try:
        with open(out, 'x') as f:
            json.dump(p, f, indent=1)
    except FileExistsError:
        pass
    return p['fingerprint']

# where the runner said the results are going; made absolute now, because trial scripts change directory
PROVENANCE_TARGET = os.path.abspath(os.environ['MPF_PROVENANCE']) if os.environ.get('MPF_PROVENANCE') else None

def emit(result):
    """Print one result as a JSON line, stamped with the fingerprint of what produced it (`source`). If the
    runner has said where the results are going (MPF_PROVENANCE), the manifest with that fingerprint is
    then written beside them if it is not there already. Every trial script ends with this. The result is
    printed first, so that a problem with the manifest cannot lose it."""
    result['source'] = provenance()['fingerprint']
    print(json.dumps(result), flush=True)
    if PROVENANCE_TARGET: write_provenance(PROVENANCE_TARGET)

# ---- the simulated data sets (several scripts must generate exactly the same data for the same trial id) ----

def draw_beta(rng, lo, hi):
    """beta = u^2 with u uniform on (0, 1), redrawn until it falls in [lo, hi) (as in the paper's scripts)."""
    while True:
        beta = rng.random() ** 2
        if lo <= beta < hi and beta >= 0.01: return beta

def table1_dataset(tid, lo, hi, mcmc=False):
    """The data of trial `tid` of the Table 1 protocol: 20 nodes, 128 complete rows, drawn by `mpf -g`.
    Writes t_data.dat and t_params.dat in the current directory; returns beta."""
    beta = draw_beta(random.Random(3000 + tid), lo, hi)
    run(['-g', 't', 20, 128, beta, '--seed', 7000 + tid] + (['--mcmc'] if mcmc else []))   # --mcmc prints its own warning only for n > 24
    return beta

def table3_dataset(tid, lo, hi, n=20, nan=5):
    """The data of trial `tid` of the Table 3 protocol: 256 exact draws from a random model, and versions of
    the last 128 with `nan` of the n answers blanked completely at random.
    Returns (beta, theta, the 256 complete rows, the 128 partial rows)."""
    rng = random.Random(1000 + tid)
    beta = draw_beta(rng, lo, hi)             # same draw as big_test_nans.rb, restricted to the regime
    nrng = np.random.default_rng(5000 + tid)
    theta = draw_model(n, beta, nrng)
    p, _ = all_probabilities(n, theta)
    lines = [state_string(d, n) + " 1.0" for d in nrng.choice(1 << n, size=256, p=p)]
    part = []
    for l in lines[128:]:
        code = list(l)
        for i in rng.sample(range(n), nan): code[i] = 'X'
        part.append("".join(code))
    return beta, theta, lines, part

def drh_reference():
    """The curated DRH answers (one row per entry and coding; 1 yes, -1 no, 0 missing) and the question columns."""
    import pandas as pd
    ref = pd.read_csv(DRH_REFERENCE_CSV)
    return ref, [c for c in ref.columns if c not in ('entry_id', 'weight')]

class DrhBootstrap:
    """Parametric bootstrap of the DRH data: 407 entries drawn exactly from a DRH-like model and given the
    real patterns of missing answers, either at random (mcar) or according to the simulated answer to
    "Scriptures" (mar: the pattern is drawn from real entries with the same answer, "Scriptures" itself
    always being observed there, so the pattern depends only on an observed answer).

    The model drawn from is fixed (drh_bootstrap_truth.dat: a frozen-weight fit of the DRH data made during
    this work), so that the simulated data do not change when the DRH is refitted. Any model of the DRH's
    kind would serve."""
    SCRIPTURES = 1                          # column of "Scriptures" (question 4729)
    def __init__(self):
        self.n = 20
        self.truth = np.loadtxt(os.path.join(HERE, 'drh_bootstrap_truth.dat')).ravel()
        ref, q = drh_reference()
        self.masks = (ref.groupby('entry_id')[q].first().values == 0)           # 407 real blank patterns
        scr_obs = ~self.masks[:, self.SCRIPTURES]
        scr_yes = ref.groupby('entry_id')[q[self.SCRIPTURES]].max().values > 0    # an entry counts as "yes" if any of its rows says yes
        self.pool = {True: self.masks[scr_obs & scr_yes], False: self.masks[scr_obs & ~scr_yes]}
        self.p, _ = all_probabilities(self.n, self.truth)
    def rows(self, rep):
        """Replicate `rep`: (the 407 drawn states, {'mcar': rows, 'mar': rows}). One generator makes the draws
        and then both sets of patterns, in that order."""
        rng = np.random.default_rng(90000 + rep)
        draws = rng.choice(1 << self.n, size=407, p=self.p)
        out = {}
        for mech in ('mcar', 'mar'):
            rows = []
            for d in draws:
                s = list(state_string(d, self.n))
                if mech == 'mcar': m = self.masks[rng.integers(len(self.masks))]
                else:
                    pl = self.pool[s[self.SCRIPTURES] == '1']; m = pl[rng.integers(len(pl))]
                for i in np.flatnonzero(m): s[i] = 'X'
                rows.append("".join(s) + " 1.0")
            out[mech] = rows
        return draws, out

def draw_model(n, beta, rng):
    """Same model as `mpf -g`: every coupling and field from N(0, beta)."""
    return rng.normal(0.0, beta, n * (n + 1) // 2)

def all_probabilities(n, theta):
    """Exact probabilities of all 2^n states; bit i of the state index is node i (as in the C code)."""
    nJ = n * (n - 1) // 2
    J = np.zeros((n, n)); J[np.triu_indices(n, 1)] = theta[:nJ]
    idx = np.arange(1 << n, dtype=np.int64)
    S = np.where((idx[:, None] >> np.arange(n)) & 1, 1.0, -1.0)
    H = np.einsum('ij,ij->i', S @ J, S) + S @ theta[nJ:]
    p = np.exp(H - H.max())
    return p / p.sum(), H

def state_string(state, n):
    return "".join('1' if (int(state) >> i) & 1 else '0' for i in range(n))

def write_params(path, theta):
    with open(path, 'w') as f:
        f.write(" ".join(f"{v:.10e}" for v in theta) + " ")

def write_data(path, n, rows):
    with open(path, 'w') as f:
        f.write(f"{len(rows)}\n{n}\n" + "\n".join(rows) + "\n")

def run(args, threads=None):
    """Run mpf; returns (stdout, stderr). A non-zero exit status raises."""
    env = dict(os.environ)
    if threads is not None:
        env['OMP_NUM_THREADS'] = str(threads)
    r = subprocess.run([MPF] + [str(a) for a in args], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"mpf {' '.join(map(str, args))} failed with status {r.returncode}: {r.stderr}")
    return r.stdout, r.stderr

def kl(datafile, truth, fit):
    return float(re.search(r'KL: (\S+)', run(['-k', datafile, truth, fit])[0]).group(1))

def count_warnings(stderr):
    """Number of fit / cross-validation warnings mpf printed."""
    return sum(1 for l in stderr.split('\n') if 'warning' in l)

def fit_end(stdout):
    """How a fixed-lambda fit (`mpf -l`) ended: (status text, max |gradient|)."""
    m = re.search(r'evaluations\); (.+?), max \|gradient\| = (\S+)', stdout)
    return m.group(1), float(m.group(2))

def fit_fixed(datafile, log_sparsity, flags=(), tag='fit', nn=1):
    """Fit at a fixed lambda; returns (parameter file, status text). A copy of the data is used so that
    different objectives do not overwrite each other's <file>_params.dat."""
    copy = f"{datafile}.{tag}"
    shutil.copy(datafile, copy)
    out, err = run(['-l', copy, log_sparsity, nn, *flags])
    return copy + '_params.dat', fit_end(out)[0]

def fit_cv(datafile, flags=(), tag='cv', threads=None):
    """Cross-validated fit; returns (parameter file, chosen log-sparsity, details). The details are the number
    of warnings printed during the search, and the number of folds and of fold fits that did not converge at
    the lambda that was chosen."""
    copy = f"{datafile}.{tag}"
    shutil.copy(datafile, copy)
    out, err = run(['-c', copy, 1, *flags], threads=threads)
    lam = float(re.findall(r'Best log_sparsity: (\S+)', out)[-1])
    m = re.search(r'at the chosen log-sparsity: (\d+) folds, (\d+) with a fit that did not converge', out)
    return copy + '_params.dat', lam, dict(warnings=count_warnings(err), folds=int(m.group(1)), cv_not_converged=int(m.group(2)))

def load_trials(path):
    """Read a JSON-lines result file, refusing duplicate trial ids."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    ids = [r['tid'] for r in rows]
    dup = sorted({t for t in ids if ids.count(t) > 1})
    if dup:
        raise SystemExit(f"{path}: duplicate trial ids {dup[:10]}{'...' if len(dup) > 10 else ''}; refusing to summarise")
    return rows

def done_ids(path):
    return {json.loads(l)['tid'] for l in open(path) if l.strip()} if os.path.exists(path) else set()
