"""Shared helpers for the validation experiments: exact sampling, file writing, calls to ../mpf, provenance."""
import hashlib, json, os, platform, re, shutil, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MPF = os.environ.get('MPF_BIN', os.path.join(HERE, '..', 'mpf'))
SOURCES = ['mpf.h', 'mpf.c', 'mcmc.c', 'nlin.c', 'nsb.c', 'main.c']

def provenance():
    """What produced a result: hashes of the C sources and of the binary, the git commit, and the environment."""
    def sha(path):
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    try:
        head = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except OSError:
        head = ''
    return dict(git_head=head, sources={s: sha(os.path.join(HERE, '..', s)) for s in SOURCES}, binary=sha(MPF),
                python=sys.version.split()[0], numpy=np.__version__, platform=platform.platform())

def write_provenance(path):
    with open(path, 'w') as f:
        json.dump(provenance(), f, indent=1)

def source_tag():
    """Short fingerprint of the C sources, stored with every trial."""
    h = hashlib.sha256()
    for s in SOURCES:
        with open(os.path.join(HERE, '..', s), 'rb') as f:
            h.update(f.read())
    return h.hexdigest()[:12]

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

def fit_fixed(datafile, log_sparsity, flags=(), tag='fit', nn=1):
    """Fit at a fixed lambda; returns (parameter file, status text). A copy of the data is used so that
    different objectives do not overwrite each other's <file>_params.dat."""
    copy = f"{datafile}.{tag}"
    shutil.copy(datafile, copy)
    out, err = run(['-l', copy, log_sparsity, nn, *flags])
    status = re.search(r'evaluations\); ([^,]+),', out)
    return copy + '_params.dat', (status.group(1) if status else '?')

def fit_cv(datafile, flags=(), tag='cv', threads=None):
    """Cross-validated fit; returns (parameter file, chosen log-sparsity, number of warnings)."""
    copy = f"{datafile}.{tag}"
    shutil.copy(datafile, copy)
    out, err = run(['-c', copy, 1, *flags], threads=threads)
    lam = float(re.findall(r'Best log_sparsity: (\S+)', out)[-1])
    return copy + '_params.dat', lam, count_warnings(err)

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
