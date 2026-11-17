"""Mutation testing of the test suite: inject one deliberate bug at a time into a temporary copy of the
sources and confirm that `make test` fails. A mutation that survives marks a gap in the tests.

Usage: CC=/path/to/compiler [OMP_NUM_THREADS=8] python3 mutation_test.py        (output: results/mutation_test.txt)
"""
import os, shutil, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(HERE, '..', 'src')     # the program and its tests
CC = os.environ.get('CC', 'gcc')
THREADS = os.environ.get('OMP_NUM_THREADS', '8')     # a mutant can make a fit run for a long time: cap what it can use

def run_limited(cmd, cwd, timeout=900):
    """Run a test in its own process group and kill the whole group on timeout, so that a mutant that
    hangs cannot leave processes behind. Returns the exit status, or 'timeout'."""
    import signal
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
                         env=dict(os.environ, OMP_NUM_THREADS=THREADS))
    try:
        return p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        return 'timeout'
M = [
 ('mpf.c', 'Hc += 0.5*(x+s[k]*h[k]);', 'Hc += 0.5*x;', 'field term dropped from H (NN>1 routine)', 0),
 ('mpf.c', 'Hc += 0.5*(x+s[k]*h[k]);', 'Hc += 0.5*x;', 'field term dropped from H (NN=1 routine)', 1),
 ('mpf.c', 'a=(objective == OBJ_LEGACY) ? mult*(m->g[c]-exp(m->lw[c])*total) : 0.0;', 'a=0.0;', 'legacy gradient ignores the weights'),
 ('mpf.c', 'm->Rk[k]=exp(0.5*m->LQ[k]);', 'm->Rk[k]=exp(m->LQ[k]);', 'marginal objective without the square root'),
 ('mpf.c', 'gJk[l] += sk*s[l]*(ck-2*m->u[l]);', 'gJk[l] += sk*s[l]*(ck-m->u[l]);', 'marginal coupling gradient'),
 ('mpf.c', 'gh[k] += sk*ck;', 'gh[k] += sk*coef;', 'marginal field gradient'),
 ('mpf.c', 'T[k]=(om[k] > 0) ? exp(m->lw[c]-X[k]) : 0.0;', 'T[k]=exp(m->lw[c]-X[k]);', 'legacy/frozen flows along missing nodes'),
 ('mpf.c', 'if (om[k] > 0) {\n\t\t\t\t\tfor(c=0;c<nc;c++) m->g[c]=m->lw[c]-2*m->X[(size_t)c*n+k];', 'if (1) {\n\t\t\t\t\tfor(c=0;c<nc;c++) m->g[c]=m->lw[c]-2*m->X[(size_t)c*n+k];', 'marginal flows along missing nodes'),
 ('mpf.c', 'mo->sparsity=mo->n_prox*exp(log_sparsity*log(10.0))*mo->total_w/mo->np;', 'mo->sparsity=mo->n_prox*exp(log_sparsity*log(10.0))*mo->total_w;', 'penalty scaling'),
 ('mpf.c', 'mo->mult[k] += d->w[pr[i].row];', 'mo->mult[k] = d->w[pr[i].row];', 'duplicate rows not summed'),
 ('mpf.c', 'if (objective == OBJ_FROZEN && !set_frozen) {', 'if (0) {', 'frozen weights not frozen (NN>1 routine)', 0),
 ('mpf.c', 'if (objective == OBJ_FROZEN && !set_frozen) {', 'if (0) {', 'frozen weights not frozen (NN=1 routine)', 1),
 ('mpf.c', 'gh[k] += ask-uk;', 'gh[k] += ask+uk;', 'sign of field gradient (partial rows)'),
 ('mpf.c', 'gh[k] -= uk;', 'gh[k] += uk;', 'sign of field gradient (complete rows)'),
 ('mpf.c', "else if (buf[j] == '0') d->v[(size_t)i*n+j]=-1;", "else if (buf[j] == '0') d->v[(size_t)i*n+j]=1;", '0 read as yes'),
 ('mpf.c', 'for(j=0;j<nb;j++) s[pos[j]]=((c >> j) & 1) ? 1.0 : -1.0;', 'for(j=0;j<nb;j++) s[pos[j]]=1.0;', 'completions not enumerated'),
 ('mpf.c', 'if (!isfinite(g[i])) return INFINITY;', 'if (0) return INFINITY;', 'non-finite gradient not detected'),
 ('mpf.c', 'if (!isfinite(f) || !isfinite(gmax)) { info.status=FIT_NUMERICAL; ended=1; }', 'if (0) { info.status=FIT_NUMERICAL; ended=1; }', 'non-finite start not reported'),
 ('mpf.c', 'if (gmax < GTOL_ACCEPT*scale) info.status=FIT_CONVERGED;', 'if (1) info.status=FIT_CONVERGED;', 'every fit reported as converged'),
 ('mpf.c', "if (end == text || *end != '\\0' || !isfinite(v)) die(", "if (end == text || *end != '\\0') die(", 'non-finite numbers accepted'),
 ('mpf.c', 'for(j=i+1;j<sz;j++) e += 2*m->J[(size_t)mem[i]*n+mem[j]]*s[mem[i]]*s[mem[j]];', 'for(j=i+1;j<sz;j++) e += m->J[(size_t)mem[i]*n+mem[j]]*s[mem[i]]*s[mem[j]];', 'multi-flip: pair correction to the flow'),
 ('mpf.c', 'for(l=k+1;l<n;l++) gJk[l] += sk*s[l]*(a-m->Ru[k]-m->Ru[l]+2*Pk[l]);', 'for(l=k+1;l<n;l++) gJk[l] += sk*s[l]*(a-m->Ru[k]-m->Ru[l]);', 'multi-flip: coupling gradient ignores sets containing both ends'),
 ('mpf.c', 'for(l=k+1;l<n;l++) gJk[l] += sk*s[l]*(coef-2*m->Ru[k]-2*m->Ru[l]+4*Pk[l]);', 'for(l=k+1;l<n;l++) gJk[l] += sk*s[l]*(coef-2*m->Ru[k]-2*m->Ru[l]);', 'multi-flip: marginal coupling gradient'),
 ('mpf.c', 'for(j=1;j<=mo->nn;j++) { comb=comb*(n-j+1)/j; mo->n_prox += comb; }', 'mo->n_prox=n;', 'multi-flip: penalty not scaled by the number of neighbours'),
 ('mpf.c', 'if (m->nn > 1) return compute_k_multi(m, theta, grad, objective, set_frozen);', '', 'NN argument ignored'),
 ('mpf.c', 'm->Rv[idx]=exp(0.5*m->LQs[idx]);', 'm->Rv[idx]=exp(m->LQs[idx]);', 'multi-flip: marginal objective without the square root'),
 ('mcmc.c', 'H -= 2*s[b]*phi[b];\n\t\t\t\tif (theta2)', 'H -= s[b]*phi[b];\n\t\t\t\tif (theta2)', 'Gray-code energy update'),
 ('mcmc.c', 'return diff-lz_t+lz_i;', 'return diff+lz_t-lz_i;', 'KL sign of log Z'),
 ('mcmc.c', 'out[j]=lo;', 'out[j]=lo^1;', 'exact sampler off by one state'),
 ('mcmc.c', 'x=2*((v[k] > 0) ? 1.0 : -1.0)*phi;', 'x=((v[k] > 0) ? 1.0 : -1.0)*phi;', 'pseudo-likelihood factor'),
 ('mcmc.c', "for(k=0;k<n;k++) line[k]=((full >> k) & 1) ? '1' : '0';", "for(k=0;k<n;k++) line[k]=((full >> (n-1-k)) & 1) ? '1' : '0';", '-z bit order'),
 ('mcmc.c', 'running *= -2*sp*scale;', 'running *= 2*sp*scale;', 'Monte Carlo acceptance sign'),
 ('nlin.c', 'wt=opt->legacy_cv ? 1.0 : d->w[a];', 'wt=1.0;', 'CV ignores weights'),
 ('nlin.c', 'mask[a]=(include == NULL || include[a]) && (fold_of[a] != in);', 'mask[a]=(include == NULL || include[a]);', 'held-out row left in the training set'),
 ('nlin.c', 'else fi=fit_staged(d, mask, log_sparsity, x, opt->objective, &first_stage);', 'else fi=fit_rows(d, mask, log_sparsity, x, opt->objective);', 'folds skip their first stage'),
 ('nlin.c', 'for(i=0;i<m;i++) complete[i]=(rows == NULL || rows[i]) && d->n_blanks[i] == 0;', 'for(i=0;i<m;i++) complete[i]=(d->n_blanks[i] == 0);', 'the first stage of a fold includes the row it holds out'),
 ('rng.c', 'if (s == 0) s=1;', '', 'a seed whose low 32 bits are zero leaves the generator at zero'),
 ('mpf.c', 'errno != 0 || v < lo || v > hi) die(', 'errno != 0) die(', 'integers (on the command line and in files) not range-checked'),
 ('nsb.c', 'if (max_intervals < 1) { *result=0; return 3; }', '', 'integration with no intervals allowed'),
 ('nlin.c', 'if (val[i] < val[best]) best=i;', 'if (val[i] > val[best]) best=i;', 'search picks the worst grid point'),
 ('nlin.c', 'if (!isfinite(fbest)) {\n\t\tif (at_boundary != NULL) *at_boundary=1;\n\t\treturn NAN;\n\t}', '', 'search returns a value when every candidate is invalid'),
 ('nlin.c', 'if (extend_low && best == 0) {', 'if (0) {', 'search never extends below -2'),
 ('nlin.c', 'if (d->n_blanks[j] > 0) ok=0;       // only fully-observed groups are held out', '', 'groups with partial rows held out'),
 ('main.c', 'opt.objective=OBJ_LEGACY; opt.legacy_cv=1;', 'opt.objective=OBJ_MARGINAL; opt.legacy_cv=1;', '--legacy does not select the legacy objective'),
]
ignore = shutil.ignore_patterns('*.o', 'mpf', 'mpf.dSYM', 'test_mpf')
# every mutation must apply to the current sources: stop at once if the list has gone stale
stale = [t[3] for t in M if (len(t) == 4 and open(os.path.join(SRC, t[0])).read().count(t[1]) != 1)
         or (len(t) == 5 and open(os.path.join(SRC, t[0])).read().count(t[1]) < t[4] + 1)]
if stale:
    sys.exit('mutation_test.py is out of date with the sources; these mutations do not apply: ' + '; '.join(stale))
killed = 0; lines = []
with tempfile.TemporaryDirectory() as td:
    W = os.path.join(td, 'src')
    for fn, old, new, desc, *which in M:
        # a mutation is (file, old, new, description[, k]); with k it changes the k-th of several identical lines
        shutil.rmtree(W, ignore_errors=True); shutil.copytree(SRC, W, ignore=ignore)
        s = open(f'{W}/{fn}').read()
        parts = s.split(old)
        k = which[0] if which else 0
        open(f'{W}/{fn}', 'w').write(old.join(parts[:k + 1]) + new + old.join(parts[k + 1:]))
        b = subprocess.run(['make', '-s', f'CC={CC}', 'mpf', 'tests/test_mpf'], cwd=W, capture_output=True, text=True)
        if b.returncode != 0:
            lines.append(f"{'DID NOT COMPILE':24s} {desc}"); print(lines[-1], flush=True); continue
        u = run_limited(['./tests/test_mpf'], W)
        c = run_limited(['bash', 'tests/test_cli.sh'], W)
        caught = [x for x, r in (('unit', u), ('cli', c)) if r != 0]
        killed += bool(caught)
        lines.append(f"{'caught by ' + '+'.join(caught) if caught else 'SURVIVED':24s} {desc}"); print(lines[-1], flush=True)
not_compiled = sum(l.startswith('DID NOT COMPILE') for l in lines)
lines.append(f'{killed} of {len(M) - not_compiled} injected bugs caught' + (f' ({not_compiled} mutations did not compile)' if not_compiled else '')); print(lines[-1])
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
open(os.path.join(HERE, 'results', 'mutation_test.txt'), 'w').write("\n".join(lines) + "\n")
