'''
Regenerate every data product, figure and table of the DRH analysis from a parameter file, in a
temporary copy of the pipeline, WITHOUT touching the committed data/, fig/ or tables/.

    python analysis/regenerate.py PARAMS_FILE OUTPUT_DIR [STAGE_DIR]

PARAMS_FILE: n(n-1)/2 couplings then n fields (the format written by mpf).
OUTPUT_DIR receives data/analysis, the small files of data/preprocessing, fig/ and tables/, plus
run_log.txt (which scripts ran, with which library versions) and params.dat. It must be new, empty,
or an earlier output of this script (which is then replaced). STAGE_DIR, if given, keeps the
working copy of the pipeline instead of using a temporary directory.

The scripts are run in the documented order with the current Python interpreter. The groups come from the
dendrogram of each fit ("Group 2" of one fit is not "Group 2" of another) and the label offsets were tuned
by hand for the published fit, so read the output with compare_regenerated.py rather than taking group
labels at face value.

If any script fails, OUTPUT_DIR is left as it was, what was produced goes to OUTPUT_DIR.failed, and the
exit status is 1.
'''
import os, shutil, subprocess, sys, tempfile

repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if len(sys.argv) not in (3, 4):
    sys.exit('usage: python analysis/regenerate.py PARAMS_FILE OUTPUT_DIR [STAGE_DIR]')
params, outdir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
stage = os.path.abspath(sys.argv[3]) if len(sys.argv) == 4 else None     # keep the working copy here (for debugging)
with open(params) as f:
    params_text = f.read()       # read now: PARAMS_FILE may live inside OUTPUT_DIR
# OUTPUT_DIR (and its .new / .failed siblings) are replaced, so insist that each is either new/empty or
# something this script wrote earlier
outdir = outdir.rstrip(os.sep)
new, failed_dir = outdir + '.new', outdir + '.failed'
for d in (outdir, new, failed_dir):
    if os.path.lexists(d) and not (os.path.isdir(d) and (not os.listdir(d) or os.path.exists(os.path.join(d, 'run_log.txt')))):
        sys.exit(f'{d} exists, is not empty and was not written by regenerate.py (no run_log.txt); refusing to replace it')
if stage is not None and os.path.exists(stage) and os.listdir(stage):
    sys.exit(f'{stage} is not empty; give a new directory for the working copy')
tag = 'questions_20_maxna_5_nrows_455_entries_407'
steps = [('preprocessing', 'prep_early.py'), ('preprocessing', 'expand_data.py'), ('preprocessing', 'prep_late.py'),
         ('analysis', 'fig_3_parameters.py'), ('analysis', 'fig_A1_dendrogram.py'), ('analysis', 'prep_fig_4_landscape.py'),
         ('analysis', 'fig_4_landscape.py'), ('analysis', 'fig_4_methodist.py'), ('analysis', 'fig_4_roman.py'),
         ('analysis', 'table_4_spartans.py'), ('analysis', 'table_5_annotations.py'), ('analysis', 'table_A1_questions.py'),
         ('analysis', 'table_A2_clades.py'), ('analysis', 'table_A3_and_A4_entries.py'), ('analysis', 'table_community_weight.py')]
log = []
with tempfile.TemporaryDirectory() as tmp:
    td = stage if stage is not None else tmp
    os.makedirs(td, exist_ok=True)
    for d in ('preprocessing', 'analysis'):
        os.makedirs(os.path.join(td, d))
        for f in os.listdir(os.path.join(repo, d)):
            if f.endswith('.py'): shutil.copy(os.path.join(repo, d, f), os.path.join(td, d, f))
    for d in ('data/reference', 'data/mdl_experiments', 'data/preprocessing', 'data/analysis', 'fig/pdf', 'fig/svg', 'tables'):
        os.makedirs(os.path.join(td, d))
    for kind in ('direct_reference', 'entry_reference', 'question_reference'):
        shutil.copy(os.path.join(repo, 'data', 'reference', f'{kind}_{tag}.csv'), os.path.join(td, 'data', 'reference'))
    with open(os.path.join(td, 'data', 'mdl_experiments', f'matrix_{tag}.txt.mpf_params.dat'), 'w') as f:
        f.write(params_text)
    env = dict(os.environ, MPLBACKEND='Agg')
    for d, script in steps:
        r = subprocess.run([sys.executable, script], cwd=os.path.join(td, d), capture_output=True, text=True, env=env)
        last = [l for l in r.stderr.strip().split('\n') if l][-1:] if r.returncode else []
        log.append(f"{'ok    ' if r.returncode == 0 else 'FAILED'} {d}/{script}" + (f"   {last[0][:200]}" if last else ''))
        print(log[-1], flush=True)
    versions = subprocess.run([sys.executable, '-c', 'import sys, numpy, pandas, sklearn, scipy, matplotlib, networkx; print("python", sys.version.split()[0], "numpy", numpy.__version__, "pandas", pandas.__version__, "scikit-learn", sklearn.__version__, "scipy", scipy.__version__, "matplotlib", matplotlib.__version__, "networkx", networkx.__version__)'], capture_output=True, text=True).stdout.strip()
    # assemble the new output next to the old one, then swap, so a failure here leaves the old output intact
    shutil.rmtree(new, ignore_errors=True)
    for d in ('data/analysis', 'fig', 'tables'):
        shutil.copytree(os.path.join(td, d), os.path.join(new, d))
    os.makedirs(os.path.join(new, 'data', 'preprocessing'))
    for f in os.listdir(os.path.join(td, 'data', 'preprocessing')):
        if f not in ('configurations.txt', 'configuration_probabilities.txt', 'matrix_expanded.txt'):   # large, and regenerated in seconds
            shutil.copy(os.path.join(td, 'data', 'preprocessing', f), os.path.join(new, 'data', 'preprocessing'))
    with open(os.path.join(new, 'params.dat'), 'w') as f:
        f.write(params_text)
    open(os.path.join(new, 'run_log.txt'), 'w').write(f"parameters: {os.path.relpath(params, repo)}\n{versions}\n" + "\n".join(log) + "\n")
    # a run in which a script failed never replaces an earlier output
    n_failed = sum(l.startswith('FAILED') for l in log)
    target = failed_dir if n_failed else outdir
    if os.path.isdir(target):
        shutil.rmtree(target)
    os.rename(new, target)
if n_failed:
    sys.exit(f'{n_failed} script(s) failed; {outdir} was left as it was and the incomplete output is in {failed_dir}')
