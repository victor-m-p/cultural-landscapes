'''
Regression checks for the Python side of the pipeline. Run from anywhere:

    python preprocessing/test_pipeline.py

Checks that
  * p_dist / bin_states agree with a brute-force calculation and with the committed data products
    (configuration_probabilities.txt, configurations.txt) for the published parameters;
  * the committed data_expanded.csv / matrix_expanded.txt are what expand_data.jl computes
    (the expansion is re-implemented here in Python);
  * curation picks questions that every entry answered, and writes the file names the rest of the
    pipeline expects.
'''
import itertools, os, sys, tempfile
import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import fun
import curation_functions as cf
data = os.path.join(here, '..', 'data')

# --- p_dist and bin_states against brute force ---
rng = np.random.default_rng(0)
for n in range(2, 9):
    h = rng.normal(0, 0.5, n); J = rng.normal(0, 0.5, n * (n - 1) // 2)
    states = np.array(list(itertools.product([-1, 1], repeat = n)))   # first column = most significant bit
    assert np.array_equal(states, fun.bin_states(n))
    assert np.array_equal((states + 1) // 2, fun.bin_states(n, sym = False))
    energy = np.array([sum(J[k] * s[a] * s[b] for k, (a, b) in enumerate(itertools.combinations(range(n), 2))) + h @ s for s in states])
    p = np.exp(energy) / np.exp(energy).sum()
    assert np.allclose(p, fun.p_dist(h, J), rtol = 1e-12, atol = 1e-15)
print('p_dist and bin_states agree with brute force for n = 2..8')

# --- committed data products ---
n = 20; nJ = n * (n - 1) // 2
params = np.loadtxt(os.path.join(data, 'mdl_experiments', 'matrix_questions_20_maxna_5_nrows_455_entries_407.txt.mpf_params.dat'))
p = fun.p_dist(params[nJ:], params[:nJ])
p_committed = np.loadtxt(os.path.join(data, 'preprocessing', 'configuration_probabilities.txt'))
assert np.abs(p - p_committed).max() < 1e-12
assert np.array_equal(fun.bin_states(n), np.loadtxt(os.path.join(data, 'preprocessing', 'configurations.txt'), dtype = int))
print('configuration_probabilities.txt and configurations.txt match the committed parameters')

flat = pd.read_csv(os.path.join(data, 'preprocessing', 'data_flattened.csv'))
committed = pd.read_csv(os.path.join(data, 'preprocessing', 'data_expanded.csv'))
rows, mats = [], []
for r in flat.itertuples(index = False):
    options = [[1, -1] if v == 0 else [v] for v in r[1:]]
    combos = [c[::-1] for c in itertools.product(*options[::-1])]     # Julia's product() varies the first iterator fastest
    ids = [int(''.join('1' if x == 1 else '0' for x in c), 2) for c in combos]
    for c, i in zip(combos, ids):
        rows.append((r[0], i, p_committed[i], p_committed[i] / p_committed[ids].sum())); mats.append([r[0]] + list(c))
mine = pd.DataFrame(rows, columns = ['entry_id', 'config_id', 'config_prob', 'entry_prob'])
assert np.array_equal(mine[['entry_id', 'config_id']].values, committed[['entry_id', 'config_id']].values)
assert np.allclose(mine['entry_prob'].values, committed['entry_prob'].values, rtol = 1e-10)
assert np.array_equal(np.array(mats), np.loadtxt(os.path.join(data, 'preprocessing', 'matrix_expanded.txt'), dtype = int))
print('data_expanded.csv and matrix_expanded.txt match a re-implementation of expand_data.jl')

# expand_data.py (the Julia-free equivalent) must give the same result when run on the committed inputs
import shutil, subprocess
with tempfile.TemporaryDirectory() as td:
    os.makedirs(os.path.join(td, 'preprocessing')); os.makedirs(os.path.join(td, 'data', 'preprocessing'))
    shutil.copy(os.path.join(here, 'expand_data.py'), os.path.join(td, 'preprocessing'))
    for f in ('configuration_probabilities.txt', 'data_flattened.csv'): shutil.copy(os.path.join(data, 'preprocessing', f), os.path.join(td, 'data', 'preprocessing'))
    subprocess.run([sys.executable, 'expand_data.py'], cwd = os.path.join(td, 'preprocessing'), check = True)
    out = pd.read_csv(os.path.join(td, 'data', 'preprocessing', 'data_expanded.csv'))
    assert np.array_equal(out[['entry_id', 'config_id']].values, committed[['entry_id', 'config_id']].values)
    assert np.allclose(out['entry_prob'].values, committed['entry_prob'].values, rtol = 1e-10)
    assert np.array_equal(np.loadtxt(os.path.join(td, 'data', 'preprocessing', 'matrix_expanded.txt'), dtype = int), np.loadtxt(os.path.join(data, 'preprocessing', 'matrix_expanded.txt'), dtype = int))
print('expand_data.py reproduces the committed data_expanded.csv and matrix_expanded.txt')

# --- curation ---
r = np.random.default_rng(1); records = []
for q in range(12):
    for e in range(30):
        if q > 0 and (e == q + 3 or r.random() < 0.05 + 0.07 * q): continue    # question 100 is answered by everyone
        for a in ([1, -1] if r.random() < 0.1 else [r.choice([1, -1])]):
            records.append(dict(question_id_drh = 100 + q, entry_id = e, answers = a))
df = pd.DataFrame(records)
grid, fractions = cf.grid_and_fractions(df, 'question_id_drh', 'entry_id', 'answers', 'weight', 'has_answer')
best = cf.subset_nbest(grid, 6, 0, 'question_id_drh', 'has_answer')
assert 100 in best['question_id_drh'].unique() and best['question_id_drh'].nunique() == 6
entries = cf.max_constraints(best, fractions, 2, 0, 'entry_id', 'has_answer', 'question_id_drh', 'answers', 'weight')
matrix = cf.to_dataframe(cf.apply_combinations(entries, 'entry_id', 'question_id_drh', 'answers', 'weight', 6))
assert np.allclose(matrix.groupby('entry_id')['weight'].sum(), 1.0)       # the expansions of an entry share its unit weight
with tempfile.TemporaryDirectory() as td:
    cf.save_output(best, pd.DataFrame({'entry_id': matrix.entry_id.unique(), 'entry_name': 'x'}),
                   pd.DataFrame({'question_id_drh': best.question_id_drh.unique(), 'question_drh': 'q'}),
                   matrix, matrix.drop(columns = 'entry_id'), 'entry_id', 'question_id_drh', 6, 2, td, td)
    assert len(os.listdir(td)) == 4 and all('_nrows_' in f for f in os.listdir(td))
# every selected question answered by every entry: nothing may be dropped
df = pd.DataFrame([dict(question_id_drh = q, entry_id = e, answers = 1) for q in (1, 2) for e in (10, 11, 12)])
grid, fractions = cf.grid_and_fractions(df, 'question_id_drh', 'entry_id', 'answers', 'weight', 'has_answer')
best = cf.subset_nbest(grid, 2, 0, 'question_id_drh', 'has_answer')
assert len(cf.max_constraints(best, fractions, 0, 0, 'entry_id', 'has_answer', 'question_id_drh', 'answers', 'weight')) == 6
assert cf.to_dataframe([]).empty
print('curation: fully-answered questions are selected and their entries kept; output file names use _nrows_')
print('all checks passed')
