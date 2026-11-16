'''
Python equivalent of expand_data.jl, for machines without Julia. Same inputs, same outputs.

For each entry_id (religious culture) find all possible configurations: if there are
inconsistencies or missing answers, expand to all possible completions, and assign each the
probability of that configuration and its probability conditional on the entry.

Run from /preprocessing after prep_early.py:   python expand_data.py
(test_pipeline.py checks that this reproduces the committed data_expanded.csv / matrix_expanded.txt)
'''
import itertools
import numpy as np
import pandas as pd

p = np.loadtxt('../data/preprocessing/configuration_probabilities.txt')
data_flattened = pd.read_csv('../data/preprocessing/data_flattened.csv')

rows, states = [], []
for record in data_flattened.itertuples(index = False):
    entry_id = record[0]
    options = [[1, -1] if v == 0 else [v] for v in record[1:]]
    # same order as Julia's product(): the first unknown question varies fastest
    combos = [c[::-1] for c in itertools.product(*options[::-1])]
    # configuration index: first question is the most significant bit, 1 = yes (as bin_states)
    ids = [int(''.join('1' if x == 1 else '0' for x in c), 2) for c in combos]
    p_raw = p[ids]
    p_norm = p_raw / p_raw.sum()
    for c, i, a, b in zip(combos, ids, p_raw, p_norm):
        rows.append((entry_id, i, a, b))
        states.append([entry_id] + list(c))

d = pd.DataFrame(rows, columns = ['entry_id', 'config_id', 'config_prob', 'entry_prob'])
d.to_csv('../data/preprocessing/data_expanded.csv', index = False)
np.savetxt('../data/preprocessing/matrix_expanded.txt', np.array(states, dtype = int), fmt = '%i', delimiter = '\t')
