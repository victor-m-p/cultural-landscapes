'''
Compare two regenerated analyses (see regenerate.py), e.g. the published parameters and a refit.

    python analysis/compare_regenerated.py regenerated/published regenerated/frozen_nn1_entrycv > report.txt

Run with the same environment that produced the two directories.
'''
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fun import p_dist, bin_states
from sklearn.metrics import adjusted_rand_score
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)

A, B = sys.argv[1], sys.argv[2]
names = (os.path.basename(os.path.normpath(A)), os.path.basename(os.path.normpath(B)))
n = 20; nJ = n * (n - 1) // 2
def load(d):
    par = np.loadtxt(os.path.join(d, 'params.dat'))
    return dict(par = par, p = p_dist(par[nJ:], par[:nJ]),
                net = pd.read_csv(os.path.join(d, 'data', 'analysis', 'network_information.csv')),
                master = pd.read_csv(os.path.join(d, 'data', 'preprocessing', 'entry_configuration_master.csv')),
                ann = pd.read_csv(os.path.join(d, 'data', 'analysis', 'landscape_annotations.csv')),
                clus = pd.read_csv(os.path.join(d, 'data', 'analysis', 'dendrogram_clusters.csv')))
a, b = load(A), load(B)
q = pd.read_csv(os.path.join(A, 'data', 'preprocessing', 'question_reference.csv'))['question'].tolist()
pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
print(f'A = {names[0]}\nB = {names[1]}\n')

print('1. PARAMETERS')
ja, jb = a['par'][:nJ], b['par'][:nJ]
print(f"   correlation of couplings {np.corrcoef(ja, jb)[0, 1]:.4f}; of fields {np.corrcoef(a['par'][nJ:], b['par'][nJ:])[0, 1]:.4f}; largest change in any parameter {np.abs(a['par'] - b['par']).max():.3f}")
ta, tb = set(np.argsort(-np.abs(ja))[:15]), set(np.argsort(-np.abs(jb))[:15])
print(f"   the 15 strongest couplings (Figure 3): {len(ta & tb)} shared")
for k in sorted(ta ^ tb): print(f"      only in {'A' if k in ta else 'B'}: {q[pairs[k][0]]} -- {q[pairs[k][1]]}  (A {ja[k]:+.3f}, B {jb[k]:+.3f})")

print('\n2. MOST PROBABLE CONFIGURATIONS (Figure 4A)')
na, nb = a['net'].set_index('config_id'), b['net'].set_index('config_id')
shared = [c for c in na.index if c in nb.index]
print(f"   top 150: {len(shared)} shared; probability mass A {na.config_prob.sum():.3f}, B {nb.config_prob.sum():.3f}")
print(f"   the 20 most probable in A have ranks in B: {[int(nb.node_id[c]) + 1 if c in nb.index else None for c in na.sort_values('node_id').index[:20]]}")

print('\n3. GROUPS (dendrogram of the top 150; Figure A1, Tables 5, A2-A4)')
for nm, x in ((names[0], a), (names[1], b)):
    g = x['net'].groupby('recode_comm').agg(configurations = ('node_id', 'size'), mass = ('config_prob', 'sum'), clusters = ('comm_label', 'nunique'))
    print(f"   {nm}: {x['clus'].node_cluster.nunique()} clusters at the paper's cut height; three top-level groups:")
    print('      ' + g.round(3).to_string().replace('\n', '\n      '))
print(f"   agreement of the three groups on the {len(shared)} shared configurations: adjusted Rand index {adjusted_rand_score(na.recode_comm[shared], nb.recode_comm[shared]):.3f}")
print('   ' + pd.crosstab(na.recode_comm[shared].rename(f'{names[0]}'), nb.recode_comm[shared].rename(f'{names[1]}')).to_string().replace('\n', '\n   '))
ann = a['ann'][['entry_name', 'config_id']].copy()
ann['A'] = [na.recode_comm.get(c) for c in ann.config_id]; ann['B'] = [nb.recode_comm.get(c, 'not in top 150') for c in ann.config_id]
ann['rank A'] = [int(na.node_id[c]) + 1 for c in ann.config_id]; ann['rank B'] = [int(nb.node_id[c]) + 1 if c in nb.index else None for c in ann.config_id]
print('   where the configurations labelled in Figure 4A fall:')
print('      ' + ann.drop(columns = 'config_id').sort_values(['A', 'rank A']).to_string(index = False).replace('\n', '\n      '))
for t in ('table_A2_clades.txt', 'table_community_weight.txt'):
    for nm, d in ((names[0], A), (names[1], B)):
        print(f"\n   {t} -- {nm}:"); print('      ' + open(os.path.join(d, 'tables', t)).read().strip().replace('\n', '\n      '))

print('\n4. ARCHAIC SPARTAN CULTS (Table 4): probability of each completion')
for nm, x in ((names[0], a), (names[1], b)):
    s = x['master'][x['master'].entry_name.str.contains('Archaic Spartan')].sort_values('config_id', ascending = False)
    print(f"   {nm}: " + ', '.join(f"{lab} {100 * pr:.1f}%" for lab, pr in zip(['ritual & sacrifice', 'no ritual & sacrifice', 'ritual & no sacrifice', 'no ritual & no sacrifice'], s.entry_prob)))

print('\n5. FOCAL LANDSCAPES (Figures 4B, 4C): single-answer changes from the focal configuration')
conf = bin_states(n)
for label, cid in (('Free Methodist Church', 362368), ('Roman Imperial Cult', 769927)):
    print(f'   {label}')
    for nm, x in ((names[0], a), (names[1], b)):
        p = x['p']; nb_ids = [cid ^ (1 << (n - 1 - k)) for k in range(n)]; pn = p[nb_ids]
        order = np.argsort(-pn)[:3]
        moves = '; '.join(f"{q[k]} -> {'yes' if conf[nb_ids[k]][k] == 1 else 'no'} ({100 * pn[k] / pn.sum():.0f}%, p ratio {pn[k] / p[cid]:.2f})" for k in order)
        print(f"      {nm}: p = {100 * p[cid]:.2f}%, rank {int((p > p[cid]).sum()) + 1}; neighbours more probable than it: {int((pn > p[cid]).sum())} of {n}; most likely changes: {moves}")
