"""How stable is the hierarchical clustering of the most probable configurations (Figure 4A / Figure A1)?

The tree is built (Ward linkage, as in analysis/fig_A1_dendrogram.py) from WHICH configurations are among
the top N; their probabilities do not enter. This script cuts the tree into the paper's three groups and
measures agreement (adjusted Rand index on the configurations two analyses share) between the published
analysis (published parameters, N = 150) and

  * the same parameters with a different N,
  * the refits in results/drh_refits, with N = 150.

Needs the pinned environment (scikit-learn 1.2.0, scipy): the tree depends on tie-breaking in the library.
Usage: python clustering_sensitivity.py      (output: results/clustering_sensitivity.txt)
"""
import os, sys, warnings
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import fcluster
warnings.filterwarnings('ignore')
here = os.path.dirname(os.path.abspath(__file__)); repo = os.path.join(here, '..', '..')
sys.path.insert(0, os.path.join(repo, 'analysis'))
from fun import top_n_idx, bin_states, p_dist, linkage_matrix
conf = bin_states(20)
def clades(params, n_top, k=3):
    par = np.loadtxt(params); p = p_dist(par[190:], par[:190])
    top = top_n_idx(n_top, p, 'config_id', 'config_prob'); ids = top['config_id'].tolist()[:n_top]
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(conf[ids])
    Z = linkage_matrix(model)
    lab = fcluster(Z, k, 'maxclust')
    return dict(zip(ids, lab)), len(set(fcluster(Z, 0.6 * Z[-1, 2], 'distance'))), sorted(np.round([p[[i for i in ids if dict(zip(ids, lab))[i] == j]].sum() for j in set(lab)], 3), reverse=True)
pub_file = os.path.join(repo, 'data', 'mdl_experiments', 'matrix_questions_20_maxna_5_nrows_455_entries_407.txt.mpf_params.dat')
base, nclus, w = clades(pub_file, 150)
out = []
def say(s):
    print(s); out.append(s)
say(f"published analysis: N=150, {nclus} clusters at the 0.6*max height, three groups with weights {w}")
say(f"{'comparison':46s} {'shared':>6s} {'ARI (3 groups)':>15s} {'clusters at 0.6*max':>20s}   group weights")
def row(label, params, n_top):
    lab, nc, w = clades(params, n_top)
    shared = [i for i in lab if i in base]
    say(f"{label:46s} {len(shared):6d} {adjusted_rand_score([base[i] for i in shared], [lab[i] for i in shared]):15.3f} {nc:20d}   {w}")
for n_top in (100, 120, 130, 140, 145, 149, 151, 155, 160, 170, 200):
    row(f"published parameters, top {n_top}", pub_file, n_top)
for key, label in (('legacy', 'refit --legacy'), ('marginal', 'refit marginal'), ('marginal_entrycv', 'refit marginal, entry-wise CV'), ('frozen_entrycv', 'refit frozen, entry-wise CV, NN=1'),
                   ('frozen_entrycv_nn2', 'refit frozen, entry-wise CV, NN=2'), ('frozen_entrycv_nn3', 'refit frozen, entry-wise CV, NN=3'), ('legacy_nn2', 'refit --legacy, NN=2')):
    f = os.path.join(here, 'results', 'drh_refits', f'drh_{key}.mpf_params.dat')
    if os.path.exists(f): row(f"{label}, top 150", f, 150)
open(os.path.join(here, 'results', 'clustering_sensitivity.txt'), 'w').write("\n".join(out) + "\n")
