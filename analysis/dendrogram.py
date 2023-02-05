import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    return dendrogram(linkage_matrix, **kwargs)

### preprocessing ###
# generate all states
n_nodes, n_top_states = 20, 150  
from fun import bin_states, top_n_idx
allstates = bin_states(n_nodes) 

# find the top 150 states 
config_prob = np.loadtxt('../data/analysis/configuration_probabilities.txt')
top_config_info = top_n_idx(n_top_states, config_prob, 'config_id', 'config_prob') 
configuration_ids = top_config_info['config_id'].tolist()

# take out top states
top_configurations = allstates[configuration_ids]

### clustering ###
# setting distance_threshold = 0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, 
                                n_clusters=None)

model = model.fit(top_configurations)

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html
# Q: how do we extract things 
# *: there is some color threshold argument

# first do not plot, but just get the list so that we can change the colors
dendrogram_dict = plot_dendrogram(model, 
                                  color_threshold=0.6*max(model.distances_),
                                  get_leaves = True,
                                  no_plot = True)

# extract the information that we need 
Z = model.children_
leaves = dendrogram_dict.get('leaves')
leaves_color = dendrogram_dict.get('leaves_color_list')

## try wes anderson
color_dict = {
    'C1': "#34649e", 
    'C2': "#78B7C5", 
    'C5': "#EBCC2A", 
    'C4':  "#c99700", 
    'C3': "#F21A00"
}

leaf_cols = {leaf:color_dict.get(c) for leaf, c in zip(leaves, leaves_color)}

link_cols = {}
for i, i12 in enumerate(Z[:,:2].astype(int)):
    c1, c2 = (link_cols[x] if x > len(Z) else leaf_cols.get(x) for x in i12)
    link_cols[i+1+len(Z)] = c1 if c1 == c2 else '#000000'
link_cols

# actually plot it 
fig, ax = plt.subplots(figsize = (12, 20), dpi = 300)
dendrogram_dict = plot_dendrogram(model, 
                                  orientation = 'left',
                                  get_leaves = True,
                                  leaf_font_size = 8,
                                  leaf_label_func = lambda x: str(x + 1),
                                  link_color_func = lambda k: link_cols[k],
                                  above_threshold_color = 'black')
ax.get_xaxis().set_visible(False)
plt.tight_layout()
plt.savefig('../fig/dendrogram.pdf')

# extract information
leaves = dendrogram_dict.get('leaves')
leaves_color = dendrogram_dict.get('leaves_color_list')
leaf_dataframe = pd.DataFrame(
    {'node_id': leaves,
     'node_cluster': leaves_color}
)

# save information
leaf_dataframe.to_csv('../data/analysis/dendrogram_clusters.csv', index = False)