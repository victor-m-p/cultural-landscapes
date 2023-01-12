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
maxlik_config_id = np.loadtxt('../data/analysis/maxlik_configuration_id.txt')
maxlik_config = np.loadtxt('../data/analysis/maxlik_configurations.txt')
maxlik_config_id = [int(x) for x in maxlik_config_id]

### clustering ###
# setting distance_threshold = 0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, 
                                n_clusters=None)

model = model.fit(maxlik_config)

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html
# Q: how do we extract things 
# *: there is some color threshold argument

# first just get number of colors

dendrogram_dict = plot_dendrogram(model, 
                                  color_threshold=0.6*max(model.distances_),
                                  get_leaves = True)

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
fig, ax = plt.subplots(figsize = (8, 30), dpi = 300)
dendrogram_dict = plot_dendrogram(model, 
                                  orientation = 'left',
                                  get_leaves = True,
                                  leaf_font_size = 7.5,
                                  labels = maxlik_config_id,
                                  #leaf_label_func = lambda x: str(x + 1),
                                  link_color_func = lambda k: link_cols[k],
                                  above_threshold_color = 'black')
ax.get_xaxis().set_visible(False)
plt.savefig('../fig/dendrogram_observed.pdf')

# extract information
leaves = dendrogram_dict.get('leaves')
leaves_color = dendrogram_dict.get('leaves_color_list')
configid = [maxlik_config_id[x] for x in leaves]

leaf_dataframe = pd.DataFrame(
    {'config_id': configid,
     'comm_color': leaves_color}
)

# save information
leaf_dataframe.to_csv('../data/analysis/dendrogram_clusters_observed.csv', index = False)