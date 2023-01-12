import pandas as pd 
import numpy as np 
from fun import bin_states, hamming_distance, hamming_edges, edge_strength, edge_information
import networkx as nx 
import matplotlib.pyplot as plt 

n_nodes = 20
sample = pd.read_csv('../data/push_forward/random_n_1000_config_769927.csv')
allstates = bin_states(n_nodes)

# weight by how number of time-steps at location 
sample_w = sample.groupby('config_id').size().reset_index(name = 'number_sampled')

# weight by how probable the state actually is 
probabilities = np.loadtxt('../data/analysis/configuration_probabilities.txt')
config_id = sample_w['config_id'].tolist()
config_p = probabilities[config_id]
sample_w['config_prob'] = config_p

# connect the points if they are within 1 hamming distance 
sample_states = allstates[config_id]

h_distances = hamming_distance(sample_states) 
h_distances = hamming_edges(len(sample_states), h_distances)
h_distances = h_distances[h_distances['hamming'] == 1]

# create network
G = nx.from_pandas_edgelist(h_distances,
                            'node_x',
                            'node_y',
                            'hamming')

pos = nx.nx_agraph.graphviz_layout(G, prog = "fdp")

# add node information
sample_w['index'] = sample_w.index
node_attr_dict = sample_w.to_dict('index')
for idx, val in node_attr_dict.items(): 
    for attr in val: 
        idx = idx 
        G.nodes[idx][attr] = val[attr]
 

##### plot by configuration probability ######
# process 
G = edge_strength(G, 'config_prob') 
edgelst_sorted, edgew_sorted = edge_information(G, 'pmass_mult', 'hamming', 30000)

## thing here is that we need to sort the node information similarly
def node_attributes(Graph, sorting_attribute, value_attribute):
    # first sort by some value (here config_prob)
    sorting_attr = nx.get_node_attributes(G, sorting_attribute)
    sorting_attr = {k: v for k, v in sorted(sorting_attr.items(), key = lambda item: item[1])}
    nodelist_sorted = list(sorting_attr.keys())
    # then take out another thing 
    value_attr = nx.get_node_attributes(G, value_attribute)
    value_attr = {k: v for k, v in sorted(value_attr.items(), key = lambda pair: nodelist_sorted.index(pair[0]))}
    value_sorted = list(value_attr.values())
    # return
    return nodelist_sorted, value_sorted

nodelst_sorted, nodesize_sorted = node_attributes(G, 'config_prob', 'config_prob')

## more prep
sample_enriched = sample.merge(sample_w, on = 'config_id', how = 'inner')
sample_enriched = sample_enriched.convert_dtypes()
sample_enriched['lag'] = sample_enriched['index'].shift(-1)
sample_enriched = sample_enriched.dropna()

# draw
fig, ax = plt.subplots(figsize = (6, 4), dpi = 500)
plt.axis('off')
cmap = plt.cm.get_cmap("Greens") 
nx.draw_networkx_nodes(G, pos, 
                        nodelist = nodelst_sorted,
                        node_size = [x*10000 for x in nodesize_sorted], 
                        node_color = 'tab:blue',
                        linewidths = 0.5, edgecolors = 'black',
                        cmap = cmap)
# foraging 
jit_oldx, jit_oldy = np.random.uniform(-10, 10, 2)
for num, row in sample_enriched.iterrows(): 
    idx_from, idx_to = row['index'], row['lag']
    pos_from, pos_to = pos.get(idx_from), pos.get(idx_to)
    # manual jitter
    jit_newx, jit_newy = np.random.uniform(-10, 10, 2)
    x_vals = [pos_from[0] + jit_oldx, pos_to[0] + jit_newx]
    y_vals = [pos_from[1] + jit_oldy, pos_to[1] + jit_newy]
    plt.plot(x_vals, y_vals, color = 'tab:orange', alpha = 0.5, linewidth = 0.5)
    jit_oldx, jit_oldy = jit_newx, jit_newy 

## NB: 
### different ways you could plot this 
### you could make it undirected, and plot 
### the number of times there is a transition
### between two cases 
### -- or -- you could plot it as Simon did earlier. 

##### plot by sampling probability ######
# process 
G = edge_strength(G, 'number_sampled') 
edgelst_sorted, edgew_sorted = edge_information(G, 'pmass_mult', 'hamming', 0.1)

## thing here is that we need to sort the node information similarly
def node_attributes(Graph, sorting_attribute, value_attribute):
    # first sort by some value (here config_prob)
    sorting_attr = nx.get_node_attributes(G, sorting_attribute)
    sorting_attr = {k: v for k, v in sorted(sorting_attr.items(), key = lambda item: item[1])}
    nodelist_sorted = list(sorting_attr.keys())
    # then take out another thing 
    value_attr = nx.get_node_attributes(G, value_attribute)
    value_attr = {k: v for k, v in sorted(value_attr.items(), key = lambda pair: nodelist_sorted.index(pair[0]))}
    value_sorted = list(value_attr.values())
    # return
    return nodelist_sorted, value_sorted

nodelst_sorted, nodesize_sorted = node_attributes(G, 'number_sampled', 'number_sampled')

# draw
fig, ax = plt.subplots(figsize = (6, 4), dpi = 500)
plt.axis('off')
cmap = plt.cm.get_cmap("Greens") 
nx.draw_networkx_nodes(G, pos, 
                        nodelist = nodelst_sorted,
                        node_size = [x*5 for x in nodesize_sorted], 
                        node_color = 'tab:blue',
                        linewidths = 0.5, edgecolors = 'black',
                        cmap = cmap)
# foraging 
jit_oldx, jit_oldy = np.random.uniform(-10, 10, 2)
for num, row in sample_enriched.iterrows(): 
    idx_from, idx_to = row['index'], row['lag']
    pos_from, pos_to = pos.get(idx_from), pos.get(idx_to)
    # manual jitter
    jit_newx, jit_newy = np.random.uniform(-10, 10, 2)
    x_vals = [pos_from[0] + jit_oldx, pos_to[0] + jit_newx]
    y_vals = [pos_from[1] + jit_oldy, pos_to[1] + jit_newy]
    plt.plot(x_vals, y_vals, color = 'tab:orange', alpha = 0.5, linewidth = 0.5)
    jit_oldx, jit_oldy = jit_newx, jit_newy 

# annotations