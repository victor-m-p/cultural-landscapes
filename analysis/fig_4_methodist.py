'''
Produces Figure 4C.
VMP 2022-02-05: save .svg and .pdf 
VMP 2022-02-06: clean this, and use as template to clean "seed_methodist".
'''

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import networkx as nx
from fun import *
import configuration as cn 
pd.set_option('display.max_colwidth', None)

# loads 
configuration_probabilities = np.loadtxt(f'../data/preprocessing/configuration_probabilities.txt')
entry_config_master = pd.read_csv(f'../data/preprocessing/entry_configuration_master.csv')
entry_reference = pd.read_csv(f'../data/preprocessing/entry_reference.csv')
question_reference = pd.read_csv('../data/preprocessing/question_reference.csv')
network_information = pd.read_csv('../data/analysis/network_information.csv')
configurations = np.loadtxt('../data/preprocessing/configurations.txt', dtype = int)

# Free Methodist Church
match_substring(entry_config_master, 'Methodist')
config_idx = 362368
n_nearest = 2  
n_top_states = 49 

# get neighbors
df_hamming_neighbors =  hamming_neighbors_N_removed(n_nearest, config_idx, configurations, configuration_probabilities)

# sample N largest and gather in different format
most_probable_neighbors = df_hamming_neighbors.nlargest(n_top_states, 'config_prob_neighbor')
most_probable_configs = (pd.concat([most_probable_neighbors[['config_id_focal', 'config_prob_focal']].drop_duplicates().rename(columns={'config_id_focal': 'config_id', 'config_prob_focal': 'config_prob'}),
                   most_probable_neighbors[['config_id_neighbor', 'config_prob_neighbor']].rename(columns={'config_id_neighbor': 'config_id', 'config_prob_neighbor': 'config_prob'})
                  ])
         .reset_index(drop=True)
        )
most_probable_configs['node_id'] = most_probable_configs.index

# hamming distance for nodes 
d_hamming = (pd.concat([pd.DataFrame([(config_idx, 0)], columns=['config_id', 'hamming']),
                        most_probable_neighbors[['config_id_neighbor', 'hamming']].rename(columns={'config_id_neighbor': 'config_id'})
                       ])
            )
node_attr = most_probable_configs.merge(d_hamming, on='config_id', how='inner')
node_attr_dict = node_attr.to_dict(orient='index')

# hamming distance for edges
config_id = most_probable_configs['config_id'].tolist()
top_states = configurations[config_id]
h_distances = hamming_distance(top_states) 
h_distances = hamming_edges(n_top_states + 1, h_distances)
h_distances = h_distances[h_distances['hamming'] == 1]

# create network
G = nx.from_pandas_edgelist(h_distances, 'node_x', 'node_y', 'hamming')
pos = nx.nx_agraph.graphviz_layout(G, prog="fdp")

# add all node information
for idx, val in node_attr_dict.items(): 
    for attr in val: 
        G.nodes[val['node_id']][attr] = val[attr]
        
# process 
G = edge_strength(G, 'config_prob') 
edgelist_sorted, edgew_sorted = sort_edge_attributes(G, 'pmass_mult', 'hamming', 40000)
nodelist_sorted, nodesize_sorted = sort_node_attributes(G, 'config_prob', 'config_prob')
color_list = [node_attr_dict[node]['hamming'] for node in nodelist_sorted]

# annotations 
entry_config_weight = entry_config_master[['config_id', 'entry_name', 'entry_id', 'entry_prob']]
annotations = entry_config_weight.merge(most_probable_configs, on = 'config_id', how = 'inner')
annotations = annotations.merge(entry_reference, on = ['entry_id', 'entry_name'], how = 'inner')

match_nodeid(annotations, 0) # Free Methodist 
match_nodeid(annotations, 1) # Jehovah 
match_nodeid(annotations, 2) # Southern Baptist
match_nodeid(annotations, 4) # No maximum likelihood configuration (or any?)
match_nodeid(annotations, 5) # Sachchai
match_nodeid(annotations, 6) # No maximum likelihood configuration (or any?)
match_nodeid(annotations, 9) # Tunisian
match_nodeid(annotations, 8) # Pauline
match_nodeid(annotations, 3) # Messalians

entryname_annotations = {
    879: 'Free Methodist',
    1311: "Jehovah's\nWitnesses", 
    1307: 'Southern\nBaptists',
    953: 'Sachchai', 
    1517: 'Tunisian Women',
    196: 'Pauline Christianity',
}

position_annotations = {
    0: (-110, 400), # Free Meth
    1: (-75, 200), # Jehova
    2: (-70, -460), # S. Baptist
    5: (-60, -270), # Sachchai
    9: (-105, -550), # Tunisian
    8: (-145, 390) # Pauline
}

d_annot = (pd.DataFrame.from_dict(entryname_annotations, orient='index', columns=['entry_name_short'])
    .assign(entry_id=lambda x: x.index)
    .merge(annotations, on=['entry_id'], how='inner')
    .loc[[0, 1, 2, 3, 5, 8]])

# main plot 
cmap = plt.cm.get_cmap("Blues") 

fig, ax = plt.subplots(figsize = (6, 4), dpi = 500)
plt.axis('off')

nx.draw_networkx_nodes(G, pos, 
                        nodelist = nodelist_sorted,
                        node_size = [x*15000 for x in nodesize_sorted], 
                        node_color = [3-x for x in color_list],
                        linewidths = 0.5, edgecolors = 'black',
                        cmap = cmap)

rgba = rgb2hex(cmap(0.8))

nx.draw_networkx_edges(G, pos, alpha = 0.7,
                       width = [x*5 if x > 0.1 else 0 for x in edgew_sorted],
                       edgelist = edgelist_sorted,
                       edge_color = rgba
                       )

for index, row in d_annot.iterrows(): 
    node_idx = row['node_id']
    name = row['entry_name_short']
    pos_x, pos_y = pos[node_idx]
    x_nudge, y_nudge = position_annotations.get(node_idx)
    color = rgb2hex(cmap(0.99))
    ax.annotate(name, xy=[pos_x, pos_y],
                color=rgba,
                xytext=[pos_x+x_nudge, pos_y+y_nudge],
                arrowprops = dict(arrowstyle="->",
                                  connectionstyle='arc3',
                                  color='black'))
plt.savefig('../fig/pdf/seed_FreeMethChurch_annotated_mix.pdf', bbox_inches = 'tight')
plt.savefig('../fig/svg/seed_FreeMethChurch_annotated_mix.svg', bbox_inches = 'tight')

# transition probabilities
import configuration as cn 
Methodist = cn.Configuration(config_idx, 
                             configurations, 
                             configuration_probabilities)

# transition probabilities
transition_probabilities = Methodist.transition_probs_to_neighbors(question_reference)
transition_probabilities['transition_prob'] = transition_probabilities['transition_prob']*100

# most likely (least stable traits)
## 15.01% (0.00029): having large-scale rituals required
## 13.34% (0.00026): having special treatment for corpses
## 12.27% (0.00024): not having formal burials

# least likely (most stable traits)
## 0.02%: having adult sacrifice required
## 0.03%: having child sacrifice required
## 0.19%: having castration required 

# which religions do these correspond to?
## most likely 
most_likely = transition_probabilities.head(3)
most_likely = most_likely[['config_id', 'question', 'transition_prob']]
most_likely = entry_config_master.merge(most_likely, on = 'config_id', how = 'inner')
most_likely = most_likely.sort_values('config_prob', ascending = False)
## 1. Soutern Baptists
## 2. Circumcellions
## 3. Sachchai

## least likely 
least_likely = transition_probabilities.tail(3)
least_likely = least_likely[['config_id', 'question', 'transition_prob']]
least_likely = entry_config_master.merge(least_likely, on = 'config_id')
least_likely = least_likely.sort_values('config_prob')
## 1. does not exist
## 2. does not exist
## 3. does not exist

# How likely compared to self 
methodist_prob = Methodist.p # 0.0055 (more probable than all neighbors)
