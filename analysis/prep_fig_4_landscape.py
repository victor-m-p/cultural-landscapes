'''
VMP 2022-12-16: 
Prepare visualization of top configurations (n = 150) for figure 4(A).
The information is used in "fig_4_landscape.py". 
'''

import matplotlib.pyplot as plt 
from matplotlib.colors import rgb2hex
import networkx as nx 
import networkx.algorithms.community as nx_comm
import numpy as np
import pandas as pd 
from fun import *

# load data
configurations = np.loadtxt('../data/preprocessing/configurations.txt', dtype = int)
configuration_probabilities = np.loadtxt('../data/preprocessing/configuration_probabilities.txt')
entry_master = pd.read_csv('../data/preprocessing/entry_configuration_master.csv')
question_reference = pd.read_csv('../data/preprocessing/question_reference.csv')

# bin states and get likelihood and index
n_top_states = 150
top_config_info = top_n_idx(n_top_states, configuration_probabilities) 
top_config_info['node_id'] = top_config_info.index

# merge with entry information 
entry_subset = entry_master[['entry_id', 'config_id', 'entry_prob', 'entry_name']]
config_entry_overlap = entry_subset.merge(top_config_info, on = 'config_id', how = 'inner')

# relative weight for observed states within these configs
datastate_entry_weight = entry_master.groupby("entry_id").size().reset_index(name = 'entry_count')
datastate_entry_weight["entry_weight"] = 1 / datastate_entry_weight["entry_count"]
datastate_entry_weight = config_entry_overlap.merge(datastate_entry_weight, on="entry_id", how="inner")
datastate_entry_weight = datastate_entry_weight.groupby("node_id").entry_weight.sum().rename("datastate_sum").reset_index()

# get the maximum likelihood datastates for each configuration (node) keeping all equally weighted
grouped = config_entry_overlap.groupby("node_id")
maxlikelihood_datastate = grouped["entry_prob"].max().reset_index(name="entry_prob")
maxlikelihood_datastate = maxlikelihood_datastate.merge(config_entry_overlap, on=["node_id", "entry_prob"], how="inner")
maxlikelihood_datastate = maxlikelihood_datastate.merge(datastate_entry_weight, on="node_id", how="inner")

# break ties between maximum likelihood datastates for each configuration (node)
maxlikelihood_sample = (maxlikelihood_datastate
                        .groupby('node_id')
                        .apply(lambda x: x.sample(n=1, random_state=421))
                        .reset_index(drop=True)).drop(columns='config_prob')
network_information = (top_config_info
                       .merge(maxlikelihood_sample,
                               on=['node_id', 'config_id'],
                               how='left')
                       .fillna(0))
network_information['entry_id'] = network_information['entry_id'].astype(int)

# Hamming distance
top_config_ids = top_config_info['config_id'].tolist()
top_configs = configurations[top_config_ids]
h_dists = hamming_distance(top_configs)
filtered_h_dists = hamming_edges(n_top_states, h_dists)
only_one_diff = filtered_h_dists[filtered_h_dists['hamming'] == 1].reset_index(drop = True)

# create network
'''
Create networkx network.
Calculate position of nodes using graphviz 'fdp' program. 
Run Louvain community detection. 
Add position and community information to network_information dataframe. 
'''

# NBNB: GOT TO HERE # 
G = nx.from_pandas_edgelist(only_one_diff,
                            source='node_x',
                            target='node_y',
                            edge_attr='hamming')

pos = nx.nx_agraph.graphviz_layout(G, prog='fdp')
pos_df = pd.DataFrame.from_dict(pos, 
                                 orient='index', 
                                 columns=['pos_x', 'pos_y'])
pos_df['node_id'] = pos_df.index

network_information = network_information.merge(pos_df, 
                                                on='node_id', 
                                                how='inner')

louvain_communities = nx_comm.louvain_communities(G, resolution=0.7, seed=152)

louvain_communities_dict = {}
for i, community in enumerate(louvain_communities):
    for node in community: 
        louvain_communities_dict[node] = i  

louvain_communities_df = pd.DataFrame.from_dict(louvain_communities_dict,
                                                orient='index',
                                                columns=['community'])
louvain_communities_df['node_id'] = louvain_communities_df.index

network_information = network_information.merge(louvain_communities_df, 
                                                on='node_id',
                                                how='inner')

# save 
network_information.to_csv('../data/analysis/top_configurations_network.csv', index = False)
only_one_diff.to_csv('../data/analysis/top_configurations_hamming.csv', index = False)
maxlikelihood_datastate.to_csv('../data/analysis/top_configurations_maxlikelihood.csv', index = False)
config_entry_overlap.to_csv('../data/analysis/top_configurations_overlap.csv', index = False)