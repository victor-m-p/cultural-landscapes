'''
VMP 2022-12-16: 
Prepare visualization of top configurations (n = 150) for figure 4A.
The information is used in "plot_main_network.py". 
'''

import matplotlib.pyplot as plt 
from matplotlib.colors import rgb2hex
import networkx as nx 
import networkx.algorithms.community as nx_comm
import numpy as np
import pandas as pd 
from fun import *

# setup
n_rows, n_nan, n_nodes, n_top_states = 455, 5, 20, 150

# load data
config_prob = np.loadtxt('../data/analysis/configuration_probabilities.txt')
entry_master = pd.read_csv('../data/analysis/entry_configuration_master.csv')
question_reference = pd.read_csv('../data/analysis/question_reference.csv')

# bin states and get likelihood and index
allstates = bin_states(n_nodes) 
top_config_info = top_n_idx(n_top_states, config_prob, 'config_id', 'config_prob') 
top_config_info['node_id'] = top_config_info.index

# merge with entry information 
## take out needed columns from entry master
entry_subset = entry_master[['entry_id', 'config_id', 'entry_prob', 'entry_name']]
## merge with the selected configurations
config_entry_overlap = entry_subset.merge(top_config_info, on = 'config_id', how = 'inner')

# relative weight for observed states within these configs
'''
Weight each entry by data representation.
This gives us the "pre-model" weight for each configuration. 
'''

## get the number of potential configurations for each entry_id
datastate_entry_weight = entry_master.groupby('entry_id').size().reset_index(name = 'entry_count')
## assign 'weight' to be the inverse 
datastate_entry_weight = datastate_entry_weight.assign(entry_weight = lambda x: 1/x['entry_count'])
## merge this with the config_entry_overlap data
datastate_entry_weight = config_entry_overlap.merge(datastate_entry_weight, on = 'entry_id', how = 'inner')
## finally sum by node (129/150 configurations have datastate weight)
datastate_entry_weight = datastate_entry_weight.groupby('node_id')['entry_weight'].sum().reset_index(name = 'datastate_sum')

# get the maximum likelihood datastates for each configuration (node)
'''
This does not break ties between entry_ids that share the same 
configuration (i.e. if they are equally weighted). 
This has to be done later (manually or randomly). 
'''

maxlikelihood_datastate = config_entry_overlap.groupby('node_id')['entry_prob'].max().reset_index(name = 'entry_prob')
maxlikelihood_datastate = config_entry_overlap.merge(maxlikelihood_datastate, on = ['node_id', 'entry_prob'], how = 'inner')
maxlikelihood_datastate = datastate_entry_weight.merge(maxlikelihood_datastate, on = 'node_id', how = 'inner')
maxlikelihood_datastate # (170/129 meaning that we have some ties)

# break ties between maximum likelihood datastates for each configuration (node)
'''
First sample one maxlikelihood datastate in case of tie.
In cases where no datastate exists for the configuration
we will still want to include this configuration in the network.
Thus, we merge with the configuration_ids and fill nan. 
'''

## sample n = 1 
maxlikelihood_sample = maxlikelihood_datastate.groupby('node_id').sample(n = 1, random_state = 421)
maxlikelihood_sample = maxlikelihood_sample.drop(columns = {'config_prob'})
## left join (i.e. configuration_ids left).
network_information = top_config_info.merge(maxlikelihood_sample, 
                              on = ['node_id', 'config_id'], 
                              how = 'left').fillna(0)
## fix dtypes corrupted by the merge 
network_information['entry_id'] = network_information['entry_id'].astype(int)

# Hamming distance
'''
Compute hamming distance for the edges. 
Select only hamming distance = 1 (could be expanded of course). 
'''

configuration_ids = top_config_info['config_id'].tolist()
top_configurations = allstates[configuration_ids]
h_distances = hamming_distance(top_configurations) # could use the class here
h_distances = hamming_edges(n_top_states, h_distances) # could use the class here
h_distances = h_distances[h_distances['hamming'] == 1]

# create network
'''
Create networkx network.
Calculate position of nodes using graphviz 'fdp' program. 
Run Louvain community detection. 
Add position and community information to network_information dataframe. 
'''

# create graph 
G = nx.from_pandas_edgelist(h_distances,
                            'node_x',
                            'node_y',
                            'hamming')

## calculate position
pos = nx.nx_agraph.graphviz_layout(G, prog = "fdp")
## to dataframe
pos_dataframe = pd.DataFrame.from_dict(pos, 
                                       orient='index', 
                                       columns = ['pos_x', 'pos_y'])
pos_dataframe['node_id'] = pos_dataframe.index

## merge with network_information
network_information = network_information.merge(pos_dataframe, 
                                                on = 'node_id', 
                                                how = 'inner')

# Louvain communities 
## resolution = 0.5 gives bias towards fewer and larger communities
louvain_communities = nx_comm.louvain_communities(G, resolution = 0.7, seed = 152)
## might be a better way to do this  
louvain_communities_dict = {}
for num, ele in enumerate(louvain_communities):
    for node in ele: 
        louvain_communities_dict[node] = num  
## to dataframe
louvain_communities_df = pd.DataFrame.from_dict(louvain_communities_dict,
                                                orient = 'index',
                                                columns = ['community'])
louvain_communities_df['node_id'] = louvain_communities_df.index

## merge with network information
network_information = network_information.merge(louvain_communities_df, 
                                                on = 'node_id',
                                                how = 'inner')

# save the work
'''
Save the information needed for visualizing the network of top configurations.
This includes the network information and the hamming distances. 
'''

network_information.to_csv('../data/analysis/top_configurations_network.csv', index = False)
h_distances.to_csv('../data/analysis/top_configurations_hamming.csv', index = False)
maxlikelihood_datastate.to_csv('../data/analysis/top_configurations_maxlikelihood.csv', index = False)
config_entry_overlap.to_csv('../data/analysis/top_configurations_overlap.csv', index = False)