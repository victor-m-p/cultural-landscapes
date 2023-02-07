'''
Visualization of top configurations (n = 150) for figure 4A.
Writes some tables for reference as well. 
VMP 2023-02-05: save to .svg and .pdf.
'''

# imports 
import pandas as pd 
import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt 

# read data
network_information = pd.read_csv('../data/analysis/top_configurations_network.csv')
hamming_information = pd.read_csv('../data/analysis/top_configurations_hamming.csv')

# add dendrogram information
network_information = network_information.drop(columns = 'community')
dendrogram_information = pd.read_csv('../data/analysis/dendrogram_clusters.csv')
dendrogram_information = dendrogram_information.rename(columns = {'node_cluster': 'community'})
network_information = network_information.merge(dendrogram_information, on = 'node_id', how = 'inner')

## add community weight to data 
community_weight = network_information.groupby('community')['config_prob'].sum().reset_index(name = 'community_weight')
network_information = network_information.merge(community_weight, on = 'community', how = 'inner')
network_information['comm_color_code'] =  network_information['community']

## community labels (descending order of total weight)
comm_order = network_information[['community_weight']].drop_duplicates().reset_index(drop=True)
comm_order['comm_label'] = comm_order.index+1
comm_order['comm_label'] = comm_order['comm_label'].apply(lambda x: f'Group {x}')
network_information = network_information.merge(comm_order, on = 'community_weight', how = 'inner')

## most weighted configurations within each community
top = network_information[['config_id', 'config_prob', 'comm_label']]
top = top.sort_values('config_prob').groupby('comm_label').tail(10)

## specific nodes to show breadth 
outlier_list = ['Pythagoreanism', 'Peyote', 'Calvinism', 'Wogeo', 'Roman', 'Method']
outliers = [network_information[network_information['entry_name'].str.contains(x)] for x in outlier_list]
outliers = pd.concat(outliers)
outliers = outliers[['config_id', 'config_prob', 'comm_label']]

## all configs to label
label_configs = pd.concat([top, outliers], axis = 0)
maxlik = pd.read_csv('../data/preprocessing/entry_maxlikelihood.csv')
maxlik = maxlik[['config_id', 'entry_name']]
label_configs = maxlik.merge(label_configs, on = 'config_id', how = 'inner')
label_configs = label_configs.sort_values(['comm_label', 'config_prob'],
                                          ascending = [True, False])
label_configs[label_configs['comm_label'] == "Group 1"].tail(10)

## translation dictionary
transl_dict = {
    # community 1 
    1025926: 'Cistercians',
    1027975: 'Ancient Egypt',
    1044359: 'Pre-Christian Ireland',
    1027974: 'Jesuits in Britain',
    1025927: 'Islam in Aceh',
    1044358: 'Vaisnava',
    894854: 'The Essenes',
    886662: 'Calvinism',
    1017734: 'Yiguan Dao', 
    # community 2
    361984: 'Messalians',
    362368: 'Free Methodist', 
    362374: "Jehovah's Witnesses",
    362370: 'Southern Baptists',
    362246: 'Valentinians',
    360960: 'Peyote Religion',
    385536: 'Pythagoreanism',
    # community 3
    1017730: 'Church of England',
    1025920: 'German Protestant',
    1025922: 'Muslims in UAE',
    632710: 'Tallensi',
    501634: 'Catholics in PRC',
    # community 4
    769927: 'Roman Imperial', 
    634758: 'Tsonga',
    769926: 'Mesopotamia',
    638854: 'Luguru',
    # community 5
    1025538: 'Sokoto',
    634496: 'Trumai',
    634370: 'Hidatsa',
    1041926: 'Wogeo',
}

## to dataframe 
annotations = pd.DataFrame.from_dict(transl_dict, 
                       orient = 'index',
                       columns = ['entry_name'])

annotations['config_id'] = annotations.index
network_color = network_information[['config_id', 'comm_color_code', 'node_id']].drop_duplicates()
annotations = annotations.merge(network_color, on = 'config_id', how = 'inner')
annotations.sort_values(['comm_color_code', 'node_id'])

## now nudge the position of labels 
pos_annot = {
    # first community 
    3: (-700, 0), # Jehovah
    16: (-750, 0), # Free Methodist
    35: (-650, 20), # Southern Baptists
    43: (-500, 0), # Messalians
    45: (-500, 0), # Valentinians
    54: (-400, -100), # Pythagoreanism
    108: (-450, -50), # Peyote
    # next community 
    7: (-900, 0), # Muslims UAE
    23: (570, 200), # German Protestant
    36: (-850, 50), # Church of England
    53: (450, 200), # Tallensi
    57: (-650, -150), # Catholics in PRC
    # next community 
    75: (200, 0), # Trumai
    79: (200, 0), # Sokoto
    91: (200, 0), # Hidatsa
    138: (200, 0), # Wogeo
    # next community 
    0: (-500, -200), # Cistercians
    1: (400, -40), # Ancient Egypt
    2: (450, -50), # Jesuits
    4: (-500, -200), # Islam Aceh 
    5: (200, 0), # Pre-Christian Ireland
    8: (500, -40), # Vaisnava
    6: (-500, -200), # Yiguan Dao
    10: (-750, -250), # The Essenes
    11: (-700, 60), # Calvinism
    # next community 
    9: (400, 30), # Tsonga
    12: (230, 0), # Mesopotamia
    19: (200, -20), # Luguru
    24: (200, 50) # Roman Imperial
}

# create network
G = nx.from_pandas_edgelist(hamming_information,
                            'node_x',
                            'node_y',
                            'hamming')

## extract position
pos = {}
for idx, row in network_information.iterrows():
    node_id = row['node_id']
    pos_x = row['pos_x']
    pos_y = row['pos_y']
    pos[node_id] = (pos_x, pos_y)

## add node information to the graph 
network_information_dict = network_information.to_dict('index')
for idx, val in network_information_dict.items(): 
    node_id = val['node_id'] # should also be idx but cautious
    for attr in val: 
        G.nodes[node_id][attr] = val[attr]

# process network information
## check up on this (i.e. can we avoid imports here and make it easy?)
from fun import * 
G = edge_strength(G, 'config_prob') # would be nice to get rid of this. 
edgelist_sorted, edgeweight_sorted = sort_edge_attributes(G, 'pmass_mult', 'hamming', 34000)
nodelist_sorted, nodesize_sorted = sort_node_attributes(G, 'config_prob', 'config_prob')
_, community_sorted = sort_node_attributes(G, 'config_prob', 'comm_color_code') 
node_scalar = 13000

# plot without labels 
fig, ax = plt.subplots(figsize = (8, 8), dpi = 500)
plt.axis('off')
nx.draw_networkx_nodes(G, pos, 
                        nodelist = nodelist_sorted,
                        node_size = [x*node_scalar for x in nodesize_sorted], 
                        node_color = community_sorted,
                        linewidths = 0.5, edgecolors = 'black')
nx.draw_networkx_edges(G, pos, alpha = 0.7,
                       width = edgeweight_sorted,
                       edgelist = edgelist_sorted,
                       edge_color = 'tab:grey'
                       )
plt.savefig('../fig/pdf/landscape_dendrogram.pdf', bbox_inches = 'tight')
plt.savefig('../fig/svg/landscape_dendrogram.svg', bbox_inches = 'tight')

# main plot (Figure 4A)
fig, ax = plt.subplots(figsize = (8, 8), dpi = 500)
plt.axis('off')

nx.draw_networkx_nodes(G, pos, 
                        nodelist = nodelist_sorted,
                        node_size = [x*node_scalar for x in nodesize_sorted], 
                        node_color = community_sorted,
                        linewidths = 0.5, edgecolors = 'black')
nx.draw_networkx_edges(G, pos, alpha = 0.7,
                       width = edgeweight_sorted,
                       edgelist = edgelist_sorted,
                       edge_color = 'tab:grey'
                       )
# annotations
for index, row in annotations.iterrows(): 
    node_idx = row['node_id']
    name = row['entry_name']
    pos_x, pos_y = pos[node_idx]
    xx, yy = pos_annot.get(node_idx)
    color_code = row['comm_color_code']
    ax.annotate(name, xy = [pos_x, pos_y],
                color = color_code,
                xytext=[pos_x+xx, pos_y+yy],
                arrowprops = dict(arrowstyle="->",
                                  connectionstyle='arc3',
                                  color='black'))
plt.subplots_adjust(left=0.15, right=0.8, top=1, bottom=0)
plt.savefig('../fig/pdf/landscape_dendrogram_labels.pdf', bbox_inches = 'tight')
plt.savefig('../fig/svg/landscape_dendrogram_labels.svg', bbox_inches = 'tight')

# recode communities and save overview csv of network information
recode_comm = {'Group 1': 'Group 1',
               'Group 2': 'Group 2',
               'Group 3': 'Group 2',
               'Group 4': 'Group 3',
               'Group 5': 'Group 3'}

network_information['recode_comm'] = [recode_comm.get(x) for x in network_information['comm_label']]
network_information.to_csv('../data/analysis/network_information.csv', index = False)

# save annotation information
annotations.to_csv('../data/analysis/landscape_annotations.csv', index = False)