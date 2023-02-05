'''
VMP 2022-12-16:
Visualization of top configurations (n = 150) for figure 4A.
Writes some tables for reference as well. 
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

#  add more community details to the network_information dataframe
'''
1. total weight of the commuities (used for table).
2. color of community (perhaps not used). 
'''

## add community weight to data 
community_weight = network_information.groupby('community')['config_prob'].sum().reset_index(name = 'community_weight')
network_information = network_information.merge(community_weight, on = 'community', how = 'inner')
network_information['comm_color_code'] =  network_information['community']

## community labels (descending order of total weight)
comm_order = network_information[['community_weight']].drop_duplicates().reset_index(drop=True)
comm_order['comm_label'] = comm_order.index+1
comm_order['comm_label'] = comm_order['comm_label'].apply(lambda x: f'Group {x}')
network_information = network_information.merge(comm_order, on = 'community_weight', how = 'inner')

# prepare annotations
'''
Manually annotate certain points entries in the plot for reference.
This is a manual process (e.g. picking and giving short names). 
*: other maximum likelihood data-states share configuration.
**: this configuration is not maximum likelihood (but still most likely for config.)
'''

### LABELS ###
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
maxlik = pd.read_csv('../data/analysis/entry_maxlikelihood.csv')
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
    362368: 'Free Methodist', # still maxlik 
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
    # 
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
edgelist_sorted, edgeweight_sorted = edge_information(G, 'pmass_mult', 'hamming', 34000)

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

# get node attributes
nodelist_sorted, nodesize_sorted = node_attributes(G, 'config_prob', 'config_prob')
_, community_sorted = node_attributes(G, 'config_prob', 'comm_color_code') 
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
plt.savefig('../fig/pdf/landscape_dendrogram.pdf')
plt.savefig('../fig/svg/landscape_dendrogram.svg')

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
                #xycoords = 'figure fraction',
                xytext=[pos_x+xx, pos_y+yy],
                #textcoords = 'figure fraction', 
                arrowprops = dict(arrowstyle="->",
                                  connectionstyle='arc3',
                                  color='black'))
plt.subplots_adjust(left=0.15, right=0.8, top=1, bottom=0)
plt.savefig('../fig/landscape_dendrogram.pdf')

# save network information
network_information.to_csv('../data/analysis/network_information_enriched.csv', index = False)

# table 4
recode_comm = {'Group 1': 'Group 1',
               'Group 2': 'Group 2',
               'Group 3': 'Group 2',
               'Group 4': 'Group 3',
               'Group 5': 'Group 3'}

network_information['recode_comm'] = [recode_comm.get(x) for x in network_information['comm_label']]


## test time-periods ##
network_information

data_raw = pd.read_csv('../data/raw/drh_20221019.csv')
entry_reference = pd.read_csv('../data/analysis/entry_reference.csv')

# for each entry id, get 
data_year = data_raw[['entry_id', 'start_year', 'end_year']]
data_year = data_year.groupby('entry_id').agg({'start_year': 'min', 'end_year': 'max'}).reset_index()

# inner join with the data we already have
entry_reference = entry_reference[['entry_id', 'entry_name']]
year_reference = entry_reference.merge(data_year, on = 'entry_id', how = 'inner')

# sort by start year
year_reference = year_reference.sort_values('start_year', ascending=True)

network_sub = network_information[['entry_id', 'recode_comm']]
year_reference = network_sub.merge(year_reference, on = 'entry_id', how = 'inner')
year_reference.groupby('recode_comm')['start_year'].mean()

## find weight ##
network_information.groupby('recode_comm')['config_prob'].sum()

network_sub = network_information[['recode_comm', 'config_prob', 'config_id']]
annotations = annotations.merge(network_sub, on = 'config_id', how = 'inner')
annotations.sort_values(['recode_comm', 'node_id'],
                        ascending = [True, True])

### make the above general and nice ###

########## TABLES ##########
# table with all entry_id that appear in a community 
'''
Locate all entry_id that are "in" a community, 
find the community that they have most probability weight in. 
Save to latex table with columns
(group, entry_id_drh, entry_name, weight)
'''

pd.set_option('display.max_colwidth', None)
## load the information on top configuration / entry overlap
config_entry_overlap = pd.read_csv('../data/analysis/top_configurations_overlap.csv')
## add community information
network_information_sub = network_information[['config_id', 'comm_label']]
config_entry_overlap = config_entry_overlap.merge(network_information_sub)
config_entry_overlap
## groupby community and entry id 
config_entry_comm = config_entry_overlap.groupby(['comm_label', 'entry_id', 'node_id'])['entry_prob'].sum().reset_index()
## if there is a tie between two communities 
## for a specific entry_id, then take first
config_entry_comm = config_entry_comm.sort_values('entry_prob', ascending=False).groupby('entry_id').head(1)
## add back name and get the DRH id 
entry_reference = pd.read_csv('../data/analysis/entry_reference.csv')
config_entry_comm = config_entry_comm.merge(entry_reference, on = 'entry_id', how = 'inner')
## select columns 
config_entry_comm = config_entry_comm[['comm_label', 'node_id', 'entry_id', 'entry_name', 'entry_prob']]

## new stuff ## 
# (1) add 1 to node_id
config_entry_comm['node_id'] = config_entry_comm['node_id'] + 1
# (2) recode groups 
new_groups = {
    'Group 1': 'Group 1', # red
    'Group 2': 'Group 2.2', # dark blue
    'Group 3': 'Group 2.1', # light blue
    'Group 4': 'Group 3.2', # dark yellow
    'Group 5': 'Group 3.1'} # light yellow
config_entry_comm = config_entry_comm.replace({'comm_label': new_groups})
# (3) round prob mass
config_entry_comm['entry_prob'] = [round(x*100, 2) for x in config_entry_comm['entry_prob']]

### looks great ###
config_entry_comm = config_entry_comm.sort_values(['comm_label', 'node_id', 'entry_prob', 'entry_id'], 
                                                  ascending = [True, True, False, True])

## rename columns 
config_entry_comm = config_entry_comm.rename(
    columns = {'comm_label': 'Group',
               'node_id': 'Node ID',
               'entry_id': 'DRH ID',
               'entry_name': 'Entry name (DRH)',
               'entry_prob': 'Weight'})

## to latex and save
config_entry_latex = config_entry_comm.to_latex(index=False)
with open('../tables/top_config_included.txt', 'w') as f: 
    f.write(config_entry_latex)

# table with all entries (entry_id) that do not appear in top states
'''
Take all of the entries in our data that do not appear in any community
(i.e. who only have configurations that are not in the top n = 150 configurations).
Save to latex table with columns (entry_id_drh, entry_name)

'''
## anti-join 
top_config_entries = config_entry_overlap[['entry_id']]

excluded_entries = entry_reference.merge(top_config_entries, on = 'entry_id', how = 'left', indicator = True)
excluded_entries = excluded_entries[excluded_entries['_merge'] == 'left_only']
## select columns
excluded_entries = excluded_entries[['entry_id', 'entry_name']]
## sort values 
excluded_entries = excluded_entries.sort_values('entry_id', ascending = True)
## rename columns 
excluded_entries = excluded_entries.rename(
    columns = {'entry_id': 'DRH ID',
               'entry_name': 'Entry name (DRH)'})

## to latex and save
excluded_entries_latex = excluded_entries.to_latex(index=False)
with open('../tables/top_config_excluded.txt', 'w') as f: 
    f.write(excluded_entries_latex)


#### old stuff ####
# table with information on the entries that we highlight/annotate in 4A
'''
Save to latex table with columns: 
(community, entry_name, entry_name_drh)
'''
## get the community labels (groups)
network_information_comm = network_information[['comm_label', 'config_id']]
## merge with annotation dataframe
annotation_table = network_information_comm.merge(annotations, on = 'config_id', how = 'inner')
## select subset of columns
annotation_table = annotation_table[['comm_label', 'entry_name', 'entry_name']]
## rename columns 
annotation_table = annotation_table.rename(
    columns = {'comm_label': 'Group',
               'entry_name': 'Entry name (short)',
               'entry_name': 'Entry name (DRH)'})
## to latex 
annotation_latex = annotation_table.style.hide(axis = 'index').to_latex()
## save 
with open('../tables/annotation_table.txt', 'w') as f: 
    f.write(annotation_latex)

# table with total probability mass per community
'''
Not included in the manuscript, but used as reference
'''
## select needed columns
community_table = network_information[['comm_label', 'community_weight']].drop_duplicates()
## convert from fraction to percentage and round to 2 decimals
community_table['community_weight'] = community_table['community_weight'].apply(lambda x: round(x*100, 2))
## rename columns
community_table = community_table.rename(
    columns = {'comm_label': 'Group',
               'community_weight': 'Weight'}
)
## to latex table (might want to come back and fix decimals)
community_table = community_table.to_latex(index = False)
## save 
with open('../tables/community_weight_table.txt', 'w') as f: 
    f.write(community_table)

# save additional information
network_information.to_csv('../data/analysis/network_information_enriched.csv', index = False)