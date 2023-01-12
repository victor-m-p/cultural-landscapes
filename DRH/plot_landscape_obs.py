'''
VMP 2022-12-16:
Visualization maxlikelihood observed configurations (n = 260).
Experimental. 
'''

# imports 
import pandas as pd 
import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt 
from fun import hamming_distance, hamming_edges

# read dendrogram 
network_information = pd.read_csv('../data/analysis/dendrogram_clusters_observed.csv')
network_information = network_information.sort_values('config_id').reset_index(drop = True)
network_information['node_id'] = network_information.index

# Hamming distance
'''
NB: with hamming distance == 1,
we appear to SPLIT UP the network.
(209 / 260). 
'''

n_states = 260
maxlik_configurations = np.loadtxt('../data/analysis/maxlik_configurations.txt')
h_distances = hamming_distance(maxlik_configurations)
h_distances = hamming_edges(n_states, h_distances)
h_distances = h_distances[h_distances['hamming'] == 1]

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
pos_dataframe.sort_values('node_id')

## merge with network_information
network_information = network_information.merge(pos_dataframe, 
                                                on = 'node_id', 
                                                how = 'inner')

# add configuration probability
maxlikelihood_entries = pd.read_csv('../data/analysis/entry_maxlikelihood.csv')
maxlikelihood_prob = maxlikelihood_entries[['config_id', 'config_prob']].drop_duplicates()
network_information = network_information.merge(maxlikelihood_prob, on = 'config_id', how = 'inner')

#  add more community details to the network_information dataframe
'''
1. total weight of the commuities (used for table).
2. color of community (perhaps not used). 
'''

## add community weight to data 
community_weight = network_information.groupby('comm_color')['config_prob'].sum().reset_index(name = 'community_weight')
network_information = network_information.merge(community_weight, on = 'comm_color', how = 'inner')

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
#outlier_list = ['Pythagoreanism', 'Peyote', 'Calvinism', 'Wogeo']
#outliers = [network_information[network_information['entry_drh'].str.contains(x)] for x in outlier_list]
#outliers = pd.concat(outliers)
#outliers = outliers[['config_id', 'config_prob', 'comm_label']]

## all configs to label
#label_configs = pd.concat([top, outliers], axis = 0)
maxlikelihood_entries
maxlik = maxlikelihood_entries[['config_id', 'entry_name']]
label_configs = maxlik.merge(top, on = 'config_id', how = 'inner')
label_configs = label_configs.sort_values(['comm_label', 'config_prob'],
                                          ascending = [True, False])
label_configs[label_configs['comm_label'] == "Group 5"].head(30)

## translation dictionary
transl_dict = {
    # community 1 
    361984: 'Messalians',
    360960: 'Peyote',
    99840: 'Spiritualism',
    377350: 'Sethian Gnostic',
    361472: 'Unitarian Universalism',
    # community 2
    634758: 'Tsonga',
    634754: 'Badjau',
    638854: 'Luguru',
    634752: 'Kapauku',
    # community 3
    1025926: 'Cistercians',
    362374: "Jehovah's Witnesses",
    1025927: 'Islam in Aceh',
    1017734: 'Yiguan Dao',
    894854: 'The Essenes',
    # community 4
    1027975: 'Ancient Egypt',
    1027974: 'Jesuits in Britain',
    1044359: 'Pre-Christian Ireland',
    1025922: 'Muslims in UAE',
    # community 5 
    1025920: 'German Protestantism',
    1041286: 'Shaiva World Renouncers',
    516998: 'American Hinduism',
    385536: 'Pythagoreanism',
    # 
    362370: 'Southern Baptists',
    362246: 'Valentinians',
    360960: 'Peyote Religion',
    385536: 'Pythagoreanism'
}

## to dataframe 
annotations = pd.DataFrame.from_dict(transl_dict, 
                       orient = 'index',
                       columns = ['entry_name'])

annotations['config_id'] = annotations.index
network_color = network_information[['config_id', 'comm_color', 'node_id']].drop_duplicates()
annotations = annotations.merge(network_color, on = 'config_id', how = 'inner')
annotations.sort_values(['comm_color', 'node_id'])

## now nudge the position of labels 
pos_annot = {
    # first comm 
    38: (700, 0), # Valentinians
    40: (700, 30), # Southern Baptists
    41: (700, 0), # Jehovah's Witnesses
    174: (700, 0), # The Essenes
    199: (700, 0), # Yiguan Dao
    209: (900, -30), # Cistercians
    210: (900, 0), # Islam in Aceh
    # next community 
    3: (-1500, 0), # Spiritualism 
    30: (-1500, 50), # Peyote
    34: (-1500, 30), # Unitarian Univ.
    35: (-1500, -20), # Messalians
    55: (1200, 0), # Sethian Gnostic
    # next community 
    207: (-2000, 0), # Muslims in UAE
    216: (-2000, 0), # Jesuits in Britain
    217: (-1800, 0), # Ancient Egypt
    255: (-2000, 0), # Pre-Christian Ireland
    # next community 
    58: (-1400, 0), # Pythagoreanism
    92: (-1300, 0), # American Hinduism
    206: (450, 0), # German Protestantism
    236: (-1800, 0), # Shaiva World Renouncers
    # next community 
    122: (-1200, 0), # Kapauku
    123: (-1200, 0), # Badjau
    124: (-1500, 20), # Tsonga
    129: (-1500, -20) # Luguru
}

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
edgelist_sorted, edgeweight_sorted = edge_information(G, 'pmass_mult', 'hamming', 30000)

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
_, community_sorted = node_attributes(G, 'config_prob', 'comm_color') 
nodesize_sorted

# main plot (Figure 4A)
node_scalar = 10000
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
    color_code = row['comm_color']
    ax.annotate(name, xy = [pos_x, pos_y],
                color = color_code,
                #xycoords = 'figure fraction',
                xytext=[pos_x+xx, pos_y+yy],
                #textcoords = 'figure fraction', 
                arrowprops = dict(arrowstyle="->",
                                  connectionstyle='arc3',
                                  color='black'))

#plt.subplots_adjust(left=0.15, right=0.8, top=1, bottom=0)
plt.savefig('../fig/landscape_dendrogram_obs.pdf')

########## TABLES ##########
# table with all entry_id that appear in a community 
'''
Locate all entry_id that are "in" a community, 
find the community that they have most probability weight in. 
Save to latex table with columns
(group, entry_id_drh, entry_name, weight)
'''

## load the information on top configuration / entry overlap
config_entry_overlap = pd.read_csv('../data/analysis/top_configurations_overlap.csv')
## add community information
network_information_sub = network_information[['config_id', 'comm_label']]
config_entry_overlap = config_entry_overlap.merge(network_information_sub)
## groupby community and entry id 
config_entry_comm = config_entry_overlap.groupby(['comm_label', 'entry_id'])['entry_prob'].sum().reset_index()
## if there is a tie between two communities 
## for a specific entry_id, then take first
config_entry_comm = config_entry_comm.sort_values('entry_prob', ascending=False).groupby('entry_id').head(1)
## add back name and get the DRH id 
entry_reference = pd.read_csv('../data/analysis/entry_reference.csv')
config_entry_comm = config_entry_comm.merge(entry_reference, on = 'entry_id', how = 'inner')
## select columns 
config_entry_comm = config_entry_comm[['comm_label', 'entry_id_drh', 'entry_drh', 'entry_prob']]
## sort 
config_entry_comm = config_entry_comm.sort_values(['comm_label', 'entry_prob', 'entry_id_drh'], ascending = [True, False, True])
## rename columns 
config_entry_comm = config_entry_comm.rename(
    columns = {'comm_label': 'Group',
               'entry_id_drh': 'DRH ID',
               'entry_drh': 'Entry name (DRH)',
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
excluded_entries = excluded_entries[['entry_id_drh', 'entry_drh']]
## sort values 
excluded_entries = excluded_entries.sort_values('entry_id_drh', ascending = True)
## rename columns 
excluded_entries = excluded_entries.rename(
    columns = {'entry_id_drh': 'DRH ID',
               'entry_drh': 'Entry name (DRH)'})
## to latex and save
excluded_entries_latex = excluded_entries.to_latex(index=False)
with open('../tables/top_config_excluded.txt', 'w') as f: 
    f.write(excluded_entries_latex)

# table with distinctive features for each community
'''
We should make this cleaner.
'''

## read question reference
question_reference = pd.read_csv('../data/analysis/question_reference.csv')
## get the list of questions
question_id_list = question_reference['question_id'].tolist()
## generate allstates 
n_nodes = 20 
allstates = bin_states(n_nodes)
## loop through five communities
## consider rewriting this (this can be made better)
bit_lst = []
for comm in range(5): # five communities 
    idx_focal = network_information[network_information['community'] == comm]['node_id'].tolist()
    idx_other = network_information[network_information['community'] != comm]['node_id'].tolist()
    bit_focal = avg_bitstring(allstates, network_information, question_id_list, idx_focal, 'node_id', 'config_id', 'question_id', 'config_prob')
    bit_other = avg_bitstring(allstates, network_information, question_id_list, idx_other, 'node_id', 'config_id', 'question_id', 'config_prob')
    bit_focal = bit_focal.rename(columns = {'weighted_avg': f'weighted_avg_focal'})
    bit_other = bit_other.rename(columns = {'weighted_avg': 'weighted_avg_other'})
    bit_diff = bit_focal.merge(bit_other, on = 'question_id', how = 'inner')
    bit_diff = bit_diff.assign(focal_minus_other = lambda x: x[f'weighted_avg_focal']-x['weighted_avg_other'])
    bit_diff['focal_minus_other_abs'] = np.abs(bit_diff['focal_minus_other'])
    bit_diff = question_reference.merge(bit_diff, on = 'question_id', how = 'inner')
    bit_diff = bit_diff.sort_values('focal_minus_other_abs', ascending = False)
    bit_diff['community'] = comm
    bit_lst.append(bit_diff)

# concat
bit_df = pd.concat(bit_lst)

# to percent, and round 
bit_df = bit_df.assign(weighted_avg_focal = lambda x: round(x['weighted_avg_focal']*100, 2),
                       weighted_avg_other = lambda x: round(x['weighted_avg_other']*100, 2),
                       focal_minus_other = lambda x: round(x['focal_minus_other']*100, 2)
                       )

# three most different per community
comm_color = network_information[['comm_label', 'community', 'comm_color']].drop_duplicates()
bit_df = bit_df.merge(comm_color, on = 'community', how = 'inner')
# top three most distinctive features
bit_diff = bit_df.sort_values(['focal_minus_other_abs'], ascending=False).groupby('community').head(3)
# sort values
bit_diff = bit_diff.sort_values(['comm_label', 'focal_minus_other_abs'], ascending = [True, False])
# select columns
bit_diff = bit_diff[['comm_label', 'comm_color', 'question', 'weighted_avg_focal', 'weighted_avg_other', 'focal_minus_other']]
# rename columns
bit_diff = bit_diff.rename(columns = {'comm_label': 'Group',
                                      'comm_color': 'Color',
                                      'question': 'Question',
                                      'weighted_avg_focal': 'Avg. S',
                                      'weighted_avg_other': 'Avg. O',
                                      'focal_minus_other': 'Diff'
                                      })
# to latex table 
bit_latex_string = bit_diff.to_latex(index=False)
with open('../tables/community_questions_table.txt', 'w') as f: 
    f.write(bit_latex_string)

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
annotation_table = annotation_table[['comm_label', 'entry_name', 'entry_drh']]
## rename columns 
annotation_table = annotation_table.rename(
    columns = {'comm_label': 'Group',
               'entry_name': 'Entry name (short)',
               'entry_drh': 'Entry name (DRH)'})
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