# COGSCI23
import pandas as pd 
import matplotlib.pyplot as plt 
import networkx as nx 
import numpy as np 

small_text, large_text = 12, 18

# load JSD 
d_JSD = pd.read_csv('../data/COGSCI23/evo_entropy/JSD_10.csv')
d_JSD['weight'] = 1-d_JSD['JSD']
d_JSD = d_JSD[['i', 'j', 'weight']]

# load GMM
d_GMM = pd.read_csv('../data/COGSCI23/evo_GMM/t_10_att_from.csv')
d_GMM['node'] = d_GMM.index

# load edgelist (bit funky concept)
d_edgelist = pd.read_csv('../data/COGSCI23/evo_clean/overview.csv')
d_from = d_edgelist[['config_from']].drop_duplicates()
d_from = d_from.rename(columns = {'config_from': 'config_to'})
d_test = d_edgelist.merge(d_from, on = 'config_to', how = 'inner')
d_test = d_test.groupby('config_to').size().reset_index(name = 'count')

# overall nodeObj
d_test = d_test.rename(columns = {'config_to': 'config_from'})
d_nodeattr = d_GMM.merge(d_test, on = 'config_from', how = 'left').fillna(0)

# create NETWORKX obj 
G = nx.from_pandas_edgelist(d_JSD, 
                            source = 'i',
                            target = 'j',
                            edge_attr = 'weight')

# add node attributes
for num, row in d_nodeattr.iterrows():
    nodesize = row['count']
    node_id = row['node']
    cluster = row['cluster']
    config_id = row['config_from']
    G.nodes[node_id]['cluster'] = cluster 
    G.nodes[node_id]['config_id'] = config_id
    G.nodes[node_id]['nodesize'] = nodesize

# prepare plot 
## edgeweight 
edge_weight = dict(nx.get_edge_attributes(G, 'weight'))
edge_list = []
edge_w = []
for x, y in edge_weight.items(): 
    edge_list.append(x)
    edge_w.append(y)
    
## degree 
degree = dict(G.degree(weight = 'weight'))
node_list = []
node_deg = []
for x, y in degree.items(): 
    node_list.append(x)
    node_deg.append(y)

## size 
node_count = dict(nx.get_node_attributes(G, 'nodesize')) 
node_size = [x for x in node_count.values()]

## color 
node_cluster = dict(nx.get_node_attributes(G, 'cluster'))
node_color = ['tab:blue' if x == 0 else 'tab:orange' for x in node_cluster.values()]

# plot 1: hairball 
fig, ax = plt.subplots(dpi = 300)
pos = nx.spring_layout(G, weight = 'weight',
                       k = 1/np.sqrt(260),
                       seed = 4)
nx.draw_networkx_nodes(G, pos, 
                       nodelist = node_list, 
                       node_size = [x*0.01 for x in node_size],
                       node_color = node_color)
nx.draw_networkx_edges(G, pos, 
                       edgelist = edge_list,
                       width = [x*0.01 for x in edge_w])

# plot 2: more spread
fig, ax = plt.subplots(dpi = 300)
pos = nx.spring_layout(G, weight = 'weight',
                       k = 0.15/np.sqrt(260),
                       seed = 12)
nx.draw_networkx_nodes(G, pos, 
                       nodelist = node_list, 
                       node_size = [x*0.01 for x in node_size],
                       node_color = node_color)
nx.draw_networkx_edges(G, pos, 
                       edgelist = edge_list,
                       width = [x*0.01 for x in edge_w])

