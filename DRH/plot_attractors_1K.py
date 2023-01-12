'''
VMP 2023-01-08:
Plot attractors for n = 1, n = 2, n = 3
'''

# COGSCI23
import pandas as pd 
import matplotlib.pyplot as plt 
import networkx as nx 
import numpy as np 
import operator 

def plot_net(d, n, graph_type, suptitle, filename, 
             edge_multiplier, node_multiplier, pos = False,
             GCC = True, remove_self = False,
             k = 1, alpha = True, in_degree = False,
             GCC_o = False, node_labels = False): 
    
    # n = 2
    d = d.sort_values('weight').groupby('config_from').tail(n)

    if remove_self: 
        d = d[d['config_from'] != d['config_to']]

    # try to just plot this as is ...
    G = nx.from_pandas_edgelist(d,
                                source = 'config_from',
                                target = 'config_to',
                                edge_attr = 'weight',
                                create_using=graph_type)

    # only GCC
    if str(GCC).isnumeric(): # find better approach
        print('GCC')
        if graph_type == nx.DiGraph:
            H = G.to_undirected()
            Gcc = sorted(nx.connected_components(H), key=len, reverse=True)
        else: 
            Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(Gcc[GCC])

    # only component with most observed states
    if str(GCC_o).isnumeric():
        print('GCC_o')
        H = G.to_undirected()
        Gcc = sorted(nx.connected_components(H), key = len, reverse = True)
        max_dct = {}
        for num, ele in enumerate(Gcc): 
            G_ele = G.subgraph(Gcc[num])
            uniq_from = np.unique([x for x, y, z in G_ele.edges(data = True)])
            n_from = len(uniq_from)
            max_dct[num] = n_from
        sorted_d = dict(sorted(max_dct.items(), key=operator.itemgetter(1),reverse=True))
        n = list(sorted_d.keys())[GCC_o]
        G = G.subgraph(Gcc[n])

    # edgeweight 
    edge_weight = dict(nx.get_edge_attributes(G, 'weight'))
    edge_list = []
    edge_w = []
    for x, y in edge_weight.items(): 
        edge_list.append(x)
        edge_w.append(y)
        
    # degree 
    if in_degree: 
        degree = dict(G.in_degree(weight = 'weight'))
    else: 
        degree = dict(G.degree(weight = 'weight')) # new change
    
    node_list = []
    node_deg = []
    for x, y in degree.items(): 
        node_list.append(x)
        node_deg.append(y)

    # color 
    uniq_from = np.unique([x for x, y, z in G.edges(data = True)])
    node_color = ['tab:red' if x in uniq_from else 'tab:orange' for x in node_list]

    # position 
    if not pos: 
        n = len(G.nodes())
        pos = nx.spring_layout(G, weight = 'weight',
                               k = k/np.sqrt(n),
                               seed = 4)

    # alpha 
    if alpha: 
        alpha_val = [x/max(edge_w)*0.8 for x in edge_w]
    else: 
        alpha_val = 1
        
    # plot 
    fig, ax = plt.subplots(dpi = 300)
    plt.axis('off')
    nx.draw_networkx_nodes(G, pos, 
                        nodelist = node_list, 
                        node_size = [x*node_multiplier for x in node_deg],
                        node_color = node_color,
                        linewidths = 0.5,
                        edgecolors = 'black')
    nx.draw_networkx_edges(G, pos, 
                        edgelist = edge_list,
                        width = [x*edge_multiplier for x in edge_w],
                        alpha = alpha_val,
                        edge_color = 'tab:blue')
    if node_labels: 
        labels = {}
        for i in G.nodes():
            label = node_labels.get(i)
            if label: 
                labels[i] = label
            else: 
                labels[i] = ''
        nx.draw_networkx_labels(G, pos, 
                                labels = labels)
    plt.suptitle(f'{suptitle}', size = large_text)
    plt.savefig(f'../fig/COGSCI23/networks/{filename}.pdf')


# prep labels 
focus_list = ['Pythagorean', #y
              'Peyote', #y
              'Tsonga', #y
              'Religion in Mesopotamia', #
              'Roman Imperial Cult', #y
              'Free Methodist Church', #
              'Jehovah', #y
              'Calvinism', #y
              'Jesuits', #y
              'Ancient Egypt - the Ramesside', #y
              'Islam in Aceh', #y
              'Wogeo', #y
              'Sokoto',
              'Cistercians'] #y
entry_maxlikelihood = pd.read_csv('../data/analysis/entry_maxlikelihood.csv')
entry_list = []
for focus_entry in focus_list:
    entry_sub = entry_maxlikelihood[entry_maxlikelihood['entry_drh'].str.contains(focus_entry)]
    entry_list.append(entry_sub)
annotations = pd.concat(entry_list)
annotations = {
    385536: 'Pythagoreanism',
    360960: 'Peyote',
    634758: 'Tsonga',
    769926: 'Mesopotamia',
    769927: 'Roman',
    362368: 'Free Methodist',
    362374: 'Jehovah',
    886662: 'Calvinism',
    1027974: 'Jesuits in Britain',
    1027975: 'Ancient Egypt',
    1025927: 'Islam in Aceh',
    1041926: 'Wogeo',
    1025538: 'Sokoto',
    1025926: 'Cistercians'
}

# setup 
small_text, large_text = 12, 18

#### SUBSAMPLE APPROACH ####
d_edgelist = pd.read_csv('../data/COGSCI23/evo_clean/subsample.csv')

### t = 10 ###
d_edgelist = d_edgelist[d_edgelist['t_to'] == 11]
d_edgelist_size = d_edgelist.groupby(['config_from', 'config_to']).size().reset_index(name = 'weight')

### t = 10 ###
# with self-loops
plot_net(d = d_edgelist_size,
         n = 3,
         graph_type = nx.DiGraph, 
         suptitle = 't = 10',
         filename = 't_10_n_3',
         edge_multiplier = 0.02,
         node_multiplier = 1,
         remove_self = False,
         GCC_o = 0,
         alpha = False,
         k = 1,
         node_labels = annotations)

# without self-loops
plot_net(d = d_edgelist_size,
         n = 3,
         graph_type = nx.DiGraph, 
         suptitle = 't = 10',
         filename = 't_10_n_3_ns',
         edge_multiplier = 0.04,
         node_multiplier = 1,
         remove_self = True,
         GCC_o = 0,
         alpha = False,
         k = 1,
         node_labels = annotations)

# only data-states
d_edgelist_from = d_edgelist_size[['config_from']].drop_duplicates()
d_edgelist_from = d_edgelist_from.rename(columns = {'config_from': 'config_to'})
d_edgelist_obs = d_edgelist_size.merge(d_edgelist_from, on = 'config_to', how ='inner')

# with self-loops
plot_net(d = d_edgelist_obs,
         n = 3,
         graph_type = nx.DiGraph, 
         suptitle = 't = 10',
         filename = 't_10_n_3_ns_obs',
         edge_multiplier = 0.02,
         node_multiplier = 0.3,
         remove_self = True,
         GCC_o = 0,
         alpha = False,
         k = 1,
         node_labels = annotations)

### t = 100 ###
d_edgelist = pd.read_csv('../data/COGSCI23/evo_clean/subsample.csv')
d_edgelist = d_edgelist[d_edgelist['t_to'] == 91]
d_edgelist_size = d_edgelist.groupby(['config_from', 'config_to']).size().reset_index(name = 'weight')
d_edgelist_from = d_edgelist_size[['config_from']].drop_duplicates()
d_edgelist_from = d_edgelist_from.rename(columns = {'config_from': 'config_to'})
d_edgelist_obs = d_edgelist_size.merge(d_edgelist_from, on = 'config_to', how ='inner')

# now we can connect with n = 2
# remove self-loops 
plot_net(d = d_edgelist_obs,
         n = 2,
         graph_type = nx.DiGraph, 
         suptitle = 't = 100',
         filename = 't_100_n_2_obs_ns',
         edge_multiplier = 0.02,
         node_multiplier = 0.3,
         remove_self = True,
         GCC_o = 0,
         alpha = False,
         k = 1,
         node_labels = annotations)

# do not remove self-loops
plot_net(d = d_edgelist_obs,
         n = 2, # often self / other
         graph_type = nx.DiGraph, 
         suptitle = 't = 100',
         filename = 't_100_n_2_obs',
         edge_multiplier = 0.02,
         node_multiplier = 0.3,
         remove_self = False,
         GCC_o = 0,
         alpha = False,
         k = 1,
         node_labels = annotations)

# based on the weighted 
weighted = pd.read_csv('../data/COGSCI23/evo_clean/weighted.csv')
plot_net(d = weighted,
         n = 2, # often self / other
         graph_type = nx.DiGraph, 
         suptitle = 'weighted',
         filename = 't_w_n_2',
         edge_multiplier = 0.001,
         node_multiplier = 0.01,
         remove_self = False,
         GCC_o = 0,
         alpha = False,
         k = 1,
         node_labels = annotations)

# remove self-loops 
plot_net(d = weighted,
         n = 2, # often self / other
         graph_type = nx.DiGraph, 
         suptitle = 'weighted',
         filename = 't_w_n_2_ns',
         edge_multiplier = 0.001,
         node_multiplier = 0.01,
         remove_self = True,
         GCC_o = 0,
         alpha = False,
         k = 1,
         node_labels = annotations)

# with other states n = 3
plot_net(d = weighted,
         n = 3, # often self / other
         graph_type = nx.DiGraph, 
         suptitle = 'weighted',
         filename = 't_w_n_3_ns',
         edge_multiplier = 0.001,
         node_multiplier = 0.01,
         remove_self = True,
         GCC_o = 0,
         alpha = False,
         k = 1,
         node_labels = annotations)

# without other states = 3
d_edgelist_from = weighted[['config_from']].drop_duplicates()
d_edgelist_from = d_edgelist_from.rename(columns = {'config_from': 'config_to'})
d_edgelist_obs = d_edgelist_size.merge(d_edgelist_from, on = 'config_to', how ='inner')
plot_net(d = d_edgelist_obs,
         n = 3, # often self / other
         graph_type = nx.DiGraph, 
         suptitle = 'weighted',
         filename = 't_w_n_3_ns_obs',
         edge_multiplier = 0.01,
         node_multiplier = 0.01,
         remove_self = True,
         GCC_o = 0,
         alpha = False,
         k = 1,
         node_labels = annotations)

# based on the shifted 
shifted = pd.read_csv('../data/COGSCI23/evo_clean/shift_weighted.csv')
plot_net(d = shifted,
         n = 2, # often self / other
         graph_type = nx.DiGraph, 
         suptitle = 'shifted',
         filename = 't_s_n_2',
         edge_multiplier = 0.1,
         node_multiplier = 3,
         remove_self = False,
         GCC_o = 0,
         alpha = False,
         k = 1,
         node_labels = annotations)

plot_net(d = shifted,
         n = 2, # often self / other
         graph_type = nx.DiGraph, 
         suptitle = 'shifted',
         filename = 't_s_n_2_ns',
         edge_multiplier = 0.3,
         node_multiplier = 10,
         remove_self = True,
         GCC_o = 0,
         alpha = False,
         k = 1,
         node_labels = annotations)
