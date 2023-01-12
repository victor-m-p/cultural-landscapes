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
             GCC_o = False): 
    
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
        print('yes')
        if graph_type == nx.DiGraph:
            H = G.to_undirected()
            Gcc = sorted(nx.connected_components(H), key=len, reverse=True)
        else: 
            Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(Gcc[GCC])

    # only component with most observed states
    if str(GCC_o).isnumeric():
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
    plt.suptitle(f'{suptitle}', size = large_text)
    plt.savefig(f'../fig/COGSCI23/networks/{filename}.pdf')

# setup 
small_text, large_text = 12, 18

# load edgelist (bit funky concept)
d_edgelist = pd.read_csv('../data/COGSCI23/evo_clean/overview.csv')
d_edgelist_10 = d_edgelist[d_edgelist['t_to'] == 101]
d_edgelist_10_max = d_edgelist_10.groupby(['config_from', 'config_to']).size().reset_index(name = 'weight')

# n = 1 
plot_net(d = d_edgelist_10_max,
         n = 1,
         graph_type = nx.DiGraph, 
         suptitle = 'Directed (n = 1)',
         filename = 't101_n1_directed',
         edge_multiplier = 0.01,
         node_multiplier = 1,
         GCC = 0)

# n = 2
plot_net(d = d_edgelist_10_max,
         n = 2, 
         graph_type = nx.DiGraph, 
         suptitle = 'Directed (n = 2)',
         filename = 't101_n2_directed',
         edge_multiplier = 0.01,
         node_multiplier = 1,
         k = 8,
         GCC = 0)

# n = 3
plot_net(d = d_edgelist_10_max,
         n = 3, 
         graph_type = nx.DiGraph, 
         suptitle = 'Directed (n = 3)',
         filename = 't101_n3_directed',
         edge_multiplier = 0.1,
         node_multiplier = 1,
         k = 10,
         GCC = 0)

#### try the sum version ####

# this does not work so well because we have a ton of 
# self-loops which means that we do not get it properly
# connected. 
# try to do it just for the whole distribution instead 
d_edgelist_sum = pd.read_csv('../data/COGSCI23/evo_clean/weighted.csv')

# n = 2 (good)
plot_net(d = d_edgelist_sum,
         n = 2, 
         graph_type = nx.DiGraph, 
         suptitle = 'Undirected (n = 2)',
         filename = 'tall_n2_directed',
         edge_multiplier = 0.001,
         node_multiplier = 0.01,
         k = 30,
         GCC_o = 0)

# n = 3 (good)
plot_net(d = d_edgelist_sum,
         n = 3, 
         graph_type = nx.DiGraph, 
         suptitle = 'Directed (n = 3)',
         filename = 'tall_n3_directed',
         edge_multiplier = 0.001,
         node_multiplier = 0.01,
         k = 50)

# we are never able to actually connect the graph
# we probably could connect the graph if we run
# for a really long time (or many simulations)
# but this is already 100 * 100
# we could also plot all of the clusters 
# and potentially (manually) connect them by
# the most "probable" connection route. 
plot_net(d = d_edgelist_sum,
         n = 10, # same as all 
         graph_type = nx.DiGraph, 
         suptitle = 'Directed (n = 10)',
         filename = 't_all_n10_directed',
         edge_multiplier = 0.001,
         node_multiplier = 0.01,
         k = 110) # notice very high k

# try to just max it out 
# we still only have a small subset 
# connected here 
# they just never get "far enough"
# perhap we should run n = 1000 
plot_net(d = d_edgelist_sum,
         n = 1000, # same as all 
         graph_type = nx.DiGraph, 
         suptitle = 'Directed (n = GCC)',
         filename = 'tall_nall_directed',
         edge_multiplier = 0.002,
         node_multiplier = 0.01,
         remove_self = True,
         k = 50, # notice very high k
         alpha = False) 

# try with in degree
# empty positions = unstable 
# end up different places a lot 
# and no-one else end up in their position
plot_net(d = d_edgelist_sum,
         n = 1000, 
         graph_type = nx.DiGraph, 
         suptitle = 'Directed (n = GCC)',
         filename = 'tall_nall_directed_indegree',
         edge_multiplier = 0.0005,
         node_multiplier = 0.03,
         remove_self = False,
         k = 1, # 10 good
         alpha = False,
         in_degree = True)

# new strategy
plot_net(d = d_edgelist_sum,
         n = 1000, 
         graph_type = nx.DiGraph, 
         suptitle = 'Directed (n = GCC)',
         filename = 'test',
         edge_multiplier = 0.0005,
         node_multiplier = 0.01,
         remove_self = False,
         k = 100, # 10 good
         alpha = False,
         in_degree = True,
         GCC = False,
         GCC_o = 2)

## just seems weird that the ones 
## that there is so much flow to 
## would not themselves flow somewhere
## else; and that there would not be
## more flow to the actual datastates?


########## labeling ###########

# find interesting states to label
pd.set_option('display.max_colwidth', None)
d_maxlik = pd.read_csv('/home/vpoulsen/humanities-glass/data/analysis/entry_maxlikelihood.csv')
d_maxlik = d_maxlik[['config_id', 'entry_drh']].drop_duplicates()
d_maxlik = d_maxlik.rename(columns = {'config_id': 'config_from'})
d_maxlik = d_maxlik.groupby('config_from')['entry_drh'].unique().reset_index(name = 'religions')

# merge it with some of our old data
# i.e. so that we can reference with our 
# main plot (perhaps also community for those where it applies)
network_landscape = pd.read_csv('../data/analysis/network_information_enriched.csv')
network_landscape = network_landscape[['config_id', 'config_prob', 'node_id', 'comm_color', 'comm_label']]
network_landscape = network_landscape.rename(columns = {'config_id': 'config_from'})

network_info = network_landscape.merge(d_maxlik, 
                                       on = 'config_from', 
                                       how = 'inner')

# try to plot some node information
network_info = network_info.sort_values('config_prob', ascending = False)

#d_annot = pd.DataFrame.from_dict(annotated, orient = 'index').reset_index()
#d_annot = d_annot.rename(columns = {'index': 'config_from',
 #                                   '0': 'label'})
#network_info = network_info.merge(d_annot, on = 'config_from', how = 'inner')

# try to plot something here 
pos_undirected_2 = get_pos(d = d_edgelist_10_max,
                           n = 2,
                           graph_type = nx.Graph,
                           bias = 1)

# basically using the function
n = 2 
d = d_edgelist_10_max
graph_type = nx.DiGraph
pos = pos_undirected_2
suptitle = 'Mixed (n = 2) labels'
filename = 't101_n2_mixed_labels'

# n = 2
d = d.sort_values('weight').groupby('config_from').tail(n)

# try to just plot this as is ...
G = nx.from_pandas_edgelist(d,
                            source = 'config_from',
                            target = 'config_to',
                            edge_attr = 'weight',
                            create_using=graph_type)

# only the main component
if graph_type == nx.DiGraph:
    H = G.to_undirected()
    Gcc = sorted(nx.connected_components(H), key=len, reverse=True)
else: 
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
G = G.subgraph(Gcc[0])

# edgeweight 
edge_weight = dict(nx.get_edge_attributes(G, 'weight'))
edge_list = []
edge_w = []
for x, y in edge_weight.items(): 
    edge_list.append(x)
    edge_w.append(y)
    
# degree 
degree = dict(G.degree())
node_list = []
node_deg = []
for x, y in degree.items(): 
    node_list.append(x)
    node_deg.append(y)

from collections import OrderedDict
sort_dict = OrderedDict(sorted(degree.items(), 
                                  key=lambda kv: kv[1], reverse=True))
sort_conf = [x for x in sort_dict]
sort_conf = sort_conf[0:15]
sort_conf = pd.DataFrame(sort_conf, columns = ['config_from'])
network_top = network_info.merge(sort_conf, on = 'config_from', how = 'inner')
network_top[['config_from', 'religions']].head(20)

annotated = {1025926: 'Cistercians*',
             1027974: 'Jesuits*',
             1027975: 'Ancient Egypt*',
             362374: 'Jehovah',
             1025927: 'Islam Aceh*',
             634758: 'Tsonga*',
             1044358: 'Vaisnava',
             1017734: 'Tao',
             769926: 'Mesopotamia*',
             362368: 'No Debt*',
             634754: 'Badjau',
             634752: 'Kapauku',
             361984: 'Messalians',
             385536: 'Pythagorean',
             377344: '?'}

# labels first 
labels = {}
for i in G.nodes(): 
    if i in [x for x in annotated.keys()]: 
        labels[i] = annotated.get(i)
    else: 
        labels[i] = ''


# position 
if not pos: 
    n = len(G.nodes())
    pos = nx.spring_layout(G, weight = 'weight',
                        k = 1/np.sqrt(n),
                        seed = 4)

# plot 
fig, ax = plt.subplots(dpi = 300)
plt.axis('off')
nx.draw_networkx_nodes(G, pos, 
                    nodelist = node_list, 
                    node_size = [x for x in node_deg],
                    node_color = 'tab:orange',
                    linewidths = 0.5,
                    edgecolors = 'black')
nx.draw_networkx_edges(G, pos, 
                    edgelist = edge_list,
                    width = [x*0.1 for x in edge_w],
                    alpha = [x/max(edge_w)*0.5 for x in edge_w],
                    edge_color = 'tab:blue')
nx.draw_networkx_labels(G, pos, 
                        labels = labels,
                        font_size = 6)
plt.suptitle(f'{suptitle}', size = large_text)
plt.savefig(f'../fig/COGSCI23/networks/{filename}.pdf')
