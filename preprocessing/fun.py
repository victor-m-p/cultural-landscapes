'''
Helper functions for the analysis of DRH data (and preprocessing)
VMP 2022-02-06: refactored with chatGPT and docstrings. 
'''

import numpy as np
import itertools 
import pandas as pd 
import networkx as nx 
import matplotlib.pyplot as plt 
from matplotlib.colors import rgb2hex
from tqdm import tqdm 

# taken from coniii enumerate
def fast_logsumexp(X, coeffs=None):
    """correlation calculation in Ising equation

    Args:
        X (np.Array): terms inside logs
        coeffs (np.Array, optional): factors in front of exponential. Defaults to None.

    Returns:
        float: sum of exponentials
    """
    Xmx = max(X)
    if coeffs is None:
        y = np.exp(X-Xmx).sum()
    else:
        y = np.exp(X-Xmx).dot(coeffs)

    if y<0:
        return np.log(np.abs(y))+Xmx, -1.
    return np.log(y)+Xmx, 1.

# still create J_combinations is slow for large number of nodes
def p_dist(h, J):
    """return probabilities for 2**h states

    Args:
        h (np.Array): local fields
        J (np.Array): pairwise couplings. 

    Returns:
        np.Array: probabilities for all configurations
    """
    n_nodes = len(h)
    hJ = np.concatenate((h, J))
    h_combinations = np.array(list(itertools.product([1, -1], repeat = n_nodes)))
    J_combinations = np.array([list(itertools.combinations(i, 2)) for i in h_combinations])
    J_combinations = np.add.reduce(J_combinations, 2)
    J_combinations[J_combinations != 0] = 1
    J_combinations[J_combinations == 0] = -1
    condition_arr = np.concatenate((h_combinations, J_combinations), axis = 1)
    flipped_arr = hJ * condition_arr
    summed_arr = np.sum(flipped_arr, axis = 1)
    logsumexp_arr = fast_logsumexp(summed_arr)[0]
    Pout = np.exp(summed_arr - logsumexp_arr)
    return Pout[::-1]

def bin_states(n, sym=True):
    """generate 2**n possible configurations

    Args:
        n (int): number of questions (features)
        sym (bool, optional): symmetric system. Defaults to True.

    Returns:
        np.Array: 2**n configurations 
    """
    v = np.array([list(np.binary_repr(i, width=n)) for i in range(2**n)]).astype(int)
    if sym is False:
        return v
    return v*2-1

# '''https://stackoverflow.com/questions/42752610/python-how-to-generate-the-pairwise-hamming-distance-matrix'''
def hamming_distance(X):
    """Calculate Hamming distance

    Args:
        X (np.Array): Array of binary values (rows = configurations, columns = binary answers)

    Returns:
        np.Array: hamming distance (rows * rows)
    """
    return (X[:, None, :] != X).sum(2)

def top_n_idx(N, p, ind_colname='config_id', prob_colname='config_prob'):
    """get the most probable N states

    Args:
        N (int): number of configurations wanted
        p (np.Array): array of probabilities for configurations
        ind_colname (str, optional): desired column name for index column. Defaults to 'config_id'.
        val_colname (str, optional): desired column name for probability column. Defaults to 'config_prob'.

    Returns:
        pd.DataFrame: Dataframe with most probable N states, their index and probability
    """
    N = N+1
    val_cutoff = np.sort(p)[-N]
    p_ind = np.argwhere(p > val_cutoff).flatten()
    p_val = p[p_ind]
    data_out = pd.DataFrame({ind_colname: p_ind, prob_colname: p_val}).nlargest(N, prob_colname)
    return data_out.reset_index(drop = True)

def sort_edge_attributes(Graph, weight_attribute, filter_attribute, scaling = 1): 
    """Return list of edges and list of edge weights, both sorted by edge weights (filtered, scaled)

    Args:
        Graph (nx.Graph): networkx graph object with weight_attribute and filter_attribute
        weight_attribute (str): weight attribute (could be other attribute, but should be numeric)
        filter_attribute (str): filter attribute (e.g. only hamming distance == 1).
        scaling (numeric): scaling of weights (for visualization purposes). Defaults to 1 (not scaled).

    Returns:
        lists: list of edges, list of edge weights. 
    """
    ## get edge attributes
    edge_weight = nx.get_edge_attributes(Graph, weight_attribute)
    edge_hdist = nx.get_edge_attributes(Graph, filter_attribute)

    ## sort edge weights by value
    edge_weights_sorted = sorted(edge_weight.items(), key=lambda x: x[1])
    edge_weights_filtered = [(k, v) for k, v in edge_weights_sorted if edge_hdist[k] == 1]
    
    # scale edge weights
    edge_weights_scaled = [(k, v * scaling) for k, v in edge_weights_filtered]
    
    # return edge list and scaled weights
    edge_list = [k for k, _ in edge_weights_scaled]
    edge_weights = [v for _, v in edge_weights_scaled]
    
    return edge_list, edge_weights

def sort_node_attributes(Graph, sorting_attribute, value_attribute):
    """Sort nodes based on attribute and return sorted node list and value list

    Args:
        Graph (nx.Graph): networkx graph object
        sorting_attribute (str): string containing sorting attribute
        value_attribute (str): string containing value attribute

    Returns:
        lst: list of sorted nodes and values
    """
    sorting_attr = nx.get_node_attributes(Graph, sorting_attribute)
    nodelist_sorted = [k for k, v in sorted(sorting_attr.items(), key=lambda item: item[1])]
    value_attr = nx.get_node_attributes(Graph, value_attribute)
    value_sorted = [v for k, v in sorted(value_attr.items(), key=lambda pair: nodelist_sorted.index(pair[0]))]
    return nodelist_sorted, value_sorted

def hamming_edges(N, H_distances):
    """Get edgelist with hamming distance for the top N states

    Args:
        N (int): Number of configurations
        H_distances (np.Array): Array of hamming distances (shape N * N)

    Returns:
        _type_: _description_
    """
    col_names = [f'hamming{x}' for x in range(N)]
    df = pd.DataFrame(H_distances, columns=col_names)
    df['node_x'] = df.index
    df = pd.wide_to_long(df, stubnames="hamming", i='node_x', j='node_y').reset_index()
    df = df[df['node_x'] != df['node_y']]
    df.drop_duplicates(inplace=True)
    return df

def edge_strength(G, nodestrength):
    """Add multiplicative and additive edge strength based on node attribute

    Args:
        G (nx.Graph): networkx graph object
        nodestrength (str): node attribute (numeric)

    Returns:
        nx.Graph: New graph with added edge attributes
    """
    Gcopy = G.copy()
    for edge_x, edge_y in Gcopy.edges():
        pmass_x = Gcopy.nodes[edge_x][nodestrength]
        pmass_y = Gcopy.nodes[edge_y][nodestrength]
        Gcopy.edges[(edge_x, edge_y)].update({
            'pmass_mult': pmass_x * pmass_y,
            'pmass_add': pmass_x + pmass_y
        })
    return Gcopy

def match_nodeid(df, node_id, N = 10):
    """get the entries associated with a particular node_id in descending order of probability

    Args:
        df (pd.DataFrame): dataframe with columns "node_id", "entry_name", "entry_id", "entry_prob"
        node_id (int): node identifier, typically integer.
        N (int, optional): number of rows to print. Defaults to 10.
    """
    df = df[df['node_id'] == node_id][['entry_name', 'entry_id', 'entry_prob']]
    df = df.sort_values('entry_prob', ascending = False)
    print(df.head(N))

def match_substring(df, entry_name, N = 10):
    """Returns entries containing substring

    Args:
        df (pd.DataFrame): dataframe with column "entry_name"
        entry_name (string): substring to match in "entry_name" column
        N (int, optional): number of rows to display. Defaults to 10.
    """
    df = df[df['entry_name'].str.contains(entry_name)]
    print(df.head(N))
    

def hamming_neighbors_N_removed(N, config_id_focal, configurations, configuration_probabilities): 
    """return Hamming neighbors within hamming distance N

    Args:
        N (int): Hamming distance from focal node (config_id_focal)
        config_id_focal (int): index of focal configuration
        configurations (np.Array): array of configurations
        configuration_probabilities (np.Array): array of probabilities

    Returns:
        pd.DataFrame: dataframe with Hamming neighbors of focal node
    """
    config_focal = configurations[config_id_focal]
    config_prob_focal = configuration_probabilities[config_id_focal]
    lst_neighbors = [(config_id_focal, config_prob_focal, config_id_neighbor, configuration_probabilities[config_id_neighbor], np.count_nonzero(config_focal!=config_neighbor))
                    for config_id_neighbor, config_neighbor in enumerate(configurations) 
                    if np.count_nonzero(config_focal!=config_neighbor) <= N and config_id_focal != config_id_neighbor]
    df_neighbors = pd.DataFrame(
        lst_neighbors,
        columns = ['config_id_focal', 'config_prob_focal', 'config_id_neighbor', 'config_prob_neighbor', 'hamming'] 
    )
    return df_neighbors