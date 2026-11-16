'''
Table A2 (and 6). Investigating differences between communities.
VMP 2022-02-07: still very opaque, but it works. 
'''

import numpy as np 
import pandas as pd 
import configuration as cn 

# helper function
def community_weight(network_information, comm_column):
    """Organize configurations and configuration weights by community

    Args:
        network_information (pd.DataFrame): dataframe with "config_id", and "config_prob" columns
        comm_column (str): column with unique community labels

    Returns:
        _type_: _description_
    """
    config_dict = {}
    weight_dict = {}
    communities = network_information[comm_column].unique()
    for comm in communities:
        network_comm = network_information[network_information[comm_column] == comm]
        config_list = [(int(row['config_id']), row['config_prob']) for _, row in network_comm.iterrows()]
        configs = [cn.Configuration(config_id, configurations, configuration_probabilities).configuration for config_id, _ in config_list]
        config_dict[comm] = configs
        weight_dict[comm] = [config_prob for _, config_prob in config_list]
    return config_dict, weight_dict


# get the proper weighting 
def clade_wrangling(c, w, question_reference):
    """Wrange dictionaries with configurations and weights (focal and comparison).

    Args:
        c (dict): configurations in focal and comparison
        w (dict): weights for focal and comparison
        question_reference (pd.DataFrame): questions and their labels

    Returns:
        pd.DataFrame: dataframe with the average answer for focal and comparison
    """
    labels = list(c.keys())
    c1, c2 = c.get(labels[0]), c.get(labels[1])
    w1, w2 = w.get(labels[0]), w.get(labels[1])
    s1, s2 = np.stack(c1, axis=1), np.stack(c2, axis=1)
    s1[s1 == -1] = 0
    s2[s2 == -1] = 0
    wn1, wn2 = np.array(w1) / sum(w1), np.array(w2) / sum(w2)
    bit1 = np.average(s1, axis=1, weights=wn1)
    bit2 = np.average(s2, axis=1, weights=wn2)
    df = pd.DataFrame({f'pct_{labels[0]}': bit1, f'pct_{labels[1]}': bit2})
    df['question_id'] = df.index + 1
    df = df.merge(question_reference, on='question_id', how='inner')
    df['difference'] = np.abs(df[f'pct_{labels[0]}'] - df[f'pct_{labels[1]}'])
    return df

def subset_groups(df, sub_list, remap_dict): 
    """subset group and remap labels

    Args:
        df (pd.DataFrame): dataframe with "comm_label" column
        sub_list (lst): list with original labels.
        remap_dict (dict): dict with mapping between original labels and new labels

    Returns:
        pd.DataFrame: dataframe with subset and remapped labels
    """
    df = df[df['comm_label'].isin(sub_list)]
    df['clade_label'] = df['comm_label'].map(remap_dict)
    return df

# read the data 
question_reference = pd.read_csv('../data/preprocessing/question_reference.csv')
network_information = pd.read_csv('../data/analysis/network_information.csv')
network_information = network_information[['config_id', 'config_prob', 'node_id', 'comm_label', 'recode_comm']]
network_information = network_information.sort_values('node_id').reset_index(drop = True)
configuration_probabilities = np.loadtxt('../data/preprocessing/configuration_probabilities.txt')
configurations = np.loadtxt('../data/preprocessing/configurations.txt', dtype = int)

# 3 communities version (Table 6 in the paper): each of the three top-level groups of the dendrogram
# ('recode_comm', written by fig_4_landscape.py) against the other two. For the published fit these
# are red (Group 1), blue (the two blue clusters) and yellow (the two yellow clusters).
group_network = network_information.assign(comm_label = network_information['recode_comm'])
groups = sorted(group_network['comm_label'].unique())
color_names = {'Group 1': 'Red', 'Group 2': 'Blue', 'Group 3': 'Yellow'}
group_tables = []
for group in groups:
    split = {g: ('1' if g == group else '0') for g in groups}
    sub = subset_groups(group_network, groups, split)
    cdict, wdict = community_weight(sub, 'clade_label')
    d_group = clade_wrangling(cdict, wdict, question_reference)
    d_group['Group'] = group
    d_group['Color'] = color_names.get(group, group)
    d_group = d_group.rename(columns = {'pct_1': 'Avg. S',
                                        'pct_0': 'Avg. O',
                                        'difference': 'Diff',
                                        'question': 'Question'})
    d_group = d_group[['Group', 'Color', 'Question', 'Avg. S', 'Avg. O', 'Diff']]
    group_tables.append(d_group.sort_values('Diff', ascending = False).head(5))

d_concat = pd.concat(group_tables)
d_concat['Diff'] = d_concat['Avg. S']-d_concat['Avg. O']

# to percent and round 
d_concat['Avg. S'] = [round(x*100, 2) for x in d_concat['Avg. S']]
d_concat['Avg. O'] = [round(x*100, 2) for x in d_concat['Avg. O']]
d_concat['Diff'] = [round(x*100, 2) for x in d_concat['Diff']] 

# to latex 
bit_latex_string = d_concat.to_latex(index=False)
with open('../tables/table_A2_clades.txt', 'w') as f: 
    f.write(bit_latex_string)

### check other possible splits ###
# (exploratory; written for the five clusters of the published fit, so only run when there are five)
if network_information['comm_label'].nunique() == 5:
    list_1 = ['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5']
    split_1 = {'Group 1': "1", # red (Group 1)
               'Group 4': "1", # dark yellow (Group 4)
               'Group 5': "1", # light yellow (Group 5)
               'Group 2': "2", # dark blue (Group 2)
               'Group 3': "2"} # light blue (Group 3)

    # warm colors 
    list_2 = ['Group 1', 'Group 4', 'Group 5']
    split_2 = {'Group 1': "1.2", # red (Group 1)
               'Group 4': "1.1", # dark yellow (Group 4) 
               'Group 5': "1.1"} # light yellow (Group 5)

    # cold colors
    list_3 = ['Group 2', 'Group 3']
    split_3 = {'Group 2': '2.2',
               'Group 3': '2.1'}

    # yellows
    list_4 = ['Group 4', 'Group 5']
    split_4 = {'Group 4': '1.1.2',
               'Group 5': '1.1.1'}

    # pick out the four splits
    df_1 = subset_groups(network_information, list_1, split_1)
    df_2 = subset_groups(network_information, list_2, split_2)
    df_3 = subset_groups(network_information, list_3, split_3)
    df_4 = subset_groups(network_information, list_4, split_4)

    # get the dictionaries with configs and weights
    cdict1, wdict1 = community_weight(df_1, 'clade_label')
    cdict2, wdict2 = community_weight(df_2, 'clade_label')
    cdict3, wdict3 = community_weight(df_3, 'clade_label')
    cdict4, wdict4 = community_weight(df_4, 'clade_label')

    # question_reference is necessary for interpretability
    d1 = clade_wrangling(cdict1, wdict1, question_reference)
    d2 = clade_wrangling(cdict2, wdict2, question_reference)
    d3 = clade_wrangling(cdict3, wdict3, question_reference)
    d4 = clade_wrangling(cdict4, wdict4, question_reference)

    # check inference
    pd.set_option('display.max_colwidth', None)
    d1.sort_values('difference', ascending = False).head(5)
    d2.sort_values('difference', ascending = False).head(5)
    d3.sort_values('difference', ascending = False).head(5)
    d4.sort_values('difference', ascending = False).head(5)

    # each community compared to the rest of the communities
    n_groups = 5
    subset_list = ['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5']
    A = np.zeros((n_groups, n_groups), int)
    np.fill_diagonal(A, 1)

    clade_list = []
    for row in A: 
        dct = {group:val for group, val in zip(subset_list, row)}
        df = subset_groups(network_information, subset_list, dct)
        cdict, wdict = community_weight(df, 'clade_label')
        clade = clade_wrangling(cdict, wdict, question_reference)
        clade_list.append(clade)

    g1 = clade_list[0] # g1 = red
    g1.sort_values('difference', ascending = False).head(5)
    g2 = clade_list[1] # g2 = dark blue
    g2.sort_values('difference', ascending = False).head(5)
    g3 = clade_list[2] # g3 = light blue
    g3.sort_values('difference', ascending = False).head(5)
    g4 = clade_list[3] # g4 = dark yellow
    g4.sort_values('difference', ascending = False).head(5)
    g5 = clade_list[4] # g5 = light yellow
    g5.sort_values('difference', ascending = False).head(5)

