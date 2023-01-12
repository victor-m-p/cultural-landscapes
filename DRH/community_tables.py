import numpy as np 
import pandas as pd 
import configuration as cn 

# helper function
def community_weight(network_information, sort_column):
    config_dict = {}
    weight_dict = {}
    for comm in network_information[sort_column].unique(): 
        config_list = []
        weight_list = []
        network_comm = network_information[network_information[sort_column] == comm]
        for _, row in network_comm.iterrows():
            config_id = int(row['config_id'])
            config_prob = row['config_prob']
            CommObj = cn.Configuration(config_id, configurations,
                                       configuration_probabilities)
            conf = CommObj.configuration
            config_list.append(conf)
            weight_list.append(config_prob)
        config_dict[comm] = config_list 
        weight_dict[comm] = weight_list
    return config_dict, weight_dict 

# preprocessing 
## from "plot_landscape_dendrogram.py"
network_information = pd.read_csv('../data/analysis/network_information_enriched.csv')
network_information = network_information[['config_id', 'config_prob', 'node_id', 'comm_label']]
network_information = network_information.sort_values('node_id').reset_index(drop = True)

# preprocessing 
from fun import bin_states 
configuration_probabilities = np.loadtxt('../data/analysis/configuration_probabilities.txt')
n_nodes = 20
configurations = bin_states(n_nodes) 

## split tables ##
# first split 
list_1 = ['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5']
split_1 = {'Group 1': "1", # red (Group 1)
           'Group 4': "1", # dark yellow (Group 4)
           'Group 5': "1", # light yellow (Group 5)
           'Group 2': "2", # dark blue (Group 2)
           'Group 3': "2"} # light blue (Group 3)

## warm colors 
list_2 = ['Group 1', 'Group 4', 'Group 5']
split_2 = {'Group 1': "1.2", # red (Group 1)
           'Group 4': "1.1", # dark yellow (Group 4) 
           'Group 5': "1.1"} # light yellow (Group 5)

## cold colors
list_3 = ['Group 2', 'Group 3']
split_3 = {'Group 2': '2.2',
           'Group 3': '2.1'}

## yellows
list_4 = ['Group 4', 'Group 5']
split_4 = {'Group 4': '1.1.2',
           'Group 5': '1.1.1'}

def subset_groups(df, sub_list, remap_dict): 
    df = df[df['comm_label'].isin(sub_list)]
    df['clade_label'] = [remap_dict.get(x) for x in df['comm_label']]
    return df

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
question_reference = pd.read_csv('../data/analysis/question_reference.csv')

# get the proper weighting 
def clade_wrangling(c, w, question_reference):
    labels_1, labels_2 = [x for x in c.keys()]
    # get values out 
    c1, w1 = c.get(labels_1), w.get(labels_1)
    c2, w2 = c.get(labels_2), w.get(labels_2)
    # stack
    s1, s2 = np.stack(c1, axis = 1), np.stack(c2, axis = 1)
    # recode
    s1[s1 == -1] = 0
    s2[s2 == -1] = 0
    # weights
    wn1, wn2 = np.array(w1)/sum(w1), np.array(w2)/sum(w2)
    # average
    bit1 = np.average(s1, axis = 1, weights = wn1)
    bit2 = np.average(s2, axis = 1, weights = wn2)
    # turn this into dataframes
    df = pd.DataFrame(
        {f'pct_{labels_1}': bit1,
         f'pct_{labels_2}': bit2})
    df['question_id'] = df.index + 1
    # merge with question reference
    df = df.merge(question_reference, on = 'question_id', how = 'inner')
    # difference 
    df = df.assign(difference = lambda x: 
        np.abs(x[f'pct_{labels_1}']-x[f'pct_{labels_2}']))
    return df 

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

### each community against the others ### 
n_groups = 5
subset_list = ['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5']
A = np.zeros((n_groups, n_groups), int)
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

## NB: save the long-versions