'''
COGSCI23
'''
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.stats import entropy

import itertools 
def create_grid(df, c1, c2):
    # create lists for both variables
    l_c1 = df[c1].drop_duplicates().to_list()
    l_c2 = df[c2].drop_duplicates().to_list()
    # create grid and to dataframe 
    l_comb = list(itertools.product(l_c1, l_c2))
    d_comb = pd.DataFrame(l_comb, columns=[c1, c2])
    return d_comb

## preprocessing 
d_edgelist = pd.read_csv('../data/COGSCI23/evo_clean/overview.csv')
d_edgelist_10 = d_edgelist[d_edgelist['t_to'] == 10]
d_edgelist_10 = d_edgelist_10.drop(columns = 't_to')
from_to_grid = create_grid(d_edgelist_10, 'config_from', 'config_to')
d_left = from_to_grid.merge(d_edgelist_10, 
                            on = ['config_from', 'config_to'],
                            how = 'left',
                            indicator = True)
d_left = d_left.rename(columns = {'_merge': 'n'})
d_left = d_left.replace({'n': {'both': 1, 'left_only': 0}})
d_left['n'] = [int(x) for x in d_left['n']]
## to probability 
d_left = d_left.groupby(['config_from', 'config_to'])['n'].sum().reset_index(name = 'total')
d_left['prob'] = d_left['total']/100
## make 100% sure that it is sorted correctly 
d_left = d_left.sort_values(['config_from', 'config_to'],
                            ascending = [True, True])
## now to array
d_probs = d_left.drop(columns = 'total')
d_wide = d_probs.pivot(index = 'config_from',
                       columns = 'config_to',
                       values = 'prob')
prob_array = d_wide.to_numpy()

# just entropy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
H = entropy(prob_array.T)
d_H = d_left[['config_from']].drop_duplicates()
d_H['entropy'] = H

## save entropy
d_H.to_csv('../data/COGSCI23/evo_entropy/entropy_10.csv', index = False)

# JSD
from scipy.spatial import distance
array_len = len(prob_array)
# could probably do this smarter
JSD_list = []
for i in range(array_len): 
    for j in range(array_len): 
        if i < j: 
            jdist = distance.jensenshannon(prob_array[i], prob_array[j])
            JSD_list.append((i, j, jdist))
## to pandas
d_JSDx = pd.DataFrame(JSD_list, columns = ['i', 'j', 'JSD'])

## merge from both sides
d_JSDy = d_left[['config_from']].drop_duplicates().reset_index(drop=True)
d_JSDy['i'] = d_JSDy.index
d_JSDy = d_JSDy.rename(columns = {'config_from': 'config_i'})
d_JSD = d_JSDx.merge(d_JSDy, on = 'i', how = 'inner')
d_JSDy = d_JSDy.rename(columns = {'i': 'j',
                                  'config_i': 'config_j'})
d_JSD = d_JSD.merge(d_JSDy, on = 'j', how = 'inner')

## save JSD 
d_JSD.to_csv('../data/COGSCI23/evo_entropy/JSD_10.csv', index = False)