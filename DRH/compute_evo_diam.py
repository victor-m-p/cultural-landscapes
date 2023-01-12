'''
VMP 2023-01-05: 
COGSCI23
come back and save the plots.
also just save the diam....
'''
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 

# plot setup
large_text = 18
small_text = 12

# usual setup
configuration_probabilities = np.loadtxt('../data/analysis/configuration_probabilities.txt')

# generate all states
n_nodes = 20
from fun import bin_states 
configurations = bin_states(n_nodes) 

# read the curated data 
d_edgelist = pd.read_csv('../data/COGSCI23/evo_clean/overview.csv')

# expand in a different way ...
## okay, first just for n = 10
d_edgelist_10 = d_edgelist[d_edgelist['t_to'] == 100]

attractors = d_edgelist_10.groupby(['config_from'])['config_to'].unique().reset_index(name = 'attractors')

# test 
def hamming_distance(X):
    '''https://stackoverflow.com/questions/42752610/python-how-to-generate-the-pairwise-hamming-distance-matrix'''
    return (X[:, None, :] != X).sum(2)

# ...
hamming_list = []
for _, row in attractors.iterrows():
    config_id_to = row['attractors']
    config_id_from = row['config_from']
    config_to = configurations[config_id_to]
    hamming = hamming_distance(config_to)
    d_hamming = pd.DataFrame(hamming, 
                             columns = [str(x) for x in config_id_to],
                             index = config_id_to)
    d_hamming['config_from'] = d_hamming.index 
    d_hamming_melt = d_hamming.melt(id_vars = ['config_from'],
                                    value_vars = [str(x) for x in config_id_to],
                                    var_name = 'config_to',
                                    value_name = 'hamming_dist')
    d_hamming_melt['config_to'] = [int(x) for x in d_hamming_melt['config_to']]
    d_hamming_melt = d_hamming_melt[d_hamming_melt['hamming_dist'] <= d_hamming_melt['hamming_dist']]
    d_hamming_melt['config_start'] = config_id_from
    hamming_list.append(d_hamming_melt)

d_hamming = pd.concat(hamming_list)
 
## weight = 100 
config_weight = d_edgelist_10.groupby(['config_from', 'config_to']).size().reset_index(name = 'weight')

### (1) need to go two ways here ###
config_weight = config_weight.rename(columns = {'config_from': 'config_start',
                                                'weight': 'weight_to'})
d_hamming = d_hamming.merge(config_weight, on = ['config_start', 'config_to'], how = 'inner')

### (2) need to go two ways here ###
config_weight = config_weight.rename(columns = {'config_to': 'config_from',
                                                'weight_to': 'weight_from'})
d_hamming = d_hamming.merge(config_weight, on = ['config_start', 'config_from'])
d_hamming['w_mult'] = d_hamming['weight_to'] * d_hamming['weight_from']
d_hamming['w_norm'] = d_hamming['w_mult']/(100*100)
d_hamming['h_diam'] = d_hamming['hamming_dist']*d_hamming['w_norm']
h_diameter = d_hamming.groupby('config_start')['h_diam'].sum().reset_index(name = 'h_diam')

# visualize diameter
fig, ax = plt.subplots(dpi = 300)
sns.kdeplot(data = h_diameter,
            x = 'h_diam',
            bw_adjust = 1)
# fix legend 
plt.suptitle('Hamming diameter', size = large_text)
plt.xlabel('Hamming diameter', size = small_text)
plt.show();
#plt.savefig('../fig/COGSCI23/evo/hamming_all_across.pdf')
