'''
VMP 2022-12-15: 
Prepares key documents for the analysis of DRH data. 
This runs after 'expand_data.jl'. 
'''

import pandas as pd 
import numpy as np 

# entry_id/configuration master dataset
## load relevant data-sets 
data_expanded = pd.read_csv('../data/analysis/data_expanded.csv')
entry_reference = pd.read_csv('../data/analysis/entry_reference.csv')

## merge them
entry_configuration_master = data_expanded.merge(entry_reference, on = 'entry_id', how = 'inner')
entry_configuration_master.to_csv('../data/analysis/entry_configuration_master.csv', index = False)

# entry_id/configuration only maxlikelihood
'''just takes one if there is a tie (which there should generally not be'''
entry_maxlikelihood = entry_configuration_master.sort_values(['entry_id', 'entry_prob'], ascending = False).groupby('entry_id').head(1)
entry_maxlikelihood = entry_maxlikelihood.sort_values('config_id')
entry_maxlikelihood.to_csv('../data/analysis/entry_maxlikelihood.csv', index = False)

# configurations only maxlikelihoood 
n_nodes = 20 
from fun import bin_states 
allstates = bin_states(n_nodes)

maxlik_config_id = entry_maxlikelihood['config_id'].unique().tolist()
maxlik_config = allstates[maxlik_config_id]

# save both
maxlik_config_id = np.array(maxlik_config_id)
np.savetxt('../data/analysis/maxlik_configuration_id.txt', maxlik_config_id)
np.savetxt('../data/analysis/maxlik_configurations.txt', maxlik_config)
