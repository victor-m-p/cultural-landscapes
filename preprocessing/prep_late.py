'''
Prepares key documents for the analysis of DRH data. 
This runs after 'expand_data.jl'. 
VMP 2022-02-06: updated and cleaned 
'''

import pandas as pd 
import numpy as np 

# loads 
data_expanded = pd.read_csv('../data/preprocessing/data_expanded.csv')
entry_reference = pd.read_csv('../data/preprocessing/entry_reference.csv')

## merge them
entry_configuration_master = data_expanded.merge(entry_reference, on = 'entry_id', how = 'inner')
entry_configuration_master.to_csv('../data/preprocessing/entry_configuration_master.csv', index = False)

# entry_id/configuration only maxlikelihood
entry_maxlikelihood = entry_configuration_master.sort_values(['entry_id', 'entry_prob'], ascending = False).groupby('entry_id').head(1)
entry_maxlikelihood = entry_maxlikelihood.sort_values('config_id')
entry_maxlikelihood.to_csv('../data/preprocessing/entry_maxlikelihood.csv', index = False)

# configurations only maxlikelihoood 
configurations = np.loadtxt('../data/preprocessing/configurations.txt', dtype = int)
maxlik_config_id = entry_maxlikelihood['config_id'].unique().tolist()
maxlik_config = configurations[maxlik_config_id]
maxlik_config_id = np.array(maxlik_config_id)
np.savetxt('../data/preprocessing/maxlik_configuration_id.txt', maxlik_config_id)
np.savetxt('../data/preprocessing/maxlik_configurations.txt', maxlik_config)
