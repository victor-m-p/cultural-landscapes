'''
This script creates the table with the annotations for Figure 4(A).
Does not include the DRH ID (added in post). 
'''

import pandas as pd 

# load data
network_information = pd.read_csv('../data/analysis/network_information.csv')
annotations = pd.read_csv('../data/analysis/landscape_annotations.csv')

# wrangle data
annotations = annotations[['entry_name', 'config_id']].rename({'entry_name': 'entry_short'}, axis = 1)
network_information = network_information[['recode_comm', 'entry_name', 'entry_id', 'config_id', 'config_prob']]
annotation_table = annotations.merge(network_information, on = 'config_id', how = 'inner')
annotation_table = annotation_table.sort_values(['recode_comm', 'config_prob'], ascending = [True, False])
annotation_table = annotation_table[['recode_comm', 'entry_short', 'entry_name']]
annotation_table = annotation_table.rename({
    'recode_comm': 'Group',
    'entry_short': 'Entry Name (Short)',
    'entry_name': 'Entry name (DRH)'})

# save table
annotation_latex = annotation_table.style.hide(axis = 'index').to_latex()
with open('../tables/table_5_annotations.txt', 'w') as f: 
    f.write(annotation_latex)
