'''
Produces table A3 and A4.
'''

# loads 
import pandas as pd
pd.set_option('display.max_colwidth', None)

# loads 
network_information = pd.read_csv('../data/analysis/network_information.csv')
entry_reference = pd.read_csv('../data/preprocessing/entry_reference.csv')
config_entry_overlap = pd.read_csv('../data/analysis/top_configurations_overlap.csv')

# table A3 
new_groups = {
    'Group 1': 'Group 1', # red
    'Group 2': 'Group 2.2', # dark blue
    'Group 3': 'Group 2.1', # light blue
    'Group 4': 'Group 3.2', # dark yellow
    'Group 5': 'Group 3.1'} # light yellow

sub_network = network_information[['config_id', 'comm_label']]
config_entry_comm = (config_entry_overlap
                     .merge(sub_network)
                     .groupby(['comm_label', 'entry_id', 'node_id'])['entry_prob']
                     .sum()
                     .reset_index()
                     .sort_values('entry_prob', ascending=False)
                     .groupby('entry_id').head(1)
                     .merge(entry_reference, on='entry_id', how='inner')
                     .assign(node_id=lambda x: x['node_id'] + 1,
                             entry_prob=lambda x: round(x['entry_prob']*100, 2),
                             comm_label=lambda x: x['comm_label'].map(new_groups))
                     .sort_values(['comm_label', 'node_id', 'entry_prob', 'entry_id'],
                                  ascending=[True, True, False, True]))
config_entry_comm = config_entry_comm[['comm_label', 'node_id', 'entry_id', 'entry_name', 'entry_prob']]
config_entry_comm = config_entry_comm.rename(
    columns={'comm_label': 'Group',
             'node_id': 'Node ID',
             'entry_id': 'DRH ID',
             'entry_name': 'Entry name (DRH)',
             'entry_prob': 'Weight'})

config_entry_comm = config_entry_comm.rename(
    columns = {'comm_label': 'Group',
               'node_id': 'Node ID',
               'entry_id': 'DRH ID',
               'entry_name': 'Entry name (DRH)',
               'entry_prob': 'Weight'})

config_entry_latex = config_entry_comm.to_latex(index=False)
with open('../tables/table_A3_included.txt', 'w') as f: 
    f.write(config_entry_latex)

# table A4
entry_reference = pd.read_csv('../data/preprocessing/entry_reference.csv')
excluded_entries = entry_reference[~entry_reference['entry_id'].isin(config_entry_overlap['entry_id'])].rename(columns={'entry_id': 'DRH ID', 'entry_name': 'Entry name (DRH)'}).sort_values(by='DRH ID').reset_index(drop=True)
excluded_entries_latex = excluded_entries.to_latex(index=False)
with open('../tables/table_A4_excluded.txt', 'w') as f: 
    f.write(excluded_entries_latex)