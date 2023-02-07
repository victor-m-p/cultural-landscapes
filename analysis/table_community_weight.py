'''
Weight of communities in the landscape network.
The numbers used in article, but not the table. 
'''

import pandas as pd 
network_information = pd.read_csv('../data/analysis/network_information.csv')
community_weight = network_information.groupby('recode_comm')['config_prob'].sum().reset_index(name = 'Weight')
community_weight['Weight'] = community_weight['Weight'].apply(lambda x: round(x*100, 2))
community_table = community_weight.rename(
    columns = {'recode_comm': 'Group'}
)
community_table = community_table.to_latex(index = False)
with open('../tables/table_community_weight.txt', 'w') as f: 
    f.write(community_table)