'''
VMP 2022-12-15: 
Prepares key documents for the analysis of DRH data. 
This runs before 'expand_data.jl'. 
'''

import numpy as np 
from fun import p_dist, bin_states
import pandas as pd 
pd.set_option('display.max_colwidth', None)

# setup
n_nodes = 20
n_nan = 5
n_rows = 455
n_entries = 407

# question reference 
question_reference = pd.read_csv(f'../data/reference/question_reference_questions_{n_nodes}_maxna_{n_nan}_nrows_{n_rows}_entries_{n_entries}.csv')

## question short-hands
question_shorthand = {
    4676: 'Official political support',
    4729: 'Scriptures',
    4745: 'Monumental religious architecture',
    4776: 'Spirit-body distinction',
    4780: 'Belief in afterlife',
    4787: 'Reincarnation in this world',
    4794: 'Special treatment for corpses',
    4808: 'Co-sacrifices in tomb/burial',
    4814: 'Grave goods',
    4821: 'Formal burials',
    4827: 'Supernatural beings present',
    4954: 'Supernatural monitoring present',
    4983: 'Supernatural beings punish',
    5127: 'Castration required',
    5132: 'Adult sacrifice required',
    5137: 'Child sacrifice required',
    5142: 'Suicide required',
    5152: 'Small-scale rituals required',
    5154: 'Large-scale rituals required',
    5220: 'Distinct written language'
}

## assign question shorthand, and new ID
question_reference['question'] = question_reference['question_id_drh'].apply(lambda x: question_shorthand.get(x))
question_reference['question_id'] = question_reference.index + 1   

## re-order columns and save 
question_reference = question_reference[['question_id', 'question_id_drh', 'question', 'question_drh']]
question_reference.to_csv('../data/preprocessing/question_reference.csv', index = False)

# move entry_reference to a new location
entry_reference = pd.read_csv(f'../data/reference/entry_reference_questions_{n_nodes}_maxna_{n_nan}_nrows_{n_rows}_entries_{n_entries}.csv')
entry_reference.to_csv('../data/preprocessing/entry_reference.csv', index = False)

## save as latex 
entry_latex = entry_reference.style.hide(axis='index').to_latex()
with open('../tables/entry_table.txt', 'w') as f: 
    f.write(entry_latex)

# create dataframe where (1, -1) - i.e. inconsistent answers - is coded as 0 
# this is just to make it easier to expand all of the possible configurations for an entry. 
# i.e. in this case it is the same whether it is (1, -1) or 0. 
direct_reference = pd.read_csv(f'../data/reference/direct_reference_questions_{n_nodes}_maxna_{n_nan}_nrows_{n_rows}_entries_{n_entries}.csv')
question_columns = direct_reference.columns[1:-1]
direct_flattened = direct_reference.groupby('entry_id')[question_columns].mean().reset_index().astype(int) # mean(1, -1) = 0
direct_flattened.to_csv('../data/preprocessing/data_flattened.csv', index = False)

# calculate probability of all configurations based on parameters h, J.
params = np.loadtxt(f'../data/mdl_experiments/matrix_questions_{n_nodes}_maxna_{n_nan}_nrows_{n_rows}_entries_{n_entries}.txt.mpf_params.dat')
nJ = int(n_nodes*(n_nodes-1)/2)
J = params[:nJ]
h = params[nJ:]
p = p_dist(h, J) # takes a minute (and a lot of memory). 
np.savetxt(f'../data/preprocessing/configuration_probabilities.txt', p)

# all configurations file allstates 
allstates = bin_states(n_nodes) # takes a minute (do not attempt with n_nodes > 20)
np.savetxt(f'../data/preprocessing/configurations.txt', allstates.astype(int), fmt='%i')