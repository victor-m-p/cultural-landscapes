import numpy as np 
import pandas as pd 
from fun import *

# setup
n_rows, n_nan, n_nodes, n_top_states = 455, 5, 20, 150

# load data
configuration_probabilities = np.loadtxt(f'../data/analysis/configuration_probabilities.txt')
question_reference = pd.read_csv(f'../data/analysis/question_reference.csv')
entry_reference = pd.read_csv('../data/analysis/entry_reference.csv')

# configs
n_states = 20
configurations = bin_states(n_states) 

# need a way to soft match
def contains(df, str): 
    return df[df['entry_name'].str.contains(str)]

## check records
def likelihood(d, configs, cols, nref):
    d_idx, n_idx = d['p_ind'].values, len(d['p_ind'].values)
    d_conf = configs[d_idx] 
    d_mat = pd.DataFrame(d_conf, columns = cols)
    d_mat['p_ind'] = d_idx 
    d_mat = pd.melt(d_mat, id_vars = 'p_ind', value_vars = cols, var_name = 'related_q_id')
    d_mat_g = d_mat.groupby(['related_q_id', 'value']).size().reset_index(name = 'count_comb')
    d_mat_g = d_mat_g[d_mat_g['count_comb'] < n_idx]
    d_inner = d_mat.merge(d_mat_g, on = ['related_q_id', 'value'], how = 'inner')
    ref_questions = nref.merge(d_inner, on = 'related_q_id', how = 'inner')
    
    # gather 
    d_total = ref_questions.merge(d, on = 'p_ind', how = 'inner')
    d_table = pd.pivot(d_total, index = 'p_norm', columns = 'related_q', values = 'value').reset_index()
    d_table = d_table.sort_values('p_norm', ascending=False)
    return d_total, d_table

def unknown_type(sref, nref, d_total): 
    s = d_total['entry_id'].unique()[0] # assumes only one
    row = sref[sref['entry_id'] == s]
    l = len(row)
    row = pd.melt(row, id_vars = 'entry_id', 
                    value_vars = row.columns[1:-1],
                    var_name = 'related_q_id')
    row['related_q_id'] = row['related_q_id'].astype(int)
    
    # nan vals
    nan_vals = row[row['value'] == 0]
    nan_vals = nan_vals.rename(columns = {'value': 'type'}) 
    nan_vals = nan_vals[['related_q_id', 'type']].drop_duplicates()

    # inconsistencies
    inconst_vals = row.groupby(['related_q_id', 'value']).size().reset_index(name = 'type')
    inconst_vals = inconst_vals[inconst_vals['type'] < l]
    inconst_vals = inconst_vals[['related_q_id', 'type']].drop_duplicates()

    # merge
    all_unknown = pd.concat([nan_vals, inconst_vals])
    all_unknown = nref.merge(all_unknown, on = 'related_q_id', how = 'inner')

    return all_unknown


## spartan (this is actually NAN)
### both NAN
dsub = pd.read_csv('../data/analysis/entry_configuration_master.csv')
pd.set_option('display.max_colwidth', None)
spartan = contains(dsub, 'Archaic Spartan Cults')

# obviously we need a better setup 
import configuration as cn 
spartan_765847 = cn.Configuration(765847,
                                  configurations, 
                                  configuration_probabilities)

spartan_765831 = cn.Configuration(765831,
                                  configurations,
                                  configuration_probabilities)

spartan_765843 = cn.Configuration(765843, 
                                  configurations,
                                  configuration_probabilities)

spartan_765827 = cn.Configuration(765827,
                                  configurations,
                                  configuration_probabilities)


spartan_765847.diverge(spartan_765831, question_reference)

spartan_765847.diverge(spartan_765843, question_reference)

# 765847: + small-scale, + child-sacrifice (1.2%)
# 765831: + small-scale, - child-sacrifice (69.7%)
# 765843: - small-scale, + child-sacrifice (0.4%)
# 765827: - small-scale, - child-sacrifice (28.8%)