import numpy as np 
import pandas as pd 
from fun import *

#d = pd.read_csv('../data/analysis/d_likelihood_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.csv')
#sref = pd.read_csv('../data/reference/sref_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv')
#nref = pd.read_csv('../data/reference/nref_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv')
#d = d.merge(sref, on = 'entry_id', how = 'inner')
#p = np.loadtxt('../data/analysis/p_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10_NN1_LAMBDA0_453839.txt') 

# setup
n_rows, n_nan, n_nodes, n_top_states = 455, 5, 20, 150

# load data
p = np.loadtxt(f'../data/analysis/p_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.txt')
d_likelihood = pd.read_csv(f'../data/analysis/d_likelihood_nrows_{n_rows}_maxna_{n_nan}_nodes_{n_nodes}.csv')
nodes_reference = pd.read_csv(f'../data/analysis/nref_nrows_455_maxna_5_nodes_20.csv')
sref = pd.read_csv('../data/analysis/sref_nrows_455_maxna_5_nodes_20.csv')
d_main = pd.read_csv('../data/analysis/d_main_nrows_455_maxna_5_nodes_20.csv')

# configs
n_states = 20
configs = bin_states(n_states) 

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


# other interesting records # 
# Han Confucianism
# Religion in Mesopotamia
# Irish Catholicism
# Inca
# Daoism
# Donatism (is there more than once, but treated separately)
# Northern Irish Roman Catholics
# Northern Irish Protestants
# Anglican Church
# American Evangelicalism
# Italy: Roman Catholic Christianity
# Atheism in the Soviet Union
# Religion in the Old Assyrian Period
# Religion in Judah
# Islamic modernists
# Secular Buddhists


## okay, get those with between 2 - 8 rows ## 
dn = d_likelihood.groupby('entry_id').size().reset_index(name = 'count')
dn = d_likelihood.merge(dn, on = 'entry_id', how = 'inner')
dn = dn.merge(nodes_reference, on = 'entry_id', how = 'inner')
dsub = dn[dn['count'] >= 2] # dn[(dn['count'] >= 2) & (dn['count'] <= 8)]
ncols = sref['related_q_id'].to_list()

# test specific ones (mostly nan)
## anglican 
anglican = contains(dsub, 'Anglican Church')
anglican_tot, anglican_tab = likelihood(anglican, configs, ncols, sref)
anglican_typ = unknown_type(sref, anglican_tot)

## spartan (this is actually NAN)
### both NAN
pd.set_option('display.max_colwidth', None)
spartan = contains(dsub, 'Archaic Spartan Cults')
spartan_total, spartan_table, qq = likelihood(spartan, configs, ncols, sref)
spartan_types = unknown_type(d_main, sref, spartan_total)

# run over all:
entry_lst = dsub['entry_id'].unique().tolist()
dct_table = {}
dct_types = {}
dct_reference = {}
for idx, e_id in enumerate(entry_lst):
    d_tmp = dsub[dsub['entry_id'] == e_id]
    d_tot, d_tab = likelihood(d_tmp, configs, ncols, sref)
    d_typ = unknown_type(d_main, sref, d_tot)
    #d_overview = d_typ.merge(d_q, on = ['related_q', 'related_q_id'], how = 'inner')
    dct_table[idx] = d_tab 
    dct_types[idx] = d_typ
    dct_reference[idx] = d_tmp


dct_types[0]

lst = []
for idx, df in dct_types.items(): 
    x = len(df[df['type']!= 0])
    if x != 0: 
        lst.append(idx)
407/len(lst)
for idx, val in dct_types.items():
    x = val

has_disagreement = [len(x[x['type'] != 0]) for x in dct_types]
x = dct_types[0]
x
len(x[x['type'] != 0])

# 13 = Iban 
n = 4
dct_table[n]
dct_types[n]
dct_reference[n]

def uniq_bitstring(allstates, config_idx, question_ids, type):
    focal_config = allstates[config_idx]
    focal_config[focal_config == -1] = 0
    focal_string = ''.join([str(x) for x in focal_config])
    focal_df = pd.DataFrame([focal_config], columns = question_ids)
    focal_df['p_ind'] = config_idx
    focal_df = pd.melt(focal_df, id_vars = 'p_ind', value_vars = question_ids, var_name = 'related_q_id')
    focal_df = focal_df.rename(columns = {
        'p_ind': f'p_ind_{type}',
        'value': f'value_{type}'})
    return focal_string, focal_df 

string_focal, df_focal = uniq_bitstring(configs, 896902, ncols, 'focal')
string_focal
df_focal
df_focal.merge(sref, on = 'related_q_id', how ='inner').sort_values('value_focal')
sref

## spartans
pd.set_option('display.max_colwidth', None)
spartan = contains(dsub, 'Archaic Spartan Cults')
spartan_total, spartan_table = likelihood(spartan, configs, ncols, sref)
spartan_types = unknown_type(d_main, sref, spartan_total)
spartan_types
spartan_table
spartan_total
string_focal, df_focal = uniq_bitstring(configs, 765831, ncols, 'focal')
df_focal['id'] = df_focal.index+1
df_focal
# 2: Pauline
# 4: Late Classic Lowland Maya
# Sri Lankan Buddhism (ritual disagreement)
# 13: Iban traditional religion
# 28: Veerashaivas
# 38: Diasporic
# 49: Manus (could be interesting.)
# 78: Raglai

#### field doesn't know for Raglai!!!: 
#### but we DO know!!!!

#### 
0.785662+0.013111
100-98.4

0.013111+0.002865
# another good NAN: Late Classic Lowland Maya (4)


d_tmp = dsub[dsub['entry_id'] == e_id]
d_tot, d_q, d_tab = likelihood(d_tmp, configs, ncols, nref)
d_typ = unknown_type(sref, nref, d_tot)
d_overview = d_typ.merge(d_q, on = ['related_q', 'related_q_id'], how = 'inner')

d_overview
d_tab

## generally old
## spartans are fun (NA)
## 967 interesting (U Unitarians, disagreement- which makes sense)

##### other potentials ######
# Roman Divination
# Pauline Christianity
# Anglican Church
# Archaic Spartan Cults
# Gaudiya
# Late Classic Lowland Maya 
# Krishna Worship in North India - Modern Period
# !Kung
# Burmese
# Maori
# Chinese Esoteric Buddhism (Tang Tantrism)
# Hinduism in Trinidad
# Temple of the Jedi Order
# Postsocialist Mongolian Buddhism
# Orphism
# Postsocialist Mongolian Shamanism
# The Church of England
# Spartan Religion
# Unitarian Universalism
# Nestorian Christianity
# Monastic Communities of Lower Egypt: Nitria, Kellia, Scetis
# Early Christianity and Monasticism in Egypt
# Religion in Greco-Roman Egypt
# Religion in Greco-Roman Alexandria
# Atheism in the Soviet Union
# Religion in Juda
# Religion of Phonecia
# Epic of Gigamesh
# The Monastic School of Gaza
# Islamic modernists
# Book of Ezekiel
# Mesopotamian Exorcistic Texts
# Secular Buddhists


