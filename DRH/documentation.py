'''
VMP 2022-12-12: 
1. Document data curation (introduction to our data set)
2. Entry Name overlap (independence of records, discussion)
'''

import pandas as pd 
from fun import *
from Civ import Civilizations


''' data curation documentation '''

# setup
n_rows, n_nan, n_nodes, n_top_states = 455, 5, 20, 150

infile = '../data/raw/drh_20221019.csv'
d = pd.read_csv(infile, low_memory=False) 

#### number of records ####
## raw data
df = d 
len(df['entry_id'].unique()) # 838 (we use this)
len(df['entry_name'].unique()) # 799
len(df['related_q_id'].unique()) # all: 1133

## only super questions
df = df[df['related_parent_q'].isna()] 
len(df['entry_id'].unique()) # 838
len(df['entry_name'].unique()) # 799
len(df['related_q_id'].unique()) # super only: 171

## binary only 
df = df[["entry_id", "entry_name", "related_q_id", "related_q", "answers"]]  
df.replace(
    {'answers': 
        {"Field doesn't know": 'Unknown', 
        "I don't know": 'Unknown'}},
    inplace = True)
civ = Civilizations(df)
civ.preprocess()

dfuniq = civ.duniq
len(dfuniq['related_q_id'].unique()) # 149
len(dfuniq['entry_id'].unique()) # 835
dfuniq['has_answer'].mean()

# run through specific params
n_questions = 20
n_nan = 4
civ.set_constraints(n_questions, n_nan/n_questions, "related_q_id")
civ.n_best()
civ.max_constraints()
civ.max_tolerance()
civ.weight_format()

dcsv = civ.dcsv
len(dcsv['s'].unique()) # 471

''' Document data overlap '''

# load data
pd.set_option('display.max_colwidth', None)
nodes_reference = pd.read_csv(f'../data/analysis/nref_nrows_455_maxna_5_nodes_20.csv')
nodes_reference.groupby(['entry_name']).size().reset_index(name = 'count').sort_values('count', ascending=False)
x = nodes_reference.sort_values('entry_name')
x.iloc[120:140]

# not much different from Egypt: 
## e.g. 
### Ancient Egypt - Early Dynastic Period,
### Ancient Egypt - First Intermediate Period,
### Ancient Egypt - Old Kingdom
### Ancient Egypt - Predynastic Perid - Early Naqada Culture
### Ancient Egyptian
### Ancient Egyptian Religion in the Early 18th Dynasty

## Church of Jesus Christ of Latter-day Saints (eraly)
## Church of Jesus Christ of Latter-day Saints (modern)