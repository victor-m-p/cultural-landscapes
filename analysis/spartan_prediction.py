'''
Implied probability for different possible Spartan Imperial Cult configurations.
'''
import configuration as cn 
import numpy as np 
import pandas as pd 
from fun import *
pd.set_option('display.max_colwidth', None)

# load data
configurations = np.loadtxt('../data/preprocessing/configurations.txt', dtype = int)
configuration_probabilities = np.loadtxt(f'../data/preprocessing/configuration_probabilities.txt')
question_reference = pd.read_csv(f'../data/preprocessing/question_reference.csv')
entry_config_master = pd.read_csv('../data/preprocessing/entry_configuration_master.csv')
match_substring(entry_config_master, 'Archaic Spartan Cults')

# probabilities for spartan variations
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