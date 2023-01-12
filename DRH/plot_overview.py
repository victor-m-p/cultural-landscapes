'''
VMP 2022-01-09
Plotting overview.
'''
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

# setup
small_text, large_text = 12, 18

# just plot their weight against time 
# spent at themselves or something really simple
d_weight = pd.read_csv('../data/COGSCI23/evo_clean/weighted.csv')
d_self = d_weight[d_weight['config_from'] == d_weight['config_to']]
d_self = d_self.rename(columns = {'config_to': 'config_id'})
d_self = d_self.drop(columns = 'config_from')

# now we can plot this against stuff 
maxlik_evo_stability = pd.read_csv('../data/COGSCI23/evo_stability/maxlik_evo_stability.csv')

# crazy stuff
overview = maxlik_evo_stability.merge(d_self, on = 'config_id', how = 'inner')
overview['log_config_prob'] = np.log(overview['config_prob'])
overview['self_pct'] = overview['weight']/100

# plot % self vs. log(p(configuration))
fig, ax = plt.subplots(dpi = 300)

sns.regplot(data = overview,
            x = 'log_config_prob',
            y = 'self_pct')
plt.xlabel('log(p(config))', size = small_text)
plt.ylabel(r'% self', size = small_text)
plt.suptitle('First 100 timesteps', size = large_text)
plt.savefig('../fig/COGSCI23/overview/self_config.pdf')

# % self vs. p(remain)
fig, ax = plt.subplots(dpi = 300)
sns.regplot(data = overview,
            x = 'remain_prob',
            y = 'self_pct')
plt.xlabel('p(remain)', size = small_text)
plt.ylabel(r'% self', size = small_text)
plt.suptitle('First 100 timesteps')
plt.savefig('../fig/COGSCI23/overview/self_remain.pdf')

# p(remain) vs. log(p(configuration))
# this has been moved to a file for itself
# because we have actually made this nice now

# hamming (weighted across all 100 t)
hamming = pd.read_csv('../data/COGSCI23/evo_clean/hamming.csv')
hamming_weight = d_weight.merge(hamming, on = ['config_from', 'config_to'], how = 'inner')
n_sim, n_t = 100, 100
hamming_weight['hamming_weight'] = hamming_weight['weight']*hamming_weight['hamming_dist']
hamming_diameter = hamming_weight.groupby('config_from')['hamming_weight'].sum().reset_index(name = 'hamming_weight')
hamming_diameter['hamming_0'] = hamming_diameter['hamming_weight']/(n_sim*n_t)
hamming_diameter = hamming_diameter.drop(columns = {'hamming_weight'})

# hamming (for n = 10)
subsample = pd.read_csv('../data/COGSCI23/evo_clean/subsample.csv')
subsample_10 = subsample[subsample['t_to'] == 11]
hamming_10 = subsample_10.merge(hamming, on = ['config_from', 'config_to'], how = 'inner')
hamming_10 = hamming_10.groupby(['config_from'])['hamming_dist'].sum().reset_index(name = 'hamming_10')
hamming_10['hamming_10'] = hamming_10['hamming_10']/n_sim

# hamming (for n = 90)
subsample_100 = subsample[subsample['t_to'] == 91]
hamming_100 = subsample_100.merge(hamming, on = ['config_from', 'config_to'], how = 'inner')
hamming_100 = hamming_100.groupby(['config_from'])['hamming_dist'].sum().reset_index(name = 'hamming_100')
hamming_100['hamming_100'] = hamming_100['hamming_100']/n_sim

# merge these 
from functools import reduce
d_hamming = reduce(lambda  left, right: 
    pd.merge(left, right, on = ['config_from'],
             how = 'outer'), 
    [hamming_diameter, hamming_10, hamming_100])
d_hamming = d_hamming.rename(columns = {'config_from': 'config_id'})
hamming_overview = overview.merge(d_hamming, on = 'config_id', how = 'inner')

# to long 
hamming_overview
hamming_long = pd.wide_to_long(hamming_overview,
                               stubnames = 'hamming_',
                               i = 'config_id',
                               j = 'hamming_type').reset_index()

# plot 
# hamming against log(p(config))
fig, ax = plt.subplots(dpi = 300)
sns.lmplot(data = hamming_long,
           x = 'log_config_prob',
           y = 'hamming_',
           hue = 'hamming_type')
plt.xlabel('log(p(config))', size = small_text)
plt.ylabel('Hamming diameter', size = small_text)
plt.savefig('../fig/COGSCI23/overview/hamming_config.pdf')

# hamming against p(remain)
fig, ax = plt.subplots(dpi = 300)
sns.lmplot(data = hamming_long,
           x = 'remain_prob',
           y = 'hamming_',
           hue = 'hamming_type')
plt.xlabel('p(remain)', size = small_text)
plt.ylabel('Hamming diameter', size = small_text)
plt.savefig('../fig/COGSCI23/overview/hamming_remain.pdf')
