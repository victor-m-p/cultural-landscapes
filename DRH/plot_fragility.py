# COGSCI23
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib.lines import Line2D

# plotting setup
small_text = 12
large_text = 18

# read data
d_fragility = pd.read_csv('../data/COGSCI23/fragility_observed.csv')
d_fragility['prob_remain'] = d_fragility['prob_remain']*100

# overall patterns (i.e. for each time-step)
## different kinds of uncertainty: 
## this basically shows that we have no uncertainty wrt. 
## the population -
## but we are not showing the individual-level uncertainty here. 
## some of the curves are very different. 
fig, ax = plt.subplots(figsize = (7, 5), dpi = 300)
sns.pointplot(data = d_fragility, x = "n_fixed_traits", y = "prob_remain")
plt.suptitle('Stability (all observed)', size = large_text)
plt.xlabel('Number of fixed traits', size = small_text)
plt.ylabel('Probability remain', size = small_text)
plt.savefig('../fig/COGSCI23/stability/global_mean.pdf')

# overall pattern (plot all of the lines) 
## here we get a much better idea about the distribution
## rather than just the global mean. 
unique_configs = d_fragility['config_id'].unique()
mean_config = d_fragility.groupby('n_fixed_traits').agg({'prob_remain': ['mean', 'median']})
mean_config.columns = mean_config.columns.droplevel()
mean_config = mean_config.reset_index()
fig, ax = plt.subplots(figsize = (7, 5), dpi = 300)
# all of the samples
for config in unique_configs: 
    tmp_config = d_fragility[d_fragility['config_id'] == config]
    x = tmp_config['n_fixed_traits'].tolist()
    y = tmp_config['prob_remain'].tolist()
    ax.plot(x, y, color = 'tab:blue', alpha = 0.05)
# global mean and median
x = mean_config['n_fixed_traits'].tolist()
y_mean = mean_config['mean'].tolist()
y_median = mean_config['median'].tolist()
ax.plot(x, y_mean, color = 'tab:orange', linewidth = 2)
ax.plot(x, y_median, color = 'tab:red', linewidth = 2)
# title, labels, and legend
custom_legend = [Line2D([0], [0], color = 'tab:blue', lw=4),
                 Line2D([0], [0], color = 'tab:orange', lw=4),
                 Line2D([0], [0], color = 'tab:red', lw=4)]
ax.legend(custom_legend, ['Sample', 'Mean', 'Median'])
plt.xticks(np.arange(0, 20, 1))
plt.suptitle('Stability (all observed)', size = large_text)
plt.xlabel('Number of fixed traits', size = small_text)
plt.ylabel('Probability remain', size = small_text)
plt.savefig('../fig/COGSCI23/stability/distribution.pdf')

# curves for each community 
## either the two small ones are just different
## or this might suggest that that curve is the MEDIAN
## while for the other ones it is driven up a bit because
## of outliers?
network_information = pd.read_csv('../data/analysis/network_information_enriched.csv')
community_information = network_information[['config_id', 'comm_label']]
community_fragility = d_fragility.merge(community_information, on = 'config_id', how = 'inner')
community_fragility = community_fragility.sort_values('comm_label', ascending = True)

fig, ax = plt.subplots(figsize = (7, 5), dpi = 300)
sns.pointplot(data = community_fragility, x = 'n_fixed_traits', 
              y = 'prob_remain', hue = 'comm_label')
plt.suptitle('Stability (by community)', size = large_text)
plt.xlabel('Number of fixed traits', size = small_text)
plt.ylabel('Probability remain', size = small_text)
plt.legend(title = 'Community')
plt.savefig('../fig/COGSCI23/stability/community.pdf')

# plot a couple of interesting ones (what is interesting?) 
## (1) one that is just very stable 
## (2) one that is very unstable, but where enforcement is effective
## (3) one that is very unstable but where enforcement is effective 
## ... find out which religions they correspond to ... 
## (4) another way to find "interesting" of course is to look 
## for theoretically interesting religions. 
## NB: lowest std. removed because that equals highest sum. 

# find based on overall sum
config_group_sum = d_fragility.groupby('config_id')['prob_remain'].sum().reset_index(name = 'sum')
## lowest sum
config_min_sum = config_group_sum.sort_values('sum', ascending = True).head(1)
## highest sum
config_max_sum = config_group_sum.sort_values('sum', ascending = False).head(1)
# find based on overall std 
config_group_std = d_fragility.groupby('config_id')['prob_remain'].std().reset_index(name = 'std')
## highest std
config_max_std = config_group_std.sort_values('std', ascending = False).head(1)

# preparation
configs = [config_min_sum,
           config_max_sum, 
           config_max_std]

colors = ['tab:blue',
          'tab:orange',
          'tab:red']

# plot these four 
fig, ax = plt.subplots(figsize = (7, 5), dpi = 300)
for config, color in zip(configs, colors):
    tmp_config = d_fragility.merge(config, on = 'config_id', how = 'inner')
    x = tmp_config['n_fixed_traits'].tolist()
    y = tmp_config['prob_remain'].tolist()
    ax.plot(x, y, color = color, linewidth = 2)
# title, 
custom_legend = [Line2D([0], [0], color = 'tab:blue', lw=4),
                 Line2D([0], [0], color = 'tab:orange', lw=4),
                 Line2D([0], [0], color = 'tab:red', lw=4)]
ax.legend(custom_legend, ['min', 'max', 'max(std)'])
plt.xticks(np.arange(0, 20, 1))
plt.suptitle('Examples', size = large_text)
plt.xlabel('Number of fixed traits', size = small_text)
plt.ylabel('Probability remain', size = small_text)
plt.savefig('../fig/COGSCI23/stability/case_study.pdf')

# what do they correspond to? 
entry_conf = pd.read_csv('../data/analysis/entry_configuration_master.csv')
config_min_sum = config_min_sum[['config_id']]
config_min_sum['type'] = 'min'
config_max_sum = config_max_sum[['config_id']]
config_max_sum['type'] = 'max'
config_max_std = config_max_std[['config_id']]
config_max_std['type'] = 'max(std)'
case_studies = pd.concat([config_min_sum, config_max_sum, config_max_std])
case_studies = entry_conf.merge(case_studies, on = 'config_id', how = 'inner')

# min: Warrau (one possible variation which is extremely unlikely)
# --> so, no full configuration
# max(std): Tractarian Movement (one possible variation which is extremely unlikely)
# --> so, no full configuration
# max: Ancient Egypt, ... 