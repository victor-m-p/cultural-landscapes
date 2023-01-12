import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import ptitprince as pt 
import seaborn as sns

data_raw = pd.read_csv('../data/raw/drh_20221019.csv')
entry_reference = pd.read_csv('../data/analysis/entry_reference.csv')

# for each entry id, get 
data_year = data_raw[['entry_id', 'start_year', 'end_year']]
data_year = data_year.groupby('entry_id').agg({'start_year': 'min', 'end_year': 'max'}).reset_index()

# inner join with the data we already have
entry_reference = entry_reference[['entry_id_drh', 'entry_drh']]
entry_reference = entry_reference.rename(columns = {'entry_id_drh': 'entry_id'})
year_reference = entry_reference.merge(data_year, on = 'entry_id', how = 'inner')

# sort by start year
year_reference = year_reference.sort_values('start_year', ascending=True)

# plotting setup
linewidth = 0.5
y_increment = 10
small_text = 12
big_text = 18

# plot (insp. by Sebastian Nehrdich)
## the basic plot of the included data
fig, ax = plt.subplots(figsize = (8, 4), dpi = 300)
y = 0
mid = (fig.subplotpars.right + fig.subplotpars.left)/2 # consistent suptitle 
for index, row in year_reference.iterrows():
    start_year, end_year = row['start_year'], row['end_year']
    ax.plot([start_year, end_year], [y, y], color = 'tab:blue', linewidth=linewidth)
    y += y_increment
plt.tick_params(left=False, labelleft=False)
plt.xlabel('Year', size = small_text)
plt.suptitle('Religion age', y = 1.0, size = big_text, x = mid)
plt.title('407 entries included in analysis', size = small_text)
plt.savefig('../fig/timeperiod/religion_age_included_lineplot.pdf')

# color by inclusion exclusion
## prep 
data_left = data_year.merge(entry_reference, on = 'entry_id', how = 'left')
data_left = data_left.assign(condition = ['Excluded' if x is np.nan else 'Included'for x in data_left['entry_drh']])
data_left = data_left.assign(color = ['tab:orange' if x is np.nan else 'tab:blue' for x in data_left['entry_drh']])
data_left = data_left.sort_values('start_year', ascending=True)

## plot 
colors = ['tab:blue', 'tab:orange']
labels = ['Included', 'Excluded']
lines = [Line2D([0], [0], color = c, linewidth = 3) for c in colors]
fig, ax = plt.subplots(figsize = (8, 4), dpi = 300)
y=0
mid = (fig.subplotpars.right + fig.subplotpars.left)/2 # consistent suptitle 
for index, row in data_left.iterrows():
    start_year, end_year = row['start_year'], row['end_year']
    ax.plot([start_year, end_year], [y, y], color = row['color'], linewidth=linewidth)
    y += y_increment
plt.tick_params(left = False, labelleft = False)
plt.xlabel('Year', size = small_text)
plt.legend(lines, labels, frameon=False, fontsize = small_text)
plt.suptitle('Religion age', y = 1.0, size = big_text, x = mid)
plt.title('838 entries (colored by inclusion)', size = small_text)
plt.savefig('../fig/timeperiod/religion_age_included_excluded_lineplot.pdf')

# color by inclusion / exclusion (violin)
dx="condition"; dy="start_year"; ort="v"; pal = "Set2"; sigma = .2
f, ax = plt.subplots(figsize=(5, 5), dpi = 300, constrained_layout = True)
ax=pt.RainCloud(x = dx, y = dy, 
                data = data_left, 
                palette = ['tab:orange', 'tab:blue'], 
                bw = sigma, width_viol = 1, 
                ax = ax, orient = ort, rain_alpha = 0.25,
                box_showfliers=False)
plt.xlabel('Inclusion in analysis', size = small_text)
plt.ylabel('Year', size = small_text)
plt.suptitle('Religion start year', size = big_text)
plt.savefig('../fig/timeperiod/religion_age_included_excluded_raincloud.pdf')

# based on community
## preprocessing
network_information = pd.read_csv('../data/analysis/network_information_enriched.csv')
network_information = network_information[['config_id', 'comm_label', 'comm_color_code']].drop_duplicates()
entry_config_master = pd.read_csv('../data/analysis/entry_configuration_master.csv')
entry_config_master = entry_config_master[['entry_drh', 'config_id']].drop_duplicates()
community_information = network_information.merge(entry_config_master, on = 'config_id', how = 'inner')
community_information = community_information.merge(data_left, on = 'entry_drh', how = 'inner')
community_information = community_information[['entry_drh', 'entry_id', 'comm_label', 'comm_color_code', 'start_year', 'end_year']]
community_information = community_information.sort_values('comm_label', ascending = True)

## only the three biggest
## NB: fix colors
community_subset = community_information[community_information['comm_label'].isin(['Group 1', 'Group 2', 'Group 3'])]
dx = "comm_label"; dy = "start_year"; ort = "v"; pal = "Set2"; sigma = .2
fig, ax = plt.subplots(figsize=(5, 7), dpi = 300, constrained_layout = True)
ax=pt.RainCloud(x = dx, y = dy, 
                data = community_subset, 
                palette = pal, 
                bw = sigma, width_viol = 1, 
                ax = ax, orient = ort, rain_alpha = 0.5,
                box_showfliers=False)
plt.xlabel('Community', size = small_text)
plt.ylabel('Year', size = small_text)
plt.suptitle('Religion start year', size = big_text)
plt.tight_layout()
plt.savefig('../fig/timeperiod/religion_age_community_large_raincloud.pdf')

## all communities (boxplot)
community_boxplot = community_information[['comm_label', 'start_year']]
fig, ax = plt.subplots(figsize = (5, 7), dpi = 300, constrained_layout = True)
sns.boxplot(x = 'comm_label', y = 'start_year', data = community_boxplot)
plt.xlabel('Community', size = small_text)
plt.ylabel('Year', size = small_text)
plt.suptitle('Community start year', size = big_text)
plt.tight_layout()
plt.savefig('../fig/timeperiod/religion_age_community_boxplot.pdf')